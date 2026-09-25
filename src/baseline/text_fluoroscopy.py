
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from datasets import Dataset
from argparse import Namespace

import torch
import argparse
import torch.nn.functional as F
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils import return_device

"""
- drop the first model from gpu
"""

torch.manual_seed(42)

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512], num_labels=2, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        self.num_labels = num_labels
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(prev_size, hidden_size),
                nn.Tanh(),
            ])
            prev_size = hidden_size
        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_labels)
    
    def forward(self, x):
        x = self.dense(x)
        x = self.classifier(x)
        return x
    
class TextFluoroscopy:
    def __init__(self):
    

        self.max_length = 512
        self.pretrained_model_name_or_path = "Qwen/Qwen3-4B"
        self.which_embedding = 'gte-qwen_KL_with_first_and_last_layer'

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path, 
                                                       trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = AutoConfig.from_pretrained(self.pretrained_model_name_or_path, trust_remote_code=True)
        if not hasattr(cfg, "rope_theta"):
            cfg.rope_theta = 10000.0
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path, 
                                                          trust_remote_code=True,
                                                          config=cfg)
        # Keep this baseline on CPU to avoid MPS placeholder-storage errors.
        self.device = return_device()
        self.model.to(self.device)
        self.model.eval()

        self.droprate = .4
        self.learning_rate = 0.003
        self.hidden_sizes = [1024,512]


    
    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]
    
    def get_kl(self, model, input_texts):
        batch_dict = self.tokenizer(input_texts, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = model(**batch_dict,output_hidden_states=True)
            last_logits = model.lm_head(outputs.hidden_states[-1]).squeeze()
            first_logits = model.lm_head(outputs.hidden_states[0]).squeeze()
        kls = []
        for i in range(1,len(outputs.hidden_states)-1):
            with torch.no_grad():
                middle_logits = model.lm_head(outputs.hidden_states[i]).squeeze()
            kls.append(F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(first_logits, dim=-1), reduction='batchmean').item()+
                    F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(last_logits, dim=-1), reduction='batchmean').item())
        return kls
        
    def get_all_embedding(self, model, input_texts):
        batch_dict = self.tokenizer(input_texts, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = model(**batch_dict,output_hidden_states=True)
        all_embed = [self.last_token_pool(outputs.hidden_states[i].cpu(), batch_dict['attention_mask'].cpu()) for i in range(len(outputs.hidden_states))]
        all_embed = torch.concat(all_embed,1).cpu()
        return all_embed
    
    def train(self, train_embeddings, train_labels):
        model = BinaryClassifier(
            train_embeddings.shape[1],
            hidden_sizes=self.hidden_sizes,
            dropout_prob=self.droprate,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for _ in range(10):
            model.train()
            for i in range(0, len(train_embeddings), 16):
                outputs = model(train_embeddings[i:i + 16])
                loss = criterion(outputs, train_labels[i:i + 16])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model.eval()

    # def train(self, 
    #           hidden_sizes, 
    #           droprate,
    #           train_embeddings,
    #           train_labels,
    #           valid_embeddings,
    #           valid_labels,
    #           ood_data
    #         ):
    #     input_size = train_embeddings.shape[1]
    #     model = BinaryClassifier(input_size,hidden_sizes=hidden_sizes,dropout_prob=droprate).to(self.device)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
    #     num_epochs = 10
    #     batch_size = 16
    #     best_valid_acc = -1.0
    #     best_epoch = -1
    #     best_ood_metrics = {}
    #     for epoch in range(num_epochs):
    #         for i in range(0, len(train_embeddings), batch_size):
    #             model.train()
    #             batch_embeddings = train_embeddings[i:i+batch_size]
    #             batch_labels = train_labels[i:i+batch_size]
    #             outputs = model(batch_embeddings)
    #             loss = criterion(outputs, batch_labels)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #         model.eval()
    #         with torch.no_grad():
    #             outputs = model(valid_embeddings)
    #             _, predicted = torch.max(outputs.data, 1)
    #             accuracy = (predicted == valid_labels).sum().item() / len(valid_labels)
    #             current_ood_metrics = {}
                
    #             for ood_item in ood_data:
    #                 test_embed = ood_item["embedding"]
    #                 test_label = ood_item["label"]
    #                 auroc, tpr_at_fpr_0_05 = self.test(model, test_embed, test_label, [], ood_item["name"])
    #                 current_ood_metrics[ood_item["name"]] = {
    #                     "auroc": float(auroc),
    #                     "tpr_at_fpr_0_05": float(tpr_at_fpr_0_05),
    #                 }

    #             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.4f}")
    #             if accuracy > best_valid_acc:
    #                 best_valid_acc = accuracy
    #                 best_epoch = epoch + 1
    #                 best_ood_metrics = {k: v.copy() for k, v in current_ood_metrics.items()}
    #     return {
    #         "best_epoch": best_epoch,
    #         "best_val_acc": float(best_valid_acc),
    #         "ood_metrics": best_ood_metrics,
    #     }
    
    def test(self, model,test_set,test_label,test_acc,testset_name):
        with torch.no_grad():
            outputs = model(test_set)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            
            # auroc
            auroc = roc_auc_score(test_label.cpu().numpy(), probabilities.cpu().numpy())

            fpr, tpr, _ = roc_curve(test_label.cpu().numpy(), probabilities.cpu().numpy())
            mask_fpr = fpr <= 0.05
            tpr_at_fpr_0_05 = float(np.max(tpr[mask_fpr])) if np.any(mask_fpr) else 0.0
            # test_acc.append(auroc)
        return auroc, tpr_at_fpr_0_05


    def get_kls(self, input_texts, labels):
        kls = []
        embeddings = []
        for text in input_texts:
            kl = self.get_kl(self.model, [text])
            # embedding 1xdim*num_layers
            embedding = self.get_all_embedding(self.model, [text])
            kls.append(kl)
            embeddings.append(embedding)

        embeddings = torch.cat(embeddings, dim=0)
        idx = np.array(kls).argmax(axis=1)
        embedding_dim = self.model.config.hidden_size 
        embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(embeddings,idx) ]).to(self.device)

        train_labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        return embeddings, train_labels

    def run(self, data: dict[str, list[dict]]) -> tuple[list[float], list[int]]:
        train_texts = [item["text"] for item in data["val"]]
        train_labels = [item["label"] for item in data["val"]]
        test_texts = [item["text"] for item in data["test"]]
        test_labels = [item["label"] for item in data["test"]]

        train_embeddings, train_targets = self.get_kls(train_texts, train_labels)
        test_embeddings, _ = self.get_kls(test_texts, test_labels)

        classifier = self.train(train_embeddings, train_targets)

        with torch.no_grad():
            scores = torch.softmax(classifier(test_embeddings), dim=1)[:, 1]

        return scores.detach().cpu().tolist(), test_labels
        
