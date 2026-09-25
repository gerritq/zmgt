import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm

from src.baseline.IntrinsicDim import PHD

import os
from datetime import datetime
from argparse import Namespace
from datasets import Dataset
import json

from transformers import (set_seed)



BASE_DIR = os.getenv("BASE_COE")

class IDEstimator:
    def __init__(self,):
        self.model_path = 'FacebookAI/xlm-roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)

        self.MIN_SUBSAMPLE = 40 
        self.INTERMEDIATE_POINTS = 7

        self.model.eval()

    def preprocess_text(self, text):
        return text.replace('\n', ' ').replace('  ', ' ')
    
    def get_phd_batch(self, items: list[str], alpha: float = 1.0, batch_size: int = 16):
        dims = []
        solver = PHD(alpha=alpha, metric="euclidean", n_points=9)

        for i in tqdm(range(0, len(items), batch_size)):
            batch_texts = [self.preprocess_text(t) for t in items[i:i + batch_size]]

            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                    outp = self.model(**inputs) # B x L x D
                    hidden = outp.last_hidden_state
                    attn = inputs["attention_mask"]
            for j in range(outp[0].shape[0]):
                seq_len = int(attn[j].sum().item())
                token_vecs = hidden[j, 1:seq_len - 1, :].detach().cpu().numpy()

                mx_points = token_vecs.shape[0]
                mn_points = self.MIN_SUBSAMPLE

                if mx_points < mn_points:
                    # need this fallback for multisocial 
                    dims.append(None)
                    continue

                step = ( mx_points - mn_points ) // self.INTERMEDIATE_POINTS
                if step == 0:
                    dims.append(None)
                    continue

                phd_val = solver.fit_transform(token_vecs,  min_points=mn_points, max_points=mx_points - step, \
                                point_jump=step)
                dims.append(phd_val)

        return np.array(dims, dtype=object).reshape(-1, 1)

    # def get_phd_single(self, text, solver):
    #     inputs = self.tokenizer(self.preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
    #     with torch.no_grad():
    #         outp = self.model(**inputs)
        
    #     # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
    #     mx_points = inputs['input_ids'].shape[1] - 2

        
    #     mn_points = self.MIN_SUBSAMPLE
    #     step = ( mx_points - mn_points ) // self.INTERMEDIATE_POINTS
            
    #     return solver.fit_transform(outp[0][0].numpy()[1:-1],  min_points=mn_points, max_points=mx_points - step, \
    #                                 point_jump=step)
    
    # def get_phd(self, items: list[str], alpha=1.0):
    #     dims = []
    #     PHD_solver = PHD(alpha=alpha, metric='euclidean', n_points=9)
    #     for x in tqdm(items):
    #         dims.append(self.get_phd_single(x, PHD_solver))

    #     return np.array(dims).reshape(-1, 1)
    
    def learn_logistic_regression(self, 
                                 x: dict, 
                                 y: list[int]
                                 ) -> None:
        valid_idx = [i for i in range(len(x)) if x[i][0] is not None]
        x = np.array([float(x[i][0]) for i in valid_idx], dtype=np.float32).reshape(-1, 1)
        y = [y[i] for i in valid_idx]
        if len(x) == 0:
            raise ValueError("No valid PHD features found for training.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_scaled, y)

        self.scaler = scaler
        self.classifier = classifier

        return None

    def run(self, 
            data: dict[str, list[float,str]],
            ) -> None:
        

        set_seed(42)

        # Get PHD features for training
        x_train = [x['text'] for x in data['val']]
        y_train = [x['label'] for x in data['val']]
        X_train = self.get_phd_batch(x_train)

        # Train logistic regression on PHD features
        self.learn_logistic_regression(X_train, y_train)

        # Test
        x_test = [x['text'] for x in data['test']]
        y_test = [x['label'] for x in data['test']]
        
        X_test = self.get_phd_batch(x_test)

        valid_idx = [i for i in range(len(X_test)) if X_test[i][0] is not None]
        
        X_test_valid = np.array([float(X_test[i][0]) for i in valid_idx], dtype=np.float32).reshape(-1, 1)
        y_test_valid = [y_test[i] for i in valid_idx]

        X_test_scaled = self.scaler.transform(X_test_valid)

        y_scores = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        return y_scores, y_test_valid
    
