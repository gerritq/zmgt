import numpy as np
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# Code taken from supplementary material of https://openreview.net/forum?id=2VdsYVXLDl

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

class IRM:
    def __init__(self,
                 base_model = 'meta-llama/Llama-3.2-1B-Instruct',
                 reference_model = 'meta-llama/Llama-3.2-1B') -> None:

        # Scoring model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model,
                                                                  trust_remote_code=True,
                                                                  device_map='auto') # .to(self.device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.base_tokenizer.pad_token_id is None:
            self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        # Reference model
        self.reference_model = AutoModelForCausalLM.from_pretrained(reference_model,
                                                                  trust_remote_code=True,
                                                                  device_map='auto') # .to(self.device)
        self.reference_tokenizer = AutoTokenizer.from_pretrained(reference_model)
        if self.reference_tokenizer.pad_token_id is None:
            self.reference_tokenizer.pad_token_id = self.reference_tokenizer.eos_token_id

        self.base_model.eval()
        self.reference_model.eval()

    def get_likelihood(self, logits, labels, return_sum=False):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1).to(logits.device)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        if return_sum:
            return log_likelihood.sum().item()
        return log_likelihood.mean().item()

    def get_score(self, text: str) -> float:
        with torch.no_grad():
            tokenized = self.base_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(self.base_model.device)
            labels = tokenized.input_ids[:, 1:]

            base_inputs = tokenized.to(self.base_model.device)
            logits = self.base_model(**base_inputs).logits[:, :-1]
            likelihood = self.get_likelihood(logits, labels, return_sum=True)

            ref_inputs = tokenized.to(self.reference_model.device)
            logits = self.reference_model(**ref_inputs).logits[:, :-1]
            ref_likelihood = self.get_likelihood(logits, labels, return_sum=True)
            
            score = likelihood - ref_likelihood
            return score


    def run(self, 
            data: dict[str, list[dict]],) -> list[float]:
        
        texts = [item["text"] for item in data['test']]
        
        results= []
        for i, text in enumerate(texts):
            score = self.get_score(text)
            results.append(score)

        return results, [item["label"] for item in data['test']]