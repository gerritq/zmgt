
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
import torch
import math
import json
import random
import torch.nn.functional as F
from torch import autocast
from scipy.stats import skew, kurtosis
import numpy as np
import os


# All code taken from: https://github.com/Shixuan-Ma/NTS/blob/main/TeSeN/detector.py


huggingface_config = {
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

class TeSeN():
    def __init__(self):
        super().__init__()

        M1_model_name = "tiiuae/falcon-7b"

        self.M1_model = AutoModelForCausalLM.from_pretrained(M1_model_name, trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16,
                                                                token=huggingface_config["TOKEN"])

        self.device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "")
        self.M1_model.to(self.device_1)
        self.M1_model.eval()

        self.M1_tokenizer = AutoTokenizer.from_pretrained(M1_model_name)
        self.M1_tokenizer.pad_token = self.M1_tokenizer.eos_token
        self.M1_tokenizer.pad_token_id = self.M1_tokenizer.eos_token_id or self.M1_tokenizer.add_special_tokens(
            {'pad_token': '[PAD]'})

    def Prob_calculating(self, text, model, tokenizer, flag=0, prompt=""):
        if flag == 0:
            tokenized = tokenizer(text, return_tensors="pt", return_token_type_ids=False,
                                  return_attention_mask=True, truncation=True, max_length=512).to(self.device_1)
        else:
            tokenized = tokenizer(text, return_tensors="pt", return_token_type_ids=False,
                                  return_attention_mask=True, truncation=True, max_length=512).to(self.device_2)

        tokenized["input_ids"] = tokenized["input_ids"].long()
        labels = tokenized["input_ids"][:, 1:]
        attention_mask = tokenized["attention_mask"]

        logits_score = model(**tokenized).logits[:, :-1]

        return logits_score, labels, attention_mask

    def getting_logits_labels_attention_mask(self, text, model, tokenizer, flag=0):
        logits, labels, attention_mask = self.Prob_calculating(text, model,
                                                               tokenizer, flag=flag)

        attention_mask = attention_mask.to(torch.float16)[:, 1:]
        return logits, labels, attention_mask

    def Temperature_Sensitivity_AVG(self, logits_M1, labels, attention_mask, T1=0.7, T2=1.4):
        logits = logits_M1 / T1

        lprobs1 = torch.log_softmax(logits, dim=-1)

        label_logprobs1 = lprobs1.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)

        logits = logits_M1 / T2

        lprobs2 = torch.log_softmax(logits, dim=-1)

        label_logprobs2 = lprobs2.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)

        if (attention_mask.sum(dim=-1).item()) != 0:
            low_T_log_PPL = ((((((label_logprobs1)) * attention_mask))).sum(
                dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())

            high_T_log_PPL = ((((((label_logprobs2)) * attention_mask))).sum(
                dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())

            TS = abs(low_T_log_PPL - high_T_log_PPL)
        else:
            TS = -5

        return TS

    def Temperature_Sensitivity_sample(self, logits_M1, labels, attention_mask, T1=0.6, T2=1.4):
        logits = logits_M1

        lprobs1 = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(lprobs1)

        logits = logits_M1 / T1

        lprobs1 = torch.log_softmax(logits, dim=-1)

        label_logprobs1 = lprobs1.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)

        logits = logits_M1 / T2

        lprobs2 = torch.log_softmax(logits, dim=-1)

        label_logprobs2 = lprobs2.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)

        per_token_ts = (lprobs1 - lprobs2)
        if (attention_mask.sum(dim=-1).item()) != 0:
            low_T_log_PPL = ((((((label_logprobs1)) * attention_mask))).sum(
                dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())

            high_T_log_PPL = ((((((label_logprobs2)) * attention_mask))).sum(
                dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())

            TS = abs(low_T_log_PPL - high_T_log_PPL)

            E = (((per_token_ts * probs).sum(dim=-1)) * attention_mask).sum(
                dim=-1).sum(dim=-1).item() / (attention_mask.sum(dim=-1).item())
            per_token_std = (((per_token_ts) ** 2) * probs).sum(dim=-1) - ((per_token_ts * probs).sum(dim=-1)) ** 2
            std = (((((per_token_std.sqrt()) * attention_mask))).sum(
                dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())
            Ts = (TS - E) / std
        else:
            Ts = -10

        return Ts

    def log_p_derative(self, logits_M1, labels, attention_mask, T=1):
        logits = logits_M1 / T
        lprobs1 = torch.log_softmax(logits, dim=-1)
        probs1 = torch.exp(lprobs1)

        logits_hat = (logits_M1 * probs1).sum(dim=-1)
        label_logits1 = logits_M1.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)

        derative = logits_hat - label_logits1

        if (attention_mask.sum(dim=-1).item()) != 0:
            derative = ((((((derative)) * attention_mask))).sum(
                dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())
        else:
            derative = 0
        return derative / (T ** 2)

    def log_p(self, logits_M1, labels, attention_mask, T=1):
        logits = logits_M1 / T
        lprobs1 = torch.log_softmax(logits, dim=-1)
        probs = lprobs1.exp()
        label_lprobs1 = lprobs1.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)

        if (attention_mask.sum(dim=-1).item()) != 0:
            logp = ((((((label_lprobs1)) * attention_mask))).sum(
                dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())
        else:
            logp = 0

        return logp

    # def compute_crit(self, text, mode):
    #     if mode == 'TS_avg':
    #         logits, labels, attention_mask = self.getting_logits_labels_attention_mask(text, self.M1_model,
    #                                                                                    self.M1_tokenizer, flag=0)
    #         feature_score = self.Temperature_Sensitivity_AVG(logits, labels, attention_mask, T1=0.7, T2=1.4)

    #     elif mode == 'TS_norm':
    #         logits, labels, attention_mask = self.getting_logits_labels_attention_mask(text, self.M1_model,
    #                                                                                    self.M1_tokenizer, flag=0)
    #         feature_score = self.Temperature_Sensitivity_sample(logits, labels, attention_mask, T1=0.7, T2=1.4)

    #     return feature_score

    @torch.inference_mode()
    def run(self, data: dict[str, list[dict]],) -> list[float]:

        scores = []
        for item in data['test']:
            text = item['text']
            # this is the implementation of def compute_crit for mode "TS_norm" (commented out above)
            logits, labels, attention_mask = self.getting_logits_labels_attention_mask(text, self.M1_model,
                                                                                        self.M1_tokenizer, flag=0)
            feature_score = self.Temperature_Sensitivity_sample(logits, labels, attention_mask, T1=0.7, T2=1.4)

            scores.append(feature_score)
        return scores, [item["label"] for item in data['test']]