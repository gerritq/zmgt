# Copied from src/baseline/fdgpt.py for the runtime analysis (src.baseline_time): run() is replaced by score(),
# which scores one text (batch size 1). Everything else is unchanged.
import numpy as np
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace

# All code taken form https://github.com/baoguangsheng/fast-detect-gpt

class FastDetectGPT:

    def __init__(self, 
                scoring_model = "EleutherAI/gpt-j-6B",
                reference_model = "EleutherAI/gpt-neo-2.7B",
                device='cuda') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model,
                                                                  trust_remote_code=True,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.bfloat16,
                                                                  use_safetensors=True,) # .to(self.device)
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model, 
                                                                  trust_remote_code=True)
        
        self.reference_model = AutoModelForCausalLM.from_pretrained(reference_model,
                                                                    torch_dtype=torch.bfloat16,
                                                                    device_map='auto',
                                                                    use_safetensors=True,) # .to(self.device)
        self.reference_tokenizer = AutoTokenizer.from_pretrained(reference_model)

        # eval mode
        self.scoring_model.eval()
        self.reference_model.eval()

    def _get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()

    def score(self, text: str) -> float:
        """Fast-DetectGPT score of one text (the body of the loop in run() of the original)."""
        tokenized = self.scoring_tokenizer(text, return_tensors="pt", truncation=True, max_length=1536,
                                    return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]

            if self.reference_model == self.scoring_model:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(text, return_tensors="pt", truncation=True, max_length=1536,
                                        return_token_type_ids=False).to(self.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]

            return self._get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)
