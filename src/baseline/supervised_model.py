from argparse import Namespace

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import return_device

class SupervisedModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = return_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        attn_impl = "eager" if "radar" in self.model_name.lower() else None
        self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                attn_implementation=attn_impl,
                use_safetensors=True,
            ).to(self.device)
        self.model.eval()

    def run(self, data: dict[str, list[dict]],) -> list[float]:
        texts = [d["text"] for d in data['test']]
        batch_size = 16
        scores: list[float] = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512, 
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits

                probs = torch.softmax(logits, dim=-1)
                batch_scores = probs[:, 1].detach().cpu().tolist()

                # if args.model in {"radar", "openai_roberta"}:
                batch_scores = [-float(s) for s in batch_scores]
                scores.extend(float(s) for s in batch_scores)

        return scores, [item["label"] for item in data['test']]
