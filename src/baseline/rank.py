import torch
from argparse import Namespace
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM)
class Rank:

    def __init__(self, 
                 model_name="EleutherAI/gpt-neo-2.7B",
                 device='cuda'):
        self.device = device
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        torch_dtype=torch.float16,
                                                        use_safetensors=True,).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _get_rank(self, text, log=False):
        with torch.no_grad():
            if text == "":
                return None
            else:
                tokenized = self.tokenizer(text, 
                                           return_tensors="pt",
                                           max_length=512,
                                           truncation=True).to(self.device)
                logits = self.model(**tokenized).logits[:, :-1]
                labels = tokenized.input_ids[:, 1:]

                matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

                assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

                ranks, timesteps = matches[:, -1], matches[:, -2]

                assert (timesteps == torch.arange(len(timesteps)).to(
                    timesteps.device)).all(), "Expected one match per timestep"

                ranks = ranks.float() + 1
                if log:
                    ranks = torch.log(ranks)

                return -ranks.float().mean().item()


    def run(self, 
            data: dict[str, list[dict]],) -> list[float]:
        texts = [item["text"] for item in data['test']]
        return [self._get_rank(text) for text in texts], [item["label"] for item in data['test']]