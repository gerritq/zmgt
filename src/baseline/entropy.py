import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace

class Entropy:
    
    def __init__(self, 
                 model_name: str = "EleutherAI/gpt-neo-2.7B", 
                 device: str = 'cuda'):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          trust_remote_code=True,
                                                          device_map='auto',
                                                          torch_dtype=torch.bfloat16,
                                                          use_safetensors=True,)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                          trust_remote_code=True)
        self.model.eval()

    def _get_entropy(self, text: str):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device) # input_ids + mask
            logits = self.model(**tokenized).logits[:, :-1]
            neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
            return neg_entropy.sum(-1).mean().item()
        
    def run(self, 
            data: dict[str, list[dict]]) -> list[float]:
        '''wrapper function to run get_entropy for list of txts'''
        texts = [item["text"] for item in data['test']]
        scores = []
        for text in texts:
            score = self._get_entropy(text)
            scores.append(score)
        return scores, [item["label"] for item in data['test']]
