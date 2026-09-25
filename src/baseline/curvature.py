import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace
import skdim
from src.config import Config

class Curvature:
    def __init__(self,
                 method: str, 
                 device='cuda'):
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.device = device
        self.target_layer = 12
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                        torch_dtype=torch.float16,
                                                        use_safetensors=True,).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    def inference(self, text: str) -> dict:
        if not text.strip():
            raise ValueError("Input text must be non-empty.")
        inputs = self.tokenizer(text, 
                                truncation=True,
                                max_length=1024,
                                return_tensors="pt")

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, 
                                 output_hidden_states=True)
        hidden_states = tuple(layer.detach().cpu().squeeze(0).float() for layer in outputs.hidden_states)
        return hidden_states

    @staticmethod
    def curvature(vectors: torch.Tensor) -> torch.Tensor:
        """Curvature alla Hoesseini but with hidden states """
        hidden_states = hidden_states.float()
        previous, following = hidden_states[:-1], hidden_states[1:]
        denominator = (previous.norm(dim=-1) * following.norm(dim=-1)).clamp_min(1e-12)
        cosine = ((previous * following).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    def run(self,
            data: dict[str, list[dict]],) -> list[float]:
        
        texts = [item["text"] for item in data['test']]
        
        scores= []
        for i, text in enumerate(texts):
            layer_scores = []
            hidden_states = self.inferece(text)[1:] # tuple of (T, D) tensors; rm embeding layer
            
            curv_zero = self.curvature(hidden_states[0])
            curv_target = self.curvature(hidden_states[self.target_layer])
            score = curv_target / curv_zero
            scores.append(-1 * score)

        return scores, [item["label"] for item in data['test']]
