from typing import Any
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from src.utils import return_device
from argparse import Namespace

from src.config import Config
cf = Config()

class Inference:
    def __init__(self, 
                 model_name: str,
                 args: Namespace | None = None) -> None:
        self.model_name = model_name
        self.model_id = cf.model_dict[model_name]
        self.device = return_device()

        model_kwargs = {"device_map": self.device}
        if getattr(args, "return_attentions", False):
            model_kwargs["attn_implementation"] = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def run(self, 
            item: dict, 
            args: Namespace) -> dict[str, Any]:
        """

        Parameters
        ------------------
        @item: a dict containing text and label (machine/human)
        @args: Namespace containing cli for returning logits/attention

        Return
        ------------------
        @out: a dict containing the model outputs (hidden states, logits, attentions) with text and label

        hs: tuple[torch.Tensor, ...] with shape  (T,H)
        logits: torch.Tensor with shape (T,V) where V is the vocab size
        attentions: tuple[torch.Tensor, ...] with shape (n_heads,T,T) 
        token_ids: torch.Tensor with shape (T,)
        """
        
        text = item["text"]
        
        if not text.strip():
            raise ValueError("Input text must be non-empty.")

        inputs = self.tokenizer(text, 
                                truncation=True,
                                add_special_tokens=False,
                                max_length=1024,
                                return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, 
                                 output_hidden_states=True, 
                                 output_attentions=getattr(args, "return_attentions", False),
                                 use_cache=False)
            
        # hidden size dim (batch_size, seq_len, hidden_dim)
        # squeeze(0) removes the batch dimensions which is 1
        out = {
            "model_id": self.model_id,
            "text": text,
            "label": item["label"],
            "n_tokens": int(inputs["input_ids"].shape[-1]),
            "hidden_states": tuple(layer.detach().cpu().squeeze(0).float() for layer in outputs.hidden_states),
        }

        # return logits
        if getattr(args, "return_logits", False):
            out["logits"] = outputs.logits.detach().cpu().squeeze(0).float()

        # return attention
        if getattr(args, "return_attentions", False):
            out["attentions"] = tuple(layer.detach().cpu().squeeze(0).float() for layer in outputs.attentions)

        # return token ids
        if getattr(args, "return_token_ids", False):
            out["token_ids"] = inputs["input_ids"].detach().cpu().squeeze(0).long()
        return out
