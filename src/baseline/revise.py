from src.baseline.bartscore import BARTScorer

from tqdm import tqdm
import torch
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import random
from argparse import Namespace

from src.utils import return_device

print('GPU available', torch.cuda.is_available())

# Code is taken from: https://github.com/thunlp/LLM-generated-text-detection/blob/main/finance_bart_auroc.py

class ReviseDetect:

    def __init__(self,
                model_name="meta-llama/Meta-Llama-3-8B-instruct",
                workers=8):
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          trust_remote_code=True,
                                                          device_map='auto',
                                                          torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       trust_remote_code=True)
        self.model.eval()
        self.prompt = "Revise the following text: {text}"
        self.device = return_device()
        self.bartscorer = BARTScorer(device=return_device(),checkpoint="facebook/bart-large-cnn")


    def batch_inference(self, items: list[str]) -> list[dict]:
        """"
        GQ: we generate this function for faster inference of summaries
        takes ages w/o batching 
        """
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        batch_size = 16
        results = []

        for start in tqdm(range(0, len(items), batch_size)):
            batch_texts = items[start:start + batch_size]
            prompts = [self.prompt.format(text=text) for text in batch_texts]

            tokenized = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536
            )
            input_ids = tokenized.input_ids.to(self.device)
            attention_mask = tokenized.attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            input_lens = input_ids.shape[1]
            for idx, text in enumerate(batch_texts):
                gen_ids = outputs[idx, input_lens:]
                gec_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                results.append({"text": text, "revise_text": gec_text})

        return results
    
    def run(self, 
            data: dict[str, list[dict]],) -> list[float]:
        ''''''

        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        texts = [item["text"] for item in data['test']]

        texts_dict = self.batch_inference(texts)

        revised = [item['revise_text'] for item in texts_dict]
        og_texts = [item['text'] for item in texts_dict]

        scores = self.bartscorer.score(revised, og_texts)
        
        return scores, [item["label"] for item in data['test']]




