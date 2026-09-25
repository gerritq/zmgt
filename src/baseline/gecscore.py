import torch
import tqdm
from rouge import Rouge
import numpy as np
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace

from src.utils import return_device
rouge = Rouge()

class GECScore:

    def __init__(self,
                # "meta-llama/Llama-3.3-70B-Instruct"
                model_name="meta-llama/Meta-Llama-3-8B-instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          trust_remote_code=True,
                                                          device_map='auto',
                                                          torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       trust_remote_code=True)
        self.model.eval()
        self.prompt = "Correct the grammar errors in the following text: {text}\nCorrected text:"
        self.device = return_device()
    

    def batch_inference(self, items: list[str]) -> list[dict]:
        """"
        GQ: we generate this function for faster inference of summaries
        takes ages w/o batching 
        """
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        batch_size = 16
        results = []

        for start in tqdm.tqdm(range(0, len(items), batch_size)):
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
                results.append({"text": text, "gec_text": gec_text})

        return results

    def gescore(self, item: dict) -> float | None:
        text = (item.get("text") or "").strip()
        gec_text = (item.get("gec_text") or "").strip()

        # rouge package throws ValueError if either side is empty.
        # Treat failed/empty generations as missing.
        if not text or not gec_text:
            return None

        try:
            rouge_score = rouge.get_scores(text, gec_text, avg=True)
            return float(rouge_score["rouge-2"]["f"])
        except ValueError:
            return None

    def run(self, 
            data: dict[str, list[dict]],) -> list[float]:
        '''wrapper function to run get_samplnig_discrepency_analytic for list of txts'''

        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        texts = [item["text"] for item in data['test']]

        texts_dict = self.batch_inference(texts)

        scores = []
        
        for item in texts_dict:
            score = self.gescore(item)
            # we need a fallback, this is very rare
            scores.append(0.0 if score is None else score)

        return scores, [item["label"] for item in data['test']]