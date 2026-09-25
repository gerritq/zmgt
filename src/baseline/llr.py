import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace

# All code taken from: https://github.com/mbzuai-nlp/DetectLLM
# Only modification is to change the tokenizer and model; and we create a class

class LLR:
    def __init__(self,
                 method: str,
                 model_name = "EleutherAI/gpt-neo-2.7B", 
                 device='cuda'):
        self.method = method
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        torch_dtype=torch.float16,
                                                        use_safetensors=True,).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def get_rank(self, text, log=False):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            logits = self.model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:,1:]
            # get rank of each label token in the model's likelihood ordering
            matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
            assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
            ranks, timesteps = matches[:,-1], matches[:,-2]
            # make sure we got exactly one match for each timestep in the sequence
            assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"
            ranks = ranks.float() + 1 # convert to 1-indexed rank
            if log:
                ranks = torch.log(ranks)
            return ranks.float().mean().item()
    
    def get_ranks(self, texts, log=True):
        return [self.get_rank(text, log=log) for text in texts]
    
    def get_ll(self, text):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            labels = tokenized['input_ids']
            return -self.model(**tokenized, labels=labels).loss.item()
    
    def get_lls(self, texts):
        return [self.get_ll(text) for text in texts]
    
    def compute_llr(self, text):
        text_ll = self.get_ll(text)
        text_logrank = self.get_rank(text, log=True)
        return -text_ll / text_logrank
    
    # def compute_llrs(self, texts):
    #     return [self.compute_llr(text) for text in texts]

    def run(self, 
            data: dict[str, list[dict]],) -> list[float]:
        
        texts = [item["text"] for item in data['test']]
        
        results= []
        for i, text in enumerate(texts):
            try:
                if i > 0 and i % 20 == 0:
                    torch.cuda.empty_cache()
                    
                if self.method == "llr":
                # llr
                    results.append(self.compute_llr(text))
                if self.method == "likelihood":
                    results.append(self.get_ll(text))
                
            except Exception as e:
                print(f"Error processing text {i}: {str(e)}")
                results.append(float('nan'))
                # Try to recover by clearing memory after an error
                torch.cuda.empty_cache()
                
        return results, [item["label"] for item in data['test']]