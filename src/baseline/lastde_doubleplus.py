import numpy as np
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.baseline import fastMDE

import argparse


# All code taken form https://github.com/TrustMedia-zju/Lastde_Detector/blob/main/py_scripts/baselines/lastde_doubleplus.py
# We use Falcon-7B as the base model as in https://proceedings.neurips.cc/paper_files/paper/2025/file/0287c393907259ae6a269fca5e3509cd-Paper-Conference.pdf

class LastdeDoublePlus:

    def __init__(self,
                 scoring_model_name = "tiiuae/falcon-7b-instruct",
                 reference_model_name = "tiiuae/falcon-7b",
                device='cuda') -> None:

        self.scoring_model_name = scoring_model_name
        self.reference_model_name = reference_model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Scoring model
        self.scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model_name,
                                                                  trust_remote_code=True,
                                                                  device_map='auto') # .to(self.device)
        self.scoring_tokenizer = self.load_tokenizer(scoring_model_name)

        # Reference model
        self.reference_model = AutoModelForCausalLM.from_pretrained(reference_model_name,
                                                                  trust_remote_code=True,
                                                                  device_map='auto') # .to(self.device)
        self.reference_tokenizer = self.load_tokenizer(reference_model_name)
        
        self.scoring_model.eval()
        self.reference_model.eval()

        # ``baseline.py`` owns the main command-line parser and passes flags
        # such as ``--model`` and ``--dataset`` via ``sys.argv``. LastDE only
        # needs the options below, so leave the runner's flags untouched.
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--embed_size', type=int, default=4)
        parser.add_argument('--epsilon', type=float, default=8)
        parser.add_argument('--tau_prime', type=int, default=15)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--n_samples', type=int, default=100)
        self.args, _ = parser.parse_known_args()
            
    def load_tokenizer(self, model_name):

        optional_tok_kwargs = {}
        optional_tok_kwargs['padding_side'] = 'right'

        base_tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       trust_remote_code=True, 
                                                       **optional_tok_kwargs)
        if base_tokenizer.pad_token_id is None:
            base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        return base_tokenizer

    def get_samples(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        nsamples = self.args.n_samples
        # CUDA multinomial sampling does not support bfloat16. Keep the
        # models in bfloat16, but use float32 for this distribution.
        lprobs = torch.log_softmax(logits.float(), dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs)
        samples = distrib.sample([nsamples]).permute([1, 2, 0])
        return samples

    def get_likelihood(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        return log_likelihood

    def get_lastde(self, log_likelihood):
        embed_size = self.args.embed_size
        epsilon = int(self.args.epsilon * log_likelihood.shape[1])
        tau_prime = self.args.tau_prime

        templl = log_likelihood.mean(dim=1)
        aggmde = fastMDE.get_tau_multiscale_DE(ori_data = log_likelihood, embed_size=embed_size, epsilon=epsilon, tau_prime=tau_prime)
        lastde = templl / aggmde 
        return lastde


    def get_sampling_discrepancy(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        samples = self.get_samples(logits_ref, labels)
        log_likelihood_x = self.get_likelihood(logits_score, labels)
        log_likelihood_x_tilde = self.get_likelihood(logits_score, samples)


        # lastde
        lastde_x = self.get_lastde(log_likelihood_x)
        sampled_lastde = self.get_lastde(log_likelihood_x_tilde)

        miu_tilde = sampled_lastde.mean()
        sigma_tilde = sampled_lastde.std()
        discrepancy = (lastde_x - miu_tilde) / sigma_tilde

        return discrepancy.cpu().item()


    def run(self, 
            data: dict[str, list[dict]],) -> list[float]:
        '''wrapper '''

        # evaluate criterion
        name = "lastde_doubleplus"
        criterion_fn = self.get_sampling_discrepancy
    
        SEED = 42
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        texts = [item["text"] for item in data['test']]
        
        results = []
        
        for text in texts:
                
            # This is the lastde++ code
            tokenized = self.scoring_tokenizer(text, 
                                        return_tensors="pt", 
                                        padding=True, 
                                        truncation=True,
                                        max_length=1024,
                                        return_token_type_ids=False).to(self.device) 
            labels = tokenized.input_ids[:, 1:]
            # not optimal but need to guard against extremely short text, this is very rate
            if labels.shape[1] < self.args.embed_size:
                results.append(0.0)
                continue
            with torch.no_grad():
                logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                if self.reference_model_name == self.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = self.reference_tokenizer(text, 
                                                    return_tensors="pt", 
                                                    padding=True, 
                                                    truncation=True,
                                                    max_length=1024,
                                                    return_token_type_ids=False).to(self.device) 
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = self.reference_model(**tokenized).logits[:, :-1]
                crit = criterion_fn(logits_ref, logits_score, labels)

                results.append(crit)

        return results, [item["label"] for item in data['test']]