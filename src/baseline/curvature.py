import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import Config

class Curvature:
    def __init__(self,
                 method: str = "curvature",
                 device='cuda'):
        if method not in ("curvature", "context_curvature"):
            raise ValueError(f"Invalid method: {method}")
        self.method = method
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.device = device
        self.target_layer = 10
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
    def curvature(hidden_states: torch.Tensor) -> torch.Tensor:
        """Curvature alla Hosseini & Fedorenko (2023): angle between
        consecutive step vectors v_k = x_{k+1} - x_k."""
        hidden_states = hidden_states.float()
        vectors = hidden_states[1:] - hidden_states[:-1]
        previous, following = vectors[:-1], vectors[1:]
        denominator = (previous.norm(dim=-1) * following.norm(dim=-1)).clamp_min(1e-12)
        cosine = ((previous * following).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    @staticmethod
    def context_curvature(curvatures: torch.Tensor, window: int = 3) -> torch.Tensor:
        """Contextual curvature alla King et al. (2026):
        C_k = (c_{k-4} + c_{k-3} + c_{k-2}) / 3, i.e. a backward-looking
        window of three curvatures per token."""
        if curvatures.shape[0] < window:
            return torch.empty(0, dtype=curvatures.dtype)
        return curvatures.unfold(0, window, 1).mean(dim=-1)

    def score_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        curvatures = self.curvature(hidden_states)
        if self.method == "context_curvature":
            curvatures = self.context_curvature(curvatures)
        return curvatures.mean()

    def run(self,
            data: dict[str, list[dict]],) -> list[float]:
        
        texts = [item["text"] for item in data['test']]
        
        scores= []
        for i, text in enumerate(texts):
            hidden_states = self.inference(text)[1:] # tuple of (T, D) tensors; rm embeding layer
            
            curv_zero = self.score_layer(hidden_states[0])
            curv_target = self.score_layer(hidden_states[self.target_layer])
            score = (curv_target - curv_zero).item()
            scores.append(-1 * score)

        return scores, [item["label"] for item in data['test']]
