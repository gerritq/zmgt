import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace
import skdim
from src.config import Config

# All code taken from: https://github.com/mbzuai-nlp/DetectLLM
# Only modification is to change the tokenizer and model; and we create a class

class Geometric:
    def __init__(self,
                 method: str, 
                 device='cuda'):
        self.center = method.endswith("_c")
        self.method = method.removesuffix("_c")
        self.model_name = Config().model_dict["l8b"] 
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                        torch_dtype=torch.float16,
                                                        use_safetensors=True,).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    def inferece(self, text: str) -> dict:
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
    def determinant(hidden_states: torch.Tensor,
                    alpha=0.001) -> np.ndarray:
        """
        Adapted from ...
        
        Log-Determinant of a single layer tensor (T,H).

        Normalization and centering tokens seemed to have worked best. May ablate the choices.
        
        Parameters
        ----------
        hidden_states : Tensor of shape (T, D)

        Returns
        -------
        log-determinant: float
        
        """
        Z = hidden_states.float().T # (H,T)
        # center tokens
        Z = Z - Z.mean(dim=1, keepdim=True) # (H, L)
        # Norm hs
        Z = Z / Z.norm(dim=0, keepdim=True) # (H,T)
        # Gram matrix
        Sigma = Z.T @ Z   # (T, T)
        # Regularization (add small alpha to diagonal); avoid log of 0 sing value which is -infinity
        Sigma = Sigma + alpha * torch.eye(Sigma.shape[0], device=Sigma.device)
        # Get singular values
        svdvals = torch.linalg.svdvals(Sigma)
        # Get the log of singular values and take the mean
        eigscore = torch.log(svdvals).mean() # scalar
        return eigscore.item()

    @staticmethod
    def id_MLE(hidden_states: torch.Tensor) -> int:
        """
        Intrinsic dimensionality of the hidden states, measured with ML.

        Adapted from https://arxiv.org/pdf/2402.18048

        Parameters
        ----------
        hidden_states : Tensor of shape (T, D)

        Returns
        -------
        LID: float
        """
        lid = skdim.id.MLE().fit_transform(np.asarray(hidden_states, dtype=np.float64))
        return float(lid)
        
    @staticmethod
    def _consecutive_angles(vectors: torch.Tensor) -> torch.Tensor:
        """Return angles between consecutive vectors along the token axis."""
        if vectors.shape[0] < 2:
            return torch.empty(0, dtype=vectors.dtype, device=vectors.device)
        previous, following = vectors[:-1], vectors[1:]
        denominator = (previous.norm(dim=-1) * following.norm(dim=-1)).clamp_min(1e-12)
        cosine = ((previous * following).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    @staticmethod
    def _contextual_curvature(curvatures: torch.Tensor, window: int = 3) -> torch.Tensor:
        """Compute C^p_k = (c_{k-4} + c_{k-3} + c_{k-2}) / 3."""
        if curvatures.shape[0] < window:
            return torch.empty(0, dtype=curvatures.dtype, device=curvatures.device)
        return curvatures.unfold(0, window, 1).mean(dim=-1)

    @staticmethod
    def _mean_or_nan(values: torch.Tensor) -> float:
        """Return the mean score, retaining insufficient-token inputs as NaN."""
        return values.mean().item() if values.numel() else float("nan")

    def token_hs_curvature(self, hidden_states: torch.Tensor, center: bool = False) -> float:
        """
        Adapted from Hosseini et al. (2023).

        Applied to hidden states.

        Parameters
        ----------
        hidden_states : Tensor of shape (T, D)

        Returns
        -------
        mean curvature across tokens: float
        """

        hidden_states = hidden_states.float()
        if center:
            hidden_states = hidden_states - hidden_states.mean(dim=0, keepdim=True)
        return self._mean_or_nan(self._consecutive_angles(hidden_states))

    def token_dv_curvature(self, hidden_states: torch.Tensor, center: bool = False) -> float:
        """
        Trajectory curvature following Hosseini & Fedorenko (2023):
        angle between consecutive step vectors v_t = h_{t+1} - h_t.

        Parameters
        ----------
        hidden_states : Tensor of shape (T, D), T >= 3

        Returns
        -------
        mean curvature across tokens (radians): float
        """
        difference_vectors = hidden_states[1:].float() - hidden_states[:-1].float()
        if center:
            difference_vectors = difference_vectors - difference_vectors.mean(
                dim=0, keepdim=True
            )
        return self._mean_or_nan(self._consecutive_angles(difference_vectors))

    def token_hs_context_curvature(
        self, hidden_states: torch.Tensor, center: bool = False
    ) -> float:
        """Return the mean three-curvature context score over token hidden states."""
        hidden_states = hidden_states.float()
        if center:
            hidden_states = hidden_states - hidden_states.mean(dim=0, keepdim=True)
        curvatures = self._consecutive_angles(hidden_states)
        return self._mean_or_nan(self._contextual_curvature(curvatures))

    def token_dv_context_curvature(
        self, hidden_states: torch.Tensor, center: bool = False
    ) -> float:
        """Return the mean three-curvature context score over token-step vectors."""
        difference_vectors = hidden_states[1:].float() - hidden_states[:-1].float()
        if center:
            difference_vectors = difference_vectors - difference_vectors.mean(
                dim=0, keepdim=True
            )
        curvatures = self._consecutive_angles(difference_vectors)
        return self._mean_or_nan(self._contextual_curvature(curvatures))

    def run(self,
            data: dict[str, list[dict]],) -> list[float]:
        
        texts = [item["text"] for item in data['test']]
        
        scores= []
        target_layer = 12
        for i, text in enumerate(texts):
            layer_scores = []
            hidden_states = self.inferece(text)[1:] # tuple of (T, D) tensors; rm embeding layer
            
            if self.method in [
                "curvature",
                "curvature_dv",
                "context_curvature",
                "context_curvature_dv",
                "curvature_our_context",
            ]:
                for l in range(len(hidden_states)):
                    if self.method == "curvature":
                        curv = self.token_hs_curvature(hidden_states[l], self.center)
                    elif self.method == "curvature_dv":
                        curv = self.token_dv_curvature(hidden_states[l], self.center)
                    elif self.method == "context_curvature":
                        curv = self.token_hs_context_curvature(
                            hidden_states[l], self.center
                        )
                    else:
                        curv = self.token_dv_context_curvature(
                            hidden_states[l], self.center
                        )
                    layer_scores.append(curv)

                # delta_layer_scores = [(layer_scores[l] - layer_scores[0]) for l in range(len(layer_scores))]
                # score = np.mean(delta_layer_scores)
                csd_layer = 10  # same layer as CSD-Detect (hidden_states[1:] indexing)
                score = layer_scores[csd_layer] / layer_scores[0]
            if self.method == "log_determinant":
                score = self.determinant(hidden_states[target_layer])
            if self.method == "id_mle":
                score = self.id_MLE(hidden_states[target_layer])

            scores.append(-1 * score)

        return scores, [item["label"] for item in data['test']]
