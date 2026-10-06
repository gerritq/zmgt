import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data

from src.config import Config
cfg = Config()

METHODS = (
    "curvature_hs",
    "curvature_context_against_current",
    "curvature_full_context_against_current",
    "curvature_context_against_context",
)


class Curvature():

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.method = args.method
        self.target_layer = 10
        self.inference = Inference(model_name=args.model)

    @staticmethod
    def _angles(previous: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
        """Row-wise angle between two (N, D) tensors."""
        denominator = (previous.norm(dim=-1) * following.norm(dim=-1)).clamp_min(1e-12)
        cosine = ((previous * following).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    @staticmethod
    def curvature_hs(hidden_states: torch.Tensor) -> torch.Tensor:
        """Curvature alla Hoesseini but with hidden states"""
        hidden_states = hidden_states.float()
        return Curvature._angles(hidden_states[:-1], hidden_states[1:])

    @staticmethod
    def curvature_context_against_current(hidden_states: torch.Tensor, window: int = 3) -> torch.Tensor:
        """For every token k >= window: angle between mean(x_{k-window}, ..., x_{k-1}) and x_k."""
        hidden_states = hidden_states.float()
        if hidden_states.shape[0] <= window:
            return torch.empty(0)
        pooled = hidden_states.unfold(0, window, 1).mean(dim=-1)  # (T-window+1, D); row j = mean(x_j..x_{j+window-1})
        return Curvature._angles(pooled[:-1], hidden_states[window:])

    @staticmethod
    def curvature_full_context_against_current(hidden_states: torch.Tensor) -> torch.Tensor:
        """For every token k >= 1: angle between mean(x_0, ..., x_{k-1}) and x_k."""
        hidden_states = hidden_states.float()
        if hidden_states.shape[0] < 2:
            return torch.empty(0)
        counts = torch.arange(1, hidden_states.shape[0], dtype=hidden_states.dtype).unsqueeze(-1)
        pooled = hidden_states.cumsum(dim=0)[:-1] / counts  # (T-1, D); row j = mean(x_0..x_j)
        return Curvature._angles(pooled, hidden_states[1:])

    @staticmethod
    def curvature_context_against_context(hidden_states: torch.Tensor, window: int = 3) -> torch.Tensor:
        """
        For every token t with window <= t <= T - window: angle between the backward context
        mean(x_{t-window}, ..., x_{t-1}) and the forward context mean(x_t, ..., x_{t+window-1}).
        """
        hidden_states = hidden_states.float()
        if hidden_states.shape[0] < 2 * window:
            return torch.empty(0)
        pooled = hidden_states.unfold(0, window, 1).mean(dim=-1)  # (T-window+1, D); row j = mean(x_j..x_{j+window-1})
        # Backward context of t is row t - window, forward context is row t.
        return Curvature._angles(pooled[:-window], pooled[window:])

    def score_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return getattr(self, self.method)(hidden_states).mean()

    def curvature_per_layer(self, hidden_states: tuple[torch.Tensor, ...]) -> np.ndarray:
        """Mean curvature of every transformer layer: (L,)."""
        # Omit the embedding output and use transformer layers.
        return np.asarray([self.score_layer(layer).item() for layer in hidden_states[1:]], dtype=float)

    @staticmethod
    def score_variants(curvature: np.ndarray) -> dict[str, np.ndarray]:
        """Text score of every layer per score variant, higher = more machine-like: (N, L) -> {variant: (N, L)}."""
        return {
            "raw": -1 * curvature,
            "ratio": -1 * curvature / curvature[:, :1],
        }

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        curvatures = []
        for item in tqdm(test_data, desc=f"Collecting {self.method} scores"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            curvatures.append(self.curvature_per_layer(hidden_states))
        variants = self.score_variants(np.asarray(curvatures, dtype=float))  # {variant: (N, L)}

        # Ratio of the mean curvature at the target layer (0-based index) to the first transformer layer.
        metrics = evaluation(labels, variants["ratio"][:, self.target_layer])
        print(json.dumps(metrics, indent=4))

        # Every layer as a fixed layer: {variant: {layer_ℓ: auroc}}.
        auroc_per_layer = {
            variant: {f"layer_{layer + 1}": self.evaluate(labels, v[:, layer])["auroc"] for layer in range(v.shape[1])}
            for variant, v in variants.items()
        }

        file_name = f"{self.method}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": self.method,
            "target_layer": self.target_layer,
            "metrics": metrics,
            "layer_numbering": "1..L over the transformer layers (embeddings excluded)",
            "auroc_per_layer": auroc_per_layer,
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        return metrics


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--method", type=str, choices=METHODS, default="curvature_hs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = Curvature(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
