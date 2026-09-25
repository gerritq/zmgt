import os
import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data, return_device

from src.config import Config
cfg = Config()


class RepresentationalNovelty():
    """Log-determinant gain of each token over its (centered) preceding context, per layer."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.device = return_device()
        self.inference = Inference(model_name=args.model)

    @staticmethod
    def novelty(hidden_states: torch.Tensor, lam: float) -> torch.Tensor:
        """
        novelty_t = log det(S_{<=t} + λI) - log det(S_{<t} + λI) for t = 1..T-1,
        where S_{<t} is the scatter matrix of the centered states x_0..x_{t-1}.

        Computed in the dual (T x T) space. For n states X_n with Gram G_n = X_n X_nᵀ,
        centering and the matrix determinant lemma give

            log det(S_n + λI) = (D - n) log λ + log det(G_n + λI) + log(λ/n · 1ᵀ(G_n + λI)⁻¹ 1).

        The Cholesky factor of G_n + λI is the leading n x n block of the one for the
        full text, so a single Cholesky gives every prefix.

        Parameters
        ----------
        hidden_states : Tensor of shape (T, D)
        lam : regularizer, relative to the average per-dimension variance of the text

        Returns
        -------
        novelty per token t = 1..T-1: Tensor of shape (T-1,)
        """
        x = hidden_states.double()
        n_tokens, dim = x.shape
        if n_tokens < 2:
            return torch.empty(0, dtype=x.dtype, device=x.device)
        # The centered scatter is translation invariant; subtract the text mean for conditioning.
        x = x - x.mean(dim=0, keepdim=True)
        # Scale λ with the layer's variance so that it means the same thing across layers.
        lam = lam * x.pow(2).sum(dim=-1).mean().item() / dim

        eye = torch.eye(n_tokens, dtype=x.dtype, device=x.device)
        chol = torch.linalg.cholesky(x @ x.T + lam * eye)
        ones = torch.ones(n_tokens, 1, dtype=x.dtype, device=x.device)
        z = torch.linalg.solve_triangular(chol, ones, upper=False).squeeze(-1)

        # mean_term[n-1] = log(1ᵀ(G_n + λI)⁻¹ 1 / n)
        counts = torch.arange(1, n_tokens + 1, dtype=x.dtype, device=x.device)
        mean_term = torch.log(torch.cumsum(z.pow(2), dim=0) / counts)
        log_diag = torch.log(torch.diagonal(chol))

        return 2 * log_diag[1:] + mean_term[1:] - mean_term[:-1] - math.log(lam)

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> np.ndarray:
        """Mean novelty over tokens, per layer (0 = embeddings): (L+1,)."""
        scores = []
        for layer in hidden_states:
            layer = layer[self.args.skip_tokens:].to(self.device)
            novelty = self.novelty(layer, self.args.lam)
            scores.append(novelty.mean().item() if novelty.numel() else float("nan"))
        # Lower novelty = more predictable continuation, taken as more machine-like.
        return -1 * np.asarray(scores, dtype=float)

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (too short texts give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def plot(self, aurocs: list[float], path: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(range(len(aurocs)), aurocs, marker="o", markersize=3)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Layer (0 = embeddings)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"Log-det novelty (λ={self.args.lam}): {self.args.model}, {self.args.dataset}")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        scores = []
        for item in tqdm(test_data, desc="Collecting representational novelty scores"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            scores.append(self.score(hidden_states))
        scores = np.asarray(scores, dtype=float)  # (N, L+1)

        metrics_per_layer = [self.evaluate(labels, scores[:, layer]) for layer in range(scores.shape[1])]
        # Single summary score: novelty averaged over all layers.
        metrics = self.evaluate(labels, scores.mean(axis=1))
        aurocs = [layer_metrics["auroc"] for layer_metrics in metrics_per_layer]
        print(json.dumps({"auroc_per_layer": aurocs, "metrics": metrics}, indent=4))

        file_name = f"representational_novelty_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "representational_novelty",
            "metrics": metrics,
            "metrics_per_layer": metrics_per_layer,
            "scores": scores.tolist(),
            "labels": labels.tolist(),
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot(aurocs, os.path.join(output_dir, f"{file_name}.pdf"))
        return metrics


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Regularizer λ as a multiple of the average per-dimension variance.")
    parser.add_argument("--skip_tokens", type=int, default=1,
                        help="Leading tokens to drop (the first token acts as an attention sink).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = RepresentationalNovelty(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
