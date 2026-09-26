import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data, return_device

from src.config import Config
cfg = Config()


class SubspaceResidual():
    """
    Fit one rank-r subspace (top-r principal components) to all token states of a text at a layer,
    then measure how far each token lies outside it.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.rank = args.rank
        self.eps = 1e-8
        self.device = return_device()
        self.inference = Inference(model_name=args.model)

    def residuals(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Centre the token states, H̃ = H − μ = U Σ Vᵀ, keep the top-r right singular vectors V_r
        (the top-r principal components), then e_t = ‖h̃_t − V_r V_rᵀ h̃_t‖² / (‖h̃_t‖² + ε) per token: (T,).
        """
        hidden_states = hidden_states.float()
        centred = hidden_states - hidden_states.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        v_r = vh[:self.rank].T  # (d, r)
        projected = centred @ v_r @ v_r.T
        residual = (centred - projected).pow(2).sum(dim=-1)
        return residual / (centred.pow(2).sum(dim=-1) + self.eps)

    def score_layer(self, hidden_states: torch.Tensor) -> float:
        """Mean residual over tokens."""
        if hidden_states.shape[0] <= self.rank:
            return float("nan")
        return self.residuals(hidden_states.to(self.device)).mean().item()

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> np.ndarray:
        """Mean residual for every transformer layer: (L,)."""
        # Omit the embedding output and use transformer layers.
        return np.asarray([self.score_layer(layer) for layer in hidden_states[1:]])

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (texts with at most r tokens give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        residuals = []
        for item in tqdm(test_data, desc=f"Collecting subspace residual scores (r={self.rank})"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            residuals.append(self.score(hidden_states))
        residuals = np.asarray(residuals, dtype=float)  # (N, L)

        # Negative so that higher = tokens better captured by the text's own rank-r subspace.
        scores = -1 * residuals
        scores_ratio = -1 * residuals / residuals[:, :1]

        metrics_per_layer = [self.evaluate(labels, scores[:, layer]) for layer in range(scores.shape[1])]
        auroc_per_layer = [m["auroc"] for m in metrics_per_layer]
        # Layer 0 is the reference itself (ratio = -1 for every text), so it is not evaluated.
        metrics_per_layer_ratio = [None] + [
            self.evaluate(labels, scores_ratio[:, layer]) for layer in range(1, scores_ratio.shape[1])
        ]
        auroc_per_layer_ratio = [None] + [m["auroc"] for m in metrics_per_layer_ratio[1:]]
        metrics = metrics_per_layer[args.layer]
        metrics_ratio = metrics_per_layer_ratio[args.layer]
        print(json.dumps({
            "metrics": metrics,
            "metrics_ratio": metrics_ratio,
            "auroc_per_layer": auroc_per_layer,
            "auroc_per_layer_ratio": auroc_per_layer_ratio,
        }, indent=4))

        file_name = f"csd_ngu_r{self.rank}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "csd_ngu",
            "metrics": metrics,
            "metrics_ratio": metrics_ratio,
            "metrics_per_layer": metrics_per_layer,
            "metrics_per_layer_ratio": metrics_per_layer_ratio,
            "auroc_per_layer": auroc_per_layer,
            "auroc_per_layer_ratio": auroc_per_layer_ratio,
            "scores": scores.tolist(),
            "scores_ratio": scores_ratio.tolist(),
            "labels": labels.tolist(),
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
    parser.add_argument("--rank", type=int, default=8, help="Number of top right singular vectors r kept.")
    parser.add_argument("--layer", type=int, default=10,
                        help="Transformer layer (embeddings excluded) reported as the main metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = SubspaceResidual(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
