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


class InfoImbalance():
    """Information Imbalance Δ(A → B) between the token states of one text at two layers."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.device = return_device()
        self.inference = Inference(model_name=args.model)

    @staticmethod
    def distances(hidden_states: torch.Tensor) -> torch.Tensor:
        """Pairwise Euclidean distances between tokens (T, D) -> (T, T), self-distance set to inf."""
        hidden_states = hidden_states.float()
        dist = torch.cdist(hidden_states, hidden_states, compute_mode="donot_use_mm_for_euclid_dist")
        dist.fill_diagonal_(float("inf"))
        return dist

    @staticmethod
    def imbalance(dist_a: torch.Tensor, dist_b: torch.Tensor) -> float:
        """
        Δ(A → B) = 2/T · mean_i r_i, where r_i is the rank at B (1 = nearest) of i's
        nearest neighbour at A. ≈ 2/T if A's neighbourhoods predict B's, ≈ 1 if unrelated.
        """
        n_tokens = dist_a.shape[0]
        if n_tokens < 3:
            return float("nan")
        nearest_a = dist_a.argmin(dim=-1)
        dist_to_nearest = dist_b.gather(1, nearest_a.unsqueeze(-1))
        ranks = (dist_b < dist_to_nearest).sum(dim=-1) + 1
        return (2 * ranks.float().mean() / n_tokens).item()

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> tuple[float, np.ndarray, np.ndarray]:
        """Return (Δ(layer_a → layer_b), Δ(ℓ → first) and Δ(ℓ → last) for every layer ℓ)."""
        # Omit the embedding output and use transformer layers.
        dists = [self.distances(layer.to(self.device)) for layer in hidden_states[1:]]
        score = self.imbalance(dists[self.args.layer_a], dists[self.args.layer_b])
        to_first = np.asarray([self.imbalance(dist, dists[0]) for dist in dists])
        to_last = np.asarray([self.imbalance(dist, dists[-1]) for dist in dists])
        return score, to_first, to_last

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (texts under 3 tokens give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        scores, to_first, to_last = [], [], []
        for item in tqdm(test_data, desc="Collecting information imbalance scores"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            score, text_to_first, text_to_last = self.score(hidden_states)
            scores.append(score)
            to_first.append(text_to_first)
            to_last.append(text_to_last)
        scores = np.asarray(scores, dtype=float)
        to_first = np.asarray(to_first, dtype=float)  # (N, L)
        to_last = np.asarray(to_last, dtype=float)  # (N, L)

        metrics = self.evaluate(labels, scores)
        metrics_to_first = [self.evaluate(labels, to_first[:, layer]) for layer in range(to_first.shape[1])]
        metrics_to_last = [self.evaluate(labels, to_last[:, layer]) for layer in range(to_last.shape[1])]
        print(json.dumps({
            "metrics": metrics,
            "auroc_to_first": [m["auroc"] for m in metrics_to_first],
            "auroc_to_last": [m["auroc"] for m in metrics_to_last],
        }, indent=4))

        file_name = f"info_imbalance_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "info_imbalance",
            "metrics": metrics,
            "metrics_to_first": metrics_to_first,
            "metrics_to_last": metrics_to_last,
            "scores": scores.tolist(),
            "scores_to_first": to_first.tolist(),
            "scores_to_last": to_last.tolist(),
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
    parser.add_argument("--layer_a", type=int, default=10,
                        help="Layer A (transformer layers, embeddings excluded) whose nearest neighbours are used.")
    parser.add_argument("--layer_b", type=int, default=0,
                        help="Layer B at which those neighbours are ranked.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = InfoImbalance(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
