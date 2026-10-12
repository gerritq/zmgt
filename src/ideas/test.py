"""Per-layer shared-energy scores for a text's hidden states.

After dropping the first ``SKIP_TOKENS`` tokens, this compares the energy in
the text mean with the energy in its top 1 and top 5 uncentred principal
components. Hidden-state index 0 is the embedding output; indices 1..L are
the corresponding transformer-layer outputs.
"""

from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from src.inference import Inference
from src.utils import evaluation, load_data

SKIP_TOKENS = 10
TOP_K = (1, 5)
SCORE_NAMES = ("mean_centroid", "top1_uncentred", "top5_uncentred")


class SharedShare:
    """Measure the fraction of token energy in a shared text direction/subspace.

    For the retained hidden-state matrix H in R^(T x D):

      mean_centroid    = ||mean_t h_t||^2 / mean_t ||h_t||^2
      top{k}_uncentred = sum_{i=1}^k s_i(H)^2 / sum_i s_i(H)^2

    The latter is uncentred PCA/SVD. Equivalently, it is the mean squared
    norm of the tokens projected onto the top-k right-singular subspace,
    divided by their total mean squared norm.
    """

    METHOD = "shared_share"

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)

    @staticmethod
    def score_layer(h: torch.Tensor) -> dict[str, float]:
        """Return all scores for one layer's retained states, shaped (T, D)."""
        h = h.float()
        total_energy = h.square().sum()
        if h.shape[0] == 0 or not torch.isfinite(total_energy) or total_energy <= 0:
            return {name: float("nan") for name in SCORE_NAMES}

        # Since mean_t ||h_t||^2 = ||H||_F^2 / T, this is exactly the
        # requested ||h_bar||^2 / mean_t ||h_t||^2.
        centroid_share = h.mean(dim=0).square().sum() * h.shape[0] / total_energy

        # Singular-value energy equals projection energy. svdvals avoids
        # constructing singular vectors because only the energy share is used.
        singular_energy = torch.linalg.svdvals(h).square()
        scores = {"mean_centroid": centroid_share.item()}
        for k in TOP_K:
            scores[f"top{k}_uncentred"] = (singular_energy[:k].sum() / total_energy).item()
        return scores

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
        """Return one (L+1,) array per score; too-short texts produce NaNs."""
        n_layers = len(hidden_states)
        if hidden_states[0].shape[0] <= SKIP_TOKENS:
            return {name: np.full(n_layers, np.nan) for name in SCORE_NAMES}

        per_layer = [self.score_layer(h[SKIP_TOKENS:]) for h in hidden_states]
        return {
            name: np.asarray([layer_scores[name] for layer_scores in per_layer], dtype=float)
            for name in SCORE_NAMES
        }

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict | None:
        """Evaluate finite text scores; return None if both classes are unavailable."""
        valid = np.isfinite(scores)
        if valid.sum() < 2 or np.unique(labels[valid]).size < 2:
            return None
        metrics = evaluation(labels[valid], scores[valid])
        metrics["n_scored"] = int(valid.sum())
        return metrics

    @staticmethod
    def group_means(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
        out = {}
        for label, group in ((0, "human"), (1, "machine")):
            values = scores[(labels == label) & np.isfinite(scores)]
            out[group] = float(values.mean()) if len(values) else None
        return out

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])
        collected = {name: [] for name in SCORE_NAMES}

        for item in tqdm(test_data, desc=f"Collecting {self.METHOD}"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            for name, values in self.score(hidden_states).items():
                collected[name].append(values)

        values = {name: np.asarray(rows, dtype=float) for name, rows in collected.items()}
        n_layers = values[SCORE_NAMES[0]].shape[1]
        metrics = {
            name: [self.evaluate(labels, score[:, layer]) for layer in range(n_layers)]
            for name, score in values.items()
        }
        means = {
            name: [self.group_means(labels, score[:, layer]) for layer in range(n_layers)]
            for name, score in values.items()
        }

        print(f"\n{self.METHOD} | model: {args.model} | dataset: {args.dataset} | "
              f"seed: {args.seed} | first {SKIP_TOKENS} tokens dropped")
        print(f"{'layer':>7}" + "".join(f"{name:>24}" for name in SCORE_NAMES))
        for layer in range(n_layers):
            cells = []
            for name in SCORE_NAMES:
                result = metrics[name][layer]
                cells.append(f"{result['auroc']:.4f}" if result else "--")
            print(f"{layer:>7}" + "".join(f"{cell:>24}" for cell in cells))

        return {"metrics": metrics, "means": means, "scores": values, "labels": labels}


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    SharedShare(args).run(args)


if __name__ == "__main__":
    main()
