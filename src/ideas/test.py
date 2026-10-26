"""Per-layer change decomposition scores for a text's hidden states.

For a token's change Δ relative to a base state b, split Δ into the part along b and the rest:

  Δ_∥ = (Δ · b / ‖b‖²) b,   Δ_⊥ = Δ − Δ_∥

and keep the signed length Δ · b / ‖b‖ (> 0 = the state grows along b) and ‖Δ_⊥‖. Three kinds of change:

  across      Δ = h_t^ℓ − h_t^{ℓ−1},  b = h_t^{ℓ−1}               (the same token at the previous layer)
  within      Δ = h_t^ℓ − h_{t−1}^ℓ,  b = h_{t−1}^ℓ               (the previous token at the same layer)
  context{k}  Δ = h_t^ℓ − c_t^ℓ,     b = c_t^ℓ = mean_{j=1..k} h_{t−j}^ℓ   (the mean of the k previous tokens)

context1 equals within. Each text's score at a layer is the mean over its tokens. The first ``SKIP_TOKENS`` tokens are
dropped before anything is computed: the first token's norm is far larger and would dominate every mean. Hidden-state
index 0 is the embedding output; indices 1..L are the transformer-layer outputs (across is NaN at 0).
"""

from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from src.inference import Inference
from src.utils import evaluation, load_data

SKIP_TOKENS = 1
CONTEXT_K = (1, 4, 8, 16)
CHANGES = ("across", "within", *(f"context{k}" for k in CONTEXT_K))
PARTS = ("parallel", "perpendicular")
SCORE_NAMES = tuple(f"{change}_{part}" for change in CHANGES for part in PARTS)


class Decomposition:
    """Mean parallel and perpendicular part of each kind of change, per layer."""

    METHOD = "change_decomposition"

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)

    @staticmethod
    def decompose(delta: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Signed length of delta along base and the length of the rest: (..., D), (..., D) -> (...,), (...,)."""
        unit = base / base.norm(dim=-1, keepdim=True)
        parallel = (delta * unit).sum(dim=-1)
        return parallel, (delta - parallel[..., None] * unit).norm(dim=-1)

    @staticmethod
    def context(h: torch.Tensor, k: int) -> torch.Tensor:
        """Mean of the k previous tokens for positions k..T−1: (L + 1, T, D) -> (L + 1, T − k, D)."""
        sums = torch.cat([torch.zeros_like(h[:, :1]), h.cumsum(dim=1)], dim=1)  # sums[:, t] = Σ_{j<t} h_j
        return (sums[:, k:-1] - sums[:, :-k - 1]) / k

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
        """Return one (L+1,) array per score; too-short texts produce NaNs."""
        h = torch.stack(hidden_states)[:, SKIP_TOKENS:].to(self.inference.device).float()  # (L + 1, T, D)
        n_layers = h.shape[0]
        if h.shape[1] <= max(CONTEXT_K):
            return {name: np.full(n_layers, np.nan) for name in SCORE_NAMES}

        nan = torch.full((1,), float("nan"), device=h.device)
        parts = {
            "across": [torch.cat([nan, part.mean(dim=1)]) for part in self.decompose(h[1:] - h[:-1], h[:-1])],
            "within": [part.mean(dim=1) for part in self.decompose(h[:, 1:] - h[:, :-1], h[:, :-1])],
        }
        for k in CONTEXT_K:
            base = self.context(h, k)
            parts[f"context{k}"] = [part.mean(dim=1) for part in self.decompose(h[:, k:] - base, base)]
        return {f"{change}_{part}": values.cpu().numpy()
                for change, both in parts.items() for part, values in zip(PARTS, both)}

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
              f"seed: {args.seed} | first {SKIP_TOKENS} token(s) dropped | AUROC, machine positive")
        for part in PARTS:
            print(f"\n{part}")
            print(f"{'layer':>7}" + "".join(f"{change:>12}" for change in CHANGES))
            for layer in range(n_layers):
                cells = []
                for change in CHANGES:
                    result = metrics[f"{change}_{part}"][layer]
                    cells.append(f"{result['auroc']:.4f}" if result else "--")
                print(f"{layer:>7}" + "".join(f"{cell:>12}" for cell in cells))

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
    Decomposition(args).run(args)


if __name__ == "__main__":
    main()
