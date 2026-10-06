"""Per-text Information Imbalance Δ(ℓ → first) and Δ(ℓ → last) across layers, coloured by human/machine."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data, return_device

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "layer_desc")
COLORS = {0: "tab:blue", 1: "tab:red"}
NAMES = {0: "Human", 1: "Machine"}
PEAK_MAX_LAYER = 20


def distances(hidden_states: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances between tokens (T, D) -> (T, T), self-distance set to inf."""
    hidden_states = hidden_states.float()
    dist = torch.cdist(hidden_states, hidden_states, compute_mode="donot_use_mm_for_euclid_dist")
    dist.fill_diagonal_(float("inf"))
    return dist


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


def compute_imbalance(args: Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Δ(ℓ → first) and Δ(ℓ → last) per text and transformer layer: (N, L) each, plus labels (N,)."""
    data = load_data(args=args)[args.split]
    inference = Inference(model_name=args.model)
    device = return_device()

    to_first, to_last = [], []
    for item in tqdm(data, desc="Computing Δ(ℓ → first), Δ(ℓ → last)"):
        hidden_states = inference.run(item, args)["hidden_states"]
        # Omit the embedding output and use transformer layers.
        dists = [distances(layer.to(device)) for layer in hidden_states[1:]]
        to_first.append([imbalance(dist, dists[0]) for dist in dists])
        to_last.append([imbalance(dist, dists[-1]) for dist in dists])
    labels = np.asarray([item["label"] for item in data])
    return np.asarray(to_first, dtype=float), np.asarray(to_last, dtype=float), labels


def global_peak(scores: np.ndarray, layers: np.ndarray) -> int:
    """Index of the global maximum."""
    return int(np.nanargmax(scores))


def peak_before(scores: np.ndarray, layers: np.ndarray, max_layer: int = PEAK_MAX_LAYER) -> int:
    """Index of the maximum among layers before `max_layer`."""
    idx = np.flatnonzero(layers < max_layer)
    return int(idx[np.nanargmax(scores[idx])])


def first_local_peak(scores: np.ndarray, layers: np.ndarray) -> int:
    """Index of the first layer higher than both neighbours; falls back to the global maximum if none."""
    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
            return i
    return global_peak(scores, layers)


PEAK_FNS = [
    (global_peak, "Raw peak (global max)"),
    (peak_before, f"Max before layer {PEAK_MAX_LAYER}"),
    (first_local_peak, "First local max"),
]


def draw_row(row: np.ndarray, values: np.ndarray, layers: np.ndarray, labels: np.ndarray, ylabel: str) -> None:
    """One panel per peak definition: a transparent line per text, with its peak marked by a dot."""
    for ax, (peak_fn, desc) in zip(row, PEAK_FNS):
        for scores, label in zip(values, labels):
            if not np.isfinite(scores).any():
                continue
            peak = peak_fn(scores, layers)
            ax.plot(layers, scores, color=COLORS[label], alpha=0.15, linewidth=0.5)
            ax.scatter(layers[peak], scores[peak], color=COLORS[label], alpha=0.4, s=4, linewidths=0, zorder=3)
        ax.set_xlabel("Layer", fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        ax.set_title(desc, fontsize=7)
    row[0].set_ylabel(ylabel, fontsize=7)


def plot(to_first: np.ndarray, to_last: np.ndarray, labels: np.ndarray, args: Namespace, path: str) -> None:
    """Δ(ℓ → first) (row 1) and Δ(ℓ → last) (row 2) per text across transformer layers, one panel per peak definition."""
    layers = np.arange(1, to_first.shape[1] + 1)
    fig, axes = plt.subplots(2, len(PEAK_FNS), figsize=(9.6, 4.8), sharey="row", squeeze=False)
    draw_row(axes[0], to_first, layers, labels, r"$\Delta(\ell \rightarrow \ell_{\mathrm{first}})$")
    draw_row(axes[1], to_last, layers, labels, r"$\Delta(\ell \rightarrow \ell_{\mathrm{last}})$")
    for label, name in NAMES.items():
        axes[0, 0].plot([], [], color=COLORS[label], label=name)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"Information Imbalance: {args.model}, {args.dataset}", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = f"info_imbalance_{args.model}_{args.dataset}_{args.split}_s{args.seed}"

    to_first, to_last, labels = compute_imbalance(args)
    plot(to_first, to_last, labels, args, os.path.join(OUTPUT_DIR, f"{file_name}.pdf"))


if __name__ == "__main__":
    main()
