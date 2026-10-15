"""
Plots of the energy decomposition norm2 = shared_centroid + token_specific from the aniso metrics
(output/metrics/aniso), one PDF per figure:
- f_aniso_auc_heatmap: AUC (machine vs. human) by layer for total energy, shared, token specific and shared share.
- f_aniso_energy: share of the machine − human gap in total energy carried by the shared and token-specific parts,
  g(part) / g(norm2) with g = mean over machine texts − mean over human texts; the two shares sum to 1 per layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.items.plot_style import configure_plot_style

cfg = Config()

BASE_DIR = Path(cfg.base_dir)
ANISO_DIR = BASE_DIR / "output" / "metrics" / "aniso"
OUTPUT_DIR = Path(cfg.item_output_dir) / "aniso"

HEATMAP_SCORES = (
    ("norm2", "Total energy"),
    ("shared_centroid", "Shared"),
    ("token_specific", "Token specific"),
    ("shared_share", "Shared share"),
)
PARTS = (("shared_centroid", "Shared", "#1f77b4"), ("token_specific", "Token specific", "#d62728"))
# Shares are meaningless where the norm2 gap is tiny or reversed: always drop the embedding, first and last layer,
# and any other layer whose norm2 gap is below MIN_REL_GAP of the human mean (left as a break in the lines).
EXCLUDED_LAYERS = (0, 1, 32)
MIN_REL_GAP = 0.01
# Entry i is hidden_states[i]: 0 = embedding output, 1..L = transformer-layer outputs.
XLABEL = "Layer (0 = embedding)"


def plot_heatmap(metrics: dict, title: str, path: Path) -> None:
    aucs = np.asarray([[entry["auroc"] for entry in metrics[score]] for score, _ in HEATMAP_SCORES], dtype=float)
    n_layers = aucs.shape[1]
    fig, ax = plt.subplots(figsize=(0.5 * n_layers + 3, 3.6), layout="constrained")
    image = ax.imshow(aucs, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    for (row, col), value in np.ndenumerate(aucs):
        if np.isfinite(value):
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value - 0.5) > 0.3 else "black")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(range(n_layers), fontsize=9)
    ax.set_yticks(range(len(HEATMAP_SCORES)))
    ax.set_yticklabels([name for _, name in HEATMAP_SCORES])
    ax.set_xlabel(XLABEL)
    fig.colorbar(image, ax=ax, label="AUC", pad=0.01)
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def group_mean(items: list[dict], score: str, label: int) -> np.ndarray:
    """Per layer, the mean over the texts with this label, ignoring nulls."""
    x = np.asarray([[np.nan if v is None else v for v in item[score]] for item in items if item["label"] == label],
                   dtype=float)
    return np.nanmean(x, axis=0)


def plot_energy(items: list[dict], title: str, path: Path) -> None:
    gap = {score: group_mean(items, score, 1) - group_mean(items, score, 0)
           for score in ("norm2", *(score for score, _, _ in PARTS))}
    layers = np.asarray([layer for layer in range(len(gap["norm2"])) if layer not in EXCLUDED_LAYERS])
    total = gap["norm2"][layers]
    dropped = layers[total < MIN_REL_GAP * group_mean(items, "norm2", 0)[layers]]
    total = np.where(np.isin(layers, dropped), np.nan, total)
    if len(dropped):
        print(f"Dropped layers with a tiny or reversed norm2 gap: {dropped.tolist()}")
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for score, name, color in PARTS:
        ax.plot(layers, gap[score][layers] / total, color=color, linewidth=1.6, marker="o", markersize=3, label=name)
    for y in (0, 1):
        ax.axhline(y, color="#888888", linewidth=0.6, linestyle=":")
    ax.set_xlabel(f"Layer (excl. {', '.join(map(str, EXCLUDED_LAYERS))})")
    ax.set_ylabel("Share of machine − human\ngap in total energy")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main(model: str, dataset: str, seed: int) -> None:
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aniso = json.loads((ANISO_DIR / f"aniso_{model}_{dataset}_s{seed}.json").read_text())
    title = f"{model}, {dataset}"
    plot_heatmap(aniso["metrics"], title, OUTPUT_DIR / f"f_aniso_auc_heatmap_{model}_{dataset}_s{seed}.pdf")
    plot_energy(aniso["items"], title, OUTPUT_DIR / f"f_aniso_energy_{model}_{dataset}_s{seed}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.model, args.dataset, args.seed)
