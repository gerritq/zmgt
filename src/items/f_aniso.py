"""
Plots of the energy decomposition norm2 = shared_centroid + token_specific from the aniso metrics
(output/metrics/aniso), for one token variant (--variant: all, skip10pct, med5, med10; see
src.geometric_metrics.aniso). One PDF per model and figure, with one panel per domain (dataset):
- f_aniso_gap_heatmap: by layer, how much bigger the machine mean is than the human mean, in %,
  100 · (machine mean − human mean) / human mean, for total energy, shared, token specific and shared share.
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
EXCLUDED_LAYERS = (0, 1, -1)  # -1 = last layer
MIN_REL_GAP = 0.01
# Entry i is hidden_states[i]: 0 = embedding output, 1..L = transformer-layer outputs. The x axis is the relative
# depth i / L, so models with different numbers of layers line up.
XLABEL = "Relative depth (layer / L; 0 = embedding)"
DEPTH_TICKS = np.linspace(0, 1, 11)
# Token variants stored by src.geometric_metrics.aniso (kept in sync with its VARIANTS).
VARIANTS = ("all", "skip10pct", "med5", "med10")


def domain_name(dataset: str) -> str:
    """drlXDomainLen_news_500_tokens -> news."""
    parts = dataset.split("_")
    return parts[1] if len(parts) > 1 else dataset


def gap_matrix(metrics: dict) -> np.ndarray:
    """(scores, layers): 100 · (machine mean − human mean) / human mean."""
    return np.asarray([[100 * (entry["mean_machine"] - entry["mean_human"]) / entry["mean_human"]
                        for entry in metrics[score]] for score, _ in HEATMAP_SCORES], dtype=float)


def plot_heatmap(runs: dict[str, dict], title: str, path: Path) -> None:
    gaps = {dataset: gap_matrix(run["metrics"]) for dataset, run in runs.items()}
    n_layers = next(iter(gaps.values())).shape[1]
    limit = max(np.nanmax(np.abs(g)) for g in gaps.values())
    fig, axes = plt.subplots(len(gaps), 1, figsize=(0.5 * n_layers + 3, 2.6 * len(gaps) + 0.8),
                             layout="constrained", squeeze=False)
    for ax, (dataset, g) in zip(axes[:, 0], gaps.items()):
        # Column i is centred at depth i / L.
        step = 1 / (n_layers - 1)
        image = ax.imshow(g, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto",
                          extent=(-step / 2, 1 + step / 2, len(HEATMAP_SCORES) - 0.5, -0.5))
        for (row, col), value in np.ndenumerate(g):
            if np.isfinite(value):
                ax.text(col * step, row, f"{value:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.6 * limit else "black")
        ax.set_xticks(DEPTH_TICKS)
        ax.set_xticklabels([f"{tick:.1f}" for tick in DEPTH_TICKS], fontsize=9)
        ax.set_yticks(range(len(HEATMAP_SCORES)))
        ax.set_yticklabels([name for _, name in HEATMAP_SCORES])
        ax.set_title(domain_name(dataset))
    axes[-1, 0].set_xlabel(XLABEL)
    fig.colorbar(image, ax=axes[:, 0].tolist(), label="Machine vs. human (%)", pad=0.01, shrink=0.6)
    fig.suptitle(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def group_mean(items: list[dict], score: str, label: int) -> np.ndarray:
    """Per layer, the mean over the texts with this label, ignoring nulls."""
    x = np.asarray([[np.nan if v is None else v for v in item[score]] for item in items if item["label"] == label],
                   dtype=float)
    return np.nanmean(x, axis=0)


def plot_energy_panel(ax: plt.Axes, items: list[dict], dataset: str) -> None:
    """One domain's shares by relative depth."""
    gap = {score: group_mean(items, score, 1) - group_mean(items, score, 0)
           for score in ("norm2", *(score for score, _, _ in PARTS))}
    n_layers = len(gap["norm2"])
    excluded = sorted(layer % n_layers for layer in EXCLUDED_LAYERS)
    layers = np.asarray([layer for layer in range(n_layers) if layer not in excluded])
    total = gap["norm2"][layers]
    dropped = layers[total < MIN_REL_GAP * group_mean(items, "norm2", 0)[layers]]
    total = np.where(np.isin(layers, dropped), np.nan, total)
    if len(dropped):
        print(f"{dataset}: dropped layers with a tiny or reversed norm2 gap: {dropped.tolist()}")
    for score, name, color in PARTS:
        ax.plot(layers / (n_layers - 1), gap[score][layers] / total, color=color, linewidth=1.6, marker="o", markersize=3, label=name)
    for y in (0, 1):
        ax.axhline(y, color="#888888", linewidth=0.6, linestyle=":")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(domain_name(dataset))


def plot_energy(runs: dict[str, dict], title: str, path: Path) -> None:
    ncols = min(len(runs), 2)
    nrows = -(-len(runs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.0 * nrows), layout="constrained",
                             squeeze=False, sharex=True)
    for ax, (dataset, run) in zip(axes.flat, runs.items()):
        plot_energy_panel(ax, run["items"], dataset)
    for ax in axes.flat[len(runs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Relative depth (layer / L; excl. embedding,\nfirst and last layer)")
        ax.set_xticks(DEPTH_TICKS[::2])
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of machine − human\ngap in total energy")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels), frameon=False)
    fig.suptitle(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main(model: str, seed: int, variant: str) -> None:
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {}
    for file in sorted(ANISO_DIR.glob(f"aniso_{model}_*_s{seed}.json")):
        aniso = json.loads(file.read_text())
        if "variants" not in aniso:
            print(f"Skipping {file.name}: no token variants (computed before they were added); rerun aniso.py")
            continue
        runs[aniso["dataset"]] = aniso["variants"][variant]
    if not runs:
        raise SystemExit(f"No aniso files with token variants for {model}, seed {seed} in {ANISO_DIR}")
    definition = next(iter(runs.values()))["definition"]
    title = f"{model}: {definition}"
    plot_heatmap(runs, title, OUTPUT_DIR / f"f_aniso_gap_heatmap_{model}_{variant}_s{seed}.pdf")
    plot_energy(runs, title, OUTPUT_DIR / f"f_aniso_energy_{model}_{variant}_s{seed}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default="all", choices=VARIANTS)
    args = parser.parse_args()
    main(args.model, args.seed, args.variant)
