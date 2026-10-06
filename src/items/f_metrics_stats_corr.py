"""
Scatter plots of the per-text mean next-token entropy against the mean angle on the raw hidden states, with a least-
squares line over all texts and Pearson / Spearman correlations: (a) within_layer, the angle against the previous token
at layer LAYER, mean over tokens; (b) across_layer, the angle against the same token at the previous layer, mean over
layers per token and then over tokens (no layer-specific view).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from src.config import Config

cfg = Config()

BASE_DIR = Path(cfg.base_dir)
DESC_STATS_DIR = BASE_DIR / "output" / "metrics" / "desc_stats"
OUTPUT_DIR = Path(cfg.item_output_dir) / "metrics_stats"

# Transformer layer (1..L, embeddings excluded) of the within_layer angle; the within_layer lists start at layer 1.
LAYER = 9
STATES = "raw"
GROUPS = (("human", 0, "#1f77b4"), ("machine", 1, "#d62728"))


def mean_or_nan(values: list) -> float:
    """Mean of the finite values (null marks a token or text too short for the metric)."""
    x = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def load(axis: str, model: str, dataset: str, seed: int) -> list[dict]:
    path = DESC_STATS_DIR / f"desc_stats_{axis}_{STATES}_{model}_{dataset}_s{seed}.json"
    return json.loads(path.read_text())["items"]


def within_angle(items: list[dict]) -> np.ndarray:
    """Mean angle over tokens at LAYER per text."""
    value = lambda item: item["angle"][LAYER - 1]
    return np.asarray([np.nan if value(item) is None else value(item) for item in items], dtype=float)


def across_angle(items: list[dict]) -> np.ndarray:
    """Per text, the mean over tokens of each token's angle averaged over the layer steps."""
    return np.asarray([mean_or_nan(item["by_token"]["angle"]) for item in items], dtype=float)


def plot(entropy: np.ndarray, angle: np.ndarray, labels: np.ndarray, ylabel: str, title: str, path: Path) -> dict:
    """Entropy (x) vs. angle (y) per text, coloured by group, with the OLS line over all texts; returns the stats."""
    mask = np.isfinite(entropy) & np.isfinite(angle)
    x, y, labels = entropy[mask], angle[mask], labels[mask]
    fit = scipy_stats.linregress(x, y)
    pearson = scipy_stats.pearsonr(x, y)
    spearman = scipy_stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    for group, label, color in GROUPS:
        ax.scatter(x[labels == label], y[labels == label], s=12, alpha=0.6, color=color, edgecolors="none",
                   label=f"{group.capitalize()} (n={int((labels == label).sum())})")
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, fit.intercept + fit.slope * grid, color="black", linewidth=1.2,
            label=f"OLS: y = {fit.intercept:.3f} + {fit.slope:.3f}x")
    ax.set_xlabel("Mean next-token entropy (nats)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nPearson r={pearson.statistic:.3f} (p={pearson.pvalue:.1e}), "
                 f"Spearman ρ={spearman.statistic:.3f}", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return {"n": int(mask.sum()), "slope": fit.slope, "intercept": fit.intercept,
            "pearson_r": pearson.statistic, "pearson_p": pearson.pvalue,
            "spearman_rho": spearman.statistic, "spearman_p": spearman.pvalue}


def main(model: str, dataset: str, seed: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_{dataset}_s{seed}"

    within = load("within_layer", model, dataset, seed)
    across = load("across_layer", model, dataset, seed)
    # Both files hold the same texts in the same order; the per-text entropy is the same in each.
    labels = np.asarray([item["label"] for item in within])
    entropy = np.asarray([item["entropy"] for item in within], dtype=float)
    assert [item["label"] for item in across] == labels.tolist(), "within/across files hold different texts"

    results = {
        "within_layer": plot(
            entropy, within_angle(within), labels, f"Mean angle, layer {LAYER} (rad)",
            f"{model}, {dataset}: within layer {LAYER} ({STATES})",
            OUTPUT_DIR / f"f_metrics_stats_corr_within_layer{LAYER}_{STATES}_{stem}.pdf",
        ),
        "across_layer": plot(
            entropy, across_angle(across), labels, "Mean angle across layers (rad)",
            f"{model}, {dataset}: across layers ({STATES})",
            OUTPUT_DIR / f"f_metrics_stats_corr_across_layer_{STATES}_{stem}.pdf",
        ),
    }
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.model, args.dataset, args.seed)
