"""
Within-layer metrics per token position: for every other transformer layer 1..17, one panel with the token position
on the x-axis and the metric on the y-axis; one PDF per metric. Every text is a thin transparent line, smoothed by a
centred rolling mean over SMOOTH_WINDOW tokens; each group's mean of these smoothed lines at every position is a solid
line with its 95% CI (± 1.96 SE over texts) as a band, and the group's mean over texts and tokens a dashed horizontal
line:
    angle: ∠(h_{t−1}, h_t), the angle to the previous token's state at the same layer
    norm:  ‖h_t‖₂
on the raw hidden states. Both drop the first SKIP tokens (the attention sink, ‖h‖ ≈ 480 vs. ≈ 20, and the large,
position-driven norms after it): the norm starts at position SKIP + 1 and the angle at SKIP + 2, the first pair of kept
tokens (positions counted from 1).
Also, per layer, a contour plot of each text's mean norm (x) against its mean angle (y), with Gaussian KDE contours
for the human and the machine texts.
Every panel's title gives the AUC P(M > H) of the per-text mean (Mann-Whitney U / n_H n_M) at that layer; the
contour panels also give the Pearson r between the per-text mean norm and mean angle, over all texts (title) and per
group (legend).
"""

import os
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from scipy import stats as scipy_stats
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data
from src.geometric_metrics.metrics import GeometricMetrics

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "metrics", "desc")
# Transformer layers 1..L (embeddings excluded), every other one from 1 to 17.
LAYERS = tuple(range(1, 18, 2))
METRICS = ("angle", "norm")
YLABELS = {"angle": r"$\angle(h_{t-1}, h_t)$ (rad)", "norm": r"$\|h_t\|_2$"}
GROUPS = ((0, "human", "#1f77b4"), (1, "machine", "#d62728"))
# Leading tokens dropped from both metrics (the first few carry large, position-driven norms).
SKIP = 10
# Tokens in the centred rolling mean of the per-token lines.
SMOOTH_WINDOW = 10
# Density levels of the contours, as shares of each group's KDE maximum.
CONTOUR_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)


def token_metrics(hidden_states: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    """
    Per selected layer and token position (0-based): {metric: (len(LAYERS), T)}, NaN for the first SKIP tokens. The
    angle of position t is against position t − 1, so it starts one later, at the first pair of kept tokens.
    """
    n_tokens = hidden_states[0].shape[0]
    out = {metric: np.full((len(LAYERS), n_tokens), np.nan) for metric in METRICS}
    for i, layer in enumerate(LAYERS):
        h = hidden_states[layer].float()  # hidden_states[0] is the embedding output
        out["norm"][i, SKIP:] = h[SKIP:].norm(dim=-1).cpu().numpy()
        if n_tokens > SKIP + 1:
            current, previous = GeometricMetrics.reference(h[SKIP:], "previous")
            out["angle"][i, SKIP + 1:] = GeometricMetrics.angle(current, previous).cpu().numpy()
    return out


def collect(args: Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Labels (N,) and {metric: (N, len(LAYERS), T_max)}, NaN-padded to the longest text."""
    inference = Inference(model_name=args.model)
    test_data = load_data(args=args)["test"]
    labels, per_text = [], []
    for item in tqdm(test_data, desc="Collecting within-layer angle and norm per token"):
        per_text.append(token_metrics(inference.run(item, args)["hidden_states"]))
        labels.append(int(item["label"]))
    n_max = max(text["norm"].shape[1] for text in per_text)
    values = {}
    for metric in METRICS:
        stacked = np.full((len(per_text), len(LAYERS), n_max), np.nan)
        for j, text in enumerate(per_text):
            stacked[j, :, :text[metric].shape[1]] = text[metric]
        values[metric] = stacked
    return np.asarray(labels), values


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """P(machine > human) of the per-text scores (Mann-Whitney U / n_H n_M), over the finite ones."""
    finite = np.isfinite(scores)
    machine, human = scores[finite & (labels == 1)], scores[finite & (labels == 0)]
    if not len(machine) or not len(human):
        return float("nan")
    return float(scipy_stats.mannwhitneyu(machine, human).statistic / (len(machine) * len(human)))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r over the entries where both are finite; NaN with fewer than three."""
    finite = np.isfinite(x) & np.isfinite(y)
    return float(scipy_stats.pearsonr(x[finite], y[finite]).statistic) if finite.sum() > 2 else float("nan")


def smooth(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """
    Centred rolling mean along the last axis, NaN unless all `window` values are finite (so the skipped leading
    tokens and the text ends stay NaN): (..., T) -> (..., T).
    """
    finite = np.isfinite(values)
    kernel = np.ones(window)
    roll = lambda x: np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), -1, x)
    total, count = roll(np.where(finite, values, 0.0)), roll(finite.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count == window, total / window, np.nan)


def plot(labels: np.ndarray, values: np.ndarray, metric: str, title: str, path: str) -> None:
    """
    3 × 3 panels, one per layer: every text a thin smoothed line, each group's per-position mean of those lines with
    its 95% CI band, and the group's overall mean (raw values) as a dashed horizontal line.
    """
    smoothed = smooth(values)  # (N, len(LAYERS), T)
    positions = np.arange(1, values.shape[2] + 1)
    n_cols = 3
    n_rows = int(np.ceil(len(LAYERS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), squeeze=False)
    for i, layer in enumerate(LAYERS):
        ax = axes[i // n_cols][i % n_cols]
        for label, group, color in GROUPS:
            for row in smoothed[labels == label, i]:
                ax.plot(positions, row, color=color, alpha=0.06, linewidth=0.4)
        for label, group, color in GROUPS:
            lines = smoothed[labels == label, i]
            # Positions where no text has a smoothed value give NaN (and "mean of empty slice" warnings).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                n = np.isfinite(lines).sum(axis=0)
                curve = np.nanmean(lines, axis=0)
                half_width = 1.96 * np.nanstd(lines, axis=0, ddof=1) / np.sqrt(n)
            ax.fill_between(positions, curve - half_width, curve + half_width, color=color, alpha=0.3, linewidth=0)
            ax.plot(positions, curve, color=color, linewidth=1.2)
            mean = np.nanmean(values[labels == label, i])
            ax.axhline(mean, color=color, linestyle="--", linewidth=1.0,
                       label=f"{group.capitalize()} mean {mean:.3f} (n={len(lines)})")
        ax.set_title(f"Layer {layer}: AUC {auc(labels, np.nanmean(values[:, i], axis=1)):.3f}", fontsize=9)
        ax.set_xlabel("Token position")
        ax.set_ylabel(YLABELS[metric])
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    for j in range(len(LAYERS), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def draw_contour(ax: plt.Axes, labels: np.ndarray, x: np.ndarray, y: np.ndarray, prefix: str = "") -> None:
    """
    Per-text points (x, y), faint, with each group's Gaussian KDE contours at CONTOUR_LEVELS of its maximum density;
    the title gives both AUCs and the Pearson r over all texts, the legend r per group.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    pad_x, pad_y = 0.1 * np.ptp(x[finite]), 0.1 * np.ptp(y[finite])
    grid_x, grid_y = np.meshgrid(np.linspace(x[finite].min() - pad_x, x[finite].max() + pad_x, 150),
                                 np.linspace(y[finite].min() - pad_y, y[finite].max() + pad_y, 150))
    for label, group, color in GROUPS:
        rows = finite & (labels == label)
        ax.scatter(x[rows], y[rows], s=6, color=color, alpha=0.25, edgecolors="none")
        if rows.sum() > 2:
            density = scipy_stats.gaussian_kde(np.vstack([x[rows], y[rows]]))(
                np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
            ax.contour(grid_x, grid_y, density, levels=[level * density.max() for level in CONTOUR_LEVELS],
                       colors=color, linewidths=1.0)
        ax.plot([], [], color=color,
                label=f"{group.capitalize()} (n={int(rows.sum())}, r={pearson(x[rows], y[rows]):.3f})")
    ax.set_title(f"{prefix}AUC norm {auc(labels, x):.3f}, angle {auc(labels, y):.3f}; r={pearson(x, y):.3f}",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def plot_contours(labels: np.ndarray, values: dict[str, np.ndarray], title: str, path: str) -> None:
    """
    3 × 3 panels, one per layer: each text's mean norm (x) against its mean angle (y) as faint points, with the
    Gaussian KDE contours of each group at CONTOUR_LEVELS of its maximum density.
    """
    mean_angle = np.nanmean(values["angle"], axis=2)  # (N, len(LAYERS))
    mean_norm = np.nanmean(values["norm"], axis=2)
    n_cols = 3
    n_rows = int(np.ceil(len(LAYERS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows), squeeze=False)
    for i, layer in enumerate(LAYERS):
        ax = axes[i // n_cols][i % n_cols]
        draw_contour(ax, labels, mean_norm[:, i], mean_angle[:, i], f"Layer {layer}: ")
        ax.set_xlabel(rf"Mean $\|h_t\|_2$ (from token {SKIP + 1})")
        ax.set_ylabel(rf"Mean $\angle(h_{{t-1}}, h_t)$ (from token {SKIP + 2}, rad)")
    for j in range(len(LAYERS), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = f"{args.model}_{args.dataset}_s{args.seed}"
    labels, values = collect(args)
    skipped = f"first {SKIP} tokens left out"
    for metric in METRICS:
        plot(labels, values[metric], metric,
             f"{args.model}, {args.dataset}: within-layer {metric} per token (raw states, {skipped}); "
             f"lines: rolling mean over {SMOOTH_WINDOW} tokens, solid: group mean with 95% CI",
             os.path.join(OUTPUT_DIR, f"within_layer_{metric}_by_token_{stem}.pdf"))
    plot_contours(labels, values,
                  f"{args.model}, {args.dataset}: mean angle vs. mean norm per text (raw states, {skipped}); "
                  f"contours at "
                  f"{', '.join(f'{level:.0%}' for level in CONTOUR_LEVELS)} of each group's KDE maximum",
                  os.path.join(OUTPUT_DIR, f"within_layer_angle_vs_norm_contour_{stem}.pdf"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
