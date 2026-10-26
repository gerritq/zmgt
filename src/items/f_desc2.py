"""
One PDF per desc2 output (output/desc2/*.npz, from src.geometric_metrics.desc), i.e. per model and dataset
(f_desc2_{model}_{dataset}_s{seed}.pdf), with ten pages:
1. Layer profiles:   per score, human and machine mean over texts at each layer, with a 95% confidence band
                     (1.96 · standard error). Norm and magnitudes grow by orders of magnitude with depth, so their y axis
                     is logarithmic; angles are in degrees.
1b. Norm growth:     (a) mean_norm and (b) mean_norm divided by each text's own embedding-layer mean_norm (so the
                     embedding-level gap cancels), mean and 95% band per label on log y. The gap is a few % on a scale
                     spanning two orders of magnitude, so the lines overlap; the bottom row shows it directly:
                     100 · (machine − human) / human per layer, with a 95% bootstrap band.
The norms are worked out here from the stored token norms: mean_norm is the mean over all tokens (page 1); the norm
pages (1b, 1c) leave out, per text and layer, tokens with ‖h_t‖ > 2 · median_t ‖h_t‖ (mostly the first token, whose
norm is far larger); the share left out is printed on the plot.
1c. Norm per text:   the ratio of (b) for every text as a thin line, human and machine in one panel (log y), each
                     group mean on top: shows the spread across texts that the means hide.
2. Separation:       per score and layer, AUROC with machine as the positive class (0.5 = no separation; below 0.5 =
                     lower for machine text).
3. Perplexity:       per-text ppl histograms on log-spaced bins (ppl is right-skewed), medians as vertical lines, AUROC
                     in the title.
4. Dispersion vs ppl: per layer, Spearman ρ over texts between representation_dispersion and ppl, within each label
                     (pooling both labels would mix in the human/machine difference in both). Spearman because ppl is
                     skewed and the relation need not be linear.
5. PCA per layer:    for each of PCA_LAYERS, the mean trajectory over the sequence per label in that layer's own PCA
                     plane (fitted on all tokens of all texts): at each position t the mean over that label's texts,
                     smoothed over TRAJ_SMOOTH positions, from position 1 (circle) to the shortest text's last token
                     (triangle, labelled with its position). The first token (position 0) is not plotted; outputs
                     whose meta has pca_first_token = "excluded" also leave it out of the PCA fit, which the plot says.
6. Change decomposition: the change Δ of each token split along the previous state h (signed length Δ · h / ‖h‖) and
                     across it (‖Δ_⊥‖), across layers (Δ = h_t^ℓ − h_t^{ℓ−1}) and within a layer (Δ = h_t^ℓ − h_{t−1}^ℓ);
                     per text the mean over tokens, then mean and 95% band per label. Tokens flagged by the norm filter
                     (at either end of Δ) are left out, as one huge-norm token would dominate a text's mean.
6b. Change gap:      machine − human per layer for the four page-6 scores, with a 95% bootstrap band: the parallel
                     parts as a plain difference (they change sign, so % of human is not defined), the perpendicular
                     parts in % of the human mean.
7. Change separation: AUROC per layer of the four page-6 scores.
The last entry (hidden_states[L], the final-norm output) is left out everywhere, as in f_aniso: its scale differs from
the residual stream and would dominate every axis. Layer 0 (the embedding output) is kept.
Human is blue and solid, machine orange and dashed (Okabe-Ito pair; the line style keeps them apart in grey print).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.items.plot_style import configure_plot_style

cfg = Config()

DESC2_DIR = Path(cfg.base_dir) / "output" / "desc2"
OUTPUT_DIR = Path(cfg.item_output_dir) / "desc2"

MODEL_NAMES = {"l8b": "Llama-3.1-8B", "m8b": "Ministral-8B", "q8b": "Qwen3-8B"}
LABELS = ((0, "Human", "#0072B2", "-"), (1, "Machine", "#D55E00", "--"))
SCORES = (
    ("representation_dispersion", "Representation dispersion\n(mean pairwise cosine distance)", False, 1.0),
    ("mean_norm", r"Mean norm $\Vert h_t \Vert$", True, 1.0),
    ("angle_within", "Angle to previous token (°)", False, 180 / np.pi),
    ("magnitude_within", "Distance to previous token", True, 1.0),
    ("angle_across", "Angle to previous layer (°)", False, 180 / np.pi),
    ("magnitude_across", "Distance to previous layer", True, 1.0),
)  # (key, label, log y, scale)
GRID = "#E2E6EA"
TRAJ_SMOOTH = 10  # positions in the moving average of the mean PCA trajectory
PCA_LAYERS = tuple(range(3, 13))  # layers shown in the PCA page (ones beyond the model's depth are skipped)
BOOTSTRAPS = 2000  # resamples for the gap bands
NORM_OUTLIER_FACTOR = 2.0  # tokens with a norm above this times the text's median norm (per layer) are left out
CHANGES = (
    ("across_parallel", "Across layers: parallel\n" r"$\Delta \cdot h \,/\, \Vert h \Vert$, $\Delta = h_t^{\ell} - h_t^{\ell-1}$",
     False),
    ("across_perpendicular", "Across layers: perpendicular\n" r"$\Vert \Delta_\perp \Vert$, $\Delta = h_t^{\ell} - h_t^{\ell-1}$",
     True),
    ("within_parallel", "Within a layer: parallel\n" r"$\Delta \cdot h \,/\, \Vert h \Vert$, $\Delta = h_t^{\ell} - h_{t-1}^{\ell}$",
     False),
    ("within_perpendicular", "Within a layer: perpendicular\n" r"$\Vert \Delta_\perp \Vert$, $\Delta = h_t^{\ell} - h_{t-1}^{\ell}$",
     True),
)  # (key, label, log y); h = the previous state


def layer_values(data: dict[str, np.ndarray], key: str, scale: float) -> np.ndarray:
    """(texts, L) values without the final-norm entry."""
    return scale * data[key][:, :-1].astype(float)


def add_norm_scores(data: dict[str, np.ndarray]) -> None:
    """
    Per text and layer (texts, L + 1): mean_norm over all tokens, mean_norm_med2 and excluded_med2 for the norm filter,
    and mean_<change> for each CHANGES key over the tokens kept at both ends of the change.
    """
    norms = data["norms"]  # (texts, L + 1, max T), NaN padded
    keep = norms <= NORM_OUTLIER_FACTOR * np.nanmedian(norms, axis=2)[..., None]  # padding (NaN) is never kept
    data["mean_norm"] = np.nanmean(norms, axis=2)
    data["mean_norm_med2"] = np.where(keep, norms, 0).sum(axis=2) / keep.sum(axis=2)  # the median token is kept
    data["excluded_med2"] = 1 - keep.sum(axis=2) / data["tokens"][:, None]
    ends = {"across": keep[:, 1:] & keep[:, :-1], "within": keep[:, :, 1:] & keep[:, :, :-1]}
    ends = {"across": np.pad(ends["across"], ((0, 0), (1, 0), (0, 0))),  # layer 0 / position 0 have no change
            "within": np.pad(ends["within"], ((0, 0), (0, 0), (1, 0)))}
    for key, _, _ in CHANGES:
        values = data[key]
        mask = ends[key.split("_")[0]] & np.isfinite(values)
        with np.errstate(invalid="ignore"):  # layer 0 of the across scores has no tokens: NaN on purpose
            data[f"mean_{key}"] = np.where(mask, values, 0).sum(axis=2) / mask.sum(axis=2)


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over texts and the half-width of its 95% confidence interval, per layer."""
    n = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():  # all-null layers (layer 0 of the across-layer scores) give NaN on purpose
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, 1.96 * se


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    if len(set(labels[mask])) < 2:
        return np.nan
    return float(roc_auc_score(labels[mask], scores[mask]))


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def finish(pdf: PdfPages, fig: plt.Figure, title: str, caption: str) -> None:
    fig.suptitle(title)
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=11, color="#455A64", wrap=True)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def label_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False)


def page_profiles(pdf: PdfPages, data: dict[str, np.ndarray], title: str) -> None:
    labels = data["label"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (key, name, log, scale) in zip(axes.ravel(), SCORES):
        values = layer_values(data, key, scale)
        layers = np.arange(values.shape[1])
        for label, label_name, color, line in LABELS:
            mean, half = mean_ci(values[labels == label])
            ax.plot(layers, mean, color=color, linestyle=line, linewidth=2, label=label_name)
            ax.fill_between(layers, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
        if log:
            ax.set_yscale("log")
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Layer (0 = embedding)")
        style(ax)
    label_legend(fig, axes[0, 0])
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, hspace=0.45, wspace=0.25)
    finish(pdf, fig, f"{title}: layer profiles",
           "Mean over texts per layer, band = 95% CI of the mean. Log y for norm and distances. "
           "Final-norm layer left out.")


def gap_percent(values: np.ndarray, labels: np.ndarray, seed: int,
                relative: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per layer, 100 · (machine mean − human mean) / human mean (or machine mean − human mean if not relative) and its
    95% bootstrap band (texts resampled within each label): (texts, L) -> gap, low, high, each (L,).
    """
    human, machine = values[labels == 0], values[labels == 1]
    if relative:
        gap = lambda h, m: 100 * (m.mean(axis=0) - h.mean(axis=0)) / h.mean(axis=0)
    else:
        gap = lambda h, m: m.mean(axis=0) - h.mean(axis=0)
    rng = np.random.default_rng(seed)
    boots = np.stack([gap(human[rng.integers(0, len(human), len(human))],
                          machine[rng.integers(0, len(machine), len(machine))]) for _ in range(BOOTSTRAPS)])
    low, high = np.percentile(boots, [2.5, 97.5], axis=0)
    return gap(human, machine), low, high


def filtered_norms(fig: plt.Figure, data: dict[str, np.ndarray], y: float) -> np.ndarray:
    """(texts, L) mean norms without the > 2 x median tokens, with a note on the figure saying what was left out."""
    share = layer_values(data, "excluded_med2", 1.0)[:, 1:]  # layer 0 (embedding) has no outliers
    tokens = data["tokens"].astype(float)[:, None]
    fig.text(0.5, y, f"Norm without outlier tokens: per text and layer, tokens with norm > 2 × the text's median "
                     f"token norm are excluded (on average {np.nanmean(share * tokens):.1f} tokens per text, "
                     f"{100 * np.nanmean(share):.2f}%, layers 1+)", ha="center", fontsize=12, color="#455A64")
    return layer_values(data, "mean_norm_med2", 1.0)


def page_norm_growth(pdf: PdfPages, data: dict[str, np.ndarray], title: str, seed: int) -> None:
    labels = data["label"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": (3, 2)})
    norms = filtered_norms(fig, data, 0.925)
    ratio = norms / norms[:, :1]  # each text relative to its own embedding layer
    layers = np.arange(ratio.shape[1])
    panels = ((norms, "Mean norm", "(a) Mean norm", None),
              (ratio, "Mean norm / embedding mean norm", "(b) Relative to the embedding", 1))
    for col, (values, ylabel, name, reference) in enumerate(panels):
        ax = axes[0, col]
        if reference is not None:
            ax.axhline(reference, color="#7F8C8D", linewidth=0.8)
        for label, label_name, color, line in LABELS:
            mean, half = mean_ci(values[labels == label])
            ax.plot(layers, mean, color=color, linestyle=line, linewidth=2, label=label_name)
            ax.fill_between(layers, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(name, fontsize=14)
        ax.legend(frameon=False, loc="upper left")
        style(ax)

        ax = axes[1, col]
        gap, low, high = gap_percent(values, labels, seed)
        ax.axhline(0, color="#7F8C8D", linewidth=0.8)
        ax.fill_between(layers, low, high, color="#263238", alpha=0.15, linewidth=0)
        ax.plot(layers, gap, color="#263238", linewidth=2)
        ax.set_xlabel("Layer (0 = embedding)")
        ax.set_ylabel("Machine − human (% of human)")
        style(ax)
    lo = min(ax.get_ylim()[0] for ax in axes[1])
    hi = max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(lo, hi)  # same gap scale in both columns
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.12, hspace=0.12, wspace=0.22)
    finish(pdf, fig, f"{title}: norm growth and the machine − human gap",
           "Top: mean over texts, band = 95% CI, log y; (b) divides each text by its own embedding-layer norm. "
           "Bottom: gap in % of the human mean, band = 95% bootstrap CI. Final-norm layer left out.")


def page_norm_per_text(pdf: PdfPages, data: dict[str, np.ndarray], title: str) -> None:
    labels = data["label"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    norms = filtered_norms(fig, data, 0.885)
    ratio = norms / norms[:, :1]
    layers = np.arange(ratio.shape[1])
    ax.axhline(1, color="#7F8C8D", linewidth=0.8)
    for label, name, color, line in LABELS:  # thin lines first, so both means sit on top
        ax.plot(layers, ratio[labels == label].T, color=color, linewidth=0.6, alpha=0.25)
    for label, name, color, line in LABELS:
        group = ratio[labels == label]
        ax.plot(layers, group.mean(axis=0), color=color, linestyle=line, linewidth=2.6,
                label=f"{name} mean ({len(group)} texts)",
                path_effects=[patheffects.withStroke(linewidth=4.5, foreground="white")])
    ax.set_yscale("log")
    ax.set_xlabel("Layer (0 = embedding)")
    ax.set_ylabel("Mean norm / embedding mean norm")
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.17)
    finish(pdf, fig, f"{title}: norm relative to the embedding, per text",
           "Thin lines = single texts, thick line = group mean. Log y. Final-norm layer left out.")


def page_separation(pdf: PdfPages, data: dict[str, np.ndarray], title: str) -> None:
    labels = data["label"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharey=True)
    for ax, (key, name, _, scale) in zip(axes.ravel(), SCORES):
        values = layer_values(data, key, scale)
        aurocs = np.asarray([auroc(labels, values[:, layer]) for layer in range(values.shape[1])])
        ax.axhline(0.5, color="#7F8C8D", linewidth=0.8)
        ax.plot(np.arange(len(aurocs)), aurocs, color="#263238", linewidth=2)
        best = int(np.nanargmax(np.abs(aurocs - 0.5)))
        ax.plot(best, aurocs[best], "o", color="#263238", markersize=8)
        right = best > len(aurocs) / 2  # label to the left of the dot near the right edge
        ax.annotate(f"layer {best}: {aurocs[best]:.2f}", (best, aurocs[best]), textcoords="offset points",
                    xytext=(-6 if right else 6, 6), ha="right" if right else "left", fontsize=11, color="#263238")
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Layer (0 = embedding)")
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC (machine positive)")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12, hspace=0.45, wspace=0.12)
    finish(pdf, fig, f"{title}: human vs machine separation per layer",
           "AUROC of each score at each layer; 0.5 = no separation, below 0.5 = lower for machine text. "
           "Dot = layer furthest from 0.5.")


def page_ppl(pdf: PdfPages, data: dict[str, np.ndarray], title: str) -> None:
    labels = data["label"]
    ppl = data["ppl"].astype(float)
    bins = np.geomspace(np.nanmin(ppl), np.nanmax(ppl), 40)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, label_name, color, line in LABELS:
        values = ppl[labels == label]
        ax.hist(values, bins=bins, histtype="step", color=color, linestyle=line, linewidth=2, label=label_name)
        median = np.nanmedian(values)
        ax.axvline(median, color=color, linestyle=line, linewidth=1)
        ax.annotate(f"median {median:.1f}", (median, 1), xycoords=("data", "axes fraction"), xytext=(4, -16),
                    textcoords="offset points", fontsize=11, color="#263238")
    ax.set_xscale("log")
    ax.set_xlabel("Perplexity (log scale)")
    ax.set_ylabel("Texts")
    ax.set_title(f"AUROC (machine positive) = {auroc(labels, ppl):.2f}", fontsize=14)
    style(ax)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.18)
    finish(pdf, fig, f"{title}: perplexity per text", "Histogram on log-spaced bins; vertical lines = medians.")


def page_dispersion_ppl(pdf: PdfPages, data: dict[str, np.ndarray], title: str) -> None:
    labels = data["label"]
    ppl = data["ppl"].astype(float)
    dispersion = layer_values(data, "representation_dispersion", 1.0)
    layers = np.arange(dispersion.shape[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="#7F8C8D", linewidth=0.8)
    for label, name, color, line in LABELS:
        mask = labels == label
        rho = [spearmanr(dispersion[mask, layer], ppl[mask], nan_policy="omit").statistic for layer in layers]
        ax.plot(layers, rho, color=color, linestyle=line, linewidth=2, label=name)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Layer (0 = embedding)")
    ax.set_ylabel("Spearman ρ (dispersion, ppl)")
    style(ax)
    ax.legend(frameon=False, ncol=2, loc="lower center")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.18)
    finish(pdf, fig, f"{title}: representation dispersion vs perplexity",
           "Per layer and label, rank correlation over texts between each text's dispersion and its ppl. "
           "Final-norm layer left out.")


def page_pca(pdf: PdfPages, data: dict[str, np.ndarray], title: str, fit_without_first: bool) -> None:
    n_layers = data["pca"].shape[1]
    layers = [layer for layer in PCA_LAYERS if layer < n_layers - 1]  # final-norm layer left out
    kernel = np.ones(TRAJ_SMOOTH) / TRAJ_SMOOTH
    cols = -(-len(layers) // 2)  # two rows
    fig, axes = plt.subplots(2, cols, figsize=(3.6 * cols + 2, 10), squeeze=False)
    for ax in axes.ravel()[len(layers):]:
        ax.set_visible(False)
    for ax, layer in zip(axes.ravel(), layers):
        for label, label_name, color, line in LABELS:
            texts = data["label"] == label
            shortest = data["tokens"][texts].min()  # every text of the label is present up to here
            path = data["pca"][texts, layer, 1:shortest].mean(axis=0)  # (positions, 2), position 0 left out
            path = np.stack([np.convolve(path[:, c], kernel, mode="valid") for c in range(2)], axis=1)
            ax.plot(path[:, 0], path[:, 1], color=color, linestyle=line, linewidth=2.2, label=f"{label_name} mean")
            ax.plot(*path[0], "o", color=color, markersize=10, markeredgecolor="white")
            ax.plot(*path[-1], "^", color=color, markersize=11, markeredgecolor="white")
            ax.annotate(f"t = {shortest - 1}", path[-1], textcoords="offset points", xytext=(7, 4), fontsize=10,
                        color=color)
        ratio = data["pca_explained_variance_ratio"][layer]
        ax.set_title(f"Layer {layer}", fontsize=14)
        ax.set_xlabel(f"PC1 ({100 * ratio[0]:.1f}% var.)")
        ax.set_ylabel(f"PC2 ({100 * ratio[1]:.1f}% var.)")
        style(ax)
    handles, names = axes.ravel()[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], marker="o", color="#455A64", linestyle="", markersize=9),
                plt.Line2D([], [], marker="^", color="#455A64", linestyle="", markersize=10)]
    names += ["start (t = 1)", "end (shortest text's last token)"]
    fit = "left out of the PCA fit and the plot" if fit_without_first else \
        "left out of the plot, but included in the PCA fit (older output)"
    fig.text(0.5, 0.905, f"First token (position 0) {fit}", ha="center", fontsize=12, color="#455A64")
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.1, hspace=0.35, wspace=0.35)
    finish(pdf, fig, f"{title}: mean trajectory over the sequence, PCA per layer",
           f"One PCA per layer on all tokens of all texts; axes differ between layers. Line = mean over texts per "
           f"position, moving average over {TRAJ_SMOOTH} positions.")


def page_changes(pdf: PdfPages, data: dict[str, np.ndarray], title: str) -> None:
    labels = data["label"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), sharex=True)
    for ax, (key, name, log) in zip(axes.ravel(), CHANGES):
        values = layer_values(data, f"mean_{key}", 1.0)
        layers = np.arange(values.shape[1])
        if not log:
            ax.axhline(0, color="#7F8C8D", linewidth=0.8)
        for label, label_name, color, line in LABELS:
            mean, half = mean_ci(values[labels == label])
            ax.plot(layers, mean, color=color, linestyle=line, linewidth=2, label=label_name)
            ax.fill_between(layers, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
        if log:
            ax.set_yscale("log")
        ax.set_title(name, fontsize=14)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Layer (0 = embedding)")
    label_legend(fig, axes[0, 0])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.11, hspace=0.3, wspace=0.2)
    finish(pdf, fig, f"{title}: change split along and across the previous state",
           "Per text the mean over tokens, then mean over texts, band = 95% CI. h = the previous state; parallel > 0 "
           "= grows along h. Tokens with norm > 2 × the text's median norm (at either end of Δ) left out. Log y for "
           "perpendicular. Final-norm layer left out.")


def page_change_gap(pdf: PdfPages, data: dict[str, np.ndarray], title: str, seed: int) -> None:
    labels = data["label"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), sharex=True)
    for ax, (key, name, log) in zip(axes.ravel(), CHANGES):
        values = layer_values(data, f"mean_{key}", 1.0)[:, 1:] if key.startswith("across") else \
            layer_values(data, f"mean_{key}", 1.0)
        first = 1 if key.startswith("across") else 0  # the across scores have no layer 0
        layers = np.arange(first, first + values.shape[1])
        gap, low, high = gap_percent(values, labels, seed, relative=log)
        ax.axhline(0, color="#7F8C8D", linewidth=0.8)
        ax.fill_between(layers, low, high, color="#263238", alpha=0.15, linewidth=0)
        ax.plot(layers, gap, color="#263238", linewidth=2)
        ax.set_ylabel("Machine − human (% of human)" if log else "Machine − human")
        ax.set_title(name, fontsize=14)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Layer (0 = embedding)")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.11, hspace=0.3, wspace=0.22)
    finish(pdf, fig, f"{title}: machine − human gap in the change parts",
           "Gap of the per-text means of page 6, band = 95% bootstrap CI. Parallel: plain difference (it changes "
           "sign); perpendicular: % of the human mean. Final-norm layer left out.")


def page_change_separation(pdf: PdfPages, data: dict[str, np.ndarray], title: str) -> None:
    labels = data["label"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for ax, (key, name, _) in zip(axes.ravel(), CHANGES):
        values = layer_values(data, f"mean_{key}", 1.0)
        aurocs = np.asarray([auroc(labels, values[:, layer]) for layer in range(values.shape[1])])
        ax.axhline(0.5, color="#7F8C8D", linewidth=0.8)
        ax.plot(np.arange(len(aurocs)), aurocs, color="#263238", linewidth=2)
        best = int(np.nanargmax(np.abs(aurocs - 0.5)))
        ax.plot(best, aurocs[best], "o", color="#263238", markersize=8)
        right = best > len(aurocs) / 2  # label to the left of the dot near the right edge
        ax.annotate(f"layer {best}: {aurocs[best]:.2f}", (best, aurocs[best]), textcoords="offset points",
                    xytext=(-6 if right else 6, 6), ha="right" if right else "left", fontsize=11, color="#263238")
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=14)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Layer (0 = embedding)")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC (machine positive)")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.11, hspace=0.3, wspace=0.1)
    finish(pdf, fig, f"{title}: human vs machine separation of the change parts",
           "AUROC of each per-text mean at each layer; 0.5 = no separation, below 0.5 = lower for machine text. "
           "Dot = layer furthest from 0.5.")


def main(seed: int) -> None:
    configure_plot_style()
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#263238", "text.color": "#263238",
        "xtick.color": "#455A64", "ytick.color": "#455A64", "axes.edgecolor": "#B0BEC5", "pdf.fonttype": 42,
    })
    paths = sorted(DESC2_DIR.glob(f"desc2_*_s{seed}.npz"))
    if not paths:
        raise FileNotFoundError(f"no desc2 outputs for seed {seed} in {DESC2_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with np.load(path) as stored:
            data = dict(stored)
        meta = json.loads(str(data.pop("meta")))
        add_norm_scores(data)
        out = OUTPUT_DIR / f"f_desc2_{meta['model']}_{meta['dataset']}_s{seed}.pdf"
        with PdfPages(out) as pdf:
            title = f"{MODEL_NAMES.get(meta['model'], meta['model'])}, {meta['dataset']}"
            page_profiles(pdf, data, title)
            page_norm_growth(pdf, data, title, seed)
            page_norm_per_text(pdf, data, title)
            page_separation(pdf, data, title)
            page_ppl(pdf, data, title)
            page_dispersion_ppl(pdf, data, title)
            page_pca(pdf, data, title, meta.get("pca_first_token") == "excluded")
            page_changes(pdf, data, title)
            page_change_gap(pdf, data, title, seed)
            page_change_separation(pdf, data, title)
        print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
