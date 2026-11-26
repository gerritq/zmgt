"""
Shared helpers for src.ideas.test and src.ideas.test2, restored from the removed src.hs_anal.hs_anal:
compare (per-layer group stats and AUC), auc_table (AUC table figure) and figure (per-layer panels).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.items.plot_style import configure_plot_style

DIFF, AUC = "#263238", "#B0B0B0"  # dark ink / light grey
HUMAN, MACHINE = "#0072B2", "#D55E00"  # Okabe-Ito blue / vermillion, colorblind-safe
INK, MUTED, GRID = "#263238", "#7F8C8D", "#E2E6EA"


def compare(score: np.ndarray, labels: np.ndarray, reverse: bool = False) -> dict[str, np.ndarray]:
    """
    score: (N, n_layers) per-text values. Per group mean ±95% CI, unpaired machine − human difference ±95% CI, AUC and
    Cohen's d = (mean machine − mean human) / pooled SD, each (n_layers,).
    """
    h, m = score[labels == 0], score[labels == 1]
    pooled_sd = np.sqrt(((len(h) - 1) * h.var(0, ddof=1) + (len(m) - 1) * m.var(0, ddof=1)) / (len(h) + len(m) - 2))
    return {
        "cohen_d": (m.mean(0) - h.mean(0)) / np.where(pooled_sd > 0, pooled_sd, np.nan),
        "human_mean": h.mean(0), "human_ci": 1.96 * h.std(0, ddof=1) / np.sqrt(len(h)),
        "machine_mean": m.mean(0), "machine_ci": 1.96 * m.std(0, ddof=1) / np.sqrt(len(m)),
        "diff_mean": m.mean(0) - h.mean(0),
        "diff_ci": 1.96 * np.sqrt(m.var(0, ddof=1) / len(m) + h.var(0, ddof=1) / len(h)),
        "auc": np.array([roc_auc_score(labels, -s if reverse else s) for s in score.T]),
    }


def auc_table(aucs: dict[str, np.ndarray], title: str, first_layer: int) -> plt.Figure:
    """One row per score, one column per layer, AUC to 2 decimals; the best layer of each score in bold."""
    names = list(aucs)
    n_layers = len(next(iter(aucs.values())))
    cells = [[f"{v:.2f}" for v in aucs[name]] for name in names]
    fig, ax = plt.subplots(figsize=(0.62 * n_layers + 2.5, 0.45 * len(names) + 1.6))
    ax.axis("off")
    table = ax.table(cellText=cells, rowLabels=names,
                     colLabels=[str(first_layer + i) for i in range(n_layers)], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for r, name in enumerate(names, start=1):
        table[r, int(np.argmax(aucs[name]))].get_text().set_fontweight("bold")
    ax.set_title(f"{title}\nAUC per transformer layer (columns); angle scores: AUC of −score; bold = best layer",
                 fontsize=11)
    fig.tight_layout()
    return fig



def figure(panels: list[list[dict]], title: str, first_layer: int, xlabel: str, groups: bool = False) -> plt.Figure:
    """
    panels: grid (rows × columns) of dicts with "stats" (from compare), "title", "quantity" (y-axis name) and
    "auc_label" (optional "auc": False hides the AUC; then "cohen_d": True shows Cohen's d instead, on the panel's own
    scale). Per panel, per layer: on the left y-axis the machine − human difference, or with groups the mean of each
    group (±95% CI); on the right y-axis the AUC, on one scale for all panels.
    """
    configure_plot_style()
    plt.rcParams.update({"pdf.fonttype": 42, "axes.edgecolor": "#B0BEC5", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK})
    n_rows, n_cols = len(panels), max(len(row) for row in panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 4.6 * n_rows), squeeze=False)
    aucs = np.concatenate([[0.5]] + [p["stats"]["auc"] for row in panels for p in row if p.get("auc", True)])
    pad = 0.05
    auc_lim = (aucs.min() - pad, aucs.max() + pad)
    for axes_row, row in zip(axes, panels):
        for ax, p in zip(axes_row, row):
            s = p["stats"]
            layers = np.arange(first_layer, first_layer + len(s["auc"]))
            if groups:
                lines = []
                for key, label, color in (("human", "human (±95% CI)", HUMAN), ("machine", "machine (±95% CI)", MACHINE)):
                    mean, ci = s[f"{key}_mean"], s[f"{key}_ci"]
                    ax.fill_between(layers, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
                    lines += ax.plot(layers, mean, color=color, linewidth=2, label=label)
                ax.set_ylabel(p["quantity"])
            else:
                mean, ci = s["diff_mean"], s["diff_ci"]
                ax.axhline(0, color=MUTED, linewidth=0.8)
                ax.fill_between(layers, mean - ci, mean + ci, color=DIFF, alpha=0.15, linewidth=0)
                lines = ax.plot(layers, mean, color=DIFF, linewidth=2, label="machine − human (±95% CI)")
                ax.set_ylabel(f"{p['quantity']}\nmachine − human")
            ax.set_xlabel(xlabel)
            ax.set_xlim(layers[0], layers[-1])
            ax.set_title(p["title"])
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)

            if p.get("auc", True):
                twin = ax.twinx()
                twin.axhline(0.5, color=AUC, linewidth=0.8, linestyle=":")
                lines += twin.plot(layers, s["auc"], color=AUC, linewidth=2, label=p["auc_label"])
                twin.set_ylim(*auc_lim)
                twin.set_ylabel("AUC")
                twin.spines["top"].set_visible(False)
            elif p.get("cohen_d", False):  # own right axis per panel, symmetric around 0
                twin = ax.twinx()
                twin.axhline(0, color=AUC, linewidth=0.8, linestyle=":")
                lines += twin.plot(layers, s["cohen_d"], color=AUC, linewidth=2, label="Cohen's d (machine − human)")
                lim = np.nanmax(np.abs(s["cohen_d"])) * 1.1 if np.isfinite(s["cohen_d"]).any() else 1.0
                twin.set_ylim(-lim, lim)
                twin.set_ylabel("Cohen's d")
                twin.spines["top"].set_visible(False)
            ax.legend(handles=lines, frameon=False, loc="best", fontsize=11)
        for ax in axes_row[len(row):]:
            ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


