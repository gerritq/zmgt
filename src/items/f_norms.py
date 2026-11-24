"""
Charts of the norm results from src.hs_anal.norm_anal (output/hidden_states/norms/norms_{model}_{dataset}_s{seed}.npz).
Six scores, one panel each: norm before the layer norm, norm after it (post_norm), log growth per layer, total energy,
shared energy and token-specific energy. Three figures in output/hidden_states/norms/:
    f_norms_groups_{name}.pdf   per layer, mean of each group (human / machine) ±95% CI (normal approximation)
    f_norms_gap_{name}.pdf      per layer, the machine − human gap ±95% CI, relative to the human mean in % (for log
                                growth, which is already a log ratio: the plain difference)
    f_norms_auc_{name}.pdf      per layer, AUC of the score (machine = 1; > 0.5: machine has the larger value)
Norm and energy panels use a log y-axis (values grow by orders of magnitude across depth). --png also writes PNGs.

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.items.f_norms --model l8b \
        --dataset drlXDomainLen_wiki_300_tokens
"""

import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.items.plot_style import configure_plot_style

cfg = Config()
NORMS_DIR = os.path.join(cfg.base_dir, "output", "hidden_states", "norms")
HUMAN, MACHINE = "#0072B2", "#D55E00"  # Okabe-Ito blue / vermillion, as in src.hs_anal.hs_anal
INK, MUTED, GRID = "#263238", "#7F8C8D", "#E2E6EA"
# key, title, y label, log y-axis
PANELS = (
    ("norm", "Norm before layer norm", r"mean $\|h_t\|$", True),
    ("post_norm", "Norm after layer norm", r"mean $\|\mathrm{RMSNorm}(h_t)\|$", False),
    ("log_growth", "Norm growth per layer", r"mean $\log(\|h_t^\ell\| / \|h_t^{\ell-1}\|)$", False),
    ("energy", "Total energy", r"mean $\|h_t\|^2$", True),
    ("shared_energy", "Shared energy", r"$\|\mu\|^2$", True),
    ("token_energy", "Token-specific energy", r"mean $\|h_t - \mu\|^2$", True),
)


def layers_of(key: str, n: int) -> np.ndarray:
    """x positions: layer ℓ for per-layer scores (0 = embeddings), ℓ for the growth step ℓ−1 → ℓ."""
    return np.arange(1, n + 1) if key == "log_growth" else np.arange(n)


def style(ax: plt.Axes, key: str, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("layer ℓ (step ℓ−1 → ℓ)" if key == "log_growth" else "layer (0 = embeddings)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def setup() -> None:
    configure_plot_style()
    plt.rcParams.update({"pdf.fonttype": 42, "axes.edgecolor": "#B0BEC5", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK})


def save(fig: plt.Figure, path: str, png: bool) -> None:
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")
    if png:
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def groups_figure(d: dict[str, np.ndarray], labels: np.ndarray, title: str) -> plt.Figure:
    """2 × 3 panels, per layer the mean of each group ±95% CI; direct labels in the first panel."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for i, (ax, (key, name, ylabel, log)) in enumerate(zip(axes.flat, PANELS)):
        x = layers_of(key, d[key].shape[1])
        for group, label, color in ((0, "human", HUMAN), (1, "machine", MACHINE)):
            v = d[key][labels == group]
            mean, ci = v.mean(0), 1.96 * v.std(0, ddof=1) / np.sqrt(len(v))
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2, label=label)
            if i == 0:
                ax.annotate(label, (x[-1], mean[-1]), xytext=(6, 6 if group else -10), textcoords="offset points",
                            color=INK, fontsize=13, va="center")
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel + (" (log)" if log else ""))
        ax.set_xlim(x[0], x[-1])
        style(ax, key, name)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, [f"{n} (±95% CI)" for n in names], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def gap_figure(d: dict[str, np.ndarray], labels: np.ndarray, title: str) -> plt.Figure:
    """2 × 3 panels, per layer the machine − human gap ±95% CI (unpaired); in % of the human mean except log growth."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for ax, (key, name, _, _) in zip(axes.flat, PANELS):
        x = layers_of(key, d[key].shape[1])
        h, m = d[key][labels == 0], d[key][labels == 1]
        diff = m.mean(0) - h.mean(0)
        ci = 1.96 * np.sqrt(m.var(0, ddof=1) / len(m) + h.var(0, ddof=1) / len(h))
        relative = key != "log_growth"
        scale = 100 / h.mean(0) if relative else 1.0
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.fill_between(x, (diff - ci) * scale, (diff + ci) * scale, color=INK, alpha=0.15, linewidth=0)
        ax.plot(x, diff * scale, color=INK, linewidth=2)
        ax.set_ylabel("machine − human (% of human)" if relative else "machine − human")
        ax.set_xlim(x[0], x[-1])
        style(ax, key, name)
    fig.suptitle(f"{title}\nmachine − human gap ±95% CI; above 0: machine has the larger value", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def auc_figure(d: dict[str, np.ndarray], labels: np.ndarray, title: str) -> plt.Figure:
    """2 × 3 panels, per layer the AUC (machine = 1), one shared y-range; the layer furthest from 0.5 is labelled."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharey=True)
    for ax, (key, name, _, _) in zip(axes.flat, PANELS):
        x = layers_of(key, d[key].shape[1])
        auc = np.array([roc_auc_score(labels, column) for column in d[key].T])
        ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
        ax.plot(x, auc, color=INK, linewidth=2, marker="o", markersize=4)
        best = int(np.argmax(np.abs(auc - 0.5)))
        ha = "right" if best > 0.8 * len(x) else "left" if best < 0.2 * len(x) else "center"
        ax.annotate(f"{auc[best]:.2f} (layer {x[best]})", (x[best], auc[best]),
                    xytext=(0, 10 if auc[best] >= 0.5 else -16), textcoords="offset points", ha=ha,
                    fontsize=12, color=INK)
        ax.set_ylim(-0.15, 1.12)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylabel("AUC (machine = 1)")
        style(ax, key, name)
    fig.suptitle(f"{title}\nAUC > 0.5: machine has the larger value; dotted line = chance", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--png", action="store_true", help="also write PNGs next to the PDFs")
    args = parser.parse_args()

    name = f"{args.model}_{args.dataset}_s{args.seed}"
    data = np.load(os.path.join(NORMS_DIR, f"norms_{name}.npz"))
    d = {key: data[key] for key in data.files}
    labels = d.pop("labels")
    title = f"Hidden-state norms, {args.model}, {args.dataset} ({(labels == 0).sum()} human, {(labels == 1).sum()} machine)"

    setup()
    save(groups_figure(d, labels, title), os.path.join(NORMS_DIR, f"f_norms_groups_{name}.pdf"), args.png)
    save(gap_figure(d, labels, title), os.path.join(NORMS_DIR, f"f_norms_gap_{name}.pdf"), args.png)
    save(auc_figure(d, labels, title), os.path.join(NORMS_DIR, f"f_norms_auc_{name}.pdf"), args.png)


if __name__ == "__main__":
    main()
