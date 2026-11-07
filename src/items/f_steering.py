"""
Plots of the steering results (output/steering/prefix_*.npz and alltoken_*.npz, see src.steering.prefix and
src.steering.steer), one PDF per file in (by default only the latest run per kind, control, model,
dataset and seed, by meta["created"] or else the file-name time stamp)
output/items/steering/<file name>.pdf:
1. Baseline suffix geometry (unsteered) by measurement layer: cross-layer angle, within-layer (adjacent-token) angle
   and representation dispersion (mean pairwise cosine), one line per class, band = 95% CI over texts.
2. Heatmaps of the mean Δ = steered − baseline over the test texts (paired: each text against its own unsteered
   pass), real direction only, for c > 0 (towards the other class). y = steering layer l, x = measurement layer m;
   cells with m <= l are blank. Columns = c, rows = class × metric (adjacent-token angle, cross-layer angle,
   dispersion = mean pairwise cosine). One page with the full states, one with the steering direction projected out,
   same layout.
3. z-scores against the control in the file (random or label-shuffled directions): per class, cell and c,
   z = (mean Δ of the real direction − mean over control vectors of their mean Δ) / SD over control vectors.
   One page with the full states, one with the injected vector projected out (real and control alike).
Every panel has its own symmetric colour scale and colour bar; a black box marks measurement layers BAND (5-15).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from src.config import Config
from src.items.plot_style import configure_plot_style

cfg = Config()

STEERING_DIR = Path(cfg.base_dir) / "output" / "steering"
OUTPUT_DIR = Path(cfg.item_output_dir) / "steering"

# label 1 = machine, 0 = human (steering sign -1 for label 1 in src.steering.prefix)
CLASSES = ((1, "Machine", "#D55E00"), (0, "Human", "#0072B2"))
BASELINE_PANELS = (("cross_angle", "Cross-layer angle (°)"),
                   ("local_angle", "Within-layer angle (°)"),
                   ("pairwise_cos", "Dispersion (mean pairwise cos)"))
HEATMAP_METRICS = (("local_angle", "adjacent-token"), ("cross_angle", "cross-layer"),
                   ("pairwise_cos", "dispersion"))
UNITS = {"local_angle": "Δ angle (°)", "cross_angle": "Δ angle (°)", "pairwise_cos": "Δ mean pairwise cos"}
CONTROL_NAME = {"rand": "random directions", "shuf": "label-shuffled directions"}
DIRECTION = {1: "machine → human", 0: "human → machine"}
# measurement layers marked with a box on every heatmap
BAND = (5, 15)


def scope(title: str) -> str:
    """What the angles are measured on: the suffix (prefix steering) or all tokens (all-token steering)."""
    return "all-token" if title.startswith("all-token") else "suffix"


def plot_baseline(f, pdf: PdfPages, title: str) -> None:
    metrics, labels, layers = list(f["metrics"]), f["labels"], f["layers"]
    base = f["baseline"]                                                             # (N, M, L)
    fig, axes = plt.subplots(1, len(BASELINE_PANELS), figsize=(6 * len(BASELINE_PANELS), 4.5))
    for ax, (metric, ylabel) in zip(axes, BASELINE_PANELS):
        x = base[:, metrics.index(metric)]
        for lab, name, color in CLASSES:
            v = x[labels == lab]
            n = np.sum(~np.isnan(v), axis=0)
            mean = np.nanmean(v, axis=0)
            ci = 1.96 * np.nanstd(v, axis=0, ddof=1) / np.sqrt(n)
            ax.plot(layers, mean, color=color, label=f"{name} (n={len(v)})")
            ax.fill_between(layers, mean - ci, mean + ci, color=color, alpha=0.2, lw=0)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xlim(layers[0], layers[-1])
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{title}: unsteered {scope(title)} geometry (band = 95% CI)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def heatmap_means(f, key: str) -> dict:
    """{(class, metric): (S+, P, L)} mean Δ over the texts of a class, real direction, c > 0 only."""
    metrics, labels, keep = list(f["metrics"]), f["labels"], f["strengths"] > 0
    d = f[key][:, :, 0][:, :, keep]                                                 # (N, M, S+, P, L)
    return {(lab, m): np.nanmean(d[labels == lab, metrics.index(m)], axis=0)
            for lab, _, _ in CLASSES for m, _ in HEATMAP_METRICS}


def control_z(f, means: dict, projected: bool = False) -> tuple[dict, str]:
    """{(class, metric): (S+, P, L)} z of the real mean Δ against the control vectors, and the control tag. With
    projected, both sides use the projected-out Δ (real: r_l removed; each control: its own vector removed)."""
    tag = "shuf" if "delta_shuf" in f.files else "rand"
    metrics, labels, keep = list(f["metrics"]), f["labels"], f["strengths"] > 0
    d = f[f"delta_{tag}_proj" if projected else f"delta_{tag}"][:, :, :, keep]                                            # (N, M, K, S+, P, L)
    z = {}
    for lab, _, _ in CLASSES:
        for m, _ in HEATMAP_METRICS:
            ctrl = np.nanmean(d[labels == lab, metrics.index(m)], axis=0)            # (K, S+, P, L)
            z[(lab, m)] = (means[(lab, m)] - ctrl.mean(axis=0)) / ctrl.std(axis=0)
    return z, tag


def plot_heatmaps(f, pdf: PdfPages, title: str, values: dict, subtitle: str, zscore: bool = False) -> None:
    strengths = f["strengths"][f["strengths"] > 0]
    steer_layers, layers = f["steer_layers"], f["layers"]
    rows = [(lab, m, mname) for lab, _, _ in CLASSES for m, mname in HEATMAP_METRICS]
    fig, axes = plt.subplots(len(rows), len(strengths), figsize=(4.6 * len(strengths), 3.4 * len(rows)),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    extent = (layers[0] - 0.5, layers[-1] + 0.5, steer_layers[-1] + 0.5, steer_layers[0] - 0.5)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    for i, (lab, m, mname) in enumerate(rows):
        for j, c in enumerate(strengths):
            ax = axes[i, j]
            v = values[(lab, m)][j]
            vmax = np.nanmax(np.abs(v[np.isfinite(v)])) if np.isfinite(v).any() else 1.0
            im = ax.imshow(v, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", extent=extent,
                           interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.9, aspect=20, label="z vs control" if zscore else UNITS[m])
            ax.add_patch(Rectangle((BAND[0] - 0.5, steer_layers[0] - 0.5), BAND[1] - BAND[0] + 1,
                                   steer_layers[-1] - steer_layers[0] + 1, fill=False, ec="black", lw=1.4))
            if i == 0:
                ax.set_title(f"c = {c:g}")
            if j == 0:
                ax.set_ylabel(f"{DIRECTION[lab]}\n{mname}\nsteering layer l", fontsize=13)
            if i == len(rows) - 1:
                ax.set_xlabel("measurement layer m")
    fig.suptitle(f"{title}: {subtitle}  (red = increase, blue = decrease; own colour scale per panel)")
    pdf.savefig(fig)
    plt.close(fig)


def plot_file(path: Path) -> Path:
    f = np.load(path)
    meta = json.loads(str(f["meta"]))
    kind = "all-token" if path.stem.startswith("alltoken") else "prefix"
    title = f"{kind} · {meta['model']} · {meta['dataset']} (n={len(f['labels'])})"
    full, proj = heatmap_means(f, "delta"), heatmap_means(f, "delta_proj")
    z, tag = control_z(f, full)
    zp, _ = control_z(f, proj, projected=True)
    out = OUTPUT_DIR / f"{path.stem}.pdf"
    with PdfPages(out) as pdf:
        plot_baseline(f, pdf, title)
        plot_heatmaps(f, pdf, title, full, "mean Δ, steered − baseline, full states")
        plot_heatmaps(f, pdf, title, proj, "mean Δ, steered − baseline, steering direction projected out")
        plot_heatmaps(f, pdf, title, z, f"z of mean Δ against {CONTROL_NAME[tag]}, full states", zscore=True)
        plot_heatmaps(f, pdf, title, zp, f"z of mean Δ against {CONTROL_NAME[tag]}, steering direction projected out",
                      zscore=True)
    return out


def run_key(path: Path) -> tuple[tuple, str]:
    """(kind, control, model, dataset, seed) of a result file and its creation time: meta["created"] if present,
    otherwise the time stamp at the end of the file name (YYYYmmdd-HHMMSS, the time the run saved it)."""
    meta = json.loads(str(np.load(path)["meta"]))
    kind = "alltoken" if path.stem.startswith("alltoken") else "prefix"
    created = meta.get("created") or path.stem.rsplit("_", 1)[-1]
    return (kind, meta.get("control"), meta["model"], meta["dataset"], meta["seed"]), created


def latest_runs() -> list[Path]:
    """All prefix_*.npz and alltoken_*.npz files, keeping only the latest run per (kind, control, model, dataset,
    seed)."""
    latest: dict[tuple, tuple[str, Path]] = {}
    for path in sorted(STEERING_DIR.glob("prefix_*.npz")) + sorted(STEERING_DIR.glob("alltoken_*.npz")):
        try:
            key, created = run_key(path)
        except Exception as err:   # e.g. a file still being written by a running job
            print(f"Skipped {path.name}: {err}")
            continue
        if key not in latest or created > latest[key][0]:
            latest[key] = (created, path)
    return sorted(p for _, p in latest.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", help="npz files (default: the latest run of every prefix_*/alltoken_* configuration in output/steering)")
    args = parser.parse_args()
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [Path(p) for p in args.files] if args.files else latest_runs()
    for path in files:
        print(f"Saved {plot_file(path)}")


if __name__ == "__main__":
    main()
