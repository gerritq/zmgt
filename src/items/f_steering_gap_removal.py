"""
How much of the unsteered machine-human gap in geometry lies along the machine direction (static, no model
processing). Per layer l: the paired gap (within each human/machine pair) in adjacent-token angle, cross-layer angle
and mean pairwise cosine at layer l, on
  full states                     baseline
  with r̂_l removed               baseline_proj, real direction       (x <- x - (r̂_l . x) r̂_l at layers l-1 and l)
  with a shuffled direction removed baseline_shuf_proj (20 label-shuffled mean differences)
Gap signed so that > 0 = expected (human angles larger, machine pairwise cosine larger).
Share of the gap carried by the direction = 1 - gap(r̂ removed) / gap(full).
Averaged over domain-generator pairs (default: every pair with an all-token random run; text pairs pooled). The
shuffled control is drawn only if every pair has a data_shuffle run; the random control is not shown.
Output: output/items/steering/f_steering_gap_removal.pdf (+ .png) and f_steering_gap_removal.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.items.f_steering import latest_runs
from src.items.f_steering_summary import pair_label
from src.items.plot_style import configure_plot_style

cfg = Config()
OUTPUT = Path(cfg.item_output_dir) / "steering" / "f_steering_gap_removal.pdf"
METRICS = (("local_angle", "Adjacent-token angle", "gap (°), human − machine", -1),
           ("cross_angle", "Cross-layer angle", "gap (°), human − machine", -1),
           ("pairwise_cos", "Mean pairwise cosine", "gap, machine − human", 1))
TABLE_LAYERS = (2, 4, 8, 12, 16, 20)


def paired_gap(x: np.ndarray, y: np.ndarray, pid: np.ndarray, sign: int) -> np.ndarray:
    """x (N, ...) per text -> (pairs, ...) machine − human per pair, times sign (> 0 = expected)."""
    pairs = np.unique(pid)
    im = np.array([np.where((pid == q) & (y == 1))[0][0] for q in pairs])
    ih = np.array([np.where((pid == q) & (y == 0))[0][0] for q in pairs])
    return sign * (x[im] - x[ih])


def load(dataset: str) -> dict:
    """Paired gaps of one domain-generator pair: full, real direction removed, and (if a data_shuffle run exists)
    shuffled directions removed. The random control is not used."""
    runs = {p.name.split("_l8b_")[0]: p for p in latest_runs()
            if p.name.startswith("alltoken_") and f"_l8b_{dataset}_" in p.name}
    fr = np.load(runs["alltoken_random"])
    fs = np.load(runs["alltoken_data_shuffle"]) if "alltoken_data_shuffle" in runs else None
    y, pid, metrics = fr["labels"], fr["pair_id"], list(fr["metrics"])
    l = np.arange(len(fr["steer_layers"]))                                               # index of layer l = 1..P
    base, proj = fr["baseline"], fr["baseline_proj"]
    shuf = fs["baseline_shuf_proj"] if fs is not None else None
    out = {}
    for m, _, _, sign in METRICS:
        mi = metrics.index(m)
        g = {"full": paired_gap(base[:, mi, l], y, pid, sign),                           # (pairs, P) at m = l
             "real": paired_gap(proj[:, mi, 0, l, l], y, pid, sign)}
        if shuf is not None:
            g["shuf"] = paired_gap(shuf[:, mi][:, :, l, l], y, pid, sign)                # (pairs, K, P)
        out[m] = g
    return out


def ci(g: np.ndarray) -> np.ndarray:
    return 1.96 * g.std(axis=0, ddof=1) / np.sqrt(len(g))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="dataset names (default: every pair with an all-token random run)")
    args = parser.parse_args()
    configure_plot_style()
    datasets = args.datasets or sorted({p.name.split("_l8b_")[1].split("_s42_")[0] for p in latest_runs()
                                        if p.name.startswith("alltoken_random_")})
    data = [load(d) for d in datasets]
    with_shuf = all("shuf" in d[METRICS[0][0]] for d in data)
    if not with_shuf:
        print(f"Shuffled control left out: data_shuffle runs exist for only "
              f"{sum('shuf' in d[METRICS[0][0]] for d in data)} of {len(data)} pairs")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 4.3), squeeze=False)
    rows = []
    for k, (m, title, ylabel, _) in enumerate(METRICS):
        ax = axes[0, k]
        g = {key: np.concatenate([d[m][key] for d in data]) for key in data[0][m]}       # pairs of all datasets
        P = g["full"].shape[1]
        layers = np.arange(1, P + 1)
        if with_shuf:
            cm = g["shuf"].mean(axis=0)                                                  # (K, P)
            ax.fill_between(layers, cm.min(0), cm.max(0), color="#c9c8c3", lw=0, zorder=1,
                            label="shuffled direction removed (range of 20)")
        for key, label, color in (("full", "full states", "#52514e"),
                                  ("real", r"machine direction $\hat{r}_l$ removed", "#2a78d6")):
            v = g[key]
            ax.plot(layers, v.mean(0), color=color, lw=2, label=label, zorder=3)
            ax.fill_between(layers, v.mean(0) - ci(v), v.mean(0) + ci(v), color=color, alpha=0.2, lw=0, zorder=2)
        ax.axhline(0, color="#8a8985", lw=1, zorder=0)
        ax.set_xlim(1, P)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title)
        ax.set_xlabel("layer l")
        ax.set_ylabel(ylabel, fontsize=13)
        full, real = g["full"].mean(0), g["real"].mean(0)
        for l in TABLE_LAYERS:
            rows.append({"metric": m, "layer": l, "gap_full": full[l - 1], "gap_dir_removed": real[l - 1],
                         "share_explained": 1 - real[l - 1] / full[l - 1],
                         "pairs_gap_full_positive": float(np.mean(g["full"][:, l - 1] > 0)),
                         "pairs_gap_removed_positive": float(np.mean(g["real"][:, l - 1] > 0))})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=12)
    n_pairs = len(np.concatenate([d[METRICS[0][0]]["full"] for d in data]))
    fig.suptitle(f"Unsteered machine–human gap at layer l, with and without the machine direction, averaged over "
                 f"{len(data)} domain–generator pairs ({n_pairs} text pairs; band = 95% CI)\n"
                 "> 0 = expected: human angles larger, machine pairwise cosine larger", fontsize=14)
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.6 / h, 1, 1 - 0.2 / h))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    with open(OUTPUT.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {OUTPUT} ({len(data)} pairs: {', '.join(d[11:-11] for d in datasets)})")
    for r in rows:
        print(f"{r['metric']:12s} L{r['layer']:<2d} gap {r['gap_full']:+.4f} -> {r['gap_dir_removed']:+.4f} | "
              f"share {r['share_explained']:+.0%} | text pairs with expected gap: full {r['pairs_gap_full_positive']:.0%}, "
              f"removed {r['pairs_gap_removed_positive']:.0%}")


if __name__ == "__main__":
    main()
