"""
Growth of the unsteered machine-human gap with depth, against the growth of the token norm (all-token steering file,
default: latest alltoken_random_* run of wiki gpt-4o). Unsteered test texts only, except panel D.

A  token norm: mean token norm u_l at layers 1..P (train tokens, both classes)
B  class position along the layer's own direction r̂_l (mean r̂_l . h over tokens, test texts), machine and human,
   with the gap between them
C  the same positions as a share of the token norm (r̂_l . h / u_l): does the gap grow faster than the norm?
D  position along the fixed direction r̂_8 at layers 8..32, unsteered (solid) and after the c = 0.05 push towards the
   other class at layer 8 (dashed): the push stays, while the unsteered classes keep drifting apart
Output: output/items/steering/f_steering_gap_<dataset>.pdf (+ .png)
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
from src.items.f_steering import latest_runs
from src.items.plot_style import configure_plot_style

cfg = Config()
OUTPUT_DIR = Path(cfg.item_output_dir) / "steering"
CLASSES = ((1, "Machine", "#D55E00"), (0, "Human", "#0072B2"))
FIXED_LAYER = 8
STRENGTH = 0.05


def ci(x: np.ndarray) -> np.ndarray:
    return 1.96 * x.std(axis=0, ddof=1) / np.sqrt(len(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="drlXpaired_wiki_gpt_4o_256_tokens")
    args = parser.parse_args()
    configure_plot_style()
    path = [p for p in latest_runs() if p.name.startswith(f"alltoken_random_l8b_{args.dataset}_")][0]
    f = np.load(path)
    meta = json.loads(str(f["meta"]))
    y, u, P = f["labels"], f["u"], len(f["steer_layers"])
    vb = f["vh_baseline"][:, 0]                                                       # (N, P, L)
    vs = f["vh_steered"][:, 0, list(f["strengths"]).index(STRENGTH)]                  # (N, P, L)
    layers = np.arange(1, P + 1)
    own = vb[:, layers - 1, layers - 1]                                               # (N, P): along r̂_l at layer l

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    # A: token norm
    ax = axes[0]
    ax.plot(layers, u, color="#52514e", lw=2, marker="o", ms=4)
    ax.set_title("A  Token norm grows")
    ax.set_ylabel(r"mean token norm $\bar{u}_l$")
    # B: position along own direction, absolute
    ax = axes[1]
    for lab, name, color in CLASSES:
        v = own[y == lab]
        ax.plot(layers, v.mean(0), color=color, lw=2, label=name)
        ax.fill_between(layers, v.mean(0) - ci(v), v.mean(0) + ci(v), color=color, alpha=0.2, lw=0)
    gap = own[y == 1].mean(0) - own[y == 0].mean(0)
    ax.plot(layers, gap, color="#52514e", lw=1.6, ls="--", label="Gap (machine − human)")
    ax.set_title(r"B  Position along the class direction $\hat{r}_l$")
    ax.set_ylabel(r"mean $\hat{r}_l \cdot h$")
    ax.legend(frameon=False, fontsize=11)
    # C: relative to the norm
    ax = axes[2]
    for lab, name, color in CLASSES:
        v = own[y == lab] / u
        ax.plot(layers, 100 * v.mean(0), color=color, lw=2, label=name)
        ax.fill_between(layers, 100 * (v.mean(0) - ci(v)), 100 * (v.mean(0) + ci(v)), color=color, alpha=0.2, lw=0)
    ax.plot(layers, 100 * gap / u, color="#52514e", lw=1.6, ls="--", label="Gap")
    ax.set_title("C  Same, as % of the token norm")
    ax.set_ylabel(r"$\hat{r}_l \cdot h \,/\, \bar{u}_l$  (%)")
    for ax in axes[:3]:
        ax.set_xlabel("layer l")
        ax.set_xlim(1, P)
    # D: fixed direction r̂_8, unsteered vs pushed at layer 8
    ax = axes[3]
    m = np.arange(FIXED_LAYER, vb.shape[-1] + 1)
    for lab, name, color in CLASSES:
        ax.plot(m, vb[y == lab, FIXED_LAYER - 1][:, m - 1].mean(0), color=color, lw=2, label=f"{name}, unsteered")
        ax.plot(m, vs[y == lab, FIXED_LAYER - 1][:, m - 1].mean(0), color=color, lw=2, ls="--",
                label=f"{name}, pushed at L{FIXED_LAYER}")
    ax.set_title(rf"D  Along $\hat{{r}}_{{{FIXED_LAYER}}}$: push at L{FIXED_LAYER}, c = {STRENGTH}")
    ax.set_xlabel("layer")
    ax.set_ylabel(rf"mean $\hat{{r}}_{{{FIXED_LAYER}}} \cdot h$")
    ax.set_xlim(FIXED_LAYER, m[-1])
    ax.legend(frameon=False, fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Unsteered machine–human gap vs token norm over depth · {meta['model']} · {meta['dataset']} "
                 f"(test texts, n = {len(y)}; band = 95% CI)", fontsize=14)
    fig.tight_layout()
    out = OUTPUT_DIR / f"f_steering_gap_{args.dataset}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    # numbers for the text
    for l in (2, 4, 8, 12, 16, 20):
        print(f"L{l}: norm {u[l-1]:.2f} | machine {own[y==1, l-1].mean():.2f} human {own[y==0, l-1].mean():.2f} "
              f"gap {gap[l-1]:.2f} | gap/norm {100*gap[l-1]/u[l-1]:.1f}%")


if __name__ == "__main__":
    main()
