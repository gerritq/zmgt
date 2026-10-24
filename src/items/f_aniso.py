"""
Where does the machine − human energy gap come from? By relative layer depth, from the aniso metrics
(output/metrics/aniso) of one token variant (--variant: all, skip10pct, med5, med10; see src.geometric_metrics.aniso):
- f_aniso_energy_gap_{variant}: one panel per model, equal-weight mean over the domains
- f_aniso_energy_gap_{model}_{variant}: one figure per model, one panel per domain
Curves:
    total:           100 · (machine mean − human mean of norm2) / human mean of norm2
    centroid:        the same for shared_centroid = ‖μ‖², over the same denominator
    token specific:  the same for token_specific = mean_t ‖h_t − μ‖², over the same denominator
The two parts add up to the total (norm2 = shared_centroid + token_specific). The embedding (layer 0) and the last
entry (the final-norm output) are left out. Shading marks the first stretch where the centroid part exceeds the
token-specific part (the intermediate layers), between the crossings of the two curves; later stretches are not shaded.
If that stretch runs to the last layer, it fades out with a gradient.
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

MODELS = (("l8b", "Llama-3.1-8B"), ("m8b", "Ministral-8B"), ("q8b", "Qwen3-8B"))
DOMAINS = ("news", "novel", "seo", "wiki")
DATASET = "drlXDomainLen_{domain}_500_tokens"
# Token variants stored by src.geometric_metrics.aniso (kept in sync with its VARIANTS).
VARIANTS = ("all", "skip10pct", "med5", "med10")
SCORES = ("norm2", "shared_centroid", "token_specific")
COLORS = ("#252525", "#0072B2", "#D55E00")
STYLES = ("-", "-", "--")
LABELS = ("Total energy gap", "Centroid contribution", "Token-specific contribution")
SHADE = "#E8EDF1"
SMOOTH_HALF_WIDTH = 2  # centred 5-layer moving average
MIN_STRETCH = 0.02  # narrower shaded stretches (in relative depth) are skipped as noise


def domain_contributions(model: str, domain: str, seed: int, variant: str) -> np.ndarray:
    """(3, L + 1): total, centroid and token-specific gap in % of the human total energy."""
    path = ANISO_DIR / f"aniso_{model}_{DATASET.format(domain=domain)}_s{seed}.json"
    metrics = json.loads(path.read_text())["variants"][variant]["metrics"]
    denominator = np.asarray([v["mean_human"] for v in metrics["norm2"]])
    values = np.asarray([[100 * (v["mean_machine"] - v["mean_human"]) for v in metrics[score]]
                         for score in SCORES]) / denominator
    assert np.max(np.abs(values[0] - values[1] - values[2])) < 0.01, f"decomposition broken for {path.name}"
    return values


def smooth(values: np.ndarray) -> np.ndarray:
    """Centred moving average over layers 1..L − 1 (the same linear filter on every row keeps total = sum of parts)."""
    out = values.copy()
    last = values.shape[1] - 1
    for j in range(1, last):
        out[:, j] = values[:, max(1, j - SMOOTH_HALF_WIDTH):min(last, j + SMOOTH_HALF_WIDTH + 1)].mean(axis=1)
    return out


def centroid_stretches(depth: np.ndarray, values: np.ndarray) -> list[tuple[float, float, bool]]:
    """
    Stretches where the centroid curve is above the token-specific one: (start, end, open_end), with the ends at the
    crossings (linear interpolation between layers); open_end when the stretch runs to the last layer.
    """
    diff = values[1] - values[2]
    above = diff > 0

    def crossing(i: int) -> float:  # between layers i and i + 1
        return float(depth[i] + (depth[i + 1] - depth[i]) * diff[i] / (diff[i] - diff[i + 1]))

    out, start = [], (float(depth[0]) if above[0] else None)
    for i in range(len(diff) - 1):
        if above[i + 1] and not above[i]:
            start = crossing(i)
        elif above[i] and not above[i + 1]:
            out.append((start, crossing(i), False))
            start = None
    if start is not None:
        out.append((start, 1.0, True))
    return out


def shade(ax: plt.Axes, stretches: list[tuple[float, float, bool]]) -> None:
    """
    The first stretch at least MIN_STRETCH wide only (the intermediate one; later ones, e.g. near the last layer, are
    not shaded): solid between its crossings, fading out to the right if it runs to the last layer.
    """
    wide = [stretch for stretch in stretches if stretch[1] - stretch[0] >= MIN_STRETCH]
    if not wide:
        return
    start, end, open_end = wide[0]
    if not open_end:
        ax.axvspan(start, end, color=SHADE, zorder=0)
        return
    edges = np.linspace(start, end, 61)
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        ax.axvspan(left, right, color=SHADE, alpha=1 - i / (len(edges) - 1), linewidth=0, zorder=0)


def prepare(values: np.ndarray, smoothed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Optional smoothing, then drop the embedding and the final-norm entry: depth (L − 1,), values (3, L − 1)."""
    if smoothed:
        values = smooth(values)
    depth = np.arange(values.shape[1]) / (values.shape[1] - 1)
    return depth[1:-1], values[:, 1:-1]


def plot_panels(panels: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    """One panel per (title, depth, values), shared y axis, legend below."""
    shown = np.concatenate([values.ravel() for _, _, values in panels])
    lo, hi = min(0, shown.min()), max(0, shown.max())
    padding = max(1.5, 0.10 * (hi - lo))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 3.2, 4.2), sharey=True)
    for col, (ax, (title, depth, values)) in enumerate(zip(axes, panels)):
        stretches = centroid_stretches(depth, values)
        print(f"{path.stem} / {title}: centroid above token specific at depth "
              f"{[(round(a, 3), round(b, 3)) for a, b, _ in stretches]}")
        shade(ax, stretches)
        ax.axhline(0, color="#7F8C8D", linewidth=0.8, zorder=1)
        for k in range(len(SCORES)):
            ax.plot(depth, values[k], color=COLORS[k], linestyle=STYLES[k], linewidth=2.4 if k == 0 else 2.0,
                    label=LABELS[k])
        ax.set_xlim(0, 1)
        ax.set_ylim(lo - padding, hi + padding)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#E2E6EA", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Relative layer depth")
        if col == 0:
            ax.set_ylabel("Machine − human energy\n(% of human total energy)")
        ax.set_title(title, pad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels), frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.25, wspace=0.12)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def main(seed: int, variant: str, smoothed: bool) -> None:
    configure_plot_style()
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#263238", "text.color": "#263238",
        "xtick.color": "#455A64", "ytick.color": "#455A64", "axes.edgecolor": "#B0BEC5", "pdf.fonttype": 42,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{variant}_s{seed}" + ("_smoothed" if smoothed else "")
    by_model = []
    for model, name in MODELS:
        raw = {domain: domain_contributions(model, domain, seed, variant) for domain in DOMAINS}
        plot_panels([(domain, *prepare(values, smoothed)) for domain, values in raw.items()],
                    OUTPUT_DIR / f"f_aniso_energy_gap_{model}_{suffix}.pdf")
        by_model.append((name, *prepare(np.mean(list(raw.values()), axis=0), smoothed)))
    plot_panels(by_model, OUTPUT_DIR / f"f_aniso_energy_gap_{suffix}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default="med5", choices=VARIANTS)
    parser.add_argument("--smooth", action="store_true", help="centred 5-layer moving average")
    args = parser.parse_args()
    main(args.seed, args.variant, args.smooth)
