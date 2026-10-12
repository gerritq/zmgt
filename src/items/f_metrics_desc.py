"""
Line plots of the human vs. machine descriptive statistics on the raw hidden states, one PDF per axis
(f_metrics_desc_{within,across}_layer_raw): one panel per metric with the group means by layer (left axis) and the AUC
of the raw value for machine vs. human (right axis).
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

STATS = (
    ("curvature", "Curvature"),
    ("angle", "Angle"),
    ("magnitude", "Magnitude"),
    ("hidden_norm", r"$\|h\|_2$"),
)
GROUPS = (("human", 0, "#1f77b4"), ("machine", 1, "#d62728"))
AUC_COLOR = "#555555"
# Metrics lower for machine text: plotted as 1 − AUC, i.e. P(machine < human), marked in the panel title.
REVERSED = ("angle",)
AXES = ("within_layer", "across_layer")
STATES = "raw"
# across_layer entry i is the layer step starting at layer i + 1 (curvature: layers i + 1..i + 3, other metrics:
# i + 1 vs. i + 2, ||h||: layer i + 1); within_layer entry i is layer i + 1.
XLABELS = {"within_layer": "Layer", "across_layer": "Layer step (from layer)"}


def finite(values: list) -> np.ndarray:
    """The finite values (null marks a text too short for the metric)."""
    x = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    return x[np.isfinite(x)]


def auc(human: np.ndarray, machine: np.ndarray) -> float:
    """P(machine > human) from the Mann–Whitney U statistic."""
    if not (len(human) and len(machine)):
        return float("nan")
    return float(scipy_stats.mannwhitneyu(machine, human).statistic / (len(machine) * len(human)))


def summarize(items: list[dict], stat: str, axis: str) -> dict[str, np.ndarray]:
    """
    Per layer, the mean of each group, its 95% CI half-width (1.96 standard errors) and the AUC:
    {"human": (L,), "machine": (L,), "human_ci": (L,), "machine_ci": (L,), "auc": (L,)}.
    """
    # across_layer keeps the per-layer values under by_layer (by_token is indexed by token position).
    values = (lambda item: item[stat]) if axis == "within_layer" else (lambda item: item["by_layer"][stat])
    count = len(values(items[0]))
    per_group = {
        group: [finite([values(item)[layer] for item in items if item["label"] == label]) for layer in range(count)]
        for group, label, _ in GROUPS
    }
    out = {group: np.asarray([x.mean() if len(x) else np.nan for x in v]) for group, v in per_group.items()}
    for group, v in per_group.items():
        out[f"{group}_ci"] = np.asarray([1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan for x in v])
    out["auc"] = np.asarray([auc(h, m) for h, m in zip(per_group["human"], per_group["machine"])])
    return out


def plot(items: list[dict], stats: list[str], axis: str, title: str, path: Path) -> None:
    shown = [(stat, name) for stat, name in STATS if stat in stats]
    fig, panels = plt.subplots(1, len(shown), figsize=(4.8 * len(shown), 4.2), layout="constrained")
    panels = np.atleast_1d(panels)
    for i, (ax, (stat, name)) in enumerate(zip(panels, shown)):
        summary = summarize(items, stat, axis)
        if stat in REVERSED:
            summary["auc"] = 1 - summary["auc"]
        layers = np.arange(1, len(summary["auc"]) + 1)
        for group, _, color in GROUPS:
            ax.plot(layers, summary[group], color=color, linewidth=1.6, label=f"{group.capitalize()} mean (95% CI)")
            ax.fill_between(layers, summary[group] - summary[f"{group}_ci"], summary[group] + summary[f"{group}_ci"],
                            color=color, alpha=0.2, linewidth=0)
        ax.set_title(f"{name} (AUC reversed)" if stat in REVERSED else name)
        ax.set_xlabel(XLABELS[axis])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("Mean")

        twin = ax.twinx()
        twin.plot(layers, summary["auc"], color=AUC_COLOR, linewidth=1.2, linestyle="--", label="AUC")
        twin.axhline(0.5, color=AUC_COLOR, linewidth=0.6, linestyle=":")
        twin.set_ylim(0, 1)
        if i == len(shown) - 1:
            twin.set_ylabel("AUC", color=AUC_COLOR)
        twin.tick_params(axis="y", colors=AUC_COLOR)

    # Legend from the first panel and its AUC axis.
    first = panels[0]
    lines = [line for ax in (first, *first.get_shared_x_axes().get_siblings(first)) for line in ax.get_lines()]
    lines = list(dict.fromkeys(line for line in lines if not line.get_label().startswith("_")))
    fig.legend(lines, [line.get_label() for line in lines], loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(lines), frameon=False)
    fig.suptitle(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main(model: str, dataset: str, seed: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for axis in AXES:
        desc = json.loads((DESC_STATS_DIR / f"desc_stats_{axis}_{STATES}_{model}_{dataset}_s{seed}.json").read_text())
        plot(
            desc["items"], desc["stats"], axis, f"{model}, {dataset}: {axis.replace('_', ' ')} ({STATES})",
            OUTPUT_DIR / f"f_metrics_desc_{axis}_{STATES}_{model}_{dataset}_s{seed}.pdf",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.model, args.dataset, args.seed)
