"""Plot detector performance by text length for selected methods."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

import matplotlib.pyplot as plt

try:
    from src.items.plot_style import configure_plot_style
except ModuleNotFoundError:
    from plot_style import configure_plot_style

configure_plot_style()


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Baselines are only computed in the sibling "zero" project.
BASELINE_DIR = REPOSITORY_ROOT.parent / "zero" / "output" / "baseline" / "sandbox"
CSD_DIR = REPOSITORY_ROOT / "output" / "zero" / "sandbox"
OUTPUT_PATH = REPOSITORY_ROOT / "output" / "items" / "f_length.pdf"
CSD_MODEL = "l8b"
SCORE = "full_log"
SCORE_TABLE = "raw_metrics_by_layer"
SCORE_LAYER = "layer_9"

METHODS = {
    "binoculars": ("Binoculars", "#E69F00", "o"),
    "nts": ("NTS", "#009E73", "^"),
    "detectllm": ("DetectLLM", "#CC79A7", "D"),
    "lastde": ("Lastde++", "#6A3D9A", "s"),
    "irm": ("IRM", "#0072B2", "P"),
    "CSD": ("Trace", "#D55E00", "*"),
}
METRIC = ("auroc", "AUC")
LENGTH_PATTERN = re.compile(r"drlXAttacks_general_(\d+)$")


def length_from_dataset(dataset: str) -> int | None:
    match = LENGTH_PATTERN.fullmatch(dataset)
    return int(match.group(1)) if match else None


SeedScores = dict[tuple[str, int], dict[int, float]]


def add_score(
    scores: SeedScores,
    method: str,
    length: int,
    seed: int,
    value: float,
) -> None:
    """Store one seed-level score, rejecting duplicate runs."""
    key = method, length
    if seed in scores[key]:
        raise ValueError(f"Duplicate seed {seed} for {method} at length {length}.")
    scores[key][seed] = value


def collect_baseline_scores(metric: str) -> SeedScores:
    scores: SeedScores = defaultdict(dict)
    for path in BASELINE_DIR.glob("*.json"):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        method = str(record["model"])
        length = length_from_dataset(record["dataset"])
        if method in METHODS and length is not None:
            add_score(
                scores, method, length, int(record["seed"]),
                float(record["metrics"][metric]),
            )
    return scores


def report_seed_coverage(label: str, scores: SeedScores) -> None:
    """Report configurations that do not have exactly three seed runs."""
    incomplete = [
        (method, length, sorted(seed_values))
        for (method, length), seed_values in scores.items()
        if len(seed_values) != 3
    ]
    if not incomplete:
        print(
            f"{label} seed coverage: all {len(scores)} method/length configurations "
            "have exactly three seed runs."
        )
        return
    print(
        f"{label} seed coverage: {len(incomplete)} of {len(scores)} method/length "
        "configurations do not have exactly three seed runs:"
    )
    for method, length, seeds in incomplete:
        print(f"  {method} at length {length}: {len(seeds)} seeds {seeds}")


def mean_and_std(values: dict[int, float]) -> tuple[float, float]:
    """Return the mean and sample standard deviation across seed runs."""
    scores = list(values.values())
    return fmean(scores), stdev(scores) if len(scores) > 1 else 0.0


def collect_csd_scores(metric: str, model: str) -> SeedScores:
    scores: SeedScores = defaultdict(dict)
    for path in CSD_DIR.glob("*.json"):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        length = length_from_dataset(record["dataset"])
        if record.get("model") == model and length is not None:
            add_score(
                scores, "CSD", length, int(record["seed"]),
                float(
                    record["metrics_by_score"][SCORE][SCORE_TABLE][SCORE_LAYER][metric]
                ),
            )
    return scores


def main(model: str = CSD_MODEL) -> None:
    metric, label = METRIC
    scores = collect_baseline_scores(metric)
    scores.update(collect_csd_scores(metric, model))
    report_seed_coverage("Length", scores)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    for method, (method_label, color, marker) in METHODS.items():
        lengths = sorted(length for candidate, length in scores if candidate == method)
        if not lengths:
            continue
        summaries = [mean_and_std(scores[method, length]) for length in lengths]
        values, stds = zip(*summaries)
        ax.errorbar(
            lengths,
            values,
            yerr=stds,
            capsize=3,
            label=method_label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Text length")
    ax.set_ylabel(label)
    ax.set_xticks(sorted({length for _, length in scores}))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=True,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.tight_layout(rect=(0, 0.19, 1, 1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=CSD_MODEL,
        help="Base-model identifier for the CSD results to include (default: %(default)s).",
    )
    parser.add_argument(
        "--score", default=SCORE,
        help="Score key in metrics_by_score (default: %(default)s).",
    )
    parser.add_argument(
        "--version", choices=("raw", "ratio"), default=SCORE_TABLE.split("_")[0],
        help="Raw per-layer metrics or ratio to layer_0 (default: %(default)s).",
    )
    parser.add_argument(
        "--layer", type=int, default=int(SCORE_LAYER.removeprefix("layer_")),
        help="Layer index (default: %(default)s).",
    )
    args = vars(parser.parse_args())
    SCORE = args.pop("score")
    SCORE_TABLE = f"{args.pop('version')}_metrics_by_layer"
    SCORE_LAYER = f"layer_{args.pop('layer')}"
    print(f"Score: {SCORE} / {SCORE_TABLE} / {SCORE_LAYER}")
    main(**args)
