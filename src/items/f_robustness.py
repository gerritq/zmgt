"""Plot robustness across detectRLX attack groups."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from src.items.plot_style import configure_plot_style
except ModuleNotFoundError:
    from plot_style import configure_plot_style

configure_plot_style()


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Baselines are only computed in the sibling "zero" project.
BASELINE_DIR = REPOSITORY_ROOT.parent / "zero" / "output" / "baseline" / "sandbox"
CSD_DIR = REPOSITORY_ROOT / "output" / "zero" / "sandbox"
OUTPUT_PATH = REPOSITORY_ROOT / "output" / "items" / "f_robustness.pdf"
CSD_MODEL = "l8b"
SCORE = "full_log"
SCORE_TABLE = "raw_metrics_by_layer"
SCORE_LAYER = "layer_9"
CSD_METHOD = "CSD"
EXPECTED_SEEDS = (42, 43, 44)

ATTACK_GROUPS = {
    "Perturbation": (
        "character_deletion",
        "character_insertion",
        "character_substitution",
        "zero_width_insertion",
    ),
    "Paraphrasing": (
        "backtranslation",
        "encoder_paraphrasing",
        "decoder_paraphrasing",
        "seq2seq_paraphrasing",
    ),
    "Structure": ("condensing", "expanding", "polishing"),
}
METHODS = {
    "detectllm": ("DetectLLM", "#CC79A7"),
    "nts": ("NTS", "#009E73"),
    "irm": ("IRM", "#0072B2"),
    "lastde": ("Lastde++", "#E69F00"),
    # "curvature": ("Curvature", "#56B4E9"),
    "CSD": ("Trace", "#D55E00"),
}
ATTACK_NAMES = {
    "character_deletion": "Char. Del.",
    "character_insertion": "Char. Ins.",
    "character_substitution": "Char. Sub.",
    "zero_width_insertion": "Zero-width Ins.",
    "backtranslation": "Backtranslation",
    "encoder_paraphrasing": "Encoder",
    "decoder_paraphrasing": "Decoder",
    "seq2seq_paraphrasing": "Seq2Seq",
}


def attack_from_dataset(dataset: str) -> str | None:
    prefix = "drlXAttacks_"
    if not dataset.startswith(prefix):
        return None
    attack = dataset.removeprefix(prefix)
    return attack if any(attack in attacks for attacks in ATTACK_GROUPS.values()) else None


SeedScores = dict[tuple[str, str], dict[int, float]]


def add_score(
    scores: SeedScores,
    method: str,
    attack: str,
    seed: int,
    value: float,
) -> None:
    """Store one seed-level AUC, rejecting duplicate runs."""
    key = method, attack
    if seed in scores[key]:
        raise ValueError(f"Duplicate seed {seed} for {method} on {attack}.")
    scores[key][seed] = value


def collect_baseline_scores() -> SeedScores:
    scores: SeedScores = defaultdict(dict)
    for path in BASELINE_DIR.glob("*.json"):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        method = str(record["model"])
        attack = attack_from_dataset(record["dataset"])
        if method in METHODS and attack is not None:
            add_score(
                scores, method, attack, int(record["seed"]),
                float(record["metrics"]["auroc"]),
            )
    return scores


def report_seed_coverage(label: str, scores: SeedScores) -> None:
    """Print present and missing seed IDs for every method/attack run."""
    print(f"{label} seed runs (expected={list(EXPECTED_SEEDS)}):")
    for method in METHODS:
        for attacks in ATTACK_GROUPS.values():
            for attack in attacks:
                present = sorted(scores.get((method, attack), {}))
                missing = [seed for seed in EXPECTED_SEEDS if seed not in present]
                print(
                    f"  {method} on {attack}: "
                    f"present={present}; missing={missing}"
                )



def mean_and_std(seed_values: dict[int, float]) -> tuple[float, float]:
    """Return the mean and sample standard deviation across seed runs."""
    values = list(seed_values.values())
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def collect_csd_scores(model: str) -> SeedScores:
    scores: SeedScores = defaultdict(dict)
    for path in CSD_DIR.glob("*.json"):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        attack = attack_from_dataset(record["dataset"])
        if record.get("model") == model and attack is not None:
            add_score(
                scores, "CSD", attack, int(record["seed"]),
                float(
                    record["metrics_by_score"][SCORE][SCORE_TABLE][SCORE_LAYER]["auroc"]
                ),
            )
    return scores


def display_name(attack: str) -> str:
    return ATTACK_NAMES.get(attack, attack.replace("_", " ").title())


def add_csd_comparison(
    axis: plt.Axes,
    position: float,
    stats: dict[str, tuple[float, float]],
    bar_positions: dict[str, float],
) -> None:
    """Annotate CSD's AUC difference from the strongest baseline for one attack."""
    baseline_methods = [method for method in METHODS if method != CSD_METHOD]
    best_baseline = max(baseline_methods, key=lambda method: stats[method][0])
    csd_mean, csd_std = stats[CSD_METHOD]
    baseline_mean, baseline_std = stats[best_baseline]
    comparison_height = max(csd_mean + csd_std, baseline_mean + baseline_std) + 0.025
    left, right = sorted((bar_positions[best_baseline], bar_positions[CSD_METHOD]))
    axis.plot(
        [left, left, right, right],
        [comparison_height - 0.008, comparison_height, comparison_height, comparison_height - 0.008],
        color="black",
        linewidth=0.8,
        clip_on=False,
    )
    delta = csd_mean - baseline_mean
    delta_color = "#009E73" if delta > 0 else "#D55E00" if delta < 0 else "black"
    axis.text(
        position,
        comparison_height + 0.012,
        f"Δ {delta:+.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color=delta_color,
        fontweight="bold",
        clip_on=False,
    )


def main(model: str = CSD_MODEL, show_csd_comparison: bool = True) -> None:
    scores = collect_baseline_scores()
    scores.update(collect_csd_scores(model))
    report_seed_coverage("Robustness", scores)
    attacks = [attack for group in ATTACK_GROUPS.values() for attack in group]
    missing = [
        f"{method}/{attack}"
        for method in METHODS
        for attack in attacks
        if not scores.get((method, attack))
    ]
    if missing:
        raise ValueError(f"Missing AUC results: {', '.join(missing)}")

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    bar_width = 0.14
    for axis, (group_name, group_attacks) in zip(axes, ATTACK_GROUPS.items()):
        ordered_attacks = group_attacks
        positions = list(range(len(ordered_attacks)))
        offsets = [
            (index - (len(METHODS) - 1) / 2) * bar_width
            for index in range(len(METHODS))
        ]
        for position, attack in zip(positions, ordered_attacks):
            stats = {
                method: mean_and_std(scores[method, attack]) for method in METHODS
            }
            # Each attack's method bars run from the lowest to highest mean AUC.
            sorted_methods = sorted(METHODS, key=lambda method: stats[method][0])
            bar_positions = {
                method: position + offsets[index]
                for index, method in enumerate(sorted_methods)
            }
            for method in sorted_methods:
                _label, color = METHODS[method]
                mean, std = stats[method]
                axis.bar(
                    bar_positions[method], mean, width=bar_width, yerr=std,
                    capsize=3, error_kw={"elinewidth": 1, "capthick": 1},
                    color=color, hatch="/", edgecolor="black", linewidth=0.8,
                )
            if show_csd_comparison:
                add_csd_comparison(axis, position, stats, bar_positions)

        axis.set_title(group_name)
        axis.set_xticks(positions, [display_name(attack) for attack in ordered_attacks])
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.set_ylim(0, 1.12 if show_csd_comparison else 1)

    axes[0].set_ylabel("AUC")
    handles = [
        Patch(facecolor=color, hatch="/", edgecolor="black", label=label)
        for label, color in METHODS.values()
    ]
    labels = [handle.get_label() for handle in handles]
    figure.legend(
        handles, labels, ncol=len(METHODS), frameon=True, loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, bbox_inches="tight")


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
    parser.add_argument(
        "--no-csd-comparison",
        action="store_false",
        dest="show_csd_comparison",
        help="Hide CSD minus best-baseline AUC annotations.",
    )
    args = vars(parser.parse_args())
    SCORE = args.pop("score")
    SCORE_TABLE = f"{args.pop('version')}_metrics_by_layer"
    SCORE_LAYER = f"layer_{args.pop('layer')}"
    print(f"Score: {SCORE} / {SCORE_TABLE} / {SCORE_LAYER}")
    main(**args)
