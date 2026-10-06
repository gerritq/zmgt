"""Compare one idea method against all available baselines on the main subsets."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

from src.config import Config

cfg = Config()

BASE_DIR = Path(cfg.base_dir)
IDEAS_SRC_DIR = BASE_DIR / "src" / "ideas"
# Baselines computed in the zero project first, zmgt sandbox runs take precedence.
BASELINE_DIRS = (
    BASE_DIR / "output" / "baselines" / "zero_folder_baselines",
    Path(cfg.baseline_output_dir),
)
IDEAS_DIR = Path(cfg.zero_output_dir) / "ideas"
OUTPUT_DIR = Path(cfg.item_output_dir)

DATASETS = (
    ("drlXAttacks_decoder_paraphrasing", "Paraphrasing"),
    ("raidDomain_wiki", "RAID Wiki"),
    ("drlXAttacks_character_deletion", "Char. Deletion"),
)

METHOD_NAMES = {
    "fdgpt": "FastDetectGPT",
    "lastde": "Lastde++",
    "irm": "IRM",
    "detectllm": "DetectLLM",
    "nts": "NTS",
    "binoculars": "Binoculars",
    "curvature": "Curvature",
}

METRICS = (
    ("auroc", r"\textbf{AUC $\uparrow$}"),
    ("tpr_at_fpr_0_05", r"\textbf{TPR@5\% $\uparrow$}"),
)

OURS = "__ours__"
AVERAGE = "average"

SeedScores = dict[tuple[str, str], dict[int, float]]
ScoreSummary = tuple[float, float]


def available_ideas() -> list[str]:
    """Idea method names, derived from ``src/ideas/<n>_<method>.py``."""
    return sorted(
        re.sub(r"^\d+_", "", path.stem)
        for path in IDEAS_SRC_DIR.glob("*.py")
        if not path.name.startswith("_")
    )


def read_baselines(metric: str, datasets: set[str]) -> SeedScores:
    """Return seed-level baseline values keyed by (method, dataset)."""
    results: SeedScores = defaultdict(dict)
    for directory in BASELINE_DIRS:
        seen: set[tuple[str, str, int]] = set()
        for path in sorted(directory.glob("*.json")):
            with path.open(encoding="utf-8") as file:
                record = json.load(file)
            if record["dataset"] not in datasets or record["model"] not in METHOD_NAMES:
                continue
            method, seed = str(record["model"]), int(record["seed"])
            if (method, record["dataset"], seed) in seen:
                raise ValueError(f"Duplicate seed {seed} for {method} on {record['dataset']}.")
            seen.add((method, record["dataset"], seed))
            results[(method, record["dataset"])][seed] = float(record["metrics"][metric])
    return results


def read_ideas(metric: str, method: str, model: str, datasets: set[str]) -> SeedScores:
    """Return seed-level idea values keyed by (OURS, dataset)."""
    results: SeedScores = defaultdict(dict)
    for path in sorted(IDEAS_DIR.glob(f"{method}_{model}_*.json")):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        if record.get("method") != method or record.get("model") != model:
            continue
        if record["dataset"] not in datasets:
            continue
        seed = int(record["seed"])
        if seed in results[(OURS, record["dataset"])]:
            raise ValueError(f"Duplicate seed {seed} for {method} on {record['dataset']}.")
        results[(OURS, record["dataset"])][seed] = float(record["metrics"][metric])
    return results


def mean_and_std(values: list[float]) -> ScoreSummary:
    """Return the mean and sample standard deviation across seed-level values."""
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def cell_summary(method: str, scores: SeedScores, column: str) -> ScoreSummary | None:
    """Summarize one dataset cell, or the per-seed dataset average for AVERAGE.

    The average only uses seeds for which every dataset is present, so methods
    with missing datasets get no average rather than a partial one.
    """
    if column != AVERAGE:
        seed_values = scores.get((method, column))
        return mean_and_std(list(seed_values.values())) if seed_values else None

    per_dataset = [scores.get((method, dataset), {}) for dataset, _ in DATASETS]
    common_seeds = set.intersection(*(set(values) for values in per_dataset))
    seed_averages = [
        fmean(values[seed] for values in per_dataset) for seed in sorted(common_seeds)
    ]
    return mean_and_std(seed_averages) if seed_averages else None


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("&", r"\&")


def format_score(score: ScoreSummary | None) -> str:
    """Render a mean with its seed standard deviation as a LaTeX subscript."""
    if score is None:
        return "--"
    mean, std = score
    return rf"{mean * 100:.1f}$_{{{std * 100:.1f}}}$"


def highlighted_score(score: ScoreSummary | None, style: str | None) -> str:
    rendered = format_score(score)
    if style == "best":
        return rf"\cellcolor{{cyan!25}}\textbf{{{rendered}}}"
    if style == "second":
        return rf"\cellcolor{{orange!25}}\underline{{{rendered}}}"
    return rendered


def columns() -> list[str]:
    return [dataset for dataset, _ in DATASETS] + [AVERAGE]


def column_highlights(
    baseline_methods: list[str],
    baseline_scores: dict[str, SeedScores],
    ours_scores: dict[str, SeedScores],
) -> dict[tuple[str, str, str], str]:
    """Return one best and second-best style per metric table column."""
    methods = [(method, baseline_scores) for method in baseline_methods]
    methods.append((OURS, ours_scores))
    highlights: dict[tuple[str, str, str], str] = {}
    for metric, _ in METRICS:
        for column in columns():
            ranked = sorted(
                (
                    (summary[0], method)
                    for method, score_sets in methods
                    if (summary := cell_summary(method, score_sets[metric], column))
                ),
                reverse=True,
            )
            if ranked:
                highlights[(ranked[0][1], column, metric)] = "best"
            if len(ranked) > 1:
                highlights[(ranked[1][1], column, metric)] = "second"
    return highlights


def render_row(
    method: str,
    score_sets: dict[str, SeedScores],
    highlights: dict[tuple[str, str, str], str],
    label: str | None = None,
) -> str:
    display_name = latex_escape(label or METHOD_NAMES.get(method, method))
    if method == OURS:
        display_name = rf"\textbf{{{display_name}}}"
    cells = [display_name]
    for column in columns():
        for metric, _ in METRICS:
            score = cell_summary(method, score_sets[metric], column)
            cells.append(highlighted_score(score, highlights.get((method, column, metric))))
    return " & ".join(cells) + r" \\"


def render_delta_row(
    baseline_methods: list[str],
    baseline_scores: dict[str, SeedScores],
    ours_scores: dict[str, SeedScores],
) -> str:
    """Report the idea's point gain/loss relative to the best displayed baseline."""
    cells = [r"$\Delta$ vs. BB"]
    for column in columns():
        for metric, _ in METRICS:
            ours = cell_summary(OURS, ours_scores[metric], column)
            baselines = [
                summary[0]
                for baseline in baseline_methods
                if (summary := cell_summary(baseline, baseline_scores[metric], column))
            ]
            if ours is None or not baselines:
                cells.append("--")
                continue
            delta = ours[0] - max(baselines)
            color = "green!60!black" if delta >= 0 else "red"
            cells.append(rf"\textcolor{{{color}}}{{{delta * 100:+.1f}}}")
    return " & ".join(cells) + r" \\"


def render_table(
    baseline_scores: dict[str, SeedScores],
    ours_scores: dict[str, SeedScores],
    method: str,
) -> str:
    available = {method for method, _ in baseline_scores["auroc"]}
    baseline_methods = [name for name in METHOD_NAMES if name in available]

    metric_count = len(METRICS)
    column_count = 1 + metric_count * len(columns())
    headers = [label for _, label in DATASETS] + ["Average"]
    header_blocks = [
        r"\multicolumn{%d}{c}{\textbf{%s}}" % (metric_count, header) for header in headers
    ]
    cmidrules = [
        rf"\cmidrule(lr){{{2 + index * metric_count}-{1 + (index + 1) * metric_count}}}"
        for index in range(len(headers))
    ]
    metric_headers = " & ".join(header for _ in headers for _, header in METRICS)
    highlights = column_highlights(baseline_methods, baseline_scores, ours_scores)

    lines = [
        r"\begin{tabular}{l" + "c" * (column_count - 1) + "}",
        r"\toprule",
        " & " + " & ".join(header_blocks) + r" \\",
        " ".join(cmidrules),
        r"\textbf{Method} & " + metric_headers + r" \\",
        r"\midrule",
    ]
    lines.extend([
        *(render_row(name, baseline_scores, highlights) for name in baseline_methods),
        r"\midrule",
        render_row(OURS, ours_scores, highlights, label=METHOD_NAMES.get(method, method)),
        render_delta_row(baseline_methods, baseline_scores, ours_scores),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    return "\n".join(lines)


def report_seed_coverage(
    baseline_scores: SeedScores, ours_scores: SeedScores, method: str
) -> None:
    """Print the seeds found for every method/dataset cell."""
    methods = sorted({name for name, _ in baseline_scores})
    rows = [(name, baseline_scores) for name in methods] + [(OURS, ours_scores)]
    print("Seed coverage (method: seeds per dataset):")
    for name, scores in rows:
        coverage = ", ".join(
            f"{label}={sorted(scores.get((name, dataset), {})) or '-'}"
            for dataset, label in DATASETS
        )
        print(f"  {method if name == OURS else name}: {coverage}")


def main(model: str, method: str) -> None:
    datasets = {dataset for dataset, _ in DATASETS}
    baseline_scores = {
        metric: read_baselines(metric, datasets) for metric, _ in METRICS
    }
    ours_scores = {
        metric: read_ideas(metric, method, model, datasets) for metric, _ in METRICS
    }
    if not ours_scores["auroc"]:
        raise ValueError(f"No {method} results for model {model} in {IDEAS_DIR}.")
    report_seed_coverage(baseline_scores["auroc"], ours_scores["auroc"], method)

    output_path = OUTPUT_DIR / f"t_main_{method}_{model}.tex"
    output_path.write_text(
        render_table(baseline_scores, ours_scores, method), encoding="utf-8"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="l8b",
        help="Base-model identifier for the idea results (default: %(default)s).",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=available_ideas(),
        help="Idea method from src/ideas to compare against the baselines.",
    )
    main(**vars(parser.parse_args()))
