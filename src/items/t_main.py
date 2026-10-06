"""Generate zero-shot baseline comparison tables."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Baselines are only computed in the sibling "zero" project.
BASELINE_DIR = REPOSITORY_ROOT.parent / "zero" / "output" / "baseline" / "sandbox"
SR_DIR = REPOSITORY_ROOT / "output" / "zero" / "sandbox"
OUTPUT_DIR = REPOSITORY_ROOT / "output" / "items"
CSD_MODEL = "l8b"
SCORE = "full_log"
SCORE_TABLE = "raw_metrics_by_layer"
SCORE_LAYER = "layer_9"

METHOD_NAMES = {
    "entropy": "Entropy",
    "likelihood": "Likelihood",
    "rank": "Rank",
    "llr": "LLR",
    "gecscore": "GECScore",
    "revise": "Revise-Detect",
    "fdgpt": "FastDetectGPT",
    "lastde": "Lastde++",
    "irm": "IRM",
    "detectllm": "DetectLLM",
    "nts": "NTS",
    "binoculars": "Binoculars",
    "openai_roberta": "OpenAI RoBERTa",
    "radar": "RADAR",
}


SUPERVISED_METHODS = {
    "openai_roberta",
    "radar",
}


EXCLUDED_METHODS = {"repreguard", "editlens", "openai_roberta", "irm_curv"}
LANGUAGE_BASELINE_METHODS = ("irm", "nts", "detectllm", "lastde")
LANGUAGE_SUBSETS = ("german", "chinese", "portuguese", "russian")
MAIN_DATASET_PREFIXES = ("drlXDomain_", "drlXLang_", "drlXModel_")


METRICS = (
    ("auroc", r"\textbf{AUC $\uparrow$}"),
    ("tpr_at_fpr_0_05", r"\textbf{TPR@5\% $\uparrow$}"),
)


SeedScores = dict[tuple[str, str], dict[int, float]]
ScoreSummary = tuple[float, float]


def read_results(
    directory: Path,
    ours: bool,
    metric: str,
    model: str | None = None,
    dataset_prefixes: tuple[str, ...] | None = None,
) -> SeedScores:
    """Return seed-level metric values keyed by (method, dataset).

    When ``dataset_prefixes`` is supplied, ignore all other datasets before
    seed de-duplication.
    """
    results: SeedScores = defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)

        if ours and record.get("model") != model:
            continue
        if ours and path.name == "l8b_drlXMix_mixed_s42_w2.json":
            print(f"Skipping {path.name}: excluded legacy mixed-data result.")
            continue
        dataset = record["dataset"]
        if dataset_prefixes is not None and not dataset.startswith(dataset_prefixes):
            continue
        method = "SR@L11" if ours else str(record["model"])
        key = method, dataset
        seed = int(record["seed"])
        if seed in results[key]:
            raise ValueError(f"Duplicate seed {seed} for {method} on {dataset}.")
        value = (
            record["metrics_by_score"][SCORE][SCORE_TABLE][SCORE_LAYER][metric]
            if ours
            else record["metrics"][metric]
        )
        results[key][seed] = float(value)
    return results


def report_seed_coverage(label: str, scores: SeedScores) -> None:
    """Report configurations that do not have exactly three seed runs."""
    incomplete = [
        (method, dataset, sorted(seed_values))
        for (method, dataset), seed_values in scores.items()
        if len(seed_values) != 3
    ]
    if not incomplete:
        print(
            f"{label} seed coverage: all {len(scores)} method/dataset configurations "
            "have exactly three seed runs."
        )
        return
    print(
        f"{label} seed coverage: {len(incomplete)} of {len(scores)} method/dataset "
        "configurations do not have exactly three seed runs:"
    )
    for method, dataset, seeds in incomplete:
        print(f"  {method} on {dataset}: {len(seeds)} seeds {seeds}")


def mean_and_std(values: list[float]) -> ScoreSummary:
    """Return the mean and sample standard deviation across seed-level values."""
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def score_summary(seed_values: dict[int, float] | None) -> ScoreSummary | None:
    """Summarize a single method/dataset cell over its seed runs."""
    return mean_and_std(list(seed_values.values())) if seed_values else None


def subset_name(dataset: str, prefix: str) -> str | None:
    start = f"{prefix}_"
    return dataset[len(start):] if dataset.startswith(start) else None


DISPLAY_NAMES = {
    "gpt_4o": "GPT-4o",
}


def title_case(value: str) -> str:
    if re.fullmatch(r"\d+_\d+", value):
        return value.replace("_", ".")
    if (scientific := re.fullmatch(r"(\d+)e_(\d+)", value)) is not None:
        return f"{scientific.group(1)}e-{scientific.group(2)}"
    return DISPLAY_NAMES.get(value, value.replace("_", " ").title())


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("&", r"\&")


def is_excluded_baseline_method(method: str, excluded_methods: set[str]) -> bool:
    """Return whether a baseline is explicitly excluded or curvature-derived."""
    return method in excluded_methods or "curvature" in method.lower()


def format_score(score: ScoreSummary | None) -> str:
    """Render a mean with its seed standard deviation as a LaTex subscript."""
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


def row_average(
    method: str,
    scores: SeedScores,
    datasets: list[str],
) -> ScoreSummary | None:
    """Average corresponding datasets within each seed, then summarize seeds."""
    values_by_seed: dict[int, list[float]] = defaultdict(list)
    for dataset in datasets:
        for seed, value in scores.get((method, dataset), {}).items():
            values_by_seed[seed].append(value)
    seed_averages = [fmean(values) for values in values_by_seed.values()]
    return mean_and_std(seed_averages) if seed_averages else None


def summary_mean(summary: ScoreSummary | None) -> float | None:
    return None if summary is None else summary[0]


def column_highlights(
    baseline_methods: list[str],
    baseline_scores: dict[str, SeedScores],
    sr_scores: dict[str, SeedScores],
    datasets: list[str],
    average_groups: list[tuple[str, list[str]]],
) -> dict[tuple[str, str, str], str]:
    """Return one best and second-best style per metric table column."""
    methods = [(method, baseline_scores) for method in baseline_methods]
    methods.append(("SR@L11", sr_scores))
    average_dataset_map = dict(average_groups)
    highlights: dict[tuple[str, str, str], str] = {}

    for metric, _ in METRICS:
        for column in [*datasets, *(key for key, _ in average_groups)]:
            ranked = []
            for method, score_sets in methods:
                scores = score_sets[metric]
                value = (
                    row_average(method, scores, average_dataset_map[column])
                    if column in average_dataset_map
                    else score_summary(scores.get((method, column)))
                )
                if (mean := summary_mean(value)) is not None:
                    ranked.append((mean, method))
            ranked.sort(key=lambda item: item[0], reverse=True)
            if ranked:
                highlights[(ranked[0][1], column, metric)] = "best"
            if len(ranked) > 1:
                highlights[(ranked[1][1], column, metric)] = "second"
    return highlights


def render_row(
    method: str,
    score_sets: dict[str, SeedScores],
    datasets: list[str],
    average_groups: list[tuple[str, list[str]]],
    highlights: dict[tuple[str, str, str], str],
    label: str | None = None,
    bold_label: bool = False,
) -> str:
    display_name = latex_escape(label or METHOD_NAMES.get(method, method))
    if bold_label:
        display_name = rf"\textbf{{{display_name}}}"

    cells = [display_name]
    for dataset in datasets:
        for metric, _ in METRICS:
            score = score_summary(score_sets[metric].get((method, dataset)))
            cells.append(highlighted_score(score, highlights.get((method, dataset, metric))))
    for average_key, average_datasets in average_groups:
        for metric, _ in METRICS:
            average = row_average(method, score_sets[metric], average_datasets)
            cells.append(highlighted_score(average, highlights.get((method, average_key, metric))))
    return " & ".join(cells) + r" \\"


def render_delta_row(
    method: str,
    sr_scores: dict[str, SeedScores],
    baseline_scores: dict[str, SeedScores],
    baseline_methods: list[str],
    datasets: list[str],
    average_groups: list[tuple[str, list[str]]],
) -> str:
    """Report CSD's point gain/loss relative to the best displayed baseline."""
    def format_delta(delta: float | None) -> str:
        if delta is None:
            return "--"
        color = "green!60!black" if delta >= 0 else "red"
        return rf"\textcolor{{{color}}}{{{delta * 100:+.1f}}}"

    cells = [r"$\Delta$ vs. BB"]
    for dataset in datasets:
        for metric, _ in METRICS:
            score = summary_mean(score_summary(sr_scores[metric].get((method, dataset))))
            baselines = [
                summary_mean(score_summary(baseline_scores[metric].get((baseline, dataset))))
                for baseline in baseline_methods
            ]
            baselines = [value for value in baselines if value is not None]
            delta = None if score is None or not baselines else score - max(baselines)
            cells.append(format_delta(delta))
    for _, average_datasets in average_groups:
        for metric, _ in METRICS:
            sr_average = summary_mean(row_average(method, sr_scores[metric], average_datasets))
            baseline_averages = [
                summary_mean(row_average(baseline, baseline_scores[metric], average_datasets))
                for baseline in baseline_methods
            ]
            baseline_averages = [value for value in baseline_averages if value is not None]
            delta = (
                None
                if sr_average is None or not baseline_averages
                else sr_average - max(baseline_averages)
            )
            cells.append(format_delta(delta))
    return " & ".join(cells) + r" \\"


def render_table(
    groups: list[tuple[str, str, list[str]]],
    baseline_scores: dict[str, SeedScores],
    sr_scores: dict[str, SeedScores],
    include_supervised: bool,
    allowed_baseline_methods: tuple[str, ...] | None = None,
    excluded_baseline_methods: set[str] | None = None,
) -> str:
    """Render one table containing one or more dimension groups."""
    group_datasets = [
        [f"{prefix}_{subset}" for subset in subsets]
        for _, prefix, subsets in groups
    ]
    datasets = [dataset for grouped in group_datasets for dataset in grouped]
    metric_count = len(METRICS)
    multiple_groups = len(groups) > 1
    average_groups = (
        [(f"average_{index}", grouped) for index, grouped in enumerate(group_datasets)]
        if multiple_groups
        else [("average", datasets)]
    )
    excluded_baseline_methods = (
        EXCLUDED_METHODS
        if excluded_baseline_methods is None
        else excluded_baseline_methods
    )
    available_methods = {
        method
        for method, dataset in baseline_scores["auroc"]
        if dataset in datasets
        and not is_excluded_baseline_method(method, excluded_baseline_methods)
    }
    if allowed_baseline_methods is None:
        baseline_methods = [
            method for method in METHOD_NAMES if method in available_methods
        ]
        baseline_methods.extend(sorted(available_methods - set(METHOD_NAMES)))
    else:
        baseline_methods = [
            method for method in allowed_baseline_methods if method in available_methods
        ]
    if not include_supervised:
        baseline_methods = [
            method for method in baseline_methods if method not in SUPERVISED_METHODS
        ]
    supervised_methods = [
        method for method in baseline_methods if method in SUPERVISED_METHODS
    ]
    zero_shot_methods = [
        method for method in baseline_methods if method not in SUPERVISED_METHODS
    ]
    column_count = 1 + metric_count * (len(datasets) + len(average_groups))

    subset_header_blocks, subset_cmidrules = [], []
    first_column = 2
    for (_, _, subsets), grouped_datasets in zip(groups, group_datasets):
        subset_header_blocks.extend(
            r"\multicolumn{%d}{c}{\textbf{%s}}" % (metric_count, title_case(subset))
            for subset in subsets
        )
        for _ in grouped_datasets:
            subset_last_column = first_column + metric_count - 1
            subset_cmidrules.append(
                rf"\cmidrule(lr){{{first_column}-{subset_last_column}}}"
            )
            first_column = subset_last_column + 1

        if multiple_groups:
            average_last_column = first_column + metric_count - 1
            subset_header_blocks.append(
                r"\multicolumn{%d}{c}{\textbf{Average}}" % metric_count
            )
            first_column = average_last_column + 1

    if not multiple_groups:
        subset_header_blocks.append(r"\multicolumn{%d}{c}{\textbf{Average}}" % metric_count)

    metric_headers = " & ".join(
        header
        for _ in range(len(datasets) + len(average_groups))
        for _, header in METRICS
    )
    lines = [
        r"\begin{tabular}{l" + "c" * (column_count - 1) + "}",
        r"\toprule",
        " & " + " & ".join(subset_header_blocks) + r"\\",
        " ".join(subset_cmidrules),
        r"\textbf{Method} & " + metric_headers + r" \\",
        r"\midrule",
    ]
    highlights = column_highlights(
        baseline_methods, baseline_scores, sr_scores, datasets, average_groups
    )
    if supervised_methods:
        lines.extend([
            rf"\rowcolor{{gray!25}}\multicolumn{{{column_count}}}{{c}}{{\textbf{{Supervised}}}} " + r"\\",
            r"\midrule",
            *(
                render_row(method, baseline_scores, datasets, average_groups, highlights)
                for method in supervised_methods
            ),
            r"\midrule",
        ])

    if include_supervised:
        lines.extend([
            rf"\rowcolor{{gray!25}}\multicolumn{{{column_count}}}{{c}}{{\textbf{{Zero-shot}}}} " + r"\\",
            r"\midrule",
        ])

    lines.extend([
        *(
            render_row(method, baseline_scores, datasets, average_groups, highlights)
            for method in zero_shot_methods
        ),
        r"\midrule",
        render_row(
            "SR@L11", sr_scores, datasets, average_groups, highlights,
            "Trace", bold_label=True,
        ),
        render_delta_row(
            "SR@L11", sr_scores, baseline_scores, baseline_methods, datasets, average_groups
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    return "\n".join(lines)


def discovered_subsets(
    scores: SeedScores,
    prefix: str,
) -> list[str]:
    return sorted(
        {
            subset
            for _, dataset in scores
            if (subset := subset_name(dataset, prefix)) is not None
        }
    )


def numeric_general_order(subsets: list[str]) -> list[str]:
    return sorted(
        {subset for subset in subsets if re.fullmatch(r"general_\d+", subset)},
        key=lambda value: int(value.split("_", 1)[1]),
    )


def report_table_missing_seed_counts(
    table_name: str,
    groups: list[tuple[str, str, list[str]]],
    baseline_scores: dict[str, SeedScores],
    sr_scores: dict[str, SeedScores],
) -> None:
    """Report missing seed runs for each displayed method in one table."""
    datasets = [
        f"{prefix}_{subset}"
        for _, prefix, subsets in groups
        for subset in subsets
    ]
    expected_count = 3 * len(datasets)
    print(
        f"{table_name} missing seed runs (expected {expected_count} per method):"
    )
    for source, scores in (("Baseline", baseline_scores["auroc"]), ("CSD", sr_scores["auroc"])):
        methods = sorted({
            method
            for method, dataset in scores
            if dataset in datasets
            and (
                source != "Baseline"
                or not is_excluded_baseline_method(method, EXCLUDED_METHODS)
            )
        })
        for method in methods:
            observed_count = sum(
                len(scores.get((method, dataset), {})) for dataset in datasets
            )
            missing_count = max(expected_count - observed_count, 0)
            print(
                f"  {source} {method}: {missing_count} missing "
                f"({observed_count}/{expected_count} present)"
            )



def main(model: str = CSD_MODEL) -> None:
    baseline_scores = {
        metric: read_results(
            BASELINE_DIR, ours=False, metric=metric,
            dataset_prefixes=MAIN_DATASET_PREFIXES,
        )
        for metric, _ in METRICS
    }
    sr_scores = {
        metric: read_results(
            SR_DIR, ours=True, metric=metric, model=model,
            dataset_prefixes=MAIN_DATASET_PREFIXES,
        )
        for metric, _ in METRICS
    }


    all_auc_scores = {**baseline_scores["auroc"], **sr_scores["auroc"]}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "domains",
            [("Domains", "drlXDomain", discovered_subsets(all_auc_scores, "drlXDomain"))],
        ),
        (
            "models",
            [("Models", "drlXModel", discovered_subsets(all_auc_scores, "drlXModel"))],
        ),
        (
            "languages",
            [("Languages", "drlXLang", list(LANGUAGE_SUBSETS))],
        ),
    ]
    for filename, groups in specs:
        report_table_missing_seed_counts(filename, groups, baseline_scores, sr_scores)
        if any(not subsets for _, _, subsets in groups):
            raise ValueError(f"No results found for the {filename} table.")
        for include_supervised, suffix in ((False, ""),):
            (OUTPUT_DIR / f"t_{filename}{suffix}.tex").write_text(
                render_table(
                    groups, baseline_scores, sr_scores, include_supervised,
                    LANGUAGE_BASELINE_METHODS if filename == "languages" else None,
                ),
                encoding="utf-8",
            )


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
