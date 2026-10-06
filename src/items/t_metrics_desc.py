"""
Tables of the human vs. machine descriptive statistics: one per axis of the geometric metrics and hidden states, raw or
centred (t_metrics_{within,across}_layer_{raw,centred}), each by layer; for across_layer also pooled over the layers
(t_metrics_across_layer_pooled_{raw,centred}); and one of the per-text scalars, text lengths and mean next-token
entropy (t_metrics_len).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    ("magnitude_unit", "Magnitude (unit)"),
    ("hidden_norm", r"$\|h\|_2$"),
)
GROUPS = (("human", "H"), ("machine", "M"))
# Columns per stat: the two groups, the mean difference M − H with the p-value of Welch's t-test as subscript, and
# the AUC of the raw value for machine vs. human.
PAIR_COLUMNS = ("H", "M", r"$\Delta_p$", "AUC")
LABELS = {"human": 0, "machine": 1}
# Per-text scalars: the text lengths and the mean next-token entropy over the sequence (nats).
TEXT_STATS = (("chars", "Characters"), ("tokens", "Tokens"), ("entropy", "Entropy (nats)"))
# Every other layer, the last layer always included.
LAYER_STEP = 2
AXES = ("within_layer", "across_layer")
# Hidden states the metrics run on: as is, or minus the mean along the axis.
STATES = ("raw", "centred")
STATE_CAPTIONS = {
    ("raw", "within_layer"): "hidden states as is",
    ("raw", "across_layer"): "hidden states as is",
    ("raw", "across_layer_pooled"): "hidden states as is",
    ("centred", "across_layer_pooled"): "hidden states minus the token's mean state over the transformer layers",
    ("centred", "within_layer"): "hidden states minus the text's mean state over its tokens at every layer",
    ("centred", "across_layer"): "hidden states minus the token's mean state over the transformer layers",
}
# Comment on top of each table; across_layer row l is the layer step starting at layer l.
CAPTIONS = {
    "within_layer": "metrics against the previous token at the same layer, mean over tokens per text",
    "across_layer": (
        "metrics against the same token at the previous layer, mean over tokens per text; row l is the step "
        "from layer l (curvature: layers l..l+2, other metrics: l vs. l+1, ||h||: layer l)"
    ),
    "across_layer_pooled": (
        "metrics against the same token at the previous layer, per token the mean over the layer steps, then the mean "
        "over tokens per text (no layer-specific view)"
    ),
    "len": "text length and mean next-token entropy over the sequence (nats) per text",
}


def finite(values: list) -> np.ndarray:
    """The finite values (null marks a text too short for the metric)."""
    x = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    return x[np.isfinite(x)]


def group_stats(values: list) -> dict:
    """Mean, std and count of the finite values."""
    x = finite(values)
    return {
        "mean": float(x.mean()) if len(x) else float("nan"),
        "std": float(x.std()) if len(x) else float("nan"),
        "n": int(len(x)),
    }


def compare(values: dict[str, list]) -> dict:
    """
    Per-group summaries plus the mean difference (machine − human), the two-sided p-value of Welch's t-test and the
    AUC P(machine > human) from the Mann–Whitney U statistic: {group: {mean, std, n}, "diff": ..., "p": ..., "auc": ...}.
    """
    out = {group: group_stats(v) for group, v in values.items()}
    human, machine = finite(values["human"]), finite(values["machine"])
    out["diff"] = out["machine"]["mean"] - out["human"]["mean"]
    out["p"] = (
        float(scipy_stats.ttest_ind(machine, human, equal_var=False).pvalue)
        if len(human) > 1 and len(machine) > 1 else float("nan")
    )
    out["auc"] = (
        float(scipy_stats.mannwhitneyu(machine, human).statistic / (len(machine) * len(human)))
        if len(human) and len(machine) else float("nan")
    )
    return out


def by_group(items: list[dict]) -> dict[str, list[dict]]:
    return {group: [item for item in items if item["label"] == label] for group, label in LABELS.items()}


def summarize_layers(items: list[dict], stats: list[str], axis: str) -> dict[str, list[dict]]:
    """Per-group summaries and tests per stat and layer from the per-text values: {stat: [compare(...) per layer]}."""
    # across_layer keeps the per-layer values under by_layer (by_token is indexed by token position).
    values = (lambda item, stat: item[stat]) if axis == "within_layer" else (lambda item, stat: item["by_layer"][stat])
    groups = by_group(items)
    return {
        stat: [
            compare({group: [values(item, stat)[layer] for item in rows] for group, rows in groups.items()})
            for layer in range(len(values(items[0], stat)))
        ]
        for stat in stats if stat in dict(STATS)
    }


def summarize_pooled(items: list[dict], stats: list[str]) -> dict[str, dict]:
    """
    across_layer pooled over the layers: per text the mean over tokens of each token's mean over the layer steps
    (by_token), then per-group summaries and tests per stat: {stat: compare(...)}.
    """
    groups = by_group(items)
    pooled = lambda item, stat: float(finite(item["by_token"][stat]).mean()) if len(finite(item["by_token"][stat])) else None
    return {
        stat: compare({group: [pooled(item, stat) for item in rows] for group, rows in groups.items()})
        for stat in stats if stat in dict(STATS)
    }


def summarize_text_stats(items: list[dict]) -> dict[str, dict]:
    """Per-group summaries and tests of the per-text scalars (TEXT_STATS): {stat: compare(...)}."""
    groups = by_group(items)
    out = {}
    for stat, _ in TEXT_STATS:
        # Files written before a stat was added lack it; rerun descriptives to fill it in.
        if stat not in items[0]:
            print(f"Warning: no {stat!r} in the descriptive statistics; rerun src.geometric_metrics.descriptives.")
            continue
        out[stat] = compare({group: [item[stat] for item in rows] for group, rows in groups.items()})
    return out


def format_number(value: float) -> str:
    """Fewer decimals for larger values, so every cell has about three to four significant digits."""
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def format_p(p: float) -> str:
    """The p-value for a math-mode subscript."""
    if not np.isfinite(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def format_cell(summary: dict, bold: bool) -> str:
    """mean$_{std}$, bold if this group has the higher mean."""
    mean = format_number(summary["mean"])
    cell = rf"{mean}$_{{{format_number(summary['std'])}}}$"
    return rf"\textbf{{{mean}}}$_{{{format_number(summary['std'])}}}$" if bold else cell


def render_pair(by_group: dict) -> list[str]:
    """
    Human and machine cells, the higher mean in bold (no bold if the displayed means tie), then the mean difference
    (M − H) with the p-value as subscript, and the AUC.
    """
    human, machine = (float(format_number(by_group[group]["mean"])) for group, _ in GROUPS)
    return [
        format_cell(by_group["human"], human > machine),
        format_cell(by_group["machine"], machine > human),
        ("+" if by_group["diff"] >= 0 else "") + format_number(by_group["diff"]) + rf"$_{{{format_p(by_group['p'])}}}$",
        f"{by_group['auc']:.3f}" if np.isfinite(by_group["auc"]) else "--",
    ]


def layers(stats: dict) -> list[int]:
    """Every LAYER_STEP-th layer from layer 1, plus the last layer of the stat with the most layers."""
    count = max(len(stats[stat]) for stat, _ in STATS if stat in stats)
    selected = list(range(1, count + 1, LAYER_STEP))
    if selected[-1] != count:
        selected.append(count)
    return selected


def render_metrics_table(stats: dict) -> list[str]:
    # Files written before a stat was added lack it; rerun descriptives to fill it in.
    for stat, _ in STATS:
        if stat not in stats:
            print(f"Warning: no {stat!r} in the descriptive statistics; rerun src.geometric_metrics.descriptives.")
    shown = [(stat, name) for stat, name in STATS if stat in stats]
    column_count = 1 + len(PAIR_COLUMNS) * len(shown)
    header = " & ".join(
        [""] + [rf"\multicolumn{{{len(PAIR_COLUMNS)}}}{{c}}{{\textbf{{{name}}}}}" for _, name in shown]
    ) + r" \\"
    rules = " ".join(
        rf"\cmidrule(lr){{{2 + len(PAIR_COLUMNS) * i}-{1 + len(PAIR_COLUMNS) * (i + 1)}}}" for i in range(len(shown))
    )
    subheader = " & ".join(
        [r"\textbf{Layer}"] + [rf"\textbf{{{column}}}" for _ in shown for column in PAIR_COLUMNS]
    ) + r" \\"
    rows = []
    for layer in layers(stats):
        cells = [str(layer)]
        for stat, _ in shown:
            # Stats with fewer layer steps (across_layer curvature) have no value in the last rows.
            cells += render_pair(stats[stat][layer - 1]) if layer <= len(stats[stat]) else ["--"] * len(PAIR_COLUMNS)
        rows.append(" & ".join(cells) + r" \\")
    return [
        r"\begin{tabular}{l" + "c" * (column_count - 1) + "}",
        r"\toprule",
        header,
        rules,
        subheader,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]


def render_length_table(stats: dict, names: tuple = TEXT_STATS, row_header: str = "Per text") -> list[str]:
    """One row per scalar stat (names: (stat, display name)), the pair columns as in the layer tables."""
    rows = [
        " & ".join([name] + render_pair(stats[stat])) + r" \\"
        for stat, name in names if stat in stats
    ]
    return [
        r"\begin{tabular}{l" + "c" * len(PAIR_COLUMNS) + "}",
        r"\toprule",
        " & ".join([rf"\textbf{{{row_header}}}"] + [rf"\textbf{{{column}}}" for column in PAIR_COLUMNS]) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]


def write_table(name: str, model: str, dataset: str, seed: int, table: list[str], states: str | None = None) -> None:
    caption = CAPTIONS[name] + (f"; {STATE_CAPTIONS[states, name]}" if states else "")
    lines = [
        f"% Descriptive statistics, {model}, {dataset}, seed {seed}: mean$_{{std}}$ per group "
        f"(H = human, M = machine), the higher mean in bold; "
        f"Delta = mean(M) - mean(H) with subscript p = two-sided Welch's t-test; "
        f"AUC = P(M > H) of the raw value (Mann-Whitney U / n_H n_M); {caption}.",
        *table,
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{name}_{states}" if states else name
    output_path = OUTPUT_DIR / f"t_metrics_{suffix}_{model}_{dataset}_s{seed}.tex"
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")


def main(model: str, dataset: str, seed: int) -> None:
    for axis in AXES:
        for states in STATES:
            desc = json.loads((DESC_STATS_DIR / f"desc_stats_{axis}_{states}_{model}_{dataset}_s{seed}.json").read_text())
            table = render_metrics_table(summarize_layers(desc["items"], desc["stats"], axis))
            write_table(axis, model, dataset, seed, table, states)
            if axis == "across_layer":
                pooled = render_length_table(summarize_pooled(desc["items"], desc["stats"]), STATS, "Metric")
                write_table("across_layer_pooled", model, dataset, seed, pooled, states)
    # The per-text scalars are the same in every file.
    write_table("len", model, dataset, seed, render_length_table(summarize_text_stats(desc["items"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.model, args.dataset, args.seed)
