"""
Summary figure of the steering experiments (output/steering/alltoken_random_*.npz, see src.steering.steer, or with
--kind prefix output/steering/prefix_random_*.npz, see src.steering.prefix, where the angles are on the suffix only):
immediate effect of steering one layer later, against random directions; by default with the steering vector
projected out (--variant proj), or on the full states (--variant full).

Panels: columns = metric (adjacent-token angle, cross-layer angle, mean pairwise cosine), rows = push (steer at
layer l, measure at layer l + 1, for l in PUSHES). y = domain-generator pair (latest run of each).
Per pair and class:
  dot + bar   mean Δ over the test texts of the class, real direction, with 95% CI over texts
  grey bar    range (min - max) of the same mean Δ over the random directions (same variant)
Δ is signed so that > 0 = in the predicted direction: machine text pushed towards human gets larger angles and lower
pairwise cosine; human text pushed towards machine gets smaller angles and higher pairwise cosine. With --raw, Δ is
plotted as is (steered − baseline), and the predicted side differs by class and metric.
Strength c = STRENGTH (in units of the mean token norm). Options: --kind, --control (random | data_shuffle; the grey
bar is then the range over that control's vectors), --variant, --pushes, --domains / --exclude, --overall (extra row
with the texts of all pairs pooled; control vector k pooled across files), --raw, --name. Output: output/items/<name>.pdf (default f_steering_summary)
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
OUTPUT_DIR = Path(cfg.item_output_dir)

PUSHES = (7, 8, 9)
STRENGTH = 0.05
METRICS = (("local_angle", "Adjacent-token angle", "Δ angle (°)"),
           ("cross_angle", "Cross-layer angle", "Δ angle (°)"),
           ("pairwise_cos", "Mean pairwise cosine", "Δ cosine"))
# label 1 = machine (pushed towards human), 0 = human (pushed towards machine); sign: + = predicted direction
CLASSES = ((1, "Machine → human", "#D55E00", -0.18), (0, "Human → machine", "#0072B2", 0.18))
SIGN = {("local_angle", 1): 1, ("cross_angle", 1): 1, ("pairwise_cos", 1): -1,
        ("local_angle", 0): -1, ("cross_angle", 0): -1, ("pairwise_cos", 0): 1}
DOMAINS = {"seo": "SEO"}
GENERATORS = {"gpt_4o": "GPT-4o", "gemini_2_5_flash": "Gemini", "deepseek_v3": "DeepSeek", "qwen_max": "Qwen"}


def pair_label(dataset: str) -> str:
    """drlXpaired_<domain>_<generator>_256_tokens -> 'Domain\\nGenerator'."""
    rest = dataset.split("drlXpaired_", 1)[1].rsplit("_256_tokens", 1)[0]
    domain, gen = rest.split("_", 1)
    return f"{DOMAINS.get(domain, domain.capitalize())} · {GENERATORS.get(gen, gen)}"


CONTROL_TAG = {"random": "rand", "data_shuffle": "shuf"}
CONTROL_LABEL = {"random": "Random directions (range of 20)", "data_shuffle": "Label-shuffled directions (range of 20)"}
OVERALL = "All pairs (pooled)"


def collect(path: Path, variant: str, pushes: tuple[int, ...], control: str) -> dict:
    """{(metric, class, push): (d (n,), q (n, K))} raw Δ per text for one file: real direction and control vectors."""
    f = np.load(path)
    labels, metrics = f["labels"], list(f["metrics"])
    j = list(f["strengths"]).index(STRENGTH)
    # read each (compressed) array once; npz decompresses on every access
    suffix = "_proj" if variant == "proj" else ""
    real = f[f"delta{suffix}"][:, :, 0, j]                                              # (N, M, P, L)
    ctrl = f[f"delta_{CONTROL_TAG[control]}{suffix}"][:, :, :, j]                       # (N, M, K, P, L)
    out = {}
    for m, *_ in METRICS:
        mi = metrics.index(m)
        for lab, *_ in CLASSES:
            for l in pushes:                                                              # measured at l + 1
                out[(m, lab, l)] = (real[labels == lab, mi, l - 1, l], ctrl[labels == lab, mi, :, l - 1, l])
    return out


def pool(parts: list[dict]) -> dict:
    """Texts of all files together; control vector k pooled with control vector k of the other files."""
    return {key: (np.concatenate([p[key][0] for p in parts]), np.concatenate([p[key][1] for p in parts]))
            for key in parts[0]}


def stats(raw_values: dict, raw: bool) -> dict:
    """{(metric, class, push): (mean, 95% CI, control min, control max)}; signed (+ = predicted) unless raw."""
    out = {}
    for (m, lab, l), (d, q) in raw_values.items():
        sg = 1 if raw else SIGN[(m, lab)]
        d, cm = d * sg, np.nanmean(q, axis=0) * sg                                       # (n,), (K,)
        ci = 1.96 * np.nanstd(d, ddof=1) / np.sqrt(np.sum(~np.isnan(d)))
        out[(m, lab, l)] = (np.nanmean(d), ci, cm.min(), cm.max())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("alltoken", "prefix"), default="alltoken",
                        help="alltoken: every token steered; prefix: prefix steered, suffix measured")
    parser.add_argument("--control", choices=("random", "data_shuffle"), default="random")
    parser.add_argument("--exclude", nargs="*", default=[], help="domains to leave out, e.g. news")
    parser.add_argument("--domains", nargs="*", default=None, help="only these domains, e.g. wiki academic")
    parser.add_argument("--variant", choices=("proj", "full"), default="proj",
                        help="proj: steering vector projected out; full: full states")
    parser.add_argument("--pushes", nargs="*", type=int, default=list(PUSHES), help="steering layers l (measure l+1)")
    parser.add_argument("--overall", action="store_true", help="add a row with all pairs pooled")
    parser.add_argument("--name", default="f_steering_summary", help="output file name (without extension)")
    parser.add_argument("--raw", action="store_true", help="plot raw Δ (steered − baseline), not signed to the prediction")
    args = parser.parse_args()
    pushes = tuple(args.pushes)
    configure_plot_style()
    files = [p for p in latest_runs() if p.name.startswith(f"{args.kind}_{args.control}_")
             and not any(f"_{d}_" in p.name for d in args.exclude)
             and (args.domains is None or any(f"drlXpaired_{d}_" in p.name for d in args.domains))]
    names = [pair_label(json.loads(str(np.load(p)["meta"]))["dataset"]) for p in files]
    order = np.argsort(names)[::-1]                       # alphabetical from the top
    files, names = [files[o] for o in order], [names[o] for o in order]
    parts = [collect(p, args.variant, pushes, args.control) for p in files]
    res = [stats(p, args.raw) for p in parts]
    if args.overall:
        res, names = res + [stats(pool(parts), args.raw)], names + [OVERALL]
    y = np.arange(len(names))
    ctrl_label = CONTROL_LABEL[args.control]

    fig, axes = plt.subplots(len(pushes), len(METRICS),
                             figsize=(4.6 * len(METRICS), max(0.36 * len(names), 2.4) * len(pushes) + 2.2),
                             sharey=True, sharex="col", squeeze=False)
    for i, l in enumerate(pushes):
        for k, (m, title, xlabel) in enumerate(METRICS):
            ax = axes[i, k]
            ax.axvline(0, color="#8a8985", lw=1, zorder=1)
            if args.overall:
                ax.axhspan(len(names) - 1.5, len(names) - 0.5, color="#f1f0ec", zorder=0)
            for lab, name, color, off in CLASSES:
                vals = np.array([r[(m, lab, l)] for r in res])                         # (rows, 4)
                ax.barh(y + off, vals[:, 3] - vals[:, 2], left=vals[:, 2], height=0.3, color="#d9d8d3",
                        zorder=2, label=ctrl_label if lab == 1 else None)
                ax.errorbar(vals[:, 0], y + off, xerr=vals[:, 1], fmt="o", ms=5, color=color, elinewidth=1.4,
                            capsize=0, zorder=3, label=name)
            ax.grid(axis="x", alpha=0.3, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(-0.6, len(names) - 0.4)
            if i == 0:
                ax.set_title(title)
            if i == len(pushes) - 1:
                ax.set_xlabel(xlabel)
            if k == 0:
                ax.set_yticks(y, names, fontsize=12)
                for t in ax.get_yticklabels():
                    if t.get_text() == OVERALL:
                        t.set_fontweight("bold")
                ax.set_ylabel(f"steer L{l} → L{l + 1}", fontsize=15, labelpad=10)
            ax.tick_params(axis="y", length=0)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    keep = [labels_.index(n) for n in ("Machine → human", "Human → machine", ctrl_label)]
    fig.legend([handles[o] for o in keep], [labels_[o] for o in keep], loc="lower center", ncol=3, frameon=False)
    states = "steering vector projected out" if args.variant == "proj" else "full states"
    kind = "All-token steering" if args.kind == "alltoken" else "Prefix steering (suffix measured)"
    fig.suptitle(f"{kind}, immediate effect one layer later (c = {STRENGTH}, {states})\n"
                 + ("raw Δ = steered − baseline. Predicted: machine → human angles ↑, cosine ↓; "
                    "human → machine angles ↓, cosine ↑" if args.raw else "Δ > 0 = predicted direction")
                 + "; dot = mean ± 95% CI over texts", fontsize=14)
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.6 / h, 1, 1 - 0.9 / h))
    out = OUTPUT_DIR / f"{args.name}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
