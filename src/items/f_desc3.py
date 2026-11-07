"""
Figures for the desc3 outputs (output/desc3/*.npz, from src.geometric_metrics.desc3).

Every figure is made once per model and domain ({model}_{dataset}_s{seed}) and once per model averaged over its domains
({model}_domain_avg_s{seed}). The average gives every domain the same weight: each statistic (class mean, Cohen's d,
class gap, binned mean) is first computed within a domain and then averaged over domains. Bands of the average:
analytic CIs are combined as half = sqrt(Σ half_i²) / K over the K domains; bootstrap CIs resample texts within each
domain and class and average the resampled statistic over domains.

f_desc3_angles_*.pdf, 1 x 2 panels, full width:
    (a) within-layer angle  ∠(h_{t-1}^ℓ, h_t^ℓ)     (per-text mean: within_angle_mean)
    (b) across-layer angle  ∠(h_t^{ℓ-1}, h_t^ℓ)     (per-text mean: across_angle_mean, undefined at layer 0)
    x:  relative depth ℓ / (L - 1), L = number of layers kept by desc3 (embedding .. last block before final norm).
    y1 (left):  mean over texts of the per-text mean angle (degrees), one line per group (human, machine),
                band = 95% CI of the mean (1.96 · SE).
    y2 (right): Cohen's d (human − machine, pooled SD) of the per-text mean angle, band = 95% CI
                (normal approximation, SE = sqrt((n1 + n2) / (n1 n2) + d² / (2 (n1 + n2)))).

f_desc3_within_*.pdf, one panel, full width: where the class gap in the within-layer angle comes from, using the exact
per-text split  within_angle = allpairs_angle + local_part.
    x:     relative depth, as above.
    bars:  class gap (human − machine, in degrees) of each part, stacked as diverging bars: positive parts are stacked
           upwards from 0, negative parts downwards from 0.
               anisotropy part: gap(allpairs_angle)   (mean angle over all pairs of kept tokens)
               local part:      gap(local_part)       (within_angle − allpairs_angle)
    line:  total gap gap(within_angle) (= sum of the two bars), black with markers, band = 95% bootstrap CI (texts
           resampled within each class).
    Per layer, only texts where all three scores are finite are used, so the bars add up to the line exactly.

f_desc3_across_*.pdf, one panel, full width: the same for the across-layer angle θ, using the per-text split
log tan θ = T1 + T2 + T3 + T4  (He et al.; h = h_t^{ℓ-1}, Δ = h_t^ℓ − h_t^{ℓ-1}):
    T1 = log ‖Δ‖ (update size), T2 = log(q / ‖Δ‖) (sideways share, q = ‖Δ_⊥‖), T3 = −log ‖h‖ (state size),
    T4 = −log(1 + α) (forward growth, α = Δ·h / ‖h‖²).
    bars:  class gap (human − machine, log units) of each term, diverging and stacked as above.
    line:  gap(log_tan), black with markers, band = 95% bootstrap CI. Layer 0 has no previous layer, so no bar.
Both split figures stop with an error if the bars do not add up to the line.

f_desc3_surprisal_*_l{layer}.pdf, 1 x 2 panels at one layer ℓ*:
    (a) across-layer angle ∠(h_t^{ℓ*-1}, h_t^{ℓ*}), (b) within-layer angle ∠(h_{t-1}^{ℓ*}, h_t^{ℓ*}).
    Tokens: (a) uses across pairs (t kept at ℓ*-1 and ℓ*, not flagged); (b) uses within pairs (t-1 and t kept).
    x:  surprisal decile, −log p(x_t | x_<t) (t ≥ 1). Decile edges come from the pooled surprisal of all tokens of both
        classes and all domains of the model, so every domain and both classes share the same bins.
    y:  two-stage mean per bin: first the mean over a text's tokens in that bin, then the mean over texts (texts with
        no token in the bin are left out of that bin).
    lines: human and machine, band = 95% bootstrap CI (texts resampled within each class).
    ℓ*: --layer, or by default the layer closest to relative depth --depth (default 0.3), i.e. round(depth · (L - 1)),
        at least 1 (one layer per model). The title gives ℓ*, L - 1 and the relative depth ℓ* / (L - 1).
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.config import Config
from src.items.plot_style import configure_plot_style

cfg = Config()

DESC3_DIR = Path(cfg.base_dir) / "output" / "desc3"
OUTPUT_DIR = Path(cfg.item_output_dir) / "desc3"

MODEL_NAMES = {"l8b": "Llama-3.1-8B", "m8b": "Ministral-8B", "q8b": "Qwen3-8B"}
GROUPS = ((0, "Human", "#0072B2", "-"), (1, "Machine", "#D55E00", "-"))  # (label, name, color, line style)
D_COLOR = "#263238"
GRID = "#E2E6EA"
PANELS = (
    ("within_angle_mean", "(a) Within-layer angle", r"$\angle(h_{t-1}^{\ell},\, h_t^{\ell})$"),
    ("across_angle_mean", "(b) Across-layer angle", r"$\angle(h_t^{\ell-1},\, h_t^{\ell})$"),
)
WITHIN_PARTS = (
    ("allpairs_angle", "Anisotropy part: gap(all-pairs angle)", "#009E73"),
    ("local_part", "Local part: gap(local part)", "#CC79A7"),
)  # (key, legend label, color)
ACROSS_PARTS = (
    ("T1_mean", r"$T_1$: update size $\log\Vert\Delta\Vert$", "#E69F00"),
    ("T2_mean", r"$T_2$: sideways share $\log(q / \Vert\Delta\Vert)$", "#56B4E9"),
    ("T3_mean", r"$T_3$: state size $-\log\Vert h\Vert$", "#009E73"),
    ("T4_mean", r"$T_4$: forward growth $-\log(1 + \alpha)$", "#CC79A7"),
)  # h = h_t^{l-1}, Δ = h_t^l − h_t^{l-1}, q = ‖Δ_⊥‖, α = Δ·h / ‖h‖²
BOOTSTRAPS = 2000
SEED = 0
N_BINS = 10
TOKEN_PANELS = (
    ("across_angle", "(a) Across-layer angle", "across"),
    ("within_angle", "(b) Within-layer angle", "within"),
)  # (token score, title, pair type); both in degrees
LAYER_KEYS = ("within_angle_mean", "across_angle_mean", "allpairs_angle", "local_part",
              "T1_mean", "T2_mean", "T3_mean", "T4_mean", "log_tan_mean")

Subset = dict[str, np.ndarray]


# ---------------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------------

def load_small(path: Path) -> Subset:
    """Per-text and per-layer arrays only (the token arrays are large and loaded later, at one layer)."""
    with np.load(path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in ("label", "meta", "logprob", *LAYER_KEYS)}
    data["meta_dict"] = json.loads(str(data["meta"]))
    data["path"] = path
    return data


def load_layer(data: Subset, layer: int) -> None:
    """Add the token angles and pair masks at one layer: (texts, max T) each."""
    with np.load(data["path"], allow_pickle=False) as npz:
        keep = npz["keep"]
        flag = npz["flag_large_angle"][:, layer]
        across = npz["across_angle"][:, layer]
        within = npz["within_angle"][:, layer]
    data["tok_across_angle"] = np.degrees(across.astype(float))
    data["tok_within_angle"] = np.degrees(within.astype(float))
    data["pair_across"] = keep[:, layer] & keep[:, layer - 1] & ~flag
    data["pair_within"] = keep[:, layer] & np.pad(keep[:, layer, :-1], ((0, 0), (1, 0)))


def domain(data: Subset) -> str:
    """'drlXDomainLen_wiki_300_tokens' -> 'wiki'."""
    parts = data["meta_dict"].get("dataset", "").split("_")
    return parts[1] if len(parts) > 1 else data["meta_dict"].get("dataset", "")


# ---------------------------------------------------------------------------------------------------------------------
# statistics (each computed within a subset, then averaged over subsets)
# ---------------------------------------------------------------------------------------------------------------------

def nanmean0(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():  # all-NaN columns (layer 0 of across-layer scores, empty bins) give NaN on purpose
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=0)


def boot_average(samples: list[tuple[np.ndarray, ...]], stat: Callable[..., np.ndarray]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    samples: per subset, a tuple of arrays whose rows are texts (e.g. (human, machine)). Returns the average over subsets
    of stat(*arrays) and its 95% bootstrap CI (rows of every array resampled independently, within each subset).
    """
    estimate = np.mean([stat(*arrays) for arrays in samples], axis=0)
    rng = np.random.default_rng(SEED)
    boots = np.stack([
        np.mean([stat(*(a[rng.integers(0, len(a), len(a))] for a in arrays)) for arrays in samples], axis=0)
        for _ in range(BOOTSTRAPS)
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return estimate, low, high


def gap(human: np.ndarray, machine: np.ndarray) -> np.ndarray:
    return nanmean0(human) - nanmean0(machine)


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(texts, L) -> mean over texts and half-width of its 95% CI, per layer."""
    n = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return nanmean0(values), 1.96 * se


def cohens_d_ci(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(texts, L) -> Cohen's d (human − machine, pooled SD) and half-width of its 95% CI, per layer."""
    n_layers = values.shape[1]
    d, half = np.full(n_layers, np.nan), np.full(n_layers, np.nan)
    for layer in range(n_layers):
        column = values[:, layer]
        human = column[(labels == 0) & np.isfinite(column)]
        machine = column[(labels == 1) & np.isfinite(column)]
        n1, n2 = len(human), len(machine)
        if n1 < 2 or n2 < 2:
            continue
        pooled = np.sqrt(((n1 - 1) * human.var(ddof=1) + (n2 - 1) * machine.var(ddof=1)) / (n1 + n2 - 2))
        if pooled <= 0:
            continue
        d[layer] = (human.mean() - machine.mean()) / pooled
        half[layer] = 1.96 * np.sqrt((n1 + n2) / (n1 * n2) + d[layer] ** 2 / (2 * (n1 + n2)))
    return d, half


def average_ci(results: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average of K independent estimates with CI half-widths: mean, sqrt(Σ half²) / K."""
    values = np.stack([r[0] for r in results])
    halves = np.stack([r[1] for r in results])
    return values.mean(axis=0), np.sqrt((halves ** 2).sum(axis=0)) / len(results)


# ---------------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------------

def save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path.with_suffix('.pdf')}")


def bottom_legend(fig: plt.Figure, handles: list, names: list | None = None) -> None:
    n = len(handles)
    ncol = n if n <= 4 else (n + 1) // 2
    rows = -(-n // ncol)
    fig.tight_layout(rect=(0, 0.04 + 0.05 * rows, 1, 1))
    if names is None:
        fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0))
    else:
        fig.legend(handles, names, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0))


def plot_angles(subsets: list[Subset], title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    d_axes = []
    for ax, (key, name, formula) in zip(axes, PANELS):
        values = [np.degrees(s[key].astype(float)) for s in subsets]  # (texts, L) per subset
        depth = np.arange(values[0].shape[1]) / (values[0].shape[1] - 1)
        for label, group, color, line in GROUPS:
            mean, half = average_ci([mean_ci(v[s["label"] == label]) for v, s in zip(values, subsets)])
            ax.plot(depth, mean, color=color, linestyle=line, linewidth=2)
            ax.fill_between(depth, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
        ax.set_title(f"{name}: {formula}")
        ax.set_xlabel("Relative depth")
        ax.set_ylabel("Mean angle (°)")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1)

        ax_d = ax.twinx()
        d, half = average_ci([cohens_d_ci(v, s["label"]) for v, s in zip(values, subsets)])
        ax_d.axhline(0, color="#7F8C8D", linewidth=0.8, linestyle=":")
        ax_d.plot(depth, d, color=D_COLOR, linestyle="-.", linewidth=1.6)
        ax_d.fill_between(depth, d - half, d + half, color=D_COLOR, alpha=0.10, linewidth=0)
        ax_d.set_ylabel("Cohen's d (human − machine)")
        d_axes.append(ax_d)

    lo = min(ax.get_ylim()[0] for ax in d_axes)
    hi = max(ax.get_ylim()[1] for ax in d_axes)
    for ax in d_axes:
        ax.set_ylim(lo, hi)  # same d scale in both panels

    handles = [Line2D([], [], color=color, linestyle=line, linewidth=2, label=f"{group} (mean angle)")
               for _, group, color, line in GROUPS]
    handles += [Line2D([], [], color=D_COLOR, linestyle="-.", linewidth=1.6, label="Cohen's d (right axis)"),
                Patch(color="#90A4AE", alpha=0.4, label="95% CI")]
    fig.suptitle(title)
    bottom_legend(fig, handles)
    save(fig, out_path)


def plot_gap_split(subsets: list[Subset], total_key: str, parts: tuple, scale: float, tol: float,
                   total_name: str, ylabel: str, title: str, out_path: Path) -> None:
    """
    Stacked diverging bars of the class gap of each part (key, legend label, color) and the class gap of total_key as a
    black line with a 95% bootstrap band. Stops if the bars do not add up to the line (max |diff| > tol).
    """
    totals, values = [], {key: [] for key, _, _ in parts}
    for s in subsets:
        total = scale * s[total_key].astype(float)
        part_values = {key: scale * s[key].astype(float) for key, _, _ in parts}
        valid = np.isfinite(total) & np.all([np.isfinite(v) for v in part_values.values()], axis=0)
        totals.append(np.where(valid, total, np.nan))
        for key, v in part_values.items():
            values[key].append(np.where(valid, v, np.nan))

    def split(arrays: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(a[s["label"] == 0], a[s["label"] == 1]) for a, s in zip(arrays, subsets)]

    gaps = {key: np.mean([gap(*hm) for hm in split(v)], axis=0) for key, v in values.items()}
    gap_total, low, high = boot_average(split(totals), gap)
    check = np.nanmax(np.abs(gap_total - sum(gaps.values())))
    if not check <= tol:
        raise ValueError(f"{out_path.name}: bars do not add up to the total gap (max |diff| = {check:.2e} > {tol})")
    print(f"{out_path.name}: bars add up to the total gap (max |diff| = {check:.1e})")

    depth = np.arange(len(gap_total)) / (len(gap_total) - 1)
    width = 0.8 / (len(gap_total) - 1)
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.axhline(0, color="#7F8C8D", linewidth=0.8)
    up, down = np.zeros_like(depth), np.zeros_like(depth)
    for key, name, color in parts:
        g = np.nan_to_num(gaps[key])
        pos, neg = np.clip(g, 0, None), np.clip(g, None, 0)
        ax.bar(depth, pos, width=width, bottom=up, color=color, alpha=0.85, linewidth=0, label=name)
        ax.bar(depth, neg, width=width, bottom=down, color=color, alpha=0.85, linewidth=0)
        up, down = up + pos, down + neg
    ax.fill_between(depth, low, high, color="black", alpha=0.15, linewidth=0, label="95% CI (total)")
    ax.plot(depth, gap_total, color="black", linewidth=1.6, marker="o", markersize=4, label=total_name)
    ax.use_sticky_edges = False  # zero-height bars would otherwise pin the y limit to the tip of the longest bar
    ax.autoscale_view()
    ax.set_xlim(-width, 1 + width)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    handles, names = ax.get_legend_handles_labels()
    order = [names.index(n) for n in [name for _, name, _ in parts] + [total_name, "95% CI (total)"]]
    fig.suptitle(title)
    bottom_legend(fig, [handles[i] for i in order], [names[i] for i in order])
    save(fig, out_path)


def n_layers(subsets: list[Subset]) -> int:
    """L, the number of layers kept by desc3 (embedding .. last block)."""
    sizes = {s["within_angle_mean"].shape[1] for s in subsets}
    if len(sizes) != 1:
        raise ValueError(f"subsets of one model have different numbers of layers: {sorted(sizes)}")
    return sizes.pop()


def depth_layer(subsets: list[Subset], depth: float) -> int:
    """Layer (>= 1) closest to relative depth `depth`: round(depth * (L - 1))."""
    last = n_layers(subsets) - 1
    return min(max(int(depth * last + 0.5), 1), last)


def surprisal(data: Subset) -> np.ndarray:
    """(texts, max T) −log p(x_t | x_<t); NaN at t = 0 and on padding."""
    return -data["logprob"].astype(float)


def surprisal_edges(subsets: list[Subset]) -> np.ndarray:
    """Decile edges of the pooled surprisal of all tokens of all given subsets (both classes)."""
    pooled = np.concatenate([surprisal(s).ravel() for s in subsets])
    return np.quantile(pooled[np.isfinite(pooled)], np.linspace(0, 1, N_BINS + 1))


def per_text_bin_means(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """values (texts, T) with NaN where unused, bins (texts, T) in 0..N_BINS-1 or -1 -> (texts, N_BINS) means."""
    used = np.isfinite(values) & (bins >= 0)
    rows, cols = np.nonzero(used)
    sums = np.zeros((values.shape[0], N_BINS))
    counts = np.zeros((values.shape[0], N_BINS))
    np.add.at(sums, (rows, bins[rows, cols]), values[rows, cols])
    np.add.at(counts, (rows, bins[rows, cols]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def plot_surprisal_bins(subsets: list[Subset], layer: int, edges: np.ndarray, title: str, out_path: Path) -> None:
    x = np.arange(1, N_BINS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (key, name, pair) in zip(axes, TOKEN_PANELS):
        per_text = []
        for s in subsets:
            sur = surprisal(s)
            bins = np.where(np.isfinite(sur), np.clip(np.searchsorted(edges, sur, side="right") - 1, 0, N_BINS - 1), -1)
            values = np.where(s[f"pair_{pair}"], s[f"tok_{key}"], np.nan)
            per_text.append((per_text_bin_means(values, bins), s["label"]))
        for label, group, color, line in GROUPS:
            mean, low, high = boot_average([(p[lab == label],) for p, lab in per_text], nanmean0)
            ax.plot(x, mean, color=color, linestyle=line, linewidth=2, marker="o", markersize=4, label=group)
            ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax.set_title(name)
        ax.set_xlabel("Surprisal decile")
        ax.set_ylabel("Mean angle (°)")
        ax.set_xticks(x)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    handles = [Line2D([], [], color=color, linestyle=line, linewidth=2, marker="o", markersize=4, label=group)
               for _, group, color, line in GROUPS]
    handles.append(Patch(color="#90A4AE", alpha=0.4, label="95% bootstrap CI"))
    last = n_layers(subsets) - 1
    fig.suptitle(f"{title}: by surprisal, layer {layer} of 0–{last} (relative depth {layer / last:.2f})")
    bottom_legend(fig, handles)
    save(fig, out_path)


def all_figures(subsets: list[Subset], layer: int, edges: np.ndarray, title: str, stem: str) -> None:
    plot_angles(subsets, title, OUTPUT_DIR / f"f_desc3_angles_{stem}")
    plot_gap_split(subsets, "within_angle_mean", WITHIN_PARTS, 180 / np.pi, 1e-3,
                   "Total gap: gap(within-layer angle)", "Human − machine (°)",
                   f"{title}: within-layer angle = all-pairs angle + local part",
                   OUTPUT_DIR / f"f_desc3_within_{stem}")
    plot_gap_split(subsets, "log_tan_mean", ACROSS_PARTS, 1.0, 1e-4,
                   r"Total gap: gap($\log\tan\theta$)", "Human − machine (log units)",
                   f"{title}: " + r"$\log\tan\theta = T_1 + T_2 + T_3 + T_4$ (across layers)",
                   OUTPUT_DIR / f"f_desc3_across_{stem}")
    plot_surprisal_bins(subsets, layer, edges, title, OUTPUT_DIR / f"f_desc3_surprisal_{stem}_l{layer}")


def main() -> None:
    args = parse_args()
    configure_plot_style()
    paths = sorted(DESC3_DIR.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files match {DESC3_DIR / args.pattern}")
    by_model: dict[str, list[Subset]] = {}
    for path in paths:
        data = load_small(path)
        by_model.setdefault(data["meta_dict"].get("model", ""), []).append(data)

    for model, subsets in by_model.items():
        layer = args.layer if args.layer is not None else depth_layer(subsets, args.depth)
        edges = surprisal_edges(subsets)
        print(f"{model}: {len(subsets)} domains, layer {layer}, surprisal decile edges {np.round(edges, 2).tolist()}")
        for s in subsets:
            load_layer(s, layer)
        name = MODEL_NAMES.get(model, model)
        for s in subsets:
            title = f"{name}, {s['meta_dict'].get('dataset', '')} (n = {len(s['label'])})"
            all_figures([s], layer, edges, title, s["path"].stem.removeprefix("desc3_"))
        if len(subsets) > 1:
            seeds = sorted({str(s["meta_dict"].get("seed", "")) for s in subsets})
            title = f"{name}, mean over {len(subsets)} domains ({', '.join(domain(s) for s in subsets)})"
            all_figures(subsets, layer, edges, title, f"{model}_domain_avg_s{'-'.join(seeds)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="desc3_*.npz", help="glob in output/desc3")
    parser.add_argument("--layer", type=int, default=None,
                        help="layer for the surprisal figure (overrides --depth)")
    parser.add_argument("--depth", type=float, default=0.3,
                        help="relative depth for the surprisal figure; the layer is round(depth * (L - 1)), per model")
    return parser.parse_args()


if __name__ == "__main__":
    main()
