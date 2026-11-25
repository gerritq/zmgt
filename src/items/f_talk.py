"""
Figures for the talk (presentation/ai_detection_talk.tex, whose \\graphicspath points at these output folders). All
for Llama-3.1-8B (--model) and the DetectRL-X domains in DATASETS, seed --seed. Unless noted, a panel shows per layer
the machine − human gap of the per-text means (unpaired): one faded line per domain and their average (black, ±95% CI;
the CI combines the domains' standard errors). The grey band marks BAND (the early-to-middle layers).

output/hidden_states/within/ (src.hs_anal.hs_within)
    f_within_talk_A.pdf     within-layer angle (°) | anisotropy (mean pairwise cosine)
    f_within_talk_B.pdf     within-layer angle on centred states (°) | neighbours vs. all pairs: hs_within's
                            within_cos − anisotropy (cosine) where the file has it, else the shuffle test of
                            src.geometric_metrics.desc3 (L = A_shuffled − A_original, degrees)
output/hidden_states/across/ (src.hs_anal.hs_across; ‖z‖ from src.hs_anal.norm_anal)
    f_across_talk_A.pdf     across-layer angle (°), one panel
    f_across_talk_B.pdf     the factors of tan φ = ‖Δ⊥‖ / ((1 + α) ‖z‖): ‖z‖ (% of human) | α | ‖Δ⊥‖ / ‖z‖ (% of human)
output/hidden_states/surprisal/ (src.hs_anal.hs_surprisal)
    f_surprisal_talk.pdf    within | across angle gap (°) per surprisal decile, on SURPRISAL_LAYERS. Per domain, deciles
                            of the pooled token surprisal (both groups); per text, the mean angle of its tokens in each
                            decile (each token's angle averaged over the layers first); then the gap of the text means
    f_harp_talk.pdf         within | across angle gap (°) per W_U subspace (full, semantic, reasoning), each the average
                            over the domains (±95% CI)
output/time_space/ (src.baseline_time.run_time)
    f_time_space_talk.pdf   latency per text (ms, mean ± std) | peak GPU memory (GB), one bar per detector

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.items.f_talk
"""

import json
import os
from argparse import ArgumentParser, Namespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import Config

cfg = Config()
HS_DIR = os.path.join(cfg.base_dir, "output", "hidden_states")
DESC3_DIR = os.path.join(cfg.base_dir, "output", "desc3")
TIME_DIR = os.path.join(cfg.base_dir, "output", "time_space")
DATASETS = {  # name in the file -> label, colour; colours validated as a categorical set (dataviz validator)
    "wiki": ("Wikipedia", "#008300"),
    "academic": ("Academic", "#6250d6"),
    "seo": ("SEO", "#e87ba4"),
    "news": ("News", "#c98500"),
}
SUBSPACES = {"full": ("full", "#008300"), "semantic": ("semantic (95%)", "#6250d6"),
             "reasoning": ("reasoning (5%)", "#e87ba4")}
DETECTORS = {  # run_time model name -> label
    "joint": "JAS (ours)", "irm": "IRM", "fdgpt": "Fast-DetectGPT", "binoculars": "Binoculars",
    "lastde": "Lastde++", "detectllm": "DNA-DetectLLM",
}
INK, MUTED, GRID, BAND_COLOR = "#263238", "#7F8C8D", "#E2E6EA", "#F1F3F5"
ACCENT, BAR = "#1DB954", "#B0BEC5"  # JAS bar (the talk's green) / other detectors
BAND = (4, 15)
SURPRISAL_LAYERS = range(4, 16)  # layers ℓ averaged in the surprisal figure (within: h^ℓ; across: step ℓ−1 → ℓ)
N_BINS = 10
WIDTH, HEIGHT = 6.2, 1.95  # inches, about the size a figure is shown at on a slide


def path_of(kind: str, name: str, args: Namespace) -> str:
    return os.path.join(HS_DIR, kind, f"{kind}_{args.model}_drlXDomainLen_{name}_300_tokens_s{args.seed}.npz")


def gap(values: np.ndarray, labels: np.ndarray, relative: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """values (N, ...) per text -> machine − human difference of means and its standard error (in % of the human mean
    if relative)."""
    h, m = values[labels == 0], values[labels == 1]
    diff = np.nanmean(m, 0) - np.nanmean(h, 0)
    se = np.sqrt(np.nanvar(m, 0, ddof=1) / np.isfinite(m).sum(0) + np.nanvar(h, 0, ddof=1) / np.isfinite(h).sum(0))
    scale = 100 / np.nanmean(h, 0) if relative else 1.0
    return diff * scale, se * np.abs(scale)


def auc(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUC of −values per column (> 0.5: machine has the smaller value); no CI (zeros)."""
    a = np.array([roc_auc_score(labels, -column) for column in values.T])
    return a, np.zeros_like(a)


def style(ax: plt.Axes, title: str, ylabel: str, xlabel: str, band: bool = True) -> None:
    if band:
        ax.axvspan(*BAND, color=BAND_COLOR, zorder=0, linewidth=0)
    ax.set_title(title, fontsize=8.5, color=INK, pad=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def domain_panel(ax: plt.Axes, x: np.ndarray, gaps: dict[str, tuple[np.ndarray, np.ndarray]], title: str,
                 ylabel: str, xlabel: str = "layer (0 = embeddings)", reference: float = 0.0,
                 band: bool = True) -> None:
    """One faded line per domain and their average (black, ±95% CI) against x."""
    style(ax, title, ylabel, xlabel, band)
    ax.axhline(reference, color=MUTED, linewidth=0.8, zorder=1)
    for name, (g, _) in gaps.items():
        label, color = DATASETS[name]
        ax.plot(x, g, color=color, linewidth=0.9, alpha=0.45, label=label, zorder=2)
    mean = np.mean([g for g, _ in gaps.values()], axis=0)
    se = np.sqrt(np.sum([s ** 2 for _, s in gaps.values()], axis=0)) / len(gaps)
    ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=INK, alpha=0.15, linewidth=0, zorder=3)
    ax.plot(x, mean, color=INK, linewidth=1.8, label="average (±95% CI)", zorder=4)
    ax.set_xlim(x[0], x[-1])


def save(fig: plt.Figure, axes, path: str, legend: bool = True) -> None:
    if legend:  # one legend for all panels, every entry once
        handles = {}
        for ax in np.ravel(axes):
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, -0.03), fontsize=6.5, handlelength=1.5, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.2)
    else:
        fig.tight_layout(w_pad=1.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def local_gap(name: str, args: Namespace) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Gap of neighbours vs. all pairs and its unit: from hs_within (within_cos − anisotropy, cosine) if the file has
    it, else desc3's shuffle test (A_shuffled − A_original, degrees)."""
    within = np.load(path_of("within", name, args))
    if "within_cos" in within.files:
        return (*gap(within["within_cos"] - within["anisotropy"], within["labels"]), "cosine")
    path = os.path.join(DESC3_DIR, f"desc3_{args.model}_drlXDomainLen_{name}_300_tokens_s{args.seed}.npz")
    if not os.path.exists(path):
        return None
    desc3 = np.load(path, allow_pickle=True)
    return (*gap(desc3["L"], desc3["label"]), "degrees")  # already degrees


def within_figures(args: Namespace) -> None:
    scores = {"within": {}, "anisotropy": {}, "within_centred": {}, "local": {}}
    units = set()
    for name in DATASETS:
        d = np.load(path_of("within", name, args))
        scores["within"][name] = gap(np.degrees(d["within"]), d["labels"])
        scores["anisotropy"][name] = gap(d["anisotropy"], d["labels"])
        scores["within_centred"][name] = gap(np.degrees(d["within_centred"]), d["labels"])
        local = local_gap(name, args)
        if local is not None:
            scores["local"][name] = local[:2]
            units.add(local[2])
    if len(units) > 1:  # mixed sources: keep only the new hs_within files
        print("mixed sources for the neighbour panel (rerun hs_within everywhere); showing cosine domains only")
        scores["local"] = {n: v for n, v in scores["local"].items() if local_gap(n, args)[2] == "cosine"}
        units = {"cosine"}
    unit = units.pop() if units else "cosine"
    local_title, local_label = (("Neighbour − all-pair cosine", "machine − human (cosine)") if unit == "cosine"
                                else ("Shuffled − original angle", "machine − human (°)"))
    print(f"neighbour panel ({unit}) for: " + ", ".join(scores["local"]))
    x = np.arange(len(scores["within"]["wiki"][0]))
    for suffix, panels in (("A", (("within", "Within-layer angle", "machine − human (°)"),
                                  ("anisotropy", "Anisotropy", "machine − human (mean cosine)"))),
                           ("B", (("within_centred", "Within-layer angle, centred", "machine − human (°)"),
                                  ("local", local_title, local_label)))):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))
        for ax, (key, title, ylabel) in zip(axes, panels):
            domain_panel(ax, x, scores[key], title, ylabel)
        save(fig, axes, os.path.join(HS_DIR, "within", f"f_within_talk_{suffix}.pdf"))


def across_figures(args: Namespace) -> None:
    scores = {"angle": {}, "z": {}, "alpha": {}, "perp_rel": {}}
    for name in DATASETS:
        d = np.load(path_of("across", name, args))
        labels = d["labels"]
        scores["angle"][name] = gap(np.degrees(d["angle"]), labels)
        scores["alpha"][name] = gap(d["alpha"], labels)
        scores["perp_rel"][name] = gap(d["perp_rel"], labels, relative=True)
        norms = np.load(path_of("norms", name, args).replace("norms_", "norms_", 1))
        assert np.array_equal(norms["labels"], labels), f"{name}: norms and across texts differ"
        scores["z"][name] = gap(norms["norm"][:, :-1], labels, relative=True)  # z = h^{ℓ−1} for step ℓ−1 → ℓ
    x = np.arange(1, len(scores["angle"]["wiki"][0]) + 1)  # step ℓ−1 → ℓ at ℓ
    xlabel = "layer ℓ (step ℓ−1 → ℓ)"

    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, HEIGHT))
    domain_panel(ax, x, scores["angle"], "Across-layer angle", "machine − human (°)", xlabel)
    save(fig, ax, os.path.join(HS_DIR, "across", "f_across_talk_A.pdf"))

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT))
    domain_panel(axes[0], x, scores["z"], r"State size $\|z\|$", "machine − human (%)", xlabel)
    domain_panel(axes[1], x, scores["alpha"], r"Growth along $z$: $\alpha$", "machine − human", xlabel)
    domain_panel(axes[2], x, scores["perp_rel"], r"Sideways push $\|\Delta_\perp\| / \|z\|$", "machine − human (%)",
                 xlabel)
    save(fig, axes, os.path.join(HS_DIR, "across", "f_across_talk_B.pdf"))


def surprisal_bins(d, layers: list[int], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Per text, the mean angle (degrees) of its tokens in each surprisal decile of the domain -> (N, N_BINS)."""
    angles = np.degrees(np.nanmean(d[key][:, layers, :], axis=1))  # (N, T), each token averaged over the layers
    s = d["surprisal"]
    edges = np.nanquantile(s, np.linspace(0, 1, N_BINS + 1))
    bins = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, N_BINS - 1)
    valid = np.isfinite(s) & np.isfinite(angles)
    out = np.full((len(s), N_BINS), np.nan)
    for b in range(N_BINS):
        mask = valid & (bins == b)
        n = mask.sum(1)
        out[:, b] = np.where(n > 0, np.where(mask, angles, 0).sum(1) / np.maximum(n, 1), np.nan)
    return out


def surprisal_figure(args: Namespace) -> None:
    scores = {"within": {}, "across": {}}
    within_layers = list(SURPRISAL_LAYERS)  # h^ℓ
    across_layers = [layer - 1 for layer in SURPRISAL_LAYERS]  # index i = step i → i + 1, so step ℓ−1 → ℓ at ℓ − 1
    for name in DATASETS:
        d = np.load(path_of("surprisal", name, args))
        scores["within"][name] = gap(surprisal_bins(d, within_layers, "within"), d["labels"])
        scores["across"][name] = gap(surprisal_bins(d, across_layers, "across"), d["labels"])
    x = np.arange(1, N_BINS + 1)
    xlabel = "surprisal decile (1 = most predictable)"
    layers = f"layers {SURPRISAL_LAYERS[0]}–{SURPRISAL_LAYERS[-1]}"
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))
    domain_panel(axes[0], x, scores["within"], f"Within-layer angle ({layers})", "machine − human (°)", xlabel,
                 band=False)
    domain_panel(axes[1], x, scores["across"], f"Across-layer angle ({layers})", "machine − human (°)", xlabel,
                 band=False)
    for ax in axes:
        ax.set_xticks(x)
    save(fig, axes, os.path.join(HS_DIR, "surprisal", "f_surprisal_talk.pdf"))


def harp_figure(args: Namespace) -> None:
    """Per W_U subspace, the machine − human angle gap (°) per layer, averaged over the domains (±95% CI)."""
    files = {name: np.load(path_of("surprisal", name, args).replace("surprisal_", "harp_", 1)) for name in DATASETS}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))
    for ax, kind, title in ((axes[0], "within", "Within-layer angle"), (axes[1], "across", "Across-layer angle")):
        first = 0 if kind == "within" else 1
        style(ax, title, "machine − human (°)",
              "layer (0 = embeddings)" if kind == "within" else "layer ℓ (step ℓ−1 → ℓ)")
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
        for space, (label, color) in SUBSPACES.items():
            gaps = [gap(np.degrees(f[f"{kind}_{space}"].astype(float)), f["labels"]) for f in files.values()]
            mean = np.mean([g for g, _ in gaps], axis=0)
            se = np.sqrt(np.sum([e ** 2 for _, e in gaps], axis=0)) / len(gaps)
            x = np.arange(first, first + len(mean))
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.18, linewidth=0, zorder=2)
            ax.plot(x, mean, color=color, linewidth=1.6, label=label, zorder=3)
            ax.set_xlim(x[0], x[-1])
    save(fig, axes, os.path.join(HS_DIR, "surprisal", "f_harp_talk.pdf"))


def time_space_figure(args: Namespace) -> None:
    rows = []
    for model, label in DETECTORS.items():
        path = os.path.join(TIME_DIR, f"{model}_drlXDomainLen_wiki_300_tokens_s{args.seed}.json")
        if os.path.exists(path):
            s = json.load(open(path))["summary"]
            rows.append((label, s["latency_ms"]["mean"], s["latency_ms"]["std"], s["peak_memory_mb"]["mean"] / 1024))
    rows.sort(key=lambda r: r[1], reverse=True)  # fastest at the top of a horizontal bar chart
    labels = [r[0] for r in rows]
    colors = [ACCENT if label.startswith("JAS") else BAR for label in labels]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True)
    for ax, values, errors, title, unit in (
            (axes[0], [r[1] for r in rows], [r[2] for r in rows], "Latency per text", "ms"),
            (axes[1], [r[3] for r in rows], None, "Peak GPU memory", "GB")):
        ax.barh(y, values, xerr=errors, color=colors, height=0.6, error_kw={"elinewidth": 0.7, "ecolor": INK})
        for yi, v, e in zip(y, values, errors or [0] * len(values)):
            ax.text(v + e + max(values) * 0.02, yi, f"{v:.0f} {unit}" if unit == "ms" else f"{v:.1f} {unit}",
                    va="center", fontsize=6.5, color=INK)
        ax.set_title(title, fontsize=8.5, color=INK, pad=3)
        ax.set_xlim(0, max(values) * 1.22)
        ax.set_xlabel(unit)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(y, labels)
    save(fig, axes, os.path.join(TIME_DIR, "f_time_space_talk.pdf"), legend=False)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7, "pdf.fonttype": 42,
                         "axes.edgecolor": "#B0BEC5", "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    within_figures(args)
    across_figures(args)
    surprisal_figure(args)
    harp_figure(args)
    time_space_figure(args)


if __name__ == "__main__":
    main()
