"""
Do hidden-state norms, angles and anisotropy differ between human and machine texts, per layer? Runs --model on every
text of --dataset (split --split) as src.inference.Inference does (no special tokens) and keeps only per-text summaries,
so no hidden states are saved. Needs a GPU. Writes three PDFs to output/hidden_states/results/, named
hs_{norm,within,across}_{model}_{dataset}_s{seed}.pdf.

Texts are not paired. Per layer, the left y-axis shows, for the norms, the unpaired difference mean(machine) −
mean(human) over the per-text values, and for the within and across plots the mean of each group; all ±95% CI (normal
approximation). The right y-axis shows an AUC with machine = 1.

1. Norms (layer 0 = embeddings, ℓ = transformer layer ℓ). Per text and layer: mean over tokens of ||h_t||_2.
   Left: all tokens. Right: only tokens whose norm is at most OUTLIER_FACTOR × the median token norm of that text and
   layer (removes outliers such as the attention sink at token 0). AUC of the norm (> 0.5: machine has the larger norm).
2. Within layer (transformer layers 1..L), one column:
   row 1: θ = mean_t ∠(h_{t−1}^ℓ, h_t^ℓ), consecutive tokens, raw states
   row 2: anisotropy = mean cosine over all token pairs t ≠ s, raw states
   row 3: θ on states centred across tokens per text and layer (h_t − mean_t h_t)
3. Across layers (transformer layers 1..L, i.e. transitions ℓ − 1 → ℓ), two columns:
   row 1: φ = mean_t ∠(h_t^{ℓ−1}, h_t^ℓ), same token, raw states (layer 1 against the embeddings) | φ on states
          centred across tokens per text and layer
   row 2: the update z^ℓ = z + Δ, z = h_t^{ℓ−1}, split along z (see update_split): α = ⟨Δ, z⟩ / ‖z‖², Δ∥ = α z,
          Δ⊥ = Δ − Δ∥. Plotted, each per token then averaged over tokens, group means with Cohen's d (no AUC):
          ‖Δ∥‖ / ‖Δ⊥‖ | α (signed: > 0 z grows, < 0 shrinks) | ‖Δ⊥‖ / ‖z‖ (how far z is turned).
AUC for angles is reversed: AUC of −angle (> 0.5: machine has the smaller angle). AUC for anisotropy is of +cosine
(> 0.5: machine has the larger cosine), so > 0.5 means "machine more aligned" in every panel of 2 and 3.
All tokens are used for angles and anisotropy. Angles are in radians, as in src.zero.
4. W_U subspaces (as src.ideas.harp_idea), written to output/harp/. Right singular vectors of the unembedding
   W_U = U Σ Vᵀ, sorted by σ: semantic = the top k = SEMANTIC_FRACTION · d (0.95 d), reasoning = the remaining
   d − k (smallest σ). The same basis is used at every layer ℓ = 0..L. States are projected through the final norm W_U
   reads, p_t^ℓ = V_Sᵀ Norm(h_t^ℓ) (the last layer is already normed by HF). Per subspace, at layers 1..L:
       within = mean_t ∠(p_{t−1}^ℓ, p_t^ℓ),   across = mean_t ∠(p_t^{ℓ−1}, p_t^ℓ)
   AUC of −angle. Output: harp_auc_{model}_{dataset}_s{seed}.csv (AUC of every score per layer; the raw full-space
   within / across of 2 and 3 as reference). No plots.

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.hs_anal.hs_anal --model l8b \
        --dataset drlXDomainLen_wiki_300_tokens
"""

import os
from argparse import ArgumentParser, Namespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.items.plot_style import configure_plot_style
from src.utils import load_data, return_device
from src.zero import angle

cfg = Config()
RESULTS_DIR = os.path.join(cfg.base_dir, "output", "hidden_states", "results")
DIFF, AUC = "#263238", "#B0B0B0"  # dark ink / light grey
HUMAN, MACHINE = "#0072B2", "#D55E00"  # Okabe-Ito blue / vermillion, colorblind-safe
INK, MUTED, GRID = "#263238", "#7F8C8D", "#E2E6EA"
OUTLIER_FACTOR = 2  # tokens with norm > OUTLIER_FACTOR × median (per text and layer) are dropped in the right norm panel
SCORES = ("norm", "norm_no_outliers", "within", "within_centred", "anisotropy", "across", "across_centred",
          "update_ratio", "update_alpha", "update_perp_rel")
HARP_DIR = os.path.join(cfg.base_dir, "output", "harp")
SEMANTIC_FRACTION = 0.95  # share of W_U's right singular vectors (largest σ) in the semantic subspace
HARP_SUBSPACES = ("semantic", "reasoning")
HARP_SCORES = tuple(f"{sub}_{kind}" for sub in HARP_SUBSPACES for kind in ("within", "across"))


def harp_bases(unembedding: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    unembedding W_U: (V, D). Right singular vectors of W_U = eigenvectors of W_Uᵀ W_U (cheaper than an SVD of (V, D)):
    semantic = the top k = round(SEMANTIC_FRACTION · D) by σ, reasoning = the remaining D − k. Each (D, n_dims).
    """
    w = unembedding.detach().double()
    _, eigenvectors = torch.linalg.eigh(w.T @ w)  # ascending σ²
    n_reasoning = w.shape[1] - round(SEMANTIC_FRACTION * w.shape[1])
    return {"semantic": eigenvectors[:, n_reasoning:].float(), "reasoning": eigenvectors[:, :n_reasoning].float()}


def harp_scores(h: torch.Tensor, bases: dict[str, torch.Tensor], final_norm: torch.nn.Module) -> dict[str, np.ndarray]:
    """
    h: (L + 1, T, D) hidden states of one text. Per subspace, within and across angles of p = Norm(h) @ basis, each
    (L,) at transformer layers 1..L. HF already applies the final norm to the last state, so it is not normed again.
    """
    dtype = next(final_norm.parameters()).dtype
    z = torch.cat([final_norm(h[:-1].to(dtype)).float(), h[-1:]])  # (L + 1, T, D)
    scores = {}
    for sub, basis in bases.items():
        p = z @ basis  # (L + 1, T, n_dims)
        scores[f"{sub}_within"] = angle(p[1:, :-1], p[1:, 1:]).mean(dim=1)
        scores[f"{sub}_across"] = angle(p[:-1], p[1:]).mean(dim=1)
    return {key: value.double().cpu().numpy() for key, value in scores.items()}


def update_split(h: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    h: (L + 1, T, D). Per token and layer transition ℓ − 1 → ℓ, with z = h^{ℓ−1} and z^ℓ = z + Δ:
        α = ⟨Δ, z⟩ / ‖z‖²,   Δ∥ = α z,   Δ⊥ = Δ − Δ∥ (⊥ z)
    Returns, each (L, T): the ratio ‖Δ∥‖ / ‖Δ⊥‖ (> 1 the update mostly rescales z, < 1 it mostly turns z), α, and
    ‖Δ⊥‖ / ‖z‖.
    """
    z = h[:-1]
    delta = h[1:] - z
    alpha = (delta * z).sum(dim=-1) / z.pow(2).sum(dim=-1).clamp_min(1e-12)  # (L, T)
    parallel = alpha.unsqueeze(-1) * z  # Δ∥
    perpendicular = delta - parallel  # Δ⊥
    ratio = parallel.norm(dim=-1) / perpendicular.norm(dim=-1).clamp_min(1e-12)
    perp_rel = perpendicular.norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-12)  # ‖Δ⊥‖ / ‖z‖
    return {"update_ratio": ratio, "update_alpha": alpha, "update_perp_rel": perp_rel}


def text_scores(h: torch.Tensor) -> dict[str, np.ndarray]:
    """
    h: (L + 1, T, D) hidden states of one text, index 0 = embeddings. Norms are (L + 1,), every other score is (L,) at
    transformer layers 1..L.
    """
    n = h.norm(dim=-1)  # (L + 1, T)
    keep = n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values
    centred = h - h.mean(dim=1, keepdim=True)
    u = torch.nn.functional.normalize(h[1:], dim=-1)  # (L, T, D)
    n_tokens = h.shape[1]
    # mean over t ≠ s of cos(h_t, h_s) = (||Σ_t u_t||² − T) / (T (T − 1)), u_t = h_t / ||h_t||
    anisotropy = (u.sum(dim=1).pow(2).sum(dim=-1) - n_tokens) / (n_tokens * (n_tokens - 1))
    scores = {
        "norm": n.mean(dim=1),
        "norm_no_outliers": (n * keep).sum(dim=1) / keep.sum(dim=1),
        "within": angle(h[1:, :-1], h[1:, 1:]).mean(dim=1),
        "within_centred": angle(centred[1:, :-1], centred[1:, 1:]).mean(dim=1),
        "anisotropy": anisotropy,
        "across": angle(h[:-1], h[1:]).mean(dim=1),
        "across_centred": angle(centred[:-1], centred[1:]).mean(dim=1),
        # per-token angles first, then the mean over tokens (not the angle of the mean components)
        **{key: value.mean(dim=1) for key, value in update_split(h).items()},
    }
    return {key: value.double().cpu().numpy() for key, value in scores.items()}


def collect(args: Namespace) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-text scores (SCORES and HARP_SCORES), each (N, n_layers), and labels (N,)."""
    device = return_device()
    inference = Inference(model_name=args.model)
    model = inference.model
    with torch.no_grad():
        bases = {k: v.to(device) for k, v in harp_bases(model.get_output_embeddings().weight.to(device)).items()}
    print("W_U subspace dims: " + ", ".join(f"{k} {v.shape[1]}" for k, v in bases.items()))
    data = load_data(args=args)[args.split]
    out = {key: [] for key in SCORES + HARP_SCORES}
    labels = []
    for item in tqdm(data, desc="Running texts"):
        hidden_states = inference.run(item, args)["hidden_states"]
        with torch.no_grad():
            h = torch.stack(hidden_states).to(device).float()
            scores = {**text_scores(h), **harp_scores(h, bases, model.model.norm)}
        for key in out:
            out[key].append(scores[key])
        labels.append(int(item["label"]))
    del inference
    torch.cuda.empty_cache()
    return {key: np.stack(value) for key, value in out.items()}, np.asarray(labels)


def compare(score: np.ndarray, labels: np.ndarray, reverse: bool = False) -> dict[str, np.ndarray]:
    """
    score: (N, n_layers) per-text values. Per group mean ±95% CI, unpaired machine − human difference ±95% CI, AUC and
    Cohen's d = (mean machine − mean human) / pooled SD, each (n_layers,).
    """
    h, m = score[labels == 0], score[labels == 1]
    pooled_sd = np.sqrt(((len(h) - 1) * h.var(0, ddof=1) + (len(m) - 1) * m.var(0, ddof=1)) / (len(h) + len(m) - 2))
    return {
        "cohen_d": (m.mean(0) - h.mean(0)) / np.where(pooled_sd > 0, pooled_sd, np.nan),
        "human_mean": h.mean(0), "human_ci": 1.96 * h.std(0, ddof=1) / np.sqrt(len(h)),
        "machine_mean": m.mean(0), "machine_ci": 1.96 * m.std(0, ddof=1) / np.sqrt(len(m)),
        "diff_mean": m.mean(0) - h.mean(0),
        "diff_ci": 1.96 * np.sqrt(m.var(0, ddof=1) / len(m) + h.var(0, ddof=1) / len(h)),
        "auc": np.array([roc_auc_score(labels, -s if reverse else s) for s in score.T]),
    }


def auc_table(aucs: dict[str, np.ndarray], title: str, first_layer: int) -> plt.Figure:
    """One row per score, one column per layer, AUC to 2 decimals; the best layer of each score in bold."""
    names = list(aucs)
    n_layers = len(next(iter(aucs.values())))
    cells = [[f"{v:.2f}" for v in aucs[name]] for name in names]
    fig, ax = plt.subplots(figsize=(0.62 * n_layers + 2.5, 0.45 * len(names) + 1.6))
    ax.axis("off")
    table = ax.table(cellText=cells, rowLabels=names,
                     colLabels=[str(first_layer + i) for i in range(n_layers)], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for r, name in enumerate(names, start=1):
        table[r, int(np.argmax(aucs[name]))].get_text().set_fontweight("bold")
    ax.set_title(f"{title}\nAUC per transformer layer (columns); angle scores: AUC of −score; bold = best layer",
                 fontsize=11)
    fig.tight_layout()
    return fig


def harp_outputs(s: dict[str, np.ndarray], labels: np.ndarray, name: str) -> None:
    """AUC table of all W_U-subspace angles as a CSV in HARP_DIR (full space as reference); no plots."""
    os.makedirs(HARP_DIR, exist_ok=True)
    reference = {"full_within": s["within"], "full_across": s["across"]}
    aucs = {key: compare(v, labels, reverse=True)["auc"]
            for key, v in {**reference, **{key: s[key] for key in HARP_SCORES}}.items()}
    n_layers = len(next(iter(aucs.values())))
    csv_path = os.path.join(HARP_DIR, f"harp_auc_{name}.csv")
    with open(csv_path, "w") as f:
        f.write("score," + ",".join(str(i) for i in range(1, n_layers + 1)) + ",best_layer,best_auc\n")
        for key, a in aucs.items():
            f.write(f"{key}," + ",".join(f"{v:.4f}" for v in a) + f",{int(np.argmax(a)) + 1},{a.max():.4f}\n")
    print(f"Wrote {csv_path}")
    for key, a in aucs.items():
        print(f"AUC {key}: best {a.max():.4f} (layer {int(np.argmax(a)) + 1}), last layer {a[-1]:.4f}")


def plot(panels: list[list[dict]], path: str, title: str, first_layer: int, xlabel: str, groups: bool = False) -> None:
    """Draws the panels with figure() and saves them to path."""
    fig = figure(panels, title, first_layer, xlabel, groups)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def figure(panels: list[list[dict]], title: str, first_layer: int, xlabel: str, groups: bool = False) -> plt.Figure:
    """
    panels: grid (rows × columns) of dicts with "stats" (from compare), "title", "quantity" (y-axis name) and
    "auc_label" (optional "auc": False hides the AUC; then "cohen_d": True shows Cohen's d instead, on the panel's own
    scale). Per panel, per layer: on the left y-axis the machine − human difference, or with groups the mean of each
    group (±95% CI); on the right y-axis the AUC, on one scale for all panels.
    """
    configure_plot_style()
    plt.rcParams.update({"pdf.fonttype": 42, "axes.edgecolor": "#B0BEC5", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK})
    n_rows, n_cols = len(panels), max(len(row) for row in panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 4.6 * n_rows), squeeze=False)
    aucs = np.concatenate([[0.5]] + [p["stats"]["auc"] for row in panels for p in row if p.get("auc", True)])
    pad = 0.05
    auc_lim = (aucs.min() - pad, aucs.max() + pad)
    for axes_row, row in zip(axes, panels):
        for ax, p in zip(axes_row, row):
            s = p["stats"]
            layers = np.arange(first_layer, first_layer + len(s["auc"]))
            if groups:
                lines = []
                for key, label, color in (("human", "human (±95% CI)", HUMAN), ("machine", "machine (±95% CI)", MACHINE)):
                    mean, ci = s[f"{key}_mean"], s[f"{key}_ci"]
                    ax.fill_between(layers, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
                    lines += ax.plot(layers, mean, color=color, linewidth=2, label=label)
                ax.set_ylabel(p["quantity"])
            else:
                mean, ci = s["diff_mean"], s["diff_ci"]
                ax.axhline(0, color=MUTED, linewidth=0.8)
                ax.fill_between(layers, mean - ci, mean + ci, color=DIFF, alpha=0.15, linewidth=0)
                lines = ax.plot(layers, mean, color=DIFF, linewidth=2, label="machine − human (±95% CI)")
                ax.set_ylabel(f"{p['quantity']}\nmachine − human")
            ax.set_xlabel(xlabel)
            ax.set_xlim(layers[0], layers[-1])
            ax.set_title(p["title"])
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)

            if p.get("auc", True):
                twin = ax.twinx()
                twin.axhline(0.5, color=AUC, linewidth=0.8, linestyle=":")
                lines += twin.plot(layers, s["auc"], color=AUC, linewidth=2, label=p["auc_label"])
                twin.set_ylim(*auc_lim)
                twin.set_ylabel("AUC")
                twin.spines["top"].set_visible(False)
            elif p.get("cohen_d", False):  # own right axis per panel, symmetric around 0
                twin = ax.twinx()
                twin.axhline(0, color=AUC, linewidth=0.8, linestyle=":")
                lines += twin.plot(layers, s["cohen_d"], color=AUC, linewidth=2, label="Cohen's d (machine − human)")
                lim = np.nanmax(np.abs(s["cohen_d"])) * 1.1 if np.isfinite(s["cohen_d"]).any() else 1.0
                twin.set_ylim(-lim, lim)
                twin.set_ylabel("Cohen's d")
                twin.spines["top"].set_visible(False)
            ax.legend(handles=lines, frameon=False, loc="best", fontsize=11)
        for ax in axes_row[len(row):]:
            ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    args.model_name = args.model

    s, labels = collect(args)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = f"{args.model}_{args.dataset}_s{args.seed}"
    tag = f"{args.model}, {args.dataset}"
    angle_auc = "AUC (−angle, machine = 1)"

    plot([[
        {"stats": compare(s["norm"], labels), "title": "all tokens", "quantity": "mean norm",
         "auc_label": "AUC (norm, machine = 1)"},
        {"stats": compare(s["norm_no_outliers"], labels), "title": f"tokens > {OUTLIER_FACTOR}× median dropped",
         "quantity": "mean norm", "auc_label": "AUC (norm, machine = 1)"},
    ]], os.path.join(RESULTS_DIR, f"hs_norm_{name}.pdf"), f"Hidden-state norms, {tag}",
        first_layer=0, xlabel="layer (0 = embeddings)")

    plot([
        [{"stats": compare(s["within"], labels, reverse=True), "title": "consecutive-token angle, raw states",
          "quantity": "within-layer angle (rad)", "auc_label": angle_auc}],
        [{"stats": compare(s["anisotropy"], labels), "title": "anisotropy, raw states",
          "quantity": "anisotropy (mean cosine)", "auc_label": "AUC (cosine, machine = 1)"}],
        [{"stats": compare(s["within_centred"], labels, reverse=True), "title": "consecutive-token angle, centred states",
          "quantity": "within-layer angle (rad)", "auc_label": angle_auc}],
    ], os.path.join(RESULTS_DIR, f"hs_within_{name}.pdf"), f"Within layer, {tag}",
        first_layer=1, xlabel="transformer layer", groups=True)

    plot([
        [{"stats": compare(s["across"], labels, reverse=True), "title": "same-token angle, raw states",
          "quantity": "across-layer angle (rad)", "auc_label": angle_auc},
         {"stats": compare(s["across_centred"], labels, reverse=True), "title": "same-token angle, centred states",
          "quantity": "across-layer angle (rad)", "auc_label": angle_auc}],
        [{"stats": compare(s["update_ratio"], labels), "auc": False, "cohen_d": True,
          "title": r"update split $\|\Delta_\parallel\| \,/\, \|\Delta_\perp\|$, $\Delta_\parallel = \alpha z$",
          "quantity": r"$\|\Delta_\parallel\| \,/\, \|\Delta_\perp\|$"},
         {"stats": compare(s["update_alpha"], labels), "auc": False, "cohen_d": True,
          "title": r"update split $\alpha = \langle \Delta, z \rangle / \|z\|^2$ (signed)", "quantity": r"$\alpha$"}],
        [{"stats": compare(s["update_perp_rel"], labels), "auc": False, "cohen_d": True,
          "title": r"update split $\|\Delta_\perp\| \,/\, \|z\|$", "quantity": r"$\|\Delta_\perp\| \,/\, \|z\|$"}],
    ], os.path.join(RESULTS_DIR, f"hs_across_{name}.pdf"), f"Across layers, {tag}",
        first_layer=1, xlabel="transformer layer", groups=True)

    harp_outputs(s, labels, name)


if __name__ == "__main__":
    main()
