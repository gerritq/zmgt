"""
Hidden-state norms of human vs. machine texts, per layer. Runs --model on every text of --dataset (split --split) as
src.inference.Inference does (no special tokens) and keeps only per-text summaries, so no hidden states are saved.
Needs a GPU.

Layers: hidden_states[0] = embeddings, hidden_states[ℓ] = output of transformer layer ℓ. The last entry has the
model's final norm applied by HF (its scale is set by that norm, not by the residual stream), so it is dropped:
layers 0..L−1 are used (as in src.geometric_metrics.desc3). Everything is computed in float64.

Token variants (VARIANTS), per text and layer:
    all:    all tokens
    clean:  tokens with ‖h_t‖ ≤ OUTLIER_FACTOR × median_t ‖h_t‖ (drops outliers such as the attention sink at t = 0)

Scores per text (h_t = hidden state of token t at layer ℓ, K = number of kept tokens, μ = (1/K) Σ_t h_t):
1. Norms, per layer ℓ = 0..L−1
       norm = (1/K) Σ_t ‖h_t‖
2. Growth, per transition ℓ−1 → ℓ (ℓ = 1..L−1), over tokens kept at both layers
       log_growth^ℓ = mean_t log(‖h_t^ℓ‖ / ‖h_t^{ℓ−1}‖)       (> 0: the norm grows from ℓ−1 to ℓ)
   and the average growth rate per layer across depth, one number per text: the geometric mean of the per-layer
   growth factors over the transitions GROWTH_FROM → GROWTH_FROM + 1, ..., L−2 → L−1
       avg_growth = exp(mean_ℓ log_growth^ℓ) − 1                (0.05 = the norm grows by 5% per layer)
   GROWTH_FROM = 1 skips embeddings → layer 1, where the norm jumps by orders of magnitude.
3. Energy split, per layer ℓ = 0..L−1
       energy        = (1/K) Σ_t ‖h_t‖²         (total)
       shared_energy = ‖μ‖²                     (shared: the part all tokens of the text have in common)
       token_energy  = (1/K) Σ_t ‖h_t − μ‖²     (token-specific: spread around the text's mean)
       shared_share  = shared_energy / energy   (in [0, 1])
   energy = shared_energy + token_energy exactly.

Per group (label 0 = human, 1 = machine), per layer: mean ±95% CI per group, machine − human difference ±95% CI,
Cohen's d and AUC (machine = 1), using src.hs_anal.hs_anal.compare. Outputs in output/hidden_states/norms/, named
norms_{model}_{dataset}_s{seed}:
    .csv  group summary, one row per variant, score and layer (layer "avg" for avg_growth)
    .npz  per-text scores ({variant}__{score}, each (N, n_layers) or (N,)) and labels (N,)
    .pdf  per variant: one page of norms and energy by layer, one page of growth by transition

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.hs_anal.norm_anal --model l8b \
        --dataset drlXDomainLen_wiki_300_tokens
"""

import csv
import os
from argparse import ArgumentParser, Namespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from src.config import Config
from src.hs_anal.hs_anal import compare, figure
from src.inference import Inference
from src.utils import load_data, return_device

cfg = Config()
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "hidden_states", "norms")
OUTLIER_FACTOR = 2  # "clean" variant: tokens with norm > OUTLIER_FACTOR × median (per text and layer) are dropped
GROWTH_FROM = 1  # avg_growth averages the transitions from layer GROWTH_FROM onwards (1 = skip embeddings → layer 1)
EPS = 1e-12
VARIANTS = ("all", "clean")
LAYER_SCORES = ("norm", "energy", "shared_energy", "token_energy", "shared_share")  # (N, L), layers 0..L−1
STEP_SCORES = ("log_growth",)  # (N, L − 1), transitions into layers 1..L−1
TEXT_SCORES = ("avg_growth",)  # (N,)
SCORES = LAYER_SCORES + STEP_SCORES + TEXT_SCORES


def token_mask(n: torch.Tensor, variant: str) -> torch.Tensor:
    """n: (L, T) token norms -> (L, T) bool, the tokens kept by the variant. "clean" always keeps at least half."""
    if variant == "all":
        return torch.ones_like(n, dtype=torch.bool)
    return n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values


def text_scores(h: torch.Tensor) -> dict[str, np.ndarray]:
    """
    h: (L, T, D) float64 hidden states of one text, layers 0..L−1. Returns {f"{variant}__{score}": array}, with
    LAYER_SCORES (L,), STEP_SCORES (L − 1,) and TEXT_SCORES scalars.
    """
    n = h.norm(dim=-1)  # (L, T)
    log_n = torch.log(n.clamp_min(EPS))
    out = {}
    for variant in VARIANTS:
        keep = token_mask(n, variant)
        w = keep.to(h.dtype)
        k = w.sum(dim=1).clamp_min(1)  # (L,)
        mu = (h * w[..., None]).sum(dim=1) / k[:, None]  # (L, D)
        energy = (n.pow(2) * w).sum(dim=1) / k
        shared = mu.pow(2).sum(dim=-1)
        token = ((h - mu[:, None]).pow(2).sum(dim=-1) * w).sum(dim=1) / k

        pair = (keep[1:] & keep[:-1]).to(h.dtype)  # token kept at both ℓ−1 and ℓ, (L − 1, T)
        log_growth = ((log_n[1:] - log_n[:-1]) * pair).sum(dim=1) / pair.sum(dim=1).clamp_min(1)

        scores = {
            "norm": (n * w).sum(dim=1) / k,
            "energy": energy,
            "shared_energy": shared,
            "token_energy": token,
            "shared_share": shared / energy.clamp_min(EPS),
            "log_growth": log_growth,
            "avg_growth": torch.expm1(log_growth[GROWTH_FROM:].mean()),
        }
        out.update({f"{variant}__{key}": value.cpu().numpy() for key, value in scores.items()})
    return out


def collect(args: Namespace) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-text scores ({variant}__{score}), each (N, ...), and labels (N,)."""
    device = return_device()
    inference = Inference(model_name=args.model)
    data = load_data(args=args)[args.split]
    out, labels = {}, []
    for item in tqdm(data, desc="Running texts"):
        hidden_states = inference.run(item, args)["hidden_states"]
        with torch.no_grad():
            h = torch.stack(hidden_states[:-1]).to(device).double()  # drop the final-normed last state
            scores = text_scores(h)
        for key, value in scores.items():
            out.setdefault(key, []).append(value)
        labels.append(int(item["label"]))
    del inference
    torch.cuda.empty_cache()
    return {key: np.stack(value) for key, value in out.items()}, np.asarray(labels)


def summarise(s: dict[str, np.ndarray], labels: np.ndarray) -> list[dict]:
    """Group summary rows (one per variant, score and layer) from compare."""
    rows = []
    for variant in VARIANTS:
        for score in SCORES:
            values = s[f"{variant}__{score}"]
            if score in TEXT_SCORES:
                values, layers = values[:, None], ["avg"]
            elif score in STEP_SCORES:
                layers = list(range(1, values.shape[1] + 1))  # transition ℓ−1 → ℓ is listed under ℓ
            else:
                layers = list(range(values.shape[1]))
            stats = compare(values, labels)
            for i, layer in enumerate(layers):
                rows.append({"variant": variant, "score": score, "layer": layer,
                             **{key: float(stats[key][i]) for key in stats}})
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def write_pdf(s: dict[str, np.ndarray], labels: np.ndarray, path: str, tag: str) -> None:
    """Per variant: one page of norms and energy by layer, one page of growth by transition (groups, ±95% CI, AUC)."""
    auc = "AUC (machine = 1)"
    with PdfPages(path) as pdf:
        for variant in VARIANTS:
            v = lambda score: compare(s[f"{variant}__{score}"], labels)
            title = f"{variant} tokens" if variant == "all" else f"tokens > {OUTLIER_FACTOR}× median dropped"
            fig = figure([
                [{"stats": v("norm"), "title": "mean norm", "quantity": r"$\|h_t\|$", "auc_label": auc},
                 {"stats": v("energy"), "title": "total energy", "quantity": r"$\|h_t\|^2$", "auc_label": auc}],
                [{"stats": v("shared_energy"), "title": "shared energy", "quantity": r"$\|\mu\|^2$",
                  "auc_label": auc},
                 {"stats": v("token_energy"), "title": "token-specific energy", "quantity": r"$\|h_t - \mu\|^2$",
                  "auc_label": auc}],
                [{"stats": v("shared_share"), "title": "shared share of energy",
                  "quantity": r"$\|\mu\|^2 \,/\, \|h_t\|^2$", "auc_label": auc}],
            ], f"Norms and energy, {tag}, {title}", first_layer=0, xlabel="layer (0 = embeddings)", groups=True)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            growth = s[f"{variant}__avg_growth"]
            h, m = growth[labels == 0], growth[labels == 1]
            fig = figure([[{"stats": v("log_growth"), "title": r"$\log(\|h_t^\ell\| / \|h_t^{\ell-1}\|)$",
                            "quantity": "log growth", "auc_label": auc}]],
                         f"Norm growth, {tag}, {title}\navg growth per layer (from layer {GROWTH_FROM}): "
                         f"human {100 * h.mean():.2f}%, machine {100 * m.mean():.2f}%",
                         first_layer=1, xlabel=r"layer $\ell$ (transition $\ell-1 \to \ell$)", groups=True)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    args.model_name = args.model

    s, labels = collect(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = f"norms_{args.model}_{args.dataset}_s{args.seed}"

    split_error = max(np.max(np.abs(s[f"{v}__energy"] - s[f"{v}__shared_energy"] - s[f"{v}__token_energy"])
                             / s[f"{v}__energy"]) for v in VARIANTS)
    print(f"check energy = shared + token: max rel error {split_error:.2e}")

    np.savez_compressed(os.path.join(OUTPUT_DIR, f"{name}.npz"), labels=labels, **s)
    print(f"Wrote {os.path.join(OUTPUT_DIR, f'{name}.npz')}")
    write_csv(summarise(s, labels), os.path.join(OUTPUT_DIR, f"{name}.csv"))
    write_pdf(s, labels, os.path.join(OUTPUT_DIR, f"{name}.pdf"), f"{args.model}, {args.dataset}")

    for variant in VARIANTS:
        g = s[f"{variant}__avg_growth"]
        print(f"avg growth per layer ({variant}): human {100 * g[labels == 0].mean():.2f}%, "
              f"machine {100 * g[labels == 1].mean():.2f}%")


if __name__ == "__main__":
    main()
