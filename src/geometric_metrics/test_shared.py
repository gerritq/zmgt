import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data, return_device
from src.zero import SKIP, MAX_SKIP_SHARE, CONTEXT_K, angle

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "metrics", "test_shared")
# Every test runs on one layer ℓ, indexed as hidden_states: 0 = embedding output, ℓ = transformer layer ℓ (the last one,
# L, is already normed by HF). Per text, H ∈ ℝ^{T×D} are the kept tokens (the first SKIP dropped, the text skipped as in
# src.zero) and h̄ = mean_t h_t its centroid. All scores are label-free; they are evaluated as is (label 1 = machine), so
# AUROC 0.5 = no class difference and < 0.5 = lower for machine text; cohens_d = (mean machine − mean human) / pooled std.
#
# A. Shared vs. token-specific part:
#   mean_norm:          mean_t ‖h_t‖                      (norm alone, reference)
#   mean_angle:         mean_t ∠(h_{t−1}, h_t)            (angle alone, reference)
#   norm2:              mean_t ‖h_t‖²
#   sigma2_diff:        ½ mean_t ‖h_t − h_{t−1}‖²         (token-specific part)
#   sigma2_centred:     mean_t ‖h_t − h̄‖²                 (token-specific part)
#   shared_dot:         mean_t ⟨h_{t−1}, h_t⟩             (shared part ‖c‖²)
#   shared_centroid:    ‖h̄‖² − sigma2_centred / T         (shared part ‖c‖²)
#   shared_share:       ‖h̄‖² / mean_t ‖h_t‖² ∈ [0, 1]
#   Decomposition: norm2 ≈ shared_dot + sigma2_diff (exact up to the first/last token); per layer, the fraction of the
#   machine − human gap in norm2 carried by each term (frac_residual = the rest).
# B. PCA within the text:
#   top_share_uncentred:  s₁² / Σ s_i², s the singular values of H
#   top_cos_centroid:     cos(v₁, h̄), v₁ the top right singular vector of H (sign: ⟨v₁, h̄⟩ ≥ 0)
#   explained_top{k}:     Σ_{i≤k} λ_i / Σ λ_i, λ the eigenvalues of the centred covariance of H, k in EXPLAINED_K
#   eff_dim:              (Σ λ)² / Σ λ²
#   spectral_entropy:     −Σ p log p, p = λ / Σ λ
#   consecutive_centred_cos: mean_t cos(h_t − h̄, h_{t−1} − h̄)
#   own_removed_k{k}_mean_norm / _mean_angle: mean_norm and mean_angle after projecting out the text's own top k
#                         uncentred singular vectors, k in REMOVE_K
# C. Token-identity control: every layer is also scored with each token divided by its embedding norm,
#   h_t^ℓ / ‖h_t^0‖ (variant emb_scaled; at ℓ = 0 the unit-norm embeddings); layer 0 itself is the embedding output.
# D. Corpus-level PCA: one uncentred SVD of the pooled kept tokens of all texts of --fit_split (never per group), its top
#   components g₁, g₂, ... (sign: ⟨g_i, pooled mean⟩ ≥ 0).
#   top_vector_cosines:   mean pairwise cos(v₁ⁱ, v₁ʲ) between texts, for human–human, machine–machine, human–machine
#   global_alignment:     mean_t cos(h_t, g₁)
#   top_cos_global:       cos(v₁, g₁)
#   With h_t = p_t·g₁ + r_t:
#   global_projection:    mean_t p_t, p_t = ⟨h_t, g₁⟩
#   global_residual_norm: mean_t ‖r_t‖ = mean_t ‖h_t − p_t·g₁‖ (same as mean_norm of global_removed_k1)
#   global_removed_k{k}:  every A and B score after projecting out g₁..g_k (variant), k in REMOVE_K
#   removal_comparison:   mean_norm and mean_angle as is, after removing the text's own top k and the global top k
EXPLAINED_K = (1, 5, 10)
REMOVE_K = (1, 2, 4, 8)
FIT_SPLITS = ("test", "val", "train")


def keep(n_tokens: int) -> bool:
    """Whether a text has enough tokens after dropping the first SKIP (same rule as src.zero)."""
    kept = n_tokens - SKIP
    return kept >= (1 - MAX_SKIP_SHARE) * n_tokens and kept >= CONTEXT_K + 2


def project_out(h: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Remove the span of the orthonormal rows of directions (k, D) from every row of h (T, D)."""
    return h - (h @ directions.T) @ directions


def align(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Flip the sign of vector so that ⟨vector, reference⟩ ≥ 0."""
    return -vector if (vector @ reference).item() < 0 else vector


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-12)


def shared_scores(h: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    """A and B scores of one text at one layer, and its top uncentred singular vector v₁: (T, D) -> {score}, (D,)."""
    n_tokens = h.shape[0]
    centroid = h.mean(dim=0)
    centred = h - centroid
    norm2 = h.pow(2).sum(dim=-1)
    sigma2_centred = centred.pow(2).sum(dim=-1).mean()
    centroid2 = centroid.pow(2).sum()

    # A. Shared vs. token-specific part.
    scores = {
        "mean_norm": norm2.sqrt().mean(),
        "mean_angle": angle(h[:-1], h[1:]).mean(),
        "norm2": norm2.mean(),
        "sigma2_diff": 0.5 * (h[1:] - h[:-1]).pow(2).sum(dim=-1).mean(),
        "sigma2_centred": sigma2_centred,
        "shared_dot": (h[:-1] * h[1:]).sum(dim=-1).mean(),
        "shared_centroid": centroid2 - sigma2_centred / n_tokens,
        "shared_share": centroid2 / norm2.mean().clamp_min(1e-12),
    }

    # B. Uncentred SVD.
    _, singular, vh = torch.linalg.svd(h, full_matrices=False)
    s2 = singular.pow(2)
    top = align(vh[0], centroid)
    scores["top_share_uncentred"] = s2[0] / s2.sum().clamp_min(1e-12)
    scores["top_cos_centroid"] = cosine(top, centroid)
    # Centred PCA, λ_i = s_i(H − h̄)² / T.
    lam = torch.linalg.svdvals(centred).pow(2) / n_tokens
    total = lam.sum().clamp_min(1e-12)
    for k in EXPLAINED_K:
        scores[f"explained_top{k}"] = lam[:k].sum() / total
    p = lam / total
    scores["eff_dim"] = total.pow(2) / lam.pow(2).sum().clamp_min(1e-24)
    scores["spectral_entropy"] = -(p * p.clamp_min(1e-30).log()).sum()
    scores["consecutive_centred_cos"] = cosine(centred[1:], centred[:-1]).mean()
    # Norm and angle after removing the text's own top k uncentred components.
    for k in REMOVE_K:
        rest = project_out(h, vh[:k])
        scores[f"own_removed_k{k}_mean_norm"] = rest.norm(dim=-1).mean()
        scores[f"own_removed_k{k}_mean_angle"] = angle(rest[:-1], rest[1:]).mean()
    return {name: value.item() for name, value in scores.items()}, top


def layer_scores(h: torch.Tensor, embedding_norm: torch.Tensor, global_directions: torch.Tensor
                 ) -> tuple[dict[str, dict[str, float]], torch.Tensor]:
    """Every variant of the scores of one text at one layer: (T, D), (T,), (K, D) -> {variant: {score}}, v₁ (D,)."""
    raw, top = shared_scores(h)
    g1 = global_directions[0]
    raw["global_alignment"] = cosine(h, g1[None]).mean().item()
    raw["top_cos_global"] = cosine(top, g1).item()
    raw["global_projection"] = (h @ g1).mean().item()
    raw["global_residual_norm"] = project_out(h, g1[None]).norm(dim=-1).mean().item()
    out = {"raw": raw, "emb_scaled": shared_scores(h / embedding_norm[:, None])[0]}
    for k in REMOVE_K:
        out[f"global_removed_k{k}"] = shared_scores(project_out(h, global_directions[:k]))[0]
    return out, top


def fit_global(inference: Inference, args: Namespace, layers: list[int], device: torch.device) -> dict[int, dict]:
    """
    Uncentred SVD of the pooled kept tokens of every text of --fit_split, per layer, via the Gram matrix Σ h hᵀ:
    {layer: {"directions": (K, D) top right singular vectors, "energy_share": (K,) s_i² / Σ s², ...}}.
    """
    splits = load_data(args=args)
    if args.fit_split not in splits:
        raise ValueError(f"--fit_split {args.fit_split} not in {args.dataset} (splits: {', '.join(splits)})")
    fit_data = splits[args.fit_split]
    gram, total, n_tokens = {}, {}, 0
    for item in tqdm(fit_data, desc=f"Fitting the global SVD on {args.fit_split}"):
        hidden_states = inference.run(item, args)["hidden_states"]
        if not keep(hidden_states[0].shape[0]):
            continue
        for layer in layers:
            h = hidden_states[layer][SKIP:].to(device, torch.float64)
            if layer not in gram:
                gram[layer] = torch.zeros(h.shape[1], h.shape[1], dtype=torch.float64, device=device)
                total[layer] = torch.zeros(h.shape[1], dtype=torch.float64, device=device)
            gram[layer] += h.T @ h
            total[layer] += h.sum(dim=0)
        n_tokens += hidden_states[0].shape[0] - SKIP

    n_components = max(REMOVE_K)
    fits = {}
    for layer in layers:
        eigenvalues, eigenvectors = torch.linalg.eigh(gram[layer])  # ascending
        directions = eigenvectors[:, -n_components:].flip(-1).T  # (K, D), largest first
        directions = torch.stack([align(d, total[layer]) for d in directions])
        fits[layer] = {
            "directions": directions,
            "energy_share": (eigenvalues[-n_components:].flip(-1) / eigenvalues.sum()).tolist(),
            "g1_cos_pooled_mean": cosine(directions[0], total[layer]).item(),
        }
    print(f"Global SVD fitted on {len(fit_data)} texts ({n_tokens} kept tokens) of {args.fit_split}")
    return fits


def group_stats(labels: np.ndarray, values: np.ndarray) -> dict:
    """Group means, AUROC of the score as is (label 1 = machine) and Cohen's d over the texts with a value."""
    valid = np.isfinite(values)
    y, x = labels[valid], values[valid]
    human, machine = x[y == 0], x[y == 1]
    out = {"n_valid": int(valid.sum()), "mean_human": None, "mean_machine": None, "auroc": None, "cohens_d": None}
    if len(human) < 2 or len(machine) < 2:
        return out
    out["mean_human"], out["mean_machine"] = float(human.mean()), float(machine.mean())
    if np.std(x) > 0:
        out["auroc"] = float(roc_auc_score(y, x))
    pooled = np.sqrt(((len(human) - 1) * human.var(ddof=1) + (len(machine) - 1) * machine.var(ddof=1))
                     / (len(human) + len(machine) - 2))
    if pooled > 0:
        out["cohens_d"] = float((machine.mean() - human.mean()) / pooled)
    return out


def decomposition(labels: np.ndarray, scores: dict[str, np.ndarray]) -> dict:
    """Fraction of the machine − human gap in norm2 carried by shared_dot and by sigma2_diff."""
    valid = np.isfinite(scores["norm2"])
    gap = lambda name: float(scores[name][valid & (labels == 1)].mean() - scores[name][valid & (labels == 0)].mean())
    gap_norm2 = gap("norm2")
    if gap_norm2 == 0 or not np.isfinite(gap_norm2):
        return {"gap_norm2": gap_norm2, "frac_shared": None, "frac_token": None, "frac_residual": None}
    frac_shared, frac_token = gap("shared_dot") / gap_norm2, gap("sigma2_diff") / gap_norm2
    return {"gap_norm2": gap_norm2, "gap_shared_dot": gap("shared_dot"), "gap_sigma2_diff": gap("sigma2_diff"),
            "frac_shared": frac_shared, "frac_token": frac_token, "frac_residual": 1 - frac_shared - frac_token}


def top_vector_cosines(labels: np.ndarray, tops: np.ndarray) -> dict:
    """Mean pairwise cosine between the texts' top uncentred singular vectors (unit, sign-aligned), by group pair."""
    valid = np.isfinite(tops).all(axis=1)
    y, cos = labels[valid], tops[valid] @ tops[valid].T
    off_diagonal = ~np.eye(len(y), dtype=bool)
    human, machine = y == 0, y == 1
    return {
        "human_human": float(cos[np.outer(human, human) & off_diagonal].mean()),
        "machine_machine": float(cos[np.outer(machine, machine) & off_diagonal].mean()),
        "human_machine": float(cos[np.outer(human, machine)].mean()),
    }


def removal_comparison(metrics: dict[str, dict[str, dict]]) -> dict:
    """mean_norm and mean_angle as is, after removing the text's own top k and after removing the global top k."""
    out = {"none": {score: metrics["raw"][score] for score in ("mean_norm", "mean_angle")}}
    for k in REMOVE_K:
        out[f"k{k}"] = {
            "own": {score: metrics["raw"][f"own_removed_k{k}_{score}"] for score in ("mean_norm", "mean_angle")},
            "global": {score: metrics[f"global_removed_k{k}"][score] for score in ("mean_norm", "mean_angle")},
        }
    return out


def fmt(stats: dict | None) -> str:
    return f"{stats['auroc']:.3f}" if stats and stats["auroc"] is not None else "--"


def print_layer(layer: int, metrics: dict, fit: dict, decomposition_by_variant: dict, cosines: dict,
                comparison: dict) -> None:
    name = "embeddings" if layer == 0 else f"transformer layer {layer}"
    shares = ", ".join(f"{s:.3f}" for s in fit["energy_share"][:3])
    print(f"\n=== {name} | global SVD energy share g1..g3: {shares} | cos(g1, pooled mean): "
          f"{fit['g1_cos_pooled_mean']:.3f} ===")
    variants = list(metrics)
    print(f"AUROC (score as is, label 1 = machine)\n{'score':>28}" + "".join(f"{v:>20}" for v in variants))
    for score in metrics["raw"]:
        print(f"{score:>28}" + "".join(f"{fmt(metrics[v].get(score)):>20}" for v in variants))
    print("Decomposition of the machine − human gap in norm2 (frac shared_dot / sigma2_diff / residual):")
    for variant, d in decomposition_by_variant.items():
        if d["frac_shared"] is None:
            print(f"{variant:>28}  gap norm2 {d['gap_norm2']:.4g}")
        else:
            print(f"{variant:>28}  gap norm2 {d['gap_norm2']:.4g}: {d['frac_shared']:.3f} / {d['frac_token']:.3f} / "
                  f"{d['frac_residual']:.3f}")
    print(f"Mean pairwise cos of the texts' top vectors: human–human {cosines['human_human']:.3f}, "
          f"machine–machine {cosines['machine_machine']:.3f}, human–machine {cosines['human_machine']:.3f}")
    print(f"Removal (AUROC mean_norm / mean_angle): none {fmt(comparison['none']['mean_norm'])} / "
          f"{fmt(comparison['none']['mean_angle'])}")
    for k in REMOVE_K:
        own, glob = comparison[f"k{k}"]["own"], comparison[f"k{k}"]["global"]
        print(f"{'k = ' + str(k):>10}  own {fmt(own['mean_norm'])} / {fmt(own['mean_angle'])}   "
              f"global {fmt(glob['mean_norm'])} / {fmt(glob['mean_angle'])}")


def to_json(values: np.ndarray) -> list:
    """NaN (skipped text) as null, so the output is valid JSON."""
    return np.where(np.isfinite(values), values, None).tolist()


def run(args: Namespace) -> dict:
    device = return_device()
    inference = Inference(model_name=args.model)
    n_layers = inference.model.config.num_hidden_layers
    layers = args.layers or sorted({0, 1, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})
    if any(layer < 0 or layer > n_layers for layer in layers):
        raise ValueError(f"--layers must be in 0..{n_layers}")
    fits = fit_global(inference, args, layers, device)

    test_data = load_data(args=args)["test"]
    labels = np.asarray([item["label"] for item in test_data])
    per_text = {layer: [] for layer in layers}  # {layer: [{variant: {score}} or None]}
    tops = {layer: [] for layer in layers}
    for item in tqdm(test_data, desc="Scoring the test texts"):
        hidden_states = inference.run(item, args)["hidden_states"]
        if not keep(hidden_states[0].shape[0]):
            for layer in layers:
                per_text[layer].append(None)
                tops[layer].append(None)
            continue
        embedding_norm = hidden_states[0][SKIP:].to(device, torch.float64).norm(dim=-1).clamp_min(1e-12)
        for layer in layers:
            h = hidden_states[layer][SKIP:].to(device, torch.float64)
            scores, top = layer_scores(h, embedding_norm, fits[layer]["directions"])
            per_text[layer].append(scores)
            tops[layer].append(top.cpu().numpy())

    n_skipped = sum(scores is None for scores in per_text[layers[0]])
    print(f"\nshared vs. token-specific | model: {args.model} | dataset: {args.dataset} | seed: {args.seed} | "
          f"global fit: {args.fit_split} | skipped texts: {n_skipped}")
    results = {key: {} for key in ("metrics", "decomposition", "top_vector_cosines", "removal_comparison", "values")}
    for layer in layers:
        template = next(scores for scores in per_text[layer] if scores is not None)
        values = {variant: {score: np.asarray([np.nan if s is None else s[variant][score] for s in per_text[layer]])
                            for score in variant_scores}
                  for variant, variant_scores in template.items()}
        metrics = {variant: {score: group_stats(labels, v) for score, v in scores.items()}
                   for variant, scores in values.items()}
        dimension = next(t for t in tops[layer] if t is not None).shape[0]
        top_matrix = np.stack([np.full(dimension, np.nan) if t is None else t for t in tops[layer]])
        key = f"layer_{layer}"
        results["metrics"][key] = metrics
        results["decomposition"][key] = {variant: decomposition(labels, scores) for variant, scores in values.items()}
        results["top_vector_cosines"][key] = top_vector_cosines(labels, top_matrix)
        results["removal_comparison"][key] = removal_comparison(metrics)
        results["values"][key] = {variant: {score: to_json(v) for score, v in scores.items()}
                                  for variant, scores in values.items()}
        print_layer(layer, metrics, fits[layer], results["decomposition"][key], results["top_vector_cosines"][key],
                    results["removal_comparison"][key])

    output = {
        **vars(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "test_shared",
        "skip_tokens": SKIP,
        "max_skip_share": MAX_SKIP_SHARE,
        "explained_k": EXPLAINED_K,
        "remove_k": REMOVE_K,
        "layers": layers,
        "layer_numbering": "layer_i = hidden_states[i]: 0 = embedding output, i = transformer layer i",
        "auroc_orientation": "score as is, label 1 = machine (0.5 = no class difference)",
        "n_skipped_items": n_skipped,
        "global_fit": {f"layer_{layer}": {k: v for k, v in fit.items() if k != "directions"}
                       for layer, fit in fits.items()},
        **{key: value for key, value in results.items() if key != "values"},
    }
    if args.save_scores:
        output["per_text"] = {"labels": labels.tolist(), "values": results["values"]}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = f"test_shared_{args.model_name}_{args.dataset}_fit{args.fit_split}_s{args.seed}"
    with open(os.path.join(OUTPUT_DIR, f"{file_name}.json"), "w") as f:
        json.dump(output, f, indent=4)
    return results


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="hidden_states indices (0 = embeddings); default 0, 1, L/4, L/2, 3L/4, L−1.")
    parser.add_argument("--fit_split", type=str, choices=FIT_SPLITS, default="test",
                        help="Split whose pooled tokens the global SVD is fitted on (test D).")
    parser.add_argument("--save_scores", type=int, choices=(0, 1), default=0,
                        help="Set to 1 to store the per-text scores (and labels) in the output json.")
    args = parser.parse_args()
    args.save_scores = bool(args.save_scores)
    return args


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    run(args)


if __name__ == "__main__":
    main()
