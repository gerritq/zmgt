"""
Does JAS detect better inside the W_U subspaces? The joint angular score (JAS) and its two parts, computed in several
spaces of the hidden state, evaluated per layer by AUROC (human vs. machine).

Runs --model on every text of --dataset (split --split) as src.inference.Inference does (no special tokens). Layers
1..L−1 (the last hidden state, already final-normed by HF, is dropped). Tokens: all of them, except outliers. Per text
and layer, token t is kept if ‖h_t‖ ≤ OUTLIER_FACTOR × median_t ‖h_t‖ (raw states; drops the attention sink at t = 0).
The same kept tokens are used in every space.

Subspaces as HARP (Hu et al., ICLR 2026): SVD of the unembedding W_U = U Σ Vᵀ (no centring), right singular vectors
sorted by σ; reasoning = the n_R directions with the smallest σ, semantic = the other D − n_R; a state is mapped to its
coordinates V_Sᵀ x (HARP Eq. 15). Two sizes (SPLITS): "5pct" = round(0.05 D) (HARP's k = 0.95 D; 205 for D = 4096) and
"256" (the best dimension in HARP's Fig. 5b). Two inputs x (INPUTS): "raw" = h, as in HARP, and "norm" = Norm(h), the
model's final RMSNorm, what W_U actually reads (its RMS scaling leaves angles unchanged; its weight γ does not).

Spaces (p_t^ℓ = the state of token t at layer ℓ seen in that space):
    raw                       p = h, the raw hidden state (JAS)
    norm                      p = Norm(h), all D directions
    {sub}_{split}_{input}     p = V_{sub}ᵀ x for sub in (semantic, reasoning), split in SPLITS, input in INPUTS
                              e.g. reasoning_256_raw = the 256 smallest-σ directions applied to the raw h
Per text and layer ℓ, with θ_t = ∠(p_{t−1}^ℓ, p_t^ℓ) (within) and φ_t = ∠(p_t^{ℓ−1}, p_t^ℓ) (across, step ℓ−1 → ℓ):
    within = mean θ_t             over t with t−1 and t kept at ℓ
    across = mean φ_t             over t kept at ℓ−1 and at ℓ
    joint  = mean √(θ_t φ_t)      over t with both angles (t−1 and t kept at ℓ, t kept at ℓ−1)   (joint = JAS)
Embedding-normalised variants (embedding_mean_ratio, as zero/src/tests/emb_norm.py), each over the same tokens as
the score it normalises; θ_t^0 = the within angle at the embedding layer, φ_t^1 = the across angle of the first step
(embedding → layer 1, the first across angle that involves the embedding; so across_emb = 1 at layer 1):
    within_emb = mean θ_t / mean θ_t^0
    across_emb = mean φ_t / mean φ_t^1
    joint_emb  = mean √((θ_t / mean θ_t^0)(φ_t / mean φ_t^1)) = joint / √(mean θ_t^0 · mean φ_t^1)
NaN where a text has no such token. Each score is evaluated as −score (lower angle = machine, as JAS): AUROC per layer,
texts with NaN at that layer dropped.

Output: output/test/jas_subspaces_{model}_{dataset}_s{seed}.json, per space (with its dims, input and split) and
score: the AUROC per layer (index i = transformer layer i + 1), at JAS_LAYER and at the best layer.

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.ideas.test --model l8b \
        --dataset drlXDomainLen_wiki_300_tokens
"""

import json
import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.utils import load_data, return_device
from src.zero import angle

cfg = Config()
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "test")
OUTLIER_FACTOR = 2  # tokens with norm > OUTLIER_FACTOR × median (per text and layer) are dropped
SPLITS = {"5pct": lambda d: round(0.05 * d), "256": lambda d: 256}  # size n_R of the reasoning subspace
INPUTS = ("raw", "norm")  # the state that is projected: h, or the final-normed Norm(h)
JAS_LAYER = 8  # the layer JAS uses (transformer layer 8)
SCORES = ("within", "across", "joint", "within_emb", "across_emb", "joint_emb")
EPS = 1e-12


def bases(unembedding: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    unembedding W_U: (V, D). Right singular vectors of W_U (eigenvectors of W_Uᵀ W_U, ascending σ², the same as the
    SVD's V): per split, {f"reasoning_{split}"} = the n_R with the smallest σ, {f"semantic_{split}"} = the rest. Each
    (D, n_dims), float32.
    """
    w = unembedding.detach().float()
    _, eigenvectors = torch.linalg.eigh((w.T @ w).double())
    out = {}
    for split, size in SPLITS.items():
        n_reasoning = size(w.shape[1])
        out[f"semantic_{split}"] = eigenvectors[:, n_reasoning:].float()
        out[f"reasoning_{split}"] = eigenvectors[:, :n_reasoning].float()
    return out


def masked_mean(values: torch.Tensor, used: torch.Tensor) -> torch.Tensor:
    """values, used: (L', T') -> (L',) mean over the used tokens; NaN where none."""
    w = used.to(values.dtype)
    n = w.sum(dim=1)
    return torch.where(n > 0, (values * w).sum(dim=1) / n.clamp_min(1), float("nan"))


def space_scores(p: torch.Tensor, keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    p: (L, T, D') states of layers 0..L−1 in one space, keep: (L, T) kept tokens. Returns within, across, joint, each
    (L − 1,) for transformer layers 1..L−1. Angles are aligned to token t = 1..T−1.
    """
    cur = p[1:]
    theta = angle(cur[:, :-1], cur[:, 1:])  # (L − 1, T − 1): ∠(p_{t−1}^ℓ, p_t^ℓ)
    phi = angle(p[:-1, 1:], cur[:, 1:])  # (L − 1, T − 1): ∠(p_t^{ℓ−1}, p_t^ℓ)
    theta_0 = angle(p[0, :-1], p[0, 1:]).expand_as(theta)  # θ_t at the embedding layer, the same for every ℓ
    phi_1 = phi[:1].expand_as(phi)  # φ_t of the first step (embedding → layer 1), the same for every ℓ
    within_used = keep[1:, :-1] & keep[1:, 1:]  # t−1 and t kept at ℓ
    across_used = keep[:-1, 1:] & keep[1:, 1:]  # t kept at ℓ−1 and ℓ
    joint_used = within_used & across_used
    within, across = masked_mean(theta, within_used), masked_mean(phi, across_used)
    joint = masked_mean((theta * phi).sqrt(), joint_used)
    # embedding_mean_ratio: divided by the mean reference angle over the same tokens as the score at ℓ.
    theta_ref_joint, phi_ref_joint = masked_mean(theta_0, joint_used), masked_mean(phi_1, joint_used)
    return {"within": within, "across": across, "joint": joint,
            "within_emb": within / masked_mean(theta_0, within_used).clamp_min(EPS),
            "across_emb": across / masked_mean(phi_1, across_used).clamp_min(EPS),
            "joint_emb": joint / (theta_ref_joint * phi_ref_joint).sqrt().clamp_min(EPS)}


@torch.no_grad()
def text_scores(hidden_states: tuple[torch.Tensor, ...], spaces: dict[str, torch.Tensor],
                final_norm: torch.nn.Module, device: torch.device) -> dict[str, np.ndarray]:
    """{f"{space}__{score}": (L − 1,)} for one text."""
    h = torch.stack(hidden_states[:-1]).to(device).float()  # (L, T, D), layers 0..L−1
    n = h.norm(dim=-1)
    keep = n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values  # (L, T), raw norms
    z = final_norm(h.to(next(final_norm.parameters()).dtype)).float()
    inputs = {"raw": h, "norm": z}
    views = {"raw": h, "norm": z,
             **{f"{name}_{kind}": x @ basis for name, basis in spaces.items() for kind, x in inputs.items()}}
    out = {}
    for space, p in views.items():
        for score, value in space_scores(p, keep).items():
            out[f"{space}__{score}"] = value.double().cpu().numpy()
    return out


def collect(args: Namespace) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    device = return_device()
    inference = Inference(model_name=args.model)
    model = inference.model
    with torch.no_grad():
        spaces = bases(model.get_output_embeddings().weight.to(device))
    dims = {"raw": model.config.hidden_size, "norm": model.config.hidden_size,
            **{f"{name}_{kind}": int(basis.shape[1]) for name, basis in spaces.items() for kind in INPUTS}}
    print("dims: " + ", ".join(f"{k} {v}" for k, v in dims.items()))
    data = load_data(args=args)[args.split]
    rows, labels = [], []
    for item in tqdm(data, desc="Running texts"):
        hidden_states = inference.run(item, args)["hidden_states"]
        with torch.no_grad():
            rows.append(text_scores(hidden_states, spaces, model.model.norm, device))
        labels.append(int(item["label"]))
    del inference
    torch.cuda.empty_cache()
    scores = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    return scores, np.asarray(labels), dims


def aucs(scores: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, list[float]]:
    """AUROC of −score per layer (machine = smaller angle); texts with NaN at that layer dropped."""
    out = {}
    for key, values in scores.items():
        out[key] = []
        for column in values.T:
            valid = np.isfinite(column)
            out[key].append(float(roc_auc_score(labels[valid], -column[valid])))
    return out


def summary(auc: dict[str, list[float]], dims: dict[str, int]) -> dict[str, dict]:
    """
    {space: {"dims", score: {"auroc_by_layer", "at_jas_layer", "best", "best_layer"}}} for every space.
    """
    out = {}
    for space, d in dims.items():
        parts = space.split("_")
        out[space] = {"dims": d, "input": parts[-1] if len(parts) == 3 else space,
                      "split": parts[1] if len(parts) == 3 else None}
        for score in SCORES:
            curve = np.asarray(auc[f"{space}__{score}"])  # index i = layer i + 1
            best = int(np.argmax(curve))
            out[space][score] = {"at_jas_layer": float(curve[JAS_LAYER - 1]), "best": float(curve[best]),
                                 "best_layer": best + 1, "auroc_by_layer": [round(float(a), 4) for a in curve]}
    return out


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    args.model_name = args.model

    scores, labels, dims = collect(args)
    output = {"model": args.model, "dataset": args.dataset, "seed": args.seed, "split": args.split,
              "outlier_factor": OUTLIER_FACTOR, "jas_layer": JAS_LAYER, "layers": "auroc_by_layer[i] = layer i + 1",
              "auroc": summary(aucs(scores, labels), dims)}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"jas_subspaces_{args.model}_{args.dataset}_s{args.seed}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
