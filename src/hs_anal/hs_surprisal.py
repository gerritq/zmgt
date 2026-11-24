"""
Per-token angles and surprisal of human vs. machine texts, so angles can be compared within surprisal bins, plus
per-text angles in the W_U subspaces (HARP). Runs
--model on every text of --dataset (split --split) as src.inference.Inference does (no special tokens), with logits.
Needs a GPU.

Layers: hidden_states[0] = embeddings, hidden_states[ℓ] = output of transformer layer ℓ. The last entry has the
model's final norm applied by HF, so it is dropped: layers 0..L−1 are used, as in the other src.hs_anal scripts.
Angles are computed in float64, in radians.

Tokens: per text and layer, a token is kept if ‖h_t‖ ≤ OUTLIER_FACTOR × median_t ‖h_t‖ (drops outliers such as the
attention sink at t = 0). Angles involving a dropped token are NaN.

Per token t (all arrays aligned to the token position t):
    surprisal[t]     = −log p(x_t | x_<t), natural log (nats), from the model's own logits; NaN at t = 0
                       (per-token perplexity = exp(surprisal))
    within[ℓ, t]     = ∠(h_{t−1}^ℓ, h_t^ℓ), layers ℓ = 0..L−1; NaN at t = 0 and where t−1 or t is dropped at ℓ
    across[i, t]     = ∠(h_t^i, h_t^{i+1}), transition i → i + 1 (i = 0..L−2); NaN where t is dropped at i or i + 1

HARP (W_U subspaces, as the former src.hs_anal.hs_anal and src.ideas.harp_idea). Right singular vectors of the
unembedding W_U = U Σ Vᵀ, sorted by σ: reasoning = the REASONING_FRACTION · D (5%) with the smallest σ (directions the
logits barely see), semantic = the other 95%. The same basis is used at every layer. States go through the final norm
W_U reads, z_t^ℓ = Norm(h_t^ℓ), then p_t^ℓ = V_Sᵀ z_t^ℓ per subspace S; full = z_t^ℓ itself (all D directions, so the
three spaces share one view; the raw-state angles are the per-token arrays above). Per text, on the kept tokens:
    harp_within_{S}[ℓ] = mean_t ∠(p_{t−1}^ℓ, p_t^ℓ), over consecutive pairs with both tokens kept, layers 0..L−1
    harp_across_{S}[i] = mean_t ∠(p_t^i, p_t^{i+1}), over tokens kept at both layers, transition i → i + 1

Outputs in output/hidden_states/surprisal/, float32, labels (N,) in both (0 = human, 1 = machine):
    surprisal_{model}_{dataset}_s{seed}.npz, texts padded with NaN to the longest text (T_max):
        n_tokens (N,), surprisal (N, T_max), within (N, L, T_max), across (N, L − 1, T_max)
    harp_{model}_{dataset}_s{seed}.npz:
        within_{full,semantic,reasoning} (N, L), across_{full,semantic,reasoning} (N, L − 1),
        dims (3,): number of dimensions of full, semantic, reasoning

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.hs_anal.hs_surprisal --model l8b \
        --dataset drlXDomainLen_wiki_300_tokens
"""

import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.utils import load_data, return_device

cfg = Config()
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "hidden_states", "surprisal")
OUTLIER_FACTOR = 2  # tokens with norm > OUTLIER_FACTOR × median (per text and layer) are dropped
REASONING_FRACTION = 0.05  # share of W_U's right singular vectors (smallest σ) in the reasoning subspace
HARP_SPACES = ("full", "semantic", "reasoning")
EPS = 1e-12
NAN = float("nan")


def angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle in radians between matching rows: (..., D), (..., D) -> (...)."""
    cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(EPS)
    return torch.acos(cos.clamp(-1.0, 1.0))


def surprisal(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """logits: (T, V), token_ids: (T,) -> (T,) −log p(x_t | x_<t) in nats; logits[t − 1] predict x_t; NaN at t = 0."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    out = torch.full((logits.shape[0],), NAN, device=logits.device)
    out[1:] = -logp[:-1].gather(1, token_ids[1:, None].to(logits.device)).squeeze(1)
    return out


def keep_mask(h: torch.Tensor) -> torch.Tensor:
    """h: (L, T, D) -> (L, T) bool, tokens with ‖h_t‖ ≤ OUTLIER_FACTOR × median_t ‖h_t‖ at each layer."""
    n = h.norm(dim=-1)
    return n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values


def harp_bases(unembedding: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    unembedding W_U: (V, D). Right singular vectors of W_U = eigenvectors of W_Uᵀ W_U (cheaper than an SVD of (V, D)):
    reasoning = the round(REASONING_FRACTION · D) with the smallest σ, semantic = the rest. Each (D, n_dims), float64.
    """
    w = unembedding.detach().float()
    _, eigenvectors = torch.linalg.eigh((w.T @ w).double())  # ascending σ²
    n_reasoning = round(REASONING_FRACTION * w.shape[1])
    return {"semantic": eigenvectors[:, n_reasoning:], "reasoning": eigenvectors[:, :n_reasoning]}


def mean_over(values: torch.Tensor, used: torch.Tensor) -> torch.Tensor:
    """values, used: (..., T) -> (...) mean of values over used entries; NaN if none."""
    w = used.to(values.dtype)
    n = w.sum(dim=-1)
    return torch.where(n > 0, (values * w).sum(dim=-1) / n.clamp_min(1), NAN)


def harp_angles(h: torch.Tensor, keep: torch.Tensor, bases: dict[str, torch.Tensor],
                final_norm: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    h: (L, T, D) float64 (layers 0..L−1, none final-normed yet), keep: (L, T). Per space in HARP_SPACES, per-text
    harp_within_{S} (L,) and harp_across_{S} (L − 1,) on p = Norm(h) @ basis (full: Norm(h)).
    """
    z = final_norm(h.to(next(final_norm.parameters()).dtype)).double()  # (L, T, D)
    within_pair = keep[:, 1:] & keep[:, :-1]
    across_pair = keep[1:] & keep[:-1]
    out = {}
    for space in HARP_SPACES:
        p = z if space == "full" else z @ bases[space]
        out[f"harp_within_{space}"] = mean_over(angle(p[:, :-1], p[:, 1:]), within_pair)
        out[f"harp_across_{space}"] = mean_over(angle(p[:-1], p[1:]), across_pair)
    return out


def token_angles(h: torch.Tensor, keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """h: (L, T, D) float64, layers 0..L−1, keep: (L, T) -> within (L, T) and across (L − 1, T), NaN where undefined
    or dropped."""
    within = torch.full(keep.shape, NAN, dtype=h.dtype, device=h.device)
    within[:, 1:] = torch.where(keep[:, 1:] & keep[:, :-1], angle(h[:, :-1], h[:, 1:]), NAN)
    across = torch.where(keep[1:] & keep[:-1], angle(h[:-1], h[1:]), NAN)
    return {"within": within, "across": across}


def pad(rows: list[np.ndarray], length: int) -> np.ndarray:
    """Stack (..., T_i) arrays into (N, ..., length) float32, NaN-padded at the end of the token axis."""
    out = np.full((len(rows), *rows[0].shape[:-1], length), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, ..., :row.shape[-1]] = row
    return out


def collect(args: Namespace) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-token arrays (surprisal file) and per-text HARP arrays (harp file), each with labels."""
    device = return_device()
    inference = Inference(model_name=args.model)
    model = inference.model
    with torch.no_grad():
        bases = harp_bases(model.get_output_embeddings().weight.to(device))
    dims = [model.config.hidden_size] + [bases[space].shape[1] for space in HARP_SPACES[1:]]
    print("HARP dims: " + ", ".join(f"{space} {d}" for space, d in zip(HARP_SPACES, dims)))
    data = load_data(args=args)[args.split]
    harp_keys = [f"harp_{kind}_{space}" for kind in ("within", "across") for space in HARP_SPACES]
    rows = {"surprisal": [], "within": [], "across": []}
    harp = {key: [] for key in harp_keys}
    labels, n_tokens = [], []
    for item in tqdm(data, desc="Running texts"):
        out = inference.run(item, args)
        with torch.no_grad():
            h = torch.stack(out["hidden_states"][:-1]).to(device).double()  # drop the final-normed last state
            keep = keep_mask(h)
            scores = {**token_angles(h, keep), **harp_angles(h, keep, bases, model.model.norm),
                      "surprisal": surprisal(out["logits"].to(device), out["token_ids"])}
        for key in rows:
            rows[key].append(scores[key].float().cpu().numpy())
        for key in harp:
            harp[key].append(scores[key].float().cpu().numpy())
        labels.append(int(item["label"]))
        n_tokens.append(out["n_tokens"])
    del inference
    torch.cuda.empty_cache()
    t_max = max(n_tokens)
    labels = np.asarray(labels)
    tokens = {"labels": labels, "n_tokens": np.asarray(n_tokens), **{key: pad(value, t_max) for key, value in rows.items()}}
    harp_out = {"labels": labels, "dims": np.asarray(dims),
                **{key.removeprefix("harp_"): np.stack(value) for key, value in harp.items()}}
    return tokens, harp_out


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    args.model_name = args.model
    args.return_logits = True
    args.return_token_ids = True

    tokens, harp = collect(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = f"{args.model}_{args.dataset}_s{args.seed}"
    for prefix, arrays in (("surprisal", tokens), ("harp", harp)):
        path = os.path.join(OUTPUT_DIR, f"{prefix}_{name}.npz")
        np.savez_compressed(path, **arrays)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
