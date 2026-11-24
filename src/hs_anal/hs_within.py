"""
Within-layer geometry of human vs. machine texts, per layer. Runs --model on every text of --dataset (split --split)
as src.inference.Inference does (no special tokens) and keeps only per-text summaries, so no hidden states are saved.
Needs a GPU.

Layers: hidden_states[0] = embeddings, hidden_states[ℓ] = output of transformer layer ℓ. The last entry has the
model's final norm applied by HF, so it is dropped: layers 0..L−1 are used, as in src.hs_anal.norm_anal and
src.hs_anal.hs_across. Everything is computed in float64; angles are in radians.

Tokens: per text and layer, a token is kept if ‖h_t‖ ≤ OUTLIER_FACTOR × median_t ‖h_t‖ (drops outliers such as the
attention sink at t = 0). A pair (s, t) is used only if both tokens are kept at that layer; pairs keep their original
distance in the text (a dropped token is not skipped over).

Scores per text, at each layer ℓ = 0..L−1 (K = number of kept tokens, μ = (1/K) Σ_t h_t over the kept tokens):
    within           = mean_t ∠(h_{t−1}, h_t), over used consecutive pairs
    anisotropy       = mean over used pairs s ≠ t of cos(h_s, h_t)
                       (= (‖Σ_t u_t‖² − K) / (K (K − 1)), u_t = h_t / ‖h_t‖)
    within_centred   = mean_t ∠(h_{t−1} − μ, h_t − μ), over used consecutive pairs
At layer LAG_LAYER only, for each k in LAGS:
    within_lag[k]    = mean_t ∠(h_{t−k}, h_t), over used pairs at distance k (raw states; k = 1 is within at that layer)

Output: output/hidden_states/within/within_{model}_{dataset}_s{seed}.npz with labels (N,) (0 = human, 1 = machine),
within, anisotropy, within_centred (N, L), index = layer ℓ; within_lag (N, len(LAGS)), column j = lag LAGS[j]; lags
(len(LAGS),) and lag_layer (scalar) for reference. NaN where a text has no used pair (e.g. fewer than k + 1 tokens).

Per-token angles, in a separate file within_tokens_{model}_{dataset}_s{seed}.npz: at LAG_LAYER, for the first
N_TOKENS tokens of each text, angles (N, N_TOKENS) with column t = ∠(h_{t−1}, h_t) (raw states). NaN at t = 0, where
t−1 or t is dropped as an outlier, and past the end of a shorter text. Also labels (N,) and layer (scalar).

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.hs_anal.hs_within --model l8b \
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
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "hidden_states", "within")
OUTLIER_FACTOR = 2  # tokens with norm > OUTLIER_FACTOR × median (per text and layer) are dropped
LAG_LAYER = 8  # layer (index into the hidden states, 8 = output of transformer layer 8) for the lag angles
LAGS = (4, 8, 16, 32)
N_TOKENS = 50  # per-token angles at LAG_LAYER are saved for the first N_TOKENS tokens
EPS = 1e-12
NAN = float("nan")


def angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle in radians between matching rows: (..., D), (..., D) -> (...)."""
    cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(EPS)
    return torch.acos(cos.clamp(-1.0, 1.0))


def lag_angle(h: torch.Tensor, keep: torch.Tensor, k: int) -> torch.Tensor:
    """
    h: (..., T, D), keep: (..., T) -> (...) mean of ∠(h_{t−k}, h_t) over t with both tokens kept; NaN if no pair.
    """
    if h.shape[-2] <= k:
        return torch.full(h.shape[:-2], NAN, dtype=h.dtype, device=h.device)
    pair = (keep[..., k:] & keep[..., :-k]).to(h.dtype)
    n = pair.sum(dim=-1)
    mean = (angle(h[..., :-k, :], h[..., k:, :]) * pair).sum(dim=-1) / n.clamp_min(1)
    return torch.where(n > 0, mean, NAN)


def token_angles(h: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """
    h: (T, D), keep: (T,) at one layer -> (N_TOKENS,): entry t = ∠(h_{t−1}, h_t) for t < N_TOKENS; NaN at t = 0, where
    either token is dropped, and for t ≥ T.
    """
    out = torch.full((N_TOKENS,), NAN, dtype=h.dtype, device=h.device)
    h, keep = h[:N_TOKENS], keep[:N_TOKENS]
    if h.shape[0] >= 2:
        out[1:h.shape[0]] = torch.where(keep[1:] & keep[:-1], angle(h[:-1], h[1:]), NAN)
    return out


def text_scores(h: torch.Tensor) -> dict[str, np.ndarray]:
    """
    h: (L, T, D) float64 hidden states of one text, layers 0..L−1. Returns within, anisotropy, within_centred (L,),
    within_lag (len(LAGS),) at LAG_LAYER and token_angle (N_TOKENS,) at LAG_LAYER.
    """
    n = h.norm(dim=-1)  # (L, T)
    keep = n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values
    w = keep.to(h.dtype)
    k = w.sum(dim=1)  # (L,), ≥ 1

    mu = (h * w[..., None]).sum(dim=1) / k[:, None]  # (L, D)
    unit_sum = (h / n[..., None].clamp_min(EPS) * w[..., None]).sum(dim=1)  # Σ_t u_t over kept tokens
    anisotropy = (unit_sum.pow(2).sum(dim=-1) - k) / (k * (k - 1))  # ‖Σ u‖² = K + Σ_{s≠t} cos

    scores = {
        "within": lag_angle(h, keep, 1),
        "anisotropy": torch.where(k >= 2, anisotropy, NAN),
        "within_centred": lag_angle(h - mu[:, None], keep, 1),
        "within_lag": torch.stack([lag_angle(h[LAG_LAYER], keep[LAG_LAYER], lag) for lag in LAGS]),
        "token_angle": token_angles(h[LAG_LAYER], keep[LAG_LAYER]),
    }
    return {key: value.cpu().numpy() for key, value in scores.items()}


def collect(args: Namespace) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-text scores, each (N, ...), and labels (N,)."""
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


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    args.model_name = args.model

    s, labels = collect(args)
    angles = s.pop("token_angle")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = f"{args.model}_{args.dataset}_s{args.seed}"
    path = os.path.join(OUTPUT_DIR, f"within_{name}.npz")
    np.savez_compressed(path, labels=labels, lags=np.asarray(LAGS), lag_layer=LAG_LAYER, **s)
    print(f"Wrote {path}")
    path = os.path.join(OUTPUT_DIR, f"within_tokens_{name}.npz")
    np.savez_compressed(path, labels=labels, layer=LAG_LAYER, angles=angles)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
