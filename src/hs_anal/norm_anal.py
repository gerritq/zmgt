"""
Hidden-state norms of human vs. machine texts, per layer. Runs --model on every text of --dataset (split --split) as
src.inference.Inference does (no special tokens) and keeps only per-text summaries, so no hidden states are saved.
Needs a GPU.

Layers: hidden_states[0] = embeddings, hidden_states[ℓ] = output of transformer layer ℓ. The last entry has the
model's final norm applied by HF (its scale is set by that norm, not by the residual stream), so it is dropped:
layers 0..L−1 are used (as in src.geometric_metrics.desc3). Everything is computed in float64.

Tokens: per text and layer, only tokens with ‖h_t‖ ≤ OUTLIER_FACTOR × median_t ‖h_t‖ are kept (drops outliers such as
the attention sink at t = 0; at least half of the tokens are always kept).

Scores per text (h_t = hidden state of token t at layer ℓ, K = number of kept tokens, μ = (1/K) Σ_t h_t):
1. Norms, per layer ℓ = 0..L−1
       norm = (1/K) Σ_t ‖h_t‖
2. Growth, per transition ℓ−1 → ℓ (ℓ = 1..L−1), over tokens kept at both layers
       log_growth^ℓ = mean_t log(‖h_t^ℓ‖ / ‖h_t^{ℓ−1}‖)       (> 0: the norm grows from ℓ−1 to ℓ)
3. Energy split, per layer ℓ = 0..L−1
       energy        = (1/K) Σ_t ‖h_t‖²         (total)
       shared_energy = ‖μ‖²                     (shared: the part all tokens of the text have in common)
       token_energy  = (1/K) Σ_t ‖h_t − μ‖²     (token-specific: spread around the text's mean)
   energy = shared_energy + token_energy exactly.

Output: output/hidden_states/norms/norms_{model}_{dataset}_s{seed}.npz with labels (N,) (0 = human, 1 = machine) and
one array per score: norm, energy, shared_energy, token_energy (N, L), index = layer ℓ; log_growth (N, L − 1),
index i = transition i → i + 1.

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.hs_anal.norm_anal --model l8b \
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
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "hidden_states", "norms")
OUTLIER_FACTOR = 2  # tokens with norm > OUTLIER_FACTOR × median (per text and layer) are dropped
EPS = 1e-12


def text_scores(h: torch.Tensor) -> dict[str, np.ndarray]:
    """
    h: (L, T, D) float64 hidden states of one text, layers 0..L−1. Returns norm, energy, shared_energy, token_energy
    (L,) and log_growth (L − 1,), on the kept tokens.
    """
    n = h.norm(dim=-1)  # (L, T)
    keep = n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values
    w = keep.to(h.dtype)
    k = w.sum(dim=1)  # (L,), ≥ 1
    mu = (h * w[..., None]).sum(dim=1) / k[:, None]  # (L, D)

    log_n = torch.log(n.clamp_min(EPS))
    pair = (keep[1:] & keep[:-1]).to(h.dtype)  # token kept at both ℓ−1 and ℓ, (L − 1, T)

    scores = {
        "norm": (n * w).sum(dim=1) / k,
        "energy": (n.pow(2) * w).sum(dim=1) / k,
        "shared_energy": mu.pow(2).sum(dim=-1),
        "token_energy": ((h - mu[:, None]).pow(2).sum(dim=-1) * w).sum(dim=1) / k,
        "log_growth": ((log_n[1:] - log_n[:-1]) * pair).sum(dim=1) / pair.sum(dim=1).clamp_min(1),
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
    split_error = np.max(np.abs(s["energy"] - s["shared_energy"] - s["token_energy"]) / s["energy"])
    print(f"check energy = shared + token: max rel error {split_error:.2e}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"norms_{args.model}_{args.dataset}_s{args.seed}.npz")
    np.savez_compressed(path, labels=labels, **s)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
