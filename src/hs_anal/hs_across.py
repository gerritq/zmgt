"""
Across-layer updates of human vs. machine texts, per token and layer transition. Runs --model on every text of
--dataset (split --split) as src.inference.Inference does (no special tokens) and keeps only per-text summaries, so no
hidden states are saved. Needs a GPU.

Layers: hidden_states[0] = embeddings, hidden_states[ℓ] = output of transformer layer ℓ. The last entry has the
model's final norm applied by HF, so it is dropped: layers 0..L−1 are used, giving L − 1 transitions ℓ−1 → ℓ
(ℓ = 1..L−1), as in src.hs_anal.norm_anal. Everything is computed in float64.

Tokens: per text and layer, a token is kept if ‖h_t‖ ≤ OUTLIER_FACTOR × median_t ‖h_t‖ (drops outliers such as the
attention sink at t = 0). A transition ℓ−1 → ℓ of token t is used only if t is kept at both layers.

Per token and transition ℓ−1 → ℓ, with z = h^{ℓ−1}, Δ = h^ℓ − h^{ℓ−1} (so h^ℓ = z + Δ, z ≠ 0):
    angle      = ∠(h^{ℓ−1}, h^ℓ) = arccos(⟨z, z + Δ⟩ / (‖z‖ ‖z + Δ‖)), radians
    alpha      = ⟨Δ, z⟩ / ‖z‖²                (signed: > 0 the update grows z, < 0 it shrinks z)
    Δ∥ = α z,  Δ⊥ = Δ − α z                   (Δ = Δ∥ + Δ⊥, ⟨z, Δ⊥⟩ = 0)
    par_norm   = ‖Δ∥‖ = |α| ‖z‖
    perp_norm  = ‖Δ⊥‖
    delta_norm = ‖Δ‖                          (‖Δ‖² = ‖Δ∥‖² + ‖Δ⊥‖²)
The angle follows from the split: tan(angle) = ‖Δ⊥‖ / ((1 + α) ‖z‖).

Per text, each score is the mean over the used tokens of that transition.

Output: output/hidden_states/across/across_{model}_{dataset}_s{seed}.npz with labels (N,) (0 = human, 1 = machine)
and one array per score: angle, alpha, par_norm, perp_norm, delta_norm (N, L − 1), index i = transition i → i + 1.

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.hs_anal.hs_across --model l8b \
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
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "hidden_states", "across")
OUTLIER_FACTOR = 2  # tokens with norm > OUTLIER_FACTOR × median (per text and layer) are dropped
EPS = 1e-12


def update_split(h: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    h: (L, T, D) float64, layers 0..L−1. Per token and transition ℓ−1 → ℓ, with z = h^{ℓ−1} and Δ = h^ℓ − z:
    angle, alpha, par_norm, perp_norm, delta_norm, each (L − 1, T).
    """
    z = h[:-1]
    delta = h[1:] - z
    z_norm = z.norm(dim=-1).clamp_min(EPS)
    alpha = (delta * z).sum(dim=-1) / z_norm.pow(2)
    perpendicular = delta - alpha[..., None] * z  # Δ⊥
    cos = (z * h[1:]).sum(dim=-1) / (z_norm * h[1:].norm(dim=-1).clamp_min(EPS))
    return {
        "angle": torch.acos(cos.clamp(-1.0, 1.0)),
        "alpha": alpha,
        "par_norm": alpha.abs() * z_norm,
        "perp_norm": perpendicular.norm(dim=-1),
        "delta_norm": delta.norm(dim=-1),
    }


def text_scores(h: torch.Tensor) -> dict[str, np.ndarray]:
    """h: (L, T, D) float64 hidden states of one text -> each score (L − 1,), mean over the used tokens."""
    n = h.norm(dim=-1)  # (L, T)
    keep = n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values
    pair = (keep[1:] & keep[:-1]).to(h.dtype)  # token kept at both ℓ−1 and ℓ, (L − 1, T)
    k = pair.sum(dim=1).clamp_min(1)
    return {key: ((value * pair).sum(dim=1) / k).cpu().numpy() for key, value in update_split(h).items()}


def collect(args: Namespace) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-text scores, each (N, L − 1), and labels (N,)."""
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"across_{args.model}_{args.dataset}_s{args.seed}.npz")
    np.savez_compressed(path, labels=labels, **s)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
