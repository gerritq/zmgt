"""
Per-text anisotropy and centroid scores at every hidden-state layer. Per text and layer, H ∈ ℝ^{T×D} are the token
states after dropping the first --skip_tokens, μ = (1/T) Σ_t h_t their centroid and μ̂ = μ / ‖μ‖:
    mean_norm:              (1/T) Σ_t ‖h_t‖                                   (reference)
    mean_angle:             mean_t ∠(h_{t−1}, h_t)                            (reference, the classic angle)
    avg_cosine:             2 / (T(T − 1)) Σ_{i<j} cos(h_i, h_j)
    avg_cosine_centred:     the same on h_t − μ
    centroid_projection:    (1/T) Σ_t p_t, p_t = ⟨h_t, μ⟩ / ‖μ‖
    centroid_distance:      (1/T) Σ_t d_t, d_t = ‖h_t − μ‖
    rel_centroid_distance:  centroid_distance / mean_norm
    cone_mean_angle:        (1/T) Σ_t α_t, α_t = arccos(⟨h_t, μ⟩ / (‖h_t‖ ‖μ‖))
    cone_alpha95:           α₀.₉₅ = Q₀.₉₅(α_1, ..., α_T), the half-angle of the cone C_α = {h : ∠(h, μ) ≤ α} around μ
                            that holds 95% of the tokens
    cone_coverage:          share of the text's tokens with α_t ≤ α₀.₉₅, its own cone (≈ 0.95 by construction; it
                            departs from it only through ties and the quantile's interpolation)
    norm2:                  (1/T) Σ_t ‖h_t‖², the total energy
    shared_centroid:        ‖μ‖²
    token_specific:         (1/T) Σ_t ‖h_t − μ‖² = norm2 − ‖μ‖²
    shared_share:           ‖μ‖² / norm2
    sideways_distance:      (1/T) Σ_t s_t, s_t = ‖h_t − p_t μ̂‖
Texts with fewer than MIN_TOKENS kept tokens get null. Per layer, every score is evaluated as is (label 1 = machine):
AUROC and Cohen's d = (mean machine − mean human) / pooled std.
"""

import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data
from src.geometric_metrics.metrics import GeometricMetrics

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "metrics", "aniso")
# Fewest kept tokens for a text to be scored (avg_cosine and mean_angle need pairs).
MIN_TOKENS = 3
# Quantile of the per-token cone angles that sets each text's cone (cone_alpha95, cone_coverage).
CONE_QUANTILE = 0.95
SCORE_DEFINITIONS = {
    "mean_norm": "mean_t ||h_t||",
    "mean_angle": "mean_t angle(h_{t-1}, h_t); radians",
    "avg_cosine": "2 / (T(T-1)) sum_{i<j} cos(h_i, h_j)",
    "avg_cosine_centred": "2 / (T(T-1)) sum_{i<j} cos(h_i - mu, h_j - mu)",
    "centroid_projection": "mean_t p_t, p_t = <h_t, mu> / ||mu||",
    "centroid_distance": "mean_t ||h_t - mu||",
    "rel_centroid_distance": "mean_t ||h_t - mu|| / mean_t ||h_t||",
    "cone_mean_angle": "mean_t alpha_t, alpha_t = arccos(<h_t, mu> / (||h_t|| ||mu||)); radians",
    "cone_alpha95": "Q_0.95(alpha_1, ..., alpha_T); radians",
    "cone_coverage": "share of tokens with alpha_t <= alpha_0.95 of the same text",
    "norm2": "mean_t ||h_t||^2",
    "shared_centroid": "||mu||^2",
    "token_specific": "mean_t ||h_t - mu||^2 = norm2 - ||mu||^2",
    "shared_share": "||mu||^2 / norm2",
    "sideways_distance": "mean_t ||h_t - p_t mu_hat||, mu_hat = mu / ||mu||",
}
SCORES = tuple(SCORE_DEFINITIONS)


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def avg_cosine(h: torch.Tensor) -> torch.Tensor:
    """Mean cosine over the pairs i < j of the rows: (L, T, D) -> (L,), from ‖Σ_t ĥ_t‖² = T + 2 Σ_{i<j} cos."""
    n = h.shape[1]
    return (unit(h).sum(dim=1).pow(2).sum(dim=-1) - n) / (n * (n - 1))


def layer_scores(h: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Every score at every layer: (L + 1, T, D) -> {score: (L + 1,)}.
    """
    mu = h.mean(dim=1, keepdim=True)  # (L + 1, 1, D)
    mu_norm = mu.norm(dim=-1).clamp_min(1e-12)  # (L + 1, 1)
    mu_hat = mu / mu_norm.unsqueeze(-1)
    norms = h.norm(dim=-1)  # (L + 1, T)
    centred = h - mu
    distance = centred.norm(dim=-1)
    projection = (h * mu_hat).sum(dim=-1)  # p_t
    cone = torch.acos((projection / norms.clamp_min(1e-12)).clamp(-1.0, 1.0))  # α_t
    alpha = torch.quantile(cone, CONE_QUANTILE, dim=1)  # α₀.₉₅ per layer
    current, previous = h[:, 1:], h[:, :-1]
    norm2 = norms.pow(2).mean(dim=1)
    shared = mu_norm.squeeze(-1).pow(2)
    scores = {
        "mean_norm": norms.mean(dim=1),
        "mean_angle": GeometricMetrics.angle(current, previous).mean(dim=1),
        "avg_cosine": avg_cosine(h),
        "avg_cosine_centred": avg_cosine(centred),
        "centroid_projection": projection.mean(dim=1),
        "centroid_distance": distance.mean(dim=1),
        "rel_centroid_distance": distance.mean(dim=1) / norms.mean(dim=1).clamp_min(1e-12),
        "cone_mean_angle": cone.mean(dim=1),
        "cone_alpha95": alpha,
        "cone_coverage": (cone <= alpha.unsqueeze(-1)).float().mean(dim=1),
        "norm2": norm2,
        "shared_centroid": shared,
        "token_specific": distance.pow(2).mean(dim=1),
        "shared_share": shared / norm2.clamp_min(1e-12),
        "sideways_distance": (h - projection.unsqueeze(-1) * mu_hat).norm(dim=-1).mean(dim=1),
    }
    return scores


def to_json(values: np.ndarray) -> list[float | None]:
    """NaN as null, so the output is valid JSON."""
    return [float(x) if np.isfinite(x) else None for x in values]


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC and Cohen's d of one score at one layer, over the texts with a finite value."""
    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    human, machine = scores[labels == 0], scores[labels == 1]
    out = {
        "n_valid": int(mask.sum()),
        "mean_human": float(human.mean()) if len(human) else None,
        "mean_machine": float(machine.mean()) if len(machine) else None,
        "auroc": None,
        "cohens_d": None,
    }
    if len(human) > 1 and len(machine) > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
        pooled = np.sqrt(((len(human) - 1) * human.var(ddof=1) + (len(machine) - 1) * machine.var(ddof=1))
                         / (len(human) + len(machine) - 2))
        out["cohens_d"] = float((machine.mean() - human.mean()) / pooled) if pooled > 0 else None
    return out


def run(args: Namespace) -> dict:
    inference = Inference(model_name=args.model)
    test_data = load_data(args=args)["test"]
    items = []
    n_layers = None
    for item in tqdm(test_data, desc="Collecting anisotropy scores"):
        hidden_states = inference.run(item, args)["hidden_states"]
        n_layers = len(hidden_states)
        # Drop the first skip_tokens after collecting the hidden states: (L + 1, T', D).
        h = torch.stack(hidden_states)[:, args.skip_tokens:].to(inference.device).float()
        meta = {
            "label": int(item["label"]), "chars": len(item["text"]),
            "tokens_before_skip": int(hidden_states[0].shape[0]), "tokens_scored": int(h.shape[1]),
        }
        if h.shape[1] < MIN_TOKENS:
            items.append({**meta, **{score: [None] * n_layers for score in SCORES}})
            continue
        scores = layer_scores(h)
        items.append({**meta, **{score: to_json(v.cpu().numpy()) for score, v in scores.items()}})

    labels = np.asarray([item["label"] for item in items])
    metrics = {}
    for score in SCORES:
        values = np.asarray([[np.nan if v is None else v for v in item[score]] for item in items], dtype=float)
        metrics[score] = [evaluate(labels, values[:, layer]) for layer in range(n_layers)]

    return {
        **vars(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "aniso",
        "scores": list(SCORES),
        "score_definitions": SCORE_DEFINITIONS,
        "layer_numbering": "entry i is hidden_states[i]: 0 = embedding output; 1..L = transformer-layer outputs",
        "auroc_orientation": "score as written, label 1 = machine; AUROC below 0.5 means the score is lower for "
                             "machine text",
        "metrics": metrics,
        "items": items,
    }


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    output = run(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"aniso_{args.model}_{args.dataset}_s{args.seed}.json")
    with open(path, "w") as f:
        json.dump(output, f)
    print(f"Wrote {path}")
    for score in SCORES:
        aurocs = [m["auroc"] for m in output["metrics"][score]]
        best = max(range(len(aurocs)), key=lambda i: abs((aurocs[i] or 0.5) - 0.5))
        print(f"{score:>22}: most separating layer {best}, AUROC {aurocs[best]}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_tokens", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main()
