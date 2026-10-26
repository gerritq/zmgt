"""
Descriptive per-text statistics at every hidden-state layer ℓ = 0..L (0 = embedding output), H^ℓ ∈ ℝ^{T×D}:
    representation_dispersion:  2 / (T(T − 1)) Σ_{i<j} (1 − cos(h_i, h_j)), the mean pairwise cosine distance
    angle_within:               mean_t ∠(h_{t−1}, h_t), consecutive tokens at the same layer
    magnitude_within:           mean_t ‖h_t − h_{t−1}‖
    angle_across:               mean_t ∠(h_t^{ℓ−1}, h_t^ℓ), the same token at the previous layer (null at layer 0)
    magnitude_across:           mean_t ‖h_t^ℓ − h_t^{ℓ−1}‖ (null at layer 0)
Per text, also ppl = exp(−(1/(T − 1)) Σ_{t≥1} log p(x_t | x_{<t})), and the norm ‖h_t^ℓ‖ of every token at every layer.
Per token, the change Δ to a state is split into the part along the previous state h and the rest:
    Δ_∥ = (Δ · h / ‖h‖²) h,  Δ_⊥ = Δ − Δ_∥;  stored as the signed length Δ · h / ‖h‖ (> 0 = the state grows along
    itself) and ‖Δ_⊥‖, for
    across layers:  Δ = h_t^ℓ − h_t^{ℓ−1}, h = h_t^{ℓ−1} (NaN at layer 0)
    within a layer: Δ = h_t^ℓ − h_{t−1}^ℓ, h = h_{t−1}^ℓ (NaN at position 0)
PCA per layer: at each layer ℓ, X^ℓ ∈ ℝ^{N×D} holds all N tokens of all texts except each text's first token (human and
machine pooled); one centred PCA with two components (sklearn) is fitted on X^ℓ and these tokens are projected onto it.
The first token is left out because its norm is far larger: it would take up PC1. The axes differ between layers, so
coordinates are comparable within a layer only.
The model runs once: every text's states are kept on the CPU in bfloat16 (float16 can overflow) ((L + 1) · N · D · 2 bytes, e.g. ~27 GB for
33 layers, 100k tokens, D = 4096), and X^ℓ (float32, N · D · 4 bytes) is built one layer at a time.
Everything goes to one .npz per model, dataset and seed, one row per text (see the "meta" entry for each array's shape);
texts shorter than the longest are padded with NaN along the token axis.
Note: hidden_states[L] is the final-norm output, so layer L describes the normed state.
"""

import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from sklearn.decomposition import PCA
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data
from src.geometric_metrics.metrics import GeometricMetrics
from src.geometric_metrics.aniso import avg_cosine

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "desc2")
# Fewest tokens for a text to be scored at a layer (dispersion and the within-layer scores need pairs).
MIN_TOKENS = 3
PCA_COMPONENTS = 2
SCORE_DEFINITIONS = {
    "representation_dispersion": "2 / (T(T-1)) sum_{i<j} (1 - cos(h_i, h_j))",
    "angle_within": "mean_t angle(h_{t-1}, h_t) at the same layer; radians",
    "magnitude_within": "mean_t ||h_t - h_{t-1}|| at the same layer",
    "angle_across": "mean_t angle(h_t^{l-1}, h_t^l); radians; null at layer 0",
    "magnitude_across": "mean_t ||h_t^l - h_t^{l-1}||; null at layer 0",
}


def layer_scores(h: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every score at every layer: (L + 1, T, D) -> {score: (L + 1,)}."""
    nan = torch.full((1,), float("nan"), device=h.device)
    if h.shape[1] < MIN_TOKENS:
        return {score: nan.expand(h.shape[0]) for score in SCORE_DEFINITIONS}
    current, previous = h[:, 1:], h[:, :-1]  # along the tokens
    upper, lower = h[1:], h[:-1]  # along the layers
    return {
        "representation_dispersion": 1 - avg_cosine(h),
        "angle_within": GeometricMetrics.angle(current, previous).mean(dim=1),
        "magnitude_within": GeometricMetrics.magnitude(current, previous).mean(dim=1),
        "angle_across": torch.cat([nan, GeometricMetrics.angle(upper, lower).mean(dim=1)]),
        "magnitude_across": torch.cat([nan, GeometricMetrics.magnitude(upper, lower).mean(dim=1)]),
    }


def decompose(delta: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed length of delta along base and the length of the rest: (..., D), (..., D) -> (...,), (...,)."""
    unit = base / base.norm(dim=-1, keepdim=True)
    parallel = (delta * unit).sum(dim=-1)
    return parallel, (delta - parallel[..., None] * unit).norm(dim=-1)


def token_scores(h: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-token norm and change decompositions: (L + 1, T, D) -> {name: (L + 1, T)}, NaN where undefined."""
    nan_layer = torch.full((1, h.shape[1]), float("nan"), device=h.device)
    nan_token = torch.full((h.shape[0], 1), float("nan"), device=h.device)
    across = decompose(h[1:] - h[:-1], h[:-1])
    within = decompose(h[:, 1:] - h[:, :-1], h[:, :-1])
    scores = {
        "norms": h.norm(dim=-1),
        "across_parallel": torch.cat([nan_layer, across[0]]),
        "across_perpendicular": torch.cat([nan_layer, across[1]]),
        "within_parallel": torch.cat([nan_token, within[0]], dim=1),
        "within_perpendicular": torch.cat([nan_token, within[1]], dim=1),
    }
    return {name: v.cpu().numpy() for name, v in scores.items()}


def perplexity(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    """exp of the mean next-token negative log-likelihood: (T, V), (T,) -> float."""
    nll = torch.nn.functional.cross_entropy(logits[:-1].float(), token_ids[1:].to(logits.device))
    return float(nll.exp())


def run(args: Namespace) -> dict[str, np.ndarray]:
    inference = Inference(model_name=args.model)
    args.return_token_ids = True
    test_data = load_data(args=args)["test"]

    args.return_logits = True
    items, states, per_token = [], [], []
    for item in tqdm(test_data, desc="Collecting desc2 statistics"):
        out = inference.run(item, args)
        h = torch.stack(out["hidden_states"]).to(inference.device).float()  # (L + 1, T, D)
        states.append(h.bfloat16().cpu())
        per_token.append(token_scores(h))  # {name: (L + 1, T)}
        items.append({
            "label": int(item["label"]), "chars": len(item["text"]), "tokens": int(h.shape[1]),
            "ppl": perplexity(out["logits"], out["token_ids"]),
            **{score: v.cpu().numpy() for score, v in layer_scores(h).items()},
        })
    del inference
    torch.cuda.empty_cache()

    tokens = np.asarray([item["tokens"] for item in items])
    n_layers = states[0].shape[0]
    token_arrays = {name: np.full((len(items), n_layers, tokens.max()), np.nan, dtype=np.float32)
                    for name in per_token[0]}
    for i, text in enumerate(per_token):
        for name, values in text.items():
            token_arrays[name][i, :, :tokens[i]] = values
    projected = np.full((len(items), n_layers, tokens.max(), PCA_COMPONENTS), np.nan, dtype=np.float32)
    ratio = np.empty((n_layers, PCA_COMPONENTS))
    for layer in tqdm(range(n_layers), desc="PCA per layer"):
        x = np.concatenate([h[layer, 1:].float().numpy() for h in states])  # (N, D), first tokens left out
        pca = PCA(n_components=PCA_COMPONENTS, random_state=args.seed)
        coords = np.split(pca.fit_transform(x), np.cumsum(tokens - 1)[:-1])
        ratio[layer] = pca.explained_variance_ratio_
        for i, text_coords in enumerate(coords):
            projected[i, layer, 1:tokens[i]] = text_coords
        del x

    meta = {
        **vars(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "desc2",
        "scores": list(SCORE_DEFINITIONS),
        "score_definitions": SCORE_DEFINITIONS,
        "pca_first_token": "excluded",
        "layer_numbering": "layer i is hidden_states[i]: 0 = embedding output; 1..L = transformer-layer outputs "
                           "(L = after the final norm)",
        "arrays": {
            "label, chars, tokens, ppl": "(texts,); ppl = exp(mean_t -log p(x_t | x_<t)), t = 1..T-1",
            "<score>": "(texts, L + 1) float32, one per entry of scores; NaN where undefined",
            "norms": "(texts, L + 1, max T) float32, ||h_t|| per token; NaN after the text's last token",
            "across_parallel, across_perpendicular": "(texts, L + 1, max T) float32, delta = h_t^l - h_t^{l-1} "
                                                     "split along h_t^{l-1}: signed length delta . h / ||h|| and "
                                                     "||delta - parallel part||; NaN at layer 0 and after the last token",
            "within_parallel, within_perpendicular": "(texts, L + 1, max T) float32, delta = h_t^l - h_{t-1}^l "
                                                     "split along h_{t-1}^l, as above; NaN at position 0 and after the "
                                                     "last token",
            "pca": "(texts, L + 1, max T, 2) float32, per layer one centred PCA on all tokens of all texts except "
                   "position 0, each token projected onto its top two components; axes differ between layers; NaN at "
                   "position 0 and after the text's last token",
            "pca_explained_variance_ratio": "(L + 1, 2)",
        },
    }
    return {
        "meta": np.asarray(json.dumps(meta)),
        "label": np.asarray([item["label"] for item in items]),
        "chars": np.asarray([item["chars"] for item in items]),
        "tokens": tokens,
        "ppl": np.asarray([item["ppl"] for item in items]),
        **{score: np.stack([item[score] for item in items]).astype(np.float32) for score in SCORE_DEFINITIONS},
        **token_arrays,
        "pca": projected,
        "pca_explained_variance_ratio": ratio,
    }


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    arrays = run(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"desc2_{args.model}_{args.dataset}_s{args.seed}.npz")
    np.savez_compressed(path, **arrays)
    print(f"Wrote {path}")
    labels, ppl = arrays["label"], arrays["ppl"]
    print(f"ppl: human mean {ppl[labels == 0].mean():.2f}, machine mean {ppl[labels == 1].mean():.2f}")
    print(f"PCA explained variance, last layer before the final norm: {arrays['pca_explained_variance_ratio'][-2]}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
