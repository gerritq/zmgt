"""
All per-text statistics of aniso.py and desc.py (no PCA) at every hidden-state layer ℓ = 0..L (0 = embedding output),
for each token-selection variant (VARIANTS):
    all:   all tokens
    med3:  tokens with ‖h_t‖ > 3 · median_t ‖h_t‖ dropped (per text and layer)
    med5:  tokens with ‖h_t‖ > 5 · median_t ‖h_t‖ dropped (per text and layer)
Per text, layer and variant:
    the aniso.py scores (mean_norm, mean_angle, avg_cosine, ..., sideways_distance), on the kept tokens of the layer
    the desc.py scores:
        representation_dispersion, angle_within, magnitude_within:  on the kept tokens of the layer (pairs are
                                                                    neighbours among the kept tokens)
        angle_across, magnitude_across:                             on the tokens kept at both ℓ − 1 and ℓ (null at
                                                                    layer 0)
Texts with fewer than MIN_TOKENS kept tokens at a layer get NaN there. Per variant, score and layer: AUROC and Cohen's d
(label 1 = machine), as in aniso.py.
Per text, also ppl and the desc.py per-token arrays (norms, across/within parallel and perpendicular parts,
update_difference), on all tokens; the kept-token mask of each variant is stored so they can be filtered later.
Note: mean_angle = angle_within and representation_dispersion = 1 − avg_cosine; both are kept so each script's
names are there.
Everything goes to one .npz per model, dataset and seed, one row per text (see the "meta" entry for each array's shape);
texts shorter than the longest are padded with NaN (False for masks) along the token axis.
Note: hidden_states[L] is the final-norm output, so layer L describes the normed state.
"""

import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data
from src.geometric_metrics.metrics import GeometricMetrics
from src.geometric_metrics import aniso, desc

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "desc_all")
# Fewest kept tokens for a text to be scored at a layer (the pairwise scores need pairs).
MIN_TOKENS = 3
SCORE_DEFINITIONS = {**aniso.SCORE_DEFINITIONS, **desc.SCORE_DEFINITIONS}
SCORES = tuple(SCORE_DEFINITIONS)
DESC_WITHIN = ("representation_dispersion", "angle_within", "magnitude_within")
VARIANTS = {
    "all": "all tokens",
    "med3": "tokens with ||h_t|| > 3 * median_t ||h_t|| dropped (per text and layer)",
    "med5": "tokens with ||h_t|| > 5 * median_t ||h_t|| dropped (per text and layer)",
}
MEDIAN_FACTORS = {"med3": 3.0, "med5": 5.0}


def variant_mask(h: torch.Tensor, variant: str) -> torch.Tensor:
    """Tokens a variant keeps per layer: (L + 1, T, D) -> (L + 1, T) bool."""
    norms = h.norm(dim=-1)
    if variant in MEDIAN_FACTORS:
        return norms <= MEDIAN_FACTORS[variant] * norms.median(dim=1, keepdim=True).values
    return torch.ones_like(norms, dtype=torch.bool)


def desc_scores(h: torch.Tensor, keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """desc.layer_scores with each layer on its own kept tokens (across: kept at both layers); NaN below MIN_TOKENS."""
    if keep.all():
        return desc.layer_scores(h)
    nan = torch.full((1,), float("nan"), device=h.device)
    within = [desc.layer_scores(h[layer:layer + 1, keep[layer]]) if keep[layer].sum() >= MIN_TOKENS else None
              for layer in range(h.shape[0])]
    scores = {score: torch.cat([nan if s is None else s[score] for s in within]) for score in DESC_WITHIN}
    both = keep[1:] & keep[:-1]  # (L, T)
    count = both.sum(dim=1)
    upper, lower = h[1:], h[:-1]
    for score, values in (("angle_across", GeometricMetrics.angle(upper, lower)),
                          ("magnitude_across", GeometricMetrics.magnitude(upper, lower))):
        mean = torch.where(both, values, 0.0).sum(dim=1) / count.clamp_min(1)
        scores[score] = torch.cat([nan, torch.where(count >= MIN_TOKENS, mean, float("nan"))])
    return scores


def run(args: Namespace) -> dict[str, np.ndarray]:
    inference = Inference(model_name=args.model)
    args.return_token_ids = True
    test_data = load_data(args=args)["test"]

    args.return_logits = True
    items, per_token = [], []
    for item in tqdm(test_data, desc="Collecting desc_all statistics"):
        out = inference.run(item, args)
        h = torch.stack(out["hidden_states"]).to(inference.device).float()  # (L + 1, T, D)
        text = {"label": int(item["label"]), "chars": len(item["text"]), "tokens": int(h.shape[1]),
                "ppl": desc.perplexity(out["logits"], out["token_ids"])}
        token_arrays = desc.token_scores(h)  # {name: (L + 1, T)}
        for variant in VARIANTS:
            keep = variant_mask(h, variant)
            scores = {**aniso.filtered_layer_scores(h, keep), **desc_scores(h, keep)}
            token_arrays[f"keep__{variant}"] = keep.cpu().numpy()
            text.update({f"{score}__{variant}": scores[score].cpu().numpy() for score in SCORES})
        items.append(text)
        per_token.append(token_arrays)
    del inference
    torch.cuda.empty_cache()

    tokens = np.asarray([item["tokens"] for item in items])
    labels = np.asarray([item["label"] for item in items])
    n_layers = per_token[0]["norms"].shape[0]
    token_arrays = {name: np.full((len(items), n_layers, tokens.max()), False if name.startswith("keep__") else np.nan,
                                  dtype=bool if name.startswith("keep__") else np.float32)
                    for name in per_token[0]}
    for i, text in enumerate(per_token):
        for name, values in text.items():
            token_arrays[name][i, :, :tokens[i]] = values

    layer_arrays, evaluation = {}, {}
    for variant in VARIANTS:
        auroc = np.full((len(SCORES), n_layers), np.nan)
        cohens_d = np.full((len(SCORES), n_layers), np.nan)
        for s, score in enumerate(SCORES):
            values = np.stack([item[f"{score}__{variant}"] for item in items]).astype(np.float32)
            layer_arrays[f"{score}__{variant}"] = values
            for layer in range(n_layers):
                result = aniso.evaluate(labels, values[:, layer])
                auroc[s, layer] = np.nan if result["auroc"] is None else result["auroc"]
                cohens_d[s, layer] = np.nan if result["cohens_d"] is None else result["cohens_d"]
        evaluation[f"auroc__{variant}"] = auroc
        evaluation[f"cohens_d__{variant}"] = cohens_d

    meta = {
        **vars(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "desc_all",
        "scores": list(SCORES),
        "score_definitions": SCORE_DEFINITIONS,
        "variants": VARIANTS,
        "layer_numbering": "layer i is hidden_states[i]: 0 = embedding output; 1..L = transformer-layer outputs "
                           "(L = after the final norm)",
        "auroc_orientation": "score as written, label 1 = machine; AUROC below 0.5 means the score is lower for "
                             "machine text",
        "arrays": {
            "label, chars, tokens, ppl": "(texts,); ppl = exp(mean_t -log p(x_t | x_<t)), t = 1..T-1",
            "<score>__<variant>": "(texts, L + 1) float32, one per entry of scores and variants; NaN where undefined "
                                  "or fewer than MIN_TOKENS kept tokens",
            "auroc__<variant>, cohens_d__<variant>": "(len(scores), L + 1), rows in the order of scores; NaN where "
                                                     "undefined",
            "keep__<variant>": "(texts, L + 1, max T) bool, tokens the variant keeps; False after the last token",
            "norms": "(texts, L + 1, max T) float32, ||h_t|| per token; NaN after the text's last token",
            "across_parallel, across_perpendicular": "(texts, L + 1, max T) float32, delta = h_t^l - h_t^{l-1} "
                                                     "split along h_t^{l-1}: signed length delta . h / ||h|| and "
                                                     "||delta - parallel part||; NaN at layer 0 and after the last token",
            "within_parallel, within_perpendicular": "(texts, L + 1, max T) float32, delta = h_t^l - h_{t-1}^l "
                                                     "split along h_{t-1}^l, as above; NaN at position 0 and after the "
                                                     "last token",
            "update_difference": "(texts, L + 1, max T) float32, ||D|| with D = Delta_t - Delta_{t-1}, Delta_t = "
                                 "h_t^l - h_t^{l-1}; NaN at layer 0, position 0 and after the last token",
        },
        "per_token_arrays_use": "all tokens; filter with keep__<variant>",
    }
    return {
        "meta": np.asarray(json.dumps(meta)),
        "label": labels,
        "chars": np.asarray([item["chars"] for item in items]),
        "tokens": tokens,
        "ppl": np.asarray([item["ppl"] for item in items]),
        **layer_arrays,
        **evaluation,
        **token_arrays,
    }


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    arrays = run(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"desc_all_{args.model}_{args.dataset}_s{args.seed}.npz")
    np.savez_compressed(path, **arrays)
    print(f"Wrote {path}")
    labels, ppl = arrays["label"], arrays["ppl"]
    print(f"ppl: human mean {ppl[labels == 0].mean():.2f}, machine mean {ppl[labels == 1].mean():.2f}")
    for variant, definition in VARIANTS.items():
        print(f"--- {variant}: {definition}")
        auroc = arrays[f"auroc__{variant}"]
        for s, score in enumerate(SCORES):
            if np.isnan(auroc[s]).all():
                print(f"{score:>26}: no valid layer")
                continue
            best = int(np.nanargmax(np.abs(auroc[s] - 0.5)))
            print(f"{score:>26}: most separating layer {best}, AUROC {auroc[s, best]:.3f}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
