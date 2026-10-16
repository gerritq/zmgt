"""
Two experiments on how a layer moves the tokens (same data, models and token variants as aniso.py), per text and layer
ℓ = 1..L on the kept tokens of layer ℓ (layer 0 = null where undefined):
1. Logit lens: does the model make up its mind earlier on machine text?
    lens_entropy:           mean_t entropy of the logit lens at layer ℓ (also layer 0), logits = W_U norm(h_t^ℓ)
    settle_layer:           per text, mean_t of the first layer whose top guess equals the final prediction and stays
                            equal (tokens kept at every layer); settle_layer_rel = settle_layer / L
    correlations:           Spearman ρ over texts of settle_layer with inter_angle and shared_share at each layer
2. Attention vs MLP: which part of the layer pushes the tokens?
    {attn,mlp}_size:        mean_t ‖u_t‖, u_t = the attention / MLP update the layer adds to the residual stream
                            (h^ℓ = h^{ℓ−1} + u_attn + u_mlp; captured with forward hooks in the same pass)
    {attn,mlp}_sideways:    mean_t ‖u_t − ⟨u_t, ĥ_t⟩ ĥ_t‖, ĥ_t = h_t^{ℓ−1} / ‖h_t^{ℓ−1}‖, the part that turns the token
    {attn,mlp}_along_mu:    mean_t ⟨u_t, μ̂⟩, μ̂ = the unit centroid of h^{ℓ−1}, the part that grows the shared direction
Reference scores: inter_angle = mean_t φ_t, φ_t = ∠(h_t^{ℓ−1}, h_t^ℓ), and shared_share = ‖μ‖² / mean_t ‖h_t‖² (as in
aniso.py). Per variant and layer every score gets AUROC and Cohen's d (label 1 = machine).
Note: hidden_states[L] is the final-norm output (transformers ties it to last_hidden_state), so layer L scores built on
hidden states describe the normed state; the updates at L are taken before the norm.
"""

import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from scipy.stats import spearmanr
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data
from src.geometric_metrics.metrics import GeometricMetrics
from src.geometric_metrics.aniso import (
    MIN_TOKENS, VARIANTS, evaluate, filtered_layer_scores, to_json, unit, variant_mask,
)

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "metrics", "across_exp")
UPDATE_PARTS = ("attn", "mlp")
UPDATE_STATS = ("size", "sideways", "along_mu")
SCORE_DEFINITIONS = {
    "inter_angle": "mean_t angle(h_t^{l-1}, h_t^l); radians; null at layer 0",
    "shared_share": "||mu||^2 / mean_t ||h_t||^2 (as in aniso.py)",
    "lens_entropy": "mean_t entropy of softmax(W_U norm(h_t^l)); nats",
    **{f"{part}_size": f"mean_t ||u_t||, u = {part} update to the residual stream; null at layer 0"
       for part in UPDATE_PARTS},
    **{f"{part}_sideways": f"mean_t ||u_t - <u_t, h_hat_t> h_hat_t||, h_hat = h^{{l-1}} / ||h^{{l-1}}||, u = {part} update"
       for part in UPDATE_PARTS},
    **{f"{part}_along_mu": f"mean_t <u_t, mu_hat>, mu_hat = unit centroid of h^{{l-1}}, u = {part} update"
       for part in UPDATE_PARTS},
}
TEXT_DEFINITIONS = {
    "settle_layer": "mean_t first layer from which the logit-lens top guess equals the final prediction",
    "settle_layer_rel": "settle_layer / L",
}
CORRELATE_WITH = ("inter_angle", "shared_share")


def update_modules(layer: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Modules whose outputs are the attention and MLP updates added to the residual stream. Gemma 3 normalizes both
    updates before adding them (post_attention_layernorm, post_feedforward_layernorm); Llama, Qwen and Mistral add the
    raw self_attn and mlp outputs (their post_attention_layernorm is the pre-MLP norm).
    """
    if hasattr(layer, "post_feedforward_layernorm"):
        return layer.post_attention_layernorm, layer.post_feedforward_layernorm
    return layer.self_attn, layer.mlp


class UpdateRecorder:
    """Forward hooks on every decoder layer that keep the attention and MLP updates of the last forward pass."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.layers = model.get_decoder().layers
        self.updates = {part: [] for part in UPDATE_PARTS}
        for layer in self.layers:
            for part, module in zip(UPDATE_PARTS, update_modules(layer)):
                module.register_forward_hook(self.hook(part))

    def hook(self, part: str):
        def record(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.updates[part].append(out.detach().squeeze(0).float())
        return record

    def pop(self) -> dict[str, torch.Tensor]:
        """{part: (L, T, D)} for the last forward pass, then clear."""
        out = {part: torch.stack(v) for part, v in self.updates.items()}
        assert all(len(v) == len(self.layers) for v in out.values()), "one update per layer and part expected"
        self.updates = {part: [] for part in UPDATE_PARTS}
        return out


class LogitLens:
    """Per layer, the next-token guess W_U norm(h) (final-logit softcapping applied where the model uses it)."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.norm = model.get_decoder().norm
        self.unembed = model.get_output_embeddings()
        self.softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    @torch.no_grad()
    def __call__(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        (L + 1, T, D) -> top guess (L + 1, T) and entropy (L + 1, T). hidden_states[L] is already the final-norm output,
        so the norm is applied to layers 0..L − 1 only.
        """
        dtype = self.unembed.weight.dtype
        top, entropy = [], []
        for layer in range(h.shape[0]):
            x = h[layer].to(dtype)
            if layer < h.shape[0] - 1:
                x = self.norm(x)
            logits = self.unembed(x).float()
            if self.softcap:
                logits = torch.tanh(logits / self.softcap) * self.softcap
            log_p = torch.log_softmax(logits, dim=-1)
            top.append(logits.argmax(dim=-1))
            entropy.append(-(log_p.exp() * log_p).sum(dim=-1))
        return torch.stack(top), torch.stack(entropy)


def settle_layers(top: torch.Tensor) -> torch.Tensor:
    """(L + 1, T) top guesses -> (T,) first layer from which every guess equals the final one."""
    match = (top == top[-1]).int()
    trailing = match.flip(0).cumprod(dim=0).sum(dim=0)  # matching layers counted back from the last
    return (top.shape[0] - trailing).float()


def masked_mean(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """(L, T) -> (L,) mean over the kept tokens of each layer; NaN where fewer than MIN_TOKENS are kept."""
    count = keep.sum(dim=1)
    mean = (x * keep).sum(dim=1) / count.clamp_min(1)
    return torch.where(count >= MIN_TOKENS, mean, torch.full_like(mean, float("nan")))


def extra_scores(h: torch.Tensor, keep: torch.Tensor, updates: dict[str, torch.Tensor],
                 entropy: torch.Tensor) -> dict[str, torch.Tensor]:
    """inter_angle, lens_entropy and the update scores per layer (L + 1,), layer 0 NaN where undefined."""
    nan = torch.full((1,), float("nan"), device=h.device)
    previous, current, kept = h[:-1], h[1:], keep[1:]  # layer ℓ = 1..L scored on its kept tokens
    out = {
        "inter_angle": torch.cat([nan, masked_mean(GeometricMetrics.angle(current, previous), kept)]),
        "lens_entropy": masked_mean(entropy, keep),
    }
    h_hat = unit(previous)
    mu = (previous * kept.unsqueeze(-1)).sum(dim=1, keepdim=True) / kept.sum(dim=1).clamp_min(1)[:, None, None]
    mu_hat = unit(mu)
    for part, u in updates.items():
        radial = (u * h_hat).sum(dim=-1, keepdim=True)
        stats = {
            "size": u.norm(dim=-1),
            "sideways": (u - radial * h_hat).norm(dim=-1),
            "along_mu": (u * mu_hat).sum(dim=-1),
        }
        for stat, x in stats.items():
            out[f"{part}_{stat}"] = torch.cat([nan, masked_mean(x, kept)])
    return out


def correlations(items: list[dict], text_score: str, n_layers: int) -> dict[str, list[float | None]]:
    """Per layer, Spearman ρ over texts between a per-text score and each CORRELATE_WITH layer score."""
    x = np.asarray([np.nan if item[text_score] is None else item[text_score] for item in items], dtype=float)
    out = {}
    for score in CORRELATE_WITH:
        y = np.asarray([[np.nan if v is None else v for v in item[score]] for item in items], dtype=float)
        rho = []
        for layer in range(n_layers):
            mask = np.isfinite(x) & np.isfinite(y[:, layer])
            value = float(spearmanr(x[mask], y[mask, layer]).statistic) if mask.sum() > 2 else float("nan")
            rho.append(value if np.isfinite(value) else None)  # constant input gives NaN
        out[score] = rho
    return out


def run(args: Namespace) -> dict:
    inference = Inference(model_name=args.model)
    recorder = UpdateRecorder(inference.model)
    lens = LogitLens(inference.model)
    test_data = load_data(args=args)["test"]
    items = {variant: [] for variant in VARIANTS}
    n_layers = None
    for item in tqdm(test_data, desc="Collecting across-layer scores"):
        hidden_states = inference.run(item, args)["hidden_states"]
        updates = recorder.pop()  # {part: (L, T, D)} from the same forward pass
        n_layers = len(hidden_states)
        h = torch.stack(hidden_states).to(inference.device).float()  # (L + 1, T, D)
        top, entropy = lens(h)
        settle = settle_layers(top)  # (T,)
        meta = {"label": int(item["label"]), "chars": len(item["text"]), "tokens": int(h.shape[1])}
        for variant in VARIANTS:
            keep = variant_mask(h, variant)
            scores = {**extra_scores(h, keep, updates, entropy),
                      "shared_share": filtered_layer_scores(h, keep)["shared_share"]}
            kept_everywhere = keep.all(dim=0)
            settle_layer = float(settle[kept_everywhere].mean()) if kept_everywhere.sum() >= MIN_TOKENS else None
            items[variant].append({
                **meta, "tokens_kept": keep.sum(dim=1).tolist(),
                "settle_layer": settle_layer,
                "settle_layer_rel": None if settle_layer is None else settle_layer / (n_layers - 1),
                **{score: to_json(scores[score].cpu().numpy()) for score in SCORE_DEFINITIONS},
            })

    variants = {}
    for variant, variant_items in items.items():
        labels = np.asarray([item["label"] for item in variant_items])
        metrics = {}
        for score in SCORE_DEFINITIONS:
            values = np.asarray([[np.nan if v is None else v for v in item[score]] for item in variant_items],
                                dtype=float)
            metrics[score] = [evaluate(labels, values[:, layer]) for layer in range(n_layers)]
        text_metrics = {
            score: evaluate(labels, np.asarray([np.nan if item[score] is None else item[score]
                                                for item in variant_items], dtype=float))
            for score in TEXT_DEFINITIONS
        }
        variants[variant] = {
            "definition": VARIANTS[variant], "metrics": metrics, "text_metrics": text_metrics,
            "correlations": {"settle_layer": correlations(variant_items, "settle_layer", n_layers)},
            "items": variant_items,
        }

    return {
        **vars(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "across_exp",
        "scores": list(SCORE_DEFINITIONS),
        "score_definitions": SCORE_DEFINITIONS,
        "text_scores": list(TEXT_DEFINITIONS),
        "text_score_definitions": TEXT_DEFINITIONS,
        "layer_numbering": "entry i is hidden_states[i]: 0 = embedding output; 1..L = transformer-layer outputs "
                           "(L = after the final norm)",
        "auroc_orientation": "score as written, label 1 = machine; AUROC below 0.5 means the score is lower for "
                             "machine text",
        "variants": variants,
    }


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    output = run(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"across_exp_{args.model}_{args.dataset}_s{args.seed}.json")
    with open(path, "w") as f:
        json.dump(output, f)
    print(f"Wrote {path}")
    for variant, entry in output["variants"].items():
        print(f"--- {variant}: {entry['definition']}")
        for score in output["scores"]:
            aurocs = [m["auroc"] for m in entry["metrics"][score]]
            best = max(range(len(aurocs)), key=lambda i: abs((aurocs[i] or 0.5) - 0.5))
            print(f"{score:>22}: most separating layer {best}, AUROC {aurocs[best]}")
        for score, m in entry["text_metrics"].items():
            print(f"{score:>22}: human {m['mean_human']}, machine {m['mean_machine']}, AUROC {m['auroc']}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
