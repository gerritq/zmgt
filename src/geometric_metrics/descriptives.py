import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data
from src.geometric_metrics.metrics import GeometricMetrics, METRICS

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "metrics", "desc_stats")
# Every metric except length (unnormalized) and projection_error, plus magnitude_unit, the magnitude between the
# unit-norm states ‖h_t/‖h_t‖ − c_t/‖c_t‖‖₂, and hidden_norm, the mean ‖h‖₂ of the states involved. Per text,
# also entropy: the mean over the sequence's tokens of the next-token predictive entropy.
DESC_METRICS = tuple(metric for metric in METRICS if metric not in ("length", "projection_error"))
LAYER_STATS = DESC_METRICS + ("magnitude_unit", "hidden_norm")
# Two axes along which the metrics run, each saved to its own JSON (desc_stats_{axis}_{states}_...):
#   within_layer: per transformer layer, along the tokens, reference h_{t−1} at the same layer; mean over tokens,
#                 one value per layer (L).
#   across_layer: per token, along the transformer layers, reference h^{l−1} of the same token; mean over layers,
#                 one value per token (T), and also mean over tokens, one value per layer step.
AXES = ("within_layer", "across_layer")
# Both axes run on the raw and on the centred states, the mean subtracted along the axis the metrics run on:
#   within_layer: per layer over tokens, h̃^l_t = h^l_t − μ^l, μ^l = (1/T) Σ_t h^l_t. The shift is the same for every
#                 token of a layer, so h̃_t − h̃_{t−1} = h_t − h_{t−1}: magnitude and curvature are unchanged; angle,
#                 magnitude_unit and hidden_norm become relative to the text's centroid at that layer.
#   across_layer: per token over layers, h̃^l_t = h^l_t − μ_t, μ_t = (1/L) Σ_l h^l_t. The shift is the same for every
#                 layer of a token, so h̃^l_t − h̃^{l−1}_t = h^l_t − h^{l−1}_t: magnitude and curvature are unchanged;
#                 angle, magnitude_unit and hidden_norm become relative to the token's centroid over the layers.
STATES = ("raw", "centred")


def magnitude_unit(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between the unit-norm hidden state and reference: (S, ..., D) -> (S, ...)."""
    unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (unit(current) - unit(reference)).norm(dim=-1)


def mean_entropy(logits: torch.Tensor) -> float:
    """Mean over tokens of the next-token predictive entropy (nats): (T, V) -> float."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return (-(log_probs.exp() * log_probs).sum(dim=-1)).mean().item()


def metrics_along_first_axis(h: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every stat along dim 0 against the previous element: (S, ..., D) -> {stat: (S', ...)}."""
    current, previous = GeometricMetrics.reference(h, "previous")
    out = {metric: getattr(GeometricMetrics, metric)(current, previous) for metric in DESC_METRICS}
    out["magnitude_unit"] = magnitude_unit(current, previous)
    out["hidden_norm"] = h.norm(dim=-1)
    return out


def within_layer(hidden_states: torch.Tensor) -> dict[str, np.ndarray]:
    """Mean over tokens per transformer layer: (L, T, D) -> {stat: (L,)}; NaN if the text is too short."""
    out = {stat: [] for stat in LAYER_STATS}
    for h in hidden_states:
        for stat, values in metrics_along_first_axis(h).items():
            out[stat].append(values.mean().item() if values.numel() else float("nan"))
    return {stat: np.asarray(v, dtype=float) for stat, v in out.items()}


def across_layer(hidden_states: torch.Tensor) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Metrics along the layers for every token at once: (L, T, D) -> {stat: (L', T)}, summarized as the mean over layers
    per token, {stat: (T,)}, and the mean over tokens per layer step, {stat: (L',)}.
    """
    values = {stat: v.cpu().numpy() for stat, v in metrics_along_first_axis(hidden_states).items()}
    return (
        {stat: v.mean(axis=0) for stat, v in values.items()},
        {stat: v.mean(axis=1) for stat, v in values.items()},
    )


def centre(hidden_states: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Subtract the mean along the axis: within_layer, over tokens at every layer; across_layer, over layers for every
    token: (L, T, D) -> (L, T, D).
    """
    dim = 1 if axis == "within_layer" else 0
    return hidden_states - hidden_states.mean(dim=dim, keepdim=True)


def to_json(values: np.ndarray) -> list[float | None]:
    """NaN (text too short) as null, so the output is valid JSON."""
    return [float(x) if np.isfinite(x) else None for x in values]


def run(args: Namespace) -> dict[tuple[str, str], list[dict]]:
    inference = Inference(model_name=args.model)
    args.return_logits = True
    test_data = load_data(args=args)["test"]

    # Raw values per text; group summaries (mean, std, ...) are computed downstream from these.
    items = {(states, axis): [] for states in STATES for axis in AXES}
    for item in tqdm(test_data, desc="Collecting descriptive statistics"):
        out = inference.run(item, args)
        # Omit the embedding output and use transformer layers: (L, T, D).
        hidden_states = torch.stack(out["hidden_states"][1:]).float()
        meta = {
            "label": int(item["label"]), "chars": len(item["text"]), "tokens": int(hidden_states.shape[1]),
            "entropy": mean_entropy(out["logits"]),
        }
        for states in STATES:
            h = hidden_states if states == "raw" else centre(hidden_states, "within_layer")
            items[(states, "within_layer")].append({
                **meta, **{stat: to_json(v) for stat, v in within_layer(h).items()},
            })
            h = hidden_states if states == "raw" else centre(hidden_states, "across_layer")
            by_token, by_layer = across_layer(h)
            items[(states, "across_layer")].append({
                **meta,
                "by_token": {stat: to_json(v) for stat, v in by_token.items()},
                "by_layer": {stat: to_json(v) for stat, v in by_layer.items()},
            })

    descriptions = {
        "within_layer": {
            "reference": "previous token at the same layer",
            "values": "per stat, the mean over tokens at each transformer layer 1..L (embeddings excluded)",
        },
        "across_layer": {
            "reference": "same token at the previous transformer layer (embeddings excluded)",
            "values": (
                "by_token: per stat, the mean over layers at each token position 0..T−1; "
                "by_layer: per stat, the mean over tokens at each layer step, entry i starting at layer i + 1 "
                "(curvature: layers i + 1..i + 3, other metrics: i + 1 vs. i + 2, hidden_norm: layer i + 1)"
            ),
        },
    }
    state_descriptions = {
        ("raw", "within_layer"): "hidden states as is",
        ("raw", "across_layer"): "hidden states as is",
        ("centred", "within_layer"): "hidden states minus the text's mean state over its tokens at every layer",
        ("centred", "across_layer"): "hidden states minus the token's mean state over the transformer layers",
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for (states, axis), axis_items in items.items():
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "axis": axis,
            "states": state_descriptions[(states, axis)],
            **descriptions[axis],
            "stats": LAYER_STATS,
            "items": axis_items,
        }
        file_name = f"desc_stats_{axis}_{states}_{args.model_name}_{args.dataset}_s{args.seed}"
        with open(os.path.join(OUTPUT_DIR, f"{file_name}.json"), "w") as f:
            json.dump(output, f)
    return items


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    run(args)


if __name__ == "__main__":
    main()
