import torch
import os
import json
import time
import numpy as np
from argparse import Namespace
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data, return_device
from datetime import datetime
from argparse import ArgumentParser

from src.config import Config
cfg = Config()

# Leading tokens dropped from every score; an item is skipped (NaN) if this drops more than MAX_SKIP_SHARE of it.
SKIP = 10
MAX_SKIP_SHARE = 0.5
# Preceding tokens (intra) / layers (inter) mean-pooled for the context angle.
CONTEXT_K = 3

# Per-layer scores, ℓ = transformer layers 1..L (embedding output only as the inter reference of layer 1). With
# θ[ℓ, t] = ∠(h_{t−1}^ℓ, h_t^ℓ) (intra), φ[ℓ, t] = ∠(h_t^{ℓ−1}, h_t^ℓ) (inter) and n[ℓ, t] = ‖h_t^ℓ‖₂, over the
# kept tokens t ≥ SKIP:
#   intra_mean_angle:             mean_t θ
#   intra_context_angle:          mean_t ∠(mean(h_{t−3}^ℓ, h_{t−2}^ℓ, h_{t−1}^ℓ), h_t^ℓ)
#   intra_mean_norm:              mean_t n
#   intra_rel_norm:               mean_t (n[ℓ, t] / ‖h_t^0‖), each token's norm relative to its embedding norm (removes
#                                 the token-identity scale)
#   intra_angle_per_norm:         mean_t θ / mean_t n
#   intra_log_angle_per_log_norm: log mean_t θ / log mean_t n
#   intra_angle_per_geo_norm:     mean_t θ / R_ℓ,  R_ℓ = exp(mean_t log n), the geometric-mean norm
#   intra_angle_cv:               std_t θ / mean_t θ, the dispersion of the token steps
#   inter_mean_angle:             mean_t φ
#   inter_context_angle:          mean_t ∠(mean(h_t^{ℓ−3}, h_t^{ℓ−2}, h_t^{ℓ−1}), h_t^ℓ); NaN for ℓ < 3
#   inter_angle_cv:               std_t φ / mean_t φ
#   withinacross:                 mean_t (θ + φ)
#   joint:                        mean_t √(θ φ)
#   withinacross_rel:             mean_t (θ / θ̄¹ + φ / φ̄¹), θ̄¹ = mean_t θ[1, t], φ̄¹ = mean_t φ[1, t]: each angle
#                                 relative to the same text's layer-1 mean before combining
#   joint_rel:                    mean_t √((θ / θ̄¹) (φ / φ̄¹))
#   joint_context:                mean_t √(θᶜ φᶜ), θᶜ / φᶜ the intra / inter context angles above (3 preceding tokens /
#                                 layers mean-pooled), over the tokens with both (t ≥ SKIP + 3); NaN for ℓ < 3
#   full:                         log mean_t n + ½ (mean_t θ + mean_t φ)
#   full_log:                     log mean_t n − ½ (log mean_t θ + log mean_t φ) = log (n̄ / √(θ̄ φ̄)): norm per unit of
#                                 turning, both parts higher for machine text (larger norm, smaller angles)
# The combined scores use the tokens that have both angles (t ≥ SKIP + 1).
LAYER_SCORES = (
    "intra_mean_angle", "intra_context_angle", "intra_angle_cv",
    "intra_mean_norm", "intra_rel_norm",
    "intra_angle_per_norm", "intra_log_angle_per_log_norm", "intra_angle_per_geo_norm",
    "inter_mean_angle", "inter_context_angle", "inter_angle_cv",
    "withinacross", "joint", "withinacross_rel", "joint_rel", "joint_context", "full", "full_log",
)
# Inter scores pooled over the layers (one per text): per token over the transformer layers 1..L−1 (the last hidden
# state is already normed by HF, the logit input), then the mean over tokens:
#   inter_all_mean_angle:          mean_{t, ℓ} φ (steps between layers 1..L−1)
#   inter_all_context_angle:       mean_{t, ℓ} ∠(mean of the 3 previous layers, h_t^ℓ)
#   inter_all_mean_norm:           mean_{t, ℓ} n
#   inter_all_log_angle_per_geo_norm: log mean_{t, ℓ} φ / R_inter,  R_inter = exp(mean_{t, ℓ} log n)
POOLED_SCORES = (
    "inter_all_mean_angle", "inter_all_context_angle",
    "inter_all_mean_norm", "inter_all_log_angle_per_geo_norm",
)
# Orientation so that higher = machine: machine text has smaller angles and larger norms. Scores led by an angle are
# negated (−1), norm scores kept (+1); full_log is already norm over angle (+1). Ratios are formed on the unsigned
# values, then oriented the same way.
SIGNS = {name: (1 if "norm" in name and "angle" not in name else -1) for name in LAYER_SCORES + POOLED_SCORES}
SIGNS["full_log"] = 1


def angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between two (..., D) tensors along the last dimension."""
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
    return torch.acos(cosine)


class AngleNormScores():
    """Intra- and inter-layer angle and norm scores of the hidden states (see LAYER_SCORES and POOLED_SCORES)."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.device = return_device()
        self.inference = Inference(model_name=args.model)

    def scores(self, hidden_states: tuple[torch.Tensor, ...]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Per-layer scores {name: (L,)} and pooled inter scores {name: float}, unsigned; NaN if the item is skipped."""
        n_layers, n_tokens = len(hidden_states) - 1, hidden_states[0].shape[0]
        kept = n_tokens - SKIP
        if kept < (1 - MAX_SKIP_SHARE) * n_tokens or kept < CONTEXT_K + 2:
            return ({name: np.full(n_layers, np.nan) for name in LAYER_SCORES},
                    {name: float("nan") for name in POOLED_SCORES})

        h_all = torch.stack(hidden_states)[:, SKIP:].to(self.device).float()  # (L + 1, T', D), with the embeddings
        h = h_all[1:]  # (L, T', D)

        # Intra: along the tokens of each layer.
        theta = angle(h[:, :-1], h[:, 1:])  # (L, T' − 1), tokens 1..T'−1 of the kept ones
        token_pool = h.unfold(1, CONTEXT_K, 1).mean(dim=-1)  # row j = mean(h_j, ..., h_{j+K−1})
        intra_context = angle(token_pool[:, :-1], h[:, CONTEXT_K:])  # (L, T' − K)
        norm = h.norm(dim=-1).clamp_min(1e-12)  # (L, T')

        # Inter: along the layers of each token; layer 1 against the embedding output.
        phi = angle(h_all[:-1], h_all[1:])  # (L, T')
        layer_pool = h_all.unfold(0, CONTEXT_K, 1).mean(dim=-1)  # row j = mean of hidden_states j..j+K−1
        inter_context = angle(layer_pool[:-1], h_all[CONTEXT_K:])  # (L − K + 1, T'), transformer layers K..L
        # Inter context on all L layers (NaN below K), restricted to the tokens that have an intra context angle.
        inter_context_full = torch.cat([torch.full((CONTEXT_K - 1, kept), float("nan"), device=h.device),
                                        inter_context])[:, CONTEXT_K:]  # (L, T' − K), aligned with intra_context

        mean_theta, mean_norm = theta.mean(dim=1), norm.mean(dim=1)
        geo_norm = norm.log().mean(dim=1).exp()
        theta_c, phi_c = theta, phi[:, 1:]  # the tokens with both angles
        # Each angle relative to the same text's layer-1 mean over these tokens.
        theta_rel = theta_c / theta_c[0].mean().clamp_min(1e-12)
        phi_rel = phi_c / phi_c[0].mean().clamp_min(1e-12)
        per_layer = {
            "intra_mean_angle": mean_theta,
            "intra_context_angle": intra_context.mean(dim=1),
            "intra_mean_norm": mean_norm,
            "intra_rel_norm": (norm / h_all[0].norm(dim=-1).clamp_min(1e-12)).mean(dim=1),
            "intra_angle_per_norm": mean_theta / mean_norm,
            "intra_log_angle_per_log_norm": mean_theta.log() / mean_norm.log(),
            "intra_angle_per_geo_norm": mean_theta / geo_norm,
            "intra_angle_cv": theta.std(dim=1) / mean_theta,
            "inter_mean_angle": phi.mean(dim=1),
            "inter_context_angle": torch.cat([torch.full((CONTEXT_K - 1,), float("nan"), device=h.device),
                                              inter_context.mean(dim=1)]),
            "inter_angle_cv": phi.std(dim=1) / phi.mean(dim=1),
            "withinacross": (theta_c + phi_c).mean(dim=1),
            "joint": (theta_c * phi_c).sqrt().mean(dim=1),
            "withinacross_rel": (theta_rel + phi_rel).mean(dim=1),
            "joint_rel": (theta_rel * phi_rel).sqrt().mean(dim=1),
            "joint_context": (intra_context * inter_context_full).sqrt().mean(dim=1),
            "full": mean_norm.log() + 0.5 * (theta_c.mean(dim=1) + phi_c.mean(dim=1)),
            "full_log": mean_norm.log() - 0.5 * (theta_c.mean(dim=1).log() + phi_c.mean(dim=1).log()),
        }

        # Pooled inter over transformer layers 1..L−1.
        hp = h[:-1]
        phi_all = angle(hp[:-1], hp[1:])
        context_all = angle(hp.unfold(0, CONTEXT_K, 1).mean(dim=-1)[:-1], hp[CONTEXT_K:])
        norm_all = hp.norm(dim=-1).clamp_min(1e-12)
        pooled = {
            "inter_all_mean_angle": phi_all.mean(),
            "inter_all_context_angle": context_all.mean(),
            "inter_all_mean_norm": norm_all.mean(),
            "inter_all_log_angle_per_geo_norm": phi_all.mean().log() / norm_all.log().mean().exp(),
        }
        return ({name: value.cpu().numpy() for name, value in per_layer.items()},
                {name: value.item() for name, value in pooled.items()})

    @staticmethod
    def metrics_by_layer(labels: np.ndarray, scores: np.ndarray) -> dict[str, dict]:
        """Metrics per layer, keyed layer_{i} with i = 0..L−1 the transformer layer index (layer 1 = layer_0)."""
        metrics = {}
        for layer, layer_scores in enumerate(scores.T):
            valid = np.isfinite(layer_scores)
            if len(np.unique(labels[valid])) == 2 and np.nanstd(layer_scores) > 0:
                metrics[f"layer_{layer}"] = evaluation(labels[valid], layer_scores[valid])
        return metrics

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict | None:
        valid = np.isfinite(scores)
        return evaluation(labels[valid], scores[valid]) if len(np.unique(labels[valid])) == 2 else None

    def run(self, args: Namespace) -> dict[str, dict]:
        """Evaluate every score at every layer (raw and ratio to the first layer), and the pooled inter scores."""
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])
        per_layer_values = {name: [] for name in LAYER_SCORES}
        pooled_values = {name: [] for name in POOLED_SCORES}
        item_latencies_ms = []
        item_peak_memory_mb = []
        using_cuda = torch.cuda.is_available()

        for item in tqdm(test_data, desc="Collecting angle and norm scores"):
            if args.benchmark and using_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            if args.benchmark:
                start_time = time.perf_counter()

            per_layer, pooled = self.scores(self.inference.run(item, args)["hidden_states"])
            for name, value in per_layer.items():
                per_layer_values[name].append(value)
            for name, value in pooled.items():
                pooled_values[name].append(value)
            if args.benchmark:
                if using_cuda:
                    torch.cuda.synchronize()
                    item_peak_memory_mb.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
                item_latencies_ms.append((time.perf_counter() - start_time) * 1000)

        n_skipped = int(np.isnan(np.asarray(pooled_values[POOLED_SCORES[0]], dtype=float)).sum())
        metrics_by_score = {}
        for name, values in per_layer_values.items():
            raw = np.asarray(values, dtype=float)  # (N, L), unsigned
            # Ratio to layer 1, or to the first layer with a value (inter_context_angle starts at layer 3).
            reference = next(j for j in range(raw.shape[1]) if np.isfinite(raw[:, j]).any())
            ratio = np.divide(raw, raw[:, [reference]], out=np.full_like(raw, np.nan),
                              where=raw[:, [reference]] != 0)
            metrics_by_score[name] = {
                "ratio_reference_layer": f"layer_{reference}",
                "raw_metrics_by_layer": self.metrics_by_layer(labels, SIGNS[name] * raw),
                "ratio_metrics_by_layer": self.metrics_by_layer(labels, SIGNS[name] * ratio),
            }
        for name, values in pooled_values.items():
            metrics_by_score[name] = {"raw_metrics": self.evaluate(labels, SIGNS[name] * np.asarray(values, float))}

        print(f"\nangle_norm | model: {args.model} | dataset: {args.dataset} | seed: {args.seed} | "
              f"skipped items: {n_skipped}")
        print(f"{'score':>34}{'raw best (layer)':>20}{'ratio best (layer)':>22}")
        for name in LAYER_SCORES:
            cells = []
            for kind in ("raw", "ratio"):
                by_layer = metrics_by_score[name][f"{kind}_metrics_by_layer"]
                if by_layer:
                    layer, m = max(by_layer.items(), key=lambda kv: kv[1]["auroc"])
                    cells.append(f"{m['auroc']:.3f} ({int(layer.split('_')[1]) + 1})")
                else:
                    cells.append("--")
            print(f"{name:>34}{cells[0]:>20}{cells[1]:>22}")
        for name in POOLED_SCORES:
            m = metrics_by_score[name]["raw_metrics"]
            cell = f"{m['auroc']:.3f} (all)" if m else "--"
            print(f"{name:>34}{cell:>20}")

        file_name = f"angle_norm_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "angle_norm",
            "skip_tokens": SKIP,
            "max_skip_share": MAX_SKIP_SHARE,
            "context_k": CONTEXT_K,
            "n_skipped_items": n_skipped,
            "score_signs": SIGNS,
            "layer_numbering": "layer_i = transformer layer i + 1 (embeddings excluded)",
            "metrics_by_score": metrics_by_score,
        }
        if args.benchmark:
            output["mean_latency_per_item_ms"] = float(np.mean(item_latencies_ms))
            output["mean_peak_gpu_memory_per_item_mb"] = (
                float(np.mean(item_peak_memory_mb)) if item_peak_memory_mb else None
            )

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        if args.benchmark:
            print(f"Mean latency per item: {output['mean_latency_per_item_ms']:.2f} ms")
        if args.benchmark and output["mean_peak_gpu_memory_per_item_mb"] is not None:
            print(f"Mean peak GPU memory per item: {output['mean_peak_gpu_memory_per_item_mb']:.2f} MB")
        return metrics_by_score


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--benchmark",
        type=int,
        choices=(0, 1),
        default=0,
        help="Set to 1 to measure per-item latency and peak CUDA memory.",
    )
    args = parser.parse_args()
    args.benchmark = bool(args.benchmark)
    return args


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = AngleNormScores(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
