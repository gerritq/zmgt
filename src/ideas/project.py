import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data

from src.config import Config
cfg = Config()

# Reference each token is projected on: its predecessor, or the mean of its k predecessors (within_layer: --context_k;
# across_layers: CONTEXT_WINDOW).
REFERENCES = ("single", "context")
CONTEXT_WINDOW = 3
# angle: ∠(h_t, r_t); residual: ‖h_t − proj_{r_t}(h_t)‖, the part of h_t the reference does not explain;
# rel_error: ‖h_t − r_t‖ / ‖r_t‖, the error of predicting "nothing changes";
# log_polar: √(ln²ρ_t + θ_t²) with ρ_t = ‖h_t‖ / ‖r_t‖, growth and shrinkage weighted symmetrically.
MEASURES = ("angle", "residual", "rel_error", "log_polar")
# Sign turning each measure into a score where higher = more machine-like: a smaller angle, rel_error and log_polar
# (less change), but a larger residual.
SIGNS = {"angle": -1, "residual": 1, "rel_error": -1, "log_polar": -1, "residual_burst": -1, "residual_energy": 1}
# within_layer only: residual_burst, the share of tokens whose residual is more than BURST_Z standard deviations above
# the text's mean residual at that layer (level removed, burstiness-style); fewer bursts = more machine-like.
BURST_Z = 2.0
# within_layer only: residual_energy = Σ_t e_t² / (Σ_t ‖h_t‖² + ε), the share of the text's energy its predecessors do
# not explain; same sign as the residual.
WITHIN_MEASURES = MEASURES + ("residual_burst", "residual_energy")
# within_layer score variants, all with the measure's sign:
#   raw:         text mean at layer ℓ
#   first_ratio: divided by the text mean at the first transformer layer (as the curvature_hs score of
#                2_curvature_tweaked); the first layer's first_ratio is always 1
#   text_norm:   divided by the text's own scale at layer ℓ, mean_t ‖h_t‖ over the scored tokens
VARIANTS = ("raw", "first_ratio", "text_norm")
# --variant: within_layer projects on the preceding tokens at the same layer; across_layers projects each token on
# its own states at the preceding layers.
AXES = ("within_layer", "across_layers")


class ProjectionOnPrevious():
    """
    Per transformer layer ℓ, project each token's raw hidden state h_t on a reference built from its predecessors:
        single:  r_t = h_{t−1}                                   for t = 1..T−1
        context: r_t = mean(h_{t−k}, ..., h_{t−1})              for t = k..T−1, k = --context_k (default 3)
    With u_t = r_t / ‖r_t‖ and the projection p_t = (h_tᵀu_t) u_t:
        angle:    θ_t = ∠(h_t, r_t)
        residual: ‖h_t − p_t‖ = ‖h_t‖ sin θ_t
    and, as one scalar for both rotation and magnitude change (ρ_t = ‖h_t‖ / ‖r_t‖):
        rel_error: ‖h_t − r_t‖ / ‖r_t‖   (0 only without rotation and magnitude change; doubling scores 1, halving 0.5)
        log_polar: √(ln²ρ_t + θ_t²)       (growth and shrinkage symmetric; log-ratio and radians are dimensionless)
    Every measure also as first_ratio: the text mean at layer ℓ divided by the text mean at layer 1; and as
    text_norm: divided by the text's typical state norm at layer ℓ, mean_t ‖h_t‖ over the scored tokens (e.g. the
    residual relative to the text's own scale).
    residual_burst (within_layer only): the share of tokens t whose residual has a z-score (over the text's tokens at
    that layer) above BURST_Z: a burstiness feature with the residual's level removed.
    residual_energy (within_layer only): Σ_t e_t² / (Σ_t ‖h_t‖² + ε), with e_t the residual of token t.
    Text score per layer: mean over tokens, negated for the angle (the more a token continues its predecessors, the
    more machine-like) but not for the residual (a larger unexplained part is more machine-like); see SIGNS.

    --variant across_layers: the same projection and measures per token along the layers instead of the tokens,
    for transformer layers ℓ (embeddings excluded):
        single:  r_t^ℓ = h_t^{ℓ−1}
        context: r_t^ℓ = mean(h_t^{ℓ−3}, h_t^{ℓ−2}, h_t^{ℓ−1})
    Each measure is averaged over the layers of a token, then over the tokens of the text: one score per text.
    *_norm: each token's layer-averaged measure divided by the token's average hidden-state norm over the layers,
    mean_ℓ ‖h_t^ℓ‖, before averaging over the tokens.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)

    @staticmethod
    def reference(hidden_states: torch.Tensor, reference: str,
                  window: int = CONTEXT_WINDOW) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Current states and their references along the first axis (tokens, or layers for across_layers):
        (N, ..., D) -> (N', ..., D), (N', ..., D).
        """
        if reference == "single":
            return hidden_states[1:], hidden_states[:-1]
        # Row j of the pooled states = mean(h_j, ..., h_{j+window−1}), the reference of token j + window.
        pooled = hidden_states.unfold(0, window, 1).mean(dim=-1)
        return hidden_states[window:], pooled[:-1]

    @staticmethod
    def project(current: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every measure of the current states against their references, per token: {measure: (N,)}."""
        reference_norm = reference.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        current_norm = current.norm(dim=-1).clamp_min(1e-12)
        u = reference / reference_norm
        coefficient = (current * u).sum(dim=-1, keepdim=True)
        residual = (current - coefficient * u).norm(dim=-1)
        angle = torch.acos((coefficient.squeeze(-1) / current_norm).clamp(-1.0, 1.0))
        log_ratio = torch.log(current_norm / reference_norm.squeeze(-1))
        return {
            "angle": angle,
            "residual": residual,
            "rel_error": (current - reference).norm(dim=-1) / reference_norm.squeeze(-1),
            "log_polar": (log_ratio.pow(2) + angle.pow(2)).sqrt(),
        }

    @staticmethod
    def burst(residual: torch.Tensor) -> float:
        """Share of tokens with a residual z-score above BURST_Z within the text (level and scale removed)."""
        z = (residual - residual.mean()) / residual.std().clamp_min(1e-8)
        return (z > BURST_Z).float().mean().item()

    @staticmethod
    def energy(residual: torch.Tensor, current: torch.Tensor) -> float:
        """Σ_t e_t² / (Σ_t ‖h_t‖² + ε) over the scored tokens."""
        return (residual.pow(2).sum() / (current.pow(2).sum() + 1e-8)).item()

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, dict[str, np.ndarray]]:
        """
        Mean over tokens per reference, measure and transformer layer, plus the text's scale ("scale": mean ‖h_t‖
        over the scored tokens): {reference: {measure / "scale": (L,)}}.
        """
        # Omit the embedding output and use transformer layers.
        layers = hidden_states[1:]
        out = {reference: {name: [] for name in WITHIN_MEASURES + ("scale",)} for reference in REFERENCES}
        for layer in layers:
            h = layer.float()
            for reference in REFERENCES:
                min_tokens = 2 if reference == "single" else self.args.context_k + 1
                if h.shape[0] < min_tokens:
                    for name in WITHIN_MEASURES + ("scale",):
                        out[reference][name].append(float("nan"))
                    continue
                current, previous = self.reference(h, reference, self.args.context_k)
                measures = self.project(current, previous)
                for measure, values in measures.items():
                    out[reference][measure].append(values.mean().item())
                out[reference]["residual_burst"].append(self.burst(measures["residual"]))
                out[reference]["residual_energy"].append(self.energy(measures["residual"], current))
                # Over the scored tokens, which leaves out the first token (its attention-sink norm is an outlier).
                out[reference]["scale"].append(current.norm(dim=-1).mean().item())
        return {
            reference: {measure: np.asarray(v) for measure, v in by_measure.items()}
            for reference, by_measure in out.items()
        }

    def score_across_layers(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, dict[str, float]]:
        """
        Mean over layers, then tokens, per reference and measure, plainly and with each token's value divided by
        its average norm over the layers: {reference: {measure / measure_norm: float}}.
        """
        # Omit the embedding output and stack the transformer layers: (L, T, D).
        h = torch.stack(hidden_states[1:]).float()
        token_norm = h.norm(dim=-1).mean(dim=0).clamp_min(1e-12)  # (T,)
        out = {}
        for reference in REFERENCES:
            measures = self.project(*self.reference(h, reference))  # {measure: (L', T)}
            out[reference] = {}
            for measure, v in measures.items():
                per_token = v.mean(dim=0)  # (T,)
                out[reference][measure] = per_token.mean().item()
                out[reference][f"{measure}_norm"] = (per_token / token_norm).mean().item()
        return out

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (too-short texts give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    @staticmethod
    def key(reference: str, measure: str, variant: str) -> str:
        """JSON key suffix; raw keys keep their names, the others get a "_{variant}" suffix."""
        return f"{reference}_{measure}" + ("" if variant == "raw" else f"_{variant}")

    def plot(self, labels: np.ndarray, values: dict[str, np.ndarray], aurocs: dict[str, list[float]],
             path: str) -> None:
        """
        Row per measure, column per reference and variant (raw, first_ratio, text_norm): per-layer AUROC (left axis) and the
        per-group mean ± 1 standard deviation across texts (right axis, transparent).
        """
        colors = {"human": "#1f77b4", "machine": "#d62728"}
        units = {"angle": "rad", "residual": "norm", "rel_error": "", "log_polar": "", "residual_burst": "share",
                 "residual_energy": "share"}
        columns = [(reference, variant) for reference in REFERENCES for variant in VARIANTS]
        fig, axes = plt.subplots(len(WITHIN_MEASURES), len(columns),
                                 figsize=(5 * len(columns), 4 * len(WITHIN_MEASURES)), squeeze=False)
        for column, (reference, variant) in enumerate(columns):
            for row, measure in enumerate(WITHIN_MEASURES):
                ax = axes[row][column]
                key = self.key(reference, measure, variant)
                v = values[key]  # (N, L)
                layers = np.arange(1, v.shape[1] + 1)
                ax.plot(layers, aurocs[key], color="black", marker="o", markersize=3, label="AUROC")
                ax.axhline(0.5, color="grey", linewidth=0.8)
                ax.set_xlabel("Layer")
                ax.set_ylabel("AUROC")
                ax_mean = ax.twinx()
                for label, group in ((0, "human"), (1, "machine")):
                    mean = np.nanmean(v[labels == label], axis=0)
                    std = np.nanstd(v[labels == label], axis=0)
                    ax_mean.plot(layers, mean, color=colors[group], alpha=0.3, linewidth=2,
                                 label=f"mean ± 1 SD {group}")
                    ax_mean.fill_between(layers, mean - std, mean + std, color=colors[group], alpha=0.1, linewidth=0)
                if variant == "raw":
                    ax_mean.set_ylabel(f"mean {measure}" + (f" ({units[measure]})" if units[measure] else ""))
                elif variant == "first_ratio":
                    ax_mean.set_ylabel(f"mean {measure} / layer 1")
                else:
                    ax_mean.set_ylabel(f"mean {measure} / text mean ‖h‖")
                lines, mean_lines = ax.get_legend_handles_labels(), ax_mean.get_legend_handles_labels()
                ax.legend(lines[0] + mean_lines[0], lines[1] + mean_lines[1], frameon=False, fontsize=7)
                title = ("previous token" if reference == "single"
                         else f"mean of previous {self.args.context_k} tokens")
                ax.set_title(f"{measure} ({variant}), projected on\n{title}", fontsize=9)
        fig.suptitle(f"{self.args.model}, {self.args.dataset}: projection on the predecessors", fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_hist(self, labels: np.ndarray, values: dict[str, np.ndarray], aurocs: dict[str, float],
                  path: str, suffix: str = "") -> None:
        """
        Row per measure, column per reference: human vs machine histograms, each group's mean dashed. suffix "_norm"
        plots the scores divided by each token's average norm.
        """
        colors = {"human": "#1f77b4", "machine": "#d62728"}
        fig, axes = plt.subplots(len(MEASURES), len(REFERENCES), figsize=(6 * len(REFERENCES), 3.5 * len(MEASURES)),
                                 squeeze=False)
        for column, reference in enumerate(REFERENCES):
            for row, measure in enumerate(MEASURES):
                ax = axes[row][column]
                key = f"{reference}_{measure}{suffix}"
                v = values[key]
                mask = np.isfinite(v)
                bins = np.histogram_bin_edges(v[mask], bins=40)
                for label, group in ((0, "human"), (1, "machine")):
                    group_values = v[mask & (labels == label)]
                    ax.hist(group_values, bins=bins, alpha=0.5, density=True, color=colors[group],
                            label=f"{group.capitalize()} (n={len(group_values)})")
                    ax.axvline(group_values.mean(), color=colors[group], linestyle="--", linewidth=1)
                title = "previous layer" if reference == "single" else f"mean of previous {CONTEXT_WINDOW} layers"
                ax.set_xlabel(f"mean {measure} across layers and tokens"
                              + (" / token's mean norm" if suffix else ""))
                ax.set_ylabel("Density")
                ax.set_title(f"{measure}, projected on {title}: AUROC={aurocs[key]:.3f}", fontsize=9)
                ax.legend(frameon=False, fontsize=7)
        fig.suptitle(f"{self.args.model}, {self.args.dataset}: projection across layers"
                     + (", divided by each token's mean norm" if suffix else ""), fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def run_across_layers(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {
            f"{reference}_{measure}{suffix}": []
            for reference in REFERENCES for measure in MEASURES for suffix in ("", "_norm")
        }
        for item in tqdm(test_data, desc="Collecting projection measures across layers"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            for reference, by_measure in self.score_across_layers(hidden_states).items():
                for measure, value in by_measure.items():
                    per_text[f"{reference}_{measure}"].append(value)
        values = {key: np.asarray(v, dtype=float) for key, v in per_text.items()}  # (N,)
        # Key "{reference}_{measure}[_norm]": the measure fixes the sign.
        scores = {key: SIGNS[key.split("_", 1)[1].removesuffix("_norm")] * v for key, v in values.items()}
        metrics = {key: self.evaluate(labels, s) for key, s in scores.items()}
        aurocs = {key: m["auroc"] for key, m in metrics.items()}
        for key, a in aurocs.items():
            print(f"AUROC {key}: {a:.4f}")

        method = "project_across_layers"
        file_name = f"{method}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "context_window": CONTEXT_WINDOW,
            "score_signs": SIGNS,
            **{f"metrics_{key}": m for key, m in metrics.items()},
            **{f"mean_{key}": {"human": float(np.nanmean(v[labels == 0])), "machine": float(np.nanmean(v[labels == 1]))}
               for key, v in values.items()},
            **{f"scores_{key}": s.tolist() for key, s in scores.items()},
            "labels": labels.tolist(),
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot_hist(labels, values, aurocs, os.path.join(output_dir, f"{file_name}.pdf"))
        self.plot_hist(labels, values, aurocs, os.path.join(output_dir, f"{file_name}_norm.pdf"), suffix="_norm")
        return aurocs

    def run(self, args: Namespace) -> dict:
        if args.variant == "across_layers":
            return self.run_across_layers(args)
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {reference: {name: [] for name in WITHIN_MEASURES + ("scale",)} for reference in REFERENCES}
        for item in tqdm(test_data, desc="Collecting projection angles / residuals"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            for reference, by_measure in self.score(hidden_states).items():
                for measure, v in by_measure.items():
                    per_text[reference][measure].append(v)
        # {key: (N, L)} for every reference, measure and variant.
        values = {}
        for reference, by_name in per_text.items():
            scale = np.asarray(by_name["scale"], dtype=float)  # (N, L)
            for measure in WITHIN_MEASURES:
                v = np.asarray(by_name[measure], dtype=float)
                values[self.key(reference, measure, "raw")] = v
                values[self.key(reference, measure, "first_ratio")] = v / v[:, :1]
                values[self.key(reference, measure, "text_norm")] = v / scale
        signs = {self.key(r, m, variant): SIGNS[m] for r in REFERENCES for m in WITHIN_MEASURES for variant in VARIANTS}
        metrics_per_layer = {
            key: [self.evaluate(labels, signs[key] * v[:, layer]) for layer in range(v.shape[1])]
            for key, v in values.items()
        }
        aurocs = {key: [m["auroc"] for m in per_layer] for key, per_layer in metrics_per_layer.items()}
        for key, a in aurocs.items():
            print(f"AUROC {key}: best {np.nanmax(a):.4f} (layer {int(np.nanargmax(a)) + 1}), last layer {a[-1]:.4f}")

        method = "project"
        file_name = f"{method}_k{args.context_k}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "context_window": args.context_k,
            "score_signs": SIGNS,
            "variants": VARIANTS,
            "labels": labels.tolist(),
        }
        for key, v in values.items():
            output.update({
                f"auroc_per_layer_{key}": aurocs[key],
                f"metrics_per_layer_{key}": metrics_per_layer[key],
                f"mean_per_layer_{key}": {
                    "human": np.nanmean(v[labels == 0], axis=0).tolist(),
                    "machine": np.nanmean(v[labels == 1], axis=0).tolist(),
                },
                f"values_per_layer_{key}": v.tolist(),
            })

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot(labels, values, aurocs, os.path.join(output_dir, f"{file_name}.pdf"))
        return aurocs


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--variant", type=str, choices=AXES, default="within_layer",
                        help="within_layer: projection on the preceding tokens, per layer; across_layers: projection "
                             "on the token's preceding layers, one score per text.")
    parser.add_argument("--context_k", type=int, default=CONTEXT_WINDOW,
                        help="within_layer: number of preceding tokens averaged into the context reference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = ProjectionOnPrevious(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
