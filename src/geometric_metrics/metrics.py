import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from datetime import datetime
from scipy.stats import spearmanr
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data

from src.config import Config
cfg = Config()

# All metrics run along the tokens of one transformer layer, comparing the hidden state h_t of token t at that layer
# with a backward-looking reference c_t:
#   previous: c_t = h_{t−1}                                for t = 1..T−1
#   context:  c_t = mean(h_{t−k}, ..., h_{t−1}), k = CONTEXT_K   for t = k..T−1
# Metrics per token:
#   curvature:        ∠(v_t, v_{t+1}) with the transition vectors v_t = h_t − c_t (h_t − h_{t−1} for previous)
#   angle:            ∠(c_t, h_t)
#   magnitude:        ‖h_t − c_t‖₂
#   length:           ‖h_t‖₂ / ‖c_t‖₂
#   projection_error: e_t = ‖r_t‖₂ / ‖h_t‖₂ with r_t = h_t − proj_{c_t}(h_t), i.e. sin ∠(h_t, c_t) ∈ [0, 1]
METRICS = ("curvature", "angle", "magnitude", "length", "projection_error")
REFERENCES = ("previous", "context")
CONTEXT_K = 3
REFERENCE_TITLES = {
    "previous": "Against the previous token: c_t = h_{t−1}",
    "context": f"Against the mean of the previous {CONTEXT_K} tokens: c_t = mean(h_{{t−{CONTEXT_K}}}, ..., h_{{t−1}})",
}
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "metrics", "desc")
# --norm, the normalization of each text's per-layer mean:
#   none:                as is
#   layer_first_to_last: divided by a text-level scale at that layer (h_0 first, h_{T−1} last token); curvature as is
#       angle:            ∠(h_0, h_{T−1})
#       magnitude:        ‖h_{T−1} − h_0‖₂
#       length:           mean_t ‖h_t‖₂
#       projection_error: ‖h_{T−1} − proj_{h_0}(h_{T−1})‖₂ / ‖h_{T−1}‖₂
#   first_layer:         divided by the text's value at the first transformer layer, every metric (layer 1 is always 1)
# none goes to OUTPUT_DIR, the others to OUTPUT_DIR/norm with the normalization in the file name.
NORMS = ("none", "layer_first_to_last", "first_layer")
NORMALIZED = ("angle", "magnitude", "length", "projection_error")


class GeometricMetrics():

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)

    @staticmethod
    def _angles(previous: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
        """Row-wise angle between two (N, D) tensors."""
        denominator = (previous.norm(dim=-1) * following.norm(dim=-1)).clamp_min(1e-12)
        cosine = ((previous * following).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    @staticmethod
    def reference(hidden_states: torch.Tensor, reference: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Current states h_t and their references c_t: (T, D) -> (T', D), (T', D)."""
        if reference == "previous":
            return hidden_states[1:], hidden_states[:-1]
        # Row j of the pooled states = mean(h_j, ..., h_{j+k−1}), the reference of token j + k.
        pooled = hidden_states.unfold(0, CONTEXT_K, 1).mean(dim=-1)
        return hidden_states[CONTEXT_K:], pooled[:-1]

    @staticmethod
    def curvature(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Angle between consecutive transition vectors v_t = h_t − c_t: (T', D) -> (T'-1,)."""
        transitions = current - reference
        return GeometricMetrics._angles(transitions[:-1], transitions[1:])

    @staticmethod
    def angle(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Angle between each hidden state and its reference: (T', D) -> (T',)."""
        return GeometricMetrics._angles(reference, current)

    @staticmethod
    def magnitude(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Euclidean distance between each hidden state and its reference: (T', D) -> (T',)."""
        return (current - reference).norm(dim=-1)

    @staticmethod
    def length(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Norm of each hidden state relative to its reference's: (T', D) -> (T',)."""
        return current.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)

    @staticmethod
    def projection_error(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Share of h_t its reference does not explain, ‖r_t‖ / ‖h_t‖: (T', D) -> (T',)."""
        u = reference / reference.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        residual = current - (current * u).sum(dim=-1, keepdim=True) * u
        return residual.norm(dim=-1) / current.norm(dim=-1).clamp_min(1e-12)

    @staticmethod
    def normalizers(hidden_states: torch.Tensor) -> dict[str, float]:
        """Text-level scale of every normalized metric at one layer (see NORMALIZED): (T, D) -> {metric: float}."""
        first, last = hidden_states[:1], hidden_states[-1:]
        return {
            "angle": GeometricMetrics.angle(last, first).item(),
            "magnitude": GeometricMetrics.magnitude(last, first).item(),
            "length": hidden_states.norm(dim=-1).mean().item(),
            "projection_error": GeometricMetrics.projection_error(last, first).item(),
        }

    def metrics_per_layer(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
        """
        Mean over tokens of every metric per reference and transformer layer: {"{reference}_{metric}": (L,)}; NaN if
        the text is too short. With --norm layer_first_to_last, divided by the text's scale at that layer.
        """
        out = {f"{reference}_{metric}": [] for reference in REFERENCES for metric in METRICS}
        # Omit the embedding output and use transformer layers.
        for layer in hidden_states[1:]:
            h = layer.float()
            scale = self.normalizers(h) if self.args.norm == "layer_first_to_last" else {}
            for reference in REFERENCES:
                current, previous = self.reference(h, reference)
                for metric in METRICS:
                    values = getattr(self, metric)(current, previous)
                    value = values.mean().item() if values.numel() else float("nan")
                    if metric in scale:
                        value = value / scale[metric] if scale[metric] > 1e-12 else float("nan")
                    out[f"{reference}_{metric}"].append(value)
        return {key: np.asarray(v, dtype=float) for key, v in out.items()}

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def plot(self, labels: np.ndarray, values: dict[str, np.ndarray], aurocs: dict[str, list[float]],
             path: str) -> None:
        """
        Row per reference, panel per metric: every text across layers as a thin transparent line and the per-group
        mean as a thick line (left axis); AUROC per layer with its three highest values marked (right axis).
        """
        colors = {"human": "#1f77b4", "machine": "#d62728"}
        fig = plt.figure(figsize=(5 * len(METRICS), 4.5 * len(REFERENCES)), layout="constrained")
        for subfig, reference in zip(fig.subfigures(len(REFERENCES), 1), REFERENCES):
            subfig.suptitle(REFERENCE_TITLES[reference], fontsize=11, fontweight="bold")
            for ax, metric in zip(subfig.subplots(1, len(METRICS)), METRICS):
                key = f"{reference}_{metric}"
                v = values[key]  # (N, L)
                layers = np.arange(1, v.shape[1] + 1)
                for label, group in ((0, "human"), (1, "machine")):
                    ax.plot(layers, v[labels == label].T, color=colors[group], alpha=0.05, linewidth=0.5)
                    ax.plot(layers, np.nanmean(v[labels == label], axis=0), color=colors[group], linewidth=2.5,
                            label=f"{group} mean")
                ax.set_xlabel("Layer")
                ax.set_ylabel(f"mean {metric} per text")
                ax_auroc = ax.twinx()
                a = np.asarray(aurocs[key], dtype=float)
                ax_auroc.plot(layers, a, color="black", linewidth=1, label="AUROC")
                top = np.argsort(np.nan_to_num(a, nan=-np.inf))[::-1][:3]
                ax_auroc.scatter(layers[top], a[top], color="black", s=30, zorder=3, label="top-3 AUROC")
                for layer in top:
                    ax_auroc.annotate(f"{a[layer]:.3f}", (layers[layer], a[layer]), textcoords="offset points",
                                      xytext=(0, 5), ha="center", fontsize=7)
                ax_auroc.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
                ax_auroc.set_ylim(0, 1)
                ax_auroc.set_ylabel("AUROC (higher value = human)")
                lines, auroc_lines = ax.get_legend_handles_labels(), ax_auroc.get_legend_handles_labels()
                ax.legend(lines[0] + auroc_lines[0], lines[1] + auroc_lines[1], frameon=False, fontsize=7)
                ax.set_title(metric, fontsize=10)
        fig.suptitle(f"{self.args.model}, {self.args.dataset}: geometric metrics per layer (norm: {self.args.norm})",
                     fontsize=12)
        fig.savefig(path)
        plt.close(fig)

    @staticmethod
    def correlation(values: list[np.ndarray]) -> np.ndarray:
        """
        Spearman correlation between metrics over texts, computed per layer (texts with a NaN dropped) and averaged
        over the layers: M × (N, L) -> (M, M).
        """
        stacked = np.stack(values, axis=-1)  # (N, L, M)
        per_layer = []
        for layer in range(stacked.shape[1]):
            x = stacked[:, layer]
            x = x[np.isfinite(x).all(axis=1)]
            if len(x) > 2:
                per_layer.append(spearmanr(x).statistic)
        return np.nanmean(per_layer, axis=0)

    def plot_correlation(self, values: dict[str, np.ndarray], path: str) -> None:
        """One heatmap per reference of the correlation between its metrics (see correlation)."""
        fig, axes = plt.subplots(1, len(REFERENCES), figsize=(5.5 * len(REFERENCES) + 1, 5.5), squeeze=False,
                                 layout="constrained")
        for ax, reference in zip(axes[0], REFERENCES):
            corr = self.correlation([values[f"{reference}_{metric}"] for metric in METRICS])
            image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(METRICS)), METRICS, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(METRICS)), METRICS, fontsize=8)
            for i in range(len(METRICS)):
                for j in range(len(METRICS)):
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(corr[i, j]) > 0.6 else "black")
            ax.set_title(REFERENCE_TITLES[reference], fontsize=9)
        fig.colorbar(image, ax=axes[0].tolist(), label="Spearman ρ (mean over layers)", shrink=0.8)
        fig.suptitle(f"{self.args.model}, {self.args.dataset}: correlation of the metrics over texts", fontsize=10)
        fig.savefig(path)
        plt.close(fig)

    def collect(self, args: Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Labels (N,) and every metric per text and layer under --norm: {"{reference}_{metric}": (N, L)}."""
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {f"{reference}_{metric}": [] for reference in REFERENCES for metric in METRICS}
        for item in tqdm(test_data, desc="Collecting geometric metrics"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            for key, v in self.metrics_per_layer(hidden_states).items():
                per_text[key].append(v)
        values = {key: np.asarray(v, dtype=float) for key, v in per_text.items()}  # {"{reference}_{metric}": (N, L)}
        if args.norm == "first_layer":
            with np.errstate(divide="ignore", invalid="ignore"):
                values = {key: np.where(np.abs(v[:, :1]) > 1e-12, v / v[:, :1], np.nan) for key, v in values.items()}
        return labels, values

    def run(self, args: Namespace) -> dict:
        labels, values = self.collect(args)

        # The negated metric as score, so a higher metric value = human; an AUROC below 0.5 means the metric is higher
        # for machine text.
        metrics_per_layer = {
            key: [self.evaluate(labels, -v[:, layer]) for layer in range(v.shape[1])]
            for key, v in values.items()
        }
        aurocs = {key: [m["auroc"] for m in per_layer] for key, per_layer in metrics_per_layer.items()}
        for key, a in aurocs.items():
            separation = np.abs(np.asarray(a) - 0.5)
            best = int(np.nanargmax(separation))
            print(f"AUROC {key}: most separating layer {best + 1} ({a[best]:.4f})")

        file_name = f"geometric_metrics_{args.model_name}_{args.dataset}_s{args.seed}" + (
            "" if args.norm == "none" else f"_{args.norm}")
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": METRICS,
            "references": REFERENCES,
            "context_k": CONTEXT_K,
            **{f"auroc_per_layer_{key}": {f"layer_{layer + 1}": a for layer, a in enumerate(a_key)}
               for key, a_key in aurocs.items()},
        }

        output_dir = OUTPUT_DIR if args.norm == "none" else os.path.join(OUTPUT_DIR, "norm")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot(labels, values, aurocs, os.path.join(output_dir, f"{file_name}.pdf"))
        self.plot_correlation(values, os.path.join(output_dir, f"{file_name}_correlation.pdf"))
        return aurocs


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--norm", type=str, choices=NORMS, default="none",
                        help="none; layer_first_to_last: divide angle, magnitude, length and projection_error by a "
                             "text-level scale per layer; first_layer: divide every metric by its first-layer value. "
                             "Normalized runs save to metrics/desc/norm.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = GeometricMetrics(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
