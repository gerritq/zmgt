import os
import json
import importlib
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

# Share of hidden dimensions, with the smallest singular values of W_U, that form the "remaining" subspace
# (1/16 = 256 of 4096 dimensions for Llama-3.1-8B).
REMAINING_FRACTION = 1 / 16
# Subspaces the last-layer states are projected onto; "remaining" is the one of interest, the others are baselines.
SUBSPACES = ("remaining", "semantic", "full")
# Curvatures of 2_curvature_tweaked, applied to the projected states.
Curvature = importlib.import_module("src.ideas.2_curvature_tweaked").Curvature
CURVATURES = ("curvature_hs", "curvature_context_against_current")
# raw: mean curvature at layer ℓ; first_ratio: divided by the mean at the first transformer layer, as the
# curvature_hs score of 2_curvature_tweaked (a ratio of text means). The first layer's first_ratio is always 1.
VARIANTS = ("raw", "first_ratio")


class RemainingSubspaceCurvature():
    """
    Curvature of the hidden-state trajectory inside the subspace the logits barely see, per transformer layer.
    1. SVD of the unembedding W_U = U Σ Vᵀ (V × D). The right singular vectors v_i are an orthonormal basis of the
       hidden space; σ_i is how strongly the logits respond to direction v_i.
    2. Split the basis: the REMAINING_FRACTION of directions with the smallest σ_i form the "remaining" subspace
       (internal computation that never reaches the output); the rest form the "semantic" subspace.
    3. For every transformer layer ℓ, project the states onto a subspace and measure curvature with the functions
       of 2_curvature_tweaked:
           curvature_hs:                      ∠(p_t, p_{t+1})
           curvature_context_against_current: ∠(mean(p_{t−3}, ..., p_{t−1}), p_t)
       remaining / semantic: p_t = V_Sᵀ Norm(h_t^ℓ), through the final norm W_U reads (the last layer's states are
       already normed by HF). full: p_t = h_t^ℓ, the raw hidden states, as in 2_curvature_tweaked.
    4. Text score per layer, negated (lower curvature = more machine-like, as in curvature_hs):
           raw:         mean curvature at layer ℓ
           first_ratio: mean curvature at layer ℓ / mean curvature at layer 1 (2_curvature_tweaked's score; for the
                        full space at layer index 10 it reproduces curvature_hs / curvature_context_against_current)
    Scores are reported for the remaining subspace (main), and for the semantic subspace and the full space as
    baselines. Main metrics: remaining, curvature_hs, last layer.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)
        model = self.inference.model
        unembedding = model.get_output_embeddings().weight.detach()  # (V, D)
        self.device = unembedding.device
        self.dtype = unembedding.dtype
        self.final_norm = model.model.norm
        self.bases, self.singular_values = self.split_basis(unembedding)

    @staticmethod
    def split_basis(unembedding: torch.Tensor) -> tuple[dict[str, torch.Tensor | None], dict[str, list[float]]]:
        """
        Right singular vectors of W_U split into the remaining and semantic subspaces: {subspace: (D, k)}, with None
        for the full space; and the singular values of each subspace.
        """
        # Eigenvectors of W_Uᵀ W_U are W_U's right singular vectors, eigenvalues σ²; cheaper than an SVD of (V, D).
        w = unembedding.float()
        eigenvalues, eigenvectors = torch.linalg.eigh((w.T @ w).double())  # ascending σ²
        singular_values = eigenvalues.clamp_min(0).sqrt()
        n_remaining = round(REMAINING_FRACTION * w.shape[1])
        bases = {
            "remaining": eigenvectors[:, :n_remaining].float(),
            "semantic": eigenvectors[:, n_remaining:].float(),
            "full": None,
        }
        return bases, {
            "remaining": singular_values[:n_remaining].tolist(),
            "semantic": singular_values[n_remaining:].tolist(),
        }

    @staticmethod
    def angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Row-wise angle between two (N, k) tensors."""
        cosine = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
        return torch.acos(cosine.clamp(-1.0, 1.0))

    def normalise(self, h: torch.Tensor, is_final: bool) -> torch.Tensor:
        """Norm(h_t), the final norm W_U reads through: (T, D)."""
        # HF already applies the final norm to the last hidden state.
        if is_final:
            return h
        return self.final_norm(h.to(self.dtype)).float()

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, dict[str, np.ndarray]]:
        """Mean curvature per subspace, curvature type and transformer layer: {subspace: {curvature: (L,)}}."""
        # Omit the embedding output and use transformer layers.
        layers = hidden_states[1:]
        out = {subspace: {curvature: [] for curvature in CURVATURES} for subspace in SUBSPACES}
        with torch.no_grad():
            for index, layer in enumerate(layers):
                h = layer.to(self.device).float()
                z = self.normalise(h, is_final=index == len(layers) - 1)
                for subspace, basis in self.bases.items():
                    # The full space uses the raw states, as 2_curvature_tweaked; the W_U subspaces the normed ones.
                    p = h if basis is None else z @ basis  # (T, k)
                    for curvature in CURVATURES:
                        angles = getattr(Curvature, curvature)(p)
                        out[subspace][curvature].append(angles.mean().item() if angles.numel() else float("nan"))
        return {
            subspace: {curvature: np.asarray(v) for curvature, v in by_curvature.items()}
            for subspace, by_curvature in out.items()
        }

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (too-short texts give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def plot(self, labels: np.ndarray, values: dict[str, dict[str, dict[str, np.ndarray]]],
             aurocs: dict[str, dict[str, dict[str, list[float]]]], path: str) -> None:
        """
        Row per curvature (curvature_hs, curvature_context_against_current), column per subspace and variant (raw,
        first_ratio): per-layer AUROC (left axis) and the per-group mean ± 1 standard deviation across texts (right
        axis, transparent).
        """
        colors = {"human": "#1f77b4", "machine": "#d62728"}
        columns = [(subspace, variant) for subspace in SUBSPACES for variant in VARIANTS]
        fig, axes = plt.subplots(len(CURVATURES), len(columns), figsize=(5 * len(columns), 4 * len(CURVATURES)),
                                 squeeze=False)
        for column, (subspace, variant) in enumerate(columns):
            dims = "D" if self.bases[subspace] is None else self.bases[subspace].shape[1]
            for row, curvature in enumerate(CURVATURES):
                ax = axes[row][column]
                v = values[subspace][curvature][variant]  # (N, L)
                layers = np.arange(1, v.shape[1] + 1)
                ax.plot(layers, aurocs[subspace][curvature][variant], color="black", marker="o", markersize=3,
                        label="AUROC")
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
                ax_mean.set_ylabel("mean curvature (rad)" if variant == "raw" else "curvature / layer 1")
                lines, mean_lines = ax.get_legend_handles_labels(), ax_mean.get_legend_handles_labels()
                ax.legend(lines[0] + mean_lines[0], lines[1] + mean_lines[1], frameon=False, fontsize=7)
                ax.set_title(f"{subspace} ({dims} dims), {variant}\n{curvature}", fontsize=9)
        fig.suptitle(f"{self.args.model}, {self.args.dataset}: curvature per subspace of $W_U$", fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {subspace: {curvature: [] for curvature in CURVATURES} for subspace in SUBSPACES}
        for item in tqdm(test_data, desc="Collecting subspace curvature per layer"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            for subspace, by_curvature in self.score(hidden_states).items():
                for curvature, v in by_curvature.items():
                    per_text[subspace][curvature].append(v)
        values = {}
        for subspace, by_curvature in per_text.items():
            values[subspace] = {}
            for curvature, v in by_curvature.items():
                v = np.asarray(v, dtype=float)  # (N, L)
                values[subspace][curvature] = {"raw": v, "first_ratio": v / v[:, :1]}
        # Lower curvature = more machine-like.
        metrics_per_layer = {
            subspace: {
                curvature: {
                    variant: [self.evaluate(labels, -1 * v[:, layer]) for layer in range(v.shape[1])]
                    for variant, v in by_variant.items()
                }
                for curvature, by_variant in by_curvature.items()
            }
            for subspace, by_curvature in values.items()
        }
        aurocs = {
            subspace: {
                curvature: {variant: [m["auroc"] for m in per_layer] for variant, per_layer in by_variant.items()}
                for curvature, by_variant in by_curvature.items()
            }
            for subspace, by_curvature in metrics_per_layer.items()
        }
        for subspace in SUBSPACES:
            for curvature in CURVATURES:
                for variant in VARIANTS:
                    a = aurocs[subspace][curvature][variant]
                    print(f"AUROC {subspace} {curvature} {variant}: last layer {a[-1]:.4f}, "
                          f"best {np.nanmax(a):.4f} (layer {int(np.nanargmax(a)) + 1})")

        method = "harp_idea"
        file_name = f"{method}_{args.model_name}_{args.dataset}_s{args.seed}"
        # Raw keys keep their names; first_ratio keys get a "_first_ratio" suffix.
        keyed = lambda key, subspace, curvature, variant: (
            f"{key}_{subspace}_{curvature}" + ("" if variant == "raw" else f"_{variant}")
        )
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "remaining_fraction": REMAINING_FRACTION,
            "n_remaining": len(self.singular_values["remaining"]),
            "singular_value_range": {
                subspace: [min(sv), max(sv)] for subspace, sv in self.singular_values.items()
            },
            "metrics": metrics_per_layer["remaining"]["curvature_hs"]["raw"][-1],
            "scores": (-1 * values["remaining"]["curvature_hs"]["raw"][:, -1]).tolist(),
            "labels": labels.tolist(),
        }
        for subspace in SUBSPACES:
            for curvature in CURVATURES:
                for variant in VARIANTS:
                    v = values[subspace][curvature][variant]
                    output.update({
                        keyed("auroc_per_layer", subspace, curvature, variant): aurocs[subspace][curvature][variant],
                        keyed("metrics_per_layer", subspace, curvature, variant):
                            metrics_per_layer[subspace][curvature][variant],
                        keyed("mean_per_layer", subspace, curvature, variant): {
                            "human": np.nanmean(v[labels == 0], axis=0).tolist(),
                            "machine": np.nanmean(v[labels == 1], axis=0).tolist(),
                        },
                        keyed("values_per_layer", subspace, curvature, variant): v.tolist(),
                    })

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot(labels, values, aurocs, os.path.join(output_dir, f"{file_name}.pdf"))
        return output["metrics"]


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = RemainingSubspaceCurvature(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
