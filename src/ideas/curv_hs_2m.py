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
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import evaluation, load_data, return_device

from src.config import Config
cfg = Config()

# Base model keys. --instruct and --base both accept a key of cfg.model_dict, a key of BASE_MODELS, or a Hugging Face id.
BASE_MODELS = {
    "q1.7bb": "Qwen/Qwen3-1.7B-Base",
    "q4bb": "Qwen/Qwen3-4B-Base",
    "q8bb": "Qwen/Qwen3-8B-Base",
    "l1bb": "meta-llama/Llama-3.2-1B",
    "l3bb": "meta-llama/Llama-3.2-3B",
    "l8bb": "meta-llama/Llama-3.1-8B",
    "g1bb": "google/gemma-3-1b-pt",
    "g4bb": "google/gemma-3-4b-pt",
    "g12bb": "google/gemma-3-12b-pt",
}


# Per-layer text scores (means over tokens).
SCORE_NAMES = ("inst", "base", "diff", "ratio")
# Sign turning each value into a score where higher = more machine-like: machine text is straighter (lower
# curvature) under either model, but curves more under the instruct model relative to the base model.
SIGNS = {"inst": -1, "base": -1, "diff": 1, "ratio": 1}
# The same scores with each model's curvature first taken relative to its first layer (per token), as curvature_hs
# does: ĉ_t^ℓ = c_t^ℓ / c_t^1. Same signs as the raw scores.
NORM_SCORE_NAMES = tuple(f"{name}_norm" for name in SCORE_NAMES)
SIGNS.update({f"{name}_norm": sign for name, sign in list(SIGNS.items())})
# Layer whose ratio is reported as the main metrics (as curvature_hs's target layer).
TARGET_LAYER = 10


class CurvatureTwoModels():
    """
    curvature_hs (angle between consecutive hidden states) under an instruct model and its base model, per
    transformer layer ℓ, on the same input ids:
        c_t^inst = ∠(h_t^inst, h_{t+1}^inst),   c_t^base = ∠(h_t^base, h_{t+1}^base)
    Text scores per layer (means over t = 0..T-2):
        inst:  c_t^inst
        base:  c_t^base
        diff:  c_t^inst − c_t^base
        ratio: c_t^inst / c_t^base
    inst and base are negated (lower curvature = machine-like); diff and ratio are not (higher instruct curvature
    relative to base = machine-like, as observed from the middle layers on). Main metrics: ratio at TARGET_LAYER.

    *_norm scores repeat this with each model's curvature relative to its first layer, per token:
        ĉ_t^{ℓ,m} = c_t^{ℓ,m} / c_t^{1,m}   (m = inst, base)
        inst_norm = ĉ^inst,  base_norm = ĉ^base,  diff_norm = ĉ^inst − ĉ^base,  ratio_norm = ĉ^inst / ĉ^base
    Layer 1 of every *_norm score is constant (1 or 0), so it carries no signal.
    """

    # Overridden by subclasses that measure curvature along another axis.
    METHOD = "curv_hs_2m"
    X_LABEL = "Layer"
    # Fewest tokens a text needs for a curvature value.
    MIN_TOKENS = 2
    # Add the *_norm scores (curvature relative to the first layer); needs a layer axis.
    NORMALISE_TO_FIRST = True

    @property
    def score_names(self) -> tuple[str, ...]:
        return SCORE_NAMES + (NORM_SCORE_NAMES if self.NORMALISE_TO_FIRST else ())

    def n_values(self) -> int:
        """Number of per-layer values per score."""
        return self.inst_model.config.num_hidden_layers

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.device = return_device()
        self.inst_id = self.resolve(args.instruct)
        self.base_id = self.resolve(args.base)
        # The same input ids are fed to both models, so their tokenizers must match.
        self.tokenizer = AutoTokenizer.from_pretrained(self.inst_id)
        if self.tokenizer.get_vocab() != AutoTokenizer.from_pretrained(self.base_id).get_vocab():
            raise ValueError(f"{self.inst_id} and {self.base_id} do not share a tokenizer vocabulary.")
        self.inst_model = self.load_model(self.inst_id)
        self.base_model = self.load_model(self.base_id)

    @staticmethod
    def resolve(model: str) -> str:
        """Hugging Face id of a model key (instruct or base), or the argument itself if it is not a key."""
        return {**cfg.model_dict, **BASE_MODELS}.get(model, model)

    def load_model(self, model_id: str) -> AutoModelForCausalLM:
        # bf16 so that both models fit on one GPU.
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map=self.device)
        model.eval()
        return model

    @staticmethod
    def angles(previous: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
        """Angle between two (..., D) tensors along the last dimension."""
        denominator = (previous.norm(dim=-1) * following.norm(dim=-1)).clamp_min(1e-12)
        cosine = ((previous * following).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    def token_curvature(self, model: AutoModelForCausalLM, inputs: dict) -> torch.Tensor:
        """curvature_hs per transformer layer and token of one model: (L, T − 1)."""
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        # Omit the embedding output and use transformer layers: (L, T, D).
        hidden_states = torch.stack(outputs.hidden_states[1:]).squeeze(1).float()
        return self.angles(hidden_states[:, :-1], hidden_states[:, 1:])

    def score(self, text: str) -> dict[str, np.ndarray]:
        """Mean over tokens of each score, per layer: {name: (L,)}."""
        if not text.strip():
            raise ValueError("Input text must be non-empty.")
        inputs = self.tokenizer(text,
                                truncation=True,
                                add_special_tokens=False,
                                max_length=1024,
                                return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if inputs["input_ids"].shape[-1] < self.MIN_TOKENS:
            return {name: np.full(self.n_values(), np.nan) for name in self.score_names}

        curv_inst = self.token_curvature(self.inst_model, inputs)
        curv_base = self.token_curvature(self.base_model, inputs)
        per_token = {
            "inst": curv_inst,
            "base": curv_base,
            "diff": curv_inst - curv_base,
            "ratio": curv_inst / curv_base.clamp_min(1e-12),
        }
        if self.NORMALISE_TO_FIRST:
            # Each model's curvature relative to its first transformer layer, per token: (L, T − 1).
            norm_inst = curv_inst / curv_inst[:1].clamp_min(1e-12)
            norm_base = curv_base / curv_base[:1].clamp_min(1e-12)
            per_token.update({
                "inst_norm": norm_inst,
                "base_norm": norm_base,
                "diff_norm": norm_inst - norm_base,
                "ratio_norm": norm_inst / norm_base.clamp_min(1e-12),
            })
        return {name: values.mean(dim=-1).cpu().numpy() for name, values in per_token.items()}

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (too-short texts give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    @staticmethod
    def group_means(values: np.ndarray, labels: np.ndarray) -> dict[str, list[float]]:
        """Per-layer mean of (N, L) values for each group."""
        return {
            "human": np.nanmean(values[labels == 0], axis=0).tolist(),
            "machine": np.nanmean(values[labels == 1], axis=0).tolist(),
        }

    def plot(self, labels: np.ndarray, values: dict[str, np.ndarray], aurocs: dict[str, list[float]],
             path: str) -> None:
        """
        One panel per score: per-layer AUROC (left axis) and the per-group mean score ± 1 standard deviation across
        texts (right axis, transparent). Row 1: raw scores; row 2: *_norm scores (relative to the first layer).
        """
        colors = {"human": "#1f77b4", "machine": "#d62728"}
        units = {"inst": "rad", "base": "rad", "diff": "rad"}
        rows = [SCORE_NAMES] + ([NORM_SCORE_NAMES] if self.NORMALISE_TO_FIRST else [])
        fig, axes = plt.subplots(len(rows), len(SCORE_NAMES), figsize=(6 * len(SCORE_NAMES), 4 * len(rows)),
                                 squeeze=False)
        for ax, name in zip(axes.flat, [name for row in rows for name in row]):
            layers = np.arange(1, values[name].shape[1] + 1)
            ax.plot(layers, aurocs[name], color="black", marker="o", markersize=3, label="AUROC")
            ax.axhline(0.5, color="grey", linewidth=0.8)
            ax.axvline(TARGET_LAYER + 1, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xlabel(self.X_LABEL)
            ax.set_ylabel("AUROC")
            ax_mean = ax.twinx()
            for label, group in ((0, "human"), (1, "machine")):
                mean = np.nanmean(values[name][labels == label], axis=0)
                std = np.nanstd(values[name][labels == label], axis=0)
                ax_mean.plot(layers, mean, color=colors[group], alpha=0.3, linewidth=2, label=f"mean ± 1 SD {group}")
                ax_mean.fill_between(layers, mean - std, mean + std, color=colors[group], alpha=0.1, linewidth=0)
            ax_mean.set_ylabel(f"mean {name}" + (f" ({units[name]})" if name in units else ""))
            lines, mean_lines = ax.get_legend_handles_labels(), ax_mean.get_legend_handles_labels()
            ax.legend(lines[0] + mean_lines[0], lines[1] + mean_lines[1], frameon=False, fontsize=7)
            ax.set_title(name, fontsize=9)
        fig.suptitle(f"instruct: {self.args.instruct}   |   base: {self.args.base}   |   dataset: {self.args.dataset}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {name: [] for name in self.score_names}
        for item in tqdm(test_data, desc=f"Collecting {self.METHOD}"):
            for name, values in self.score(item["text"]).items():
                per_text[name].append(values)
        values = {name: np.asarray(v, dtype=float) for name, v in per_text.items()}  # (N, L)

        metrics_per_layer = {
            name: [self.evaluate(labels, SIGNS[name] * v[:, layer]) for layer in range(v.shape[1])]
            for name, v in values.items()
        }
        aurocs = {name: [m["auroc"] for m in per_layer] for name, per_layer in metrics_per_layer.items()}
        metrics = metrics_per_layer["ratio"][TARGET_LAYER]
        print(json.dumps({f"auroc_per_layer_{n}": a for n, a in aurocs.items()}, indent=4))
        for name in self.score_names:
            print(f"AUROC {name} (layer index {TARGET_LAYER}): {aurocs[name][TARGET_LAYER]:.4f}")

        method = self.METHOD
        file_name = f"{method}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "inst_model": self.inst_id,
            "base_model": self.base_id,
            "target_layer": TARGET_LAYER,
            "score_signs": SIGNS,
            "metrics": metrics,
            **{f"auroc_per_layer_{name}": a for name, a in aurocs.items()},
            **{f"metrics_per_layer_{name}": m for name, m in metrics_per_layer.items()},
            **{f"mean_per_layer_{name}": self.group_means(v, labels) for name, v in values.items()},
            "scores": (SIGNS["ratio"] * values["ratio"][:, TARGET_LAYER]).tolist(),
            **{f"values_per_layer_{name}": v.tolist() for name, v in values.items()},
            "labels": labels.tolist(),
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot(labels, values, aurocs, os.path.join(output_dir, f"{file_name}.pdf"))
        return metrics


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--instruct", type=str, required=True,
                        help="First model: a key of cfg.model_dict (e.g. l8b), of BASE_MODELS (e.g. l8bb), or a Hugging Face id.")
    parser.add_argument("--base", type=str, required=True,
                        help="Second model: a key of cfg.model_dict (e.g. l8b), of BASE_MODELS (e.g. l8bb), or a Hugging Face id.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Hugging Face ids contain "/", which cannot go into a file name.
    args.model_name = f"{args.instruct}_vs_{args.base}".replace("/", "-")

    model = CurvatureTwoModels(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
