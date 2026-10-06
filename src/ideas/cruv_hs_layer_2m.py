import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import Namespace
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from src.ideas.curv_hs_2m import CurvatureTwoModels, SCORE_NAMES, SIGNS, parse_args
from src.utils import load_data

from src.config import Config
cfg = Config()


class CurvatureAcrossLayersTwoModels(CurvatureTwoModels):
    """
    The two-model contrast of curv_hs_2m, with each token's curvature measured across layers instead of across tokens.
    For token t, over the transformer layers ℓ = 1..L (embeddings excluded):
        c_t = mean_{ℓ=1..L−1} ∠(h_t^ℓ, h_t^{ℓ+1})
    under the instruct and the base model on the same input ids. Per token:
        inst:  c_t^inst
        base:  c_t^base
        diff:  c_t^inst − c_t^base
        ratio: c_t^inst / c_t^base
    each averaged over the tokens of a text, giving one value per score and text. Signs as in curv_hs_2m (SIGNS).
    Main metrics: ratio.
    """

    METHOD = "cruv_hs_layer_2m"
    # A single token already has a trajectory across layers.
    MIN_TOKENS = 1
    # One value per token, so there is no first layer to normalise by.
    NORMALISE_TO_FIRST = False

    def n_values(self) -> tuple:
        """One value per score and text."""
        return ()

    def token_curvature(self, model: AutoModelForCausalLM, inputs: dict) -> torch.Tensor:
        """Mean angle between consecutive layers' hidden states of each token: (T,)."""
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        # Omit the embedding output and use transformer layers: (L, T, D).
        hidden_states = torch.stack(outputs.hidden_states[1:]).squeeze(1).float()
        return self.angles(hidden_states[:-1], hidden_states[1:]).mean(dim=0)

    def plot(self, labels: np.ndarray, values: dict[str, np.ndarray], aurocs: dict[str, float], path: str) -> None:
        """One histogram per score: human vs machine, with each group's mean dashed."""
        colors = {"human": "#1f77b4", "machine": "#d62728"}
        units = {"inst": " (rad)", "base": " (rad)", "diff": " (rad)", "ratio": ""}
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, name in zip(axes.flat, SCORE_NAMES):
            mask = np.isfinite(values[name])
            bins = np.histogram_bin_edges(values[name][mask], bins=40)
            for label, group in ((0, "human"), (1, "machine")):
                group_values = values[name][mask & (labels == label)]
                ax.hist(group_values, bins=bins, alpha=0.5, density=True, color=colors[group],
                        label=f"{group.capitalize()} (n={len(group_values)})")
                ax.axvline(group_values.mean(), color=colors[group], linestyle="--", linewidth=1)
            ax.set_xlabel(f"mean token curvature across layers: {name}{units[name]}")
            ax.set_ylabel("Density")
            ax.set_title(f"{name}: AUROC={aurocs[name]:.3f}", fontsize=9)
            ax.legend(frameon=False)
        fig.suptitle(f"instruct: {self.args.instruct}   |   base: {self.args.base}   |   dataset: {self.args.dataset}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {name: [] for name in SCORE_NAMES}
        for item in tqdm(test_data, desc=f"Collecting {self.METHOD}"):
            for name, value in self.score(item["text"]).items():
                per_text[name].append(float(value))
        values = {name: np.asarray(v, dtype=float) for name, v in per_text.items()}  # (N,)
        scores = {name: SIGNS[name] * v for name, v in values.items()}

        metrics_by_score = {name: self.evaluate(labels, scores[name]) for name in SCORE_NAMES}
        aurocs = {name: m["auroc"] for name, m in metrics_by_score.items()}
        print(json.dumps(metrics_by_score, indent=4))
        for name in SCORE_NAMES:
            print(f"AUROC {name}: {aurocs[name]:.4f}")

        file_name = f"{self.METHOD}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": self.METHOD,
            "inst_model": self.inst_id,
            "base_model": self.base_id,
            "score_signs": SIGNS,
            "metrics": metrics_by_score["ratio"],
            **{f"metrics_{name}": m for name, m in metrics_by_score.items()},
            **{f"mean_{name}": {"human": float(np.nanmean(v[labels == 0])), "machine": float(np.nanmean(v[labels == 1]))}
               for name, v in values.items()},
            "scores": scores["ratio"].tolist(),
            **{f"scores_{name}": s.tolist() for name, s in scores.items()},
            "labels": labels.tolist(),
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot(labels, values, aurocs, os.path.join(output_dir, f"{file_name}.pdf"))
        return output["metrics"]


def main() -> None:
    args = parse_args()
    # Hugging Face ids contain "/", which cannot go into a file name.
    args.model_name = f"{args.instruct}_vs_{args.base}".replace("/", "-")

    model = CurvatureAcrossLayersTwoModels(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
