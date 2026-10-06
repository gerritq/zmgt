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


DISTANCES = ("cosine", "euclidean")
SCORE_NAMES = ("target", "k", "ratio", "diff", "z")
# Scores compared against the top-K neighbourhood (saved in the second file).
K_SCORE_NAMES = ("ratio", "diff", "z")


class TargetUnembeddingDistance():
    """
    Distance of the (final-normalised) hidden state z_t = LN(h_t) to the unembedding vector of the token that
    actually comes next, and to the K vocabulary tokens z_t points to most strongly:
        d_t^target = d(z_t, w_{x_{t+1}})
        μ_{K,t}    = mean_{v ∈ N_K(z_t)} d(z_t, w_v),   N_K = K closest w_v with v ≠ x_{t+1}
        σ_{K,t}    = std_{v ∈ N_K(z_t)} d(z_t, w_v)
    Text scores (means over t = 1..T-1):
        target: d_t^target
        k:      μ_{K,t}
        ratio:  d_t^target / μ_{K,t}
        diff:   d_t^target − μ_{K,t}
        z:      (d_t^target − μ_{K,t}) / (σ_{K,t} + ε)
    Each is computed for two distances d:
        cosine:    1 − cos(z, w)
        euclidean: ‖z − w‖
    Hypothesis: machine text follows the model's expectation, so the actual token is as close as the locally
    plausible ones (all scores lower). Scores are negated so that higher = more machine-like.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.k = args.k
        self.eps = 1e-8
        self.inference = Inference(model_name=args.model)
        self.unembedding = self.inference.model.get_output_embeddings().weight.detach().float()  # (V, D)
        self.device = self.unembedding.device
        self.unembedding_sq = self.unembedding.pow(2).sum(dim=-1)  # (V,)

    def distance_matrix(self, z: torch.Tensor, distance: str) -> torch.Tensor:
        """Distance from every z_t to every unembedding vector: (T − 1, V)."""
        dot = z @ self.unembedding.T
        z_sq = z.pow(2).sum(dim=-1)
        if distance == "cosine":
            norms = z_sq.sqrt()[:, None] * self.unembedding_sq.sqrt()[None, :]
            return 1 - dot / norms.clamp_min(self.eps)
        # ‖z − w‖² = ‖z‖² + ‖w‖² − 2 zᵀw
        return (z_sq[:, None] + self.unembedding_sq[None, :] - 2 * dot).clamp_min(0).sqrt()

    def distances(self, hidden_states: tuple[torch.Tensor, ...], token_ids: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """Per-token d_t^target, μ_{K,t} and σ_{K,t} for t = 0..T-2: {distance: {"target"/"mean"/"std": (T − 1,)}}."""
        # HF already applies the final norm to the last hidden state, so this is z_t = LN(h_t).
        z = hidden_states[-1][:-1].to(self.device).float()
        targets = token_ids[1:].to(self.device)
        rows = torch.arange(len(targets), device=self.device)
        out = {}
        with torch.no_grad():
            for distance in DISTANCES:
                d = self.distance_matrix(z, distance)
                target = d[rows, targets]
                # Exclude the observed token from its comparison neighbourhood.
                d[rows, targets] = float("inf")
                nearest = d.topk(self.k, dim=-1, largest=False).values  # (T − 1, K)
                out[distance] = {
                    "target": target.cpu(),
                    "mean": nearest.mean(dim=-1).cpu(),
                    "std": nearest.std(dim=-1).cpu(),
                }
        return out

    def score(self, hidden_states: tuple[torch.Tensor, ...], token_ids: torch.Tensor) -> dict[str, dict[str, float]]:
        """Mean over tokens of every score in SCORE_NAMES, per distance."""
        if token_ids.numel() < 2:
            return {distance: {name: float("nan") for name in SCORE_NAMES} for distance in DISTANCES}
        return {
            distance: {
                "target": d["target"].mean().item(),
                "k": d["mean"].mean().item(),
                "ratio": (d["target"] / d["mean"].clamp_min(self.eps)).mean().item(),
                "diff": (d["target"] - d["mean"]).mean().item(),
                "z": ((d["target"] - d["mean"]) / (d["std"] + self.eps)).mean().item(),
            }
            for distance, d in self.distances(hidden_states, token_ids).items()
        }

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (too-short texts give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def plot(self, labels: np.ndarray, panels: list[list[tuple[np.ndarray, str, float]]], path: str) -> None:
        """Human vs machine histograms: a grid of panels, each (values, x-label, AUROC)."""
        n_rows, n_cols = len(panels), len(panels[0])
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False)
        for row_axes, row_panels in zip(axes, panels):
            for ax, (values, xlabel, auroc) in zip(row_axes, row_panels):
                mask = np.isfinite(values)
                bins = np.histogram_bin_edges(values[mask], bins=40)
                for label, name, color in ((0, "Human", "#1f77b4"), (1, "Machine", "#d62728")):
                    group = values[mask & (labels == label)]
                    ax.hist(group, bins=bins, alpha=0.5, density=True, color=color, label=f"{name} (n={len(group)})")
                    ax.axvline(group.mean(), color=color, linestyle="--", linewidth=1)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Density")
                ax.set_title(f"{self.args.model}, {self.args.dataset}: AUROC={auroc:.3f}", fontsize=9)
                ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def save(self, output: dict, file_name: str, labels: np.ndarray, panels: list[list[tuple[np.ndarray, str, float]]]) -> None:
        output_dir = os.path.join(cfg.zero_output_dir, self.args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot(labels, panels, os.path.join(output_dir, f"{file_name}.pdf"))

    @staticmethod
    def group_means(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        return {"human": float(np.nanmean(values[labels == 0])), "machine": float(np.nanmean(values[labels == 1]))}

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {distance: {name: [] for name in SCORE_NAMES} for distance in DISTANCES}
        for item in tqdm(test_data, desc=f"Collecting target / top-{self.k} unembedding distances"):
            out = self.inference.run(item, args)
            for distance, text_scores in self.score(out["hidden_states"], out["token_ids"]).items():
                for name, value in text_scores.items():
                    per_text[distance][name].append(value)
        values = {
            distance: {name: np.asarray(v, dtype=float) for name, v in by_name.items()}
            for distance, by_name in per_text.items()
        }
        scores = {distance: {name: -1 * v for name, v in by_name.items()} for distance, by_name in values.items()}
        metrics = {
            distance: {name: self.evaluate(labels, scores[distance][name]) for name in ("target",) + K_SCORE_NAMES}
            for distance in DISTANCES
        }
        print(json.dumps(metrics, indent=4))
        for distance, by_name in metrics.items():
            for name, m in by_name.items():
                print(f"AUROC {distance} {name}: {m['auroc']:.4f}")

        base = {**vars(args), "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        suffix = f"{args.model_name}_{args.dataset}_s{args.seed}"

        # 1. Target only.
        method = "hs_unem_target"
        self.save({
            **base,
            "method": method,
            "metrics": {distance: metrics[distance]["target"] for distance in DISTANCES},
            "mean_distance": {distance: self.group_means(values[distance]["target"], labels) for distance in DISTANCES},
            "scores": {distance: scores[distance]["target"].tolist() for distance in DISTANCES},
            "labels": labels.tolist(),
        }, f"{method}_{suffix}", labels, [[
            (values[distance]["target"], rf"{distance}: mean $d(z_t, w_{{x_{{t+1}}}})$", metrics[distance]["target"]["auroc"])
            for distance in DISTANCES
        ]])

        # 2. Target against the top-K neighbourhood.
        method = f"hs_unem_target_k{self.k}"
        xlabels = {
            "ratio": r"mean $d_t^{\mathrm{target}} / \mu_{K,t}$",
            "diff": r"mean $d_t^{\mathrm{target}} - \mu_{K,t}$",
            "z": r"mean $(d_t^{\mathrm{target}} - \mu_{K,t}) / (\sigma_{K,t} + \epsilon)$",
        }
        self.save({
            **base,
            "method": method,
            **{
                distance: {
                    **{f"metrics_{name}": metrics[distance][name] for name in K_SCORE_NAMES},
                    "mean_distance_target": self.group_means(values[distance]["target"], labels),
                    "mean_distance_k": self.group_means(values[distance]["k"], labels),
                    **{f"mean_{name}": self.group_means(values[distance][name], labels) for name in K_SCORE_NAMES},
                    **{f"scores_{name}": scores[distance][name].tolist() for name in K_SCORE_NAMES},
                }
                for distance in DISTANCES
            },
            "labels": labels.tolist(),
        }, f"{method}_{suffix}", labels, [
            [
                (values[distance][name], f"{distance}: {xlabels[name]} (K={self.k})", metrics[distance][name]["auroc"])
                for name in K_SCORE_NAMES
            ]
            for distance in DISTANCES
        ])
        return metrics


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--k", type=int, default=10,
                        help="Size of the nearest-unembedding neighbourhood (observed token excluded).")
    args = parser.parse_args()
    if args.k < 2:
        parser.error("--k must be >= 2 for the neighbourhood standard deviation.")
    args.return_token_ids = True
    return args


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = TargetUnembeddingDistance(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
