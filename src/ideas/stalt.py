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

# Softmax temperatures over layers. The deltas are cosine distances in [0, 2], so the paper's τ = 1 is close to uniform;
# smaller τ sharpens the layer selection. "uniform" (τ → ∞) weights every layer equally, i.e. the plain time-wise
# amplitude (the paper's "Time-wise" baseline), as an ablation of the layer weighting.
TAUS = ("0.01", "0.1", "1", "uniform")
# Higher StALT = larger temporal amplitude; machine text is assumed to move less between tokens (as for the angle in
# project.py), so the detection score is −StALT. AUROC < 0.5 means the direction is the other way round (with raw L2
# deltas it was: AUROC 0.04–0.24 on l8b).
SIGN = -1


class StALT():
    """
    Spatiotemporal Amplitude of Latent Transition (Furuya & Tanimura, 2026, arXiv:2605.01853), training-free.
    With h_t^l the hidden state of token t at layer l (l = 0 the embedding output, l = 1..L the transformer layers) and
    the cosine distance d(a, b) = 1 − cos(a, b) in place of the paper's L2 norm ‖a − b‖₂:
        Δ_time[t, l]  = d(h_t^l, h_{t−1}^l)    t = 2..T, l = 0..L   (change across tokens, per layer)
        Δ_layer[t, l] = d(h_t^l, h_t^{l−1})    t = 1..T, l = 1..L   (change across layers, per token)
    Align both on t = 2..T, l = 1..L: drop the embedding column of Δ_time (it reflects which token was read, not
    computation) and the first-token row of Δ_layer (no Δ_time entry). Per token, layer weights
        W[t, l] = softmax_l(Δ_layer[t, l] / τ)
    and
        StALT = 1/(T−1) Σ_{t=2..T} Σ_{l=1..L} W[t, l] · Δ_time[t, l]
    the across-token movement, weighted toward the layers where the token's representation changes most.
    Cosine distances ignore the hidden-state norm, which grows with depth and has outliers (and the final RMSNorm
    rescales the last layer), so neither the weights nor the amplitude are driven by scale alone. With deltas in
    [0, 2], τ = 1 gives near-uniform weights; smaller τ concentrates them (see the saved mean layer weights).
    --skip_first 1 drops the first token (attention sink, outlier norm) before computing the deltas.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)

    @staticmethod
    def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """1 − cos(a, b) along the last axis."""
        return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """StALT per τ and the token-averaged layer weights per τ: {τ: float}, {τ: (L,)}."""
        h = torch.stack(hidden_states).float()  # (L+1, T, D)
        if self.args.skip_first:
            h = h[:, 1:]
        if h.shape[1] < 2:
            return ({tau: float("nan") for tau in TAUS},
                    {tau: np.full(h.shape[0] - 1, np.nan) for tau in TAUS})
        # Aligned grids (L, T−1): Δ_time without the embedding layer, Δ_layer without the first token.
        delta_time = self.cosine_distance(h[1:, 1:], h[1:, :-1])
        delta_layer = self.cosine_distance(h[1:, 1:], h[:-1, 1:])
        scores, weights = {}, {}
        for tau in TAUS:
            if tau == "uniform":
                w = torch.full_like(delta_layer, 1.0 / delta_layer.shape[0])
            else:
                w = torch.softmax(delta_layer / float(tau), dim=0)
            scores[tau] = (w * delta_time).sum(dim=0).mean().item()
            weights[tau] = w.mean(dim=1).cpu().numpy()
        return scores, weights

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (too-short texts give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def plot_hist(self, labels: np.ndarray, values: dict[str, np.ndarray], aurocs: dict[str, float],
                  path: str) -> None:
        """One panel per τ: human vs machine StALT histograms, each group's mean dashed."""
        colors = {"human": "#1f77b4", "machine": "#d62728"}
        fig, axes = plt.subplots(1, len(TAUS), figsize=(5 * len(TAUS), 3.8), squeeze=False)
        for column, tau in enumerate(TAUS):
            ax = axes[0][column]
            v = values[tau]
            mask = np.isfinite(v)
            bins = np.histogram_bin_edges(v[mask], bins=40)
            for label, group in ((0, "human"), (1, "machine")):
                group_values = v[mask & (labels == label)]
                ax.hist(group_values, bins=bins, alpha=0.5, density=True, color=colors[group],
                        label=f"{group.capitalize()} (n={len(group_values)})")
                ax.axvline(group_values.mean(), color=colors[group], linestyle="--", linewidth=1)
            ax.set_xlabel("StALT, cosine" + (" (uniform layer weights)" if tau == "uniform" else f" (τ={tau})"))
            ax.set_ylabel("Density")
            ax.set_title(f"τ={tau}: AUROC(−StALT)={aurocs[tau]:.3f}", fontsize=9)
            ax.legend(frameon=False, fontsize=7)
        fig.suptitle(f"{self.args.model}, {self.args.dataset}: StALT"
                     + (", first token dropped" if self.args.skip_first else ""), fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {tau: [] for tau in TAUS}
        per_text_weights = {tau: [] for tau in TAUS}
        for item in tqdm(test_data, desc="Collecting StALT"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            scores, weights = self.score(hidden_states)
            for tau in TAUS:
                per_text[tau].append(scores[tau])
                per_text_weights[tau].append(weights[tau])
        values = {tau: np.asarray(v, dtype=float) for tau, v in per_text.items()}  # (N,)
        weights = {tau: np.asarray(w, dtype=float) for tau, w in per_text_weights.items()}  # (N, L)
        metrics = {tau: self.evaluate(labels, SIGN * v) for tau, v in values.items()}
        aurocs = {tau: m["auroc"] for tau, m in metrics.items()}
        for tau, a in aurocs.items():
            print(f"AUROC StALT tau={tau}: {a:.4f}")

        method = "stalt_cos"
        file_name = f"{method}_skip{args.skip_first}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "taus": TAUS,
            "score_sign": SIGN,
            **{f"metrics_tau_{tau}": m for tau, m in metrics.items()},
            **{f"mean_tau_{tau}": {"human": float(np.nanmean(v[labels == 0])),
                                   "machine": float(np.nanmean(v[labels == 1]))}
               for tau, v in values.items()},
            # Token-averaged softmax weight per transformer layer, averaged over the texts of each group.
            **{f"mean_layer_weights_tau_{tau}": {"human": np.nanmean(w[labels == 0], axis=0).tolist(),
                                                 "machine": np.nanmean(w[labels == 1], axis=0).tolist()}
               for tau, w in weights.items()},
            **{f"values_tau_{tau}": v.tolist() for tau, v in values.items()},
            "labels": labels.tolist(),
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot_hist(labels, values, aurocs, os.path.join(output_dir, f"{file_name}.pdf"))
        return aurocs


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--skip_first", type=int, choices=(0, 1), default=0,
                        help="1: drop the first token (attention sink) before computing the deltas; 0: as the paper.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = StALT(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
