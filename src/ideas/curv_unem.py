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

# Per-layer text scores (means over t); the text score is the negation, so higher = more machine-like.
SCORE_NAMES = ("abs_delta", "delta", "actual", "readout")
VARIANTS = ("hs_readout", "readout")
# How the top-K candidates are combined: prob = p̃_j (renormalised softmax), uniform = 1 / K.
WEIGHTINGS = ("prob", "uniform")
# Two reference angles from a query q_t to the top-K candidates w_{v_j}:
#   avg_emb:   ∠(q_t, Σ_j p̃_j w_{v_j})     (angle to the average embedding)
#   avg_angle: Σ_j p̃_j ∠(q_t, w_{v_j})     (average of the angles)
# Per variant, the key suffix of each reference; "" is the variant's original (main) reference.
REFERENCE_SUFFIXES = {
    "hs_readout": {"avg_emb": "", "avg_angle": "_avg_angle"},
    "readout": {"avg_angle": "", "avg_emb": "_avg_emb"},
}


class CurvatureUnembedding():
    """
    Actual hidden-state curvature vs. a readout-reference angle, per transformer layer ℓ:
        actual:   c_t  = ∠(h_t, h_{t+1})
        readout:  c̃_t = ∠(Norm(h_t), w̄_t),   w̄_t = Σ_{j ≤ K} p̃_j w_j
    where w_1..w_K are the unembedding vectors of the top-K logit-lens tokens of h_t with the observed next token
    x_{t+1} excluded (both variants), logits_t = W_U Norm(h_t), and p̃ is their softmax probability renormalised over
    the top K.
        Δc_t = c_t − c̃_t
    Hypothesis: machine text follows the model-preferred readout direction, so |Δc_t| is smaller.
    Text scores per layer (means over t = 0..T-2): abs_delta = |Δc|, delta = Δc, actual = c, readout = c̃.
    Main score: −mean |Δc_t| at the final layer (the cleanest readout; earlier layers are logit-lens approximations).

    Variants (--variant):
        hs_readout: the above (hidden-state curvature vs. readout-reference angle), query q_t = Norm(h_t).
        readout:    everything in unembedding space, so there is no hidden-state vs. unembedding mismatch:
                        actual:  c_t^U = ∠(w_{x_t}, w_{x_{t+1}})               (the same for every layer)
                        readout: μ_t^U = Σ_{j ≤ K} p̃_j ∠(w_{x_t}, w_{v_j})    (expected counterfactual curvature)
                    query q_t = w_{x_t}, and Δ_t = c_t^U − μ_t^U.

    Both variants compute both reference angles (see REFERENCE_SUFFIXES): the variant's original one under the plain
    names, the other under suffixed names (hs_readout: *_avg_angle, readout: *_avg_emb). actual is shared.
    Every score is computed twice: with the probability weights p̃_j (prob) and with uniform weights 1 / K (uniform).
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.k = args.k
        self.variant = args.variant
        self.suffixes = REFERENCE_SUFFIXES[self.variant]
        self.score_names = SCORE_NAMES + tuple(
            f"{name}{suffix}" for suffix in self.suffixes.values() if suffix
            for name in ("abs_delta", "delta", "readout")
        )
        self.inference = Inference(model_name=args.model)
        model = self.inference.model
        self.unembedding = model.get_output_embeddings().weight.detach()  # (V, D), model dtype
        self.device = self.unembedding.device
        self.final_norm = model.model.norm

    @staticmethod
    def angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Row-wise angle between two (..., D) tensors."""
        cosine = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
        return torch.acos(cosine.clamp(-1.0, 1.0))

    def normalise(self, h: torch.Tensor, is_final: bool) -> torch.Tensor:
        """Norm(h_t), the final norm the unembedding reads from: (N, D)."""
        # HF already applies the final norm to the last hidden state.
        if is_final:
            return h
        return self.final_norm(h.to(self.unembedding.dtype)).float()

    def top_k(self, normed: torch.Tensor, exclude: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Top-K logit-lens token ids of each Norm(h_t) and their probabilities p̃ renormalised over the top K.
        exclude: (N,) token ids removed from each row's candidates before taking the top K.
        """
        logits = (normed.to(self.unembedding.dtype) @ self.unembedding.T).float()  # (N, V)
        if exclude is not None:
            logits[torch.arange(len(exclude), device=logits.device), exclude] = -float("inf")
        top_logits, top_ids = logits.topk(self.k, dim=-1)
        # Softmax over the top K = full softmax renormalised over the top K.
        return top_ids, top_logits.softmax(dim=-1)  # (N, K), (N, K)

    @staticmethod
    def weightings(probs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Candidate weights per weighting: {name: (N, K)}."""
        return {"prob": probs, "uniform": torch.full_like(probs, 1 / probs.shape[-1])}

    def reference_angles(self, query: torch.Tensor, normed: torch.Tensor,
                         following_ids: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """
        Both reference angles from each query q_t to the top-K logit-lens tokens v_j of Norm(h_t), with the observed
        next token x_{t+1} excluded so the candidates are true alternatives: {weighting: {reference: (N,)}}.
        """
        top_ids, probs = self.top_k(normed, exclude=following_ids)
        candidates = self.unembedding[top_ids].float()  # (N, K, D)
        candidate_angles = self.angles(query[:, None], candidates)  # (N, K)
        return {
            name: {
                "avg_emb": self.angles(query, (weights[:, :, None] * candidates).sum(dim=1)),
                "avg_angle": (weights * candidate_angles).sum(dim=-1),
            }
            for name, weights in self.weightings(probs).items()
        }

    def score(self, hidden_states: tuple[torch.Tensor, ...], token_ids: torch.Tensor) -> dict[str, dict[str, np.ndarray]]:
        """Mean of each per-token score for every transformer layer, per weighting: {weighting: {name: (L,)}}."""
        # Omit the embedding output and use transformer layers.
        layers = hidden_states[1:]
        if layers[0].shape[0] < 2:
            return {weighting: {name: np.full(len(layers), np.nan) for name in self.score_names}
                    for weighting in WEIGHTINGS}
        out = {weighting: {name: [] for name in self.score_names} for weighting in WEIGHTINGS}
        with torch.no_grad():
            token_ids = token_ids.to(self.device)
            if self.variant == "readout":
                # The observed unembedding trajectory w_{x_1}, ..., w_{x_T} does not depend on the layer.
                w = self.unembedding[token_ids].float()
                actual_unembedding = self.angles(w[:-1], w[1:])
            for index, layer in enumerate(layers):
                h = layer.to(self.device).float()
                # Read out from Norm(h_t), the vector the logits are actually computed from.
                normed = self.normalise(h[:-1], is_final=index == len(layers) - 1)
                if self.variant == "readout":
                    actual, query = actual_unembedding, w[:-1]
                else:
                    actual, query = self.angles(h[:-1], h[1:]), normed
                for weighting, references in self.reference_angles(query, normed, token_ids[1:]).items():
                    out[weighting]["actual"].append(actual.mean().item())
                    for reference, readout in references.items():
                        suffix = self.suffixes[reference]
                        delta = actual - readout
                        out[weighting][f"abs_delta{suffix}"].append(delta.abs().mean().item())
                        out[weighting][f"delta{suffix}"].append(delta.mean().item())
                        out[weighting][f"readout{suffix}"].append(readout.mean().item())
        return {
            weighting: {name: np.asarray(values) for name, values in by_name.items()}
            for weighting, by_name in out.items()
        }

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

    def plot(self, labels: np.ndarray, values: dict[str, dict[str, np.ndarray]],
             aurocs: dict[str, dict[str, list[float]]], path: str) -> None:
        """One row per weighting: per-layer AUROC with per-group mean |Δc|, and a histogram of the main score."""
        colors = {"human": "#1f77b4", "machine": "#d62728"}
        markers = {"abs_delta": "o", "delta": "s", "actual": "^", "readout": "D"}
        for suffix in self.suffixes.values():
            if suffix:
                markers.update({f"abs_delta{suffix}": "v", f"delta{suffix}": "P", f"readout{suffix}": "X"})
        fig, axes = plt.subplots(len(WEIGHTINGS), 2, figsize=(12, 4 * len(WEIGHTINGS)), squeeze=False)
        for (ax_layer, ax_hist), weighting in zip(axes, WEIGHTINGS):
            by_name = values[weighting]

            # 1. Per-layer AUROC (left axis) and per-group mean |Δc| (right axis).
            layers = np.arange(1, by_name["abs_delta"].shape[1] + 1)
            for name in self.score_names:
                # Scores with the variant's secondary reference in grey.
                color = "grey" if name.endswith(("_avg_emb", "_avg_angle")) else "black"
                ax_layer.plot(layers, aurocs[weighting][name], color=color, linewidth=0.8, marker=markers[name],
                              markersize=4, markerfacecolor="none", label=f"AUROC {name}")
            ax_layer.axhline(0.5, color="grey", linewidth=0.8)
            ax_layer.set_xlabel("Layer")
            ax_layer.set_ylabel("AUROC")
            ax_mean = ax_layer.twinx()
            for group, mean in self.group_means(by_name["abs_delta"], labels).items():
                ax_mean.plot(layers, mean, color=colors[group], alpha=0.3, linewidth=2,
                             label=rf"mean $|\Delta c|$ {group}")
            ax_mean.set_ylabel(r"mean $|\Delta c_t|$ (rad)")
            lines = ax_layer.get_legend_handles_labels()
            mean_lines = ax_mean.get_legend_handles_labels()
            ax_layer.legend(lines[0] + mean_lines[0], lines[1] + mean_lines[1], frameon=False, fontsize=7)
            ax_layer.set_title(f"{self.variant} ({weighting}-weighted): {self.args.model}, {self.args.dataset}, "
                               f"K={self.k}", fontsize=9)

            # 2. Histogram of the main score (final-layer mean |Δc|).
            final = by_name["abs_delta"][:, -1]
            mask = np.isfinite(final)
            bins = np.histogram_bin_edges(final[mask], bins=40)
            for label, group in ((0, "human"), (1, "machine")):
                group_values = final[mask & (labels == label)]
                ax_hist.hist(group_values, bins=bins, alpha=0.5, density=True, color=colors[group],
                             label=f"{group.capitalize()} (n={len(group_values)})")
                ax_hist.axvline(group_values.mean(), color=colors[group], linestyle="--", linewidth=1)
            ax_hist.set_xlabel(r"final layer: mean $|c^U_t - \mu^U_t|$ (rad)" if self.variant == "readout"
                               else r"final layer: mean $|c_t - \tilde c_t|$ (rad)")
            ax_hist.set_ylabel("Density")
            ax_hist.set_title(f"{weighting}-weighted: AUROC={aurocs[weighting]['abs_delta'][-1]:.3f}", fontsize=9)
            ax_hist.legend(frameon=False)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {weighting: {name: [] for name in self.score_names} for weighting in WEIGHTINGS}
        for item in tqdm(test_data, desc=f"Collecting {self.variant} angles (K={self.k})"):
            out = self.inference.run(item, args)
            for weighting, by_name in self.score(out["hidden_states"], out["token_ids"]).items():
                for name, values in by_name.items():
                    per_text[weighting][name].append(values)
        values = {
            weighting: {name: np.asarray(v, dtype=float) for name, v in by_name.items()}  # (N, L)
            for weighting, by_name in per_text.items()
        }

        metrics_per_layer = {
            weighting: {
                name: [self.evaluate(labels, -1 * v[:, layer]) for layer in range(v.shape[1])]
                for name, v in by_name.items()
            }
            for weighting, by_name in values.items()
        }
        aurocs = {
            weighting: {name: [m["auroc"] for m in per_layer] for name, per_layer in by_name.items()}
            for weighting, by_name in metrics_per_layer.items()
        }
        for weighting in WEIGHTINGS:
            print(f"AUROC {weighting} (final layer, -mean |delta c|): {aurocs[weighting]['abs_delta'][-1]:.4f}")

        # Probability-weighted keys keep their original names; uniform-weighted keys get a "_uniform" suffix.
        def keyed(weighting: str, key: str) -> str:
            return key if weighting == "prob" else f"{key}_uniform"

        method = f"curv_unem{'_readout' if self.variant == 'readout' else ''}_k{self.k}"
        file_name = f"{method}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
        }
        for weighting in WEIGHTINGS:
            by_name = values[weighting]
            output.update({
                keyed(weighting, "metrics"): metrics_per_layer[weighting]["abs_delta"][-1],
                **{keyed(weighting, f"auroc_per_layer_{name}"): a for name, a in aurocs[weighting].items()},
                **{keyed(weighting, f"metrics_per_layer_{name}"): m
                   for name, m in metrics_per_layer[weighting].items()},
                **{keyed(weighting, f"mean_per_layer_{name}"): self.group_means(v, labels) for name, v in by_name.items()},
                keyed(weighting, "scores"): (-1 * by_name["abs_delta"][:, -1]).tolist(),
                **{keyed(weighting, f"values_per_layer_{name}"): v.tolist() for name, v in by_name.items()},
            })
        output["labels"] = labels.tolist()

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
    parser.add_argument("--k", type=int, default=10,
                        help="Top-K logit-lens tokens forming the expected readout direction.")
    parser.add_argument("--variant", type=str, choices=VARIANTS, default="hs_readout",
                        help="hs_readout: hidden-state curvature vs. readout-reference angle; readout: unembedding "
                             "curvature of the observed tokens vs. expected curvature to the top-K candidates "
                             "(observed next token excluded from the candidates in both variants).")
    args = parser.parse_args()
    args.return_token_ids = True
    return args


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = CurvatureUnembedding(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
