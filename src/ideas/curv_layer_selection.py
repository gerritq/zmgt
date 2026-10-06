import os
import json
import numpy as np
import torch
import skdim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data, return_device

from src.config import Config
cfg = Config()

# Per-layer statistic used for the layer selection.
STATISTICS = ("information_imbalance", "id_mle", "jsd_to_first", "jsd_to_first_token")
# Rule that picks the layer from the per-layer statistic.
CRITERIA = ("first_local_max", "first_effective_peak")
# Centred moving-average window over layers when the statistic is smoothed (--smooth 1).
SMOOTH_WINDOW = 3
# Nearest neighbours per token for the MLE intrinsic dimension (skdim default).
MLE_NEIGHBORS = 20
CURVATURES = ("curvature_hs", "curvature_context_against_current")
# raw: −curv(ℓ); ratio: −curv(ℓ) / curv(1), the score of 4_curv_layer_selection (curv(1): first transformer layer).
SCORE_VARIANTS = ("raw", "ratio")
# Fixed-layer baselines, numbered 1..L over the transformer layers (embeddings excluded).
FIXED_LAYERS = tuple(range(7, 13))
# Preceding tokens averaged by curvature_context_against_current.
CONTEXT_WINDOW = 3
# Token each curvature value is attributed to: value j of a curvature belongs to token j + offset (its current token).
TOKEN_OFFSET = {"curvature_hs": 0, "curvature_context_against_current": CONTEXT_WINDOW}


class CurvatureLayerSelection():
    """
    Curvature of the hidden states at a layer selected per text. Layers are the transformer layers ℓ = 1..L.
    1. For every layer, the mean curvature of the text under each curvature variant:
           curvature_hs:                      ∠(h_t, h_{t+1})
           curvature_context_against_current: ∠(mean(h_{t−3}, ..., h_{t−1}), h_t)
    2. The statistic (--statistic) scores every layer of the text (its "method score"), optionally smoothed over
       layers (--smooth, moving average of SMOOTH_WINDOW layers):
           information_imbalance: Δ(ℓ → first layer)
           id_mle:                MLE intrinsic dimension of the token cloud (skdim, MLE_NEIGHBORS neighbours)
           jsd_to_first:          mean over tokens of JSD(p_ℓ, p_first) between logit-lens next-token distributions
           jsd_to_first_token:    JSD(p_ℓ, p_first) of every token on its own; the criterion picks one layer per token
       The criterion (--criterion) picks one layer from it:
           first_local_max:      first layer higher than both neighbours
           first_effective_peak: first local max that is not followed by a strictly increasing run within the
                                 horizon w (--horizon) ending above it
    3. Text score at the selected layer, per curvature and score variant (higher = more machine-like):
           raw:   −curv(ℓ)
           ratio: −curv(ℓ) / curv(1)
       With a per-token selection, curv(ℓ) is the mean over tokens of each token's curvature at its own layer, where a
       curvature value belongs to its current token (t+1 for ∠(h_t, h_{t+1}), k for the context variant).
    Baseline: the same scores at the fixed layers 7..12 for every text.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        # Name of the layer selection, used in titles and file names.
        self.method = f"{args.statistic}_{args.criterion}" + ("_smooth" if args.smooth else "")
        self.device = return_device()
        self.inference = Inference(model_name=args.model)

    # ---------- Curvature ----------

    @staticmethod
    def _angles(previous: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
        """Row-wise angle between two (N, D) tensors."""
        denominator = (previous.norm(dim=-1) * following.norm(dim=-1)).clamp_min(1e-12)
        cosine = ((previous * following).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    @staticmethod
    def curvature_hs(hidden_states: torch.Tensor) -> torch.Tensor:
        """Curvature alla Hoesseini but with hidden states"""
        hidden_states = hidden_states.float()
        return CurvatureLayerSelection._angles(hidden_states[:-1], hidden_states[1:])

    @staticmethod
    def curvature_context_against_current(hidden_states: torch.Tensor, window: int = CONTEXT_WINDOW) -> torch.Tensor:
        """For every token k >= window: angle between mean(x_{k-window}, ..., x_{k-1}) and x_k."""
        hidden_states = hidden_states.float()
        if hidden_states.shape[0] <= window:
            return torch.empty(0)
        pooled = hidden_states.unfold(0, window, 1).mean(dim=-1)  # (T-window+1, D); row j = mean(x_j..x_{j+window-1})
        return CurvatureLayerSelection._angles(pooled[:-1], hidden_states[window:])

    def token_curvatures(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """Curvature of every token and transformer layer, per curvature variant: {curvature: (L, T')}."""
        return {curvature: torch.stack([getattr(self, curvature)(layer) for layer in hidden_states])
                for curvature in CURVATURES}

    @staticmethod
    def at_layers(token_curvature: torch.Tensor, token_layers: np.ndarray, offset: int) -> float:
        """Mean over tokens of each token's curvature at its selected layer: (L, T') and (T,) -> scalar."""
        n_values = token_curvature.shape[1]
        if n_values == 0:
            return float("nan")
        layers = torch.as_tensor(token_layers[offset:offset + n_values], device=token_curvature.device)
        return token_curvature[layers, torch.arange(n_values, device=token_curvature.device)].mean().item()

    # ---------- Layer selection ----------

    @staticmethod
    def _distances(hidden_states: torch.Tensor) -> torch.Tensor:
        """Pairwise Euclidean distances between tokens (T, D) -> (T, T), self-distance set to inf."""
        hidden_states = hidden_states.float()
        dist = torch.cdist(hidden_states, hidden_states, compute_mode="donot_use_mm_for_euclid_dist")
        dist.fill_diagonal_(float("inf"))
        return dist

    @staticmethod
    def _imbalance(dist_a: torch.Tensor, dist_b: torch.Tensor) -> float:
        """Δ(A → B) = 2/T · mean_i r_i, where r_i is the rank at B of i's nearest neighbour at A."""
        n_tokens = dist_a.shape[0]
        if n_tokens < 3:
            return float("nan")
        nearest_a = dist_a.argmin(dim=-1)
        dist_to_nearest = dist_b.gather(1, nearest_a.unsqueeze(-1))
        ranks = (dist_b < dist_to_nearest).sum(dim=-1) + 1
        return (2 * ranks.float().mean() / n_tokens).item()

    def _imbalance_scores(self, hidden_states: tuple[torch.Tensor, ...]) -> np.ndarray:
        """Δ(ℓ → first) per layer: (L,)."""
        dists = [self._distances(layer.to(self.device)) for layer in hidden_states]
        return np.asarray([self._imbalance(dist, dists[0]) for dist in dists])

    def _id_mle_scores(self, hidden_states: tuple[torch.Tensor, ...]) -> np.ndarray:
        """MLE intrinsic dimension of the text's token cloud per layer: (L,)."""
        scores = np.full(len(hidden_states), np.nan)
        n_neighbors = min(MLE_NEIGHBORS, hidden_states[0].shape[0] - 1)
        if n_neighbors < 2:
            return scores
        for i, layer in enumerate(hidden_states):
            # Sorted nearest-neighbour distances and indices (self excluded), computed on the device.
            nearest = self._distances(layer.to(self.device)).topk(n_neighbors, dim=-1, largest=False)
            knn = (nearest.values.double().cpu().numpy(), nearest.indices.cpu().numpy())
            with np.errstate(divide="ignore", invalid="ignore"):
                dimension = float(skdim.id.MLE().fit(layer.float().cpu().numpy(), precomputed_knn_arrays=knn,
                                                     n_neighbors=n_neighbors).dimension_)
            if np.isfinite(dimension):
                scores[i] = dimension
        return scores

    def _logit_lens(self, hidden_states: torch.Tensor, apply_norm: bool) -> torch.Tensor:
        """Unembed each token's hidden state: (T, D) -> next-token log-probs (T, V)."""
        model = self.inference.model
        hidden_states = hidden_states.to(self.device, model.lm_head.weight.dtype)
        with torch.no_grad():
            if apply_norm:
                hidden_states = model.model.norm(hidden_states)
            return torch.log_softmax(model.lm_head(hidden_states).float(), dim=-1)

    @staticmethod
    def _jsd(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        """Per-token JSD(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m), m = ½ (p + q) (nats, in [0, log 2]): (T,)."""
        log_mix = torch.logaddexp(log_p, log_q) - np.log(2)
        kl = lambda log_r: (log_r.exp() * (log_r - log_mix)).sum(dim=-1)
        return 0.5 * (kl(log_p) + kl(log_q))

    def _jsd_to_first_token_scores(self, hidden_states: tuple[torch.Tensor, ...]) -> np.ndarray:
        """JSD(p_ℓ, p_first) of every token over the logit lens: (L, T); 0 at the first layer."""
        # HF already applies the final norm to the last hidden state.
        last = len(hidden_states) - 1
        log_p_first = self._logit_lens(hidden_states[0], apply_norm=True)
        return torch.stack([self._jsd(self._logit_lens(layer, apply_norm=i < last), log_p_first)
                            for i, layer in enumerate(hidden_states)]).cpu().numpy()

    def _jsd_to_first_scores(self, hidden_states: tuple[torch.Tensor, ...]) -> np.ndarray:
        """JSD(p_ℓ, p_first) per layer, mean over tokens: (L,); 0 at the first layer."""
        return self._jsd_to_first_token_scores(hidden_states).mean(axis=1)

    def statistic(self, hidden_states: tuple[torch.Tensor, ...]) -> np.ndarray:
        """Per-layer statistic of --statistic, smoothed over layers with --smooth: (L,)."""
        if self.args.statistic == "information_imbalance":
            scores = self._imbalance_scores(hidden_states)
        elif self.args.statistic == "id_mle":
            scores = self._id_mle_scores(hidden_states)
        else:
            scores = self._jsd_to_first_scores(hidden_states)
        return self._smooth(scores) if self.args.smooth else scores

    @staticmethod
    def _smooth(scores: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
        """Centred moving average over layers; NaNs are ignored and the window shrinks at the edges."""
        half = window // 2
        smoothed = np.full(len(scores), np.nan)
        for i in range(len(scores)):
            segment = scores[max(0, i - half):i + half + 1]
            segment = segment[np.isfinite(segment)]
            if segment.size:
                smoothed[i] = segment.mean()
        return smoothed

    @staticmethod
    def _argmax(scores: np.ndarray) -> int:
        """nanargmax that falls back to the first layer when every score is NaN (too-short text)."""
        return int(np.nanargmax(scores)) if np.isfinite(scores).any() else 0

    @staticmethod
    def _local_maxima(scores: np.ndarray) -> list[int]:
        """Layers higher than both neighbours, shallow to deep."""
        return [i for i in range(1, len(scores) - 1) if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]]

    def first_local_max(self, scores: np.ndarray) -> int:
        """First local max; falls back to the global max if none."""
        peaks = self._local_maxima(scores)
        return peaks[0] if peaks else self._argmax(scores)

    def first_effective_peak(self, scores: np.ndarray) -> int:
        """
        First local max ℓ that survives the horizon N+(ℓ, w) = {ℓ+1, ..., min(ℓ+w, L)}: it is discarded if
        d(ℓ) < d(min(ℓ+w, L)) and d(ℓ+1) < d(ℓ+2) < ... < d(min(ℓ+w, L)), i.e. the statistic keeps increasing past ℓ.
        Defaults to the first local max if none survive, and to the global max if there is no local max.
        """
        peaks = self._local_maxima(scores)
        if not peaks:
            return self._argmax(scores)
        for peak in peaks:
            end = min(peak + self.args.horizon, len(scores) - 1)
            rising = np.all(np.diff(scores[peak + 1:end + 1]) > 0)
            if not (scores[peak] < scores[end] and rising):
                return peak
        return peaks[0]

    def select(self, hidden_states: tuple[torch.Tensor, ...]) -> tuple[np.ndarray, np.ndarray]:
        """
        Method score per layer (L,) and the selected layer index of every token (T,). A per-text statistic selects the
        same layer for every token; jsd_to_first_token selects on each token's own curve, and its method score is the
        mean of those curves.
        """
        criterion = getattr(self, self.args.criterion)
        n_tokens = hidden_states[0].shape[0]
        if self.args.statistic != "jsd_to_first_token":
            scores = self.statistic(hidden_states)
            return scores, np.full(n_tokens, criterion(scores), dtype=int)
        token_scores = self._jsd_to_first_token_scores(hidden_states).T  # (T, L)
        if self.args.smooth:
            token_scores = np.stack([self._smooth(scores) for scores in token_scores])
        return token_scores.mean(axis=0), np.asarray([criterion(scores) for scores in token_scores], dtype=int)

    # ---------- Scoring ----------

    @staticmethod
    def score_variants(curvature: np.ndarray) -> dict[str, np.ndarray]:
        """Text score of every layer per score variant, higher = more machine-like: (N, L) -> {variant: (N, L)}."""
        return {
            "raw": -1 * curvature,
            "ratio": -1 * curvature / curvature[:, :1],
        }

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    # ---------- Plot ----------

    def plot(self, labels: np.ndarray, method_scores: np.ndarray, selected: np.ndarray,
             fixed_aurocs: dict[str, dict[str, list[float]]], selected_aurocs: dict[str, dict[str, float]],
             path: str) -> None:
        """
        Row 1: the method score per layer of every text (transparent, by group) with a dot at its selected layer.
        Row 2: selected layers by group, and AUROC of the selected layer vs the fixed layers 7..12.
        With a per-token selection the selected layer of a text is its most frequent one.
        """
        colors = {"human": "#1f77b4", "machine": "#d62728"}
        groups = ((0, "human"), (1, "machine"))
        fig = plt.figure(figsize=(16, 9))
        grid = fig.add_gridspec(2, 2)

        # 1. Method score per layer and text, dot at the selected layer.
        ax = fig.add_subplot(grid[0, :])
        layers = np.arange(1, method_scores.shape[1] + 1)
        for label, group in groups:
            rows = np.flatnonzero(labels == label)
            for i in rows:
                ax.plot(layers, method_scores[i], color=colors[group], alpha=0.08, linewidth=0.8)
            chosen = selected[rows]
            ax.scatter(chosen + 1, method_scores[rows, chosen], color=colors[group], alpha=0.4, s=10,
                       edgecolor="none", label=f"selected layer, {group} (n={len(rows)})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("method score")
        ax.set_title(self.method, fontsize=9)
        ax.legend(frameon=False, fontsize=7)

        # 2. Distribution of the selected layer by group.
        ax_hist = fig.add_subplot(grid[1, 0])
        bins = np.arange(0.5, len(layers) + 1.5)
        for label, group in groups:
            ax_hist.hist(selected[labels == label] + 1, bins=bins, alpha=0.4, color=colors[group],
                         histtype="stepfilled", label=group)
        ax_hist.set_xlabel("selected layer")
        ax_hist.set_ylabel("texts")
        ax_hist.legend(frameon=False, fontsize=7)
        ax_hist.set_title("selected layers", fontsize=9)

        # 3. AUROC: selected layer vs fixed layers.
        ax_bar = fig.add_subplot(grid[1, 1])
        combos = [(curvature, variant) for curvature in CURVATURES for variant in SCORE_VARIANTS]
        width = 0.8 / (len(FIXED_LAYERS) + 1)
        shades = plt.cm.Greys(np.linspace(0.3, 0.7, len(FIXED_LAYERS)))
        for j, layer in enumerate(FIXED_LAYERS):
            ax_bar.bar(np.arange(len(combos)) + j * width, [fixed_aurocs[c][v][j] for c, v in combos], width,
                       color=shades[j], label=f"layer {layer}" if j in (0, len(FIXED_LAYERS) - 1) else None)
        ax_bar.bar(np.arange(len(combos)) + len(FIXED_LAYERS) * width, [selected_aurocs[c][v] for c, v in combos],
                   width, color="#2ca02c", label="selected")
        ax_bar.axhline(0.5, color="grey", linewidth=0.8)
        ax_bar.set_xticks(np.arange(len(combos)) + 0.4 - width / 2)
        ax_bar.set_xticklabels([f"{c.replace('curvature_', '')}\n{v}" for c, v in combos], fontsize=7)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("AUROC")
        ax_bar.legend(frameon=False, fontsize=7, ncol=3)
        ax_bar.set_title(f"fixed layers {FIXED_LAYERS[0]}-{FIXED_LAYERS[-1]} vs selected", fontsize=9)

        fig.suptitle(f"{self.args.model}, {self.args.dataset}: layer selection {self.method}", fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    # ---------- Run ----------

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        curvature_values = {curvature: [] for curvature in CURVATURES}
        selected_curvature = {curvature: [] for curvature in CURVATURES}
        method_scores, selected, token_layer_counts = [], [], 0
        for item in tqdm(test_data, desc=f"Collecting curvature with layer selection ({self.method})"):
            # Omit the embedding output and use transformer layers.
            hidden_states = self.inference.run(item, args)["hidden_states"][1:]
            # The selected layers are shared by all curvature variants.
            scores, token_layers = self.select(hidden_states)
            for curvature, token_curvature in self.token_curvatures(hidden_states).items():
                curvature_values[curvature].append(token_curvature.mean(dim=1).cpu().numpy() if token_curvature.shape[1]
                                                   else np.full(len(hidden_states), np.nan))
                selected_curvature[curvature].append(self.at_layers(token_curvature, token_layers,
                                                                    TOKEN_OFFSET[curvature]))
            counts = np.bincount(token_layers, minlength=len(hidden_states))
            method_scores.append(scores)
            # Per text: the most frequent selected layer (the selected layer for a per-text statistic).
            selected.append(int(counts.argmax()))
            token_layer_counts = token_layer_counts + counts
        curvature_values = {c: np.asarray(v, dtype=float) for c, v in curvature_values.items()}  # (N, L)
        selected_curvature = {c: np.asarray(v, dtype=float) for c, v in selected_curvature.items()}  # (N,)
        method_scores = np.asarray(method_scores, dtype=float)  # (N, L)
        selected = np.asarray(selected, dtype=int)  # (N,)

        fixed, fixed_aurocs, at_selected, selected_metrics, selected_aurocs = {}, {}, {}, {}, {}
        aurocs_per_layer = {}
        for curvature in CURVATURES:
            variants = self.score_variants(curvature_values[curvature])  # {variant: (N, L)}
            aurocs_per_layer[curvature] = {
                variant: [self.evaluate(labels, v[:, layer])["auroc"] for layer in range(v.shape[1])]
                for variant, v in variants.items()
            }
            fixed[curvature] = {
                variant: {f"layer_{layer}": self.evaluate(labels, v[:, layer - 1]) for layer in FIXED_LAYERS}
                for variant, v in variants.items()
            }
            fixed_aurocs[curvature] = {
                variant: [m["auroc"] for m in by_layer.values()] for variant, by_layer in fixed[curvature].items()
            }
            at_selected[curvature] = {
                "raw": -1 * selected_curvature[curvature],
                "ratio": -1 * selected_curvature[curvature] / curvature_values[curvature][:, 0],
            }
            selected_metrics[curvature] = {
                variant: self.evaluate(labels, s) for variant, s in at_selected[curvature].items()
            }
            selected_aurocs[curvature] = {variant: m["auroc"] for variant, m in selected_metrics[curvature].items()}

        for curvature in CURVATURES:
            for variant in SCORE_VARIANTS:
                fixed_text = " ".join(f"L{layer} {a:.3f}" for layer, a in zip(FIXED_LAYERS, fixed_aurocs[curvature][variant]))
                print(f"AUROC {curvature} {variant}: selected {selected_aurocs[curvature][variant]:.4f} | fixed {fixed_text}")

        method = "curv_layer_selection"
        horizon = f"_w{args.horizon}" if args.criterion == "first_effective_peak" else ""
        file_name = f"{method}_{self.method}{horizon}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "selection": self.method,
            "smooth_window": SMOOTH_WINDOW if args.smooth else None,
            "mle_neighbors": MLE_NEIGHBORS if args.statistic == "id_mle" else None,
            "layer_numbering": "1..L over the transformer layers (embeddings excluded)",
            # Fixed-layer baselines: {curvature: {variant: {layer_ℓ: metrics}}}.
            "default": fixed,
            # Per-text selected layer: {curvature: {variant: metrics}}.
            "layer_selection": selected_metrics,
            # Every layer as a fixed layer: {curvature: {variant: (L,)}}.
            "auroc_per_layer": aurocs_per_layer,
            # Per text: the selected layer (per-token selection: the most frequent one).
            "selected_layers": (selected + 1).tolist(),
            # Share of tokens, over all texts, whose selected layer is layer_ℓ.
            "token_layer_share": {f"layer_{layer + 1}": share
                                  for layer, share in enumerate((token_layer_counts / token_layer_counts.sum()).tolist())},
            "scores_at_selected_layer": {
                c: {variant: s.tolist() for variant, s in by_variant.items()} for c, by_variant in at_selected.items()
            },
            "method_scores_per_layer": method_scores.tolist(),
            "curvature_per_layer": {c: v.tolist() for c, v in curvature_values.items()},
            "labels": labels.tolist(),
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot(labels, method_scores, selected, fixed_aurocs, selected_aurocs,
                  os.path.join(output_dir, f"{file_name}.pdf"))
        return selected_metrics


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--statistic", type=str, choices=STATISTICS, default="information_imbalance",
                        help="Per-layer statistic the layer is selected on.")
    parser.add_argument("--criterion", type=str, choices=CRITERIA, default="first_local_max",
                        help="Rule that picks the layer from the statistic.")
    parser.add_argument("--smooth", type=int, choices=(0, 1), default=0,
                        help=f"1: smooth the statistic over layers (moving average of {SMOOTH_WINDOW}) before selecting.")
    parser.add_argument("--horizon", type=int, default=3,
                        help="Forward horizon w of first_effective_peak.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = CurvatureLayerSelection(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
