import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data, return_device

from src.config import Config
cfg = Config()

SELECTIONS = (
    "info_imbalance_first_local_max",
    "fluoroscopy_kl",
    "raw_global_max_token_entropy",
    "min_normalized_curvature",
    "fixed_layer",
)


class CurvatureLayerSelection():
    """Curvature of hidden states at a layer selected per sample, relative to the first transformer layer."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.selection = args.selection
        self.device = return_device()
        self.inference = Inference(model_name=args.model)

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

    def info_imbalance_first_local_max(self, hidden_states: tuple[torch.Tensor, ...]) -> int:
        """First layer whose Δ(ℓ → first) is higher than both neighbours; falls back to the global max if none."""
        dists = [self._distances(layer.to(self.device)) for layer in hidden_states]
        scores = np.asarray([self._imbalance(dist, dists[0]) for dist in dists])
        for i in range(1, len(scores) - 1):
            if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
                return i
        return int(np.nanargmax(scores))

    def _logit_lens(self, hidden_states: torch.Tensor, apply_norm: bool) -> torch.Tensor:
        """Unembed each token's hidden state: (T, D) -> next-token log-probs (T, V)."""
        model = self.inference.model
        hidden_states = hidden_states.to(self.device, model.lm_head.weight.dtype)
        with torch.no_grad():
            if apply_norm:
                hidden_states = model.model.norm(hidden_states)
            return torch.log_softmax(model.lm_head(hidden_states).float(), dim=-1)

    @staticmethod
    def _kl(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
        """Mean over tokens of KL(p ‖ q)."""
        return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean().item()

    def fluoroscopy_kl(self, hidden_states: tuple[torch.Tensor, ...]) -> int:
        """
        Text Fluoroscopy layer selection: M = argmax_j KL(q_N ‖ q_j) + KL(q_0 ‖ q_j) over the
        layers strictly between the first (q_0) and last (q_N) transformer layers.
        """
        # HF already applies the final norm to the last hidden state.
        log_q_first = self._logit_lens(hidden_states[0], apply_norm=True)
        log_q_last = self._logit_lens(hidden_states[-1], apply_norm=False)
        kls = []
        for layer in hidden_states[1:-1]:
            log_q = self._logit_lens(layer, apply_norm=True)
            kls.append(self._kl(log_q_last, log_q) + self._kl(log_q_first, log_q))
        return int(np.argmax(kls)) + 1

    def raw_global_max_token_entropy(self, hidden_states: tuple[torch.Tensor, ...]) -> int:
        """Layer with the highest logit-lens token entropy (mean over tokens)."""
        entropies = []
        for i, layer in enumerate(hidden_states):
            # HF already applies the final norm to the last hidden state.
            log_probs = self._logit_lens(layer, apply_norm=i < len(hidden_states) - 1)
            entropies.append((-(log_probs.exp() * log_probs).sum(dim=-1)).mean().item())
        return int(np.nanargmax(entropies))

    def min_normalized_curvature(self, hidden_states: tuple[torch.Tensor, ...]) -> int:
        """Layer where the model straightens the text most: argmin_ℓ curv(ℓ) / curv(first) over layers after the first."""
        return int(np.argmax(self.ratios(hidden_states)[1:])) + 1

    # ---------- Scoring ----------

    def score_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.curvature_hs(hidden_states).mean()

    def ratios(self, hidden_states: tuple[torch.Tensor, ...]) -> np.ndarray:
        """-curv(ℓ) / curv(first) for every transformer layer ℓ: (L,)."""
        curvatures = np.asarray([self.score_layer(layer).item() for layer in hidden_states])
        return -1 * curvatures / curvatures[0]

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> tuple[float, int]:
        """Ratio of mean curvature at the selected layer to the first transformer layer, plus that layer."""
        # Omit the embedding output and use transformer layers.
        hidden_states = hidden_states[1:]
        layer = getattr(self, self.selection)(hidden_states)
        return float(self.ratios(hidden_states)[layer]), layer

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def run(self, args: Namespace) -> dict:
        if self.selection == "fixed_layer":
            return self.run_fixed_layer(args)

        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        scores, layers = [], []
        for item in tqdm(test_data, desc=f"Collecting curvature_hs scores ({self.selection})"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            score, layer = self.score(hidden_states)
            scores.append(score)
            layers.append(layer)
        scores = np.asarray(scores, dtype=float)

        metrics = self.evaluate(labels, scores)
        print(json.dumps(metrics, indent=4))

        self.save(args, {
            "metrics": metrics,
            "scores": scores.tolist(),
            "selected_layers": layers,
            "labels": labels.tolist(),
        })
        return metrics

    def run_fixed_layer(self, args: Namespace) -> dict:
        """Baseline: the same layer for every text, evaluated at each transformer layer."""
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        scores = []
        for item in tqdm(test_data, desc="Collecting curvature_hs scores (fixed_layer)"):
            # Omit the embedding output and use transformer layers.
            scores.append(self.ratios(self.inference.run(item, args)["hidden_states"][1:]))
        scores = np.asarray(scores, dtype=float)  # (N, L)

        # Layer 0 is the reference itself (ratio = -1 for every text), so it is not evaluated.
        metrics_per_layer = [None] + [self.evaluate(labels, scores[:, layer]) for layer in range(1, scores.shape[1])]
        auroc_per_layer = [None] + [m["auroc"] for m in metrics_per_layer[1:]]
        best_layer = int(np.nanargmax(np.asarray(auroc_per_layer[1:], dtype=float))) + 1
        metrics = {**metrics_per_layer[best_layer], "best_layer": best_layer}
        print(json.dumps({"auroc_per_layer": auroc_per_layer, "best_layer": best_layer}, indent=4))

        self.save(args, {
            "metrics": metrics,
            "metrics_per_layer": metrics_per_layer,
            "auroc_per_layer": auroc_per_layer,
            "scores": scores.tolist(),
            "labels": labels.tolist(),
        })
        return metrics

    def save(self, args: Namespace, results: dict) -> None:
        file_name = f"curv_layer_selection_{self.selection}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "curvature_hs",
            "selection": self.selection,
            **results,
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--selection", type=str, choices=SELECTIONS, default="info_imbalance_first_local_max")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = CurvatureLayerSelection(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
