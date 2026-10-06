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

METHODS = ("surrounding_context", "preceding_context", "cross_layer")


class SurroundingContextNovelty():
    """
    Per layer, measure how far each token state lies outside the subspace spanned by its context
    (GEM, Yang et al., 2019): novelty and significance, for two contexts:
    - surrounding: the m neighbours on either side.
    - preceding: the m preceding states only; also the magnitude-weighted exploration ratio.
    - cross_layer: the token and its m preceding states, taken from layer l − g (novelty gained between layers).
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.method = args.method
        self.window = args.window
        self.layer_gap = args.layer_gap
        self.score_names = (f"{self.method}_novelty", f"{self.method}_significance")
        if self.method == "preceding_context":
            self.score_names += (f"{self.method}_exploration_ratio",)
        self.eps = 1e-8
        self.device = return_device()
        self.inference = Inference(model_name=args.model)

    def context_basis(self, context_states: torch.Tensor, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Orthonormal basis Q_t of C_t = [c_{t+o} for o in offsets] (clipped at the edges) for every token:
        Q (T, d, K) with columns of out-of-range positions zeroed, and valid (T, K).
        """
        num_tokens = context_states.shape[0]
        index = torch.arange(num_tokens, device=self.device)[:, None] + offsets[None]  # (T, K)
        valid = (index >= 0) & (index < num_tokens)
        # Put valid context columns first so the leading Q columns span exactly the valid context.
        order = torch.argsort((~valid).int(), dim=1, stable=True)
        index, valid = index.gather(1, order), valid.gather(1, order)
        context = context_states[index.clamp(0, num_tokens - 1)] * valid[..., None]  # (T, K, d)
        q, _ = torch.linalg.qr(context.transpose(1, 2))  # (T, d, K)
        q = q * valid[:, None, :]  # drop columns belonging to padded (out-of-range) positions
        return q, valid

    def context_scores(
        self, hidden_states: torch.Tensor, offsets: torch.Tensor, context_states: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        GEM scores. For token h_t, C_t = [c_{t+o} for o in offsets] (clipped at the edges), where c are
        context_states (default: hidden_states themselves),
        C_t = Q_t R_t, u_t = (I − Q_t Q_tᵀ) h_t (‖u_t‖ = r_{-1} in the paper), then per token (T,):
        novelty α_n = exp(‖u_t‖ / ‖h_t‖) (eq. 4) and significance α_s = ‖u_t‖ / (|offsets| + 1) (eq. 6).
        Tokens without any in-range context are dropped.
        """
        hidden_states = hidden_states.float()
        context_states = hidden_states if context_states is None else context_states.float()
        q, valid = self.context_basis(context_states, offsets)
        projected = torch.einsum("tdk,tk->td", q, torch.einsum("tdk,td->tk", q, hidden_states))
        residual_norm = (hidden_states - projected).norm(dim=-1)
        novelty = torch.exp(residual_norm / (hidden_states.norm(dim=-1) + self.eps))
        significance = residual_norm / (offsets.numel() + 1)
        has_context = valid.any(dim=1)
        return {"novelty": novelty[has_context], "significance": significance[has_context]}

    def surrounding_context_scores(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Context C_t = [h_{t-m}, ..., h_{t-1}, h_{t+1}, ..., h_{t+m}]; significance divides by 2m + 1."""
        offsets = torch.cat([torch.arange(-self.window, 0), torch.arange(1, self.window + 1)]).to(self.device)
        return self.context_scores(hidden_states, offsets)

    def preceding_context_scores(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Context C_t = [h_{t-m}, ..., h_{t-1}]; significance divides by m + 1. The first token is dropped."""
        offsets = torch.arange(-self.window, 0, device=self.device)
        return self.context_scores(hidden_states, offsets)

    def exploration_ratio(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Magnitude-weighted pooling over the preceding context S_t = span(h_{t-m}, ..., h_{t-1}). The step
        Δ_t = h_t − h_{t-1} minus its component along the previous direction u = Δ_{t-1} / ‖Δ_{t-1}‖ is the turn
        Δ_t^{⊥u}; c_t = ‖(I − P_{S_t}) Δ_t^{⊥u}‖ is the part of it that leaves the context and sin θ_t = c_t / ‖Δ_t^{⊥u}‖.
        Over scorable tokens (full m-window and a defined u, i.e. t ≥ max(m, 2) and Δ_{t-1} ≠ 0), with w_t = ‖Δ_t^{⊥u}‖:
        R_m = Σ_t c_t / Σ_t w_t = Σ_t w_t sin θ_t / Σ_t w_t ∈ [0, 1]. Returns a scalar tensor (NaN if nothing is scorable).
        """
        hidden_states = hidden_states.float()
        index = torch.arange(max(self.window, 2), hidden_states.shape[0], device=self.device)
        if index.numel() == 0:
            return torch.tensor(float("nan"))
        q, _ = self.context_basis(hidden_states, torch.arange(-self.window, 0, device=self.device))
        q = q[index]  # full windows only, so every column is in range
        step = hidden_states[index] - hidden_states[index - 1]  # Δ_t
        previous_step = hidden_states[index - 1] - hidden_states[index - 2]  # Δ_{t-1}
        previous_norm = previous_step.norm(dim=-1, keepdim=True)
        u = previous_step / previous_norm.clamp_min(self.eps)
        turn = step - (step * u).sum(dim=-1, keepdim=True) * u  # Δ_t^{⊥u}
        in_context = torch.einsum("tdk,tk->td", q, torch.einsum("tdk,td->tk", q, turn))  # P_S Δ_t^{⊥u}
        defined = previous_norm.squeeze(-1) > self.eps
        new_part = (turn - in_context).norm(dim=-1)[defined]  # c_t
        weight = turn.norm(dim=-1)[defined]  # w_t
        if weight.sum() <= self.eps:
            return torch.tensor(float("nan"))
        return new_part.sum() / weight.sum()

    def cross_layer_scores(self, hidden_states: torch.Tensor, lower_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Context C_t = [h^{l-g}_{t-m}, ..., h^{l-g}_{t-1}, h^{l-g}_t] from the lower layer, so u_t is the part of h^l_t
        not explained by the token's own lower-layer state and its m predecessors there; significance divides by m + 2.
        """
        offsets = torch.arange(-self.window, 1, device=self.device)
        return self.context_scores(hidden_states, offsets, context_states=lower_states)

    def score_layer(self, hidden_states: torch.Tensor, lower_states: torch.Tensor | None = None) -> dict[str, float]:
        """Mean GEM novelty and significance over tokens (and, for preceding, the exploration ratio)."""
        hidden_states = hidden_states.to(self.device)
        if hidden_states.shape[0] < 2:
            return {name: float("nan") for name in self.score_names}
        if self.method == "cross_layer":
            token_scores = self.cross_layer_scores(hidden_states, lower_states.to(self.device))
        elif self.method == "preceding_context":
            token_scores = self.preceding_context_scores(hidden_states)
        else:
            token_scores = self.surrounding_context_scores(hidden_states)
        scores = {f"{self.method}_{name}": value.mean().item() for name, value in token_scores.items()}
        if f"{self.method}_exploration_ratio" in self.score_names:
            scores[f"{self.method}_exploration_ratio"] = self.exploration_ratio(hidden_states).item()
        return scores

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
        """Mean of each score for every transformer layer: {name: (L,)}."""
        # Omit the embedding output and use transformer layers; hidden_states[i + 1] is transformer layer i.
        per_layer = []
        for layer in range(len(hidden_states) - 1):
            if self.method != "cross_layer":
                per_layer.append(self.score_layer(hidden_states[layer + 1]))
            elif layer + 1 - self.layer_gap < 0:  # no lower layer (the embeddings are index 0)
                per_layer.append({name: float("nan") for name in self.score_names})
            else:
                per_layer.append(self.score_layer(hidden_states[layer + 1], hidden_states[layer + 1 - self.layer_gap]))
        return {name: np.asarray([layer[name] for layer in per_layer]) for name in self.score_names}

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (too-short texts give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

    def run(self, args: Namespace) -> dict[str, dict]:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        residuals_by_score = {name: [] for name in self.score_names}
        for item in tqdm(test_data, desc=f"Collecting {self.method} scores (m={self.window})"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            for name, values in self.score(hidden_states).items():
                residuals_by_score[name].append(values)

        return {
            name: self.report(args, name, labels, np.asarray(residuals, dtype=float))  # (N, L)
            for name, residuals in residuals_by_score.items()
        }

    def report(self, args: Namespace, name: str, labels: np.ndarray, residuals: np.ndarray) -> dict:
        """Evaluate one score per layer and write its JSON."""
        # Negative so that higher = tokens better captured by the subspace (less novel).
        scores = -1 * residuals
        scores_ratio = -1 * residuals / residuals[:, :1]

        metrics_per_layer = [
            self.evaluate(labels, scores[:, layer]) if np.isfinite(scores[:, layer]).any() else None
            for layer in range(scores.shape[1])
        ]
        auroc_per_layer = [None if m is None else m["auroc"] for m in metrics_per_layer]
        # Layer 0 is the reference itself (ratio = -1 for every text), so it is not evaluated.
        # Layers without a finite score for any text (e.g. cross_layer below the gap) are not evaluated either.
        metrics_per_layer_ratio = [None] + [
            self.evaluate(labels, scores_ratio[:, layer]) if np.isfinite(scores_ratio[:, layer]).any() else None
            for layer in range(1, scores_ratio.shape[1])
        ]
        auroc_per_layer_ratio = [None if m is None else m["auroc"] for m in metrics_per_layer_ratio]
        metrics = metrics_per_layer[args.layer]
        metrics_ratio = metrics_per_layer_ratio[args.layer]
        print(json.dumps({
            "score": name,
            "metrics": metrics,
            "metrics_ratio": metrics_ratio,
            "auroc_per_layer": auroc_per_layer,
            "auroc_per_layer_ratio": auroc_per_layer_ratio,
        }, indent=4))

        setting = f"m{self.window}_g{self.layer_gap}" if self.method == "cross_layer" else f"m{self.window}"
        file_name = f"{name}_{setting}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": name,
            "metrics": metrics,
            "metrics_ratio": metrics_ratio,
            "metrics_per_layer": metrics_per_layer,
            "metrics_per_layer_ratio": metrics_per_layer_ratio,
            "auroc_per_layer": auroc_per_layer,
            "auroc_per_layer_ratio": auroc_per_layer_ratio,
            "scores": scores.tolist(),
            "scores_ratio": scores_ratio.tolist(),
            "labels": labels.tolist(),
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        return metrics


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--method", type=str, default="surrounding_context", choices=METHODS)
    parser.add_argument("--window", type=int, default=3,
                        help="Context size m: neighbours on either side (surrounding) or preceding states "
                             "(preceding, cross_layer).")
    parser.add_argument("--layer_gap", type=int, default=1, help="cross_layer: context taken from layer l − g.")
    parser.add_argument("--layer", type=int, default=10,
                        help="Transformer layer (embeddings excluded) reported as the main metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = SurroundingContextNovelty(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
