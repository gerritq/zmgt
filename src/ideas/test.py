"""Layer-wise logit-lens trajectories of the observed token.

For every token x_t and layer ℓ = 1..L (the transformer-layer outputs), apply the logit lens at position t − 1:

  p_ℓ(· | x_<t) = softmax(W_U LN(h_{t−1}^ℓ))

(layer L uses the model's own logits, its hidden state is already normed) and track

  q_t^ℓ = log p_ℓ(x_t)                       observed-token log-probability
  r_t^ℓ = rank of x_t under p_ℓ (0 = top-1)  observed-token rank
  H_t^ℓ = −Σ_v p_ℓ(v) log p_ℓ(v)             predictive entropy

From these, per token:

  norm      q̃_t^ℓ = (q_t^ℓ − q_t^1) / (q_t^L − q_t^1)   0 at layer 1, 1 at the final layer
  rank_dev  d_t^ℓ = |r_t^ℓ − r_t^L|                    absolute rank deviation from the final layer

Tokens with |q_t^L − q_t^1| < MIN_SPAN are left out of norm, since the ratio blows up there. Each text is averaged
over its tokens, then texts are averaged per group. One PDF with a panel per measure (and entropy) goes to output/test.
"""

import os
from argparse import ArgumentParser, Namespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.utils import load_data

cf = Config()

OUTPUT_DIR = os.path.join(cf.base_dir, "output", "test")
GROUPS = ((0, "human", "#2a78d6"), (1, "machine", "#eb6834"))
MIN_SPAN = 0.5
MEASURES = (
    ("norm", "normalised log p  (q − q¹) / (qᴸ − q¹)", False),
    ("rank_dev", "mean |rank − final rank|", True),
    ("entropy", "mean predictive entropy (nats)", False),
)


class LayerLens:
    """Per-layer observed-token log-probability, rank and entropy from the logit lens."""

    METHOD = "layer_lens"

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)

    def final_norm(self) -> torch.nn.Module:
        """The norm applied before the unembedding (text-only or multimodal wrapper)."""
        inner = self.inference.model.model
        return inner.norm if hasattr(inner, "norm") else inner.language_model.norm

    @staticmethod
    def token_stats(logits: torch.Tensor, token_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """q, rank and entropy of x_t from position t − 1 for t = 2..T: (T, V), (T,) -> 3 x (T − 1,)."""
        log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
        q = log_probs.gather(-1, token_ids[1:, None])
        rank = (log_probs > q).sum(dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return q.squeeze(-1), rank, entropy

    @torch.no_grad()
    def layer_stats(self, hidden_states: tuple[torch.Tensor, ...], logits: torch.Tensor,
                    token_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """q, rank and entropy for ℓ = 1..L: (L + 1) x (T, D), (T, V), (T,) -> 3 x (L, T − 1)."""
        lm_head = self.inference.model.get_output_embeddings()
        norm = self.final_norm()
        dtype = lm_head.weight.dtype
        device_ids = token_ids.to(self.inference.device)
        rows = []
        for h in hidden_states[1:-1]:
            layer_logits = lm_head(norm(h.to(self.inference.device, dtype)))
            rows.append([stat.cpu() for stat in self.token_stats(layer_logits, device_ids)])
        rows.append(list(self.token_stats(logits, token_ids)))
        q, rank, entropy = (torch.stack(stat).numpy() for stat in zip(*rows))
        return q, rank.astype(np.int64), entropy

    @staticmethod
    def text_curves(q: np.ndarray, rank: np.ndarray, entropy: np.ndarray) -> dict[str, np.ndarray]:
        """One text's (L,) curve per measure, averaged over its tokens; NaN if no token qualifies."""
        n_layers = q.shape[0]
        span = q[-1] - q[0]
        keep = np.abs(span) >= MIN_SPAN
        return {
            "norm": ((q[:, keep] - q[0, keep]) / span[keep]).mean(axis=1) if keep.any()
                    else np.full(n_layers, np.nan),
            "rank_dev": np.abs(rank - rank[-1]).mean(axis=1) if q.shape[1] else np.full(n_layers, np.nan),
            "entropy": entropy.mean(axis=1) if q.shape[1] else np.full(n_layers, np.nan),
        }

    @staticmethod
    def plot_groups(ax: plt.Axes, labels: np.ndarray, per_text: np.ndarray) -> None:
        """Group mean over layers with a ±1 standard-error band: per_text is (N, L)."""
        layers = np.arange(1, per_text.shape[1] + 1)
        for label, name, color in GROUPS:
            rows = per_text[(labels == label) & np.isfinite(per_text).all(axis=1)]
            mean = rows.mean(axis=0)
            se = rows.std(axis=0) / np.sqrt(len(rows))
            ax.plot(layers, mean, color=color, linewidth=2, label=f"{name} (n={len(rows)})")
            ax.fill_between(layers, mean - se, mean + se, color=color, alpha=0.2, linewidth=0)

    def run(self, args: Namespace) -> None:
        args.return_logits = True
        args.return_token_ids = True
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])
        collected = {name: [] for name, _, _ in MEASURES}

        for item in tqdm(test_data, desc=f"Collecting {self.METHOD}"):
            out = self.inference.run(item, args)
            stats = self.layer_stats(out["hidden_states"], out["logits"], out["token_ids"])
            for name, curve in self.text_curves(*stats).items():
                collected[name].append(curve)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, len(MEASURES), figsize=(6 * len(MEASURES), 4))
        for ax, (name, ylabel, log_y) in zip(axes, MEASURES):
            self.plot_groups(ax, labels, np.stack(collected[name]))
            ax.set(xlabel="layer", ylabel=ylabel, title=name)
            if log_y:
                ax.set_yscale("symlog", linthresh=1)  # final layer is 0 by definition
            ax.grid(alpha=0.3)
            ax.legend(frameon=False)
        fig.suptitle(f"{args.model} | {args.dataset} | seed {args.seed}")
        path = os.path.join(OUTPUT_DIR, f"layer_lens_{args.model}_{args.dataset}_s{args.seed}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    LayerLens(args).run(args)


if __name__ == "__main__":
    main()
