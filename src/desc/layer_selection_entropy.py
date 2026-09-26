"""Per-text prompt entropy (Skean et al., 2025, "Layer by Layer"), logit-lens token entropy and token KL to the first layer across layers, coloured by human/machine."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data, return_device

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "desc")
COLORS = {0: "tab:blue", 1: "tab:red"}
NAMES = {0: "Human", 1: "Machine"}
PEAK_MAX_LAYER = 20


def prompt_entropy(hidden_states: torch.Tensor) -> float:
    """
    Matrix-based entropy (α = 1) of the token Gram matrix K = Z Zᵀ / tr(K), normalised by log T.
    ≈ 0 if all tokens collapse onto one direction, ≈ 1 if they spread uniformly.
    """
    n_tokens = hidden_states.shape[0]
    if n_tokens < 2:
        return float("nan")
    hidden_states = hidden_states.double()
    gram = hidden_states @ hidden_states.T
    eigvals = torch.linalg.eigvalsh(gram / gram.trace()).clamp(min=0)
    eigvals = eigvals[eigvals > 0]
    return (-(eigvals * eigvals.log()).sum() / np.log(n_tokens)).item()


def logit_lens(hidden_states: torch.Tensor, inference: Inference, apply_norm: bool) -> torch.Tensor:
    """Unembed each token's hidden state: (T, D) -> next-token log-probs (T, V)."""
    model = inference.model
    hidden_states = hidden_states.to(model.lm_head.weight.dtype)
    with torch.no_grad():
        if apply_norm:
            hidden_states = model.model.norm(hidden_states)
        return torch.log_softmax(model.lm_head(hidden_states).float(), dim=-1)


def token_entropy(log_probs: torch.Tensor) -> float:
    """Mean over tokens of the next-token entropy (nats)."""
    return (-(log_probs.exp() * log_probs).sum(dim=-1)).mean().item()


def token_kl(log_probs: torch.Tensor, log_probs_ref: torch.Tensor) -> float:
    """Mean over tokens of KL(p_ℓ ‖ p_ref) between next-token distributions (nats)."""
    return (log_probs.exp() * (log_probs - log_probs_ref)).sum(dim=-1).mean().item()


def compute_entropy(args: Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prompt entropy, token entropy and token KL to first layer per text and transformer layer: (N, L) each, plus labels (N,)."""
    data = load_data(args=args)[args.split]
    inference = Inference(model_name=args.model)
    device = return_device()

    prompt_entropies, token_entropies, token_kls = [], [], []
    for item in tqdm(data, desc="Computing entropies"):
        # Omit the embedding output and use transformer layers.
        hidden_states = [layer.to(device) for layer in inference.run(item, args)["hidden_states"][1:]]
        prompt_entropies.append([prompt_entropy(layer) for layer in hidden_states])
        entropies, kls, log_probs_first = [], [], None
        for i, layer in enumerate(hidden_states):
            # HF already applies the final norm to the last hidden state.
            log_probs = logit_lens(layer, inference, apply_norm=i < len(hidden_states) - 1)
            if log_probs_first is None:
                log_probs_first = log_probs
            entropies.append(token_entropy(log_probs))
            kls.append(token_kl(log_probs, log_probs_first))
        token_entropies.append(entropies)
        token_kls.append(kls)
    labels = np.asarray([item["label"] for item in data])
    return (np.asarray(prompt_entropies, dtype=float), np.asarray(token_entropies, dtype=float),
            np.asarray(token_kls, dtype=float), labels)


def global_peak(scores: np.ndarray, layers: np.ndarray) -> int:
    """Index of the global maximum."""
    return int(np.nanargmax(scores))


def peak_before(scores: np.ndarray, layers: np.ndarray, max_layer: int = PEAK_MAX_LAYER) -> int:
    """Index of the maximum among layers before `max_layer`."""
    idx = np.flatnonzero(layers < max_layer)
    return int(idx[np.nanargmax(scores[idx])])


def first_local_peak(scores: np.ndarray, layers: np.ndarray) -> int:
    """Index of the first layer higher than both neighbours; falls back to the global maximum if none."""
    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
            return i
    return global_peak(scores, layers)


PEAK_FNS = [
    (global_peak, "Raw peak (global max)"),
    (peak_before, f"Max before layer {PEAK_MAX_LAYER}"),
    (first_local_peak, "First local max"),
]


def draw_row(row: np.ndarray, values: np.ndarray, layers: np.ndarray, labels: np.ndarray, ylabel: str) -> None:
    """One panel per peak definition: a transparent line per text, with its peak marked by a dot."""
    for ax, (peak_fn, desc) in zip(row, PEAK_FNS):
        for scores, label in zip(values, labels):
            if not np.isfinite(scores).any():
                continue
            peak = peak_fn(scores, layers)
            ax.plot(layers, scores, color=COLORS[label], alpha=0.15, linewidth=0.5)
            ax.scatter(layers[peak], scores[peak], color=COLORS[label], alpha=0.4, s=4, linewidths=0, zorder=3)
        ax.set_xlabel("Layer", fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        ax.set_title(desc, fontsize=7)
    row[0].set_ylabel(ylabel, fontsize=7)


def plot(metrics: list[tuple[np.ndarray, np.ndarray, str]], labels: np.ndarray, args: Namespace, path: str) -> None:
    """One row per (values, layers, ylabel) metric, one panel per peak definition."""
    fig, axes = plt.subplots(len(metrics), len(PEAK_FNS), figsize=(9.6, 2.4 * len(metrics)), sharey="row", squeeze=False)
    for row, (values, layers, ylabel) in zip(axes, metrics):
        draw_row(row, values, layers, labels, ylabel)
    for label, name in NAMES.items():
        axes[0, 0].plot([], [], color=COLORS[label], label=name)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"Layer Entropy: {args.model}, {args.dataset}", fontsize=8, y=1.01)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = f"layer_selection_entropy_{args.model}_{args.dataset}_{args.split}_s{args.seed}"

    prompt_entropies, token_entropies, token_kls, labels = compute_entropy(args)
    layers = np.arange(1, prompt_entropies.shape[1] + 1)
    metrics = [
        (prompt_entropies, layers, r"Prompt entropy $H(\ell) / \log T$"),
        (prompt_entropies - prompt_entropies[:, :1], layers,
         r"Prompt entropy $H(\ell) - H(\ell_{\mathrm{first}})$"),
        # Drop transformer layer 1 for token entropy.
        (token_entropies[:, 1:], layers[1:], "Token entropy (logit lens, nats)\nlayer 1 excluded"),
        (token_entropies[:, 1:] - token_entropies[:, :1], layers[1:],
         "Token entropy $H(\\ell) - H(\\ell_{\\mathrm{first}})$\nlayer 1 excluded"),
        (token_kls[:, 1:], layers[1:],
         "Token KL$(p_\\ell \\,\\|\\, p_{\\ell_{\\mathrm{first}}})$ (nats)\nlayer 1 excluded"),
    ]
    plot(metrics, labels, args, os.path.join(OUTPUT_DIR, f"{file_name}.pdf"))


if __name__ == "__main__":
    main()
