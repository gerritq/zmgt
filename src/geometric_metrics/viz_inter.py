"""
Across-layer (inter-layer) angle and norm per text, as a contour plot. Per token t, on the raw hidden states:
    angle: ∠(h_t^{ℓ−1}, h_t^ℓ), the angle to the same token's state at the previous layer, averaged over the layer steps
    norm:  ‖h_t^ℓ‖₂, averaged over the layers
then each averaged over the text's tokens, giving one point (mean norm, mean angle) per text, with Gaussian KDE
contours for the human and the machine texts. Transformer layers 1..L−1: the last hidden state is already normed by
HF (the logit input), so its scale differs from the rest. The first SKIP tokens are left out, as in viz.py.
The title gives the AUC P(M > H) of each per-text mean (Mann-Whitney U / n_H n_M) and the Pearson r between them over
all texts; the legend gives r per group.
Also a depth profile: per text, the across-layer angle of every layer step and the norm of every layer, each averaged
over the tokens, as thin lines; each group's mean with its 95% CI (± 1.96 SE over texts) as a band; and below each
panel the AUC P(M > H) per layer step / layer.
"""

import os
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data
from src.geometric_metrics.metrics import GeometricMetrics
from src.geometric_metrics.viz import CONTOUR_LEVELS, GROUPS, METRICS, OUTPUT_DIR, SKIP, auc, draw_contour


def depth_profiles(hidden_states: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    """
    Per text, over the tokens from SKIP + 1: the mean across-layer angle of every step between layers 1..L−1
    (L − 2,) and the mean norm of every layer 1..L−1 (L − 1,): {metric: profile}; NaN if no token is left.
    """
    h = torch.stack(hidden_states[1:-1])[:, SKIP:].float()  # (L − 1, T', D), transformer layers 1..L−1
    if h.shape[1] == 0:
        return {"angle": np.full(h.shape[0] - 1, np.nan), "norm": np.full(h.shape[0], np.nan)}
    return {
        "angle": GeometricMetrics.angle(h[1:], h[:-1]).mean(dim=1).cpu().numpy(),  # step ℓ → ℓ + 1
        "norm": h.norm(dim=-1).mean(dim=1).cpu().numpy(),
    }


def collect(args: Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Labels (N,) and the depth profiles {metric: (N, steps or layers)}. Every token has every layer, so the per-text
    mean over tokens of each token's mean over layers is the mean of its profile.
    """
    inference = Inference(model_name=args.model)
    test_data = load_data(args=args)["test"]
    labels, per_text = [], []
    for item in tqdm(test_data, desc="Collecting across-layer angle and norm"):
        per_text.append(depth_profiles(inference.run(item, args)["hidden_states"]))
        labels.append(int(item["label"]))
    return np.asarray(labels), {metric: np.stack([t[metric] for t in per_text]) for metric in METRICS}


def plot_depth(labels: np.ndarray, profiles: dict[str, np.ndarray], title: str, path: str) -> None:
    """
    Two columns (angle per layer step, norm per layer): per-text profiles as thin lines, each group's mean with its
    95% CI band, and below each panel the AUC P(M > H) per layer step / layer with the 0.5 line.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex="col", gridspec_kw={"height_ratios": (3, 1)})
    specs = {
        "angle": ("Layer step $\\ell \\to \\ell+1$", r"Mean $\angle(h_t^{\ell}, h_t^{\ell+1})$ (rad)"),
        "norm": ("Layer $\\ell$", r"Mean $\|h_t^\ell\|_2$"),
    }
    for column, (metric, (xlabel, ylabel)) in enumerate(specs.items()):
        ax, ax_auc = axes[0][column], axes[1][column]
        values = profiles[metric]  # (N, X)
        x = np.arange(1, values.shape[1] + 1)
        for label, group, color in GROUPS:
            for row in values[labels == label]:
                ax.plot(x, row, color=color, alpha=0.06, linewidth=0.5)
        for label, group, color in GROUPS:
            rows = values[labels == label]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                curve = np.nanmean(rows, axis=0)
                half_width = 1.96 * np.nanstd(rows, axis=0, ddof=1) / np.sqrt(np.isfinite(rows).sum(axis=0))
            ax.fill_between(x, curve - half_width, curve + half_width, color=color, alpha=0.3, linewidth=0)
            ax.plot(x, curve, color=color, linewidth=1.4, label=f"{group.capitalize()} (n={len(rows)})")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7)
        aucs = [auc(labels, values[:, j]) for j in range(values.shape[1])]
        ax_auc.plot(x, aucs, color="black", marker="o", markersize=3, linewidth=1)
        ax_auc.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
        ax_auc.set_ylim(0, 1)
        ax_auc.set_ylabel("AUC")
        ax_auc.set_xlabel(xlabel)
        ax_auc.set_xticks(x[::2])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def plot(labels: np.ndarray, values: dict[str, np.ndarray], title: str, path: str) -> None:
    """One panel: each text's across-layer mean norm (x) against its across-layer mean angle (y), with contours."""
    fig, ax = plt.subplots(figsize=(6, 4.8))
    draw_contour(ax, labels, values["norm"], values["angle"])
    ax.set_xlabel(rf"Mean over tokens of mean $\|h_t^\ell\|_2$ over layers 1..L$-$1 (from token {SKIP + 1})")
    ax.set_ylabel(r"Mean over tokens of mean $\angle(h_t^{\ell-1}, h_t^\ell)$ (rad)")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = f"{args.model}_{args.dataset}_s{args.seed}"
    labels, profiles = collect(args)
    values = {metric: profile.mean(axis=1) for metric, profile in profiles.items()}  # (N,) per-text means
    plot(labels, values,
         f"{args.model}, {args.dataset}: across-layer mean angle vs. mean norm per text\n(raw states, layers 1..L-1, "
         f"first {SKIP} tokens left out); contours at "
         f"{', '.join(f'{level:.0%}' for level in CONTOUR_LEVELS)} of each group's KDE maximum",
         os.path.join(OUTPUT_DIR, f"across_layer_angle_vs_norm_contour_{stem}.pdf"))
    plot_depth(labels, profiles,
               f"{args.model}, {args.dataset}: across-layer depth profile (raw states, layers 1..L-1, first {SKIP} "
               f"tokens left out); lines: texts, solid: group mean with 95% CI; bottom: AUC P(M > H)",
               os.path.join(OUTPUT_DIR, f"across_layer_depth_profile_{stem}.pdf"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
