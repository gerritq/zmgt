"""
Across-layer (inter-layer) angle and norm per text, as a contour plot. Per token t, on the raw hidden states:
    angle: ∠(h_t^{ℓ−1}, h_t^ℓ), the angle to the same token's state at the previous layer, averaged over the layer steps
    norm:  ‖h_t^ℓ‖₂, averaged over the layers
then each averaged over the text's tokens, giving one point (mean norm, mean angle) per text, with Gaussian KDE
contours for the human and the machine texts. Transformer layers 1..L−1: the last hidden state is already normed by
HF (the logit input), so its scale differs from the rest. The first SKIP tokens are left out, as in viz.py.
The title gives the AUC P(M > H) of each per-text mean (Mann-Whitney U / n_H n_M) and the Pearson r between them over
all texts; the legend gives r per group.
Also, for the token-balanced datasets (HEATMAP_DATASETS, every text the same number of tokens), token × layer heatmaps
of two quantities per token position t and layer ℓ = 2..L−1 (the first and last layers left out), each the mean over
the texts of one group (2 × 2 panels: rows angle and norm, columns human and machine):
    angle: ∠(h_t^{ℓ−1}, h_t^ℓ), against the same token at the previous layer
    norm:  ‖h_t^ℓ‖₂
Both groups of a quantity share one colour range, the HEATMAP_PERCENTILES of their values, so the attention sink and the
last layer do not wash it out.
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

# Datasets with the same number of tokens in every text, for which the token × layer heatmaps are drawn.
HEATMAP_DATASETS = ("drlXDomainLen_wiki_500_tokens",)
HEATMAP_METRICS = ("angle", "norm")
HEATMAP_LABELS = {
    "angle": r"$\angle(h_t^{\ell-1}, h_t^\ell)$ (rad)",
    "norm": r"$\|h_t^\ell\|_2$",
}
# Lower and upper percentile of each quantity's values (both groups) that bound its colour range.
HEATMAP_PERCENTILES = (2, 98)


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


def token_layer_maps(hidden_states: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    """
    Per layer ℓ = 2..L−1 and token position: the angle against layer ℓ − 1 and the norm, {metric: (L − 2, T)}.
    """
    h = torch.stack(hidden_states[1:]).float()  # (L, T, D), transformer layers 1..L
    return {
        "angle": GeometricMetrics.angle(h[1:-1], h[:-2]).cpu().numpy(),
        "norm": h[1:-1].norm(dim=-1).cpu().numpy(),
    }


def mean_maps(labels: np.ndarray, maps: list[dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """
    Per group, the mean over its texts of the token × layer maps, NaN-padded to the longest text:
    {group: {metric: (L, T)}}.
    """
    n_layers, n_max = maps[0]["angle"].shape[0], max(m["angle"].shape[1] for m in maps)
    out = {}
    for label, group, _ in GROUPS:
        rows = [m for m, l in zip(maps, labels) if l == label]
        out[group] = {}
        for metric in HEATMAP_METRICS:
            stacked = np.full((len(rows), n_layers, n_max), np.nan)
            for j, m in enumerate(rows):
                stacked[j, :, :m[metric].shape[1]] = m[metric]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[group][metric] = np.nanmean(stacked, axis=0)
    return out


def collect(args: Namespace, heatmap: bool) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """
    Labels (N,), the depth profiles {metric: (N, steps or layers)} and, if heatmap, the per-group mean token × layer
    maps (see mean_maps; else {}). Every token has every layer, so the per-text mean over tokens of each token's mean
    over layers is the mean of its profile.
    """
    inference = Inference(model_name=args.model)
    test_data = load_data(args=args)["test"]
    labels, per_text, maps = [], [], []
    for item in tqdm(test_data, desc="Collecting across-layer angle and norm"):
        hidden_states = inference.run(item, args)["hidden_states"]
        per_text.append(depth_profiles(hidden_states))
        if heatmap:
            maps.append(token_layer_maps(hidden_states))
        labels.append(int(item["label"]))
    labels = np.asarray(labels)
    profiles = {metric: np.stack([t[metric] for t in per_text]) for metric in METRICS}
    return labels, profiles, (mean_maps(labels, maps) if heatmap else {})


def plot_heatmap(maps: dict[str, dict[str, np.ndarray]], title: str, path: str) -> None:
    """
    2 × 2 panels, rows angle and norm, columns human and machine: token position (x) by layer (y), the
    group's mean over texts as colour, one colour range per row.
    """
    fig, axes = plt.subplots(len(HEATMAP_METRICS), len(GROUPS), figsize=(7 * len(GROUPS), 4 * len(HEATMAP_METRICS)),
                             squeeze=False, layout="constrained")
    for row, metric in enumerate(HEATMAP_METRICS):
        vmin, vmax = np.nanpercentile(np.stack([maps[group][metric] for _, group, _ in GROUPS]), HEATMAP_PERCENTILES)
        for column, (_, group, _) in enumerate(GROUPS):
            ax, values = axes[row][column], maps[group][metric]  # (L, T)
            image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                              interpolation="nearest",
                              extent=(0.5, values.shape[1] + 0.5, 1.5, values.shape[0] + 1.5))  # layers 2..L−1
            ax.set_title(group.capitalize())
            ax.set_xlabel("Token position")
            ax.set_ylabel(r"Layer $\ell$")
        fig.colorbar(image, ax=axes[row].tolist(), label=HEATMAP_LABELS[metric])
    fig.suptitle(title, fontsize=9)
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
    heatmap = args.dataset in HEATMAP_DATASETS
    labels, profiles, maps = collect(args, heatmap)
    values = {metric: profile.mean(axis=1) for metric, profile in profiles.items()}  # (N,) per-text means
    plot(labels, values,
         f"{args.model}, {args.dataset}: across-layer mean angle vs. mean norm per text\n(raw states, layers 1..L-1, "
         f"first {SKIP} tokens left out); contours at "
         f"{', '.join(f'{level:.0%}' for level in CONTOUR_LEVELS)} of each group's KDE maximum",
         os.path.join(OUTPUT_DIR, f"across_layer_angle_vs_norm_contour_{stem}.pdf"))
    if heatmap:
        plot_heatmap(maps,
                     f"{args.model}, {args.dataset}: across-layer angle and norm per token and layer, "
                     f"mean over each group's texts (raw states, layers 2..L-1; angle vs. previous layer); "
                     f"colour range per row: percentiles {HEATMAP_PERCENTILES[0]}-{HEATMAP_PERCENTILES[1]}",
                     os.path.join(OUTPUT_DIR, f"across_layer_token_layer_heatmap_{stem}.pdf"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_500_tokens")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
