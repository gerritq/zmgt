"""Heatmap of the per-text ESS intrinsic dimension of token states across layers."""

import os
import json
import numpy as np
import torch
import skdim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from joblib import Parallel, delayed
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data, return_device

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "desc")


def get_knn_from_torch_distance_matrix(dist_matrix: torch.Tensor, k: int):
    """
    dist_matrix: torch.Tensor of shape (N, N) — pairwise distances
    k: number of nearest neighbors to extract (excluding self-distance)

    Returns:
        knn_dists, knn_neighbors: tensors of shape (N, k)
    """
    assert dist_matrix.shape[0] == dist_matrix.shape[1], "Distance matrix must be square"
    # Exclude the self-distance (0) by sorting and skipping the first column.
    sorted_dists, sorted_neighbors = torch.sort(dist_matrix, dim=1)
    return sorted_dists[:, 1:k+1], sorted_neighbors[:, 1:k+1]


def ess_id(hidden_states: np.ndarray, knn_dists: np.ndarray, knn_neighbors: np.ndarray) -> float:
    """Global ESS ID of the token states (T, D): mean of the local estimates over tokens."""
    return float(skdim.id.ESS().fit_transform(hidden_states, precomputed_knn_arrays=[knn_dists, knn_neighbors]))


def text_ids(hidden_states: tuple[torch.Tensor, ...], args: Namespace, device: torch.device,
             parallel: Parallel) -> list[float]:
    """ESS ID for every transformer layer of one text."""
    # Omit the embedding output and use transformer layers.
    layers = hidden_states[1:]
    if layers[0].shape[0] <= args.k:
        return [float("nan")] * len(layers)

    jobs = []
    for layer in layers:
        layer = layer.to(device)
        knn_dists, knn_neighbors = get_knn_from_torch_distance_matrix(torch.cdist(layer, layer), k=args.k)
        jobs.append(delayed(ess_id)(layer.cpu().numpy(),
                                    knn_dists.double().cpu().numpy(),
                                    knn_neighbors.cpu().numpy()))
    # The local ESS fits are CPU-bound Python, so layers run in separate processes.
    return parallel(jobs)


def compute_ids(args: Namespace) -> np.ndarray:
    """ESS ID per text and transformer layer: (N, L)."""
    data = load_data(args=args)[args.split]
    inference = Inference(model_name=args.model)
    device = return_device()

    ids = []
    with Parallel(n_jobs=args.n_jobs) as parallel:
        for item in tqdm(data, desc="Computing ESS ID"):
            hidden_states = inference.run(item, args)["hidden_states"]
            ids.append(text_ids(hidden_states, args, device, parallel))
    return np.asarray(ids, dtype=float)


def plot(ids: np.ndarray, args: Namespace, path: str) -> None:
    """Heatmap with layers on x and texts on y; each layer column sorted independently."""
    # Descending sort puts NaNs last; flipping gives NaNs on top, then values rising
    # towards the bottom, i.e. high to low from the bottom up.
    sorted_ids = np.sort(-ids, axis=0)[::-1] * -1

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    image = ax.imshow(sorted_ids, aspect="auto", interpolation="nearest", cmap="viridis",
                      extent=(0.5, ids.shape[1] + 0.5, ids.shape[0], 0))
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Texts (sorted per layer)", fontsize=7)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_title(f"ESS ID: {args.model}, {args.dataset}", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.05)
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.set_label("ID", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--k", type=int, default=10, help="Nearest neighbours per local ESS estimate.")
    parser.add_argument("--n_jobs", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", "1")),
                        help="Processes for the per-layer ESS fits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = f"id_heatmap_{args.model}_{args.dataset}_{args.split}_s{args.seed}"

    ids = compute_ids(args)
    with open(os.path.join(OUTPUT_DIR, f"{file_name}.json"), "w") as f:
        json.dump({**vars(args), "ids": ids.tolist()}, f)
    plot(ids, args, os.path.join(OUTPUT_DIR, f"{file_name}.pdf"))


if __name__ == "__main__":
    main()
