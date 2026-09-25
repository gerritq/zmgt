import torch
import os
import json
import time
import numpy as np
from argparse import Namespace
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data
from datetime import datetime
from argparse import ArgumentParser

from src.config import Config
cfg = Config()

NEGATED_SCORE_NAMES = frozenset({
    "perp_parallel",
    "perp_scaled",
    "context_angle",
})


class CSD():

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)

    def scores(self, hs: torch.Tensor, window: int) -> dict[str, np.ndarray]:
        """Return uncentred token-level CSD score variants."""
        hs = hs.float()
        if hs.shape[0] < window + 1:
            raise ValueError("Not enough hidden states for the requested window.")

        eps = torch.finfo(hs.dtype).eps
        score_names = (
            "perpendicular_ratio",
            "parallel_ratio",
            "squared_perpendicular_ratio",
            "squared_parallel_ratio",
            "perp_parallel",
            "perp_scaled",
            "parallel_scaled",
            "context_angle",
        )
        scores = {name: [] for name in score_names}
        for index in range(window, hs.shape[0]):
            context = hs[index - window:index]
            basis = context.T
            current = hs[index]

            q, r = torch.linalg.qr(basis, mode="reduced")
            diagonal = torch.abs(torch.diag(r))
            tolerance = eps * max(basis.shape) * diagonal.max().clamp_min(eps)
            rank = int((diagonal > tolerance).sum())
            subspace = q[:, :rank]
            current_parallel = (
                subspace @ (subspace.T @ current)
                if rank else torch.zeros_like(current)
            )
            current_perp = current - current_parallel

            perp_norm = current_perp.norm()
            parallel_norm = current_parallel.norm()
            current_norm = current.norm().clamp_min(eps)
            perpendicular_ratio = perp_norm / current_norm
            parallel_ratio = parallel_norm / current_norm
            perp_parallel = perp_norm / parallel_norm.clamp_min(eps)
            context_scale = context.norm(dim=-1).mean().clamp_min(eps)
            perp_scaled = perp_norm / context_scale
            parallel_scaled = parallel_norm / context_scale

            context_angle = torch.acos(parallel_ratio.clamp(-1.0, 1.0))

            values = {
                "perpendicular_ratio": 1 - perpendicular_ratio,
                "parallel_ratio": parallel_ratio,
                "squared_perpendicular_ratio": 1 - (perpendicular_ratio ** 2),
                "squared_parallel_ratio": parallel_ratio ** 2,
                "perp_parallel": perp_parallel,
                "perp_scaled": perp_scaled,
                "parallel_scaled": parallel_scaled,
                "context_angle": context_angle,
            }
            for name, value in values.items():
                if name in NEGATED_SCORE_NAMES:
                    value = -value
                scores[name].append(value)

        return {
            name: torch.stack(values).detach().cpu().numpy()
            for name, values in scores.items()
        }

    @staticmethod
    def metrics_by_layer(labels: np.ndarray, scores: np.ndarray) -> dict[str, dict]:
        metrics = {}
        for layer, layer_scores in enumerate(scores.T):
            valid = np.isfinite(layer_scores)
            if len(np.unique(labels[valid])) == 2:
                metrics[f"layer_{layer}"] = evaluation(labels[valid], layer_scores[valid])
        return metrics

    def run(self, args: Namespace) -> dict[str, dict[str, dict]]:
        """Evaluate every uncentred CSD score separately at every layer."""
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])
        scores_by_method: dict[str, list[list[float]]] = {}
        item_latencies_ms = []
        item_peak_memory_mb = []
        using_cuda = torch.cuda.is_available()

        for item in tqdm(test_data, desc="Collecting CSD scores"):
            if args.benchmark and using_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            if args.benchmark:
                start_time = time.perf_counter()

            # Omit the embedding output and use transformer layers.
            hidden_states = self.inference.run(item, args)["hidden_states"][1:]
            token_scores_by_layer = [
                self.scores(layer, window=args.window)
                for layer in hidden_states
            ]
            for name in token_scores_by_layer[0]:
                scores_by_method.setdefault(name, []).append([
                    float(layer_scores[name].mean())
                    for layer_scores in token_scores_by_layer
                ])
            if args.benchmark:
                if using_cuda:
                    torch.cuda.synchronize()
                    item_peak_memory_mb.append(
                        torch.cuda.max_memory_allocated() / (1024 ** 2)
                    )
                item_latencies_ms.append((time.perf_counter() - start_time) * 1000)

        metrics_by_score = {}
        for name, values in scores_by_method.items():
            raw_scores = np.asarray(values, dtype=float)
            delta_scores = raw_scores - raw_scores[:, [0]]
            ratio_scores = np.divide(
                raw_scores,
                raw_scores[:, [0]],
                out=np.full_like(raw_scores, np.nan),
                where=raw_scores[:, [0]] != 0,
            )
            # A ratio of two negated values loses the detector orientation.
            # Restore it so larger values consistently indicate machine text.
            if name in NEGATED_SCORE_NAMES:
                ratio_scores = -ratio_scores
            metrics_by_score[name] = {
                "raw_metrics_by_layer": self.metrics_by_layer(labels, raw_scores),
                "delta_metrics_by_layer": self.metrics_by_layer(labels, delta_scores),
                "ratio_metrics_by_layer": self.metrics_by_layer(labels, ratio_scores),
            }

        file_name = f"{args.model_name}_{args.dataset}_s{args.seed}_w{args.window}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "csd",
            "metrics_by_score": metrics_by_score,
        }
        if args.benchmark:
            output["mean_latency_per_item_ms"] = float(np.mean(item_latencies_ms))
            output["mean_peak_gpu_memory_per_item_mb"] = (
                float(np.mean(item_peak_memory_mb))
                if item_peak_memory_mb else None
            )

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        if args.benchmark:
            print(f"Mean latency per item: {output['mean_latency_per_item_ms']:.2f} ms")
        if args.benchmark and output["mean_peak_gpu_memory_per_item_mb"] is not None:
            print(
                "Mean peak GPU memory per item: "
                f"{output['mean_peak_gpu_memory_per_item_mb']:.2f} MB"
            )
        return metrics_by_score


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--benchmark",
        type=int,
        choices=(0, 1),
        default=0,
        help="Set to 1 to measure per-item latency and peak CUDA memory.",
    )
    args = parser.parse_args()
    args.benchmark = bool(args.benchmark)
    return args


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = CSD(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
