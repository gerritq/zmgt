import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data

from src.config import Config
cfg = Config()


class TokenChange():

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)

    @staticmethod
    def score(hidden_states: tuple[torch.Tensor, ...]) -> float:
        """Mean over tokens of each token's mean L2 drift from the first layer."""
        # Omit the embedding output and use transformer layers: (L, T, H).
        hs = torch.stack(hidden_states[1:])
        # Distance of every later layer to the first layer, per token: (L-1, T).
        drift = (hs[1:] - hs[0]).norm(dim=-1)
        token_scores = drift.mean(dim=0)
        return float(token_scores.mean())

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        scores = []
        for item in tqdm(test_data, desc="Collecting token change scores"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            scores.append(self.score(hidden_states))
        scores = np.asarray(scores, dtype=float)

        metrics = evaluation(labels, scores)
        print(json.dumps(metrics, indent=4))

        file_name = f"token_change_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "token_change",
            "metrics": metrics,
            "scores": scores.tolist(),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = TokenChange(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
