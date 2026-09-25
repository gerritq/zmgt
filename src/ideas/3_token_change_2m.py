import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import evaluation, load_data, return_device

from src.config import Config
cfg = Config()

# Base (pretrained-only) counterpart of each instruct model in cfg.model_dict.
BASE_MODELS = {
    "q1.7b": "Qwen/Qwen3-1.7B-Base",
    "q4b": "Qwen/Qwen3-4B-Base",
    "q8b": "Qwen/Qwen3-8B-Base",
    "l1b": "meta-llama/Llama-3.2-1B",
    "l3b": "meta-llama/Llama-3.2-3B",
    "l8b": "meta-llama/Llama-3.1-8B",
    "g1b": "google/gemma-3-1b-pt",
    "g4b": "google/gemma-3-4b-pt",
    "g12b": "google/gemma-3-12b-pt",
}


class TokenChange2M():
    """Per-token divergence between an instruct model and its base model."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.device = return_device()
        self.inst_id = cfg.model_dict[args.model]
        self.base_id = BASE_MODELS[args.model]
        # Both models share the tokenizer, so the same input ids are fed to both.
        self.tokenizer = AutoTokenizer.from_pretrained(self.inst_id)
        self.inst_model = self.load_model(self.inst_id)
        self.base_model = self.load_model(self.base_id)

    def load_model(self, model_id: str) -> AutoModelForCausalLM:
        # bf16 so that both models fit on one GPU.
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map=self.device)
        model.eval()
        return model

    def hidden_states(self, model: AutoModelForCausalLM, inputs: dict) -> torch.Tensor:
        """Return stacked hidden states (L+1, T, H), including the embedding output."""
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        return torch.stack(outputs.hidden_states).squeeze(1).float()

    def score(self, text: str) -> np.ndarray:
        """Mean over tokens of the cosine distance between instruct and base, per layer: (L+1,)."""
        if not text.strip():
            raise ValueError("Input text must be non-empty.")
        inputs = self.tokenizer(text,
                                truncation=True,
                                add_special_tokens=False,
                                max_length=1024,
                                return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        hs_inst = self.hidden_states(self.inst_model, inputs)
        hs_base = self.hidden_states(self.base_model, inputs)
        # Cosine distance per layer and token: (L+1, T).
        distance = 1 - torch.nn.functional.cosine_similarity(hs_inst, hs_base, dim=-1)
        return distance.mean(dim=-1).cpu().numpy()

    def plot(self, aurocs: list[float], path: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(range(len(aurocs)), aurocs, marker="o", markersize=3)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Layer (0 = embeddings)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"Instruct vs. base divergence: {self.args.model}, {self.args.dataset}")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        scores = []
        for item in tqdm(test_data, desc="Collecting instruct vs. base divergence"):
            scores.append(self.score(item["text"]))
        scores = np.asarray(scores, dtype=float)  # (N, L+1)

        metrics_per_layer = [evaluation(labels, scores[:, layer]) for layer in range(scores.shape[1])]
        # Single summary score: divergence averaged over all layers.
        metrics = evaluation(labels, scores.mean(axis=1))
        aurocs = [layer_metrics["auroc"] for layer_metrics in metrics_per_layer]
        print(json.dumps({"auroc_per_layer": aurocs, "metrics": metrics}, indent=4))

        file_name = f"token_change_2m_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "token_change_2m",
            "inst_model": self.inst_id,
            "base_model": self.base_id,
            "metrics": metrics,
            "metrics_per_layer": metrics_per_layer,
            "scores": scores.tolist(),
            "labels": labels.tolist(),
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        self.plot(aurocs, os.path.join(output_dir, f"{file_name}.pdf"))
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

    model = TokenChange2M(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
