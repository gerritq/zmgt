import os
import json
import importlib
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import evaluation, load_data, return_device

from src.config import Config
cfg = Config()

BASE_MODELS = importlib.import_module("src.ideas.3_token_change_2m").BASE_MODELS


class ContrastiveStraighteningTokens():
    """Per-layer curvature across tokens under the instruct model relative to its base model."""

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

    @staticmethod
    def _angles(previous: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
        """Row-wise angle between two (N, D) tensors."""
        denominator = (previous.norm(dim=-1) * following.norm(dim=-1)).clamp_min(1e-12)
        cosine = ((previous * following).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    def curvature(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Curvature of each layer's trajectory across tokens: (L, T, D) -> (L,)."""
        hidden_states = hidden_states.float()
        if self.args.curvature == "dv":
            # Hosseini & Fedorenko (2023): angle between token steps v_k = x_{k+1} - x_k.
            hidden_states = hidden_states[:, 1:] - hidden_states[:, :-1]
        # Angles between consecutive tokens (L, T'), averaged over tokens.
        return self._angles(hidden_states[:, :-1], hidden_states[:, 1:]).mean(dim=-1)

    def layer_curvature(self, model: AutoModelForCausalLM, inputs: dict) -> torch.Tensor:
        """Per-layer curvature across tokens for the transformer layers of one model: (L,)."""
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        # Omit the embedding output and use transformer layers: (L, T, D).
        hidden_states = torch.stack(outputs.hidden_states[1:]).squeeze(1)
        curvatures = self.curvature(hidden_states)
        if self.args.normalize == "first":
            # Curvature of every later layer relative to the first transformer layer: (L-1,).
            curvatures = curvatures[1:] / curvatures[:1].clamp_min(1e-12)
        return curvatures

    def score(self, text: str) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Return (score, per-layer score, instruct curvatures, base curvatures) for one text."""
        if not text.strip():
            raise ValueError("Input text must be non-empty.")
        inputs = self.tokenizer(text,
                                truncation=True,
                                add_special_tokens=False,
                                max_length=1024,
                                return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Per-layer curvature across tokens under each model: (L,).
        curv_inst = self.layer_curvature(self.inst_model, inputs)
        curv_base = self.layer_curvature(self.base_model, inputs)
        if self.args.combine == "ratio":
            contrast = curv_inst / curv_base.clamp_min(1e-12)
        else:
            contrast = curv_inst - curv_base
        # Straighter under the instruct model than under the base model is taken as machine-like.
        score_per_layer = -1 * contrast.cpu().numpy()
        # Aggregate over layers.
        return (float(score_per_layer.mean()),
                score_per_layer,
                curv_inst.cpu().numpy(),
                curv_base.cpu().numpy())

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        scores, scores_per_layer, curvs_inst, curvs_base = [], [], [], []
        for item in tqdm(test_data, desc="Collecting contrastive straightening scores"):
            score, score_per_layer, curv_inst, curv_base = self.score(item["text"])
            scores.append(score)
            scores_per_layer.append(score_per_layer)
            curvs_inst.append(curv_inst)
            curvs_base.append(curv_base)
        scores = np.asarray(scores, dtype=float)
        scores_per_layer = np.asarray(scores_per_layer, dtype=float)  # (N, L)

        metrics = evaluation(labels, scores)
        metrics_per_layer = [evaluation(labels, scores_per_layer[:, layer]) for layer in range(scores_per_layer.shape[1])]
        print(json.dumps({"auroc_per_layer": [m["auroc"] for m in metrics_per_layer], "metrics": metrics}, indent=4))

        method = f"curv_tokens_2m_{args.curvature}_{args.combine}"
        if args.normalize == "first":
            method += "_norm"
        file_name = f"{method}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "inst_model": self.inst_id,
            "base_model": self.base_id,
            "metrics": metrics,
            "metrics_per_layer": metrics_per_layer,
            "scores": scores.tolist(),
            "scores_per_layer": scores_per_layer.tolist(),
            "curvatures_inst": np.asarray(curvs_inst).tolist(),
            "curvatures_base": np.asarray(curvs_base).tolist(),
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
    parser.add_argument("--curvature", type=str, choices=("dv", "hs"), default="hs",
                        help="Across tokens, per layer. dv: angles between token steps; hs: angles between hidden states.")
    parser.add_argument("--combine", type=str, choices=("ratio", "diff"), default="ratio",
                        help="ratio: C_inst / C_base; diff: C_inst - C_base.")
    parser.add_argument("--normalize", type=str, choices=("none", "first"), default="first",
                        help="first: divide each model's curvature by its first-layer curvature before comparing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = ContrastiveStraighteningTokens(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
