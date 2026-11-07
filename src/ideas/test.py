"""AUROC per layer of across-layer, within-layer and token-to-mean angles of the hidden states (printed only).

Layers ℓ = 0..L (0 = embeddings, ℓ = output of decoder layer ℓ, taken with forward hooks, so layer L comes before the
final norm). Position 0 (the attention sink) is left out everywhere, also from the text mean. Per token t and layer ℓ,
angles in degrees:
    φ_t = ∠(h_t^{ℓ−1}, h_t^ℓ)            across layers
    θ_t = ∠(h_{t−1}^ℓ, h_t^ℓ)            within a layer, to the previous token
    ψ_t = ∠(h_t^ℓ, m_{−t}^ℓ)             token to the text mean, m_{−t} = mean of h_s^ℓ over the other tokens s ≠ t
Per text and layer ℓ = 1..L (means over tokens):
    cross_angle    mean_t φ
    within_angle   mean_t θ
    joint          mean_t √(θ_t φ_t)     over the tokens with both angles: the sandbox `joint` score (angle_norm in
                                         src/zero.py)
    token_to_mean  mean_t ψ
    joint2         mean_t √(ψ_t φ_t)
Unlike the sandbox, only position 0 is dropped (the sandbox drops the first 10 tokens and truncates at 1024 tokens), and
layer L is taken before the final norm.
AUROC per metric and layer on the test split, label 1 = machine (above 0.5: higher on machine text).
"""

from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.inference import Inference
from src.utils import load_data

# position 0 is skipped and the within-layer angle needs two tokens, so a text needs at least 3 tokens
MIN_TOKENS = 3
METRICS = ("cross_angle", "within_angle", "joint", "token_to_mean", "joint2")


def angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle in degrees between matching rows of a and b (..., d) -> (...)."""
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-12).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos))


def layer_metrics(H: torch.Tensor) -> dict[str, np.ndarray]:
    """(L + 1, T, d) states of layers 0..L, position 0 already removed -> each metric (L,) for layers 1..L."""
    T = H.shape[1]
    phi = angles(H[:-1], H[1:])                            # (L, T)
    theta = angles(H[1:, :-1], H[1:, 1:])                  # (L, T − 1), tokens 1..T − 1
    loo_mean = (H[1:].sum(dim=1, keepdim=True) - H[1:]) / (T - 1)  # (L, T, d), mean of the other tokens
    psi = angles(H[1:], loo_mean)                          # (L, T)
    out = {
        "cross_angle": phi.mean(dim=-1),
        "within_angle": theta.mean(dim=-1),
        "joint": (theta * phi[:, 1:]).sqrt().mean(dim=-1),
        "token_to_mean": psi.mean(dim=-1),
        "joint2": (psi * phi).sqrt().mean(dim=-1),
    }
    return {key: v.cpu().numpy() for key, v in out.items()}


class AngleAUC:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model, args=args)
        self.model, self.tokenizer = self.inference.model, self.inference.tokenizer
        self.device = self.inference.device
        self.decoder = self.model.get_decoder()
        self.n_layers = self.model.config.num_hidden_layers
        self.states: list[torch.Tensor] = []
        self.decoder.embed_tokens.register_forward_hook(self.record)
        for layer in self.decoder.layers:
            layer.register_forward_hook(self.record)

    def record(self, module, inputs, output) -> None:
        self.states.append(output[0] if isinstance(output, tuple) else output)

    @torch.no_grad()
    def text_metrics(self, text: str) -> dict[str, np.ndarray] | None:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if ids.shape[1] < MIN_TOKENS:
            return None
        self.states = []
        self.decoder(input_ids=ids.to(self.device), use_cache=False)
        H = torch.stack(self.states)[:, 0, 1:].float()  # (L + 1, T - 1, d)
        self.states = []
        return layer_metrics(H)

    def collect(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        data = load_data(args=self.args)["test"]
        values, labels = {key: [] for key in METRICS}, []
        for item in tqdm(data, desc="Texts"):
            metrics = self.text_metrics(item["text"])
            if metrics is None:
                continue
            for key in METRICS:
                values[key].append(metrics[key])
            labels.append(item["label"])
        return {key: np.stack(v) for key, v in values.items()}, np.asarray(labels)  # (N, L)

    @staticmethod
    def auroc(labels: np.ndarray, x: np.ndarray) -> float:
        """AUROC over the texts with a value; nan if one class has none."""
        ok = np.isfinite(x)
        if len(np.unique(labels[ok])) < 2:
            return float("nan")
        return float(roc_auc_score(labels[ok], x[ok]))

    def run(self) -> None:
        args = self.args
        values, labels = self.collect()
        aucs = {key: [self.auroc(labels, values[key][:, i]) for i in range(self.n_layers)] for key in METRICS}

        print(f"\nAUROC per layer (machine = 1) | {args.model} {args.dataset} s{args.seed} | "
              f"human={(labels == 0).sum()} machine={(labels == 1).sum()}")
        print(f"{'layer':>6} " + " ".join(f"{key:>13}" for key in METRICS))
        for i in range(self.n_layers):
            print(f"{i + 1:>6} " + " ".join(f"{aucs[key][i]:>13.4f}" for key in METRICS))
        # layer furthest from chance per metric
        best = {key: int(np.nanargmax(np.abs(np.asarray(aucs[key]) - 0.5))) for key in METRICS}
        print(f"{'best':>6} " + " ".join(f"{f'L{best[key] + 1}: {aucs[key][best[key]]:.3f}':>13}"
                                         for key in METRICS))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.model_name = args.model
    return args


if __name__ == "__main__":
    AngleAUC(parse_args()).run()
