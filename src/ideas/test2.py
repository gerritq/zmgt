"""
Per-layer scores of test.py inside the semantic and remaining subspaces of the unembedding W_U, split as in harp_idea
(raw and first). With θ[ℓ, t] = ∠(c_t^ℓ, p_t^ℓ) the within-layer angle to the reference and φ[ℓ, t] = ∠(p_t^{ℓ−1}, p_t^ℓ)
the across-layer angle of the same token into layer ℓ, over the tokens t that have a reference:
  within layer:
    angle:              log mean_t θ[ℓ, t]
    withinacross:       log mean_t θ[ℓ, t] + log mean_t φ[ℓ, t]
    anglepernorm:       ½ withinacross − log α̂^ℓ = log (√(θ̄^ℓ φ̄^ℓ) / α̂^ℓ), α̂^ℓ the position-adjusted norm (test.py)
  across layers:
    across_angle:       log mean_t φ[ℓ, t]
    across_magnitude:   log mean_t ‖p_t^ℓ − p_t^{ℓ−1}‖, the Euclidean step into layer ℓ
    across_anglepernorm: log mean_t (φ[ℓ, t] / n̄_t), n̄_t = mean_{ℓ'=2..L−1} ‖p_t^ℓ'‖, each token's angle divided by
                         its norm averaged over the layers (as tokenpernorm in test.py: the first and last layer left out)
    The across scores leave out the last layer (already normed by HF, the logit input) and show --.
"""

import importlib
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from src.ideas.test import AngleNorm, CONTEXT_K, REFERENCES, VARIANTS
from src.utils import load_data

split_basis = importlib.import_module("src.ideas.harp_idea").RemainingSubspaceCurvature.split_basis

SUBSPACES = ("semantic", "remaining")
COMBOS = ("angle", "withinacross", "anglepernorm", "across_angle", "across_magnitude", "across_anglepernorm")


class SubspaceAngleNorm(AngleNorm):
    """
    AngleNorm on p_t^ℓ = V_Sᵀ (g ⊙ h_t^ℓ), with V_S the right singular vectors of W_U of a subspace S (harp_idea) and
    g the final-norm gain, so the angles equal those of V_Sᵀ Norm(h_t^ℓ) in harp_idea (the RMS rescaling is per
    token and leaves angles unchanged) while ‖p_t^ℓ‖ keeps the norm for posnorm. The last layer is already normed by
    HF and is projected as it is. Across-layer angles into layer 1 start from the projected embedding output.
    """

    METHOD = "angle_norm_unembedding_subspaces"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        model = self.inference.model
        unembedding = model.get_output_embeddings().weight.detach()
        self.bases, _ = split_basis(unembedding)
        self.bases = {subspace: self.bases[subspace].to(self.device) for subspace in SUBSPACES}
        self.gain = model.model.norm.weight.detach().float().to(self.device)

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
        """{f"{subspace}_{reference}_{variant}_{combo}": (L,) per-layer scores}; NaN if the text is too short."""
        h_all = torch.stack(hidden_states).to(self.device).float()  # (L + 1, T, D), with the embedding output
        h_all = torch.cat([h_all[:-1] * self.gain, h_all[-1:]])
        n_layers = h_all.shape[0] - 1
        out = {}
        for subspace, basis in self.bases.items():
            p_all = h_all @ basis  # (L + 1, T, k)
            p = p_all[1:]
            across = self.angle(p_all[:-1], p_all[1:]).clamp_min(1e-12)  # (L, T)
            step = (p_all[1:] - p_all[:-1]).norm(dim=-1).clamp_min(1e-12)  # (L, T): Euclidean step into layer ℓ
            # Each token's norm averaged over layers 2..L−1: (T,); the last layer is normed by HF (as tokenpernorm).
            token_norm = p[1:-1].norm(dim=-1).mean(dim=0).clamp_min(1e-12)
            pos_norm = (-self.position_adjusted_norm(p).clamp_min(1e-12).log() if p.shape[1] >= 3
                        else torch.full((n_layers,), float("nan"), device=p.device))
            for reference in REFERENCES:
                if p.shape[1] <= (1 if reference == "previous" else CONTEXT_K):
                    for variant in VARIANTS:
                        for combo in COMBOS:
                            out[f"{subspace}_{reference}_{variant}_{combo}"] = np.full(n_layers, np.nan)
                    continue
                current, ref = self.reference(p, reference)
                within = self.angle(ref, current).clamp_min(1e-12)  # (L, T')
                # Across-layer quantities of the same tokens t that have a reference (this also drops the first
                # token, the attention sink, whose norm is far larger than the rest).
                tokens = slice(p.shape[1] - current.shape[1], None)
                across_current = across[:, tokens]
                combos = {"angle": within.mean(dim=1).log()}
                combos["withinacross"] = combos["angle"] + across_current.mean(dim=1).log()
                combos["anglepernorm"] = 0.5 * combos["withinacross"] + pos_norm
                combos["across_angle"] = across_current.mean(dim=1).log()
                combos["across_magnitude"] = step[:, tokens].mean(dim=1).log()
                combos["across_anglepernorm"] = (across_current / token_norm[tokens]).mean(dim=1).log()
                # The last layer is already normed by HF (the logit input), so the step into it mixes scales: left out.
                for combo in ("across_angle", "across_magnitude", "across_anglepernorm"):
                    combos[combo][-1] = float("nan")
                first = {combo: per_layer - per_layer[0] for combo, per_layer in combos.items()}
                first["anglepernorm"] = 0.5 * first["withinacross"] + pos_norm
                for combo, per_layer in combos.items():
                    out[f"{subspace}_{reference}_raw_{combo}"] = per_layer.cpu().numpy()
                    out[f"{subspace}_{reference}_first_{combo}"] = first[combo].cpu().numpy()
        return out

    def run(self, args: Namespace) -> dict:
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        per_text = {}
        for item in tqdm(test_data, desc=f"Collecting {self.METHOD}"):
            hidden_states = self.inference.run(item, args)["hidden_states"]
            for name, value in self.score(hidden_states).items():
                per_text.setdefault(name, []).append(value)
        values = {name: np.asarray(v, dtype=float) for name, v in per_text.items()}

        print(f"\n{self.METHOD} | model: {args.model} | dataset: {args.dataset} | seed: {args.seed}")
        print(", ".join(f"{subspace}: {basis.shape[1]} dims" for subspace, basis in self.bases.items()))
        metrics = {}
        for subspace in SUBSPACES:
            for reference in REFERENCES:
                for variant in VARIANTS:
                    scores = {combo: values[f"{subspace}_{reference}_{variant}_{combo}"] for combo in COMBOS}
                    metrics[f"{subspace}_{reference}_{variant}"] = self.print_table(
                        labels, scores, f"subspace: {subspace}, reference: {reference}, score: {variant}")
        return metrics


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = SubspaceAngleNorm(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
