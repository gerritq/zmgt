import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data, return_device

from src.config import Config
cfg = Config()

# AUROCs are computed on the negated score (higher −score = machine).
SIGN = -1
# Within-layer angle and within/across-layer angle combinations (see AngleNorm docstring).
COMBOS = ("angle", "withinacross", "rms", "joint", "posnorm", "anglepernorm", "tokenpernorm", "zscore", "ratio", "layerratio")
# zscore: standardise each axis over tokens by median and IQR instead of mean and SD.
ROBUST = False
# Fixed token position t_0 (1-based) at which the per-text norm-position line is read off (posnorm).
T0 = 200
# References c_t of token t: the previous state, or the mean of the last CONTEXT_K states.
REFERENCES = ("previous", "context")
CONTEXT_K = 3
# Scores as they are, or relative to the same text's layer-1 score (log ratio to the first layer).
VARIANTS = ("raw", "first")


class AngleNorm():
    """
    With h_t^ℓ the hidden state of token t at transformer layer ℓ = 1..L (embedding output omitted), c_t^ℓ its reference
    (previous: h_{t−1}^ℓ; context: mean(h_{t−3}^ℓ, h_{t−2}^ℓ, h_{t−1}^ℓ)), θ[ℓ, t] = ∠(c_t^ℓ, h_t^ℓ) the within-layer
    angle and φ[ℓ, t] = ∠(h_t^{ℓ−1}, h_t^ℓ) the across-layer angle of the same token into layer ℓ (h^0: embedding output),
    over the N tokens t that have a reference. Both angles are smaller for machine text. Per text, one score per layer;
    higher = human:
        angle:            log mean_t θ[ℓ, t]
        withinacross:     log mean_t θ[ℓ, t] + log mean_t φ[ℓ, t]
        rms:              log (1/N) Σ_t √((θ[ℓ, t]² + φ[ℓ, t]²) / 2)
        joint:            log (1/N) Σ_t √(θ[ℓ, t] φ[ℓ, t])
        posnorm:          −log α̂^ℓ, the position-adjusted norm: per text and layer, the OLS fit
            r_t = α + β (t − t_0) + ε_t of r_t = ‖h_t^ℓ‖ on the token position t, so α̂ = r̄ − β̂ (t̄ − t_0) is the
            fitted norm at the fixed position t_0 = T0 for every text. Fit over t ≥ 2: the first token is the
            attention sink (‖h‖ ≈ 480 vs. ≈ 20). The same for both references; negated because machine text has the
            larger norm.
        anglepernorm:     ½ withinacross + posnorm = log (√(θ̄^ℓ φ̄^ℓ) / α̂^ℓ)
            the geometric-mean angle per unit of position-adjusted norm, with θ̄^ℓ = mean_t θ[ℓ, t] and
            φ̄^ℓ = mean_t φ[ℓ, t]. Machine text turns less (smaller angles) on larger states (larger norm), so both
            push it down. Half of withinacross gives the angle pair and the norm equal weight; as a ratio it is
            scale-free and needs no statistics over other texts.
        tokenpernorm:     log mean_t (√(θ[ℓ, t] φ[ℓ, t]) / n̄_t),  n̄_t = mean_{ℓ'=2..L−1} ‖h_t^ℓ'‖
            each token's joint angle divided by its own norm averaged over the layers, then the mean over tokens:
            removes the token-identity scale (rarer, longer tokens have larger norms) token by token. The first and
            last layer are left out of n̄_t: the last layer's norm (≈ 140) is far larger than the rest and would dominate.
        zscore:           log std_t s_t,  s_t = θ̃[ℓ, t] + φ̃[ℓ, t]
            each axis standardised per text and layer over its tokens (mean and SD, or median and IQR if ROBUST).
            mean_t s_t is 0 by construction, so the spread is the score; with mean and SD, std_t s_t = √(2 + 2ρ),
            ρ the within-text correlation of θ and φ over tokens.
        ratio:            log std_t s_t,  s_t = θ[ℓ, t] / θ̄^ℓ + φ[ℓ, t] / φ̄^ℓ
            two scale-free ratios around 1; mean_t s_t is 2 by construction, so again the spread is the score
            (the coefficients of variation of θ and φ and their correlation).
        layerratio:       log mean_t (θ[ℓ, t] / θ̄ + φ[ℓ, t] / φ̄) = log (θ̄^ℓ / θ̄ + φ̄^ℓ / φ̄)
            each axis divided by its own mean over all layers and tokens of the same text (θ̄, φ̄), so both are
            ratios around 1 with equal weight; self-contained per text. Removes the text's overall angle level and
            keeps how layer ℓ compares with the text's other layers.
    first: s^ℓ − s^1, every score relative to the same text's layer-1 score (a log ratio; layer 1 is constant), except
    anglepernorm: ½ (withinacross^ℓ − withinacross^1) + posnorm^ℓ, only the angle part relative to layer 1.
    Every score is in log space so that its first variant is a ratio too; the log does not change the AUROC.
    """

    METHOD = "angle_norm"

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model)
        self.device = return_device()

    @staticmethod
    def angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Angle between two (..., D) tensors along the last dimension."""
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    @staticmethod
    def reference(h: torch.Tensor, reference: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Current states h_t and their references c_t along the tokens: (L, T, D) -> (L, T', D), (L, T', D)."""
        if reference == "previous":
            return h[:, 1:], h[:, :-1]
        # Row j of the pooled states = mean(h_j, ..., h_{j+k−1}), the reference of token j + k.
        pooled = h.unfold(1, CONTEXT_K, 1).mean(dim=-1)
        return h[:, CONTEXT_K:], pooled[:, :-1]

    @staticmethod
    def position_adjusted_norm(h: torch.Tensor) -> torch.Tensor:
        """Per-layer intercept α̂ at t_0 of the OLS line ‖h_t‖ = α + β (t − t_0), over t ≥ 2: (L, T, D) -> (L,)."""
        r = h[:, 1:].norm(dim=-1)  # (L, T − 1), positions t = 2..T
        t = torch.arange(2, h.shape[1] + 1, device=h.device, dtype=r.dtype)
        t_centered, r_centered = t - t.mean(), r - r.mean(dim=1, keepdim=True)
        beta = (r_centered * t_centered).sum(dim=1) / (t_centered ** 2).sum()
        return r.mean(dim=1) - beta * (t.mean() - T0)

    @staticmethod
    def standardise(x: torch.Tensor) -> torch.Tensor:
        """Per-layer standardisation over the tokens, by mean and SD or median and IQR: (L, T') -> (L, T')."""
        if ROBUST:
            q = torch.tensor([0.25, 0.5, 0.75], device=x.device, dtype=x.dtype)
            q1, median, q3 = torch.quantile(x, q, dim=1, keepdim=True)
            return (x - median) / (q3 - q1).clamp_min(1e-12)
        return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-12)

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
        """{f"{reference}_{variant}_{combo}": (L,) per-layer scores}; NaN if the text is too short."""
        h_all = torch.stack(hidden_states).to(self.device).float()  # (L + 1, T, D), with the embedding output
        h = h_all[1:]  # (L, T, D)
        n_layers = h.shape[0]
        # Across-layer angle of each token into layer ℓ: (L, T).
        across = self.angle(h_all[:-1], h_all[1:]).clamp_min(1e-12)
        # Position-adjusted norm per layer: (L,); the line needs two tokens after the sink.
        pos_norm = (-self.position_adjusted_norm(h).clamp_min(1e-12).log() if h.shape[1] >= 3
                    else torch.full((n_layers,), float("nan"), device=h.device))
        out = {}
        for reference in REFERENCES:
            if h.shape[1] <= (1 if reference == "previous" else CONTEXT_K):
                for variant in VARIANTS:
                    for combo in COMBOS:
                        out[f"{reference}_{variant}_{combo}"] = np.full(n_layers, np.nan)
                continue
            current, ref = self.reference(h, reference)
            log_angle = self.angle(ref, current).clamp_min(1e-12).log()  # (L, T')
            # Across-layer angles of the same tokens t that have a reference.
            across_current = across[:, h.shape[1] - current.shape[1]:]  # (L, T')
            log_mean_angle = log_angle.exp().mean(dim=1).log()
            combos = {
                "angle": log_mean_angle,
                "withinacross": log_mean_angle + across_current.mean(dim=1).log(),
                "rms": ((log_angle.exp() ** 2 + across_current ** 2) / 2).sqrt().mean(dim=1).log(),
                "joint": (log_angle.exp() * across_current).sqrt().mean(dim=1).log(),
                "posnorm": pos_norm,
            }
            combos["anglepernorm"] = 0.5 * combos["withinacross"] + pos_norm
            # Each token's norm averaged over layers 2..L−1, for the same tokens t: (T',).
            token_norm = h[1:-1, h.shape[1] - current.shape[1]:].norm(dim=-1).mean(dim=0)
            combos["tokenpernorm"] = ((log_angle.exp() * across_current).sqrt() / token_norm).mean(dim=1).log()
            within = log_angle.exp()
            combos["zscore"] = (self.standardise(within) + self.standardise(across_current)).std(dim=1).log()
            combos["ratio"] = (within / within.mean(dim=1, keepdim=True)
                               + across_current / across_current.mean(dim=1, keepdim=True)).std(dim=1).log()
            combos["layerratio"] = (within / within.mean() + across_current / across_current.mean()).mean(dim=1).log()
            first = {combo: per_layer - per_layer[0] for combo, per_layer in combos.items()}
            # anglepernorm, first: withinacross relative to layer 1, posnorm as it is (its own first variant flips).
            first["anglepernorm"] = 0.5 * first["withinacross"] + pos_norm
            for combo, per_layer in combos.items():
                out[f"{reference}_raw_{combo}"] = per_layer.cpu().numpy()
                out[f"{reference}_first_{combo}"] = first[combo].cpu().numpy()
        return out

    @staticmethod
    def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
        """Evaluate on texts with a finite score (too-short texts give NaN)."""
        mask = np.isfinite(scores)
        metrics = evaluation(labels[mask], scores[mask])
        metrics["n_scored"] = int(mask.sum())
        return metrics

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
        metrics = {}
        for reference in REFERENCES:
            for variant in VARIANTS:
                scores = {combo: values[f"{reference}_{variant}_{combo}"] for combo in COMBOS}
                metrics[f"{reference}_{variant}"] = self.print_table(
                    labels, scores, f"reference: {reference}, score: {variant}")
        return metrics

    def print_table(self, labels: np.ndarray, scores: dict[str, np.ndarray], title: str) -> dict[str, list[dict]]:
        """Per-layer AUROC (TPR@5%) table, one column per score, best layer marked; constant layers show --."""
        results = {
            name: [self.evaluate(labels, SIGN * v[:, layer])
                   if np.isfinite(v[:, layer]).any() and np.nanstd(v[:, layer]) > 0 else None
                   for layer in range(v.shape[1])]
            for name, v in scores.items()
        }
        best = {name: int(np.nanargmax([m["auroc"] if m else np.nan for m in r])) for name, r in results.items()}
        print(f"\n{title} | AUROC (TPR@5%) per layer; * = best layer")
        print(f"{'layer':>5}" + "".join(f"{name:>18}" for name in scores))
        for layer in range(len(next(iter(results.values())))):
            cells = []
            for name, r in results.items():
                m = r[layer]
                mark = "*" if layer == best[name] else " "
                cells.append(f"{m['auroc']:>10.3f} ({m['tpr_at_fpr_0_05']:.3f}){mark}" if m else "--")
            print(f"{layer + 1:>5}" + "".join(f"{c:>18}" for c in cells))
        return results


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = AngleNorm(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
