"""
Geometry of human vs. machine text, scalars only (no hidden-state vectors are saved).

Setup
    One forward pass per text with all hidden states. Layer ℓ is hidden_states[ℓ]: 0 = embedding output, 1..L =
    transformer-layer outputs. hidden_states[L] has the final norm applied and is dropped, so layers 0..L-1 are used.
    Token mask keep[ℓ, t]: ‖h_t^ℓ‖ ≤ 2 · median_t ‖h_t^ℓ‖ (median over all tokens of the text at ℓ) and t ≥ 10.
    Within-layer pair (t-1, t): both kept at ℓ. Across-layer pair (ℓ-1, ℓ): t kept at both layers.
    Angles in degrees; logs are natural, log(x + EPS); per-token arrays are NaN where masked.

Analysis 1.1 (global: energy split), per text and layer, over the K kept tokens, μ = mean of the kept h_t
    mean_norm, energy = mean ‖h_t‖², shared_energy = ‖μ‖², token_energy = mean ‖h_t − μ‖², s = shared / energy,
    s_corr = (shared − token_energy / (K − 1)) / energy, avg_cos = mean cos over kept pairs i < j, n_kept = K
    (energy = shared_energy + token_energy exactly)
Analysis 1.2 (local: shuffle test), per text and layer, on the kept tokens in order with masked tokens removed
    A_original = mean ∠(h_t, h_{t+1}) over consecutive kept tokens; A_shuffled / A_shuffled_std = mean / std (ddof 0)
    of the same score over N_SHUFFLES random orders (rng seeded with [42, text_index, layer]); L = A_shuffled −
    A_original (> 0: neighbours are closer than random pairs)
Analysis 2 (across layers), per token, ℓ ≥ 1, across-layer pairs only, h = h_t^{ℓ-1}, Δ = h_t^ℓ − h
    across_angle = ∠(h, h + Δ), alpha = Δ·h / ‖h‖², beta = ‖Δ − alpha h‖ / ‖h‖, log_tan = log beta − log(1 + alpha),
    T1 = log ‖Δ‖, T2 = log(‖Δ − alpha h‖ / ‖Δ‖), T3 = −log ‖h‖, T4 = −log(1 + alpha)  (log_tan = T1 + T2 + T3 + T4)
    All NaN where 1 + alpha ≤ 0 (flag_large_angle). alpha and beta come from desc.decompose (relative to h^{ℓ-1}).
Analysis 3 (predictability), per token aligned to t: logprob = log p(x_t | x_<t) (NaN at t = 0), entropy_prev =
    H(p(· | x_<t)) (NaN at t = 0), entropy_next = H(p(· | x_≤t)); per text: ppl_all = exp(−mean logprob) over t ≥ 1,
    mean_entropy_next over t ≥ 10.

One .npz per model, dataset and seed in output/desc3 (see the "meta" entry for each array's shape).
"""

import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data
from src.geometric_metrics.desc import decompose

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "desc3")
MEDIAN_FACTOR = 2.0
MIN_POSITION = 10
N_SHUFFLES = 50
SHUFFLE_SEED = 42
EPS = 1e-12
NAN = float("nan")

ENERGY_SCORES = ("mean_norm", "energy", "shared_energy", "token_energy", "s", "s_corr", "avg_cos")
SHUFFLE_SCORES = ("A_original", "A_shuffled", "A_shuffled_std", "L")
LAYER_SCORES = (*ENERGY_SCORES, *SHUFFLE_SCORES)  # (texts, layers), plus n_kept
ACROSS_SCORES = ("across_angle", "alpha", "beta", "log_tan", "T1", "T2", "T3", "T4")  # (texts, layers, T)
TOKEN_MASKS = ("keep", "flag_large_angle")  # (texts, layers, T) bool
PRED_SCORES = ("logprob", "entropy_prev", "entropy_next")  # (texts, T)
TEXT_SCORES = ("ppl_all", "mean_entropy_next")  # (texts,)


def log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x + EPS)


def angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle in degrees between matching rows: (..., D), (..., D) -> (...)."""
    cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(EPS)
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def token_mask(h: torch.Tensor) -> torch.Tensor:
    """(L, T, D) -> (L, T) bool: ‖h_t‖ ≤ 2 · median_t ‖h_t‖ and t ≥ MIN_POSITION."""
    norms = h.norm(dim=-1)
    keep = norms <= MEDIAN_FACTOR * norms.median(dim=1, keepdim=True).values
    keep[:, :MIN_POSITION] = False
    return keep


def energy_split(x: torch.Tensor, keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """Analysis 1.1. x: (L, T, D) float64, keep: (L, T) -> {score: (L,)}; NaN where undefined (K = 0, or K < 2)."""
    w = keep.to(x.dtype)
    k = w.sum(dim=1)
    k1 = k.clamp_min(1)
    norms = x.norm(dim=-1)
    mu = (x * w[..., None]).sum(dim=1) / k1[:, None]
    energy = (norms ** 2 * w).sum(dim=1) / k1
    shared = (mu ** 2).sum(dim=-1)
    token = (((x - mu[:, None]) ** 2).sum(dim=-1) * w).sum(dim=1) / k1
    unit_sum = (x / norms[..., None].clamp_min(EPS) * w[..., None]).sum(dim=1)
    out = {
        "mean_norm": (norms * w).sum(dim=1) / k1,
        "energy": energy,
        "shared_energy": shared,
        "token_energy": token,
        "s": shared / energy,
        "s_corr": (shared - token / (k - 1)) / energy,
        "avg_cos": ((unit_sum ** 2).sum(dim=-1) - k) / (k * (k - 1)),  # ‖Σ u‖² = K + 2 Σ_{i<j} cos
    }
    for name, value in out.items():
        out[name] = torch.where(k >= (2 if name in ("s_corr", "avg_cos") else 1), value, NAN)
    return out


def shuffle_test(x: torch.Tensor, keep: torch.Tensor, text_index: int) -> dict[str, torch.Tensor]:
    """Analysis 1.2. x: (L, T, D) float64, keep: (L, T) -> {score: (L,)}; NaN where fewer than 2 kept tokens."""
    n_layers = x.shape[0]
    out = {name: torch.full((n_layers,), NAN, dtype=x.dtype) for name in SHUFFLE_SCORES}
    for layer in range(n_layers):
        kept = x[layer, keep[layer]]  # (K, D), original order
        k = kept.shape[0]
        if k < 2:
            continue
        unit = kept / kept.norm(dim=-1, keepdim=True).clamp_min(EPS)
        angles = torch.rad2deg(torch.acos((unit @ unit.T).clamp(-1.0, 1.0))).cpu()  # (K, K)
        rng = np.random.default_rng([SHUFFLE_SEED, text_index, layer])
        order = torch.from_numpy(np.stack([np.arange(k)] + [rng.permutation(k) for _ in range(N_SHUFFLES)]))
        scores = angles[order[:, :-1], order[:, 1:]].mean(dim=1)  # (1 + N_SHUFFLES,)
        out["A_original"][layer] = scores[0]
        out["A_shuffled"][layer] = scores[1:].mean()
        out["A_shuffled_std"][layer] = scores[1:].std(correction=0)
    out["L"] = out["A_shuffled"] - out["A_original"]
    return out


def across_split(x: torch.Tensor, keep: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Analysis 2. x: (L, T, D) float64 -> {score: (L, T)} NaN at layer 0, off across pairs and where flagged; flag."""
    prev, cur = x[:-1], x[1:]
    delta = cur - prev
    parallel, perpendicular = decompose(delta, prev)  # along h = h^{l-1}: Δ·h / ‖h‖ and ‖Δ − alpha h‖
    h_norm = prev.norm(dim=-1)
    alpha = parallel / h_norm
    beta = perpendicular / h_norm
    growth = torch.log1p(alpha.clamp_min(-1 + EPS))
    raw = {
        "across_angle": angle(prev, cur),
        "alpha": alpha,
        "beta": beta,
        "log_tan": log(beta) - growth,
        "T1": log(delta.norm(dim=-1)),
        "T2": log(perpendicular / delta.norm(dim=-1).clamp_min(EPS)),
        "T3": -log(h_norm),
        "T4": -growth,
    }
    pair = keep[1:] & keep[:-1]
    flag = pair & (1 + alpha <= 0)
    used = pair & ~flag
    first = torch.zeros_like(keep[:1])
    scores = {name: torch.cat([torch.full_like(v[:1], NAN), torch.where(used, v, NAN)]) for name, v in raw.items()}
    return scores, torch.cat([first, flag])


def predictability(logits: torch.Tensor, token_ids: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Analysis 3, aligned to the token at t: logits[t - 1] predict x_t. (T, V), (T,) -> {score: (T,)}, per text."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    entropy = torch.special.entr(logp.exp()).sum(dim=-1)  # H(p(· | x_≤t)) at t; 0 · log 0 = 0
    nan = torch.full((1,), NAN, device=logits.device)
    logprob = logp[:-1].gather(1, token_ids[1:, None].to(logits.device)).squeeze(1)
    tokens = {
        "logprob": torch.cat([nan, logprob]),
        "entropy_prev": torch.cat([nan, entropy[:-1]]),
        "entropy_next": entropy,
    }
    text = {
        "ppl_all": float((-logprob).mean().exp()) if len(logprob) else NAN,
        "mean_entropy_next": float(entropy[MIN_POSITION:].mean()) if len(entropy) > MIN_POSITION else NAN,
    }
    return tokens, text


def text_statistics(h: torch.Tensor, logits: torch.Tensor, token_ids: torch.Tensor, text_index: int) -> dict:
    """All scalars of one text. h: (L, T, D), layers 0..L-1."""
    x = h.double()
    keep = token_mask(x)
    layers = {**energy_split(x, keep), **shuffle_test(x, keep, text_index)}
    across, flag = across_split(x, keep)
    pred, text = predictability(logits, token_ids)
    to_np = lambda d: {name: v.cpu().numpy() for name, v in d.items()}
    return {
        "layers": {**to_np(layers), "n_kept": keep.sum(dim=1).cpu().numpy()},
        "across": to_np(across),
        "masks": {"keep": keep.cpu().numpy(), "flag_large_angle": flag.cpu().numpy()},
        "pred": {**to_np(pred), "token_ids": token_ids.cpu().numpy()},
        "text": text,
    }


def pad(rows: list[np.ndarray], length: int, fill, dtype) -> np.ndarray:
    """Stack (..., T_i) arrays into (texts, ..., max T), padded at the end of the token axis."""
    out = np.full((len(rows), *rows[0].shape[:-1], length), fill, dtype=dtype)
    for i, row in enumerate(rows):
        out[i, ..., :row.shape[-1]] = row
    return out


def run(args: Namespace) -> dict[str, np.ndarray]:
    inference = Inference(model_name=args.model)
    args.return_token_ids = True
    args.return_logits = True
    data = load_data(args=args)[args.split]

    rows, labels, n_layers_total = [], [], None
    for text_index, item in enumerate(tqdm(data, desc="Collecting desc3 statistics")):
        out = inference.run(item, args)
        n_layers_total = len(out["hidden_states"]) - 1  # L transformer layers
        h = torch.stack(out["hidden_states"][:-1]).to(inference.device).float()  # (L, T, D), final-norm layer dropped
        rows.append(text_statistics(h, out["logits"].to(inference.device), out["token_ids"], text_index))
        labels.append(int(item["label"]))
        del h, out
    del inference
    torch.cuda.empty_cache()

    max_t = max(len(row["pred"]["token_ids"]) for row in rows)
    n_layers = len(rows[0]["layers"]["n_kept"])
    arrays = {
        "text_id": np.arange(len(rows)),
        "label": np.asarray(labels),
        "layers": np.arange(n_layers),
        "model": np.asarray(args.model),
        "dataset": np.asarray(args.dataset),
        "n_layers_total": np.asarray(n_layers_total),
        "n_kept": np.stack([row["layers"]["n_kept"] for row in rows]).astype(np.int32),
        "token_ids": pad([row["pred"]["token_ids"] for row in rows], max_t, -1, np.int64),
    }
    arrays.update({name: np.asarray([row["text"][name] for row in rows], dtype=np.float32) for name in TEXT_SCORES})
    arrays.update({name: np.stack([row["layers"][name] for row in rows]).astype(np.float32) for name in LAYER_SCORES})
    arrays.update({name: pad([row["across"][name] for row in rows], max_t, np.nan, np.float32)
                   for name in ACROSS_SCORES})
    arrays.update({name: pad([row["masks"][name] for row in rows], max_t, False, bool) for name in TOKEN_MASKS})
    arrays.update({name: pad([row["pred"][name] for row in rows], max_t, np.nan, np.float32) for name in PRED_SCORES})

    meta = {
        **vars(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "desc3",
        "median_factor": MEDIAN_FACTOR,
        "min_position": MIN_POSITION,
        "n_shuffles": N_SHUFFLES,
        "shuffle_seed": f"np.random.default_rng([{SHUFFLE_SEED}, text_index, layer])",
        "eps": EPS,
        "units": "angles in degrees; natural logs, log(x + eps)",
        "layer_numbering": "layers[i] is hidden_states[i]: 0 = embedding output, 1..L-1 = transformer-layer outputs; "
                           "hidden_states[L] (final norm applied) is not used; n_layers_total = L",
        "arrays": {
            "text_id, label": "(texts,); text_id = index in the split, label 0 = human, 1 = machine",
            "layers": "(layers,) layer index of each position on the layer axis",
            "model, dataset, n_layers_total": "scalars",
            "n_kept, " + ", ".join(LAYER_SCORES): "(texts, layers); NaN where undefined (s_corr, avg_cos and the "
                "shuffle scores need >= 2 kept tokens)",
            ", ".join(ACROSS_SCORES): "(texts, layers, max T) float32; NaN at layer 0, off across-layer pairs, where "
                "flag_large_angle and after the last token",
            "keep": "(texts, layers, max T) bool, ||h_t|| <= 2 * median_t ||h_t|| and t >= 10",
            "flag_large_angle": "(texts, layers, max T) bool, across-layer pair with 1 + alpha <= 0",
            "token_ids": "(texts, max T) int64, -1 after the last token",
            ", ".join(PRED_SCORES): "(texts, max T) float32, aligned to the token at t; logprob and entropy_prev NaN "
                "at t = 0",
            "ppl_all, mean_entropy_next": "(texts,); ppl_all over t >= 1, mean_entropy_next over t >= 10",
        },
    }
    arrays["meta"] = np.asarray(json.dumps(meta))
    return arrays


def quick_checks(arrays: dict[str, np.ndarray]) -> None:
    labels = arrays["label"]
    split = arrays["energy"] - arrays["shared_energy"] - arrays["token_energy"]
    print(f"energy - shared - token:          max rel |.| = {np.nanmax(np.abs(split) / arrays['energy']):.2e}")
    t_sum = arrays["log_tan"] - sum(arrays[f"T{i}"] for i in range(1, 5))
    print(f"log_tan - (T1 + T2 + T3 + T4):    max |.| = {np.nanmax(np.abs(t_sum)):.2e}")
    pairs = (arrays["keep"][:, 1:] & arrays["keep"][:, :-1]).sum()
    print(f"flag_large_angle: {arrays['flag_large_angle'].sum()} of {pairs} across-layer pairs")
    print(f"kept tokens: {arrays['keep'].mean():.1%} of all (text, layer, position) slots; "
          f"min n_kept {arrays['n_kept'].min()}")
    print(f"s in [{np.nanmin(arrays['s']):.3f}, {np.nanmax(arrays['s']):.3f}]; "
          f"A_shuffled_std median {np.nanmedian(arrays['A_shuffled_std']):.3f} deg")
    for name in TEXT_SCORES:
        print(f"{name:>18}: human {np.nanmean(arrays[name][labels == 0]):.3f}, "
              f"machine {np.nanmean(arrays[name][labels == 1]):.3f}")
    for name in ("s", "avg_cos", "A_original", "A_shuffled", "L"):
        human = np.nanmean(arrays[name][labels == 0], axis=0)
        machine = np.nanmean(arrays[name][labels == 1], axis=0)
        print(f"{name:>18} (human - machine, per layer): {np.round(human - machine, 3).tolist()}")


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    arrays = run(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"desc3_{args.model}_{args.dataset}_s{args.seed}.npz")
    np.savez_compressed(path, **arrays)
    print(f"Wrote {path}")
    quick_checks(arrays)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


if __name__ == "__main__":
    main()
