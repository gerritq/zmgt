"""
Paired texts (one human, one machine per pair_id) whose joint score differs most.

Data are loaded as in src.zero (load_data(args)["test"]). Only the joint score of src.zero is computed, the same way
except that no leading tokens are dropped (SKIP = 0 here; src.zero drops 10), so every token from t = 1 is scored:
    joint[ℓ] = mean_t √(θ φ),  θ = ∠(h_{t−1}^ℓ, h_t^ℓ) (intra),  φ = ∠(h_t^{ℓ−1}, h_t^ℓ) (inter), in radians,
over the kept tokens with both angles; ℓ = transformer layers 1..L, stored as layer_i = transformer layer i + 1 (as
src.zero). Unsigned: machine text has smaller angles, so a lower joint.
Per pair, at --layer: divergence = joint(human) − joint(machine) (> 0: the pair is ordered as expected). Only the --top_k
pairs with the largest |divergence| are saved, to output/inter/inter_sample_{model}_{dataset}_s{seed}.json: per text the
text, the joint at every layer and, at --layer, every token with its θ, φ and √(θ φ) (null at t = 0, which has no
previous token). The mean of the per-token √(θ φ) is the text's joint exactly (checked before saving).
"""

import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import load_data, return_device
from src.zero import angle, MAX_SKIP_SHARE, CONTEXT_K, FORMULAS

from src.config import Config
cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "inter")
SKIP = 0  # leading tokens dropped; src.zero uses 10, here every token is kept


def joint_scores(hidden_states: tuple[torch.Tensor, ...], device: torch.device
                 ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    As src.zero.AngleNormScores.scores: joint at every transformer layer (L,) and the per-token θ, φ and √(θ φ), each
    (L, T) aligned to the token position t, NaN for t <= SKIP; everything NaN if the item is skipped.
    """
    n_layers, n_tokens = len(hidden_states) - 1, hidden_states[0].shape[0]
    tokens = {name: np.full((n_layers, n_tokens), np.nan) for name in ("theta", "phi", "joint")}
    kept = n_tokens - SKIP
    if kept < (1 - MAX_SKIP_SHARE) * n_tokens or kept < CONTEXT_K + 2:
        return np.full(n_layers, np.nan), tokens
    h_all = torch.stack(hidden_states)[:, SKIP:].to(device).float()  # (L + 1, T', D), with the embeddings
    theta = angle(h_all[1:, :-1], h_all[1:, 1:])  # (L, T' − 1): ∠(h_{t−1}, h_t) for t = SKIP + 1 .. T − 1
    phi = angle(h_all[:-1], h_all[1:])[:, 1:]  # (L, T' − 1): ∠(h_t^{ℓ−1}, h_t^ℓ), same tokens
    joint = (theta * phi).sqrt()
    for name, value in (("theta", theta), ("phi", phi), ("joint", joint)):
        tokens[name][:, SKIP + 1:] = value.cpu().double().numpy()
    return joint.mean(dim=1).cpu().numpy(), tokens


def run(args: Namespace) -> dict:
    device = return_device()
    inference = Inference(model_name=args.model)
    args.return_token_ids = True
    test_data = load_data(args=args)["test"]

    pairs: dict[str, dict] = {}
    for item in tqdm(test_data, desc="Collecting joint scores"):
        out = inference.run(item, args)
        side = "machine" if int(item["label"]) == 1 else "human"
        pair = pairs.setdefault(item["pair_id"], {"pair_id": item["pair_id"]})
        if side in pair:
            raise ValueError(f"pair {item['pair_id']} has two {side} texts")
        joint, tokens = joint_scores(out["hidden_states"], device)
        token_ids = out["token_ids"].tolist()
        pair[side] = {"text": item["text"], "n_tokens": out["n_tokens"], "joint_by_layer": joint,
                      "token_ids": token_ids, "token_strings": [inference.tokenizer.decode([i]) for i in token_ids],
                      **{f"token_{name}": value[args.layer] for name, value in tokens.items()}}
    del inference
    torch.cuda.empty_cache()

    incomplete = [pair_id for pair_id, pair in pairs.items() if not {"human", "machine"} <= pair.keys()]
    if incomplete:
        raise ValueError(f"{len(incomplete)} pairs without both a human and a machine text, e.g. {incomplete[:3]}")

    layer = f"layer_{args.layer}"
    rows = []
    for pair in pairs.values():
        human = float(pair["human"]["joint_by_layer"][args.layer])
        machine = float(pair["machine"]["joint_by_layer"][args.layer])
        rows.append({"pair_id": pair["pair_id"], "joint_human": human, "joint_machine": machine,
                     "divergence": human - machine})
    valid = sorted((r for r in rows if np.isfinite(r["divergence"])), key=lambda r: -abs(r["divergence"]))

    def number(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    def text_entry(text: dict) -> dict:
        per_token = text["token_joint"]
        check = abs(np.nanmean(per_token) - text["joint_by_layer"][args.layer])
        if not check < 1e-6:
            raise ValueError(f"mean of the per-token joint differs from the joint by {check:.2e}")
        return {
            "text": text["text"],
            "n_tokens": text["n_tokens"],
            "joint": number(text["joint_by_layer"][args.layer]),
            "joint_by_layer": {f"layer_{i}": number(v) for i, v in enumerate(text["joint_by_layer"])},
            "tokens": [
                {"position": t, "token_id": token_id, "token": token,
                 "theta": number(text["token_theta"][t]), "phi": number(text["token_phi"][t]),
                 "joint": number(text["token_joint"][t])}
                for t, (token_id, token) in enumerate(zip(text["token_ids"], text["token_strings"]))
            ],
        }

    top = [{"rank": rank, **row, **{side: text_entry(pairs[row["pair_id"]][side]) for side in ("human", "machine")}}
           for rank, row in enumerate(valid[:args.top_k], start=1)]

    divergences = np.asarray([r["divergence"] for r in valid])
    print(f"\ninter_sample | model: {args.model} | dataset: {args.dataset} | seed: {args.seed} | {layer} "
          f"(transformer layer {args.layer + 1})")
    print(f"pairs: {len(pairs)}, scored: {len(valid)}, ordered as expected (human > machine): "
          f"{(divergences > 0).mean():.1%}, mean divergence {divergences.mean():.4f} rad")
    for row in top:
        print(f"  #{row['rank']} {row['pair_id']}: human {row['joint_human']:.4f}, machine {row['joint_machine']:.4f}, "
              f"divergence {row['divergence']:+.4f} rad")

    return {
        **vars(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "inter_sample",
        "score": "joint",
        "formula": FORMULAS["joint"] + " over tokens t >= SKIP + 1; theta intra, phi inter angle; radians",
        "skip_tokens": SKIP,
        "max_skip_share": MAX_SKIP_SHARE,
        "layer_numbering": "layer_i = transformer layer i + 1 (embeddings excluded)",
        "selected_layer": layer,
        "divergence": "joint(human) - joint(machine) at selected_layer (unsigned joint; > 0 = machine has the "
                      "lower joint, as expected); pairs ranked by |divergence|",
        "n_pairs": len(pairs),
        "n_pairs_scored": len(valid),
        "share_ordered_as_expected": float((divergences > 0).mean()),
        "tokens": "per text, at selected_layer: every token (position, id, decoded string) with theta, phi and joint = "
                  "sqrt(theta * phi) in radians; null for positions <= skip_tokens; the mean of the non-null joint "
                  "values is the text's joint",
        "top_pairs": top,
    }


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=7,
                        help="layer_i of src.zero (transformer layer i + 1) at which pairs are ranked")
    parser.add_argument("--top_k", type=int, default=5, help="number of pairs to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model
    output = run(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"inter_sample_{args.model}_{args.dataset}_s{args.seed}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
