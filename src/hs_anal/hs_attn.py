"""
Do tokens point along the main directions of the attention weights? Per layer and head, the angle between each token
and the main direction of (1) the value weights W_V and (2) the OV circuit W_O W_V, human vs. machine. Runs --model on
every text of --dataset (split --split) as src.inference.Inference does (no special tokens) and keeps only per-text
summaries. Needs a GPU.

Transformer layer ℓ (ℓ = 1..L) reads x = input_layernorm(h^{ℓ−1}) (RMSNorm with its weight γ, what v_proj actually
reads) and adds the attention output to the raw residual r = h^{ℓ−1}. Singular vectors have an arbitrary sign, so every
angle uses |cos| and lies in [0, π/2] radians (small: the token points along the main direction).

1. W_V, per value head g (num_key_value_heads; with grouped-query attention several query heads share one value head):
       W_V^g = U Σ Vᵀ (SVD),   v_1^g = the first right singular vector (largest σ): the input direction W_V^g amplifies most
       wv_angle_t^{ℓ,g} = arccos |cos(x_t, v_1^g)|
2. OV circuit, per attention head h (num_attention_heads), reading the value head g(h) = h // (H / H_kv) (as HF's
   repeat_kv) and writing through its own head_dim columns of W_O:
       OV_h = W_O^h W_V^{g(h)} ∈ ℝ^{D × D},   rank ≤ head_dim,   OV_h = Σ_i σ_i u_i v_iᵀ
   v_1 = the main input direction (what the head reads most), u_1 = the main output direction (where it writes most).
       ov_in_angle_t^{ℓ,h}  = arccos |cos(x_t, v_1)|    the normed input, as the head reads it
       ov_out_angle_t^{ℓ,h} = arccos |cos(r_t, u_1)|    the raw residual the head writes into (small: the head's main
                                                         write runs along the token, i.e. stretches rather than turns it)
   The SVD is exact through the thin factors: W_O^h = Q_o R_o, (W_V^g)ᵀ = Q_v R_v (QR), so OV_h = Q_o (R_o R_vᵀ) Q_vᵀ
   and the SVD of the head_dim × head_dim matrix R_o R_vᵀ = U' Σ V'ᵀ gives u_1 = Q_o U'[:, 0], v_1 = Q_v V'[:, 0].

Tokens: per text and layer, token t is kept if ‖h_t^{ℓ−1}‖ ≤ OUTLIER_FACTOR × median_t ‖h_t^{ℓ−1}‖ (the raw residual
state entering the layer; drops outliers such as the attention sink at t = 0). Per text, the angle is the mean over the
kept tokens.

x is recomputed from the hidden states as layers[ℓ − 1].input_layernorm(hidden_states[ℓ − 1]) in the model's dtype,
which is exactly what v_proj receives in the forward pass (hidden_states[ℓ − 1] is the input of transformer layer ℓ).

Output: output/hidden_states/attn/attn_{model}_{dataset}_s{seed}.npz with labels (N,) (0 = human, 1 = machine),
wv_angle (N, L, H_kv) and ov_in_angle, ov_out_angle (N, L, H): index [i, g or h] = transformer layer i + 1, value head
g or attention head h; NaN where no token is kept.

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.hs_anal.hs_attn --model l8b \
        --dataset drlXDomainLen_wiki_300_tokens
"""

import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.utils import load_data, return_device

cfg = Config()
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "hidden_states", "attn")
OUTLIER_FACTOR = 2  # tokens with norm > OUTLIER_FACTOR × median (per text and layer) are dropped
EPS = 1e-12


def head_sizes(config) -> tuple[int, int, int]:
    """(H, H_kv, head_dim): attention heads, value heads, size of each head."""
    n_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_heads
    return n_heads, config.num_key_value_heads, head_dim


def main_directions(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Per transformer layer, float64: wv_in (L, H_kv, D) the first right singular vector of each W_V^g; ov_in, ov_out
    (L, H, D) the first right / left singular vector of each OV_h = W_O^h W_V^{g(h)}.
    """
    n_heads, n_kv, head_dim = head_sizes(model.config)
    group = n_heads // n_kv  # query heads per value head
    out = {"wv_in": [], "ov_in": [], "ov_out": []}
    for layer in model.model.layers:
        w_v = layer.self_attn.v_proj.weight.detach().double()  # (H_kv · head_dim, D)
        w_o = layer.self_attn.o_proj.weight.detach().double()  # (D, H · head_dim)
        d = w_v.shape[1]
        v_blocks = w_v.reshape(n_kv, head_dim, d)  # value head g = rows g · head_dim .. (g + 1) · head_dim
        _, _, vh = torch.linalg.svd(v_blocks, full_matrices=False)  # rows of vh: right singular vectors, σ descending
        out["wv_in"].append(vh[:, 0])
        o_blocks = w_o.reshape(d, n_heads, head_dim).permute(1, 0, 2)  # (H, D, head_dim): W_O^h = columns of head h
        q_o, r_o = torch.linalg.qr(o_blocks)  # W_O^h = Q_o R_o
        q_v, r_v = torch.linalg.qr(v_blocks.transpose(1, 2))  # (W_V^g)ᵀ = Q_v R_v, (H_kv, D, head_dim)
        g = torch.arange(n_heads, device=w_v.device) // group  # value head read by each attention head
        u_s, _, vh_s = torch.linalg.svd(r_o @ r_v[g].transpose(1, 2))  # small (H, head_dim, head_dim)
        out["ov_out"].append((q_o @ u_s[:, :, :1]).squeeze(-1))  # u_1 = Q_o U'[:, 0]
        out["ov_in"].append((q_v[g] @ vh_s[:, :1, :].transpose(1, 2)).squeeze(-1))  # v_1 = Q_v V'[:, 0]
    return {key: torch.stack(value) for key, value in out.items()}


def mean_angle(vectors: torch.Tensor, directions: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """vectors (L, T, D), unit directions (L, K, D), keep (L, T) -> (L, K) mean over kept tokens of arccos |cos|."""
    unit = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(EPS)
    angle = torch.acos(torch.einsum("ltd,lkd->ltk", unit, directions).abs().clamp(max=1.0))  # (L, T, K)
    w = keep.to(angle.dtype)[..., None]
    k = w.sum(dim=1)  # (L, 1)
    return torch.where(k > 0, (angle * w).sum(dim=1) / k.clamp_min(1), torch.nan)


def text_scores(h: torch.Tensor, x: torch.Tensor, directions: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """
    h: (L, T, D) raw residual states entering layers 1..L (outlier mask, OV output side), x: (L, T, D) the v_proj
    inputs, directions: from main_directions. Returns wv_angle (L, H_kv), ov_in_angle and ov_out_angle (L, H).
    """
    n = h.norm(dim=-1)  # (L, T)
    keep = n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values
    scores = {
        "wv_angle": mean_angle(x, directions["wv_in"], keep),
        "ov_in_angle": mean_angle(x, directions["ov_in"], keep),
        "ov_out_angle": mean_angle(h, directions["ov_out"], keep),
    }
    return {key: value.cpu().numpy() for key, value in scores.items()}


def collect(args: Namespace) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-text scores, each (N, L, H_kv or H), and labels (N,)."""
    device = return_device()
    inference = Inference(model_name=args.model)
    model = inference.model
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        directions = {key: value.to(device) for key, value in main_directions(model).items()}
    print("main directions: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in directions.items()))
    data = load_data(args=args)[args.split]
    out, labels = {}, []
    for item in tqdm(data, desc="Running texts"):
        hidden_states = inference.run(item, args)["hidden_states"]
        with torch.no_grad():
            h = torch.stack(hidden_states[:-1]).to(device)  # (L, T, D): input of transformer layer ℓ at index ℓ − 1
            x = torch.stack([layer.input_layernorm(h[i].to(dtype)) for i, layer in enumerate(model.model.layers)])
            scores = text_scores(h.double(), x.double(), directions)
        for key, value in scores.items():
            out.setdefault(key, []).append(value)
        labels.append(int(item["label"]))
    del inference
    torch.cuda.empty_cache()
    return {key: np.stack(value) for key, value in out.items()}, np.asarray(labels)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    args.model_name = args.model

    s, labels = collect(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"attn_{args.model}_{args.dataset}_s{args.seed}.npz")
    np.savez_compressed(path, labels=labels, **s)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
