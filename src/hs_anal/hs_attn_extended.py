"""
Attention behaviour of human vs. machine texts, per layer and attention head: (1) summaries of the attention maps and
(2) the size of each head's contribution to the residual stream. Runs --model on every text of --dataset (split
--split) with eager attention (needed for the attention weights) and keeps only per-text summaries. Needs a GPU.

Input: tokenised as src.inference.Inference does (no special tokens, truncation at 1024 tokens), so by default there is
no BOS token and position 0 is the first word of the text; with --add_bos 1 the tokenizer's BOS token is put in front
(position 0 is then the BOS token). Rows are destination tokens i, columns source tokens j; A^{ℓ,h} is the causal
attention matrix of layer ℓ (1..L) and head h (1..H), each row sums to 1 over j ≤ i.

1. Attention maps, per row i (all rows i = 0..T−1 are averaged, as in the definition):
       self      = A_ii
       prev_5    = Σ_{j = i−5}^{i−1} A_ij      the 5 previous tokens, the token itself excluded (fewer for i < 5)
       prev_15   = Σ_{j = i−15}^{i−1} A_ij     the same with 15
       bos       = A_i0                         the weight on position 0: the BOS token with --add_bos 1, otherwise
                                                the first token of the text, which takes the attention-sink role
       entropy   = −Σ_{j ≤ i} A_ij log A_ij     natural log; over all i + 1 visible tokens (row 0 has entropy 0)
2. Head contributions. With x_j = input_layernorm(h_j^{ℓ−1}) (what v_proj reads) and the head's value head
   g(h) = h // (H / H_kv) (as HF's repeat_kv), the head's OV map W_OV^h = W_O^h W_V^{g(h)} and
       c_ij^h = A_ij^h W_OV^h x_j                 what source j writes into destination i through head h
       Δ_i^h  = Σ_{j ≤ i} c_ij^h = W_O^h z_i^h,   z_i^h = Σ_j A_ij^h W_V^{g(h)} x_j   (the head's real output at i)
       contrib = C^h = (1/T) Σ_i ‖Δ_i^h‖₂
   Σ_h Δ_i^h is the attention block's output (o_proj has no bias in Llama/Qwen/Mistral). ‖W_O^h z‖² = zᵀ (W_O^hᵀ W_O^h) z
   is computed in the head dimension, so no D-dimensional vector per (i, j) pair is ever formed.

Output: output/hidden_states/attn_extended/attn_ext_{model}_{dataset}_s{seed}[_bos].npz with labels (N,)
(0 = human, 1 = machine), n_tokens (N,), and self, prev_5, prev_15, bos, entropy, contrib, each (N, L, H): index
[i, h] = transformer layer i + 1, attention head h.

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.hs_anal.hs_attn_extended --model l8b \
        --dataset drlXDomainLen_wiki_300_tokens
"""

import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.utils import load_data

cfg = Config()
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "hidden_states", "attn_extended")
PREV_KS = (5, 15)  # windows of previous tokens for the attention mass
MAX_LENGTH = 1024  # truncation, as src.inference.Inference
SCORES = ("self", *(f"prev_{k}" for k in PREV_KS), "bos", "entropy", "contrib")


def head_sizes(config) -> tuple[int, int, int]:
    """(H, H_kv, head_dim): attention heads, value heads, size of each head."""
    n_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_heads
    return n_heads, config.num_key_value_heads, head_dim


def map_scores(attn: torch.Tensor) -> dict[str, torch.Tensor]:
    """attn: (H, T, T) causal attention of one layer -> per head (H,) the row means of self, prev_k, bos, entropy."""
    n_tokens = attn.shape[-1]
    rows = torch.arange(n_tokens, device=attn.device)
    offset = rows[:, None] - rows[None, :]  # i − j
    out = {"self": attn.diagonal(dim1=-2, dim2=-1).mean(dim=-1), "bos": attn[..., 0].mean(dim=-1)}
    for k in PREV_KS:
        window = ((offset >= 1) & (offset <= k)).to(attn.dtype)  # j = i − k .. i − 1
        out[f"prev_{k}"] = (attn * window).sum(dim=-1).mean(dim=-1)
    plogp = torch.where(attn > 0, attn * torch.log(attn.clamp_min(1e-30)), torch.zeros_like(attn))
    out["entropy"] = (-plogp.sum(dim=-1)).mean(dim=-1)
    return out


def contribution(attn: torch.Tensor, x: torch.Tensor, w_v: torch.Tensor, gram_o: torch.Tensor,
                 n_kv: int) -> torch.Tensor:
    """
    attn (H, T, T), x (T, D) the v_proj inputs, w_v (H_kv · d, D) the v_proj weight, gram_o (H, d, d) = W_O^hᵀ W_O^h.
    Returns (H,) C^h = mean_i ‖W_O^h z_i^h‖, z_i^h = Σ_j A_ij^h W_V^{g(h)} x_j.
    """
    n_heads, head_dim = gram_o.shape[0], gram_o.shape[-1]
    values = (x @ w_v.T).view(x.shape[0], n_kv, head_dim).transpose(0, 1)  # (H_kv, T, d)
    values = values.repeat_interleave(n_heads // n_kv, dim=0)  # (H, T, d), head h reads value head h // group
    z = attn @ values  # (H, T, d)
    norm2 = torch.einsum("htd,hde,hte->ht", z, gram_o, z)  # ‖W_O^h z_i‖²
    return norm2.clamp_min(0).sqrt().mean(dim=-1)


class AttentionAnalysis:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model, args=Namespace(return_attentions=True))  # eager attention
        self.model, self.tokenizer = self.inference.model, self.inference.tokenizer
        self.device = self.inference.device
        self.n_heads, self.n_kv, self.head_dim = head_sizes(self.model.config)
        with torch.no_grad():
            self.w_v, self.gram_o = [], []
            for layer in self.model.model.layers:
                w_o = layer.self_attn.o_proj.weight.detach().float()  # (D, H · d)
                blocks = w_o.view(w_o.shape[0], self.n_heads, self.head_dim).permute(1, 0, 2)  # (H, D, d)
                self.gram_o.append(blocks.transpose(1, 2) @ blocks)  # (H, d, d)
                self.w_v.append(layer.self_attn.v_proj.weight.detach().float())
        if args.add_bos and self.tokenizer.bos_token_id is None:
            raise ValueError(f"{args.model}'s tokenizer has no BOS token; run without --add_bos.")

    @torch.no_grad()
    def text_scores(self, text: str) -> tuple[dict[str, np.ndarray], int]:
        """{score: (L, H)} for one text, and its number of tokens."""
        ids = self.tokenizer(text, truncation=True, add_special_tokens=False, max_length=MAX_LENGTH,
                             return_tensors="pt")["input_ids"]
        if self.args.add_bos:
            ids = torch.cat([torch.tensor([[self.tokenizer.bos_token_id]]), ids[:, :MAX_LENGTH - 1]], dim=1)
        out = self.model(input_ids=ids.to(self.device), output_hidden_states=True, output_attentions=True,
                         use_cache=False)
        dtype = next(self.model.parameters()).dtype
        scores = {key: [] for key in SCORES}
        for index, layer in enumerate(self.model.model.layers):
            attn = out.attentions[index][0].float()  # (H, T, T)
            x = layer.input_layernorm(out.hidden_states[index][0].to(dtype)).float()  # (T, D), what v_proj reads
            for key, value in map_scores(attn).items():
                scores[key].append(value)
            scores["contrib"].append(contribution(attn, x, self.w_v[index], self.gram_o[index], self.n_kv))
        return {key: torch.stack(value).cpu().numpy() for key, value in scores.items()}, int(ids.shape[1])

    def run(self) -> dict[str, np.ndarray]:
        data = load_data(args=self.args)[self.args.split]
        rows, labels, n_tokens = {key: [] for key in SCORES}, [], []
        for item in tqdm(data, desc="Running texts"):
            scores, n = self.text_scores(item["text"])
            for key, value in scores.items():
                rows[key].append(value)
            labels.append(int(item["label"]))
            n_tokens.append(n)
        return {"labels": np.asarray(labels), "n_tokens": np.asarray(n_tokens),
                **{key: np.stack(value).astype(np.float32) for key, value in rows.items()}}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--add_bos", type=int, choices=(0, 1), default=0,
                        help="1: put the tokenizer's BOS token in front (default 0: no special tokens, as elsewhere)")
    args = parser.parse_args()
    args.model_name = args.model

    out = AttentionAnalysis(args).run()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = "_bos" if args.add_bos else ""
    path = os.path.join(OUTPUT_DIR, f"attn_ext_{args.model}_{args.dataset}_s{args.seed}{suffix}.npz")
    np.savez_compressed(path, **out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
