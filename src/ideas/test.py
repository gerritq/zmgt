"""
Joint angle scores per text and layer, human vs machine. (The previous difference-of-means probe that lived in this
file is in git, commit 1f43c88.)

Runs --model on every text of --dataset (split --split) as src.inference.Inference does (no special tokens). For each
text, at each transformer layer ℓ = 1..L (layer 0 = embeddings), tokens whose norm is more than OUTLIER_FACTOR × the
median token norm of that text and layer are removed (e.g. the attention sink at position 0). A token t enters a layer-ℓ
score only if every state the score uses is kept (see the masks below).

Per token t at layer ℓ (h_t^ℓ = hidden state, ĥ = h / ‖h‖, μ^ℓ = mean of the kept h_t^ℓ = the layer's centroid):
    θ_t = ∠(h_{t−1}^ℓ, h_t^ℓ)      within layer, angle to the predecessor token        (src.zero intra angle)
    φ_t = ∠(h_t^{ℓ−1}, h_t^ℓ)      across layers, angle of the token to itself one layer down  (src.zero inter angle)
    ψ_t = ∠(μ^ℓ, h_t^ℓ)            within layer, angle to the centroid (the dominant direction of the layer)
    ψ̄_t = ∠(μ^ℓ, h̄_t^ℓ)           h̄_t^ℓ = mean(h_{t−W+1}^ℓ, ..., h_t^ℓ), the last W = CONTEXT_W tokens incl. t
    θ̄_t = mean_{j=1..K} ∠(h_{t−j}^ℓ, h_t^ℓ)   mean angle to the K = CONTEXT_K previous tokens
Scores (mean over the tokens t that pass the mask):
    joint             = mean_t √(θ_t φ_t)     as src.zero; mask: t, t−1 kept at ℓ and t kept at ℓ − 1
    joint_ani         = mean_t √(ψ_t φ_t)     mask: t kept at ℓ and at ℓ − 1
    joint_context_ani = mean_t √(ψ̄_t φ_t)    mask: t−W+1..t kept at ℓ and t kept at ℓ − 1
    joint_context     = mean_t √(θ̄_t φ_t)    mask: t−K..t kept at ℓ and t kept at ℓ − 1
    joint_semantic    = joint with θ and φ computed on p_t^ℓ = V_Sᵀ Norm(h_t^ℓ), semantic subspace of W_U
    joint_reasoning   = the same in the reasoning subspace of W_U
W_U subspaces as src.hs_anal.hs_anal.harp_bases: right singular vectors of the unembedding W_U, semantic = the top
k = 0.95 d by singular value, reasoning = the remaining 0.05 d (smallest). Same basis at every layer; states go
through the final norm W_U reads before projecting (the last layer is already normed by HF). The subspace scores use
the same mask as joint (outliers found on the full-space norms), so they average over the same tokens.
AUC (machine = 1): AUC of −score (> 0.5: machine has the smaller angles = more aligned).

Output: output/test/joint_scores_{model}_{dataset}_s{seed}.pdf. Page 1: table of the AUC of every score (rows) at every
layer (columns). Page 2: one row per score, per layer the mean per group ±95% CI (left y-axis) and the AUC (right
y-axis).

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.ideas.test --model l8b \
        --dataset drlXDomainLen_wiki_300_tokens
"""

import os
from argparse import ArgumentParser, Namespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from src.config import Config
from src.hs_anal.hs_anal import HARP_SUBSPACES, auc_table, compare, figure, harp_bases
from src.inference import Inference
from src.utils import load_data, return_device

cfg = Config()
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "test")
OUTLIER_FACTOR = 2
EPS = 1e-12
CONTEXT_W = 3  # joint_context_ani: tokens averaged (incl. t)
CONTEXT_K = 3  # joint_context: previous tokens compared with t
SCORES = ("joint", "joint_ani", "joint_context_ani", "joint_context") + tuple(f"joint_{s}" for s in HARP_SUBSPACES)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the last dim of x where mask is True; NaN if nothing passes."""
    count = mask.sum(dim=-1)
    total = torch.where(mask, x, torch.zeros_like(x)).sum(dim=-1)
    return torch.where(count > 0, total / count.clamp_min(1), torch.full_like(total, float("nan")))


def text_scores(h: torch.Tensor, bases: dict[str, torch.Tensor], final_norm: torch.nn.Module) -> dict[str, np.ndarray]:
    """
    h: (L + 1, T, D) hidden states of one text, index 0 = embeddings; bases: {subspace: (D, n_dims)} from harp_bases;
    final_norm: the model's final norm. Every score is (L,) at layers 1..L.
    """
    n = h.norm(dim=-1).clamp_min(EPS)  # (L + 1, T)
    keep = n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values  # (L + 1, T)
    u = h / n[..., None]  # unit states
    mu = (h * keep[..., None]).sum(dim=1) / keep.sum(dim=1, keepdim=True).clamp_min(1)  # (L + 1, D), kept tokens
    mu_u = mu / mu.norm(dim=-1, keepdim=True).clamp_min(EPS)  # (L + 1, D) unit centroid
    T = h.shape[1]
    W, K = CONTEXT_W, CONTEXT_K

    def angle(c: torch.Tensor) -> torch.Tensor:
        return torch.acos(c.clamp(-1.0, 1.0))

    def unit(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)

    # angles at layers 1..L
    theta = angle((u[1:, :-1] * u[1:, 1:]).sum(dim=-1))  # (L, T − 1): ∠(h_{t−1}, h_t), t = 1..T−1
    phi = angle((u[:-1] * u[1:]).sum(dim=-1))  # (L, T): ∠(h_t^{ℓ−1}, h_t^ℓ)
    psi = angle((u[1:] * mu_u[1:, None]).sum(dim=-1))  # (L, T): ∠(μ^ℓ, h_t^ℓ)
    # ψ̄: centroid vs mean of h_{t−W+1..t}, t = W−1..T−1
    h_ctx = torch.stack([h[1:, W - 1 - j:T - j] for j in range(W)]).mean(dim=0)  # (L, T − W + 1, D)
    psi_ctx = angle((unit(h_ctx) * mu_u[1:, None]).sum(dim=-1))  # (L, T − W + 1)
    # θ̄: mean over j = 1..K of ∠(h_{t−j}, h_t), t = K..T−1
    theta_ctx = torch.stack([angle((u[1:, K - j:T - j] * u[1:, K:]).sum(dim=-1))
                             for j in range(1, K + 1)]).mean(dim=0)  # (L, T − K)

    # masks over tokens: every state a score uses must be kept
    mask_t = keep[1:] & keep[:-1]  # (L, T): h_t^ℓ, h_t^{ℓ−1}
    mask_prev = mask_t[:, 1:] & keep[1:, :-1]  # (L, T − 1), t = 1..T−1
    mask_w = mask_t[:, W - 1:].clone()  # (L, T − W + 1), t = W−1..T−1
    for j in range(1, W):
        mask_w &= keep[1:, W - 1 - j:T - j]
    mask_k = mask_t[:, K:].clone()  # (L, T − K), t = K..T−1
    for j in range(1, K + 1):
        mask_k &= keep[1:, K - j:T - j]

    scores = {
        "joint": masked_mean((theta * phi[:, 1:]).sqrt(), mask_prev),
        "joint_ani": masked_mean((psi * phi).sqrt(), mask_t),
        "joint_context_ani": masked_mean((psi_ctx * phi[:, W - 1:]).sqrt(), mask_w),
        "joint_context": masked_mean((theta_ctx * phi[:, K:]).sqrt(), mask_k),
    }

    # joint in the W_U subspaces: p = Norm(h) @ basis, same mask as joint
    dtype = next(final_norm.parameters()).dtype
    z = torch.cat([final_norm(h[:-1].to(dtype)).float(), h[-1:]])  # (L + 1, T, D); last layer already normed
    for sub, basis in bases.items():
        p = unit(z @ basis)  # (L + 1, T, n_dims)
        theta_s = angle((p[1:, :-1] * p[1:, 1:]).sum(dim=-1))  # (L, T − 1)
        phi_s = angle((p[:-1] * p[1:]).sum(dim=-1))  # (L, T)
        scores[f"joint_{sub}"] = masked_mean((theta_s * phi_s[:, 1:]).sqrt(), mask_prev)
    return {key: value.double().cpu().numpy() for key, value in scores.items()}


def collect(args: Namespace) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-text scores, each (N, L), and labels (N,)."""
    device = return_device()
    inference = Inference(model_name=args.model)
    model = inference.model
    with torch.no_grad():
        bases = {k: v.to(device) for k, v in harp_bases(model.get_output_embeddings().weight.to(device)).items()}
    print("W_U subspace dims: " + ", ".join(f"{k} {v.shape[1]}" for k, v in bases.items()))
    data = load_data(args=args)[args.split]
    out = {key: [] for key in SCORES}
    labels = []
    for item in tqdm(data, desc="Running texts"):
        hidden_states = inference.run(item, args)["hidden_states"]
        with torch.no_grad():
            scores = text_scores(torch.stack(hidden_states).to(device).float(), bases, model.model.norm)
        for key in SCORES:
            out[key].append(scores[key])
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
    bad = {key: int((~np.isfinite(v)).sum()) for key, v in s.items() if not np.isfinite(v).all()}
    if bad:
        raise ValueError(f"scores with NaN (no token passed the mask): {bad}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = [
        ("joint", "joint: √(θ φ), predecessor & layer below"),
        ("joint_ani", "joint_ani: √(ψ φ), centroid & layer below"),
        ("joint_context_ani", f"joint_context_ani: √(ψ̄ φ), centroid vs mean of last {CONTEXT_W} tokens & layer below"),
        ("joint_context", f"joint_context: √(θ̄ φ), mean angle to {CONTEXT_K} previous tokens & layer below"),
        ("joint_semantic", "joint_semantic: √(θ φ) in the semantic subspace of W_U (top 95%)"),
        ("joint_reasoning", "joint_reasoning: √(θ φ) in the reasoning subspace of W_U (bottom 5%)"),
    ]
    panels = [[{"stats": compare(s[key], labels, reverse=True), "title": title, "quantity": "angle (rad)",
                "auc_label": "AUC (−score, machine = 1)"}] for key, title in rows]
    path = os.path.join(OUTPUT_DIR, f"joint_scores_{args.model}_{args.dataset}_s{args.seed}.pdf")
    title = f"Joint scores, tokens > {OUTLIER_FACTOR}× median removed, {args.model}, {args.dataset}"
    with PdfPages(path) as pdf:  # page 1: AUC table, page 2: plots
        for fig in (auc_table({key: p[0]["stats"]["auc"] for (key, _), p in zip(rows, panels)}, title, first_layer=1),
                    figure(panels, title, first_layer=1, xlabel="transformer layer", groups=True)):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
