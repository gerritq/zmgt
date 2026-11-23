"""
Effective dimension of the perpendicular layer updates, per text and layer, human vs machine. (The previous
logistic-regression "AI direction" script that lived in this file is in git.)

Runs --model on every text of --dataset (split --split) as src.inference.Inference does (no special tokens), with the
hidden states h^ℓ, ℓ = 0..L (0 = embeddings; as in HF, the last one is after the final norm).

1. Update split, as src.hs_anal.hs_anal.update_split. Per token t and transition ℓ − 1 → ℓ, with z = h_t^{ℓ−1} and
   Δ = h_t^ℓ − z:
       α = ⟨Δ, z⟩ / ‖z‖²,   Δ∥ = α z,   Δ⊥ = Δ − Δ∥ (the part of the update that turns z)
2. Outliers: a token is dropped from transition ℓ − 1 → ℓ if its norm is more than OUTLIER_FACTOR × the median token
   norm of that text at layer ℓ − 1 or at layer ℓ (e.g. the attention sink at position 0).
3. Per text and transition, stack the kept Δ⊥ into X (n × D), centre it, and take the eigenvalues λ_i of its
   covariance (= squared singular values of the centred X, up to a constant that cancels). PCA effective rank:
       q_i = λ_i / Σ_j λ_j,   d_eff = exp(−Σ_i q_i log q_i)
   d_eff runs from 1 (all updates along one direction) to n − 1 (spread evenly over every direction the n tokens can
   span), so it depends on n; texts of the same length are needed for a fair comparison.

Output: output/test/perp_eff_rank_{model}_{dataset}_s{seed}.pdf. Page 1: AUC table per layer. Page 2: d_eff per
group (mean ±95% CI) and the AUC of +d_eff (> 0.5: machine updates spread over more directions).

    PYTHONPATH="${BASE_ZMGT}" uv run --project "${BASE_ZERO}" -m src.ideas.test2 --model l8b \
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
from src.hs_anal.hs_anal import auc_table, compare, figure
from src.inference import Inference
from src.utils import load_data, return_device

cfg = Config()
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "test")
OUTLIER_FACTOR = 2
MIN_TOKENS = 3  # kept tokens a transition needs for d_eff (else NaN)
EPS = 1e-12


def perpendicular_eff_rank(h: torch.Tensor) -> np.ndarray:
    """h: (L + 1, T, D) hidden states of one text. PCA effective rank of the kept Δ⊥ per transition: (L,)."""
    n = h.norm(dim=-1)  # (L + 1, T)
    keep = n <= OUTLIER_FACTOR * n.median(dim=1, keepdim=True).values
    mask = keep[:-1] & keep[1:]  # (L, T): token kept at both ends of the transition

    z = h[:-1]
    delta = h[1:] - z
    alpha = (delta * z).sum(dim=-1, keepdim=True) / z.pow(2).sum(dim=-1, keepdim=True).clamp_min(EPS)
    perp = delta - alpha * z  # (L, T, D) Δ⊥

    # centre on the kept tokens; dropped tokens become zero rows, which leave the singular values unchanged
    w = mask.unsqueeze(-1).float()
    count = w.sum(dim=1, keepdim=True)  # (L, 1, 1)
    centred = (perp - (perp * w).sum(dim=1, keepdim=True) / count.clamp_min(1)) * w
    lam = torch.linalg.svdvals(centred.double()).pow(2)  # (L, min(T, D)), ∝ covariance eigenvalues
    q = lam / lam.sum(dim=-1, keepdim=True).clamp_min(EPS)
    entropy = -(q * torch.log(q.clamp_min(EPS))).sum(dim=-1)  # q = 0 terms contribute 0
    d_eff = torch.exp(entropy)
    d_eff[count[:, 0, 0] < MIN_TOKENS] = float("nan")
    return d_eff.cpu().numpy()


def collect(args: Namespace) -> tuple[np.ndarray, np.ndarray]:
    """d_eff per text and transition (N, L), and labels (N,)."""
    device = return_device()
    inference = Inference(model_name=args.model)
    data = load_data(args=args)[args.split]
    scores, labels = [], []
    for item in tqdm(data, desc="Running texts"):
        hidden_states = inference.run(item, args)["hidden_states"]
        with torch.no_grad():
            scores.append(perpendicular_eff_rank(torch.stack(hidden_states).to(device).float()))
        labels.append(int(item["label"]))
    del inference
    torch.cuda.empty_cache()
    return np.stack(scores), np.asarray(labels)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    args.model_name = args.model

    d_eff, labels = collect(args)
    ok = np.isfinite(d_eff).all(axis=1)
    if not ok.all():
        print(f"Dropped {int((~ok).sum())} texts with < {MIN_TOKENS} kept tokens at some layer")
    d_eff, labels = d_eff[ok], labels[ok]

    stats = compare(d_eff, labels)
    for layer, (a, hm, mm) in enumerate(zip(stats["auc"], stats["human_mean"], stats["machine_mean"]), start=1):
        print(f"layer {layer:2d}: d_eff human {hm:7.2f}, machine {mm:7.2f}, AUC {a:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"perp_eff_rank_{args.model}_{args.dataset}_s{args.seed}.pdf")
    title = f"Effective rank of Δ⊥ per text, tokens > {OUTLIER_FACTOR}× median removed, {args.model}, {args.dataset}"
    panels = [[{"stats": stats, "title": r"PCA effective rank of $\Delta_\perp$, $\exp(-\sum_i q_i \log q_i)$",
                "quantity": r"$d_{\mathrm{eff}}$", "auc_label": "AUC (d_eff, machine = 1)"}]]
    with PdfPages(path) as pdf:  # page 1: AUC table, page 2: plot
        for fig in (auc_table({"d_eff": stats["auc"]}, title, first_layer=1),
                    figure(panels, title, first_layer=1, xlabel="transformer layer", groups=True)):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
