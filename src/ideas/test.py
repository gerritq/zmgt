"""
Alignment of tokens with the top singular directions of W_V (per layer) and of the OV circuit (per head), human vs
machine, and a local ablation of each head's top output direction.

Runs --model on every text of --dataset (split --split) as src.inference.Inference does (no special tokens). At each
transformer layer ℓ = 1..L, hooks record for every token t
    r_t = h_t^{ℓ−1}, the raw residual-stream state entering layer ℓ (input of input_layernorm)
    x_t = the input of self_attn.v_proj (for Llama: RMSNorm(r_t) with its weight γ)
    z_t = the input of self_attn.o_proj (the H concatenated head outputs, each d_h wide), a_t = its output
and x̂ = x / ‖x‖, r̂ = r / ‖r‖.

A. W_V alignment (whole layer). W_V^ℓ ∈ ℝ^{d_out × d_in} (all KV heads) = U Σ Vᵀ; rows v_1, v_2, ... of Vᵀ are the
   input-side singular directions:
       A_1 = |x̂ᵀ v_1|,      A_k = Σ_{i=1..k} (x̂ᵀ v_i)²   for k in KS (k ≤ rank W_V)
   Random unit vector in ℝ^d: E[A_k] = k / d, E[A_1] ≈ sqrt(2 / (π d)).

B. OV alignment (per head h). With GQA, head h reads the value rows of its KV group g = h // (H / H_kv) and writes
   through its own d_h columns of W_O:
       OV_h = W_O^h W_V[g] ∈ ℝ^{d_model × d_in},   rank ≤ d_h,   OV_h = Σ_i σ_i u_i v_iᵀ
   v_1 = top input direction, u_1 = top output direction. The SVD is exact through the small factors: W_O^h = Q_o R_o,
   W_V[g]ᵀ = Q_v R_v, so OV_h = Q_o (R_o R_vᵀ) Q_vᵀ and the SVD of R_o R_vᵀ = U' Σ V'ᵀ gives u_1 = Q_o U'[:, 1],
   v_1 = Q_v V'[:, 1].
       θ_in  = arccos |x̂ᵀ v_1|   normalised input token vs the head's top input direction
       θ_out = arccos |r̂ᵀ u_1|   raw residual token (entering the layer) vs the head's top output direction
   In degrees, in [0°, 90°] (singular vectors have an arbitrary sign); random in ℝ^d ≈ 90° − (180/π) sqrt(2 / (π d)).

C. Attention turn with and without the u_1 writes (all heads of the layer at once). Per token:
       m      = r + a                               post-attention residual (input of post_attention_layernorm)
       φ      = ∠(r, m)                             the turn caused by attention, in [0°, 180°]
       c_h    = u_1ᵀ W_O^h z^h                      head h's actual write along its own u_1
       m_abl  = m − Σ_h c_h u_1^h                   every head's u_1 component removed
       φ_abl  = ∠(r, m_abl)
   m_abl equals running o_proj with every W_O^h ← (I − u_1 u_1ᵀ) W_O^h on the same inputs, everything else fixed.
   Per layer: gap = mean_human φ − mean_machine φ, gap_abl = the same for φ_abl, each ±95% CI (unpaired).

D. Turn decomposition, for two updates Δ of the residual r (both start from r = h^{ℓ−1}):
       full: Δ = h^ℓ − h^{ℓ−1}   (attention + MLP, the whole layer; h^ℓ = raw decoder-layer output, before any norm)
       attn: Δ = a               (the attention output only, i.e. m − r)
   Per token, r̂ = r / ‖r‖, a_∥ = r̂ᵀΔ (signed parallel part), b = Δ − a_∥ r̂ (perpendicular part), so
   r + Δ = (‖r‖ + a_∥) r̂ + b and tan φ = ‖b‖ / (‖r‖ + a_∥). Taking logs:
       log tan φ = log ‖b‖ − log ‖r‖ − log(1 + a_∥ / ‖r‖)
   (the last two terms are −log(‖r‖ + a_∥) split into token length and parallel growth). The split is exact per token,
   so the per-text means and the human − machine gaps add up exactly. Tokens with ‖r‖ + a_∥ ≤ 0 (φ ≥ 90°, log tan
   undefined) or ‖b‖ = 0 are left out of all terms; their share per text is printed.

Per text: mean over all its tokens (no tokens removed). Per group (human, machine): mean of the per-text means ±95% CI
and Cohen's d = (mean machine − mean human) / pooled SD. For A: d > 0 = machine more aligned; for θ: d < 0 = machine
more aligned (smaller angle).

Output: output/test/vproj_align_{model}_{dataset}_s{seed}.pdf.
    Page 1: one row, A_1, A_10, A_50 per layer (group means ±95% CI, Cohen's d on the right axis).
    Page 2: one row, θ_in and θ_out per layer, mean over heads.
    Page 3: gap and gap_abl per layer (human − machine turn φ, ±95% CI).
    Page 4: one row, full and attn: human − machine gap of log tan φ and of its three terms (±95% CI).
Per-layer group means, Cohen's d and AUCs (machine = 1) and the gaps are printed.

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
from src.hs_anal.hs_anal import compare, figure
from src.inference import Inference
from src.items.plot_style import configure_plot_style
from src.utils import load_data

cfg = Config()
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "test")
KS = (10, 50)  # top-k sizes for A_k (A_1 is always computed)
EPS = 1e-12
HEAD_SCORES = ("theta_in", "theta_out")  # (L, H) per text
TURN_SCORES = ("phi", "phi_abl")  # (L,) per text
GAP_COLORS = {"gap": "#000000", "gap_abl": "#009E73"}
UPDATES = {"full": "full layer update (attention + MLP)", "attn": "attention update only"}
TERMS = ("log_tan", "log_b", "log_r", "log_1pa", "excluded")  # per update, (L,) per text
# (term, sign in log tan φ, label, colour); the signed term gaps add up to the log tan φ gap
TERM_PARTS = (("log_b", 1, r"$+\log\|b\|$ (sideways push)", "#E69F00"),
              ("log_r", -1, r"$-\log\|r\|$ (token length)", "#56B4E9"),
              ("log_1pa", -1, r"$-\log(1 + a_\parallel/\|r\|)$ (parallel part)", "#CC79A7"))
TURN_NAMES = tuple(f"{u}_{t}" for u in UPDATES for t in TERMS)


def ov_top_directions(w_v: torch.Tensor, w_o: torch.Tensor, n_heads: int, n_kv: int,
                      head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    w_v: v_proj weight (n_kv · d_h, d_in), w_o: o_proj weight (d_model, n_heads · d_h). Top output (u_1, (H, d_model))
    and input (v_1, (H, d_in)) singular directions of OV_h = W_O^h W_V[g(h)] for every head h.
    """
    group = n_heads // n_kv
    wv = w_v.view(n_kv, head_dim, -1)[torch.arange(n_heads, device=w_v.device) // group]  # (H, d_h, d_in)
    wo = w_o.view(w_o.shape[0], n_heads, head_dim).permute(1, 0, 2)  # (H, d_model, d_h)
    q_o, r_o = torch.linalg.qr(wo)  # (H, d_model, d_h), (H, d_h, d_h)
    q_v, r_v = torch.linalg.qr(wv.transpose(1, 2))  # (H, d_in, d_h), (H, d_h, d_h)
    u, _, vh = torch.linalg.svd(r_o @ r_v.transpose(1, 2))  # OV_h = q_o (r_o r_vᵀ) q_vᵀ
    u1 = (q_o @ u[:, :, :1]).squeeze(-1)
    v1 = (q_v @ vh[:, :1, :].transpose(1, 2)).squeeze(-1)
    return u1, v1


def angle_deg(cos: torch.Tensor) -> torch.Tensor:
    """Unsigned angle to a singular direction (arbitrary sign), in [0°, 90°]."""
    return torch.rad2deg(torch.arccos(cos.abs().clamp(max=1.0)))


def turn_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """∠(a_t, b_t) per row, in [0°, 180°]."""
    cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(EPS)
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def turn_terms(r: torch.Tensor, delta: torch.Tensor) -> dict[str, float]:
    """
    r, delta: (T, D) float64. Per-text means of log tan φ = log ‖b‖ − log ‖r‖ − log(1 + a_∥/‖r‖) and of its terms over
    the tokens with ‖r‖ + a_∥ > 0 and ‖b‖ > 0, and the share of tokens left out ("excluded").
    """
    r_norm = r.norm(dim=-1).clamp_min(EPS)
    a_par = (r * delta).sum(dim=-1) / r_norm
    b_norm = (delta - (a_par / r_norm)[:, None] * r).norm(dim=-1)
    growth = 1 + a_par / r_norm
    keep = (growth > 0) & (b_norm > 0)
    out = {"excluded": 1 - keep.double().mean().item()}
    if not keep.any():
        return {**out, **{t: float("nan") for t in ("log_tan", "log_b", "log_r", "log_1pa")}}
    log_b, log_r, log_1pa = b_norm[keep].log(), r_norm[keep].log(), growth[keep].log()
    return {**out, "log_tan": (log_b - log_r - log_1pa).mean().item(), "log_b": log_b.mean().item(),
            "log_r": log_r.mean().item(), "log_1pa": log_1pa.mean().item()}


class Alignment:
    """Hooks on every input_layernorm, v_proj and o_proj; after a forward pass, self.scores holds the per-text means."""

    def __init__(self, model: torch.nn.Module) -> None:
        conf = model.config
        n_heads = conf.num_attention_heads
        n_kv = getattr(conf, "num_key_value_heads", None) or n_heads
        self.head_dim = getattr(conf, "head_dim", None) or conf.hidden_size // n_heads
        self.layers = model.model.layers
        self.bases, self.u1, self.v1, self.w_u, self.hooks = [], [], [], [], []
        self._resid: torch.Tensor | None = None
        with torch.no_grad():
            for layer in self.layers:
                w_v = layer.self_attn.v_proj.weight.float()  # (d_out, d_in)
                w_o = layer.self_attn.o_proj.weight.float()  # (d_model, H · d_h)
                _, _, vh = torch.linalg.svd(w_v, full_matrices=False)  # vh: (rank, d_in), rows = v_i
                self.bases.append(vh[:max(KS)].contiguous())  # (k_max, d_in)
                u1, v1 = ov_top_directions(w_v, w_o, n_heads, n_kv, self.head_dim)
                self.u1.append(u1.contiguous())
                self.v1.append(v1.contiguous())
                # w_u[h] = (W_O^h)ᵀ u_1^h ∈ ℝ^{d_h}, so c_h = u_1ᵀ W_O^h z^h = w_u[h]ᵀ z^h
                self.w_u.append(torch.einsum("dhk,hd->hk", w_o.view(w_o.shape[0], n_heads, self.head_dim), u1))
        self.n_heads, self.d_in, self.d_model = n_heads, self.bases[0].shape[1], self.u1[0].shape[1]
        self.ks = tuple(k for k in KS if k <= self.bases[0].shape[0])
        if len(self.ks) < len(KS):
            print(f"W_V rank {self.bases[0].shape[0]}: dropped k in {sorted(set(KS) - set(self.ks))}")
        self.a_names = ("A_1", *(f"A_{k}" for k in self.ks))
        self.reset()
        for i, layer in enumerate(self.layers):
            self.hooks.append(layer.input_layernorm.register_forward_pre_hook(self._residual_hook(i)))
            self.hooks.append(layer.self_attn.v_proj.register_forward_pre_hook(self._input_hook(i)))
            self.hooks.append(layer.self_attn.o_proj.register_forward_hook(self._write_hook(i)))
            self.hooks.append(layer.register_forward_hook(self._layer_hook(i)))

    @staticmethod
    def _flat(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, t.shape[-1]).float()  # (T, D), batch size 1

    @staticmethod
    def _unit(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)

    def _residual_hook(self, i: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            with torch.no_grad():
                self._resid = self._flat(inputs[0])
                r_hat = self._unit(self._resid)
                self.scores["theta_out"][i] = angle_deg(r_hat @ self.u1[i].T).mean(dim=0).cpu().numpy()  # (H,)
        return hook

    def _input_hook(self, i: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            with torch.no_grad():
                x_hat = self._unit(self._flat(inputs[0]))
                p = x_hat @ self.bases[i].T  # (T, k_max), x̂ᵀ v_i of W_V
                self.scores["A_1"][i] = p[:, 0].abs().mean().item()
                cum = p.pow(2).cumsum(dim=-1)
                for k in self.ks:
                    self.scores[f"A_{k}"][i] = cum[:, k - 1].mean().item()
                self.scores["theta_in"][i] = angle_deg(x_hat @ self.v1[i].T).mean(dim=0).cpu().numpy()  # (H,)
        return hook

    def _write_hook(self, i: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            with torch.no_grad():
                z = self._flat(inputs[0]).view(-1, self.n_heads, self.head_dim)  # (T, H, d_h)
                r = self._resid.double()  # (T, D) residual entering the layer
                m = r + self._flat(output).double()  # (T, D) post-attention residual
                c = (z * self.w_u[i]).sum(dim=-1).double()  # (T, H), c_h = u_1ᵀ W_O^h z^h
                m_abl = m - c @ self.u1[i].double()  # (T, D), m − Σ_h c_h u_1^h
                self.scores["phi"][i] = turn_deg(r, m).mean().item()
                self.scores["phi_abl"][i] = turn_deg(r, m_abl).mean().item()
                for t, v in turn_terms(r, m - r).items():
                    self.scores[f"attn_{t}"][i] = v
        return hook

    def _layer_hook(self, i: int):
        def hook(module: torch.nn.Module, inputs: tuple, output) -> None:
            with torch.no_grad():
                h_out = output[0] if isinstance(output, tuple) else output  # raw decoder-layer output h^ℓ
                r = self._resid.double()  # h^{ℓ−1}, set by this layer's input_layernorm pre-hook
                for t, v in turn_terms(r, self._flat(h_out).double() - r).items():
                    self.scores[f"full_{t}"][i] = v
        return hook

    def reset(self) -> None:
        n = len(self.layers)
        self.scores = {**{name: np.full(n, np.nan) for name in (*self.a_names, *TURN_SCORES, *TURN_NAMES)},
                       **{name: np.full((n, self.n_heads), np.nan) for name in HEAD_SCORES}}

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()


def collect(args: Namespace) -> tuple[dict[str, np.ndarray], np.ndarray, Alignment]:
    """Per-text scores (A_*: (N, L), HEAD_SCORES: (N, L, H)), labels (N,) and the hook object."""
    inference = Inference(model_name=args.model)
    align = Alignment(inference.model)
    data = load_data(args=args)[args.split]
    rows, labels = {name: [] for name in align.scores}, []
    for item in tqdm(data, desc="Running texts"):
        align.reset()
        inference.run(item, args)
        for name, value in align.scores.items():
            rows[name].append(value)
        labels.append(int(item["label"]))
    align.remove()
    del inference
    torch.cuda.empty_cache()
    return {name: np.stack(v) for name, v in rows.items()}, np.asarray(labels), align


def report(name: str, s: dict[str, np.ndarray]) -> None:
    for layer, (a, d, hm, mm) in enumerate(zip(s["auc"], s["cohen_d"], s["human_mean"], s["machine_mean"]), start=1):
        print(f"  {name} layer {layer:2d}: human {hm:.4f}, machine {mm:.4f}, Cohen's d {d:+.3f}, AUC {a:.4f}")


def panel(stats: dict[str, np.ndarray], quantity: str, title: str) -> dict:
    return {"stats": stats, "quantity": quantity, "title": title, "auc": False, "cohen_d": True}


def gap_figure(gaps: dict[str, tuple[np.ndarray, np.ndarray]], title: str) -> plt.Figure:
    """gaps: {name: (human − machine mean, 95% CI half-width)}, each (L,); one line per name, layers 1..L."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    labels = {"gap": "gap: φ = ∠(r, m), full attention write",
              "gap_abl": "gap_abl: φ_abl = ∠(r, m_abl), all heads' u₁ writes removed"}
    for name, (mean, ci) in gaps.items():
        layers = np.arange(1, len(mean) + 1)
        ax.fill_between(layers, mean - ci, mean + ci, color=GAP_COLORS[name], alpha=0.15, linewidth=0)
        ax.plot(layers, mean, color=GAP_COLORS[name], linewidth=2, label=f"{labels[name]} (±95% CI)")
    ax.axhline(0, color="#90A4AE", linewidth=0.8)
    ax.set_xlim(1, len(mean))
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("mean human − mean machine (°)")
    ax.set_title("Turn caused by attention, human − machine")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def decomposition_figure(stats: dict[str, dict[str, np.ndarray]], title: str) -> plt.Figure:
    """One panel per update: human − machine gap of log tan φ (black) and of its signed terms, each ±95% CI."""
    configure_plot_style()
    fig, axes = plt.subplots(1, len(UPDATES), figsize=(7.5 * len(UPDATES), 4.6), squeeze=False)
    for ax, (u, u_label) in zip(axes[0], UPDATES.items()):
        lines = [(f"{u}_log_tan", 1, r"$\log\tan\varphi$ (total)", "#000000")]
        lines += [(f"{u}_{t}", sign, label, color) for t, sign, label, color in TERM_PARTS]
        for name, sign, label, color in lines:
            gap = -sign * stats[name]["diff_mean"]  # signed contribution of human − machine
            ci = stats[name]["diff_ci"]
            layers = np.arange(1, len(gap) + 1)
            ax.fill_between(layers, gap - ci, gap + ci, color=color, alpha=0.15, linewidth=0)
            ax.plot(layers, gap, color=color, linewidth=2.4 if color == "#000000" else 1.8, label=label)
        ax.axhline(0, color="#90A4AE", linewidth=0.8)
        ax.set_xlim(1, len(gap))
        ax.set_xlabel("transformer layer")
        ax.set_ylabel("human − machine (log units)")
        ax.set_title(u_label)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="l8b")
    parser.add_argument("--dataset", type=str, default="drlXDomainLen_wiki_300_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    args.model_name = args.model

    scores, labels, align = collect(args)
    assert all(np.isfinite(v).all() for v in scores.values()), "a hook did not fire for some text"
    head_mean = {name: scores[name].mean(axis=-1) for name in HEAD_SCORES}  # (N, L)
    stats = {**{name: compare(scores[name], labels) for name in (*align.a_names, *TURN_SCORES)},
             **{name: compare(v, labels) for name, v in head_mean.items()}}
    for name, s in stats.items():
        report(name, s)
    # gap = mean human − mean machine = −(machine − human); same CI half-width
    gaps = {"gap": (-stats["phi"]["diff_mean"], stats["phi"]["diff_ci"]),
            "gap_abl": (-stats["phi_abl"]["diff_mean"], stats["phi_abl"]["diff_ci"])}
    for layer, (g, ga) in enumerate(zip(gaps["gap"][0], gaps["gap_abl"][0]), start=1):
        print(f"  layer {layer:2d}: gap {g:+.4f}°, gap_abl {ga:+.4f}°")
    turn_stats = {name: compare(scores[name], labels) for name in TURN_NAMES if not name.endswith("excluded")}
    for u in UPDATES:
        print(f"{u}: human − machine gap of log tan φ = log‖b‖ − log‖r‖ − log(1 + a/‖r‖)")
        for layer in range(len(align.layers)):
            g = {t: -turn_stats[f"{u}_{t}"]["diff_mean"][layer] for t in ("log_tan", "log_b", "log_r", "log_1pa")}
            excluded = scores[f"{u}_excluded"][:, layer].mean()
            print(f"  layer {layer + 1:2d}: log tan {g['log_tan']:+.4f} = log‖b‖ {g['log_b']:+.4f} "
                  f"− log‖r‖ {g['log_r']:+.4f} − log(1+a/‖r‖) {g['log_1pa']:+.4f}; excluded {100 * excluded:.2f}%")

    rand_in = 90 - np.rad2deg(np.sqrt(2 / (np.pi * align.d_in)))
    rand_out = 90 - np.rad2deg(np.sqrt(2 / (np.pi * align.d_model)))
    a_ref = {"A_1": np.sqrt(2 / (np.pi * align.d_in)), **{f"A_{k}": k / align.d_in for k in align.ks}}
    a_quantity = {"A_1": r"$A_1 = |\hat{x}^\top v_1|$",
                  **{f"A_{k}": rf"$A_{{{k}}} = \sum_{{i \leq {k}}} (\hat{{x}}^\top v_i)^2$" for k in align.ks}}
    a_panels = [panel(stats[name], a_quantity[name], f"{name} (random unit vector ≈ {a_ref[name]:.3g})")
                for name in align.a_names]
    theta_panels = [
        panel(stats["theta_in"], r"$\theta_{in} = \arccos|\hat{x}^\top v_1|$ (°)",
              f"θ_in: input token vs OV top input direction (random ≈ {rand_in:.1f}°)"),
        panel(stats["theta_out"], r"$\theta_{out} = \arccos|\hat{r}^\top u_1|$ (°)",
              f"θ_out: residual token vs OV top output direction (random ≈ {rand_out:.1f}°)"),
    ]
    run = f"{args.model}, {args.dataset}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"vproj_align_{args.model}_{args.dataset}_s{args.seed}.pdf")
    figures = (
        figure([a_panels], f"Alignment of v_proj inputs with the top input directions of W_V, {run}",
               first_layer=1, xlabel="transformer layer", groups=True),
        figure([theta_panels], f"Angles to the top singular directions of the per-head OV circuit, mean over heads, "
               f"{run}", first_layer=1, xlabel="transformer layer", groups=True),
        gap_figure(gaps, f"Ablating every head's top OV output direction u₁ (local, all heads at once), {run}"),
        decomposition_figure(turn_stats, r"Turn split, $\log\tan\varphi = \log\|b\| - \log\|r\| - \log(1 + a_\parallel/\|r\|)$, "
                             + run),
    )
    with PdfPages(path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
