import os
import json
import inspect
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm
from src.inference import Inference
from src.utils import evaluation, load_data, return_device

from src.config import Config
cfg = Config()

READOUTS = ("norm", "norm_layers", "kl", "entropy")
DIRECTIONS = ("subspace", "orthogonal")
POOLS = ("mean", "median")
PROFILES = ("subspace", "orthogonal", "ratio")
# ratio: subspace / orthogonal, per token (and layer) before pooling.
RATIOS = {"ratio": ("subspace", "orthogonal")}
# Hypothesis: machine text is more stable (smaller response to the nudge), so the detection score is −response.
SIGN = -1
# Positions 0..SKIP_FIRST−1 are never probed (attention sink and very short contexts).
SKIP_FIRST = 4


class PrefixCache():
    """
    Minimal stand-in for an HF Cache: attention layer i concatenates the clean keys/values of positions 0..t−1
    (from the clean pass, shared across the batch) in front of the probe position's own keys/values. Nothing is stored,
    so the clean prefix is reused unchanged for every probe.
    """

    def __init__(self, keys: dict[int, torch.Tensor], values: dict[int, torch.Tensor], t: int, batch: int) -> None:
        self.keys, self.values, self.t, self.batch = keys, values, t, batch

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs):
        prefix_k = self.keys[layer_idx][:, :, :self.t].expand(self.batch, -1, -1, -1)
        prefix_v = self.values[layer_idx][:, :, :self.t].expand(self.batch, -1, -1, -1)
        return torch.cat([prefix_k, key_states], dim=-2), torch.cat([prefix_v, value_states], dim=-2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.t


class ContextPerturbation():
    """
    Causal perturbation probe. At probe layer l (hidden_states[l], the output of transformer layer l, embeddings = 0),
    for probe token t with k preceding states C_t = [h_{t−k}, ..., h_{t−1}]:
        C_tᵀ = Q R (QR), keep the columns of Q with |R_ii| > tol (tol as in the CSD code), skip t if rank < 2;
        u = Q a / ‖Q a‖, a ~ N(0, I_r)            (uniform random direction within span(C_t))
        h_t ← h_t + ε u,  ε = rel_eps · ‖h_t‖      (position t only)
    then layers l+1..L, final norm and lm_head are re-run, and
        s  = ‖h^L_pert,t − h^L_clean,t‖ / ‖δ‖      (h^L: last-layer output before the final norm; δ the nudge as
                                                  applied in the model dtype, ≈ ε)
        s_j = ‖h^j_pert,t − h^j_clean,t‖ / ‖δ‖ for every layer j = l+1..L after the nudge (s_L = s): the response
        profile through depth; norm_layers = mean_j s_j,
        kl = KL(p_clean,t ‖ p_pert,t),
        entropy = |H(p_pert,t) − H(p_clean,t)|, how much the next-token entropy at t changes.
    Because attention is causal, nudging h_t leaves positions < t unchanged and position t's readout depends only on
    positions ≤ t; the tail is therefore re-run for position t alone, attending to the clean keys/values of positions
    < t cached from the clean pass. This is exact (equal to re-running all positions) and far cheaper.
    Contrast: orthogonal directions u = (I − Q Qᵀ) g / ‖·‖, g ~ N(0, I_d) (uniform within the complement of span(C_t),
    i.e. away from the features the context uses). The clean reference is an unperturbed row in the same batch, so
    all readouts are differences from the same computation path. Per token: mean over n_dirs directions of each kind;
    per text: mean and median over probe tokens, plus the per-token ratio subspace / orthogonal pooled the same way.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.device = return_device()
        self.inference = Inference(model_name=args.model)
        self.model = self.inference.model  # same dtype as Inference (the checkpoint's, bf16 for l8b)
        self.dtype = next(self.model.parameters()).dtype
        self.tokenizer = self.inference.tokenizer
        self.backbone = self.model.model
        self.layers = self.backbone.layers
        self.num_layers = len(self.layers)
        if not 1 <= args.probe_layer < self.num_layers:
            raise ValueError(f"--probe_layer must be in [1, {self.num_layers - 1}] (hidden_states index).")
        # Verify the decoder-layer signature: newer HF versions take the rotary embeddings precomputed.
        params = inspect.signature(self.layers[0].forward).parameters
        self.pass_position_embeddings = "position_embeddings" in params
        self.pass_position_ids = "position_ids" in params
        self.cache_kwarg = "past_key_values" if "past_key_values" in params else "past_key_value"

    def clean_pass(self, text: str) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """hidden_states[l] (T, d) and the clean keys/values of layers l..L−1 (1, kv_heads, T, head_dim)."""
        inputs = self.tokenizer(text, truncation=True, add_special_tokens=False, max_length=1024, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=True)
        cache = outputs.past_key_values
        keys, values = {}, {}
        for i in range(self.args.probe_layer, self.num_layers):
            if hasattr(cache, "layers"):
                keys[i], values[i] = cache.layers[i].keys, cache.layers[i].values
            else:
                keys[i], values[i] = cache.key_cache[i], cache.value_cache[i]
        return outputs.hidden_states[self.args.probe_layer][0].float(), keys, values

    def subspace_basis(self, context: torch.Tensor) -> torch.Tensor:
        """Orthonormal basis of span(context rows) (d, r), dropping directions with |R_ii| ≤ tol (as in CSD)."""
        basis = context.T
        q, r = torch.linalg.qr(basis, mode="reduced")
        diagonal = torch.abs(torch.diag(r))
        eps = torch.finfo(basis.dtype).eps
        tolerance = eps * max(basis.shape) * diagonal.max().clamp_min(eps)
        return q[:, diagonal > tolerance]

    @torch.no_grad()
    def tail(self, x: torch.Tensor, t: int, keys: dict, values: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run x (B, 1, d) at position t through layers l..L−1: the states after every layer, hidden_states l+1..L
        (B, L − l, d; the last one before the final norm), and the log-probs (B, V).
        """
        batch = x.shape[0]
        position_ids = torch.full((batch, 1), t, device=self.device, dtype=torch.long)
        cache = PrefixCache(keys, values, t, batch)
        kwargs = {"attention_mask": None, self.cache_kwarg: cache}
        if self.pass_position_ids:
            kwargs["position_ids"] = position_ids
        if self.pass_position_embeddings:
            kwargs["position_embeddings"] = self.backbone.rotary_emb(x, position_ids=position_ids)
        states = []
        for layer in self.layers[self.args.probe_layer:]:
            x = layer(x, **kwargs)
            x = x[0] if isinstance(x, tuple) else x
            states.append(x[:, 0].float())
        logits = self.model.lm_head(self.backbone.norm(x[:, 0]))
        return torch.stack(states, dim=1), torch.log_softmax(logits.float(), dim=-1)

    def probe_token(self, h: torch.Tensor, t: int, keys: dict, values: dict,
                    generator: torch.Generator) -> dict | None:
        """
        Direction-averaged readouts at token t for each kind of direction (DIRECTIONS), and the per-layer response
        profile {"profile": {direction: (L − l,)}}; None if rank < 2.
        """
        q = self.subspace_basis(h[t - self.args.csd_window:t])
        rank, d, n = q.shape[1], h.shape[1], self.args.n_dirs
        if rank < 2:
            return None
        a = torch.randn(rank, n, generator=generator).to(self.device)
        u_sub = (q @ a).T  # (n, d)
        g = torch.randn(n, d, generator=generator).to(self.device)
        u_orth = g - (g @ q) @ q.T  # (n, d): g with its span(C_t) component removed
        # Rows per kind, in DIRECTIONS order.
        directions = torch.cat([u_sub, u_orth])
        directions = directions / directions.norm(dim=-1, keepdim=True)
        eps = self.args.rel_eps * h[t].norm()
        x = torch.cat([h[t][None], h[t][None] + eps * directions])[:, None].to(self.dtype)  # (1 + 2n, 1, d); row 0 = clean
        # In bf16 the applied nudge is rounded, so normalise by its realised size rather than ε.
        applied = (x[1:, 0].float() - x[0, 0].float()).norm(dim=-1)
        states, log_probs = self.tail(x, t, keys, values)
        profile = (states[1:] - states[:1]).norm(dim=-1) / applied[:, None]  # (2n, L − l): s_j per direction
        kl = (log_probs[0].exp() * (log_probs[0] - log_probs[1:])).sum(dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (1 + 2n,)
        readouts = {"norm": profile[:, -1], "norm_layers": profile.mean(dim=1), "kl": kl,
                    "entropy": (entropy[1:] - entropy[0]).abs()}
        rows = {dr: slice(i * n, (i + 1) * n) for i, dr in enumerate(DIRECTIONS)}
        return {
            readout: {dr: v[rows[dr]].mean().item() for dr in DIRECTIONS}
            for readout, v in readouts.items()
        } | {"rank": rank,
             "profile": {dr: profile[rows[dr]].mean(dim=0).cpu().numpy() for dr in DIRECTIONS}}

    def score(self, text: str, index: int) -> tuple[dict[str, float], dict[str, int], dict[str, np.ndarray]]:
        """
        Per-text pooled readouts {readout_direction_pool: float}, token counts, and the response profile averaged over
        tokens {PROFILES: (L − l,)} (ratios: per token and layer, then the mean).
        """
        h, keys, values = self.clean_pass(text)
        generator = torch.Generator().manual_seed(self.args.seed + index)
        start = max(SKIP_FIRST, self.args.csd_window)
        per_token = {(r, dr): [] for r in READOUTS for dr in DIRECTIONS}
        ranks, skipped = [], 0
        profiles = {dr: [] for dr in DIRECTIONS}
        for t in range(start, h.shape[0], self.args.token_stride):
            result = self.probe_token(h, t, keys, values, generator)
            if result is None:
                skipped += 1
                continue
            ranks.append(result["rank"])
            for dr in DIRECTIONS:
                profiles[dr].append(result["profile"][dr])
            for r in READOUTS:
                for dr in DIRECTIONS:
                    per_token[(r, dr)].append(result[r][dr])
        scores = {}
        for r in READOUTS:
            series = {dr: np.asarray(per_token[(r, dr)]) for dr in DIRECTIONS}
            series |= {name: series[a] / np.maximum(series[b], 1e-12) for name, (a, b) in RATIOS.items()}
            for name, v in series.items():
                for pool in POOLS:
                    scores[f"{r}_{name}_{pool}"] = float(getattr(np, pool)(v)) if v.size else float("nan")
        counts = {"n_probed": len(ranks), "n_skipped_rank": skipped,
                  "mean_rank": float(np.mean(ranks)) if ranks else float("nan")}
        n_after = self.num_layers - self.args.probe_layer
        if ranks:
            stacked = {dr: np.stack(profiles[dr]) for dr in DIRECTIONS}  # (tokens, L − l)
            profile = {dr: v.mean(axis=0) for dr, v in stacked.items()}
            profile |= {name: (stacked[a] / np.maximum(stacked[b], 1e-12)).mean(axis=0)
                        for name, (a, b) in RATIOS.items()}
        else:
            profile = {name: np.full(n_after, np.nan) for name in PROFILES}
        return scores, counts, profile

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

        per_text, per_text_counts, per_text_profiles = [], [], []
        for index, item in enumerate(tqdm(test_data, desc=f"Perturbation probe (l={args.probe_layer})")):
            scores, counts, profile = self.score(item["text"], index)
            per_text.append(scores)
            per_text_counts.append(counts)
            per_text_profiles.append(profile)
        names = list(per_text[0].keys())
        values = {name: np.asarray([s[name] for s in per_text], dtype=float) for name in names}  # (N,)
        metrics = {name: self.evaluate(labels, SIGN * v) for name, v in values.items()}
        for name, m in metrics.items():
            print(f"AUROC {name}: {m['auroc']:.4f}")
        main_name = f"{args.readout}_subspace_mean"
        print(f"Main ({main_name}): {metrics[main_name]['auroc']:.4f}, "
              f"orthogonal ({args.readout}_orthogonal_mean): {metrics[f'{args.readout}_orthogonal_mean']['auroc']:.4f}, "
              f"ratio ({args.readout}_ratio_mean): {metrics[f'{args.readout}_ratio_mean']['auroc']:.4f}")
        # Response profile through depth: hidden_states l+1..L, one AUROC per layer after the nudge.
        profile_layers = list(range(args.probe_layer + 1, self.num_layers + 1))
        profiles = {name: np.stack([p[name] for p in per_text_profiles]) for name in PROFILES}  # (N, L − l)
        auroc_per_layer = {
            name: [self.evaluate(labels, SIGN * v[:, j])["auroc"] for j in range(v.shape[1])]
            for name, v in profiles.items()
        }
        for name, aurocs in auroc_per_layer.items():
            print(f"AUROC per layer, norm profile {name}: "
                  + ", ".join(f"{layer}: {a:.3f}" for layer, a in zip(profile_layers, aurocs)))
        counts = {key: np.asarray([c[key] for c in per_text_counts], dtype=float) for key in per_text_counts[0]}
        print(f"Probed tokens/text: {counts['n_probed'].mean():.1f}, skipped (rank < 2): "
              f"{int(counts['n_skipped_rank'].sum())} in total")

        method = "perturb"
        setting = (f"l{args.probe_layer}_k{args.csd_window}_eps{args.rel_eps}_n{args.n_dirs}"
                   f"_stride{args.token_stride}")
        file_name = f"{method}_{setting}_{args.model_name}_{args.dataset}_s{args.seed}"
        output = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "score_sign": SIGN,
            "metrics": metrics[main_name],
            "metrics_orthogonal": metrics[f"{args.readout}_orthogonal_mean"],
            "metrics_ratio": metrics[f"{args.readout}_ratio_mean"],
            "metrics_all": metrics,
            "mean_by_group": {name: {"human": float(np.nanmean(v[labels == 0])),
                                     "machine": float(np.nanmean(v[labels == 1]))}
                              for name, v in values.items()},
            "profile_layers": profile_layers,
            "profile_auroc_per_layer": auroc_per_layer,
            "profile_mean_by_group": {name: {"human": np.nanmean(v[labels == 0], axis=0).tolist(),
                                             "machine": np.nanmean(v[labels == 1], axis=0).tolist()}
                                      for name, v in profiles.items()},
            "profiles": {name: v.tolist() for name, v in profiles.items()},
            "n_skipped_rank_total": int(counts["n_skipped_rank"].sum()),
            "counts": {key: v.tolist() for key, v in counts.items()},
            "values": {name: v.tolist() for name, v in values.items()},
            "labels": labels.tolist(),
        }

        output_dir = os.path.join(cfg.zero_output_dir, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            json.dump(output, f, indent=4)
        return metrics


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--probe_layer", type=int, default=16,
                        help="hidden_states index l (0 = embeddings) to perturb; layers l+1..L are re-run.")
    parser.add_argument("--csd_window", type=int, default=8, help="k preceding states spanning the subspace.")
    parser.add_argument("--rel_eps", type=float, default=0.02, help="ε = rel_eps · ‖h_t‖.")
    parser.add_argument("--n_dirs", type=int, default=8, help="Probe directions per token (each of subspace/orthogonal).")
    parser.add_argument("--token_stride", type=int, default=4, help="Probe every stride-th token.")
    parser.add_argument("--readout", type=str, default="norm", choices=READOUTS,
                        help="Readout reported as the main metric (all are computed and saved).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name = args.model

    model = ContextPerturbation(args=args)
    model.run(args)


if __name__ == "__main__":
    main()
