"""All-token steering: does pushing every token along the human-machine direction change the geometry of later layers?

Analogous to prefix.py (same data split, hook point, layer list, strength grid, norm-outlier filter and angle
definitions), but the vector is added to every token position, not only to a prefix. Only computes and saves results;
no analysis.

1. Split: as in prefix.py (70% train for the directions, the rest steered; whole pairs on one side for paired data).
2. Direction (the AI direction), per layer l = 1..P (P = min(--max_layer, L)), over positions 1..T-1 of all train
   texts (position 0, the attention sink, is left out everywhere):
     r_l = (mu_M - mu_H) / ||mu_M - mu_H||   (+r = towards machine)
     u_l = mean token norm ||h_t|| over all kept train tokens (both classes)
   Norm outliers (as in prefix.py): at every layer, tokens whose norm is > MAX_NORM_RATIO (2) x the median token norm
   of their text at that layer (unsteered pass) are left out of the means, u_l, the test scores, the angles and v . h
   at that layer. The mask is computed once per text from an unsteered pass and used for every steered row.
3. Controls (--control picks one per run; the real direction is always run), unit length:
     random        N_RANDOM (20) vectors uniform on the sphere, g / ||g||, g ~ N(0, I_d); drawn once (fixed seed,
                   seed + RANDOM_SEED_OFFSET) and the same at every layer (prefix.py's random directions)
     data_shuffle  N_SHUFFLE (20) text-level permutations of the train labels (class sizes kept, seed +
                   SHUFFLE_SEED_OFFSET); difference in means (machine - human) over all kept tokens, per layer,
                   normalized (prefix.py's shuffle)
   Controls are scaled and injected exactly like the real direction.
4. Passes, per test text. One unsteered single-text pass caches the states at all layers (norm mask, test score).
   Then every batch holds one unsteered row (row 0) plus up to batch_size - 1 steered rows, so each effect is taken
   against an unsteered pass from the same batch (cancels bf16 batch noise). Steered row, for steered layer l,
   strength c (prefix.py's STRENGTHS, both signs) and vector v:
     h_t <- h_t + s * c * u_l * v   at layer l only, at every position t = 1..T-1
   with s = -1 for machine texts and s = +1 for human texts: c > 0 pushes every text towards the other class
   (machine along -r towards human, human along +r towards machine), c < 0 further towards its own class. The same
   class sign is used for the control vectors.
   Sanity check: in every steered row, the hidden states at layers < l (and at layer l before the vector is added)
   must equal the baseline row of the same batch; the run stops if the largest |difference| exceeds SANITY_ATOL.
5. Angles at every layer m (layer 0 = embeddings), over the kept positions among 1..T-1:
     local_angle  mean angle (deg) between adjacent tokens (t, t+1), both kept at layer m
     cross_angle  mean angle (deg) between token t at layers m - 1 and m, kept at both
     pairwise_cos representation dispersion: mean cosine similarity over all pairs of distinct kept tokens at
                  layer m (higher = less dispersed)
   each twice: on the full states, and after projecting out the injected vector v of that row from the states of
   every layer (x <- x - (v . x) v). The baseline is computed on the full states and projected with every v used, so
   each steered result has a matching baseline. Effects are kept at layers m > l only (nan at m <= l).
6. Persistence: mean over the kept positions of v . h at every layer, for the steered row and for the baseline row
   (same v), i.e. how much of the injected vector is still there.

Everything goes to one new file output/steering/alltoken_<control>_<model>_<dataset>_s<seed>_<timestamp>.npz, with the
metadata as a JSON string under "meta"; nothing is overwritten.
"""

import json
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.inference import Inference
from src.steering.prefix import (MAX_NORM_RATIO, METRICS, MIN_TOKENS, N_RANDOM, N_SHUFFLE, OUTPUT_DIR, RANDOM_SEED_OFFSET,
                                 SHUFFLE_SEED_OFFSET, STRENGTHS, TRAIN_FRACTION, PrefixSteering, keep_mask,
                                 masked_layer_stats, masked_mean_angle)

APPROACHES = ("token",)
# largest allowed |h_steered - h_baseline| at layers before the steered layer (same batch, expected exactly 0)
SANITY_ATOL = 1e-3
EPS = 1e-12


class AllTokenHooks:
    """Forward hooks on the embeddings and every decoder layer. Row k of a batch may add one vector to all positions
    1..T-1 at one layer; at every layer the angles of every row (full and with that row's v projected out), the
    matching projected baseline angles (row 0 with row k's v) and the mean v . h are recorded over the kept tokens.
    In cache mode the states of row 0 at every layer are stored instead."""

    def __init__(self, decoder: torch.nn.Module) -> None:
        self.active = False
        self.caching = False
        # (add (R, d) float32, at (R,) long, 0 = not steered; v (R, d) unit vector to project out, 0 for row 0)
        self.plan: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        # (L+1, T') bool over positions 1..T-1: tokens kept at each layer (0 = embeddings)
        self.keep: torch.Tensor | None = None
        self.prev: dict[str, torch.Tensor] | None = None
        self.out: dict[str, list[torch.Tensor]] = {}
        self.cache: list[torch.Tensor] = []
        self.max_diff = torch.zeros(())
        decoder.embed_tokens.register_forward_hook(self.embedding_hook)
        for i, layer in enumerate(decoder.layers):
            layer.register_forward_hook(self.hook(i + 1))

    def reset(self, active: bool = False, plan=None, keep: torch.Tensor | None = None, cache: bool = False) -> None:
        self.active, self.plan, self.keep, self.caching = active, plan, keep, cache
        self.prev, self.out, self.cache = None, {}, []
        self.max_diff = torch.zeros(())

    def prepare(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """States of positions 1..T-1 as float32, their squared norms, the coefficient of every row's own v (a) and the
        coefficients of the baseline row on every row's v (a0, (T', R))."""
        x = h[:, 1:].float()                                                       # (R, T', d)
        v = self.plan[2]                                                           # (R, d)
        return {"x": x, "n": (x * x).sum(dim=-1), "a": (x * v[:, None]).sum(dim=-1), "a0": x[0] @ v.T}

    def record(self, cur: dict[str, torch.Tensor], layer: int) -> None:
        x, n, a, a0 = cur["x"], cur["n"], cur["a"], cur["a0"]
        kl, kp = self.keep[layer], self.keep[layer - 1]                            # (T',) kept at this / previous layer
        n0 = n[0][:, None]                                                         # (T', 1) baseline row
        out = {}
        # adjacent tokens (t, t+1), both kept at this layer
        w = kl[1:] & kl[:-1]
        dot = (x[:, 1:] * x[:, :-1]).sum(dim=-1)                                   # (R, T'-1)
        out["local_angle"] = masked_mean_angle(dot / (n[:, 1:] * n[:, :-1]).clamp_min(EPS).sqrt(), w)
        num = dot - a[:, 1:] * a[:, :-1]
        den = ((n[:, 1:] - a[:, 1:] ** 2) * (n[:, :-1] - a[:, :-1] ** 2)).clamp_min(EPS).sqrt()
        out["local_angle_proj"] = masked_mean_angle(num / den, w)
        num = dot[0][:, None] - a0[1:] * a0[:-1]                                   # (T'-1, R)
        den = ((n0[1:] - a0[1:] ** 2) * (n0[:-1] - a0[:-1] ** 2)).clamp_min(EPS).sqrt()
        out["local_angle_proj_base"] = masked_mean_angle(num / den, w[:, None], dim=0)
        # same token at the previous layer, kept at both layers
        w = kl & kp
        p = self.prev
        dot = (x * p["x"]).sum(dim=-1)                                             # (R, T')
        out["cross_angle"] = masked_mean_angle(dot / (n * p["n"]).clamp_min(EPS).sqrt(), w)
        num = dot - a * p["a"]
        den = ((n - a ** 2) * (p["n"] - p["a"] ** 2)).clamp_min(EPS).sqrt()
        out["cross_angle_proj"] = masked_mean_angle(num / den, w)
        num = dot[0][:, None] - a0 * p["a0"]                                       # (T', R)
        den = ((n0 - a0 ** 2) * (p["n"][0][:, None] - p["a0"] ** 2)).clamp_min(EPS).sqrt()
        out["cross_angle_proj_base"] = masked_mean_angle(num / den, w[:, None], dim=0)
        # dispersion: mean cosine over pairs of distinct kept tokens, (||sum_t z_t||^2 - n) / (n (n - 1)) with unit
        # z_t; projected: z_t = (x_t - a_t v) / rho_t, rho_t = sqrt(n_t - a_t^2), and since v . sum_t x_t / rho_t =
        # sum_t a_t / rho_t, ||sum_t z_t||^2 = ||sum_t x_t / rho_t||^2 - (sum_t a_t / rho_t)^2
        k = kl.float()
        nk = k.sum()
        disp = lambda sq: (sq - nk) / (nk * (nk - 1))
        g = k / n.clamp_min(EPS).sqrt()                                            # (R, T')
        tot = (x * g[..., None]).sum(dim=1)                                        # (R, d)
        out["pairwise_cos"] = disp((tot * tot).sum(dim=-1))
        g = k / (n - a ** 2).clamp_min(EPS).sqrt()
        tot, b = (x * g[..., None]).sum(dim=1), (a * g).sum(dim=-1)
        out["pairwise_cos_proj"] = disp((tot * tot).sum(dim=-1) - b ** 2)
        g = k[:, None] / (n0 - a0 ** 2).clamp_min(EPS).sqrt()                      # (T', R)
        tot, b = x[0].T @ g, (a0 * g).sum(dim=0)                                   # (d, R), (R,)
        out["pairwise_cos_proj_base"] = disp((tot * tot).sum(dim=0) - b ** 2)
        # persistence of the injected vector, over the tokens kept at this layer
        k = kl.float()
        out["vh"] = (a * k).sum(dim=-1) / k.sum()
        out["vh_base"] = (a0 * k[:, None]).sum(dim=0) / k.sum()
        for key, val in out.items():
            self.out.setdefault(key, []).append(val)

    def embedding_hook(self, module, inputs, output) -> None:
        if self.caching:
            self.cache.append(output[0, 1:])
        if self.active:
            self.prev = self.prepare(output)

    def hook(self, layer: int):
        def forward(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.caching:
                self.cache.append(h[0, 1:])
            if not self.active:
                return output
            add, at, _ = self.plan
            # rows steered at this layer or later must still equal the baseline row (before the vector is added)
            same = at >= layer
            if bool(same.any()):
                diff = (h[same, 1:].float() - h[:1, 1:].float()).abs().max()
                self.max_diff = torch.maximum(self.max_diff.to(diff.device), diff)
            on = at == layer
            if bool(on.any()):
                x = h[:, 1:].float() + (add * on[:, None].float())[:, None]
                h = torch.cat([h[:, :1], x.to(h.dtype)], dim=1)
            cur = self.prepare(h)
            self.record(cur, layer)
            self.prev = cur
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return forward


class AllTokenSteering(PrefixSteering):
    """Reuses PrefixSteering.split, .tokenize, .cache_states and .directions (AI direction, random directions, label
    shuffles, norm filter); hooks and steering are all-token versions."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model, args=args)
        self.model = self.inference.model
        self.tokenizer = self.inference.tokenizer
        self.device = self.inference.device
        self.dtype = next(self.model.parameters()).dtype
        self.decoder = self.model.get_decoder()
        self.n_layers = self.model.config.num_hidden_layers
        self.P = min(args.max_layer, self.n_layers)
        self.strengths = np.asarray(STRENGTHS, dtype=np.float64)
        self.hooks = AllTokenHooks(self.decoder)

    # ------------------------------------------------------------------ steering
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, plan, keep: torch.Tensor) -> dict[str, np.ndarray]:
        """(R, T) ids -> {name: (R, L)} per row and layer, over the kept tokens."""
        self.hooks.reset(active=True, plan=plan, keep=keep)
        self.decoder(input_ids=ids, use_cache=False)
        out = {k: torch.stack(v, dim=1).cpu().numpy() for k, v in self.hooks.out.items()}
        out["max_diff"] = float(self.hooks.max_diff)
        self.hooks.reset()
        return out

    @torch.no_grad()
    def steer(self, test: list[dict], dirs: dict) -> dict:
        """Steer every test text along the real direction and the control vectors, kept in one bank
        (K = 1 real + controls, (K, P, d)); results are split into real and control at the end."""
        S, P, L = len(self.strengths), self.P, self.n_layers
        M = len(METRICS)
        # random directions are the same at every layer: (N_RANDOM, d) -> (N_RANDOM, P, d)
        ctrl = np.repeat(dirs["rand"][:, None], P, axis=1) if self.args.control == "random" else dirs["shuf"]
        bank = np.concatenate([dirs["r"], ctrl])                                            # (K, P, d), unit
        K = bank.shape[0]
        vec = torch.tensor(self.strengths[None, :, None, None] * dirs["u"][None, None, :, None] * bank[:, None],
                           device=self.device, dtype=torch.float32)                         # (K, S, P, d)
        vdir = torch.tensor(bank, device=self.device, dtype=torch.float32)
        conds = np.asarray([(k, j, l) for k in range(K) for j in range(S) for l in range(1, P + 1)])
        B = max(1, self.args.batch_size - 1)
        keys = ("baseline", "delta", "delta_proj", "baseline_proj", "vh_steered", "vh_baseline")
        acc = {k: [] for k in keys}
        kept, T_all, scores, removed, max_diff = [], [], [], [], 0.0
        for i, item in enumerate(tqdm(test, desc=f"Steering ({len(conds)} conditions per text)")):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            # unsteered single-text pass: norm-outlier mask (same for every row) and the text mean for the test AUROC
            states = self.cache_states(ids)
            keep = keep_mask(states)                                                        # (L+1, T-1)
            sums, n, _ = masked_layer_stats(states, keep, P)
            scores.append(((sums / n[:, None])[None] * dirs["r"]).sum(axis=-1))             # (1, P)
            removed.append(1 - keep.float().mean(dim=1).cpu().numpy())                      # (L+1,)
            del states
            # class sign: c > 0 pushes machine along -r (towards human) and human along +r (towards machine);
            # c < 0 pushes each text further towards its own class
            sign = -1.0 if item["label"] == 1 else 1.0
            doc = np.full((M, K, S, P, L), np.nan, dtype=np.float32)
            dproj = np.full((M, K, S, P, L), np.nan, dtype=np.float32)
            bproj = np.full((M, K, P, L), np.nan, dtype=np.float32)
            vh_st = np.full((K, S, P, L), np.nan, dtype=np.float32)
            vh_b = np.full((K, P, L), np.nan, dtype=np.float32)
            base = None
            for s in range(0, len(conds), B):
                c = conds[s:s + B]
                kk, jj, ll = c[:, 0], c[:, 1], c[:, 2]
                R = len(c) + 1
                add = torch.zeros((R, vec.shape[-1]), device=self.device)
                at = torch.zeros(R, dtype=torch.long, device=self.device)
                v = torch.zeros_like(add)   # row 0 (baseline): nothing projected out
                kt, jt, lt = (torch.as_tensor(a, device=self.device) for a in (kk, jj, ll))
                add[1:] = sign * vec[kt, jt, lt - 1]
                at[1:] = lt
                v[1:] = vdir[kt, lt - 1]
                res = self.forward(ids.expand(R, -1), (add, at, v), keep)
                if res["max_diff"] > SANITY_ATOL:
                    raise AssertionError(f"text {i}: hidden states before the steered layer differ from the "
                                         f"baseline by {res['max_diff']:.3g} > {SANITY_ATOL}")
                max_diff = max(max_diff, res["max_diff"])
                if base is None:
                    base = np.stack([res[m][0] for m in METRICS])                           # (M, L)
                for mi, m in enumerate(METRICS):
                    doc[mi, kk, jj, ll - 1] = res[m][1:] - res[m][:1]
                    dproj[mi, kk, jj, ll - 1] = res[f"{m}_proj"][1:] - res[f"{m}_proj_base"][1:]
                    bproj[mi, kk, ll - 1] = res[f"{m}_proj_base"][1:]
                vh_st[kk, jj, ll - 1] = res["vh"][1:]
                vh_b[kk, ll - 1] = res["vh_base"][1:]
            for k, val in zip(keys, (base, doc, dproj, bproj, vh_st, vh_b)):
                acc[k].append(val)
            kept.append(i), T_all.append(ids.shape[1])
        # effects only at layers m > l
        after = np.arange(1, L + 1)[None, :] > np.arange(1, P + 1)[:, None]                 # (P, L)
        baseline = np.stack(acc["baseline"])                                                # (N, M, L)
        delta = np.where(after, np.stack(acc["delta"]), np.nan).astype(np.float32)          # (N, M, K, S, P, L)
        delta_proj = np.where(after, np.stack(acc["delta_proj"]), np.nan).astype(np.float32)
        baseline_proj = np.stack(acc["baseline_proj"])                                      # (N, M, K, P, L)
        vh_steered, vh_baseline = np.stack(acc["vh_steered"]), np.stack(acc["vh_baseline"])  # (N,K,S,P,L) (N,K,P,L)
        labels = np.asarray([test[i]["label"] for i in kept])
        scores = np.stack(scores)                                                           # (N, 1, P)
        test_auroc = np.asarray([[roc_auc_score(labels, scores[:, 0, l]) for l in range(P)]])
        real, ctl = slice(0, 1), slice(1, K)
        tag = "rand" if self.args.control == "random" else "shuf"
        return {"baseline": baseline, "labels": labels, "kept": np.asarray(kept), "n_tokens": np.asarray(T_all),
                "removed_share": np.stack(removed),
                "test_score": scores, "test_auroc": test_auroc, "sanity_max_change_before_l": max_diff,
                "delta": delta[:, :, real],
                "steered": delta[:, :, real] + baseline[:, :, None, None, None],
                "delta_proj": delta_proj[:, :, real], "baseline_proj": baseline_proj[:, :, real],
                "steered_proj": delta_proj[:, :, real] + baseline_proj[:, :, real, None],
                "vh_steered": vh_steered[:, real], "vh_baseline": vh_baseline[:, real],
                f"delta_{tag}": delta[:, :, ctl], f"delta_{tag}_proj": delta_proj[:, :, ctl],
                f"baseline_{tag}_proj": baseline_proj[:, :, ctl],
                f"vh_{tag}_steered": vh_steered[:, ctl], f"vh_{tag}_baseline": vh_baseline[:, ctl]}

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        args = self.args
        train, test, split = self.split()
        dirs = self.directions(train)
        res = self.steer(test, dirs)

        created = time.strftime('%Y%m%d-%H%M%S')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"alltoken_{args.control}_{args.model}_{args.dataset}_s{args.seed}_"
                                        f"{created}.npz")
        if os.path.exists(path):
            raise FileExistsError(path)
        y = res["labels"]
        test_kept = [test[i] for i in res["kept"]]
        tag = "rand" if args.control == "random" else "shuf"
        meta = {
            "created": created,
            "model": args.model, "model_id": self.inference.model_id, "dtype": str(self.dtype),
            "dataset": args.dataset, "seed": args.seed, "train_fraction": TRAIN_FRACTION, "min_tokens": MIN_TOKENS,
            "max_layer": self.P, "n_layers": self.n_layers, "strengths": self.strengths.tolist(),
            "approaches": list(APPROACHES), "metrics": list(METRICS), "batch_size": args.batch_size,
            "n_train": dirs["n_train"], "n_train_tokens": dirs["n_train_tokens"],
            "n_test": {"human": int((y == 0).sum()), "machine": int((y == 1).sum())},
            "sanity_max_change_before_l": res["sanity_max_change_before_l"], "sanity_atol": SANITY_ATOL,
            "direction": "AI direction: r_l = mu_M - mu_H normalized (+r = machine), class means over all kept "
                         "train tokens 1..T-1",
            "u": "mean train token norm at layer l (kept tokens 1..T-1, both classes)",
            "norm_outliers": f"at every layer, tokens with norm > {MAX_NORM_RATIO} x the median token norm of their "
                             "text (unsteered pass, positions 1..T-1) are left out of the means, u, test scores, "
                             "angles and v . h at that layer; steering vectors are still added at every position",
            "removed_share_train": dirs["removed_share_train"],
            "intervention": "h_t <- h_t + s * c * u_l * v at every position t = 1..T-1, at layer l only; "
                            "s = -1 for machine texts, s = +1 for human texts; c > 0 = towards the other class "
                            "(machine along -r towards human, human along +r towards machine), c < 0 = further "
                            "towards the own class; the same sign for the control vectors",
            "baseline": "an unsteered single-text pass gives the norm mask and test score; the effect of a steered "
                        "row is taken against the unsteered row 0 of its own batch; 'baseline' holds the first "
                        "batch's row 0",
            "local_angle": "mean angle (deg) between adjacent tokens (t, t+1), t = 1..T-2, both kept",
            "cross_angle": "mean angle (deg) between token t at layers m-1 and m (layer 0 = embeddings), kept at both",
            "pairwise_cos": "representation dispersion: mean cosine similarity over all pairs of distinct kept tokens "
                            "at layer m",
            "effect": "delta = steered - baseline per text; nan at measurement layers m <= l",
            "projected": "*_proj: the same angles after projecting the injected unit vector v (of that row: r_l for "
                         "the real direction, the control vector for controls) out of the states at every layer "
                         "(x <- x - (v . x) v); baseline_*proj = baseline row projected with the same v; "
                         "delta_proj = steered_proj - baseline_proj (same batch)",
            "persistence": "vh_* = mean over kept t of v . h_t at every layer 1..L (unsigned v; the injected vector "
                           "is s * c * u_l * v), for the steered row and the baseline row",
            "control": args.control,
            "random": {"n": N_RANDOM, "seed": args.seed + RANDOM_SEED_OFFSET,
                       "sampling": "v = g / ||g||, g ~ N(0, I_d) (numpy default_rng), drawn once, the same at every layer; "
                                   "scaled and injected like the real direction"},
            "shuffle": {"n": N_SHUFFLE, "seed": args.seed + SHUFFLE_SEED_OFFSET,
                        "sampling": "text labels of the train split permuted (class sizes kept), every token takes "
                                    "its text's label; mu_M - mu_H over all kept tokens per layer, unit length; "
                                    "scaled and injected like the real direction"},
            "auroc": "train/test AUROC of the text mean (kept tokens) along r_l for detecting machine texts",
            "layer_convention": "layer l = output of decoder layer l (1..L), before the final norm",
            "shapes": {
                "baseline": "(N, metric, L)", "removed_share": "(N, layer 0..L) share of tokens left out",
                "steered|delta|steered_proj|delta_proj": "(N, metric, approach, strength, steer layer 1..P, "
                                                         "measurement layer 1..L)",
                "baseline_proj": "(N, metric, approach, steer layer 1..P, measurement layer 1..L)",
                "vh_steered": "(N, approach, strength, steer layer 1..P, measurement layer 1..L)",
                "vh_baseline": "(N, approach, steer layer 1..P, measurement layer 1..L)",
                "r|mu_h|mu_m": "(approach, P, d)", "u": "(P,)", "gap_norm|train_auroc|test_auroc": "(approach, P)",
                "test_score": "(N, approach, P)",
                f"{tag}": "(n_rand, d)" if tag == "rand" else "(n_shuf, P, d)",
                f"delta_{tag}|delta_{tag}_proj": f"(N, metric, n_{tag}, strength, steer layer 1..P, "
                                                 f"measurement layer 1..L)",
                f"baseline_{tag}_proj": f"(N, metric, n_{tag}, steer layer 1..P, measurement layer 1..L)",
                f"vh_{tag}_steered": f"(N, n_{tag}, strength, steer layer 1..P, measurement layer 1..L)",
                f"vh_{tag}_baseline": f"(N, n_{tag}, steer layer 1..P, measurement layer 1..L)",
            },
        }
        meta.pop("shuffle" if args.control == "random" else "random")
        control = {tag: (dirs["rand"] if args.control == "random" else dirs["shuf"]).astype(np.float32)}
        control.update({k: res[k] for k in (f"delta_{tag}", f"delta_{tag}_proj", f"baseline_{tag}_proj",
                                            f"vh_{tag}_steered", f"vh_{tag}_baseline")})
        if args.control == "data_shuffle":
            control["shuf_labels"] = dirs["shuf_labels"]
        extra = {}
        if all("pair_id" in item for item in test_kept):
            extra["pair_id"] = np.asarray([str(item["pair_id"]) for item in test_kept])
        np.savez_compressed(
            path, meta=np.asarray(json.dumps(meta)),
            baseline=res["baseline"], steered=res["steered"], delta=res["delta"], labels=y,
            n_tokens=res["n_tokens"], removed_share=res["removed_share"],
            test_index=split["test_index"][res["kept"]], train_index=split["train_index"],
            metrics=np.asarray(METRICS), approaches=np.asarray(APPROACHES), strengths=self.strengths,
            steer_layers=np.arange(1, self.P + 1), layers=np.arange(1, self.n_layers + 1),
            r=dirs["r"].astype(np.float32), u=dirs["u"], gap_norm=dirs["gap_norm"],
            mu_h=dirs["mu_h"].astype(np.float32), mu_m=dirs["mu_m"].astype(np.float32),
            train_auroc=dirs["train_auroc"], test_auroc=res["test_auroc"], test_score=res["test_score"],
            baseline_proj=res["baseline_proj"], steered_proj=res["steered_proj"], delta_proj=res["delta_proj"],
            vh_steered=res["vh_steered"], vh_baseline=res["vh_baseline"],
            **control, **extra,
        )
        print(f"\nSaved {path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control", choices=("random", "data_shuffle"), required=True,
                        help="random: N_RANDOM random unit vectors (same at every layer); data_shuffle: N_SHUFFLE label-shuffled "
                             "mean-difference directions")
    parser.add_argument("--max_layer", type=int, default=20, help="steer at layers 1..max_layer")
    parser.add_argument("--batch_size", type=int, default=81, help="rows per forward pass (incl. one baseline)")
    args = parser.parse_args()
    args.model_name = args.model
    return args


def main() -> None:
    AllTokenSteering(parse_args()).run()


if __name__ == "__main__":
    main()
