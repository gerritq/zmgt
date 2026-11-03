"""Prefix steering: does pushing the prefix along the machine-human direction change the geometry of the suffix?

1. Split. All splits of the dataset are pooled and split TRAIN_FRACTION (70%) for the directions, the rest steered.
   Stratified by label; if the items carry a "pair_id" (paired data, e.g. drlXpaired), whole pairs go to one side.
2. Direction (mean differencing over tokens), at layers l = 1..P, P = min(--max_layer, L), over positions 1..T-1 of
   every train text (position 0, the attention sink, excluded): mu_H, mu_M = mean over all tokens of all texts of a
   class (every token weighs the same).
   r_l = (mu_M - mu_H) / ||mu_M - mu_H||   (+r = machine, -r = human)
   u_l = mean token norm ||h_t|| over all train tokens 1..T-1 (both classes).
   (Per-text mean pooling was dropped: on fixed-length texts it gives exactly the same direction. The arrays keep an
   "approach" axis of size 1, so the file layout is unchanged.)
3. Split point. Positions 1..T-1 of a text are split at p = floor(SPLIT_FRAC * (T-1)): prefix = positions 1..p,
   suffix = positions p+1..T-1.
4. Passes, per steered text, all in the same batch:
     baseline   no intervention
     steered    at layer l only, h_t <- h_t + c * u_l * r_l for the prefix positions t = 1..p; for every
                strength c (STRENGTHS; c > 0 towards machine, c < 0 towards human) and l = 1..P
   The forward pass then runs on through layers l+1..L. The suffix is never touched directly: it changes only
   through attention to the steered prefix.
5. Suffix geometry at every layer m = 1..L (layer 0 = embeddings), on suffix positions only:
     local_angle  mean angle (degrees) between adjacent suffix tokens (t, t+1), t >= p+1 (the boundary pair (p, p+1)
                  is left out)
     cross_angle  mean angle (degrees) between the same suffix token at layers m - 1 and m
6. Effect = steered - baseline per text (paired), at each layer m > l; nan at m <= l, where the suffix cannot have
   changed (the largest |change| there is printed as a sanity check, expected ~0).
7. Projected-out variant. The same angles, computed after removing the steering direction r_l (of the steered layer
   l and approach) from the suffix states of both passes at every measurement layer (and the embeddings, for the
   cross-layer angle at layer 1):  x_t <- x_t - (r_l . x_t) r_l.  The effect is again steered - baseline, both
   projected with the same r_l, so it shows whether the change in the suffix angles lies outside the direction
   itself (i.e. is not just the suffix moving along r_l).
Controls (--control picks one per run, "random" or "data_shuffle"; only that one is steered and saved):
8. Random control (--control random). The same steering with N_RANDOM (30) random unit directions r = g / ||g||, g ~ N(0, I_d) (uniform on
   the sphere), drawn once with a fixed seed and reused at every layer and for every text: same layers, prefix
   positions, scale c * u_l, texts and batches, and the same projected-out variant (projecting out the random
   direction itself). The report compares the real effect with the distribution over random directions.
9. Label-shuffle control (--control data_shuffle). N_SHUFFLE (20) permutations of the human/machine labels of the train texts (text level,
   class sizes kept; every token gets its text's shuffled label). For each permutation, the token-level mean
   difference on the same train split, per layer, normalized to unit length, gives one direction per layer; it is
   steered, measured and projected out exactly like the real direction.
Hypothesis: pushing machine text towards human (c < 0) raises the suffix angles; pushing human text towards machine
(c > 0) lowers them.

Layer convention: layer l = 1..L is the output of decoder layer l (hidden_states[l] in HF), recorded with forward
hooks, so layer L is taken before the final norm. Texts are used in full; texts with fewer than MIN_TOKENS tokens are
left out. Everything goes to one new file output/steering/prefix_<model>_<dataset>_s<seed>_<timestamp>.npz, with
the metadata as a JSON string under "meta"; nothing is overwritten.
"""

import json
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.utils import load_data

cf = Config()

OUTPUT_DIR = os.path.join(cf.base_dir, "output", "steering")
TRAIN_FRACTION = 0.7
# at least 3 suffix tokens, so that there are adjacent suffix pairs
MIN_TOKENS = 8
APPROACHES = ("token",)
METRICS = ("local_angle", "cross_angle")
# variants of every metric: as is, and with the steering direction projected out of the suffix states
VARIANTS = ("standard", "projected")
# c in units of the mean token norm u_l; > 0 towards machine, < 0 towards human. All run in one job.
STRENGTHS = (-0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4)
# random control: N_RANDOM unit directions r = g / ||g||, g ~ N(0, I_d), drawn once with seed + RANDOM_SEED_OFFSET
N_RANDOM = 30
RANDOM_SEED_OFFSET = 1000
# label-shuffle control: N_SHUFFLE permutations of the train text labels (class sizes kept), seed + SHUFFLE_SEED_OFFSET
N_SHUFFLE = 20
SHUFFLE_SEED_OFFSET = 2000


class PrefixHooks:
    """Forward hooks on every decoder layer. Each batch row may add one vector to the prefix positions 1..p at one
    layer; at every layer the suffix angles of every row are recorded (the suffix is never changed by the hook)."""

    def __init__(self, decoder: torch.nn.Module) -> None:
        self.p: int | None = None
        # (add (R, d) float, layer (R,) long; 0 = no steering, r (R, d) unit direction to project out, 0 = none)
        # for the rows of the current batch
        self.plan: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.collect = False
        # normalized suffix states of the previous layer: as is, projected (steered rows), projected baseline row
        self.prev: dict[str, torch.Tensor] = {}
        self.metrics: dict[str, list[torch.Tensor]] = {}
        self.stats: list[tuple[torch.Tensor, torch.Tensor]] = []
        decoder.embed_tokens.register_forward_hook(self.embedding_hook)
        for i, layer in enumerate(decoder.layers):
            layer.register_forward_hook(self.hook(i + 1))

    def reset(self, p: int | None = None, plan=None, collect: bool = False) -> None:
        self.p, self.plan, self.collect, self.prev = p, plan, collect, {}
        self.metrics = {}
        self.stats = []

    def suffix_states(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Normalized suffix states (R, S, d): "std" as is; "proj" with each row's r projected out; "base" the
        baseline row 0 with each row's r projected out (the matching baseline of every row)."""
        x = h[:, self.p + 1:].float()
        out = {"std": torch.nn.functional.normalize(x, dim=-1)}
        if self.plan is not None:
            r = self.plan[2][:, None]                                                    # (R, 1, d)
            out["proj"] = torch.nn.functional.normalize(x - (x * r).sum(dim=-1, keepdim=True) * r, dim=-1)
            out["base"] = torch.nn.functional.normalize(x[:1] - (x[:1] * r).sum(dim=-1, keepdim=True) * r, dim=-1)
        return out

    def record(self, name: str, z: torch.Tensor) -> None:
        adjacent = (z[:, 1:] * z[:, :-1]).sum(dim=-1).clamp(-1, 1)
        cross = (self.prev[name] * z).sum(dim=-1).clamp(-1, 1)
        suffix = {"std": "", "proj": "_proj", "base": "_proj_base"}[name]
        self.metrics.setdefault(f"local_angle{suffix}", []).append(torch.rad2deg(torch.arccos(adjacent)).mean(dim=-1))
        self.metrics.setdefault(f"cross_angle{suffix}", []).append(torch.rad2deg(torch.arccos(cross)).mean(dim=-1))

    def embedding_hook(self, module, inputs, output) -> None:
        if self.p is not None:
            self.prev = self.suffix_states(output)

    def hook(self, layer: int):
        def forward(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.collect:
                # train statistics over positions 1..T-1 of the single row: token sum (d,), sum of token norms
                x = h[0, 1:].float()
                self.stats.append((x.sum(dim=0), x.norm(dim=-1).sum()))
            if self.p is None:
                return output
            if self.plan is not None:
                add, at, _ = self.plan
                on = at == layer
                if bool(on.any()):
                    x = h[:, 1:self.p + 1].float() + (add * on[:, None].float())[:, None]
                    h = torch.cat([h[:, :1], x.to(h.dtype), h[:, self.p + 1:]], dim=1)
            states = self.suffix_states(h)
            for name, z in states.items():
                self.record(name, z)
            self.prev = states
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return forward


class PrefixSteering:
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
        self.hooks = PrefixHooks(self.decoder)

    def tokenize(self, text: str) -> torch.Tensor | None:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return None if ids.shape[1] < MIN_TOKENS else ids.to(self.device)

    def split_point(self, T: int) -> int:
        return max(1, int(self.args.split_frac * (T - 1)))

    # ------------------------------------------------------------------ split
    def split(self) -> tuple[list[dict], list[dict], dict]:
        data = load_data(args=self.args)
        items = [item for split in data.values() for item in split]
        for item in items:
            item["label"] = int(item["label"])
        idx = np.arange(len(items))
        if all("pair_id" in item for item in items):
            # paired data: both texts of a pair on the same side
            pair_ids = np.asarray([str(item["pair_id"]) for item in items])
            groups = np.unique(pair_ids)
            tr_groups, _ = train_test_split(groups, test_size=round(1 - TRAIN_FRACTION, 6),
                                            random_state=self.args.seed)
            in_train = np.isin(pair_ids, tr_groups)
            tr, te = idx[in_train], idx[~in_train]
            how = f"by pair ({len(groups)} pairs)"
        else:
            tr, te = train_test_split(idx, test_size=round(1 - TRAIN_FRACTION, 6), random_state=self.args.seed,
                                      stratify=[item["label"] for item in items])
            how = "stratified by label"
        return [items[i] for i in tr], [items[i] for i in te], {"train_index": tr, "test_index": te}

    # ------------------------------------------------------------------ directions
    @torch.no_grad()
    def directions(self, train: list[dict]) -> dict:
        P = self.P
        token_sum, doc_means, doc_n, norm_sum, n_tokens, labels = [], [], [], np.zeros(P), 0, []
        token_sums = {0: 0.0, 1: 0.0}
        token_counts = {0: 0, 1: 0}
        for item in tqdm(train, desc="Train states"):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            self.hooks.reset(collect=True)
            self.decoder(input_ids=ids, use_cache=False)
            sums = torch.stack([s for s, _ in self.hooks.stats[:P]]).double().cpu().numpy()       # (P, d)
            norms = torch.stack([n for _, n in self.hooks.stats[:P]]).double().cpu().numpy()      # (P,)
            self.hooks.reset()
            n = ids.shape[1] - 1
            y = item["label"]
            token_sums[y] = token_sums[y] + sums
            token_counts[y] += n
            doc_means.append(sums / n)
            doc_n.append(n)
            norm_sum += norms
            n_tokens += n
            labels.append(y)
        doc_means, labels = np.stack(doc_means), np.asarray(labels)                              # (N, P, d)
        mu = {
            "token": (token_sums[0] / token_counts[0], token_sums[1] / token_counts[1]),
        }
        R, gap, mu_h, mu_m, auroc = [], [], [], [], []
        for name in APPROACHES:
            h, m = mu[name]
            d = m - h
            g = np.linalg.norm(d, axis=1)
            r = d / g[:, None]
            score = (doc_means * r[None]).sum(axis=-1)                                           # (N, P)
            R.append(r), gap.append(g), mu_h.append(h), mu_m.append(m)
            auroc.append([roc_auc_score(labels, score[:, i]) for i in range(P)])
        R, auroc = np.stack(R), np.asarray(auroc)
        u = norm_sum / n_tokens
        # random control directions: uniform on the unit sphere, fixed once (seeded) and reused at every layer and text
        n_rand = N_RANDOM if self.args.control == "random" else 0
        g = np.random.default_rng(self.args.seed + RANDOM_SEED_OFFSET).standard_normal((n_rand, R.shape[-1]))
        rand = g / np.linalg.norm(g, axis=1, keepdims=True)
        # label-shuffle control: permute the text labels (class sizes kept), every token takes its text's label, then
        # the same token-level mean difference per layer
        doc_n = np.asarray(doc_n, dtype=np.float64)
        doc_sums = doc_means * doc_n[:, None, None]                                              # (N, P, d)
        rng = np.random.default_rng(self.args.seed + SHUFFLE_SEED_OFFSET)
        shuf, perms = [np.zeros((0, P, R.shape[-1]))], [np.zeros((0, len(labels)), dtype=labels.dtype)]
        for _ in range(N_SHUFFLE if self.args.control == "data_shuffle" else 0):
            yp = rng.permutation(labels)
            mh = doc_sums[yp == 0].sum(axis=0) / doc_n[yp == 0].sum()
            mm = doc_sums[yp == 1].sum(axis=0) / doc_n[yp == 1].sum()
            d = mm - mh
            shuf.append((d / np.linalg.norm(d, axis=1, keepdims=True))[None])
            perms.append(yp[None])
        shuf = np.concatenate(shuf)                                                              # (NS, P, d)
        return {"r": R, "u": u, "rand": rand, "shuf": shuf, "shuf_labels": np.concatenate(perms), "gap_norm": np.stack(gap), "mu_h": np.stack(mu_h), "mu_m": np.stack(mu_m),
                "train_auroc": auroc,
                "n_train": {"human": int((labels == 0).sum()), "machine": int((labels == 1).sum())},
                "n_train_tokens": {"human": token_counts[0], "machine": token_counts[1]}}

    # ------------------------------------------------------------------ steering
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, p: int, plan, collect: bool = False) -> dict[str, np.ndarray]:
        """(R, T) ids -> suffix metrics {metric: (R, L)}; with collect, also "mean": the mean state of row 0 (the
        baseline row) over positions 1..T-1 at layers 1..L, (L, d)."""
        self.hooks.reset(p=p, plan=plan, collect=collect)
        self.decoder(input_ids=ids, use_cache=False)
        out = {m: torch.stack(v, dim=1).cpu().numpy() for m, v in self.hooks.metrics.items()}
        if collect:
            out["mean"] = (torch.stack([s for s, _ in self.hooks.stats]).double().cpu().numpy()
                           / (ids.shape[1] - 1))
        self.hooks.reset()
        return out

    @torch.no_grad()
    def steer(self, test: list[dict], dirs: dict) -> dict:
        """Steer every test text along the real direction(s) and the random control directions. Directions are kept
        in one bank (K = A real + N_RANDOM random, (K, P, d)); results are split into real and random at the end."""
        A, S, P, L = len(APPROACHES), len(self.strengths), self.P, self.n_layers
        NR, NS = dirs["rand"].shape[0], dirs["shuf"].shape[0]
        K = A + NR + NS
        bank = np.concatenate([dirs["r"], np.repeat(dirs["rand"][:, None], P, axis=1),
                               dirs["shuf"]])                                           # (K, P, d), unit
        vec = (self.strengths[None, :, None, None] * dirs["u"][None, None, :, None]
               * bank[:, None])                                                         # (K, S, P, d)
        vec = torch.tensor(vec, device=self.device, dtype=torch.float32)
        rdir = torch.tensor(bank, device=self.device, dtype=torch.float32)
        conditions = [(k, j, l) for k in range(K) for j in range(S) for l in range(1, P + 1)]
        # one baseline row in every batch, so that each effect is taken within its own batch
        B = max(1, self.args.batch_size - 1)
        baseline, delta, delta_proj, baseline_proj = [], [], [], []
        kept, p_all, T_all, noise = [], [], [], 0.0
        scores = []  # (A, P) per text: the unsteered text mean along r_l, for the test AUROC
        for i, item in enumerate(tqdm(test, desc=f"Steering ({len(conditions)} conditions per text)")):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            T = ids.shape[1]
            p = self.split_point(T)
            doc = np.full((len(METRICS), K, S, P, L), np.nan, dtype=np.float32)
            dproj = np.full((len(METRICS), K, S, P, L), np.nan, dtype=np.float32)
            bproj = np.full((len(METRICS), K, P, L), np.nan, dtype=np.float32)
            base = None
            for s in range(0, len(conditions), B):
                conds = conditions[s:s + B]
                add = torch.zeros((len(conds) + 1, vec.shape[-1]), device=self.device)
                at = torch.zeros(len(conds) + 1, dtype=torch.long, device=self.device)
                proj = torch.zeros_like(add)  # row 0 (baseline): nothing projected out
                for k, (b, j, l) in enumerate(conds, start=1):
                    add[k], at[k], proj[k] = vec[b, j, l - 1], l, rdir[b, l - 1]
                res = self.forward(ids.expand(len(conds) + 1, -1), p, (add, at, proj), collect=base is None)
                if base is None:
                    base = np.stack([res[m][0] for m in METRICS])                           # (M, L)
                    scores.append((res["mean"][None, :P] * dirs["r"]).sum(axis=-1))
                for mi, m in enumerate(METRICS):
                    for k, (b, j, l) in enumerate(conds, start=1):
                        # effect against the baseline row of the same batch
                        doc[mi, b, j, l - 1] = res[m][k] - res[m][0]
                        noise = max(noise, float(np.abs(res[m][k, :l] - res[m][0, :l]).max()))
                        # projected: row k and the baseline row, both with the direction of row k projected out
                        st, bs = res[f"{m}_proj"][k], res[f"{m}_proj_base"][k]
                        dproj[mi, b, j, l - 1] = st - bs
                        bproj[mi, b, l - 1] = bs
                        noise = max(noise, float(np.abs(st[:l] - bs[:l]).max()))
            baseline.append(base)
            delta.append(doc)
            delta_proj.append(dproj)
            baseline_proj.append(bproj)
            kept.append(i), p_all.append(p), T_all.append(T)
        # the suffix can only change at layers m > l
        after = np.arange(1, L + 1)[None, :] > np.arange(1, P + 1)[:, None]             # (P, L)
        baseline = np.stack(baseline)                                                   # (N, M, L)
        delta = np.where(after, np.stack(delta), np.nan).astype(np.float32)             # (N, M, K, S, P, L)
        delta_proj = np.where(after, np.stack(delta_proj), np.nan).astype(np.float32)   # (N, M, K, S, P, L)
        baseline_proj = np.stack(baseline_proj)                                         # (N, M, K, P, L)
        labels = np.asarray([test[i]["label"] for i in kept])
        scores = np.stack(scores)                                                       # (N, A, P)
        test_auroc = np.asarray([[roc_auc_score(labels, scores[:, a, l]) for l in range(P)]
                                 for a in range(A)])
        real, rand, shuf = slice(0, A), slice(A, A + NR), slice(A + NR, K)
        return {"baseline": baseline, "labels": labels,
                "kept": np.asarray(kept), "split_point": np.asarray(p_all), "n_tokens": np.asarray(T_all),
                "test_score": scores, "test_auroc": test_auroc, "sanity_max_change_before_l": noise,
                # real direction
                "delta": delta[:, :, real],
                "steered": delta[:, :, real] + baseline[:, :, None, None, None],
                "delta_proj": delta_proj[:, :, real], "baseline_proj": baseline_proj[:, :, real],
                "steered_proj": delta_proj[:, :, real] + baseline_proj[:, :, real, None],
                # random control directions
                "delta_rand": delta[:, :, rand], "delta_rand_proj": delta_proj[:, :, rand],
                "baseline_rand_proj": baseline_proj[:, :, rand],
                # label-shuffle control directions
                "delta_shuf": delta[:, :, shuf], "delta_shuf_proj": delta_proj[:, :, shuf],
                "baseline_shuf_proj": baseline_proj[:, :, shuf]}

    # ------------------------------------------------------------------ report
    def report(self, res: dict) -> None:
        for variant, key in (("standard", "delta"), ("direction projected out", "delta_proj")):
            ctrl = "delta_rand" if self.args.control == "random" else "delta_shuf"
            self.report_delta(res["labels"], res[key], res[key.replace("delta", ctrl)], variant)

    def report_delta(self, y: np.ndarray, delta: np.ndarray, rand: np.ndarray, variant: str) -> None:
        """Per class, strength and steered layer l: the mean effect over layers l+1..l+4 of the real direction, the
        random directions (mean and 2.5-97.5% range of the per-direction class means) and the share of random
        directions with an effect at least as large in absolute value (p_rand, with +1 correction)."""
        layers = [l for l in (2, 5, 8, 9, 10, 12, 16, 20) if l <= self.P]
        NR = rand.shape[2]
        for mi, m in enumerate(METRICS):
            print(f"\n{m} [{variant}]: mean effect over layers l+1..l+4 (degrees); real | random mean "
                  f"[2.5%, 97.5%] over {NR} directions | p_rand")
            for label, group in ((1, "machine"), (0, "human")):
                print(f"  {group}")
                for s, c in enumerate(self.strengths):
                    row = []
                    for l in layers:
                        win = slice(l, l + 4)
                        real = float(np.nanmean(delta[y == label, mi, 0, s, l - 1, win]))
                        r = np.nanmean(rand[y == label, mi, :, s, l - 1, win], axis=(0, 2))       # (NR,)
                        p = (1 + int((np.abs(r) >= abs(real)).sum())) / (1 + NR)
                        lo, hi = np.percentile(r, [2.5, 97.5])
                        row.append(f"L{l} {real:+.3f} | {r.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.2f}")
                    print(f"    c={c:+.2f}  " + " ;  ".join(row))

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        args = self.args
        train, test, split = self.split()
        dirs = self.directions(train)
        res = self.steer(test, dirs)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"prefix_{args.control}_{args.model}_{args.dataset}_s{args.seed}_"
                                        f"{time.strftime('%Y%m%d-%H%M%S')}.npz")
        if os.path.exists(path):
            raise FileExistsError(path)
        y = res["labels"]
        test_kept = [test[i] for i in res["kept"]]
        meta = {
            "model": args.model, "model_id": self.inference.model_id, "dtype": str(self.dtype),
            "dataset": args.dataset, "seed": args.seed, "train_fraction": TRAIN_FRACTION,
            "split_frac": args.split_frac, "min_tokens": MIN_TOKENS, "max_layer": self.P,
            "n_layers": self.n_layers, "strengths": self.strengths.tolist(), "approaches": list(APPROACHES),
            "metrics": list(METRICS), "batch_size": args.batch_size,
            "n_train": dirs["n_train"], "n_train_tokens": dirs["n_train_tokens"],
            "n_test": {"human": int((y == 0).sum()), "machine": int((y == 1).sum())},
            "sanity_max_change_before_l": res["sanity_max_change_before_l"],
            "direction": {"token": "r_l = mu_M - mu_H normalized, class means over all train tokens 1..T-1"},
            "u": "mean train token norm at layer l (tokens 1..T-1, both classes)",
            "intervention": "h_t <- h_t + c * u_l * r_l at prefix positions t = 1..p, at layer l only "
                            "(c > 0 towards machine)",
            "split_point": "p = max(1, floor(split_frac * (T - 1))); prefix 1..p, suffix p+1..T-1",
            "local_angle": "mean angle (deg) between adjacent suffix tokens (t, t+1), t >= p+1",
            "cross_angle": "mean angle (deg) between suffix token t at layers m-1 and m (layer 0 = embeddings)",
            "effect": "delta = steered - baseline per text; nan at measurement layers m <= l",
            "projected": "*_proj: the same angles after projecting r_l (steered layer l, same approach) out of the "
                         "suffix states of both passes at every measurement layer and the embeddings: "
                         "x <- x - (r_l . x) r_l; delta_proj = steered_proj - baseline_proj (same batch)",
            "control": args.control,
            "random": {"n": N_RANDOM, "seed": self.args.seed + RANDOM_SEED_OFFSET,
                       "sampling": "r = g / ||g||, g ~ N(0, I_d) (numpy default_rng); the same directions at every "
                                   "layer and text; scale c * u_l as for the real direction",
                       "projected": "the random direction itself is projected out"},
            "shuffle": {"n": N_SHUFFLE, "seed": self.args.seed + SHUFFLE_SEED_OFFSET,
                        "sampling": "text labels of the train split permuted (class sizes kept), every token takes "
                                    "its text's label; token-level mean difference per layer, unit length; scale "
                                    "c * u_l as for the real direction",
                        "projected": "the shuffled direction of the steered layer is projected out"},
            "layer_convention": "layer l = output of decoder layer l (1..L), before the final norm",
            "shapes": {
                "baseline": "(N, metric, L)",
                "steered": "(N, metric, approach, strength, steer layer 1..P, measurement layer 1..L)",
                "delta": "(N, metric, approach, strength, steer layer 1..P, measurement layer 1..L)",
                "r": "(approach, P, d)", "u": "(P,)", "train_auroc": "(approach, P)",
                "test_auroc": "(approach, P)", "test_score": "(N, approach, P)",
                "baseline_proj": "(N, metric, approach, steer layer 1..P, measurement layer 1..L)",
                "steered_proj": "(N, metric, approach, strength, steer layer 1..P, measurement layer 1..L)",
                "delta_proj": "(N, metric, approach, strength, steer layer 1..P, measurement layer 1..L)",
                "rand": "(n_random, d)",
                "delta_rand": "(N, metric, n_random, strength, steer layer 1..P, measurement layer 1..L)",
                "delta_rand_proj": "(N, metric, n_random, strength, steer layer 1..P, measurement layer 1..L)",
                "baseline_rand_proj": "(N, metric, n_random, steer layer 1..P, measurement layer 1..L)",
                "shuf": "(n_shuffle, P, d)", "shuf_labels": "(n_shuffle, N_train) permuted labels",
                "delta_shuf": "(N, metric, n_shuffle, strength, steer layer 1..P, measurement layer 1..L)",
                "delta_shuf_proj": "(N, metric, n_shuffle, strength, steer layer 1..P, measurement layer 1..L)",
                "baseline_shuf_proj": "(N, metric, n_shuffle, steer layer 1..P, measurement layer 1..L)",
            },
        }
        # only the control that was run
        if args.control == "random":
            control = dict(rand=dirs["rand"].astype(np.float32), delta_rand=res["delta_rand"],
                           delta_rand_proj=res["delta_rand_proj"], baseline_rand_proj=res["baseline_rand_proj"])
            meta.pop("shuffle")
            meta["shapes"] = {k: v for k, v in meta["shapes"].items() if "shuf" not in k}
        else:
            control = dict(shuf=dirs["shuf"].astype(np.float32), shuf_labels=dirs["shuf_labels"],
                           delta_shuf=res["delta_shuf"], delta_shuf_proj=res["delta_shuf_proj"],
                           baseline_shuf_proj=res["baseline_shuf_proj"])
            meta.pop("random")
            meta["shapes"] = {k: v for k, v in meta["shapes"].items() if "rand" not in k}
        extra = {}
        if all("pair_id" in item for item in test_kept):
            extra["pair_id"] = np.asarray([str(item["pair_id"]) for item in test_kept])
        np.savez_compressed(
            path, meta=np.asarray(json.dumps(meta)),
            baseline=res["baseline"], steered=res["steered"], delta=res["delta"], labels=y,
            split_point=res["split_point"], n_tokens=res["n_tokens"],
            test_index=split["test_index"][res["kept"]], train_index=split["train_index"],
            metrics=np.asarray(METRICS), approaches=np.asarray(APPROACHES), strengths=self.strengths,
            steer_layers=np.arange(1, self.P + 1), layers=np.arange(1, self.n_layers + 1),
            r=dirs["r"].astype(np.float32), u=dirs["u"], gap_norm=dirs["gap_norm"],
            mu_h=dirs["mu_h"].astype(np.float32), mu_m=dirs["mu_m"].astype(np.float32),
            train_auroc=dirs["train_auroc"],
            test_auroc=res["test_auroc"], test_score=res["test_score"],
            baseline_proj=res["baseline_proj"], steered_proj=res["steered_proj"], delta_proj=res["delta_proj"],
            **control,
            **extra,
        )
        print(f"\nSaved {path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_frac", type=float, default=0.5, help="prefix share of positions 1..T-1")
    parser.add_argument("--control", choices=("random", "data_shuffle"), required=True,
                        help="random: N_RANDOM random unit directions; data_shuffle: N_SHUFFLE label-shuffled "
                             "mean-difference directions")
    parser.add_argument("--max_layer", type=int, default=20, help="steer at layers 1..max_layer")
    parser.add_argument("--batch_size", type=int, default=81, help="rows per forward pass (incl. one baseline)")
    args = parser.parse_args()
    args.model_name = args.model
    return args


def main() -> None:
    PrefixSteering(parse_args()).run()


if __name__ == "__main__":
    main()
