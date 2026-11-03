"""Ablating the machine-human direction in early to mid layers.

Instead of steering along the direction (test.py), the direction is removed: in human text the machine
direction is ablated, in machine text the human direction is ablated.

1. Split. All splits of the dataset are pooled and split stratified, TRAIN_FRACTION (70%) to find the
   directions, the rest (30%) to ablate.
2. Direction (mean differencing). For every train document, the mean state over positions 1..T-1 (position 0,
   the attention sink, excluded) at layer l = 1..P, P = min(MAX_LAYER, L) = 20. With mu_H, mu_M the class means of
   these document means:
       d_l = mu_M - mu_H,   v_l = d_l / ||d_l||   (machine direction +v_l, human direction -v_l)
       c_l = v_l . (mu_H + mu_M) / 2              (the class midpoint along v_l)
   The signed score s_l(h) = v_l . h - c_l is > 0 on the machine side and < 0 on the human side.
3. Ablation. +v and -v span the same line, so removing the machine direction from human text and the human
   direction from machine text is the same operation, applied to each class:
       h_t <- h_t - (v_l . h_t - c_l) v_l          (--center midpoint, default; "mean ablation")
       h_t <- h_t - (v_l . h_t) v_l                (--center zero; plain projection out)
   on every position 1..T-1. After it, s_l = 0 (midpoint) for every token: no human or machine side is left
   along v_l. Conditions, all run in one batch with the clean pass:
       clean       no ablation
       dir_L<j>    ablate v_j at layer j only, j = 1..P, then let layers j+1..L run
       dir_all     ablate v_l at every layer l = 1..P
       rand_L<j>, rand_all   the same with a random unit direction per layer (numpy default_rng(seed + l)),
                   centred the same way: a control for removing any one direction
4. Measured at every layer m = 1..L (positions 1..T-1):
       proj        mean_t s_m(h_t): where the text sits along the direction (m <= P; nan above). Shows whether
                   the direction comes back in the layers after the ablation
       global_cos  mean off-diagonal cosine between all token pairs
       local_angle mean angle (degrees) between adjacent tokens t, t+1
       cross_angle mean angle (degrees) between the same token at layers m - 1 and m (layer 0 = embeddings)
   and once per text:
       loglik      mean log p(x_{t+1} | x_{<=t}) of the model's own next-token prediction
   AUROC (label 1 = machine) per condition and layer, oriented by the clean sign at that layer.

Layer convention: layer l = 1..L is the output of decoder layer l (hidden_states[l] in HF), recorded with forward
hooks, so layer L is taken before the final norm. Every document is used in full; documents with fewer than
MIN_TOKENS tokens are left out.
All files go to a new folder output/test/ablate_<model>_<dataset>_s<seed>_<timestamp>; nothing is overwritten.
"""

import json
import os
import time
from argparse import ArgumentParser, Namespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.utils import load_data

cf = Config()

OUTPUT_DIR = os.path.join(cf.base_dir, "output", "test")
MIN_TOKENS = 3
TRAIN_FRACTION = 0.7
# directions are found and ablated at layers 1..MAX_LAYER
MAX_LAYER = 20
GEOMETRY = ("global_cos", "local_angle", "cross_angle")
METRICS = ("proj",) + GEOMETRY
GROUPS = ((0, "human", "#2a78d6"), (1, "machine", "#eb6834"))
LABELS = {"proj": "score along v (+ = machine)", "global_cos": "mean pairwise cosine",
          "local_angle": "adjacent-token angle (°)", "cross_angle": "cross-layer angle (°)",
          "loglik": "mean token log-likelihood"}
# single-layer ablations drawn in the line plots
SHOW_LAYERS = (2, 5, 8, 12, 16, 20)


class Ablation:
    """Forward hooks on every decoder layer: ablate one direction per batch row at the planned layers, and record
    every output; plus a hook on the embedding that keeps layer 0 for the cross-layer angle at layer 1."""

    def __init__(self, decoder: torch.nn.Module) -> None:
        # (V (R, P, d), c (R, P), on (R, P) bool) for the rows of the current batch
        self.plan: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.states: list[torch.Tensor] = []
        self.embedding: torch.Tensor | None = None
        decoder.embed_tokens.register_forward_hook(self.record_embedding)
        for i, layer in enumerate(decoder.layers):
            layer.register_forward_hook(self.hook(i + 1))

    def record_embedding(self, module, inputs, output) -> None:
        self.embedding = output

    def hook(self, layer: int):
        def forward(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.plan is not None and layer <= self.plan[0].shape[1]:
                V, c, on = (x[:, layer - 1] for x in self.plan)
                if bool(on.any()):
                    x = h[:, 1:].float()
                    coef = (x * V[:, None]).sum(dim=-1) - c[:, None]          # (R, T-1)
                    x = x - (coef * on[:, None].float())[..., None] * V[:, None]
                    h = torch.cat([h[:, :1], x.to(h.dtype)], dim=1)
            self.states.append(h)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return forward


def geometry(H: torch.Tensor, E: torch.Tensor) -> dict[str, torch.Tensor]:
    """(L, T, d) states of layers 1..L and (T, d) embeddings -> global_cos, local_angle, cross_angle, each (L,),
    over positions 1..T-1 (same as test.py)."""
    H, E = H[:, 1:], E[1:]
    Z = torch.nn.functional.normalize(H.float(), dim=-1)
    T = Z.shape[1]
    K = Z @ Z.transpose(1, 2)
    trace = K.diagonal(dim1=1, dim2=2).sum(dim=-1)
    adjacent = (Z[:, 1:] * Z[:, :-1]).sum(dim=-1).clamp(-1, 1)
    previous = torch.cat([torch.nn.functional.normalize(E.float(), dim=-1)[None], Z[:-1]])
    cross = (previous * Z).sum(dim=-1).clamp(-1, 1)
    return {
        "global_cos": (K.sum(dim=(1, 2)) - trace) / (T * (T - 1)),
        "local_angle": torch.rad2deg(torch.arccos(adjacent)).mean(dim=-1),
        "cross_angle": torch.rad2deg(torch.arccos(cross)).mean(dim=-1),
    }


class DirectionAblation:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model, args=args)
        self.model = self.inference.model
        self.tokenizer = self.inference.tokenizer
        self.device = self.inference.device
        self.dtype = next(self.model.parameters()).dtype
        self.decoder = self.model.get_decoder()
        self.lm_head = self.model.get_output_embeddings()
        self.n_layers = self.model.config.num_hidden_layers
        self.layers = np.arange(1, self.n_layers + 1)
        self.P = min(MAX_LAYER, self.n_layers)
        self.hooks = Ablation(self.decoder)
        # conditions: (name, kind, ablated layers)
        P = self.P
        self.conditions = ([("clean", "clean", ())]
                           + [(f"dir_L{j}", "dir", (j,)) for j in range(1, P + 1)]
                           + [("dir_all", "dir", tuple(range(1, P + 1)))]
                           + [(f"rand_L{j}", "rand", (j,)) for j in range(1, P + 1)]
                           + [("rand_all", "rand", tuple(range(1, P + 1)))])
        self.names = [c[0] for c in self.conditions]

    def idx(self, name: str) -> int:
        return self.names.index(name)

    def tokenize(self, text: str) -> torch.Tensor | None:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return None if ids.shape[1] < MIN_TOKENS else ids.to(self.device)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, plan=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(R, T) ids -> states of layers 1..L (R, L, T, d), embeddings (R, T, d), final normed states (R, T, d)."""
        self.hooks.plan, self.hooks.states = plan, []
        out = self.decoder(input_ids=ids, use_cache=False)
        states, embedding = torch.stack(self.hooks.states, dim=1), self.hooks.embedding
        self.hooks.plan, self.hooks.states, self.hooks.embedding = None, [], None
        return states, embedding, out.last_hidden_state

    # ------------------------------------------------------------------ directions
    def split(self) -> tuple[list[dict], list[dict], dict]:
        data = load_data(args=self.args)
        items = [item for split in data.values() for item in split]
        labels = [item["label"] for item in items]
        tr, te = train_test_split(np.arange(len(items)), test_size=round(1 - TRAIN_FRACTION, 6),
                                  random_state=self.args.seed, stratify=labels)
        print(f"Pooled splits {list(data)}: {len(items)} documents -> {len(tr)} for directions, {len(te)} ablated")
        return [items[i] for i in tr], [items[i] for i in te], {"train_index": tr, "test_index": te}

    def directions(self, train: list[dict]) -> dict:
        """Mean-difference direction per layer 1..P from the per-document mean states."""
        means, labels = [], []
        for item in tqdm(train, desc="Train states"):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            H, _, _ = self.forward(ids)
            means.append(H[0, :self.P, 1:].float().mean(dim=1).cpu().numpy())
            labels.append(item["label"])
        means, labels = np.stack(means).astype(np.float64), np.asarray(labels)  # (N, P, d)
        mu_h, mu_m = means[labels == 0].mean(axis=0), means[labels == 1].mean(axis=0)
        d = mu_m - mu_h
        gap = np.linalg.norm(d, axis=1)
        V = d / gap[:, None]
        c = (V * (mu_h + mu_m) / 2).sum(axis=1)
        if self.args.center == "zero":
            c = np.zeros_like(c)
        score = (means * V[None]).sum(axis=-1) - c                                  # (N, P)
        train_auroc = np.asarray([roc_auc_score(labels, score[:, i]) for i in range(self.P)])
        rng_dirs, rng_c = [], []
        for l in range(1, self.P + 1):
            r = np.random.default_rng(self.args.seed + l).standard_normal(V.shape[1])
            r /= np.linalg.norm(r)
            rng_dirs.append(r)
            rng_c.append(0.0 if self.args.center == "zero" else float(r @ (mu_h[l - 1] + mu_m[l - 1]) / 2))
        print("Mean-difference direction, train AUROC of s_l by layer 1..P: "
              + " ".join(f"{a:.3f}" for a in train_auroc))
        return {"V": V, "c": c, "mu_h": mu_h, "mu_m": mu_m, "gap_norm": gap, "train_auroc": train_auroc,
                "R": np.stack(rng_dirs), "c_rand": np.asarray(rng_c),
                "n_train": {"human": int((labels == 0).sum()), "machine": int((labels == 1).sum())}}

    def build_plan(self, dirs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C, P = len(self.conditions), self.P
        d = dirs["V"].shape[1]
        V = np.zeros((C, P, d), dtype=np.float32)
        c = np.zeros((C, P), dtype=np.float32)
        on = np.zeros((C, P), dtype=bool)
        for i, (_, kind, layers) in enumerate(self.conditions):
            if kind == "clean":
                continue
            V[i] = dirs["V"] if kind == "dir" else dirs["R"]
            c[i] = dirs["c"] if kind == "dir" else dirs["c_rand"]
            for l in layers:
                on[i, l - 1] = True
        to = lambda x: torch.tensor(x, device=self.device)
        return to(V), to(c), to(on)

    # ------------------------------------------------------------------ ablation
    @torch.no_grad()
    def ablate_test(self, test: list[dict], dirs: dict) -> dict:
        plan = self.build_plan(dirs)
        Vd = torch.tensor(dirs["V"], device=self.device, dtype=torch.float32)       # (P, d)
        cd = torch.tensor(dirs["c"], device=self.device, dtype=torch.float32)
        C, B = len(self.conditions), self.args.batch_size
        values, loglik, labels, check = {m: [] for m in METRICS}, [], [], {"below": 0.0, "at": 0.0}
        for item in tqdm(test, desc="Ablating"):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            doc, ll = {m: [] for m in METRICS}, []
            for s in range(0, C, B):
                rows = slice(s, min(s + B, C))
                R = rows.stop - rows.start
                H, E, last = self.forward(ids.expand(R, -1), tuple(x[rows] for x in plan))
                # score along the (true) direction at layers 1..P, nan above
                proj = torch.full((R, self.n_layers), float("nan"), device=self.device)
                proj[:, :self.P] = (H[:, :self.P, 1:].float().mean(dim=2) * Vd).sum(dim=-1) - cd
                doc["proj"].append(proj.cpu().numpy())
                for r in range(R):
                    g = geometry(H[r], E[r])
                    for m in GEOMETRY:
                        doc[m].append(g[m].cpu().numpy()[None])
                    logp = torch.log_softmax(self.lm_head(last[r, :-1]).float(), dim=-1)
                    ll.append(float(logp.gather(-1, ids[0, 1:, None]).mean()))
                del H, E, last
            doc = {m: np.concatenate(v) for m, v in doc.items()}  # (C, L)
            # sanity: below the ablated layer nothing changes; at it, s = 0 (midpoint) along v
            for j in range(2, self.P + 1):
                i = self.idx(f"dir_L{j}")
                check["below"] = max(check["below"],
                                     float(np.abs(doc["global_cos"][i, :j - 1] - doc["global_cos"][0, :j - 1]).max()))
                check["at"] = max(check["at"], float(abs(doc["proj"][i, j - 1])))
            for m in METRICS:
                values[m].append(doc[m])
            loglik.append(ll)
            labels.append(item["label"])
        print(f"Sanity: max |change| of global_cos below the ablated layer = {check['below']:.2e} (expected ~0); "
              f"max |score along v| left at the ablated layer = {check['at']:.2e} (expected ~0)")
        return {"values": {m: np.stack(v) for m, v in values.items()},  # (N, C, L)
                "loglik": np.asarray(loglik), "labels": np.asarray(labels)}  # (N, C), (N,)

    # ------------------------------------------------------------------ summary
    @staticmethod
    def auroc(y: np.ndarray, x: np.ndarray) -> float:
        ok = ~np.isnan(x)
        return float("nan") if ok.sum() < 2 or len(np.unique(y[ok])) < 2 else float(roc_auc_score(y[ok], x[ok]))

    @staticmethod
    def stats(delta: np.ndarray) -> dict:
        try:
            p = float(wilcoxon(delta).pvalue)
        except ValueError:
            p = float("nan")
        return {"mean": float(delta.mean()), "se": float(delta.std(ddof=1) / np.sqrt(len(delta))), "wilcoxon_p": p}

    def summarize(self, res: dict) -> dict:
        y, C, L = res["labels"], len(self.conditions), self.n_layers
        out = {"groups": {}, "auroc": {}, "gap_closed": {}, "loglik": {}}
        series = dict(res["values"], loglik=res["loglik"][..., None])  # loglik as a 1-layer metric
        for name, v in series.items():
            delta = v - v[:, :1]
            out["groups"][name] = {
                group: {"mean": v[y == label].mean(axis=0).tolist(),
                        "se": (v[y == label].std(axis=0, ddof=1) / np.sqrt((y == label).sum())).tolist(),
                        "delta_mean": delta[y == label].mean(axis=0).tolist()}
                for label, group, _ in GROUPS}
            raw = np.asarray([[self.auroc(y, v[:, i, m]) for m in range(v.shape[2])] for i in range(C)])
            flip = raw[0] < 0.5  # orient every condition by the clean sign at that layer
            out["auroc"][name] = np.where(flip[None], 1 - raw, raw).tolist()
            # share of the clean machine - human gap removed: 1 - gap_ablated / gap_clean
            gap = v[y == 1].mean(axis=0) - v[y == 0].mean(axis=0)  # (C, layers)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["gap_closed"][name] = (1 - gap / gap[:1]).tolist()
        for label, group, _ in GROUPS:
            d = res["loglik"][y == label] - res["loglik"][y == label][:, :1]
            out["loglik"][group] = {n: self.stats(d[:, i]) for i, n in enumerate(self.names) if n != "clean"}
        return out

    def report(self, summary: dict, dirs: dict) -> None:
        print(f"\n{'=' * 100}\nAUROC (oriented by clean) at each layer: clean | dir_all | rand_all\n{'=' * 100}")
        rows = [self.idx(n) for n in ("clean", "dir_all", "rand_all")]
        for name in METRICS:
            a = np.asarray(summary["auroc"][name])
            print(f"\n{name}\n{'layer':>6} {'clean':>8} {'dir_all':>8} {'rand_all':>9}")
            for m in self.layers:
                print(f"{m:>6} " + " ".join(f"{a[i, m - 1]:>8.3f}" for i in rows))
        a = np.asarray(summary["auroc"]["loglik"])[:, 0]
        print(f"\nloglik AUROC: clean {a[0]:.3f} | dir_all {a[rows[1]]:.3f} | rand_all {a[rows[2]]:.3f}")
        print(f"\n{'=' * 100}\nShare of the clean machine-human gap removed by dir_all (rand_all), per layer\n{'=' * 100}")
        print(f"{'layer':>6} " + " ".join(f"{m:>22}" for m in METRICS))
        for m in self.layers:
            cells = []
            for name in METRICS:
                g = np.asarray(summary["gap_closed"][name])
                cells.append(f"{g[rows[1], m - 1]:>+10.2f} ({g[rows[2], m - 1]:>+8.2f})")
            print(f"{m:>6} " + " ".join(f"{c:>22}" for c in cells))
        print(f"\n{'=' * 100}\nChange in mean token log-likelihood (ablated - clean)\n{'=' * 100}")
        for group in ("human", "machine"):
            for n in ("dir_all", "rand_all", *(f"dir_L{j}" for j in SHOW_LAYERS if j <= self.P)):
                s = summary["loglik"][group][n]
                print(f"{group:>8} {n:>9}: {s['mean']:+.4f} ± {s['se']:.4f}  (Wilcoxon p = {s['wilcoxon_p']:.2e})")

    # ------------------------------------------------------------------ plots
    @staticmethod
    def save(fig: plt.Figure, out_dir: str, name: str) -> None:
        path = os.path.join(out_dir, f"{name}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    def plot(self, res: dict, summary: dict, out_dir: str) -> None:
        y, x, P = res["labels"], self.layers, self.P
        show = [j for j in SHOW_LAYERS if j <= P]
        cmap = plt.get_cmap("viridis")
        shades = {j: cmap(k / max(1, len(show) - 1)) for k, j in enumerate(show)}

        # 1. score along v per class: does the direction come back after a single-layer ablation?
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        v = res["values"]["proj"]
        for ax, (label, group, color) in zip(axes, GROUPS):
            sel = v[y == label]
            other = v[y != label][:, 0].mean(axis=0)
            ax.plot(x, sel[:, 0].mean(axis=0), color="black", lw=2, label="clean")
            ax.plot(x, other, color="black", lw=1, ls=":", label="clean, other class")
            for j in show:
                ax.plot(x, sel[:, self.idx(f"dir_L{j}")].mean(axis=0), color=shades[j], lw=1.3, label=f"ablate L{j}")
            ax.plot(x, sel[:, self.idx("dir_all")].mean(axis=0), color=color, lw=2.5, label=f"ablate L1-{P}")
            ax.plot(x, sel[:, self.idx("rand_all")].mean(axis=0), color="grey", lw=1.5, ls="--",
                    label=f"random L1-{P}")
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlim(1, P)
            ax.set_title(f"{group} text")
            ax.set_xlabel("measurement layer")
        axes[0].set_ylabel(LABELS["proj"])
        axes[1].legend(fontsize=8, ncol=2)
        fig.suptitle("Score along the mean-difference direction")
        self.save(fig, out_dir, "projection")

        # 2. geometry by layer: clean vs ablated, both classes
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
        for ax, name in zip(axes, GEOMETRY):
            v = res["values"][name]
            for label, group, color in GROUPS:
                sel = v[y == label]
                ax.plot(x, sel[:, 0].mean(axis=0), color=color, lw=2, label=f"{group} clean")
                ax.plot(x, sel[:, self.idx("dir_all")].mean(axis=0), color=color, lw=2, ls="--",
                        label=f"{group} ablate L1-{P}")
                ax.plot(x, sel[:, self.idx("rand_all")].mean(axis=0), color=color, lw=1, ls=":", alpha=0.8,
                        label=f"{group} random L1-{P}")
            ax.axvline(P, color="grey", lw=0.6)
            ax.set_xlabel("measurement layer")
            ax.set_ylabel(LABELS[name])
        axes[0].legend(fontsize=8)
        fig.suptitle("Geometry by layer, clean vs. ablated")
        self.save(fig, out_dir, "geometry_layers")

        # 3. heatmaps: ablation layer j x measurement layer m, change per class and share of the gap removed
        fig, axes = plt.subplots(len(METRICS), 3, figsize=(16, 3.4 * len(METRICS)), squeeze=False)
        rows = [self.idx(f"dir_L{j}") for j in range(1, P + 1)]
        for r, name in enumerate(METRICS):
            v = res["values"][name][..., :P]
            for c, (label, group, _) in enumerate(GROUPS):
                d = (v[y == label][:, rows] - v[y == label][:, :1]).mean(axis=0)  # (P, P)
                lim = np.nanmax(np.abs(d)) or 1.0
                im = axes[r, c].imshow(d, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto",
                                       extent=(0.5, P + 0.5, 0.5, P + 0.5))
                fig.colorbar(im, ax=axes[r, c])
                axes[r, c].set_title(f"{name}: Δ {group}")
            g = np.asarray(summary["gap_closed"][name])[rows][:, :P]
            im = axes[r, 2].imshow(g, origin="lower", cmap="PuOr", vmin=-1, vmax=1, aspect="auto",
                                   extent=(0.5, P + 0.5, 0.5, P + 0.5))
            fig.colorbar(im, ax=axes[r, 2])
            axes[r, 2].set_title(f"{name}: share of gap removed")
            for ax in axes[r]:
                ax.set_ylabel("ablated layer")
                ax.plot([0.5, P + 0.5], [0.5, P + 0.5], color="black", lw=0.5)
        for ax in axes[-1]:
            ax.set_xlabel("measurement layer")
        fig.suptitle(f"Single-layer ablation (layers 1-{P}); below the diagonal is unchanged by construction")
        fig.tight_layout()
        self.save(fig, out_dir, "heatmaps")

        # 4. AUROC by layer
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 4.2), sharey=True)
        for ax, name in zip(axes, METRICS):
            a = np.asarray(summary["auroc"][name])
            ax.plot(x, a[0], color="black", lw=2, label="clean")
            for j in show:
                ax.plot(x, a[self.idx(f"dir_L{j}")], color=shades[j], lw=1.2, label=f"ablate L{j}")
            ax.plot(x, a[self.idx("dir_all")], color="#eb6834", lw=2.5, label=f"ablate L1-{P}")
            ax.plot(x, a[self.idx("rand_all")], color="grey", lw=1.5, ls="--", label=f"random L1-{P}")
            ax.axhline(0.5, color="grey", lw=0.6)
            ax.axvline(P, color="grey", lw=0.6)
            ax.set_title(name)
            ax.set_xlabel("measurement layer")
        axes[0].set_ylabel("AUROC (oriented by clean)")
        axes[-1].legend(fontsize=8)
        self.save(fig, out_dir, "auroc")

        # 5. log-likelihood: change per ablated layer, and its AUROC
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        j = np.arange(1, P + 1)
        ll = res["loglik"]
        for label, group, color in GROUPS:
            d = ll[y == label] - ll[y == label][:, :1]
            for kind, ls in (("dir", "-"), ("rand", "--")):
                cols = [self.idx(f"{kind}_L{k}") for k in j]
                m, se = d[:, cols].mean(axis=0), d[:, cols].std(axis=0, ddof=1) / np.sqrt(len(d))
                axes[0].plot(j, m, color=color, ls=ls, label=f"{group}, {'direction' if kind == 'dir' else 'random'}")
                axes[0].fill_between(j, m - se, m + se, color=color, alpha=0.15)
        axes[0].axhline(0, color="grey", lw=0.6)
        axes[0].set_xlabel("ablated layer")
        axes[0].set_ylabel("Δ " + LABELS["loglik"])
        axes[0].legend(fontsize=8)
        a = np.asarray(summary["auroc"]["loglik"])[:, 0]
        axes[1].plot(j, a[[self.idx(f"dir_L{k}") for k in j]], color="#eb6834", label="direction")
        axes[1].plot(j, a[[self.idx(f"rand_L{k}") for k in j]], color="grey", ls="--", label="random")
        axes[1].axhline(a[0], color="black", lw=1.5, label="clean")
        axes[1].axhline(a[self.idx("dir_all")], color="#eb6834", lw=1, ls=":", label=f"ablate L1-{P}")
        axes[1].set_xlabel("ablated layer")
        axes[1].set_ylabel("log-likelihood AUROC (oriented by clean)")
        axes[1].legend(fontsize=8)
        self.save(fig, out_dir, "loglik")

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        args = self.args
        train, test, split = self.split()
        dirs = self.directions(train)
        out_dir = os.path.join(OUTPUT_DIR, f"ablate_{args.model}_{args.dataset}_s{args.seed}_"
                                           f"{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(out_dir, exist_ok=False)
        res = self.ablate_test(test, dirs)
        summary = self.summarize(res)
        self.report(summary, dirs)
        self.plot(res, summary, out_dir)

        path = os.path.join(out_dir, "per_document.npz")
        np.savez_compressed(path, labels=res["labels"], conditions=np.asarray(self.names), layers=self.layers,
                            loglik=res["loglik"], **{m: v for m, v in res["values"].items()}, **split)
        print(f"Saved {path}  (metrics: documents x condition x layer 1..L; loglik: documents x condition)")
        path = os.path.join(out_dir, "directions.npz")
        np.savez_compressed(path, **{k: dirs[k] for k in ("V", "c", "mu_h", "mu_m", "gap_norm", "train_auroc",
                                                          "R", "c_rand")})
        print(f"Saved {path}")
        path = os.path.join(out_dir, "summary.json")
        y = res["labels"]
        with open(path, "w") as f:
            json.dump({
                "config": {
                    "model": args.model, "model_id": self.inference.model_id, "dtype": str(self.dtype),
                    "dataset": args.dataset, "seed": args.seed, "train_fraction": TRAIN_FRACTION,
                    "ablation_layers": list(range(1, self.P + 1)), "center": args.center,
                    "ablation": "h <- h - (v.h - c) v on positions 1..T-1; c = class midpoint along v (midpoint) "
                                "or 0 (zero)",
                    "direction": "v_l = (mu_M - mu_H) / ||.||, class means of per-document mean states (positions "
                                 "1..T-1) on the train part",
                    "random": "isotropic Gaussian unit vector per layer, numpy default_rng(seed + layer)",
                    "layer_convention": "layer l = output of decoder layer l (1..L), before the final norm",
                    "conditions": self.names, "n_train": dirs["n_train"],
                    "n_test": {"human": int((y == 0).sum()), "machine": int((y == 1).sum())},
                },
                "directions": {"train_auroc": dirs["train_auroc"].tolist(), "gap_norm": dirs["gap_norm"].tolist(),
                               "c": dirs["c"].tolist()},
                "auroc_note": "conditions x layers, oriented by the clean sign at each layer",
                "gap_closed_note": "1 - (machine - human)_ablated / (machine - human)_clean, conditions x layers",
                **summary,
            }, f, indent=1)
        print(f"Saved {path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center", choices=("midpoint", "zero"), default="midpoint",
                        help="midpoint: set every token to the class midpoint along v; zero: plain projection out")
    parser.add_argument("--batch_size", type=int, default=22, help="conditions per forward pass")
    args = parser.parse_args()
    args.model_name = args.model
    return args


def main() -> None:
    DirectionAblation(parse_args()).run()


if __name__ == "__main__":
    main()
