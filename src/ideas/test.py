"""Does the machine-human direction control representation geometry?

Two experiments, chosen with --experiment:
  steer        (SteeringGeometry)      steering along the probe direction, described below
  specificity  (DirectionSpecificity)  the probe direction against random directions of the same size; see that
                                       class

Layer convention: layer l = 1..L is the output of decoder layer l (hidden_states[l] in HF), recorded with forward
hooks, so layer L is taken before the final norm. Every document is cut to its first N_TOKENS tokens (shorter
documents are left out), so all metrics compare equal-length sequences.

1. Direction. On the train split (or, if the dataset has none, a stratified half of test, with the other half
   steered), the last-token state of every document at every layer l is fed to a linear probe
   (StandardScaler + LogisticRegression, label 1 = machine). The probe's normal in raw hidden-state space is
   w_l = coef / scale, the score is s_l(h) = w_l . h + b_l (positive = machine), and v_l = w_l / ||w_l||.
2. Intervention. At layer l every token gets h_t <- h_t + alpha * n_l * v_l, with n_l the mean train token norm at
   layer l (position 0, the attention sink, excluded), so alpha is relative to a typical hidden-state norm. The
   forward pass then continues through layers l+1..L. alpha > 0 pushes towards machine, alpha < 0 towards human.
3. Layers. The target layer is the one where the clean human/machine angular separation on the train split is
   strongest (max |AUROC - 0.5| of global_cos or local_angle over layers 2..L), unless --target_layer is given.
   Intervention layers default to every layer 2..target - 1, plus the target itself as a reference; effects are
   always read at the target layer. Only small strengths are used (|alpha| <= 0.4).
4. Geometry per document, layer and alpha (test split):
     global_cos:  mean off-diagonal cosine between all token pairs
     local_angle: mean angle (degrees) between adjacent tokens t, t+1
     cross_angle: mean angle (degrees) between the same token at layers m - 1 and m (layer 0 = embeddings); when
                  steering at the target m itself, only the h^m end of this angle is moved
     probe:       s_m(last token) with the probe trained at the measurement layer m
   alpha = 0 is run in the same batch as the steered rows, so differences are not batch noise.
Hypothesis checked at the target layer: human + alpha (H->M) raises global_cos and lowers local_angle and
cross_angle; machine - alpha (M->H) does the opposite. Each change D(alpha) is split, on all documents, into
  directional = (D(+a) - D(-a)) / 2, what flips with the sign of the push (the effect of the direction itself), and
  shared      = (D(+a) + D(-a)) / 2, what does not (generic collapse from adding any vector to every token).
The probe score is reported separately, since moving along v changes it by construction.
Results go to a new folder under output/test; nothing is overwritten.
"""

import inspect
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.utils import load_data

cf = Config()

OUTPUT_DIR = os.path.join(cf.base_dir, "output", "test")
N_TOKENS = 256
# small strengths only: |alpha| >= 0.5 collapsed the geometry whatever the sign
ALPHAS = (-0.4, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.4)
MAGNITUDES = tuple(a for a in ALPHAS if a > 0)
PROBE_C = 1.0
GEOMETRY = ("global_cos", "local_angle", "cross_angle")
# the target layer is chosen on the within-layer metrics only, as before
TARGET_METRICS = ("global_cos", "local_angle")
SCORES = GEOMETRY + ("probe",)
# expected sign of the change under H->M (human docs, alpha > 0); M->H is the opposite
EXPECTED = {"global_cos": 1, "local_angle": -1, "cross_angle": -1, "probe": 1}
GROUPS = ((0, "human", "#2a78d6"), (1, "machine", "#eb6834"))
MAGNITUDE_COLORS = ("#c6dbf5", "#9ec5f0", "#5c9ee6", "#2a78d6", "#17498a")
LABELS = {"global_cos": "mean pairwise cosine", "local_angle": "adjacent-token angle (°)",
          "cross_angle": "cross-layer angle (°)", "probe": "probe score (+ = machine)"}


class Steering:
    """
    Forward hooks on every decoder layer: add a shift to the output of one layer, and record every output; plus a
    hook on the embedding that keeps layer 0 for the cross-layer angle at layer 1.
    """

    def __init__(self, decoder: torch.nn.Module) -> None:
        self.shift: tuple[int, torch.Tensor] | None = None  # (layer l, (B, 1, d))
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
            if self.shift is not None and self.shift[0] == layer:
                h = h + self.shift[1]
            self.states.append(h)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return forward


def geometry(H: torch.Tensor, E: torch.Tensor) -> dict[str, torch.Tensor]:
    """(L, T, d) states of layers 1..L and (T, d) embeddings -> global_cos, local_angle, cross_angle, each (L,)."""
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


class SteeringGeometry:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model, args=args)
        self.model = self.inference.model
        self.tokenizer = self.inference.tokenizer
        self.device = self.inference.device
        self.dtype = next(self.model.parameters()).dtype
        self.decoder = self.model.get_decoder()
        self.n_layers = self.model.config.num_hidden_layers
        self.layers = np.arange(1, self.n_layers + 1)
        self.steering = Steering(self.decoder)
        self.alphas = torch.tensor(ALPHAS, device=self.device, dtype=torch.float32)

    def tokenize(self, text: str) -> torch.Tensor | None:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if ids.shape[1] < N_TOKENS:
            return None
        return ids[:, :N_TOKENS].to(self.device)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor,
                shift: tuple[int, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T) ids -> states of layers 1..L, (B, L, T, d), with the optional shift applied, and embeddings (B, T, d)."""
        self.steering.shift, self.steering.states = shift, []
        self.decoder(input_ids=ids, use_cache=False)
        states, embedding = torch.stack(self.steering.states, dim=1), self.steering.embedding
        self.steering.shift, self.steering.states, self.steering.embedding = None, [], None
        return states, embedding

    def collect_train(self, data: list[dict]) -> dict:
        """Last-token states (N, L, d), clean geometry (N, L), token norms (L,) and labels of the train split."""
        last, values, norms, labels = [], {name: [] for name in GEOMETRY}, [], []
        for item in tqdm(data, desc="Train states"):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            H, E = self.forward(ids)
            H = H[0]
            last.append(H[:, -1].float().cpu().numpy())
            for name, value in geometry(H, E[0]).items():
                values[name].append(value.cpu().numpy())
            norms.append(H[:, 1:].float().norm(dim=-1).mean(dim=-1).cpu().numpy())
            labels.append(item["label"])
        return {"last": np.stack(last), "labels": np.asarray(labels), "norms": np.mean(norms, axis=0),
                "values": {name: np.stack(v) for name, v in values.items()}}

    def fit_probes(self, last: np.ndarray, labels: np.ndarray) -> dict:
        """Per layer, the probe in raw space: w (L, d), b (L,), and its train AUROC."""
        W, b, train_auroc = [], [], []
        for i in tqdm(range(self.n_layers), desc="Probes"):
            X = last[:, i]
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=PROBE_C, max_iter=5000))
            probe.fit(X, labels)
            scaler, clf = probe[0], probe[1]
            assert list(clf.classes_) == [0, 1], "label 1 must be machine"
            w = clf.coef_[0] / scaler.scale_
            W.append(w)
            b.append(clf.intercept_[0] - (w * scaler.mean_).sum())
            train_auroc.append(roc_auc_score(labels, X @ w + b[-1]))
        return {"W": np.stack(W), "b": np.asarray(b), "train_auroc": np.asarray(train_auroc)}

    def select_layers(self, train: dict) -> tuple[int, list[int], dict]:
        """Target layer (strongest clean angular separation on train) and the intervention layers."""
        separation = {name: np.array([roc_auc_score(train["labels"], train["values"][name][:, i])
                                      for i in range(self.n_layers)]) for name in GEOMETRY}
        strength = np.max([np.abs(separation[name] - 0.5) for name in TARGET_METRICS], axis=0)
        target = self.args.target_layer or int(self.layers[1:][np.argmax(strength[1:])])
        if self.args.intervene_layers:
            intervene = sorted(set(self.args.intervene_layers))
        else:
            intervene = list(range(2, target + 1))
        dropped = [l for l in intervene if not 1 <= l <= target]
        if dropped:
            print(f"Dropping intervention layers {dropped}: geometry is measured at the target layer {target}")
        intervene = [l for l in intervene if 1 <= l <= target]
        return target, intervene, {name: v.tolist() for name, v in separation.items()}

    def steer_test(self, data: list[dict], intervene: list[int], probes: dict, scale: np.ndarray) -> dict:
        """Metrics (N, J, A, L) for every test document, intervention layer j, alpha a and measurement layer."""
        V = torch.tensor(probes["W"] / np.linalg.norm(probes["W"], axis=1, keepdims=True),
                         device=self.device, dtype=torch.float32)
        W = torch.tensor(probes["W"], device=self.device, dtype=torch.float32)
        b = torch.tensor(probes["b"], device=self.device, dtype=torch.float32)
        values, labels, below_diff = {name: [] for name in SCORES}, [], 0.0
        for item in tqdm(data, desc="Steering"):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            ids = ids.expand(len(ALPHAS), -1)
            doc = {name: [] for name in SCORES}
            for l in intervene:
                shift = (self.alphas * float(scale[l - 1]))[:, None, None] * V[l - 1]
                H, E = self.forward(ids, (l, shift.to(self.dtype)))  # (A, L, T, d), (A, T, d)
                rows = [geometry(H[a], E[a]) for a in range(len(ALPHAS))]
                for name in GEOMETRY:
                    doc[name].append(torch.stack([r[name] for r in rows]).cpu().numpy())
                doc["probe"].append(((H[:, :, -1].float() * W).sum(dim=-1) + b).cpu().numpy())
                # sanity: layers below l must not depend on alpha
                if l > 1:
                    g = doc["global_cos"][-1][:, :l - 1]
                    below_diff = max(below_diff, float(np.abs(g - g[ALPHAS.index(0.0)]).max()))
            for name in SCORES:
                values[name].append(np.stack(doc[name]))
            labels.append(item["label"])
        print(f"Sanity: max |change| of global_cos below the intervention layer = {below_diff:.2e} (expected 0)")
        return {"values": {name: np.stack(v) for name, v in values.items()}, "labels": np.asarray(labels)}

    @staticmethod
    def stats(delta: np.ndarray, expected: int | None = None) -> dict:
        """Mean, SE, Wilcoxon p and, given an expected sign, the fraction of documents that move that way."""
        try:
            p = float(wilcoxon(delta).pvalue)
        except ValueError:
            p = float("nan")
        out = {"mean": float(delta.mean()), "se": float(delta.std(ddof=1) / np.sqrt(len(delta))), "wilcoxon_p": p}
        if expected is not None:
            out["expected_sign"] = expected
            out["frac_expected"] = float(np.mean(np.sign(delta) == expected))
        return out

    def summarize(self, values: dict, labels: np.ndarray, intervene: list[int], target: int) -> dict:
        zero = ALPHAS.index(0.0)
        m = target - 1
        groups = {}
        for label, group, _ in GROUPS:
            groups[group] = {}
            for name in SCORES:
                v = values[name][labels == label]  # (n, J, A, L)
                delta = v - v[:, :, zero:zero + 1]
                n = len(v)
                groups[group][name] = {
                    "mean": v.mean(axis=0).tolist(), "se": (v.std(axis=0, ddof=1) / np.sqrt(n)).tolist(),
                    "delta_mean": delta.mean(axis=0).tolist(),
                    "delta_se": (delta.std(axis=0, ddof=1) / np.sqrt(n)).tolist(),
                }

        # clean machine - human difference at the target, the size a full "class switch" would have
        gap = {name: float(values[name][labels == 1, 0, zero, m].mean() - values[name][labels == 0, 0, zero, m].mean())
               for name in SCORES}
        geometry_rows, probe_rows = [], []
        for j, l in enumerate(intervene):
            for magnitude in MAGNITUDES:
                plus, minus = ALPHAS.index(magnitude), ALPHAS.index(-magnitude)
                row = {"intervene_layer": l, "measure_layer": target, "alpha": magnitude}
                for name in GEOMETRY:
                    v = values[name][:, j, :, m]
                    delta = v - v[:, zero:zero + 1]
                    expected = EXPECTED[name]
                    directional = (delta[:, plus] - delta[:, minus]) / 2
                    row[name] = {
                        "H->M": self.stats(delta[labels == 0, plus], expected),
                        "M->H": self.stats(delta[labels == 1, minus], -expected),
                        "directional": self.stats(directional, expected)
                                       | {"frac_of_gap": float(directional.mean() / gap[name])},
                        "shared": self.stats((delta[:, plus] + delta[:, minus]) / 2),
                    }
                geometry_rows.append(row)
            v = values["probe"][:, j, :, m]
            for a, alpha in enumerate(ALPHAS):
                if alpha != 0:
                    delta = v[:, a] - v[:, zero]
                    probe_rows.append({"intervene_layer": l, "measure_layer": target, "alpha": alpha} | {
                        group: self.stats(delta[labels == label], int(np.sign(alpha))) for label, group, _ in GROUPS})

        # human/machine AUROC of the unsteered metrics (alpha = 0) at each layer
        clean_auroc = {name: [roc_auc_score(labels, values[name][:, 0, zero, i]) for i in range(self.n_layers)]
                       for name in SCORES}
        return {"groups": groups, "clean_gap_at_target": gap, "geometry": geometry_rows, "probe_change": probe_rows,
                "clean_test_auroc": clean_auroc}

    def report(self, summary: dict, probes: dict, test_probe_auroc: list, target: int, intervene: list[int],
               train: dict, test_labels: np.ndarray) -> None:
        args = self.args
        print(f"\nSteering geometry | model={args.model} dataset={args.dataset} seed={args.seed} | "
              f"first {N_TOKENS} tokens | train human={(train['labels'] == 0).sum()} "
              f"machine={(train['labels'] == 1).sum()} | test human={(test_labels == 0).sum()} "
              f"machine={(test_labels == 1).sum()}")
        print(f"Target layer {target} | intervention layers {intervene} | alphas {ALPHAS} x mean token norm")

        print("\n=== Probes and clean geometry per layer ===")
        print(f"{'layer':>6} {'norm':>9} {'probe tr':>9} {'probe te':>9} {'AUC cos':>8} {'AUC ang':>8} "
              f"{'AUC cross':>9}")
        for i, layer in enumerate(self.layers):
            print(f"{layer:>6} {train['norms'][i]:>9.2f} {probes['train_auroc'][i]:>9.4f} "
                  f"{test_probe_auroc[i]:>9.4f} {summary['clean_test_auroc']['global_cos'][i]:>8.4f} "
                  f"{summary['clean_test_auroc']['local_angle'][i]:>8.4f} "
                  f"{summary['clean_test_auroc']['cross_angle'][i]:>9.4f}")

        print(f"\n=== Geometry change at target layer {target}: mean (fraction of docs in the expected direction) ===")
        print("H->M: human docs, +alpha; M->H: machine docs, -alpha. directional = (D(+a) - D(-a)) / 2 on all docs, "
              "with % of the clean machine - human gap and Wilcoxon p; shared = (D(+a) + D(-a)) / 2, the part that "
              "does not depend on the sign (generic collapse)")
        for name in GEOMETRY:
            print(f"\n--- {name} | expected: H->M {EXPECTED[name]:+d}, M->H {-EXPECTED[name]:+d} | "
                  f"clean machine - human gap = {summary['clean_gap_at_target'][name]:+.4f} ---")
            print(f"{'int':>4} {'|a|':>6} {'H->M':>17} {'M->H':>17} {'directional':>17} {'% gap':>7} "
                  f"{'p':>8} {'shared':>10}")
            for row in summary["geometry"]:
                r = row[name]
                hm, mh, d = r["H->M"], r["M->H"], r["directional"]
                print(f"{row['intervene_layer']:>4} {row['alpha']:>6.3f} "
                      f"{hm['mean']:>+10.4f} ({hm['frac_expected']:.2f}) {mh['mean']:>+10.4f} ({mh['frac_expected']:.2f}) "
                      f"{d['mean']:>+10.4f} ({d['frac_expected']:.2f}) {100 * d['frac_of_gap']:>+6.1f}% "
                      f"{d['wilcoxon_p']:>8.1e} {r['shared']['mean']:>+10.4f}")

        print(f"\n=== Probe score change at target layer {target} (reported separately: moving along v changes it "
              f"by construction) | clean gap = {summary['clean_gap_at_target']['probe']:+.3f} ===")
        print(f"{'int':>4} {'alpha':>7} {'human':>17} {'machine':>17}")
        for row in summary["probe_change"]:
            h, mc = row["human"], row["machine"]
            print(f"{row['intervene_layer']:>4} {row['alpha']:>+7.3f} {h['mean']:>+10.3f} ({h['frac_expected']:.2f}) "
                  f"{mc['mean']:>+10.3f} ({mc['frac_expected']:.2f})")

    def plot(self, values: dict, labels: np.ndarray, intervene: list[int], target: int, out_dir: str) -> None:
        args = self.args
        title = f"{args.model} | {args.dataset} | seed {args.seed} | target layer {target}"
        alphas = np.asarray(ALPHAS)
        zero = ALPHAS.index(0.0)
        m = target - 1
        J = len(intervene)

        def band(ax: plt.Axes, x: np.ndarray, rows: np.ndarray, color: str, label: str, style: str = "-",
                 marker: str | None = None) -> None:
            mean = rows.mean(axis=0)
            se = rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
            ax.plot(x, mean, style, color=color, linewidth=2, marker=marker, markersize=4, label=label)
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, linewidth=0)

        def delta(name: str, j: int) -> np.ndarray:
            """(N, A) change at the target layer relative to alpha = 0."""
            v = values[name][:, j, :, m]
            return v - v[:, zero:zero + 1]

        def column_title(l: int) -> str:
            return f"steer at {l}" + (" (reference)" if l == target else "")

        # 1. intervention strength vs change in geometry at the target layer
        fig, axes = plt.subplots(len(GEOMETRY), J, figsize=(3.2 * J, 3.2 * len(GEOMETRY)), sharex=True, sharey="row",
                                 squeeze=False)
        for j, l in enumerate(intervene):
            for row, name in enumerate(GEOMETRY):
                ax = axes[row, j]
                d = delta(name, j)
                for label, group, color in GROUPS:
                    band(ax, alphas, d[labels == label], color, f"{group} (n={(labels == label).sum()})", marker="o")
                ax.axhline(0, color="#888888", linewidth=1)
                ax.axvline(0, color="#888888", linewidth=1)
                ax.grid(alpha=0.3)
                if j == 0:
                    ax.set_ylabel(f"Δ {LABELS[name]}")
            axes[0, j].set_title(column_title(l))
            axes[-1, j].set_xlabel("α (× mean token norm)")
        axes[0, 0].legend(frameon=False)
        fig.suptitle(f"{title} | change at layer {target} vs α = 0 (α > 0: towards machine)")
        self.save(fig, out_dir, "strength.pdf")

        # 2. layer-wise effects: original vs steered (H->M on human, M->H on machine)
        a_plus, a_minus = ALPHAS.index(args.plot_alpha), ALPHAS.index(-args.plot_alpha)
        fig, axes = plt.subplots(len(GEOMETRY), J, figsize=(3.2 * J, 3.2 * len(GEOMETRY)), sharex=True,
                                 squeeze=False)
        for j, l in enumerate(intervene):
            for row, name in enumerate(GEOMETRY):
                ax = axes[row, j]
                for (label, group, color), a, arrow in zip(GROUPS, (a_plus, a_minus), ("H→M", "M→H")):
                    v = values[name][labels == label, j]
                    band(ax, self.layers, v[:, zero], color, f"{group} original")
                    band(ax, self.layers, v[:, a], color, f"{group} {arrow} (α={ALPHAS[a]:+g})", "--")
                ax.axvline(l, color="#888888", linewidth=1)
                ax.axvline(target, color="#888888", linewidth=1, linestyle=":")
                ax.grid(alpha=0.3)
                if j == 0:
                    ax.set_ylabel(LABELS[name])
            axes[0, j].set_title(column_title(l))
            axes[-1, j].set_xlabel("measurement layer")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{title} | solid line: steering layer, dotted: target")
        self.save(fig, out_dir, "layers.pdf")

        # 3. depth comparison at the target layer: both directions, then split into directional and shared parts
        kinds = (("H→M: human docs, +α", lambda d, p, n: d[labels == 0, p]),
                 ("M→H: machine docs, −α", lambda d, p, n: d[labels == 1, n]),
                 ("directional: (Δ(+α) − Δ(−α)) / 2", lambda d, p, n: (d[:, p] - d[:, n]) / 2),
                 ("shared: (Δ(+α) + Δ(−α)) / 2", lambda d, p, n: (d[:, p] + d[:, n]) / 2))
        fig, axes = plt.subplots(len(kinds), len(GEOMETRY), figsize=(5.5 * len(GEOMETRY), 3 * len(kinds)),
                                 sharex=True, squeeze=False)
        for col, name in enumerate(GEOMETRY):
            deltas = [delta(name, j) for j in range(J)]
            for row, (kind, select) in enumerate(kinds):
                ax = axes[row, col]
                for magnitude, color in zip(MAGNITUDES, MAGNITUDE_COLORS):
                    p, n = ALPHAS.index(magnitude), ALPHAS.index(-magnitude)
                    rows = np.stack([select(d, p, n) for d in deltas], axis=1)
                    mean = rows.mean(axis=0)
                    se = rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
                    ax.errorbar(intervene, mean, yerr=se, color=color, linewidth=2, marker="o", markersize=5,
                                capsize=3, label=f"|α|={magnitude:g}")
                ax.axhline(0, color="#888888", linewidth=1)
                if target in intervene:
                    ax.axvline(target, color="#888888", linewidth=1, linestyle=":")
                ax.grid(alpha=0.3)
                ax.set_title(f"{LABELS[name]} | {kind}", fontsize=9)
                if col == 0:
                    ax.set_ylabel("Δ at target")
            axes[-1, col].set_xlabel("steering layer")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{title} | expected: cosine ↑, both angles ↓ for H→M and directional, opposite for M→H; "
                     f"dotted: target (reference)", fontsize=10)
        self.save(fig, out_dir, "depth.pdf")

        # 4. probe score, kept apart from the geometry: change at the target vs alpha, and across layers
        fig, axes = plt.subplots(2, J, figsize=(3.2 * J, 6.4), squeeze=False)
        for j, l in enumerate(intervene):
            d = delta("probe", j)
            for label, group, color in GROUPS:
                band(axes[0, j], alphas, d[labels == label], color, group, marker="o")
                v = values["probe"][labels == label, j]
                a = a_plus if label == 0 else a_minus
                band(axes[1, j], self.layers, v[:, zero], color, f"{group} original")
                band(axes[1, j], self.layers, v[:, a], color, f"{group} steered (α={ALPHAS[a]:+g})", "--")
            axes[0, j].axhline(0, color="#888888", linewidth=1)
            axes[0, j].set(title=column_title(l), xlabel="α (× mean token norm)")
            axes[1, j].axvline(l, color="#888888", linewidth=1)
            axes[1, j].axvline(target, color="#888888", linewidth=1, linestyle=":")
            axes[1, j].set_xlabel("measurement layer")
            for ax in axes[:, j]:
                ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel(f"Δ probe score at layer {target}")
        axes[1, 0].set_ylabel(LABELS["probe"])
        axes[0, 0].legend(frameon=False, fontsize=8)
        axes[1, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{title} | probe of the measurement layer")
        self.save(fig, out_dir, "probe.pdf")

    @staticmethod
    def save(fig: plt.Figure, out_dir: str, name: str) -> None:
        path = os.path.join(out_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    def run(self) -> None:
        args = self.args
        if args.plot_alpha not in ALPHAS or -args.plot_alpha not in ALPHAS or args.plot_alpha <= 0:
            raise ValueError(f"--plot_alpha must be a positive value with ±alpha in {ALPHAS}")
        out_dir = os.path.join(OUTPUT_DIR, f"steer_{args.model}_{args.dataset}_s{args.seed}_"
                                           f"{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(out_dir, exist_ok=False)

        data = load_data(args=args)
        probe_split = "train"
        if "train" not in data:
            # no train split: fit the probes on a stratified half of test, steer the other half
            probe_split = "half of test"
            items = data["test"]
            fit_idx, steer_idx = train_test_split(np.arange(len(items)), test_size=0.5, random_state=args.seed,
                                                  stratify=[item["label"] for item in items])
            data = {"train": [items[i] for i in fit_idx], "test": [items[i] for i in steer_idx]}
            print(f"No train split: probes on {len(fit_idx)} test documents, steering on the other {len(steer_idx)}")
        train = self.collect_train(data["train"])
        probes = self.fit_probes(train["last"], train["labels"])
        target, intervene, separation = self.select_layers(train)
        test = self.steer_test(data["test"], intervene, probes, train["norms"])
        values, labels = test["values"], test["labels"]

        summary = self.summarize(values, labels, intervene, target)
        test_probe_auroc = summary["clean_test_auroc"]["probe"]
        self.report(summary, probes, test_probe_auroc, target, intervene, train, labels)
        self.plot(values, labels, intervene, target, out_dir)

        path = os.path.join(out_dir, "per_document.npz")
        np.savez_compressed(path, labels=labels, alphas=np.asarray(ALPHAS), intervene_layers=np.asarray(intervene),
                            layers=self.layers, **values)
        print(f"Saved {path}  (arrays: documents x intervention layer x alpha x measurement layer)")
        path = os.path.join(out_dir, "probes.npz")
        np.savez_compressed(path, W=probes["W"], b=probes["b"], token_norm=train["norms"], layers=self.layers)
        print(f"Saved {path}")

        path = os.path.join(out_dir, "summary.json")
        with open(path, "w") as f:
            json.dump({
                "config": {
                    "model": args.model, "model_id": self.inference.model_id, "dtype": str(self.dtype),
                    "dataset": args.dataset, "seed": args.seed, "n_tokens": N_TOKENS, "probe_split": probe_split,
                    "layer_convention": "layer l = output of decoder layer l (1..L), before the final norm",
                    "alphas": list(ALPHAS), "alpha_scale": "alpha x mean train token norm at the steering layer "
                                                           "(position 0 excluded)",
                    "token_norm": train["norms"].tolist(),
                    "target_layer": target, "target_from_args": args.target_layer is not None,
                    "intervene_layers": intervene, "measure_layer": target,
                    "reference_layer": target if target in intervene else None,
                    "steer_tokens": "all tokens", "plot_alpha": args.plot_alpha,
                    "metrics": {"global_cos": "mean pairwise cosine within a layer",
                                "local_angle": "mean angle between adjacent tokens within a layer",
                                "cross_angle": "mean angle of each token between layers m - 1 and m (0 = embeddings);"
                                               " steering at m itself moves only the h^m end"},
                    "target_metrics": list(TARGET_METRICS),
                    "n_train": {"human": int((train["labels"] == 0).sum()),
                                "machine": int((train["labels"] == 1).sum())},
                    "n_test": {"human": int((labels == 0).sum()), "machine": int((labels == 1).sum())},
                },
                "probe": {
                    "features": "last-token state per document", "model": "StandardScaler + LogisticRegression",
                    "C": PROBE_C, "orientation": "positive = machine", "direction": "w_raw / ||w_raw||",
                    "train_auroc": probes["train_auroc"].tolist(), "test_auroc_clean": test_probe_auroc,
                },
                "train_clean_separation_auroc": separation,
                "summary_axes": "groups[group][metric][stat] is (intervention layer, alpha, measurement layer)",
                **summary,
            }, f, indent=1)
        print(f"Saved {path}")


# ==================================================================================================
# Experiment 2 (--experiment specificity): probe direction against random directions
# ==================================================================================================
# directions are compared at these |alpha| (each run with + and -)
SPEC_ALPHAS = (0.1,)
# row order within a direction: +a, -a for each a
SPEC_SIGNED = tuple(sign * a for a in SPEC_ALPHAS for sign in (1, -1))
N_RANDOM = 50
KINDS = ("probe", "random")
KIND_COLORS = {"probe": "#1f1f1f", "random": "#2a78d6"}
EFFECTS = ("directional", "H->M", "M->H")
# expected sign of each effect relative to EXPECTED[name] (the H->M sign)
EFFECT_SIGN = {"directional": 1, "H->M": 1, "M->H": -1}
# sanity threshold: relative difference between the re-run tail and the full forward pass
TAIL_TOLERANCE = 0.02


def target_geometry(prev: torch.Tensor, cur: torch.Tensor) -> dict[str, torch.Tensor]:
    """(B, T, d) states at layers m - 1 and m -> global_cos, local_angle at m and cross_angle, each (B,)."""
    Z = torch.nn.functional.normalize(cur.float(), dim=-1)
    T = Z.shape[1]
    K = Z @ Z.transpose(1, 2)
    trace = K.diagonal(dim1=1, dim2=2).sum(dim=-1)
    adjacent = (Z[:, 1:] * Z[:, :-1]).sum(dim=-1).clamp(-1, 1)
    cross = (torch.nn.functional.normalize(prev.float(), dim=-1) * Z).sum(dim=-1).clamp(-1, 1)
    return {
        "global_cos": (K.sum(dim=(1, 2)) - trace) / (T * (T - 1)),
        "local_angle": torch.rad2deg(torch.arccos(adjacent)).mean(dim=-1),
        "cross_angle": torch.rad2deg(torch.arccos(cross)).mean(dim=-1),
    }


def compare(probe: float, others: np.ndarray) -> dict:
    """Rank one value against a distribution: percentile, empirical one-sided p (larger = better), z-score."""
    return {
        "percentile": float(100 * np.mean(others < probe)),
        "p": float((1 + np.sum(others >= probe)) / (1 + len(others))),
        "z": float((probe - others.mean()) / others.std(ddof=1)),
    }


class DirectionSpecificity(SteeringGeometry):
    """
    Direction-specific causal attribution: is the geometry change caused by the probe direction itself?

    Same setup as src/ideas/test.py (layer convention, first N_TOKENS tokens, last-token probes, token-norm scaling,
    target layer m chosen the same way). At each intervention layer l the probe direction v_l is compared with
    N_RANDOM random unit directions r / ||r||, r ~ N(0, I), drawn once per layer with seed args.seed + l and kept
    fixed across documents. Every direction d is applied as h_t <- h_t + alpha * n_l * d at every token of layer l,
    for alpha = +-SPEC_ALPHAS, and read at the target m:
      global_cos   mean pairwise cosine at m
      local_angle  mean adjacent-token angle at m
      cross_angle  mean angle of each token between layers m - 1 and m
      probe        last-token score of the probe trained at m (moves along v by construction; kept for reference)
    All directions, documents and strengths share the same probes, target and n_l.

    Cost: layers 1..l do not depend on the direction, so they run once per document; only layers l+1..m are re-run,
    for all steered rows in chunks of --batch_size. Each chunk carries its own unsteered row, so every change D(alpha)
    is taken within one batch. Attention is causal because no mask is passed (SDPA); this is checked against the full
    forward pass on the first document.

    Per direction k, layer l and |alpha| = a (means over documents):
      directional  (D(+a) - D(-a)) / 2 on all documents
      H->M         D(+a) on human documents;   M->H   D(-a) on machine documents
    Specificity, with e the expected sign of each effect (directional and H->M: cosine +, angles -; M->H opposite):
    the probe's e * effect is ranked against the random directions: percentile,
    empirical one-sided p = (1 + #{random >= probe}) / (1 + N_RANDOM) and z-score; the same on |effect|, i.e. whether
    the probe moves the geometry more than random directions whatever the sign. Results go to a new folder under
    output/test; nothing is overwritten.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        params = inspect.signature(self.decoder.layers[0].forward).parameters
        self.pass_position_ids = "position_ids" in params
        self.pass_position_embeddings = "position_embeddings" in params
        self.kinds = np.array(["probe"] + ["random"] * N_RANDOM)
        self.signed = torch.tensor(SPEC_SIGNED, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def tail(self, x: torch.Tensor, start: int, stop: int) -> list[torch.Tensor]:
        """x (B, T, d), the output of layer `start`, through layers start+1..stop -> [h^start, ..., h^stop]."""
        batch, T = x.shape[:2]
        position_ids = torch.arange(T, device=self.device)[None].expand(batch, -1)
        kwargs = {"attention_mask": None}  # causal under SDPA when no mask is given
        if self.pass_position_ids:
            kwargs["position_ids"] = position_ids
        if self.pass_position_embeddings:
            kwargs["position_embeddings"] = self.decoder.rotary_emb(x, position_ids=position_ids)
        states = [x]
        for layer in self.decoder.layers[start:stop]:
            x = layer(x, **kwargs)
            x = x[0] if isinstance(x, tuple) else x
            states.append(x)
        self.steering.states = []  # the recording hooks fired as well; drop their copies
        return states

    def directions(self, probes: dict, intervene: list[int]) -> dict[int, np.ndarray]:
        """Per layer, (1 + N_RANDOM, d) unit directions: probe, then random (rows in self.kinds order)."""
        out = {}
        for l in intervene:
            v = probes["W"][l - 1] / np.linalg.norm(probes["W"][l - 1])
            g = np.random.default_rng(self.args.seed + l).standard_normal((N_RANDOM, v.shape[0]))
            out[l] = np.concatenate([v[None], g / np.linalg.norm(g, axis=1, keepdims=True)])
        return out

    @torch.no_grad()
    def document(self, ids: torch.Tensor, intervene: list[int], shifts: dict[int, torch.Tensor], target: int,
                 W: torch.Tensor, b: torch.Tensor, check: bool) -> tuple[dict, dict]:
        """Changes (J, K, len(SPEC_SIGNED)) and unsteered values (J,) at the target for one document."""
        H, E = self.forward(ids)
        H, E = H[0], E[0]
        n_dirs, size = len(self.kinds), self.args.batch_size - 1
        delta = {name: np.empty((len(intervene), n_dirs * len(SPEC_SIGNED)), dtype=np.float32) for name in SCORES}
        base = {name: np.empty(len(intervene), dtype=np.float32) for name in SCORES}
        for j, l in enumerate(intervene):
            for start in range(0, len(shifts[l]), size):
                chunk = shifts[l][start:start + size]
                chunk = torch.cat([torch.zeros_like(chunk[:1]), chunk])  # row 0: unsteered
                states = self.tail(H[l - 1][None] + chunk[:, None].to(self.dtype), l, target)
                cur = states[-1]
                if len(states) > 1:
                    prev = states[-2]
                else:  # steering at the target: h^{m-1} is the clean state
                    prev = (H[target - 2] if target > 1 else E)[None].expand_as(cur)
                values = target_geometry(prev, cur)
                values["probe"] = cur[:, -1].float() @ W[target - 1] + b[target - 1]
                if check and start == 0:
                    rel = ((cur[0].float() - H[target - 1].float()).norm() / H[target - 1].float().norm()).item()
                    print(f"Sanity: steer at {l}, unsteered tail vs full pass at layer {target}: "
                          f"relative difference {rel:.2e}")
                    if rel > TAIL_TOLERANCE:
                        raise RuntimeError("re-running the tail does not reproduce the full forward pass; "
                                           "check that the attention is causal without a mask (SDPA)")
                for name in SCORES:
                    v = values[name]
                    delta[name][j, start:start + len(chunk) - 1] = (v[1:] - v[0]).cpu().numpy()
                    if start == 0:
                        base[name][j] = v[0].item()
        return {name: d.reshape(len(intervene), n_dirs, len(SPEC_SIGNED)) for name, d in delta.items()}, base

    def steer(self, data: list[dict], intervene: list[int], dirs: dict, probes: dict, scale: np.ndarray,
              target: int) -> dict:
        W = torch.tensor(probes["W"], device=self.device, dtype=torch.float32)
        b = torch.tensor(probes["b"], device=self.device, dtype=torch.float32)
        # (K * len(SPEC_SIGNED), d) shifts per layer, row = direction k, then signed alpha
        shifts = {l: ((self.signed[None, :, None] * torch.tensor(dirs[l], device=self.device,
                                                                 dtype=torch.float32)[:, None])
                      * float(scale[l - 1])).reshape(-1, dirs[l].shape[1]) for l in intervene}
        deltas, bases, labels = {name: [] for name in SCORES}, {name: [] for name in SCORES}, []
        for item in tqdm(data, desc="Steering"):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            delta, base = self.document(ids, intervene, shifts, target, W, b, check=not labels)
            for name in SCORES:
                deltas[name].append(delta[name])
                bases[name].append(base[name])
            labels.append(item["label"])
        return {"delta": {name: np.stack(v) for name, v in deltas.items()},
                "base": {name: np.stack(v) for name, v in bases.items()}, "labels": np.asarray(labels)}

    def effects(self, delta: np.ndarray, labels: np.ndarray, j: int, a: float) -> dict[str, np.ndarray]:
        """Per-document effects (n, K) of one metric at layer index j and strength a."""
        plus, minus = SPEC_SIGNED.index(a), SPEC_SIGNED.index(-a)
        d = delta[:, j]
        return {"directional": (d[:, :, plus] - d[:, :, minus]) / 2,
                "H->M": d[labels == 0][:, :, plus], "M->H": d[labels == 1][:, :, minus]}

    def analyse(self, delta: dict, labels: np.ndarray, intervene: list[int]) -> list[dict]:
        rows = []
        for name in SCORES:
            for j, l in enumerate(intervene):
                for a in SPEC_ALPHAS:
                    row = {"metric": name, "intervene_layer": l, "alpha": a}
                    signs_ok = {}
                    for effect, per_doc in self.effects(delta[name], labels, j, a).items():
                        sign = EXPECTED[name] * EFFECT_SIGN[effect]
                        mean = per_doc.mean(axis=0)  # (K,)
                        probe = mean[0]
                        entry = {"expected_sign": sign, "probe": float(probe),
                                 "probe_se": float(per_doc[:, 0].std(ddof=1) / np.sqrt(len(per_doc))),
                                 "probe_frac_docs_expected": float(np.mean(np.sign(per_doc[:, 0]) == sign))}
                        for kind in KINDS[1:]:
                            others = mean[self.kinds == kind]
                            entry[kind] = {"mean": float(others.mean()), "sd": float(others.std(ddof=1)),
                                           "q2.5": float(np.quantile(others, 0.025)),
                                           "q97.5": float(np.quantile(others, 0.975)),
                                           "signed": compare(sign * probe, sign * others),
                                           "abs": compare(abs(probe), np.abs(others)),
                                           "frac_expected_sign": float(np.mean(np.sign(others) == sign))}
                        row[effect] = entry
                        signs_ok[effect] = np.sign(mean) == sign  # (K,)
                    both = signs_ok["H->M"] & signs_ok["M->H"]
                    row["both_directions_expected"] = {"probe": bool(both[0])} | {
                        kind: float(both[self.kinds == kind].mean()) for kind in KINDS[1:]}
                    rows.append(row)
        return rows

    def report(self, rows: list[dict], target: int, intervene: list[int], labels: np.ndarray) -> None:
        args = self.args
        print(f"\nDirection specificity | model={args.model} dataset={args.dataset} seed={args.seed} | "
              f"first {N_TOKENS} tokens | test human={(labels == 0).sum()} machine={(labels == 1).sum()}")
        print(f"Target layer {target} | intervention layers {intervene} | ±alpha {SPEC_ALPHAS} x mean token norm | "
              f"{N_RANDOM} random directions per layer")
        print("Each effect: probe value (mean over documents) | random mean [2.5, 97.5] | percentile and one-sided "
              "empirical p of the probe in the expected direction among random directions | |p|: the same on the "
              "size of the effect. directional D = (D(+a) - D(-a)) / 2 on all documents; H->M = D(+a) on human, "
              "M->H = D(-a) on machine documents. both: probe has the expected sign in H->M and M->H (fraction of "
              "random directions that do)")
        for name in SCORES:
            note = " (moves along the probe by construction)" if name == "probe" else ""
            print(f"\n--- {name}: expected sign directional / H->M {EXPECTED[name]:+d}, "
                  f"M->H {-EXPECTED[name]:+d}{note} ---")
            print(f"{'int':>4} {'|a|':>5} {'effect':>12} {'probe':>17} {'random mean [2.5, 97.5]':>32} "
                  f"{'pct':>6} {'p':>6} {'|p|':>6}")
            for row in rows:
                if row["metric"] != name:
                    continue
                for effect in EFFECTS:
                    e, r = row[effect], row[effect]["random"]
                    print(f"{row['intervene_layer']:>4} {row['alpha']:>5.2f} {effect:>12} "
                          f"{e['probe']:>+10.4f}±{e['probe_se']:.4f} "
                          f"{r['mean']:>+10.4f} [{r['q2.5']:>+9.4f}, {r['q97.5']:>+9.4f}] "
                          f"{r['signed']['percentile']:>6.1f} {r['signed']['p']:>6.3f} {r['abs']['p']:>6.3f}")
                both = row["both_directions_expected"]
                print(f"{'':>4} {'':>5} {'both':>12} {'yes' if both['probe'] else 'no':>17} "
                      f"(random: {both['random']:.2f})")

    def plot(self, delta: dict, labels: np.ndarray, intervene: list[int], target: int, out_dir: str) -> None:
        args = self.args
        title = f"{args.model} | {args.dataset} | seed {args.seed} | target layer {target}"
        J = len(intervene)

        def directional(name: str, j: int, a: float) -> np.ndarray:
            """Mean directional effect per direction (K,), and the probe's SE across documents."""
            per_doc = self.effects(delta[name], labels, j, a)["directional"]
            return per_doc.mean(axis=0), per_doc[:, 0].std(ddof=1) / np.sqrt(len(per_doc))

        def ylabel(name: str) -> str:
            return f"D {LABELS[name]} (expected {'+' if EXPECTED[name] > 0 else '−'})"

        # 1. distribution of random directional effects with the probe's effect
        fig, axes = plt.subplots(len(GEOMETRY), J, figsize=(3.4 * J, 3 * len(GEOMETRY)), squeeze=False)
        for j, l in enumerate(intervene):
            for row, name in enumerate(GEOMETRY):
                ax = axes[row, j]
                mean, _ = directional(name, j, args.plot_alpha)
                values = np.concatenate([mean[self.kinds == kind] for kind in KINDS[1:]] + [mean[:1]])
                bins = np.linspace(values.min(), values.max(), 25)
                for kind in KINDS[1:]:
                    ax.hist(mean[self.kinds == kind], bins=bins, color=KIND_COLORS[kind], alpha=0.5, label=kind)
                ax.axvline(mean[0], color=KIND_COLORS["probe"], linewidth=2, label="probe")
                ax.axvline(0, color="#888888", linewidth=1, linestyle=":")
                ax.grid(alpha=0.3)
                if j == 0:
                    ax.set_ylabel(f"{LABELS[name]}\n# directions")
            axes[0, j].set_title(f"steer at {l}")
            axes[-1, j].set_xlabel("directional effect D")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{title} | |α| = {args.plot_alpha:g} | expected D: cosine +, angles −")
        self.save(fig, out_dir, "specificity_hist.pdf")

        def against(ax: plt.Axes, x: np.ndarray, effects: list[tuple[np.ndarray, float]]) -> None:
            """Probe (± SE) against the random mean and 95% interval (bars, plus a band when x has several points)."""
            dist = np.stack([mean[self.kinds == "random"] for mean, _ in effects])  # (len(x), N_RANDOM)
            centre, lo, hi = dist.mean(axis=1), np.quantile(dist, 0.025, axis=1), np.quantile(dist, 0.975, axis=1)
            ax.errorbar(x, centre, yerr=[centre - lo, hi - centre], color=KIND_COLORS["random"], linewidth=2,
                        marker="s", markersize=5, capsize=3, label="random mean, 95%")
            ax.fill_between(x, lo, hi, color=KIND_COLORS["random"], alpha=0.15, linewidth=0)
            ax.errorbar(x, [mean[0] for mean, _ in effects], yerr=[se for _, se in effects],
                        color=KIND_COLORS["probe"], linewidth=2, marker="o", markersize=5, capsize=3, label="probe")
            ax.axhline(0, color="#888888", linewidth=1)
            ax.grid(alpha=0.3)

        # 2. directional effect across intervention layers, one column per |alpha|
        fig, axes = plt.subplots(len(GEOMETRY), len(SPEC_ALPHAS), figsize=(4 * len(SPEC_ALPHAS), 3 * len(GEOMETRY)),
                                 sharex=True, squeeze=False)
        for col, a in enumerate(SPEC_ALPHAS):
            for row, name in enumerate(GEOMETRY):
                against(axes[row, col], np.asarray(intervene), [directional(name, j, a) for j in range(J)])
                if col == 0:
                    axes[row, col].set_ylabel(ylabel(name))
            axes[0, col].set_title(f"|α| = {a:g}")
            axes[-1, col].set_xlabel("steering layer")
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle(title)
        self.save(fig, out_dir, "specificity_layers.pdf")

        # 3. directional effect against |alpha|, one column per intervention layer
        fig, axes = plt.subplots(len(GEOMETRY), J, figsize=(3.4 * J, 3 * len(GEOMETRY)), sharex=True,
                                 squeeze=False)
        for j, l in enumerate(intervene):
            for row, name in enumerate(GEOMETRY):
                against(axes[row, j], np.asarray(SPEC_ALPHAS), [directional(name, j, a) for a in SPEC_ALPHAS])
                if j == 0:
                    axes[row, j].set_ylabel(ylabel(name))
            axes[0, j].set_title(f"steer at {l}")
            axes[-1, j].set_xlabel("|α| (× mean token norm)")
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle(title)
        self.save(fig, out_dir, "specificity_strength.pdf")

    def run(self) -> None:
        args = self.args
        if args.plot_alpha not in SPEC_ALPHAS:
            raise ValueError(f"--plot_alpha must be one of {SPEC_ALPHAS} for --experiment specificity")
        out_dir = os.path.join(OUTPUT_DIR, f"specificity_{args.model}_{args.dataset}_s{args.seed}_"
                                           f"{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(out_dir, exist_ok=False)

        data = load_data(args=args)
        probe_split = "train"
        if "train" not in data:
            # no train split: fit the probes on a stratified half of test, steer the other half (as in test.py)
            probe_split = "half of test"
            items = data["test"]
            fit_idx, steer_idx = train_test_split(np.arange(len(items)), test_size=0.5, random_state=args.seed,
                                                  stratify=[item["label"] for item in items])
            data = {"train": [items[i] for i in fit_idx], "test": [items[i] for i in steer_idx]}
            print(f"No train split: probes on {len(fit_idx)} test documents, steering on the other {len(steer_idx)}")
        test_items = data["test"]
        if args.max_docs:
            # balanced subset: max_docs / 2 per class, in a seeded random order, among documents long enough to use
            per_class, kept = args.max_docs // 2, {0: [], 1: []}
            for i in np.random.default_rng(args.seed).permutation(len(test_items)):
                item = test_items[i]
                if len(kept[item["label"]]) < per_class and self.tokenize(item["text"]) is not None:
                    kept[item["label"]].append(int(i))
            test_items = [test_items[i] for i in sorted(kept[0] + kept[1])]
            print(f"Steering a balanced subset: human={len(kept[0])} machine={len(kept[1])} test documents")

        train = self.collect_train(data["train"])
        probes = self.fit_probes(train["last"], train["labels"])
        target, _, separation = self.select_layers(train)
        if args.intervene_layers:
            intervene = sorted(l for l in set(args.intervene_layers) if 1 <= l <= target)
        else:
            intervene = list(range(max(2, target - 5), target))
        print(f"Target layer {target} | intervention layers {intervene}")

        dirs = self.directions(probes, intervene)
        result = self.steer(test_items, intervene, dirs, probes, train["norms"], target)
        delta, labels = result["delta"], result["labels"]
        rows = self.analyse(delta, labels, intervene)
        self.report(rows, target, intervene, labels)
        self.plot(delta, labels, intervene, target, out_dir)

        path = os.path.join(out_dir, "per_document.npz")
        np.savez_compressed(path, labels=labels, intervene_layers=np.asarray(intervene),
                            signed_alphas=np.asarray(SPEC_SIGNED), kinds=self.kinds,
                            **{f"delta_{name}": v for name, v in delta.items()},
                            **{f"base_{name}": v for name, v in result["base"].items()})
        print(f"Saved {path}  (delta_*: documents x intervention layer x direction x signed alpha, "
              f"change against alpha = 0 at layer {target}; base_*: documents x intervention layer)")
        path = os.path.join(out_dir, "directions.npz")
        np.savez_compressed(path, kinds=self.kinds, seeds=np.asarray([args.seed + l for l in intervene]),
                            **{f"layer_{l}": dirs[l].astype(np.float32) for l in intervene})
        print(f"Saved {path}")

        path = os.path.join(out_dir, "summary.json")
        with open(path, "w") as f:
            json.dump({
                "config": {
                    "model": args.model, "model_id": self.inference.model_id, "dtype": str(self.dtype),
                    "dataset": args.dataset, "seed": args.seed, "n_tokens": N_TOKENS, "probe_split": probe_split,
                    "layer_convention": "layer l = output of decoder layer l (1..L), before the final norm",
                    "alphas": list(SPEC_ALPHAS), "signed_alphas": list(SPEC_SIGNED),
                    "alpha_scale": "alpha x mean train token norm at the steering layer (position 0 excluded)",
                    "token_norm": train["norms"].tolist(), "target_layer": target,
                    "target_from_args": args.target_layer is not None, "intervene_layers": intervene,
                    "steer_tokens": "all tokens, same direction for every token and document",
                    "n_random": N_RANDOM,
                    "directions": {"probe": "w_raw / ||w_raw|| of the last-token probe at the steering layer",
                                   "random": "isotropic Gaussian, normalised; numpy default_rng(seed + layer)"},
                    "max_docs": args.max_docs, "batch_size": args.batch_size,
                    "n_test": {"human": int((labels == 0).sum()), "machine": int((labels == 1).sum())},
                },
                "probe": {"C": PROBE_C, "train_auroc": probes["train_auroc"].tolist()},
                "train_clean_separation_auroc": separation,
                "stats": "percentile / p / z rank the probe among the random directions; 'signed' in "
                         "the expected direction, 'abs' on the size of the effect; p = (1 + #>=) / (1 + n)",
                "results": rows,
            }, f, indent=1)
        print(f"Saved {path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, default="specificity", choices=("steer", "specificity"),
                        help="steer: probe direction only, all alphas and layers; specificity: probe against "
                             "random directions")
    parser.add_argument("--target_layer", type=int, default=None,
                        help="layer whose geometry is measured; default: strongest clean separation on train")
    parser.add_argument("--intervene_layers", type=int, nargs="+", default=None,
                        help="steering layers, at most the target; default: 2..target - 1 plus the target as "
                             "reference (steer), target - 5 .. target - 1 (specificity)")
    parser.add_argument("--plot_alpha", type=float, default=0.1,
                        help="|alpha| for the layer-wise figures (steer) or the histograms (specificity)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="specificity: rows per re-run of layers l+1..m")
    parser.add_argument("--max_docs", type=int, default=None, help="specificity: stratified subset of test documents")
    args = parser.parse_args()
    args.model_name = args.model
    return args


def main() -> None:
    args = parse_args()
    experiment = {"steer": SteeringGeometry, "specificity": DirectionSpecificity}[args.experiment]
    experiment(args).run()


if __name__ == "__main__":
    main()
