"""Does the machine-human direction control representation geometry?

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


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_layer", type=int, default=None,
                        help="layer whose geometry is measured; default: strongest clean separation on train")
    parser.add_argument("--intervene_layers", type=int, nargs="+", default=None,
                        help="steering layers, at most the target; default: 2..target - 1 plus the target as "
                             "reference")
    parser.add_argument("--plot_alpha", type=float, default=0.1, help="|alpha| for the layer-wise figures")
    args = parser.parse_args()
    args.model_name = args.model
    return args


def main() -> None:
    args = parse_args()
    SteeringGeometry(args).run()


if __name__ == "__main__":
    main()
