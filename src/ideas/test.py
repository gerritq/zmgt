"""Does the machine-human direction control representation geometry?

Two experiments, run one after the other on the same probes, target layers and token norms, repeated for each of
three probes (PROBES), each in its own subfolder of the run folder:
  last    one example per document: the last token's state
  mean    one example per document: the mean state over tokens 1..T-1 (position 0, the attention sink, excluded)
  tokens  one example per token: every position 1..T-1 of every document, each with the document's label. Too
          large for memory at once, so the states are collected TOKEN_PROBE_GROUP layers at a time on the GPU and
          the probe is fitted there (L-BFGS on the same objective as the sklearn probes: standardised features,
          sum of log losses + ||w||^2 / (2 C), intercept not penalised)
The document score of the mean and tokens probes is s_l(mean_t h_t), which equals the mean of the per-token scores
since the probe is linear.

  1. steer        (SteeringGeometry)      steering along the probe direction at every layer 1..MAX_LAYER, 11 strengths
  2. specificity  (DirectionSpecificity)  the probe direction against random directions of the same size, at every
                                          layer 1..MAX_LAYER; see that class
Probes are trained for layers 1..MAX_LAYER only; the probe score is nan above.

Layer convention: layer l = 1..L is the output of decoder layer l (hidden_states[l] in HF), recorded with forward
hooks, so layer L is taken before the final norm. Every document is used in full (no truncation); documents with
fewer than MIN_TOKENS tokens are left out. Position 0 (the attention sink) is skipped everywhere: it is not pushed,
not part of any metric and not a probe feature.

1. Direction. On the train split (or, if the dataset has none, a stratified PROBE_FRACTION of test, with the rest
   steered), the probe's features (above) at every layer l <= MAX_LAYER are fed to a linear probe
   (StandardScaler + LogisticRegression, label 1 = machine). The probe's normal in raw hidden-state space is
   w_l = coef / scale, the score is s_l(h) = w_l . h + b_l (positive = machine), and v_l = w_l / ||w_l||.
2. Intervention. At layer l, h_t <- h_t + alpha * n_l * v_l for the pushed tokens t: positions 1..T-1 whose clean
   norm at layer l is at most NORM_OUTLIER x the median norm of the document's positions 1..T-1 at that layer
   (high-norm outlier tokens are left alone). n_l is the mean train token norm at layer l (position 0 excluded),
   so alpha is relative to a typical hidden-state norm. The forward pass then continues through layers l+1..L,
   and every metric is recorded at every layer. alpha > 0 pushes towards machine, alpha < 0 towards human.
3. Targets. For each metric separately, the target layer is the layer with the largest absolute difference of the
   clean human and machine means on the train split (over layers 1..L). Effects of a metric are read at its own
   target, for the steering layers at or below it. The probe score is read at MAX_LAYER.
4. Geometry per document, layer and alpha (test split), over positions 1..T-1:
     global_cos:  mean off-diagonal cosine between all token pairs
     local_angle: mean angle (degrees) between adjacent tokens t, t+1
     cross_angle: mean angle (degrees) between the same token at layers m - 1 and m (layer 0 = embeddings); when
                  steering at m itself, only the h^m end of this angle is moved
     probe:       document score (last token or token mean) of the probe trained at the measurement layer m
   alpha = 0 is run in the same batch as the steered rows, so differences are not batch noise.
Hypothesis checked at each target: human + alpha (H->M) raises global_cos and lowers local_angle and
cross_angle; machine - alpha (M->H) does the opposite. Each change D(alpha) is split, on all documents, into
  directional = (D(+a) - D(-a)) / 2, what flips with the sign of the push (the effect of the direction itself), and
  shared      = (D(+a) + D(-a)) / 2, what does not (generic collapse from adding any vector to every token).
The probe score is reported separately, since moving along v changes it by construction.
All files of a run go to one new folder, output/test/steer_spec_<model>_<dataset>_s<seed>_<timestamp>, with one
subfolder per probe (last/, mean/, tokens/) holding steer_* and specificity_* files, probes.npz and directions.npz;
nothing is overwritten.
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
# documents are used in full; shorter ones are left out (adjacent-token and pairwise metrics need a few tokens)
MIN_TOKENS = 3
# small strengths only: |alpha| >= 0.5 collapsed the geometry whatever the sign
ALPHAS = (-0.4, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.4)
MAGNITUDES = tuple(a for a in ALPHAS if a > 0)
PROBE_C = 1.0
# share of test used to fit the probes when a dataset has no train split (the rest is steered)
PROBE_FRACTION = 0.7
PROBES = ("last", "mean", "tokens")
PROBE_FEATURES = {
    "last": "last-token state per document",
    "mean": "mean state over tokens 1..T-1 per document",
    "tokens": "every token (positions 1..T-1) of every document with its document's label, fitted on the GPU; "
              "document score = score of the token mean",
}
# layers whose token states are held on the GPU at once while fitting the tokens probe
TOKEN_PROBE_GROUP = 4
# probes are trained and steering is done at layers 1..MAX_LAYER
MAX_LAYER = 20
# tokens whose clean norm exceeds NORM_OUTLIER x the document's median token norm (at the steering layer) are not pushed
NORM_OUTLIER = 2.0
GEOMETRY = ("global_cos", "local_angle", "cross_angle")
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
        self.shift: tuple[int, torch.Tensor] | None = None  # (layer l, (B, T, d), zero on tokens not pushed)
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
    """(L, T, d) states of layers 1..L and (T, d) embeddings -> global_cos, local_angle, cross_angle, each (L,),
    over positions 1..T-1."""
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
        self.n_probe = min(MAX_LAYER, self.n_layers)
        self.intervene = list(range(1, self.n_probe + 1))
        self.steering = Steering(self.decoder)
        self.alphas = torch.tensor(ALPHAS, device=self.device, dtype=torch.float32)
        # which probe's document score to read (set per run in main)
        self.probe = "last"

    def tokenize(self, text: str) -> torch.Tensor | None:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if ids.shape[1] < MIN_TOKENS:
            return None
        return ids.to(self.device)

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
        """Probe features (last and mean: (N, L, d)), token ids (for the tokens probe), clean geometry (N, L),
        token norms (L,) and labels of the train split."""
        last, mean, ids_all, values, norms, labels = [], [], [], {name: [] for name in GEOMETRY}, [], []
        for item in tqdm(data, desc="Train states"):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            H, E = self.forward(ids)
            H = H[0]
            last.append(H[:, -1].float().cpu().numpy())
            mean.append(H[:, 1:].float().mean(dim=1).cpu().numpy())
            ids_all.append(ids.cpu())
            for name, value in geometry(H, E[0]).items():
                values[name].append(value.cpu().numpy())
            norms.append(H[:, 1:].float().norm(dim=-1).mean(dim=-1).cpu().numpy())
            labels.append(item["label"])
        return {"last": np.stack(last), "mean": np.stack(mean), "ids": ids_all,
                "labels": np.asarray(labels), "norms": np.mean(norms, axis=0),
                "values": {name: np.stack(v) for name, v in values.items()}}

    def fit_probes(self, features: np.ndarray, labels: np.ndarray, doc_features: np.ndarray,
                   doc_labels: np.ndarray) -> dict:
        """Per layer 1..MAX_LAYER, the probe in raw space: w (P, d), b (P,), and its document-level train AUROC."""
        W, b, train_auroc = [], [], []
        for i in tqdm(range(self.n_probe), desc="Probes"):
            X = features[:, i].astype(np.float32)
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=PROBE_C, max_iter=5000))
            probe.fit(X, labels)
            scaler, clf = probe[0], probe[1]
            assert list(clf.classes_) == [0, 1], "label 1 must be machine"
            w = clf.coef_[0] / scaler.scale_
            W.append(w)
            b.append(clf.intercept_[0] - (w * scaler.mean_).sum())
            train_auroc.append(roc_auc_score(doc_labels, doc_features[:, i] @ w + b[-1]))
        return {"W": np.stack(W), "b": np.asarray(b), "train_auroc": np.asarray(train_auroc)}

    @staticmethod
    def gpu_logistic(X: torch.Tensor, y: torch.Tensor) -> tuple[np.ndarray, float]:
        """The sklearn probe's optimum on the GPU: standardise X (n, d), minimise
        sum_i logloss_i + ||w||^2 / (2 C) with an unpenalised intercept; returns w, b in raw space."""
        X = X.float()
        mu = X.mean(dim=0)
        sd = X.std(dim=0, unbiased=False)
        sd[sd == 0] = 1.0
        X.sub_(mu).div_(sd)
        n = X.shape[0]
        w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
        b = torch.zeros(1, device=X.device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1, max_iter=1000, history_size=20, tolerance_grad=1e-7,
                                tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            z = X @ w + b
            # mean form of the same objective, divided by n for scale
            loss = (torch.nn.functional.softplus(z) - y * z).mean() + w.pow(2).sum() / (2 * PROBE_C * n)
            loss.backward()
            return loss

        opt.step(closure)
        w_raw = (w.detach() / sd).double()
        b_raw = float(b.detach().double() - (w_raw * mu.double()).sum())
        return w_raw.cpu().numpy(), b_raw

    @torch.no_grad()
    def collect_layer_tokens(self, ids_all: list[torch.Tensor], layers: list[int]) -> dict[int, torch.Tensor]:
        """States of positions 1..T_k-1 of every train document k at the given layers, (sum_k (T_k - 1), d) each,
        on GPU, documents in order."""
        sizes = [ids.shape[1] - 1 for ids in ids_all]
        out = {l: torch.empty(sum(sizes), self.model.config.hidden_size, device=self.device, dtype=self.dtype)
               for l in layers}
        start = 0
        for ids, size in zip(ids_all, sizes):
            H, _ = self.forward(ids.to(self.device))
            for l in layers:
                out[l][start:start + size] = H[0, l - 1, 1:]
            start += size
        return out

    def fit_token_probes(self, train: dict) -> dict:
        """The tokens probe per layer: all token states, collected TOKEN_PROBE_GROUP layers at a time."""
        sizes = [ids.shape[1] - 1 for ids in train["ids"]]
        y = torch.tensor(np.repeat(train["labels"], sizes), device=self.device, dtype=torch.float32)
        print(f"Tokens probe: {sum(sizes)} train tokens per layer ({len(sizes)} documents, "
              f"{min(sizes) + 1}-{max(sizes) + 1} tokens each)")
        W, b, train_auroc = [None] * self.n_probe, [None] * self.n_probe, [None] * self.n_probe
        groups = [list(self.layers[i:min(i + TOKEN_PROBE_GROUP, self.n_probe)])
                  for i in range(0, self.n_probe, TOKEN_PROBE_GROUP)]
        for group in tqdm(groups, desc="Token probes (layer groups)"):
            states = self.collect_layer_tokens(train["ids"], group)
            for l in group:
                with torch.enable_grad():
                    W[l - 1], b[l - 1] = self.gpu_logistic(states.pop(l), y)
                train_auroc[l - 1] = roc_auc_score(train["labels"], train["mean"][:, l - 1] @ W[l - 1] + b[l - 1])
            torch.cuda.empty_cache()
        return {"W": np.stack(W), "b": np.asarray(b), "train_auroc": np.asarray(train_auroc)}

    def pooled(self, H: torch.Tensor) -> torch.Tensor:
        """(..., T, d) -> (..., d): the state the current probe scores (last token, or mean over tokens 1..T-1)."""
        return H[..., -1, :] if self.probe == "last" else H[..., 1:, :].mean(dim=-2)

    def probe_score(self, H: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(..., L, T, d) states of layers 1..L -> (..., L) document scores of each layer's probe, nan above the
        probed layers."""
        pooled = self.pooled(H.float())
        out = torch.full(pooled.shape[:-1], float("nan"), device=pooled.device)
        out[..., :self.n_probe] = (pooled[..., :self.n_probe, :] * W).sum(dim=-1) + b
        return out

    @staticmethod
    def push_mask(H: torch.Tensor) -> torch.Tensor:
        """(L, T, d) clean states -> (L, T) bool, the tokens pushed at each layer: positions 1..T-1 with a norm of at
        most NORM_OUTLIER x the median norm of positions 1..T-1 at that layer."""
        norms = H.float().norm(dim=-1)
        median = norms[:, 1:].median(dim=1, keepdim=True).values
        mask = norms <= NORM_OUTLIER * median
        mask[:, 0] = False
        return mask

    def select_targets(self, train: dict) -> tuple[dict, dict]:
        """Per metric, the layer with the largest |machine mean - human mean| of the clean train values; the probe
        is read at the last probed layer."""
        labels, criterion, targets = train["labels"], {}, {}
        for name in GEOMETRY:
            v = train["values"][name]
            human, machine = v[labels == 0].mean(axis=0), v[labels == 1].mean(axis=0)
            diff = machine - human
            targets[name] = int(self.layers[np.argmax(np.abs(diff))])
            criterion[name] = {"human_mean": human.tolist(), "machine_mean": machine.tolist(),
                               "machine_minus_human": diff.tolist()}
            print(f"Target {name}: layer {targets[name]} (train machine - human = {diff[targets[name] - 1]:+.4f})")
        targets["probe"] = self.n_probe
        return targets, criterion

    def prepare(self) -> dict:
        """Data split, train states, probes and layers, shared by both experiments."""
        args = self.args
        data = load_data(args=args)
        probe_split = "train"
        if "train" not in data:
            # no train split: fit the probes on a stratified PROBE_FRACTION of test, steer the rest
            probe_split = f"{PROBE_FRACTION:.0%} of test"
            items = data["test"]
            fit_idx, steer_idx = train_test_split(np.arange(len(items)), test_size=round(1 - PROBE_FRACTION, 6), random_state=args.seed,
                                                  stratify=[item["label"] for item in items])
            data = {"train": [items[i] for i in fit_idx], "test": [items[i] for i in steer_idx]}
            print(f"No train split: probes on {len(fit_idx)} test documents, steering on the other {len(steer_idx)}")
        train = self.collect_train(data["train"])
        probes = {
            "last": self.fit_probes(train["last"], train["labels"], train["last"], train["labels"]),
            "mean": self.fit_probes(train["mean"], train["labels"], train["mean"], train["labels"]),
            "tokens": self.fit_token_probes(train),
        }
        targets, separation = self.select_targets(train)
        out_dir = os.path.join(OUTPUT_DIR, f"steer_spec_{args.model}_{args.dataset}_s{args.seed}_"
                                           f"{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(out_dir, exist_ok=False)
        return {"out_dir": out_dir, "data": data, "probe_split": probe_split, "train": train, "probes": probes,
                "targets": targets, "intervene": self.intervene, "separation": separation}

    def steer_test(self, data: list[dict], intervene: list[int], probes: dict, scale: np.ndarray) -> dict:
        """Metrics (N, J, A, L) for every test document, intervention layer j, alpha a and measurement layer."""
        V = torch.tensor(probes["W"] / np.linalg.norm(probes["W"], axis=1, keepdims=True),
                         device=self.device, dtype=torch.float32)
        W = torch.tensor(probes["W"], device=self.device, dtype=torch.float32)
        b = torch.tensor(probes["b"], device=self.device, dtype=torch.float32)
        values, labels, below_diff, pushed = {name: [] for name in SCORES}, [], 0.0, []
        for item in tqdm(data, desc="Steering"):
            ids = self.tokenize(item["text"])
            if ids is None:
                continue
            clean, _ = self.forward(ids)
            mask = self.push_mask(clean[0])  # (L, T)
            pushed.append(mask[:self.n_probe, 1:].float().mean(dim=1).cpu().numpy())
            ids = ids.expand(len(ALPHAS), -1)
            doc = {name: [] for name in SCORES}
            for l in intervene:
                shift = ((self.alphas * float(scale[l - 1]))[:, None, None] * V[l - 1]) * mask[l - 1][None, :, None]
                H, E = self.forward(ids, (l, shift.to(self.dtype)))  # (A, L, T, d), (A, T, d)
                rows = [geometry(H[a], E[a]) for a in range(len(ALPHAS))]
                for name in GEOMETRY:
                    doc[name].append(torch.stack([r[name] for r in rows]).cpu().numpy())
                doc["probe"].append(self.probe_score(H, W, b).cpu().numpy())
                # sanity: layers below l must not depend on alpha
                if l > 1:
                    g = doc["global_cos"][-1][:, :l - 1]
                    below_diff = max(below_diff, float(np.abs(g - g[ALPHAS.index(0.0)]).max()))
            for name in SCORES:
                values[name].append(np.stack(doc[name]))
            labels.append(item["label"])
        print(f"Sanity: max |change| of global_cos below the intervention layer = {below_diff:.2e} (expected 0)")
        pushed = np.stack(pushed)  # (N, P): share of positions 1..T-1 pushed per steering layer
        print(f"Pushed tokens (share of positions 1..T-1, mean over documents) per layer 1..{self.n_probe}: "
              + " ".join(f"{x:.3f}" for x in pushed.mean(axis=0)))
        return {"values": {name: np.stack(v) for name, v in values.items()}, "labels": np.asarray(labels),
                "pushed": pushed}

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

    @staticmethod
    def auroc(labels: np.ndarray, x: np.ndarray) -> float:
        return float("nan") if np.isnan(x).any() else float(roc_auc_score(labels, x))

    def summarize(self, values: dict, labels: np.ndarray, intervene: list[int], targets: dict) -> dict:
        zero = ALPHAS.index(0.0)
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

        # clean machine - human difference at each metric's target, the size a full "class switch" would have
        gap = {name: float(values[name][labels == 1, 0, zero, targets[name] - 1].mean()
                           - values[name][labels == 0, 0, zero, targets[name] - 1].mean()) for name in SCORES}
        geometry_rows, probe_rows = [], []
        for name in GEOMETRY:
            m = targets[name] - 1
            for j, l in enumerate(intervene):
                if l > targets[name]:
                    continue
                v = values[name][:, j, :, m]
                delta = v - v[:, zero:zero + 1]
                expected = EXPECTED[name]
                for magnitude in MAGNITUDES:
                    plus, minus = ALPHAS.index(magnitude), ALPHAS.index(-magnitude)
                    directional = (delta[:, plus] - delta[:, minus]) / 2
                    geometry_rows.append({
                        "metric": name, "intervene_layer": l, "measure_layer": targets[name], "alpha": magnitude,
                        "H->M": self.stats(delta[labels == 0, plus], expected),
                        "M->H": self.stats(delta[labels == 1, minus], -expected),
                        "directional": self.stats(directional, expected)
                                       | {"frac_of_gap": float(directional.mean() / gap[name])},
                        "shared": self.stats((delta[:, plus] + delta[:, minus]) / 2),
                    })
        m = targets["probe"] - 1
        for j, l in enumerate(intervene):
            v = values["probe"][:, j, :, m]
            for a, alpha in enumerate(ALPHAS):
                if alpha != 0:
                    delta = v[:, a] - v[:, zero]
                    probe_rows.append({"intervene_layer": l, "measure_layer": targets["probe"], "alpha": alpha} | {
                        group: self.stats(delta[labels == label], int(np.sign(alpha))) for label, group, _ in GROUPS})

        # human/machine AUROC of the unsteered metrics (alpha = 0) at each layer (nan where the probe is undefined)
        clean_auroc = {name: [self.auroc(labels, values[name][:, 0, zero, i]) for i in range(self.n_layers)]
                       for name in SCORES}
        return {"groups": groups, "clean_gap_at_target": gap, "geometry": geometry_rows, "probe_change": probe_rows,
                "clean_test_auroc": clean_auroc}

    def report(self, summary: dict, probes: dict, targets: dict, separation: dict, intervene: list[int],
               train: dict, test_labels: np.ndarray) -> None:
        args = self.args
        print(f"\nSteering geometry | model={args.model} dataset={args.dataset} seed={args.seed} | "
              f"full documents, position 0 skipped | train human={(train['labels'] == 0).sum()} "
              f"machine={(train['labels'] == 1).sum()} | test human={(test_labels == 0).sum()} "
              f"machine={(test_labels == 1).sum()}")
        print(f"Targets {targets} | intervention layers {intervene[0]}..{intervene[-1]} | alphas {ALPHAS} x mean "
              f"token norm | tokens above {NORM_OUTLIER:g} x median norm not pushed")

        print("\n=== Probes and clean geometry per layer (train machine - human; * = target; test AUROC) ===")
        print(f"{'layer':>6} {'norm':>8} {'probe tr':>9} {'probe te':>9} {'D cos':>10} {'D ang':>9} {'D cross':>9} "
              f"{'AUC cos':>8} {'AUC ang':>8} {'AUC cross':>9}")
        auc = summary["clean_test_auroc"]
        for i, layer in enumerate(self.layers):
            tr = f"{probes['train_auroc'][i]:>9.4f}" if i < self.n_probe else f"{'-':>9}"
            te = f"{auc['probe'][i]:>9.4f}" if i < self.n_probe else f"{'-':>9}"
            diffs = " ".join(f"{separation[name]['machine_minus_human'][i]:>+{w - 1}.4f}"
                             f"{'*' if targets[name] == layer else ' '}"
                             for name, w in zip(GEOMETRY, (10, 9, 9)))
            print(f"{layer:>6} {train['norms'][i]:>8.2f} {tr} {te} {diffs} {auc['global_cos'][i]:>8.4f} "
                  f"{auc['local_angle'][i]:>8.4f} {auc['cross_angle'][i]:>9.4f}")

        print("\n=== Geometry change at each metric's target: mean (fraction of docs in the expected direction) ===")
        print("H->M: human docs, +alpha; M->H: machine docs, -alpha. directional = (D(+a) - D(-a)) / 2 on all docs, "
              "with % of the clean machine - human gap and Wilcoxon p; shared = (D(+a) + D(-a)) / 2, the part that "
              "does not depend on the sign (generic collapse)")
        for name in GEOMETRY:
            print(f"\n--- {name} at layer {targets[name]} | expected: H->M {EXPECTED[name]:+d}, "
                  f"M->H {-EXPECTED[name]:+d} | clean machine - human gap = {summary['clean_gap_at_target'][name]:+.4f} ---")
            print(f"{'int':>4} {'|a|':>6} {'H->M':>17} {'M->H':>17} {'directional':>17} {'% gap':>7} "
                  f"{'p':>8} {'shared':>10}")
            for r in summary["geometry"]:
                if r["metric"] != name:
                    continue
                hm, mh, d = r["H->M"], r["M->H"], r["directional"]
                print(f"{r['intervene_layer']:>4} {r['alpha']:>6.3f} "
                      f"{hm['mean']:>+10.4f} ({hm['frac_expected']:.2f}) {mh['mean']:>+10.4f} ({mh['frac_expected']:.2f}) "
                      f"{d['mean']:>+10.4f} ({d['frac_expected']:.2f}) {100 * d['frac_of_gap']:>+6.1f}% "
                      f"{d['wilcoxon_p']:>8.1e} {r['shared']['mean']:>+10.4f}")

        print(f"\n=== Probe score change at layer {targets['probe']} (reported separately: moving along v changes it "
              f"by construction) | clean gap = {summary['clean_gap_at_target']['probe']:+.3f} ===")
        print(f"{'int':>4} {'alpha':>7} {'human':>17} {'machine':>17}")
        for row in summary["probe_change"]:
            h, mc = row["human"], row["machine"]
            print(f"{row['intervene_layer']:>4} {row['alpha']:>+7.3f} {h['mean']:>+10.3f} ({h['frac_expected']:.2f}) "
                  f"{mc['mean']:>+10.3f} ({mc['frac_expected']:.2f})")

    def plot(self, values: dict, labels: np.ndarray, intervene: list[int], targets: dict, out_dir: str) -> None:
        args = self.args
        title = f"{args.model} | {args.dataset} | seed {args.seed}"
        zero = ALPHAS.index(0.0)
        J = len(intervene)

        def band(ax: plt.Axes, x: np.ndarray, rows: np.ndarray, color: str, label: str, style: str = "-",
                 marker: str | None = None) -> None:
            mean = rows.mean(axis=0)
            se = rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
            ax.plot(x, mean, style, color=color, linewidth=2, marker=marker, markersize=4, label=label)
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, linewidth=0)

        def delta(name: str, j: int) -> np.ndarray:
            """(N, A) change at the metric's target layer relative to alpha = 0."""
            v = values[name][:, j, :, targets[name] - 1]
            return v - v[:, zero:zero + 1]

        # 1. layer-wise effects at every measurement layer: original vs steered (H->M on human, M->H on machine)
        a_plus, a_minus = ALPHAS.index(args.plot_alpha), ALPHAS.index(-args.plot_alpha)
        fig, axes = plt.subplots(len(GEOMETRY), J, figsize=(2.8 * J, 3 * len(GEOMETRY)), sharex=True,
                                 squeeze=False)
        for j, l in enumerate(intervene):
            for row, name in enumerate(GEOMETRY):
                ax = axes[row, j]
                for (label, group, color), a, arrow in zip(GROUPS, (a_plus, a_minus), ("H→M", "M→H")):
                    v = values[name][labels == label, j]
                    band(ax, self.layers, v[:, zero], color, f"{group} original")
                    band(ax, self.layers, v[:, a], color, f"{group} {arrow} (α={ALPHAS[a]:+g})", "--")
                ax.axvline(l, color="#888888", linewidth=1)
                ax.axvline(targets[name], color="#888888", linewidth=1, linestyle=":")
                ax.grid(alpha=0.3)
                if j == 0:
                    ax.set_ylabel(LABELS[name])
            axes[0, j].set_title(f"steer at {l}")
            axes[-1, j].set_xlabel("measurement layer")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{title} | solid line: steering layer, dotted: the metric's target")
        self.save(fig, out_dir, "steer_layers.pdf")

        # 2. depth comparison at each metric's target: both directions, then split into directional and shared parts
        kinds = (("H→M: human docs, +α", lambda d, p, n: d[labels == 0, p]),
                 ("M→H: machine docs, −α", lambda d, p, n: d[labels == 1, n]),
                 ("directional: (Δ(+α) − Δ(−α)) / 2", lambda d, p, n: (d[:, p] - d[:, n]) / 2),
                 ("shared: (Δ(+α) + Δ(−α)) / 2", lambda d, p, n: (d[:, p] + d[:, n]) / 2))
        fig, axes = plt.subplots(len(kinds), len(GEOMETRY), figsize=(5.5 * len(GEOMETRY), 3 * len(kinds)),
                                 squeeze=False)
        for col, name in enumerate(GEOMETRY):
            steer = [(j, l) for j, l in enumerate(intervene) if l <= targets[name]]
            deltas = [delta(name, j) for j, _ in steer]
            x = [l for _, l in steer]
            for row, (kind, select) in enumerate(kinds):
                ax = axes[row, col]
                for magnitude, color in zip(MAGNITUDES, MAGNITUDE_COLORS):
                    p, n = ALPHAS.index(magnitude), ALPHAS.index(-magnitude)
                    rows = np.stack([select(d, p, n) for d in deltas], axis=1)
                    mean = rows.mean(axis=0)
                    se = rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
                    ax.errorbar(x, mean, yerr=se, color=color, linewidth=2, marker="o", markersize=5,
                                capsize=3, label=f"|α|={magnitude:g}")
                ax.axhline(0, color="#888888", linewidth=1)
                ax.grid(alpha=0.3)
                ax.set_title(f"{LABELS[name]} at layer {targets[name]} | {kind}", fontsize=9)
                if col == 0:
                    ax.set_ylabel("Δ at target")
            axes[-1, col].set_xlabel("steering layer")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{title} | expected: cosine ↑, both angles ↓ for H→M and directional, opposite for M→H",
                     fontsize=10)
        self.save(fig, out_dir, "steer_depth.pdf")

    @staticmethod
    def save(fig: plt.Figure, out_dir: str, name: str) -> None:
        path = os.path.join(out_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    def run_steer(self, prep: dict) -> None:
        args = self.args
        if args.plot_alpha not in ALPHAS or -args.plot_alpha not in ALPHAS or args.plot_alpha <= 0:
            raise ValueError(f"--plot_alpha must be a positive value with ±alpha in {ALPHAS}")
        out_dir = prep["out_dir"]

        data, probe_split, train, probes = prep["data"], prep["probe_split"], prep["train"], prep["probes"]
        targets, intervene, separation = prep["targets"], prep["intervene"], prep["separation"]
        test = self.steer_test(data["test"], intervene, probes, train["norms"])
        values, labels = test["values"], test["labels"]

        summary = self.summarize(values, labels, intervene, targets)
        test_probe_auroc = summary["clean_test_auroc"]["probe"]
        self.report(summary, probes, targets, separation, intervene, train, labels)
        self.plot(values, labels, intervene, targets, out_dir)

        path = os.path.join(out_dir, "steer_per_document.npz")
        np.savez_compressed(path, labels=labels, alphas=np.asarray(ALPHAS), intervene_layers=np.asarray(intervene),
                            layers=self.layers, pushed=test["pushed"], **values)
        print(f"Saved {path}  (arrays: documents x intervention layer x alpha x measurement layer; pushed: documents "
              f"x steering layer, share of positions 1..T-1 pushed)")
        path = os.path.join(out_dir, "probes.npz")
        np.savez_compressed(path, W=probes["W"], b=probes["b"], token_norm=train["norms"],
                            layers=self.layers[:self.n_probe])
        print(f"Saved {path}")

        path = os.path.join(out_dir, "steer_summary.json")
        with open(path, "w") as f:
            json.dump({
                "config": {
                    "model": args.model, "model_id": self.inference.model_id, "dtype": str(self.dtype),
                    "dataset": args.dataset, "seed": args.seed, "n_tokens": "full documents", "min_tokens": MIN_TOKENS, "probe_split": probe_split,
                    "layer_convention": "layer l = output of decoder layer l (1..L), before the final norm",
                    "alphas": list(ALPHAS), "alpha_scale": "alpha x mean train token norm at the steering layer "
                                                           "(position 0 excluded)",
                    "token_norm": train["norms"].tolist(),
                    "target_layers": targets, "target_rule": "per metric, largest |machine - human| of the clean "
                                                             "train means over layers 1..L; probe at MAX_LAYER",
                    "intervene_layers": intervene, "probe_layers": self.n_probe,
                    "steer_tokens": f"positions 1..T-1 with clean norm <= {NORM_OUTLIER:g} x the document's median "
                                    f"token norm at the steering layer",
                    "pushed_share_mean_per_layer": test["pushed"].mean(axis=0).tolist(),
                    "skip_position_0": "not pushed, not in any metric, not a probe feature",
                    "plot_alpha": args.plot_alpha,
                    "metrics": {"global_cos": "mean pairwise cosine within a layer",
                                "local_angle": "mean angle between adjacent tokens within a layer",
                                "cross_angle": "mean angle of each token between layers m - 1 and m (0 = embeddings);"
                                               " steering at m itself moves only the h^m end"},
                    "n_train": {"human": int((train["labels"] == 0).sum()),
                                "machine": int((train["labels"] == 1).sum())},
                    "n_test": {"human": int((labels == 0).sum()), "machine": int((labels == 1).sum())},
                },
                "probe": {
                    "name": self.probe, "features": PROBE_FEATURES[self.probe],
                    "model": "StandardScaler + LogisticRegression",
                    "C": PROBE_C, "orientation": "positive = machine", "direction": "w_raw / ||w_raw||",
                    "train_auroc": probes["train_auroc"].tolist(), "test_auroc_clean": test_probe_auroc,
                },
                "train_clean_means": separation,
                "summary_axes": "groups[group][metric][stat] is (intervention layer, alpha, measurement layer)",
                **summary,
            }, f, indent=1)
        print(f"Saved {path}")


# ==================================================================================================
# Experiment 2 (specificity, run after steering): probe direction against random directions
# ==================================================================================================
# directions are compared at these |alpha| (each run with + and -)
SPEC_ALPHAS = (0.1,)
# row order within a direction: +a, -a for each a
SPEC_SIGNED = tuple(sign * a for a in SPEC_ALPHAS for sign in (1, -1))
N_RANDOM = 30
KINDS = ("probe", "random")
KIND_COLORS = {"probe": "#1f1f1f", "random": "#2a78d6"}
EFFECTS = ("directional", "H->M", "M->H")
# expected sign of each effect relative to EXPECTED[name] (the H->M sign)
EFFECT_SIGN = {"directional": 1, "H->M": 1, "M->H": -1}
# sanity threshold: relative difference between the re-run tail and the full forward pass
TAIL_TOLERANCE = 0.02


def target_geometry(prev: torch.Tensor, cur: torch.Tensor) -> dict[str, torch.Tensor]:
    """(B, T, d) states at layers m - 1 and m -> global_cos, local_angle at m and cross_angle, each (B,), over
    positions 1..T-1."""
    prev, cur = prev[:, 1:], cur[:, 1:]
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

    Same setup as the steering experiment (layer convention, full documents, per-probe features, token-norm scaling,
    pushed tokens, per-metric targets). At each layer l = 1..MAX_LAYER the probe direction v_l is compared with
    N_RANDOM random unit directions r / ||r||, r ~ N(0, I), drawn once per layer with seed args.seed + l and kept
    fixed across documents. Every direction d is applied as h_t <- h_t + alpha * n_l * d at the pushed tokens of
    layer l, for alpha = +-SPEC_ALPHAS, recorded at every layer l..L and analysed at each metric's target m
    (steering layers l <= m):
      global_cos   mean pairwise cosine at m
      local_angle  mean adjacent-token angle at m
      cross_angle  mean angle of each token between layers m - 1 and m
      probe        document score of the probe trained at MAX_LAYER (moves along v by construction)
    All directions, documents and strengths share the same probes, target and n_l.

    Cost: layers 1..l do not depend on the direction, so they run once per document; only layers l+1..L are re-run,
    for all steered rows in chunks of --batch_size. Each chunk carries its own unsteered row, so every change D(alpha)
    is taken within one batch. Changes are recorded at every layer l..L (statistics and figures use the target m). Attention is causal because no mask is passed (SDPA); this is checked against the full
    forward pass on the first document.

    Per direction k, layer l and |alpha| = a (means over documents):
      directional  (D(+a) - D(-a)) / 2 on all documents
      H->M         D(+a) on human documents;   M->H   D(-a) on machine documents
    Specificity, with e the expected sign of each effect (directional and H->M: cosine +, angles -; M->H opposite):
    the probe's e * effect is ranked against the random directions: percentile,
    empirical one-sided p = (1 + #{random >= probe}) / (1 + N_RANDOM) and z-score; the same on |effect|, i.e. whether
    the probe moves the geometry more than random directions whatever the sign. Results go to the run folder
    shared with the steering experiment (specificity_* files).
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        params = inspect.signature(self.decoder.layers[0].forward).parameters
        self.pass_position_ids = "position_ids" in params
        self.pass_position_embeddings = "position_embeddings" in params
        self.kinds = np.array(["probe"] + ["random"] * N_RANDOM)
        self.signed = torch.tensor(SPEC_SIGNED, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def tail(self, x: torch.Tensor, start: int, stop: int):
        """x (B, T, d), the output of layer `start`, through layers start+1..stop; yields h^start, ..., h^stop."""
        batch, T = x.shape[:2]
        position_ids = torch.arange(T, device=self.device)[None].expand(batch, -1)
        kwargs = {"attention_mask": None}  # causal under SDPA when no mask is given
        if self.pass_position_ids:
            kwargs["position_ids"] = position_ids
        if self.pass_position_embeddings:
            kwargs["position_embeddings"] = self.decoder.rotary_emb(x, position_ids=position_ids)
        yield x
        for layer in self.decoder.layers[start:stop]:
            x = layer(x, **kwargs)
            x = x[0] if isinstance(x, tuple) else x
            self.steering.states = []  # the recording hooks fired as well; drop their copies
            yield x

    def directions(self, probes: dict, intervene: list[int]) -> dict[int, np.ndarray]:
        """Per layer, (1 + N_RANDOM, d) unit directions: probe, then random (rows in self.kinds order)."""
        out = {}
        for l in intervene:
            v = probes["W"][l - 1] / np.linalg.norm(probes["W"][l - 1])
            g = np.random.default_rng(self.args.seed + l).standard_normal((N_RANDOM, v.shape[0]))
            out[l] = np.concatenate([v[None], g / np.linalg.norm(g, axis=1, keepdims=True)])
        return out

    @torch.no_grad()
    def document(self, ids: torch.Tensor, intervene: list[int], shifts: dict[int, torch.Tensor],
                 W: torch.Tensor, b: torch.Tensor, check: bool) -> tuple[dict, dict]:
        """Changes (J, K, len(SPEC_SIGNED), L) at every measurement layer 1..L (0 below the steering layer) and the
        clean values (L,) for one document."""
        H, E = self.forward(ids)
        H, E = H[0], E[0]
        clean = geometry(H, E)
        clean["probe"] = self.probe_score(H, W, b)
        base = {name: v.float().cpu().numpy() for name, v in clean.items()}
        mask = self.push_mask(H)[:, :, None].to(self.dtype)  # (L, T, 1)
        check_layer = self.n_probe
        n_dirs, size = len(self.kinds), self.args.batch_size - 1
        delta = {name: np.zeros((len(intervene), n_dirs * len(SPEC_SIGNED), self.n_layers), dtype=np.float32)
                 for name in SCORES}
        for j, l in enumerate(intervene):
            for start in range(0, len(shifts[l]), size):
                chunk = shifts[l][start:start + size]
                chunk = torch.cat([torch.zeros_like(chunk[:1]), chunk])  # row 0: unsteered
                # at m = l, h^{m-1} is the clean state; above, the previous steered state
                prev = (H[l - 2] if l > 1 else E)[None].expand(len(chunk), -1, -1)
                rows = slice(start, start + len(chunk) - 1)
                x = H[l - 1][None] + chunk[:, None].to(self.dtype) * mask[l - 1][None]
                for m, cur in enumerate(self.tail(x, l, self.n_layers), start=l):
                    values = target_geometry(prev, cur)
                    values["probe"] = (self.pooled(cur.float()) @ W[m - 1] + b[m - 1] if m <= self.n_probe
                                       else torch.full((len(cur),), float("nan"), device=cur.device))
                    for name in SCORES:
                        v = values[name]
                        delta[name][j, rows, m - 1] = (v[1:] - v[0]).cpu().numpy()
                    if check and start == 0 and m == check_layer:
                        rel = ((cur[0].float() - H[m - 1].float()).norm() / H[m - 1].float().norm()).item()
                        print(f"Sanity: steer at {l}, unsteered tail vs full pass at layer {m}: "
                              f"relative difference {rel:.2e}")
                        if rel > TAIL_TOLERANCE:
                            raise RuntimeError("re-running the tail does not reproduce the full forward pass; "
                                               "check that the attention is causal without a mask (SDPA)")
                    prev = cur
        return {name: d.reshape(len(intervene), n_dirs, len(SPEC_SIGNED), self.n_layers)
                for name, d in delta.items()}, base

    def steer(self, data: list[dict], intervene: list[int], dirs: dict, probes: dict, scale: np.ndarray) -> dict:
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
            delta, base = self.document(ids, intervene, shifts, W, b, check=not labels)
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

    def analyse(self, delta: dict, labels: np.ndarray, intervene: list[int], targets: dict) -> list[dict]:
        """delta[name] is (N, J, K, S) at the metric's target; only steering layers at or below it are analysed."""
        rows = []
        for name in SCORES:
            for j, l in enumerate(intervene):
                if l > targets[name]:
                    continue
                for a in SPEC_ALPHAS:
                    row = {"metric": name, "intervene_layer": l, "measure_layer": targets[name], "alpha": a}
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

    def report_specificity(self, rows: list[dict], targets: dict, intervene: list[int], labels: np.ndarray) -> None:
        args = self.args
        print(f"\nDirection specificity | model={args.model} dataset={args.dataset} seed={args.seed} | "
              f"full documents | test human={(labels == 0).sum()} machine={(labels == 1).sum()}")
        print(f"Targets {targets} | intervention layers {intervene[0]}..{intervene[-1]} | ±alpha {SPEC_ALPHAS} x "
              f"mean token norm | {N_RANDOM} random directions per layer")
        print("Each effect: probe value (mean over documents) | random mean [2.5, 97.5] | percentile and one-sided "
              "empirical p of the probe in the expected direction among random directions | |p|: the same on the "
              "size of the effect. directional D = (D(+a) - D(-a)) / 2 on all documents; H->M = D(+a) on human, "
              "M->H = D(-a) on machine documents. both: probe has the expected sign in H->M and M->H (fraction of "
              "random directions that do)")
        for name in SCORES:
            note = " (moves along the probe by construction)" if name == "probe" else ""
            print(f"\n--- {name} at layer {targets[name]}: expected sign directional / H->M {EXPECTED[name]:+d}, "
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

    def plot_specificity(self, delta: dict, labels: np.ndarray, intervene: list[int], targets: dict,
                         out_dir: str) -> None:
        args = self.args
        title = f"{args.model} | {args.dataset} | seed {args.seed} | targets " + \
                ", ".join(f"{name} {targets[name]}" for name in GEOMETRY)
        J = len(intervene)

        def directional(name: str, j: int, a: float) -> np.ndarray:
            """Mean directional effect per direction (K,), and the probe's SE across documents."""
            per_doc = self.effects(delta[name], labels, j, a)["directional"]
            return per_doc.mean(axis=0), per_doc[:, 0].std(ddof=1) / np.sqrt(len(per_doc))

        def ylabel(name: str) -> str:
            return f"D {LABELS[name]} at {targets[name]} (expected {'+' if EXPECTED[name] > 0 else '−'})"

        # 1. distribution of random directional effects with the probe's effect
        fig, axes = plt.subplots(len(GEOMETRY), J, figsize=(3.4 * J, 3 * len(GEOMETRY)), squeeze=False)
        for j, l in enumerate(intervene):
            for row, name in enumerate(GEOMETRY):
                ax = axes[row, j]
                if l > targets[name]:
                    ax.axis("off")
                    continue
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
        fig, axes = plt.subplots(len(GEOMETRY), len(SPEC_ALPHAS), figsize=(6 * len(SPEC_ALPHAS), 3 * len(GEOMETRY)),
                                 squeeze=False)
        for col, a in enumerate(SPEC_ALPHAS):
            for row, name in enumerate(GEOMETRY):
                steer = [(j, l) for j, l in enumerate(intervene) if l <= targets[name]]
                against(axes[row, col], np.asarray([l for _, l in steer]), [directional(name, j, a) for j, _ in steer])
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
                if l > targets[name]:
                    axes[row, j].axis("off")
                    continue
                against(axes[row, j], np.asarray(SPEC_ALPHAS), [directional(name, j, a) for a in SPEC_ALPHAS])
                if j == 0:
                    axes[row, j].set_ylabel(ylabel(name))
            axes[0, j].set_title(f"steer at {l}")
            axes[-1, j].set_xlabel("|α| (× mean token norm)")
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle(title)
        self.save(fig, out_dir, "specificity_strength.pdf")

    def run_specificity(self, prep: dict) -> None:
        args = self.args
        if args.plot_alpha not in SPEC_ALPHAS:
            raise ValueError(f"--plot_alpha must be one of {SPEC_ALPHAS} for the specificity experiment")
        out_dir = prep["out_dir"]

        data, probe_split, train, probes = prep["data"], prep["probe_split"], prep["train"], prep["probes"]
        targets, intervene, separation = prep["targets"], prep["intervene"], prep["separation"]
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

        print(f"Targets {targets} | intervention layers {intervene[0]}..{intervene[-1]}")
        dirs = self.directions(probes, intervene)
        result = self.steer(test_items, intervene, dirs, probes, train["norms"])
        delta_all, labels = result["delta"], result["labels"]
        # statistics and figures at each metric's target; every layer is saved
        delta = {name: v[..., targets[name] - 1] for name, v in delta_all.items()}
        rows = self.analyse(delta, labels, intervene, targets)
        self.report_specificity(rows, targets, intervene, labels)
        self.plot_specificity(delta, labels, intervene, targets, out_dir)

        path = os.path.join(out_dir, "specificity_per_document.npz")
        np.savez_compressed(path, labels=labels, intervene_layers=np.asarray(intervene),
                            signed_alphas=np.asarray(SPEC_SIGNED), kinds=self.kinds,
                            layers=self.layers, **{f"delta_{name}": v for name, v in delta_all.items()},
                            **{f"base_{name}": v for name, v in result["base"].items()})
        print(f"Saved {path}  (delta_*: documents x intervention layer x direction x signed alpha x measurement "
              f"layer 1..L, change against alpha = 0, 0 below the steering layer; base_*: documents x layer, "
              f"clean values)")
        path = os.path.join(out_dir, "directions.npz")
        np.savez_compressed(path, kinds=self.kinds, seeds=np.asarray([args.seed + l for l in intervene]),
                            **{f"layer_{l}": dirs[l].astype(np.float32) for l in intervene})
        print(f"Saved {path}")

        path = os.path.join(out_dir, "specificity_summary.json")
        with open(path, "w") as f:
            json.dump({
                "config": {
                    "model": args.model, "model_id": self.inference.model_id, "dtype": str(self.dtype),
                    "dataset": args.dataset, "seed": args.seed, "n_tokens": "full documents", "min_tokens": MIN_TOKENS, "probe_split": probe_split,
                    "layer_convention": "layer l = output of decoder layer l (1..L), before the final norm",
                    "alphas": list(SPEC_ALPHAS), "signed_alphas": list(SPEC_SIGNED),
                    "alpha_scale": "alpha x mean train token norm at the steering layer (position 0 excluded)",
                    "token_norm": train["norms"].tolist(), "target_layers": targets,
                    "intervene_layers": intervene, "probe_layers": self.n_probe,
                    "steer_tokens": f"positions 1..T-1 with clean norm <= {NORM_OUTLIER:g} x the document's median "
                                    f"token norm at the steering layer; same direction for every token and document",
                    "n_random": N_RANDOM,
                    "directions": {"probe": "w_raw / ||w_raw|| of the probe at the steering layer",
                                   "random": "isotropic Gaussian, normalised; numpy default_rng(seed + layer)"},
                    "max_docs": args.max_docs, "batch_size": args.batch_size,
                    "n_test": {"human": int((labels == 0).sum()), "machine": int((labels == 1).sum())},
                },
                "probe": {"name": self.probe, "features": PROBE_FEATURES[self.probe], "C": PROBE_C,
                          "train_auroc": probes["train_auroc"].tolist()},
                "train_clean_means": separation,
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
    parser.add_argument("--plot_alpha", type=float, default=0.1,
                        help="|alpha| for the layer-wise figures (steer) or the histograms (specificity)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="specificity: rows per re-run of layers l+1..L")
    parser.add_argument("--max_docs", type=int, default=None, help="specificity: stratified subset of test documents")
    args = parser.parse_args()
    args.model_name = args.model
    return args


def main() -> None:
    args = parse_args()
    # one model and train pass; per probe: steering first, then the random-direction control
    experiment = DirectionSpecificity(args)
    prep = experiment.prepare()
    for probe in PROBES:
        print(f"\n{'=' * 100}\nProbe: {probe} ({PROBE_FEATURES[probe]})\n{'=' * 100}")
        experiment.probe = probe
        run = dict(prep, probes=prep["probes"][probe], out_dir=os.path.join(prep["out_dir"], probe))
        os.makedirs(run["out_dir"], exist_ok=False)
        experiment.run_steer(run)
        experiment.run_specificity(run)


if __name__ == "__main__":
    main()
