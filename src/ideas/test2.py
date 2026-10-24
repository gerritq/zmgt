"""Layer-update features measured against the embedding and the previous token.

With e_t = hidden_states[0] (embeddings), h_t^(l) = hidden_states[l] and the
layer update delta_t^(l) = h_t^(l) - h_t^(l-1), for each text and layer l >= 1:
  E: mean_t angle(h_t^(l), e_t)
       how far the token's direction has moved from its embedding
  A: mean_t angle(delta_t^(l), delta_{t-1}^(l))
       whether the layer moves neighbouring tokens in similar directions
  B: mean_t angle(h_t^(l) - e_t, h_t^(l-1) - e_t), l >= 2
       whether the displacement from the embedding keeps its direction
  M: mean_t ||delta_t^(l) - delta_{t-1}^(l)|| / (||h_t^(l-1) - h_{t-1}^(l-1)|| + eps)
       how much the layer changes a token's position relative to its neighbour
Each angle feature X also has a magnitude twin Xn: mean_t ||a - b|| for the same
two vectors a, b (En: ||h_t^(l) - e_t||, An: ||delta_t - delta_{t-1}||,
Bn: ||h_t^(l) - h_t^(l-1)|| = ||delta_t^(l)||).
Variants:
  Er:  mean_t ||h_t^(l) - e_t|| / ||h_t^(l)||, En without the growth of norms
  S:   mean_t ||h_t^(l) - e_t|| / sum_{j=1..l} ||delta_t^(j)||, l >= 2
       straightness of the path from the embedding (1 = straight line)
  Bk:  B with a gap of k layers, angle(h_t^(l) - e_t, h_t^(l-k) - e_t), l >= k + 1
  EnB: En / B, per text
Within / across angles, with theta_t = angle(h_t^(l), h_{t-1}^(l)) and
phi_t = angle(h_t^(l), h_t^(l-1)) (layer 1 against the embeddings):
  within:        mean_t theta_t, the step to the previous token in the same layer
  across:        mean_t phi_t, the step from the previous layer for the same token
  joint:         mean_t sqrt(theta_t * phi_t), the per-token geometric mean
  joint_means:   sqrt(mean_t theta_t * mean_t phi_t), the geometric mean of the means
  joint_lagk:    joint with the within angle lagged by k tokens,
                 angle(h_t^(l), h_{t-k}^(l))
Displacement from the embedding, with E_t^(l) = angle(h_t^(l), e_t):
  embed_disp:     mean_t ||h_t^(l) - e_t|| / ||e_t||, the change in direction and
                  magnitude relative to the embedding's size; at the last layer
                  (the final state before the logits) this is the endpoint
                  displacement D = sqrt((r - 1)^2 + 2r(1 - cos E)), r = ||f_t|| / ||e_t||
  embed_progress: mean_t (E_t^(l) - E_t^(l-1)) / phi_t, l >= 2, the share of the
                  layer's rotation that moves the token away from its embedding
                  (+: further away, -: back towards it); tokens with
                  phi_t < MIN_ROTATION are left out, nan if none are left
All features use the same tokens, t >= skip (the first --skip_tokens tokens,
at most half the text, at least 1, are dropped), except the joint family
(joint, joint_means, joint_lagk), which all use t >= max(skip, max lag) so
they are averaged over the same tokens; texts too short for that are nan.
No hypothesis on direction, so each
(feature, layer) AUROC is flipped to >= 0.5 and its sign printed (+1: machine
higher, -1: human higher). Results are printed only.
"""

from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from src.inference import Inference
from src.utils import evaluation, load_data

GAPS = (2, 4)
LAGS = (5, 10, 20)
FEATURES = (
    "E", "En", "Er", "S", "A", "An", "B", *(f"B{k}" for k in GAPS), "Bn", "EnB", "M",
    "within", "across", "embed_disp", "embed_progress",
    "joint", "joint_means", *(f"joint_lag{k}" for k in LAGS),
)
EPS = 1e-6
# radians; embed_progress leaves out tokens the layer barely rotates
MIN_ROTATION = 1e-3
SUMMARY_COLUMNS = 8
WIDTH = max(8, *(len(name) for name in FEATURES))


def angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)
    return torch.acos(cos.clamp(-1.0, 1.0))


class LayerUpdates:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model, args=args)
        self.model = self.inference.model
        self.tokenizer = self.inference.tokenizer
        self.device = self.inference.device
        self.n_layers = self.model.config.num_hidden_layers
        layers = list(range(1, self.n_layers + 1)) if args.layers is None else args.layers
        if any(layer < 1 or layer > self.n_layers for layer in layers):
            raise ValueError(f"--layers must be hidden-state indices in 1..{self.n_layers}.")
        self.layers = sorted(set(layers))

    def scores(self, hs: torch.Tensor) -> dict[str, np.ndarray]:
        """All features from (L + 1, T, d) hidden states; each is (L,) for layers 1..L,
        nan where a feature is undefined."""
        hs = hs.float()
        T = hs.shape[1]
        s = max(1, min(self.args.skip_tokens, T // 2))
        e = hs[0]
        # delta[l - 1] = delta^(l) = h^(l) - h^(l-1)
        delta = hs[1:] - hs[:-1]

        def pad(x: torch.Tensor, k: int) -> torch.Tensor:
            """Prepend k nans for the first k layers, where x is undefined."""
            return torch.cat([hs.new_full((k,), float("nan")), x])

        # tokens t >= s; [s - 1:-1] gives their previous token
        # disp[l] = h^(l) - e; disp[0] = 0, so B starts at l = 2
        disp = hs[:, s:] - e[s:]
        dist = disp.norm(dim=-1)
        step = delta[:, s:].norm(dim=-1)
        # E_tok[l - 1] = E_t^(l)
        E_tok = angle(hs[1:, s:], e[s:].expand_as(hs[1:, s:]))
        E = E_tok.mean(dim=-1)
        En = dist[1:].mean(dim=-1)
        Er = (dist[1:] / (hs[1:, s:].norm(dim=-1) + EPS)).mean(dim=-1)
        # at l = 1 the path is a single step, so S = 1
        S = pad((dist[2:] / (step.cumsum(dim=0)[1:] + EPS)).mean(dim=-1), 1)

        moved = (delta[:, s:] - delta[:, s - 1:-1]).norm(dim=-1)
        A = angle(delta[:, s:], delta[:, s - 1:-1]).mean(dim=-1)
        An = moved.mean(dim=-1)

        # Bk[l] compares disp[l] with disp[l - k], for l = k + 1..L
        B = {k: pad(angle(disp[k + 1:], disp[1:-k]).mean(dim=-1), k) for k in (1, *GAPS)}
        Bn = pad(step[1:].mean(dim=-1), 1)

        separation = (hs[:-1, s:] - hs[:-1, s - 1:-1]).norm(dim=-1)
        M = (moved / (separation + EPS)).mean(dim=-1)

        # within: previous token in the same layer; across: same token one layer down
        theta = angle(hs[1:, s:], hs[1:, s - 1:-1])
        phi = angle(hs[1:, s:], hs[:-1, s:])

        embed_disp = (dist[1:] / (e[s:].norm(dim=-1) + EPS)).mean(dim=-1)
        # at l = 1, E^(0) = 0 and E^(1) = phi, so the ratio is 1; start at l = 2
        rotated = phi[1:] >= MIN_ROTATION
        progress = (E_tok[1:] - E_tok[:-1]) / phi[1:].clamp_min(MIN_ROTATION)
        embed_progress = pad(
            (progress * rotated).sum(dim=-1) / rotated.sum(dim=-1), 1
        )

        # the joint family on the same tokens, t >= max(s, max lag)
        start = max(s, *LAGS)
        if start < T:
            phi_j = phi[:, start - s:]
            theta_j = theta[:, start - s:]
            joint = {1: (theta_j * phi_j).sqrt().mean(dim=-1)}
            joint_means = (theta_j.mean(dim=-1) * phi_j.mean(dim=-1)).sqrt()
            for k in LAGS:
                theta_k = angle(hs[1:, start:], hs[1:, start - k:-k])
                joint[k] = (theta_k * phi_j).sqrt().mean(dim=-1)
        else:
            nan = hs.new_full((hs.shape[0] - 1,), float("nan"))
            joint = {k: nan for k in (1, *LAGS)}
            joint_means = nan

        values = (
            E, En, Er, S, A, An, B[1], *(B[k] for k in GAPS), Bn, En / B[1], M,
            theta.mean(dim=-1), phi.mean(dim=-1), embed_disp, embed_progress,
            joint[1], joint_means, *(joint[k] for k in LAGS),
        )
        return {name: value.cpu().numpy() for name, value in zip(FEATURES, values)}

    @torch.no_grad()
    def text_scores(self, text: str) -> dict[str, np.ndarray] | None:
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.args.max_tokens,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        if ids.shape[1] < 3:
            return None
        outputs = self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=True,
            use_cache=False,
        )
        return self.scores(torch.stack(outputs.hidden_states)[:, 0])

    def run(self) -> dict:
        args = self.args
        test_data = load_data(args=args)["test"]
        labels = np.asarray([item["label"] for item in test_data])

        rows = [self.text_scores(item["text"]) for item in tqdm(test_data, desc="Scoring texts")]
        valid = np.asarray([row is not None for row in rows])
        y = labels[valid]
        rows = [row for row in rows if row is not None]

        print(
            f"\nLayer-update features | model={args.model} dataset={args.dataset} "
            f"| scored={valid.sum()} skipped={(~valid).sum()} skip_tokens={args.skip_tokens}"
        )
        print("AUROC is flipped to >= 0.5; sign=+1: machine higher, sign=-1: human higher.")

        results = {name: {} for name in FEATURES}
        for name in FEATURES:
            print(f"\n=== {name} ===")
            print(
                f"{'layer':>6} {'sign':>5} {'human mean':>12} {'machine mean':>14} "
                f"{'AUROC':>8} {'TPR@5%':>8}"
            )
            for layer in self.layers:
                values = np.asarray([row[name][layer - 1] for row in rows])
                # the joint family is nan for texts too short for the lags; score the rest
                keep = ~np.isnan(values)
                values, y_kept = values[keep], y[keep]
                if len(np.unique(y_kept)) < 2:
                    continue
                sign = 1
                metric = evaluation(y_kept, values)
                if metric["auroc"] < 0.5:
                    sign = -1
                    metric = evaluation(y_kept, -values)
                metric["sign"] = sign
                results[name][layer] = metric
                print(
                    f"{layer:>6} {sign:>+5d} {values[y_kept == 0].mean():>12.4f} "
                    f"{values[y_kept == 1].mean():>14.4f} "
                    f"{metric['auroc']:>8.4f} {metric['tpr_at_fpr_0_05']:>8.4f}"
                )

        print("\n=== Summary: AUROC by layer ===")
        for i in range(0, len(FEATURES), SUMMARY_COLUMNS):
            names = FEATURES[i:i + SUMMARY_COLUMNS]
            print(f"\n{'layer':>6} " + " ".join(f"{name:>{WIDTH}}" for name in names))
            for layer in self.layers:
                cells = [
                    f"{results[name][layer]['auroc']:>{WIDTH}.4f}"
                    if layer in results[name] else f"{'-':>{WIDTH}}"
                    for name in names
                ]
                print(f"{layer:>6} " + " ".join(cells))

        print("\n=== Best layer per feature ===")
        for name in FEATURES:
            if not results[name]:
                continue
            layer, metric = max(results[name].items(), key=lambda kv: kv[1]["auroc"])
            print(
                f"{name:>{WIDTH}}: layer {layer:>3}  sign={metric['sign']:+d}  "
                f"AUROC={metric['auroc']:.4f}"
            )
        return results


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--skip_tokens", type=int, default=10)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Hidden-state indices; default is every transformer layer (1..L).",
    )
    args = parser.parse_args()
    args.model_name = args.model
    return args


def main() -> None:
    args = parse_args()
    LayerUpdates(args).run()


if __name__ == "__main__":
    main()
