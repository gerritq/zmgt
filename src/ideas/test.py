"""Difference-of-means "AI direction" per layer, and every text's tokens projected onto it.

Layers ℓ = 0..L (0 = embeddings, ℓ = output of decoder layer ℓ, taken with forward hooks, so layer L comes before the
final norm). Position 0 (the attention sink, with a norm far above the rest) is left out everywhere.

One forward pass per text; all activations are kept on the CPU in bfloat16 ((L + 1) · N · d · 2 bytes for N tokens,
e.g. ~21 GB for 33 layers, 80k tokens, d = 4096); the class means are summed exactly (float64) during the pass.
Probe, per layer, pooled over all tokens of all texts of the test split:
    μ_M^ℓ, μ_H^ℓ = mean h_t^ℓ over the machine / human tokens,   w^ℓ = (μ_M − μ_H) / ‖μ_M − μ_H‖   (unit AI direction)
Projection, per token t and layer ℓ:
    proj_t = ⟨h_t^ℓ, w^ℓ⟩                       position along the AI direction (higher = more machine-like)
    align_t = ⟨h_t^ℓ, w^ℓ⟩ / ‖h_t^ℓ‖ = cos(h_t, w)  the share of the token's norm that lies along the AI direction
    per text: proj_mean, align_mean = means over its tokens.
    References per layer: midpoint_proj = ⟨(μ_M + μ_H) / 2, w⟩ (the probe's decision boundary), human_proj /
    machine_proj = ⟨μ_H, w⟩ / ⟨μ_M, w⟩ (token-pooled class means).
Five text couples whose proj_mean diverges most at the selection layer (--layer; default: the layer with the highest
AUROC of proj_mean): with pair_id in the data, the pairs with the largest |proj_mean(machine) − proj_mean(human)|;
without, the k-th most machine-like machine text with the k-th most human-like human text, k = 1..5. Their per-token
proj and align at every layer are saved.
Note: the direction is fit on the same texts it scores (in-sample), so AUROC is optimistic.

Output: output/test/dom_probe_{model}_{dataset}_s{seed}.npz (see "meta" for the shapes).
"""

import json
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.utils import load_data

cfg = Config()

OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "test")
MIN_TOKENS = 2  # position 0 is dropped, so a text needs one more token
TOP_K = 5
EPS = 1e-12


class DomProbe:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model, args=args)
        self.model, self.tokenizer = self.inference.model, self.inference.tokenizer
        self.device = self.inference.device
        self.decoder = self.model.get_decoder()
        self.n_layers = self.model.config.num_hidden_layers + 1  # embeddings + decoder layers
        self.states: list[torch.Tensor] = []
        self.decoder.embed_tokens.register_forward_hook(self.record)
        for layer in self.decoder.layers:
            layer.register_forward_hook(self.record)

    def record(self, module, inputs, output) -> None:
        self.states.append(output[0] if isinstance(output, tuple) else output)

    @torch.no_grad()
    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """(T,) token ids -> (L + 1, T − 1, d) float32 states, position 0 dropped."""
        self.states = []
        self.decoder(input_ids=ids[None].to(self.device), use_cache=False)
        h = torch.stack(self.states)[:, 0, 1:].float()
        self.states = []
        return h

    def run(self) -> dict[str, np.ndarray]:
        args = self.args
        data = load_data(args=args)["test"]
        ids = [self.tokenizer(item["text"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
               for item in data]
        labels = np.asarray([int(item["label"]) for item in data])
        used = np.asarray([len(i) >= MIN_TOKENS for i in ids])

        # one forward pass per text: all activations kept (CPU, bfloat16), class sums exact (float64)
        states: list[torch.Tensor | None] = [None] * len(data)
        sums = torch.zeros(2, self.n_layers, self.model.config.hidden_size, dtype=torch.float64, device=self.device)
        counts = np.zeros(2)
        for i in tqdm(np.nonzero(used)[0], desc="Activations"):
            h = self.hidden(ids[i])
            sums[labels[i]] += h.sum(dim=1).double()
            counts[labels[i]] += h.shape[1]
            states[i] = h.bfloat16().cpu()
        del self.model, self.decoder, self.inference
        torch.cuda.empty_cache()

        # difference of means over all tokens, per layer
        mu_h, mu_m = sums[0] / counts[0], sums[1] / counts[1]
        gap = mu_m - mu_h
        w = gap / gap.norm(dim=-1, keepdim=True).clamp_min(EPS)  # (L + 1, d)

        # per-token projection and alignment
        proj = [None] * len(data)
        align = [None] * len(data)
        for i in np.nonzero(used)[0]:
            h = states[i].to(self.device).double()
            states[i] = None
            p = torch.einsum("ltd,ld->lt", h, w)
            nan = torch.full((self.n_layers, 1), float("nan"), dtype=p.dtype, device=p.device)  # position 0
            proj[i] = torch.cat([nan, p], dim=1).cpu().numpy()
            align[i] = torch.cat([nan, p / h.norm(dim=-1).clamp_min(EPS)], dim=1).cpu().numpy()

        empty = np.full(self.n_layers, np.nan)
        proj_mean = np.stack([np.nanmean(p, axis=1) if p is not None else empty for p in proj])
        align_mean = np.stack([np.nanmean(a, axis=1) if a is not None else empty for a in align])
        auroc_proj = np.asarray([auroc(labels, proj_mean[:, l]) for l in range(self.n_layers)])
        auroc_align = np.asarray([auroc(labels, align_mean[:, l]) for l in range(self.n_layers)])
        layer = args.layer if args.layer is not None else int(np.nanargmax(auroc_proj))

        couples, divergence, pair_ids = select(data, labels, proj_mean[:, layer], used)
        max_t = max(len(ids[i]) for c in couples for i in c)
        top_proj = np.full((len(couples), 2, self.n_layers, max_t), np.nan, dtype=np.float32)
        top_align = np.full_like(top_proj, np.nan)
        top_ids = np.full((len(couples), 2, max_t), -1, dtype=np.int64)
        top_tokens = np.full((len(couples), 2, max_t), "", dtype=object)
        for k, couple in enumerate(couples):
            for side, i in enumerate(couple):  # side 0 = human, 1 = machine
                n = len(ids[i])
                top_proj[k, side, :, :n] = proj[i]
                top_align[k, side, :, :n] = align[i]
                top_ids[k, side, :n] = ids[i].numpy()
                top_tokens[k, side, :n] = [self.tokenizer.decode([t]) for t in ids[i].tolist()]

        reference = {
            "midpoint_proj": ((mu_m + mu_h) / 2 * w).sum(dim=-1),
            "human_proj": (mu_h * w).sum(dim=-1),
            "machine_proj": (mu_m * w).sum(dim=-1),
            "gap_norm": gap.norm(dim=-1),
        }
        meta = {
            **vars(args),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "dom_probe",
            "layer_numbering": "index l = layer l: 0 = embeddings, l = output of decoder layer l (layer L before the "
                               "final norm)",
            "selection_layer": layer,
            "selection": "pairs by pair_id, ranked by |divergence|" if pair_ids is not None else
                         "k-th most machine-like machine text with the k-th most human-like human text",
            "token_counts": {"human": int(counts[0]), "machine": int(counts[1])},
            "arrays": {
                "label, used": "(texts,); label 1 = machine; used = scored (>= MIN_TOKENS tokens)",
                "proj_mean, align_mean": "(texts, L + 1) float32, per-text mean over tokens 1..T-1 of <h_t, w> and "
                                         "<h_t, w> / ||h_t||; NaN if not used",
                "auroc_proj, auroc_align": "(L + 1,), label 1 = machine, in-sample",
                "midpoint_proj, human_proj, machine_proj, gap_norm": "(L + 1,), <(mu_M + mu_H)/2, w>, <mu_H, w>, "
                                                                     "<mu_M, w>, ||mu_M - mu_H||",
                "top_text_index": "(5, 2) index in the test split, [:, 0] human, [:, 1] machine",
                "top_pair_id, top_divergence": "(5,), pair_id ('' if unpaired) and proj_mean(machine) - "
                                               "proj_mean(human) at selection_layer",
                "top_proj, top_align": "(5, 2, L + 1, max T) float32, per token; index = token position; NaN at "
                                       "position 0 and after the last token",
                "top_token_ids, top_tokens, top_text": "(5, 2, max T) ids (-1 pad) and decoded strings ('' pad); "
                                                       "(5, 2) texts",
            },
        }
        return {
            "meta": np.asarray(json.dumps(meta)),
            "label": labels,
            "used": used,
            "proj_mean": proj_mean.astype(np.float32),
            "align_mean": align_mean.astype(np.float32),
            "auroc_proj": auroc_proj,
            "auroc_align": auroc_align,
            **{name: v.cpu().numpy() for name, v in reference.items()},
            "top_text_index": np.asarray(couples),
            "top_pair_id": np.asarray(pair_ids if pair_ids is not None else [""] * len(couples)),
            "top_divergence": np.asarray(divergence),
            "top_proj": top_proj,
            "top_align": top_align,
            "top_token_ids": top_ids,
            "top_tokens": top_tokens.astype(str),
            "top_text": np.asarray([[data[i]["text"] for i in c] for c in couples]),
        }


def auroc(labels: np.ndarray, x: np.ndarray) -> float:
    """AUROC over the texts with a value; nan if one class has none."""
    ok = np.isfinite(x)
    if len(np.unique(labels[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[ok], x[ok]))


def select(data: list[dict], labels: np.ndarray, score: np.ndarray, used: np.ndarray
           ) -> tuple[list[tuple[int, int]], list[float], list[str] | None]:
    """TOP_K (human index, machine index) couples whose score diverges most; pair ids if the data are paired."""
    ok = used & np.isfinite(score)
    if all("pair_id" in item for item in data):
        pairs: dict[str, dict[int, int]] = {}
        for i, item in enumerate(data):
            pairs.setdefault(item["pair_id"], {})[int(item["label"])] = i
        rows = [(pid, p[0], p[1]) for pid, p in pairs.items() if {0, 1} <= p.keys() and ok[p[0]] and ok[p[1]]]
        rows.sort(key=lambda r: -abs(score[r[2]] - score[r[1]]))
        rows = rows[:TOP_K]
        return [(h, m) for _, h, m in rows], [float(score[m] - score[h]) for _, h, m in rows], [r[0] for r in rows]
    human = [i for i in np.argsort(score) if ok[i] and labels[i] == 0][:TOP_K]
    machine = [i for i in np.argsort(-score) if ok[i] and labels[i] == 1][:TOP_K]
    couples = [(int(h), int(m)) for h, m in zip(human, machine)]
    return couples, [float(score[m] - score[h]) for h, m in couples], None


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=None,
                        help="layer for picking the top couples (default: highest AUROC of proj_mean)")
    args = parser.parse_args()
    args.model_name = args.model
    return args


def main() -> None:
    args = parse_args()
    arrays = DomProbe(args).run()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"dom_probe_{args.model}_{args.dataset}_s{args.seed}.npz")
    np.savez_compressed(path, **arrays)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
