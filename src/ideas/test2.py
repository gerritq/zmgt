"""Candidate-calibrated hidden-state geometry.

For each text, tokenize the untouched text, use the first j token IDs as a
prefix, and the next T IDs as the real continuation. Sample k continuations
of exactly T tokens from the scoring model. A batched forward pass over the
real and sampled sequences keeps only continuation hidden states H_i at the
requested layers. Nothing is written to disk; results are printed only.
"""

from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from src.inference import Inference
from src.utils import evaluation, load_data


class CandidateCalibration:
    """Compare a real continuation with samples drawn from the same prefix."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(model_name=args.model, args=args)
        self.model = self.inference.model
        self.tokenizer = self.inference.tokenizer
        self.device = self.inference.device
        self.layers = self.resolve_layers(args.layers)
        self.ranks = tuple(sorted(set(args.ranks)))
        self.global_directions: dict[int, torch.Tensor] = {}

        if args.n_candidates < 3:
            raise ValueError("--n_candidates must be at least 3 for leave-one-out calibration.")
        if not 0.0 < args.prefix_fraction < 1.0:
            raise ValueError("--prefix_fraction must be strictly between 0 and 1.")
        if args.max_tokens < 3:
            raise ValueError("--max_tokens must be at least 3.")
        if any(rank < 1 for rank in self.ranks):
            raise ValueError("--ranks must contain positive integers.")
        if max(self.ranks) > self.model.config.hidden_size:
            raise ValueError("A requested --ranks value exceeds the model hidden size.")

    def resolve_layers(self, layers: list[int] | None) -> list[int]:
        n_layers = self.model.config.num_hidden_layers
        chosen = list(range(n_layers + 1)) if layers is None else layers
        if any(layer < 0 or layer > n_layers for layer in chosen):
            raise ValueError(f"--layers must be hidden-state indices in 0..{n_layers}.")
        return sorted(set(chosen))

    def split_point(self, ids: torch.Tensor) -> int:
        """Prefix length: floor(prefix_fraction times the tokenized text length)."""
        return max(1, int(ids.numel() * self.args.prefix_fraction))

    def token_ids(self, text: str) -> torch.Tensor | None:
        """Tokenize the full text (up to max_tokens), preserving its continuation length."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.args.max_tokens,
            return_tensors="pt",
        )["input_ids"][0]
        if encoded.numel() < 3:
            return None
        continuation = encoded.numel() - self.split_point(encoded)
        min_continuation = max(
            2,
            (max(self.ranks) + self.args.n_candidates - 2) // (self.args.n_candidates - 1),
        )
        return encoded.to(self.device) if continuation >= min_continuation else None

    @torch.no_grad()
    def continuation_states(
        self, ids: torch.Tensor, item_index: int
    ) -> dict[int, torch.Tensor]:
        """Return {layer: (1+k, T, D)} for real row 0 and sampled rows 1..k."""
        j = self.split_point(ids)
        t = ids.numel() - j
        k = self.args.n_candidates
        prefix = ids[:j][None]
        real = ids[j : j + t]

        # Reset per text so sampling is reproducible independently of skipped texts.
        seed = self.args.seed + item_index
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        generated = self.model.generate(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            do_sample=True,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            min_new_tokens=t,
            max_new_tokens=t,
            num_return_sequences=k,
            pad_token_id=pad_id,
            use_cache=True,
        )
        candidates = generated[:, j : j + t]
        if candidates.shape != (k, t):
            raise RuntimeError(
                f"Generation returned {tuple(candidates.shape)} candidate tokens; expected {(k, t)}."
            )

        prefix_batch = prefix.expand(k + 1, -1)
        continuations = torch.cat([real[None], candidates], dim=0)
        sequences = torch.cat([prefix_batch, continuations], dim=1)
        outputs = self.model(
            input_ids=sequences,
            attention_mask=torch.ones_like(sequences),
            output_hidden_states=True,
            use_cache=False,
        )
        return {
            layer: outputs.hidden_states[layer][:, j : j + t].detach()
            for layer in self.layers
        }

    @staticmethod
    def shared_share(h: torch.Tensor) -> torch.Tensor:
        """||mean_t h_t||^2 / mean_t ||h_t||^2 for each sequence: (B,T,D)->(B,)."""
        numerator = h.mean(dim=1).square().sum(dim=-1)
        denominator = h.square().sum(dim=-1).mean(dim=1)
        return numerator / denominator.clamp_min(1e-12)

    @staticmethod
    def mean_angle(h: torch.Tensor) -> torch.Tensor:
        """Mean consecutive-token angle for each sequence."""
        cosine = torch.nn.functional.cosine_similarity(h[:, :-1], h[:, 1:], dim=-1, eps=1e-12)
        return torch.acos(cosine.clamp(-1.0, 1.0)).mean(dim=1)

    @staticmethod
    def calibrated_z(real: torch.Tensor, candidates: torch.Tensor) -> float:
        """(real - candidate mean) / candidate sample SD."""
        scale = candidates.std(correction=1)
        if not torch.isfinite(scale) or scale <= 1e-12:
            return float("nan")
        return ((real - candidates.mean()) / scale).item()

    @staticmethod
    def align(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Resolve a singular vector's sign using the pooled candidate mean."""
        return -vector if torch.dot(vector, reference) < 0 else vector

    @staticmethod
    def top_directions(
        h: torch.Tensor, rank: int, gram: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Top uncentred right-singular directions as orthonormal rows.

        The token Gram matrix is used when T <= D, which is much smaller than
        forming a D x D covariance for the candidate pools used here.
        """
        n, d = h.shape
        rank = min(rank, n, d)
        if n <= d:
            gram = h @ h.T if gram is None else gram
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            values = eigenvalues[-rank:].flip(0).clamp_min(1e-12)
            left = eigenvectors[:, -rank:].flip(1)
            directions = (left.T @ h) / values.sqrt()[:, None]
        else:
            gram_d = h.T @ h
            _, eigenvectors = torch.linalg.eigh(gram_d)
            directions = eigenvectors[:, -rank:].flip(1).T
        # Remove small numerical loss of orthogonality from the Gram route.
        return torch.linalg.qr(directions.T, mode="reduced").Q.T

    def candidate_bases(
        self, candidates: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Top max(r) bases for all candidates and each leave-one-out pool."""
        k, t, d = candidates.shape
        pooled = candidates.reshape(k * t, d)
        rank = max(self.ranks)
        full = self.top_directions(pooled, rank)
        loo = []
        for candidate in range(k):
            h_loo = torch.cat([candidates[:candidate], candidates[candidate + 1:]], dim=0)
            loo.append(self.top_directions(h_loo.reshape((k - 1) * t, d), rank))
        return full, loo

    @staticmethod
    def alignment(h: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Mean token cosine with a direction."""
        return torch.nn.functional.cosine_similarity(
            h, direction[None], dim=-1, eps=1e-12
        ).mean()

    @staticmethod
    def subspace_share(h: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """Mean per-token energy fraction in the row span of directions."""
        projected = (h @ directions.T).square().sum(dim=-1)
        total = h.square().sum(dim=-1)
        return (projected / total.clamp_min(1e-12)).mean()

    def direction_scores(
        self,
        h: torch.Tensor,
        full_basis: torch.Tensor,
        loo_bases: list[torch.Tensor],
    ) -> dict[str, float]:
        """Options 2, 3, and the global/local analysis for one layer."""
        real, candidates = h[0], h[1:]
        k = candidates.shape[0]
        candidate_mean = candidates.mean(dim=(0, 1))

        # Option 2a: the mean of all candidate tokens, with exact LOO means.
        m_mean = torch.nn.functional.normalize(candidate_mean, dim=0)
        real_mean_alignment = self.alignment(real, m_mean)
        candidate_mean_alignments = []
        candidate_sums = candidates.sum(dim=(0, 1))
        denom = (k - 1) * candidates.shape[1]
        for i in range(k):
            m_loo = (candidate_sums - candidates[i].sum(dim=0)) / denom
            m_loo = torch.nn.functional.normalize(m_loo, dim=0)
            candidate_mean_alignments.append(self.alignment(candidates[i], m_loo))
        scores = {
            "alignment_mean_z": self.calibrated_z(
                real_mean_alignment, torch.stack(candidate_mean_alignments)
            )
        }

        # Option 2b: top uncentred singular direction, sign-aligned to pool mean.
        m_svd = self.align(full_basis[0], candidate_mean)
        real_svd_alignment = self.alignment(real, m_svd)
        candidate_svd_alignments = []
        loo_svd_directions = []
        for i in range(k):
            reference = (
                candidate_sums - candidates[i].sum(dim=0)
            ) / denom
            direction = self.align(loo_bases[i][0], reference)
            loo_svd_directions.append(direction)
            candidate_svd_alignments.append(self.alignment(candidates[i], direction))
        scores["alignment_svd_z"] = self.calibrated_z(
            real_svd_alignment, torch.stack(candidate_svd_alignments)
        )

        # Option 3: fraction of each token's energy in the candidate subspace.
        for rank in self.ranks:
            real_share = self.subspace_share(real, full_basis[:rank])
            candidate_shares = torch.stack(
                [
                    self.subspace_share(candidates[i], loo_bases[i][:rank])
                    for i in range(k)
                ]
            )
            scores[f"subspace_r{rank}_z"] = self.calibrated_z(real_share, candidate_shares)

        # Option 5: split each local SVD direction into global and prefix-specific parts.
        if self.global_directions:
            g1 = self.global_directions[self.current_layer]
            global_direction = torch.sign(torch.dot(m_svd, g1)).clamp(min=-1, max=1) * g1
            if global_direction.norm() <= 1e-12:
                global_direction = g1
            residual = m_svd - torch.dot(m_svd, g1) * g1
            scores["local_global_abs_cos"] = torch.dot(m_svd, g1).abs().item()

            real_global = self.alignment(real, global_direction)
            real_specific = (
                self.alignment(real, torch.nn.functional.normalize(residual, dim=0))
                if residual.norm() > 1e-8
                else torch.tensor(float("nan"), device=h.device)
            )
            candidate_global, candidate_specific = [], []
            for i, direction in enumerate(loo_svd_directions):
                signed_global = torch.sign(torch.dot(direction, g1)).clamp(min=-1, max=1) * g1
                if signed_global.norm() <= 1e-12:
                    signed_global = g1
                specific = direction - torch.dot(direction, g1) * g1
                candidate_global.append(self.alignment(candidates[i], signed_global))
                candidate_specific.append(
                    self.alignment(
                        candidates[i], torch.nn.functional.normalize(specific, dim=0)
                    )
                    if specific.norm() > 1e-8
                    else torch.tensor(float("nan"), device=h.device)
                )
            scores["alignment_svd_global_z"] = self.calibrated_z(
                real_global, torch.stack(candidate_global)
            )
            scores["alignment_svd_specific_z"] = self.calibrated_z(
                real_specific, torch.stack(candidate_specific)
            )
        return scores

    def score_layer(self, h: torch.Tensor, layer: int) -> dict[str, float]:
        """All candidate-calibrated scores for one (1+k,T,D) tensor."""
        h = h.float()
        self.current_layer = layer
        real, candidates = h[0], h[1:]

        # Option 1 and inexpensive scalar controls.
        shares = self.shared_share(h)
        norms = h.norm(dim=-1).mean(dim=1)
        angles = self.mean_angle(h)
        combined = shares.clamp_min(1e-12).log() - angles.clamp_min(1e-12).log()
        scores = {
            "shared_share_z": self.calibrated_z(shares[0], shares[1:]),
            "mean_norm_z": self.calibrated_z(norms[0], norms[1:]),
            "mean_angle_z": self.calibrated_z(angles[0], angles[1:]),
            "shared_angle_z": self.calibrated_z(combined[0], combined[1:]),
        }

        # Option 4: real distance to all-candidate centroid versus candidate LOO distances.
        centroids = h.mean(dim=1)
        candidate_sum = centroids[1:].sum(dim=0)
        candidate_centre = candidate_sum / candidates.shape[0]
        real_distance = (centroids[0] - candidate_centre).norm()
        loo_distances = torch.stack(
            [
                (
                    centroids[i + 1]
                    - (candidate_sum - centroids[i + 1]) / (candidates.shape[0] - 1)
                ).norm()
                for i in range(candidates.shape[0])
            ]
        )
        scores["centroid_distance_z"] = self.calibrated_z(real_distance, loo_distances)

        full_basis, loo_bases = self.candidate_bases(candidates)
        scores.update(self.direction_scores(h, full_basis, loo_bases))
        return scores

    @torch.no_grad()
    def fit_global(self, data: list[dict]) -> None:
        """Fit corpus-wide uncentred g1 on real continuation states, per layer."""
        grams: dict[int, torch.Tensor] = {}
        totals: dict[int, torch.Tensor] = {}
        n_fit = 0
        for item in tqdm(data, desc="Fitting global g1 on real continuations"):
            ids = self.token_ids(item["text"])
            if ids is None:
                continue
            j = self.split_point(ids)
            outputs = self.model(
                input_ids=ids[None],
                attention_mask=torch.ones_like(ids[None]),
                output_hidden_states=True,
                use_cache=False,
            )
            for layer in self.layers:
                h = outputs.hidden_states[layer][0, j:].float()
                if layer not in grams:
                    grams[layer] = torch.zeros(
                        h.shape[1], h.shape[1], device=h.device, dtype=torch.float32
                    )
                    totals[layer] = torch.zeros(h.shape[1], device=h.device, dtype=torch.float32)
                grams[layer].addmm_(h.T, h)
                totals[layer] += h.sum(dim=0)
            n_fit += 1

        if not n_fit:
            raise ValueError("No texts were long enough to fit global directions.")
        for layer, gram in grams.items():
            direction = torch.nn.functional.normalize(totals[layer], dim=0)
            for _ in range(50):
                direction = torch.nn.functional.normalize(gram @ direction, dim=0)
            self.global_directions[layer] = direction
        print(f"Global g1 fitted on {n_fit} real continuations from {self.args.fit_split}.")

    @staticmethod
    def evaluate(labels: np.ndarray, values: np.ndarray, sign: int) -> dict | None:
        valid = np.isfinite(values)
        if valid.sum() < 2 or np.unique(labels[valid]).size < 2:
            return None
        result = evaluation(labels[valid], sign * values[valid])
        result["n_scored"] = int(valid.sum())
        result["mean_z_human"] = float(values[valid & (labels == 0)].mean())
        result["mean_z_machine"] = float(values[valid & (labels == 1)].mean())
        return result

    def run(self, args: Namespace) -> dict:
        splits = load_data(args=args)
        test_data = splits["test"]
        labels_all = np.asarray([item["label"] for item in test_data])

        if args.global_analysis:
            if args.fit_split not in splits:
                raise ValueError(f"--fit_split {args.fit_split!r} is unavailable.")
            self.fit_global(splits[args.fit_split])

        per_layer: dict[int, list[dict | None]] = {layer: [] for layer in self.layers}
        prefix_lengths, continuation_lengths = [], []
        for index, item in enumerate(tqdm(test_data, desc="Sampling and scoring continuations")):
            ids = self.token_ids(item["text"])
            if ids is None:
                for layer in self.layers:
                    per_layer[layer].append(None)
                continue
            j = self.split_point(ids)
            prefix_lengths.append(j)
            continuation_lengths.append(ids.numel() - j)
            hidden = self.continuation_states(ids, index)
            for layer in self.layers:
                per_layer[layer].append(self.score_layer(hidden[layer], layer))

        signs = {
            "mean_angle_z": -1,
            "centroid_distance_z": -1,
            "local_global_abs_cos": 1,
        }
        results = {}
        n_skipped = sum(row is None for row in per_layer[self.layers[0]])
        print(
            f"\nCandidate calibration | model={args.model} dataset={args.dataset} "
            f"prefix={args.prefix_fraction:.1%} of each text, full remainder sampled "
            f"| k={args.n_candidates} r={list(self.ranks)} | skipped={n_skipped}"
        )
        print(
            f"Observed prefix lengths: {min(prefix_lengths)}..{max(prefix_lengths)} "
            f"(mean {np.mean(prefix_lengths):.1f}); continuation lengths: "
            f"{min(continuation_lengths)}..{max(continuation_lengths)} "
            f"(mean {np.mean(continuation_lengths):.1f})."
        )
        print("AUROC is oriented so higher detector score means machine.")
        print("For mean_angle_z and centroid_distance_z, the raw z-score is negated for AUROC.")

        for layer in self.layers:
            rows = per_layer[layer]
            names = next(row for row in rows if row is not None).keys()
            values = {
                name: np.asarray(
                    [np.nan if row is None else row[name] for row in rows], dtype=float
                )
                for name in names
            }
            metrics = {
                name: self.evaluate(labels_all, value, signs.get(name, 1))
                for name, value in values.items()
            }
            results[layer] = metrics
            print(f"\n=== hidden_states[{layer}] ===")
            print(
                f"{'score':>30} {'human mean':>14} {'machine mean':>16} "
                f"{'AUROC':>9} {'TPR@5%':>9} {'n':>6}"
            )
            for name, metric in metrics.items():
                if metric is None:
                    print(f"{name:>30} {'--':>14} {'--':>16} {'--':>9} {'--':>9} {'0':>6}")
                    continue
                print(
                    f"{name:>30} {metric['mean_z_human']:>14.4f} "
                    f"{metric['mean_z_machine']:>16.4f} {metric['auroc']:>9.4f} "
                    f"{metric['tpr_at_fpr_0_05']:>9.4f} {metric['n_scored']:>6}"
                )
        return results


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefix_fraction", type=float, default=0.1,
        help="Fraction of each tokenized text used as prefix; the remainder is the continuation.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum full-text token count before the 10%/90% split.",
    )
    parser.add_argument("--n_candidates", type=int, default=5, help="Number of sampled continuations k.")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Hidden-state indices; default is every layer, including embeddings and the final layer.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[1, 5, 20],
        help="Candidate subspace ranks.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--global_analysis",
        type=int,
        choices=(0, 1),
        default=1,
        help="Fit global uncentred g1 and print global/prefix-specific alignment scores.",
    )
    parser.add_argument(
        "--fit_split",
        type=str,
        choices=("train", "val", "test"),
        default="train",
        help="Split used for the label-free global g1 analysis.",
    )
    args = parser.parse_args()
    args.global_analysis = bool(args.global_analysis)
    args.model_name = args.model
    return args


def main() -> None:
    args = parse_args()
    CandidateCalibration(args).run(args)


if __name__ == "__main__":
    main()
