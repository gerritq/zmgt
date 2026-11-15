"""Text and token states against an L2 logistic-regression "AI direction", per layer (plotted only).

Layers ℓ = 0..L (0 = embeddings, ℓ = output of decoder layer ℓ, taken with forward hooks, so layer L comes before the
final norm). Position 0 (the attention sink) is left out everywhere. One forward pass per text; all token states are kept
on the CPU in bfloat16 ((L + 1) · N · d · 2 bytes for N tokens, e.g. ~21 GB for 33 layers, 80k tokens, d = 4096).

Direction, per layer: an L2-penalised logistic regression (sklearn, C = --C, balanced class weights, intercept) on all
tokens of all texts of the test split, token label = its text's label (1 = machine). The states are divided by the
layer's mean token norm before fitting (one scalar per layer, so the direction is unchanged and C means the same at
every layer). w^ℓ = coef / ‖coef‖ (unit; points to machine).
Scores, with m = mean_t h_t (a text's mean state):
    row 1  cos(m, w)                                         per text
    row 2  ⟨m, w⟩                                            per text, the mean state projected onto w
    row 3  ⟨m, w⟩ / mean_t ‖h_t‖                             per text, that projection as a share of the mean token norm
    row 4  mean over all tokens of a group of ⟨h_t, w⟩ / ‖h_t‖ = cos(h_t, w)   per group, pooled over tokens
Left: human and machine means; right: the gap machine − human. Bands are ±1 SD of the measure: over the texts of a group
(rows 1-3) or over all tokens of a group (row 4); for the gap ±sqrt(SD_human² + SD_machine²), the SD of the difference
between a random machine and a random human text (token).
    row 5  left: per text, the SD of cos(h_t, w) over its tokens (within-text SD); mean over the texts of a group,
           band ±1 SD over texts. right: ICC = between / (between + within) per group, between = variance of the text
           means of cos(h_t, w), within = mean within-text variance (near 1: a text's tokens share one level; near 0:
           the share varies mostly from token to token).
The direction is fit on the same texts it scores (in-sample).
Saved as lr_text_token_{model}_{dataset}_s{seed}.pdf in the main directory.
"""

import os
from argparse import ArgumentParser, Namespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.config import Config
from src.inference import Inference
from src.items.plot_style import configure_plot_style
from src.utils import load_data

cfg = Config()

MIN_TOKENS = 2  # position 0 is dropped, so a text needs one more token
EPS = 1e-12
HUMAN, MACHINE, INK = "#0072B2", "#D55E00", "#24292F"


class TextTokenDirection:
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

    def collect(self, data: list[dict]) -> dict:
        """One forward pass per text: token states (L + 1, N, d) bf16, per-text means and mean token norms."""
        ids = [self.tokenizer(item["text"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
               for item in data]
        keep = [i for i in range(len(data)) if len(ids[i]) >= MIN_TOKENS]
        sizes = np.asarray([len(ids[i]) - 1 for i in keep])
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        d = self.model.config.hidden_size
        states = torch.empty((self.n_layers, int(sizes.sum()), d), dtype=torch.bfloat16)
        means = np.empty((len(keep), self.n_layers, d))
        norms = np.empty((len(keep), self.n_layers))
        for j, i in enumerate(tqdm(keep, desc="Texts")):
            h = self.hidden(ids[i])
            states[:, starts[j]:starts[j] + sizes[j]] = h.bfloat16().cpu()
            means[j] = h.double().mean(dim=1).cpu().numpy()
            norms[j] = h.double().norm(dim=-1).mean(dim=1).cpu().numpy()
        labels = np.asarray([int(data[i]["label"]) for i in keep])
        return {"states": states, "means": means, "norms": norms, "labels": labels,
                "text_of_token": np.repeat(np.arange(len(keep)), sizes)}

    def directions(self, states: torch.Tensor, token_labels: np.ndarray, text_of_token: np.ndarray, n_texts: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per layer: unit LR direction (L + 1, d); per text the sums of cos(h_t, w) and cos² over its tokens, and the
        token count."""
        w = np.empty((self.n_layers, states.shape[2]))
        cos_sum = np.zeros((n_texts, self.n_layers))
        cos_sq_sum = np.zeros((n_texts, self.n_layers))
        for layer in tqdm(range(self.n_layers), desc="Logistic regression"):
            x = states[layer].float().numpy()  # (N, d)
            token_norm = np.linalg.norm(x, axis=1)
            # L2 penalty is sklearn's default (the penalty argument is deprecated since 1.8)
            clf = LogisticRegression(C=self.args.C, class_weight="balanced", max_iter=self.args.max_iter)
            clf.fit(x / token_norm.mean(), token_labels)
            w[layer] = clf.coef_[0] / max(np.linalg.norm(clf.coef_[0]), EPS)
            cos = (x @ w[layer]) / np.maximum(token_norm, EPS)
            np.add.at(cos_sum[:, layer], text_of_token, cos)
            np.add.at(cos_sq_sum[:, layer], text_of_token, cos ** 2)
            del x
        return w, cos_sum, cos_sq_sum, np.bincount(text_of_token, minlength=n_texts)

    def run(self) -> None:
        args = self.args
        data = load_data(args=args)["test"]
        got = self.collect(data)
        del self.model, self.decoder, self.inference
        torch.cuda.empty_cache()
        labels, means, norms = got["labels"], got["means"], got["norms"]
        w, cos_sum, cos_sq_sum, n_tokens = self.directions(got["states"], labels[got["text_of_token"]], got["text_of_token"],
                                               len(labels))
        del got

        proj = np.einsum("nld,ld->nl", means, w)  # (texts, L + 1), <m, w>
        text_rows = (
            (proj / np.maximum(np.linalg.norm(means, axis=2), EPS), r"$\cos(\bar h, w)$"),
            (proj, r"$\langle \bar h, w \rangle$"),
            (proj / norms, r"$\langle \bar h, w \rangle \,/\, \overline{\Vert h_t\Vert}$"),
        )
        layers = np.arange(self.n_layers)
        configure_plot_style()
        fig, axes = plt.subplots(5, 2, figsize=(13, 22.5))
        letters = iter("abcdefgh")
        for (score, name), (ax_group, ax_gap) in zip(text_rows, axes):
            human, machine = score[labels == 0], score[labels == 1]
            for values, group, color in ((human, "Human", HUMAN), (machine, "Machine", MACHINE)):
                mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                draw(ax_group, layers, mean, mean - sd, mean + sd, color, group)
            diff = machine.mean(axis=0) - human.mean(axis=0)
            sd = np.sqrt(machine.var(axis=0, ddof=1) + human.var(axis=0, ddof=1))
            draw(ax_gap, layers, diff, diff - sd, diff + sd, INK, "Machine − human")
            ax_group.set(title=f"({next(letters)}) Mean {name} per group (over texts)", ylabel=f"mean {name}")
            ax_gap.set(title=f"({next(letters)}) Gap: machine − human", ylabel=rf"$\Delta$ mean {name}")

        # row 4: mean and SD of cos(h_t, w) over all tokens of a group
        name = r"$\langle h_t, w \rangle / \Vert h_t\Vert$"
        mean, var = {}, {}
        for c in (0, 1):
            n = n_tokens[labels == c].sum()
            mean[c] = cos_sum[labels == c].sum(axis=0) / n
            var[c] = (cos_sq_sum[labels == c].sum(axis=0) - n * mean[c] ** 2) / (n - 1)
        ax_group, ax_gap = axes[3]
        for c, group, color in ((0, "Human", HUMAN), (1, "Machine", MACHINE)):
            sd = np.sqrt(var[c])
            draw(ax_group, layers, mean[c], mean[c] - sd, mean[c] + sd, color, group)
        diff, sd = mean[1] - mean[0], np.sqrt(var[0] + var[1])
        draw(ax_gap, layers, diff, diff - sd, diff + sd, INK, "Machine − human")
        ax_group.set(title=f"(g) Mean {name} per group (over all tokens)", ylabel=f"mean {name}")
        ax_gap.set(title="(h) Gap: machine − human", ylabel=rf"$\Delta$ mean {name}")

        # row 5: spread of cos(h_t, w) within a text, and the share of the variance that lies between texts
        with np.errstate(invalid="ignore", divide="ignore"):
            text_mean = cos_sum / n_tokens[:, None]
            within_var = (cos_sq_sum - n_tokens[:, None] * text_mean ** 2) / (n_tokens[:, None] - 1)
        within_sd = np.sqrt(np.clip(within_var, 0, None))  # (texts, L + 1); NaN for one-token texts
        ax_within, ax_icc = axes[4]
        for c, group, color in ((0, "Human", HUMAN), (1, "Machine", MACHINE)):
            sd_c = within_sd[labels == c]
            mean, spread = np.nanmean(sd_c, axis=0), np.nanstd(sd_c, axis=0, ddof=1)
            draw(ax_within, layers, mean, mean - spread, mean + spread, color, group)
            between = np.nanvar(text_mean[labels == c], axis=0, ddof=1)
            within = np.nanmean(within_var[labels == c], axis=0)
            ax_icc.plot(layers, between / (between + within), color=color, lw=1.6, marker="o", markersize=3.5,
                        label=group)
        ax_within.set(title=f"(i) Within-text SD of {name}", ylabel="SD over a text's tokens")
        ax_icc.set(title="(j) Share of the variance between texts (ICC)", ylabel="between / (between + within)",
                   ylim=(0, 1))

        for ax in axes.flat:
            ax.axhline(0, color="#57606A", lw=0.8)
            ax.set_xlabel("Layer (0 = embeddings)")
            ax.grid(axis="y", color="#E2E6EA", lw=0.6)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, title=None if ax is ax_icc else "band: ±1 SD over texts" if ax is ax_within
                      else "band: ±1 SD", title_fontsize=11)
        fig.suptitle(f"L2 logistic-regression direction (C = {args.C}), {args.model}, {args.dataset} "
                     f"({(labels == 0).sum()} + {(labels == 1).sum()} texts)")
        fig.tight_layout()
        path = os.path.join(cfg.base_dir, f"lr_text_token_{args.model}_{args.dataset}_s{args.seed}.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"Wrote {path}")


def draw(ax, x: np.ndarray, y: np.ndarray, low: np.ndarray, high: np.ndarray, color: str, label: str) -> None:
    ax.fill_between(x, low, high, color=color, alpha=0.25, lw=0)
    ax.plot(x, y, color=color, lw=1.6, marker="o", markersize=3.5, label=label)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the logistic regression")
    parser.add_argument("--max_iter", type=int, default=1000, help="lbfgs iterations of the logistic regression")
    args = parser.parse_args()
    args.model_name = args.model
    return args


if __name__ == "__main__":
    TextTokenDirection(parse_args()).run()
