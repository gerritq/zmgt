"""
One saved pair of src.inter_sample (output/inter/inter_sample_*.json), human text left and machine text right, every
token shaded by its joint score √(θ φ) at the saved layer (θ intra, φ inter angle, radians). Both panels share one
colour scale (single-hue sequential, light = low, dark = high; clipped to the 2nd..98th percentile of the pair's
tokens). Tokens without a score (t = 0) are grey with a dashed outline.

Output: output/items/f_inter_sample_{pair_id}.pdf. --rank picks the pair (1 = first saved = largest |divergence|).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

from src.config import Config
from src.items.plot_style import configure_plot_style

cfg = Config()

INTER_DIR = Path(cfg.base_dir) / "output" / "inter"
OUTPUT_DIR = Path(cfg.item_output_dir)
DEFAULT_INPUT = INTER_DIR / "inter_sample_l8b_drlXpaired_wiki_gpt_4o_256_tokens_s42.json"

CMAP = plt.get_cmap("Blues")
NO_SCORE = "#E3E6EA"
INK, INK_ON_DARK = "#1F2328", "#FFFFFF"
FONT_SIZE = 9.5          # points
PANEL_WIDTH = 7.2        # inches of text per panel
LINE_HEIGHT = 15.0       # points
PAD = 1.2                # points of box padding on each side of a token
PANEL_GAP = 0.45         # inches between the panels


def layout(tokens: list[dict], font: FontProperties, renderer, dpi: float, width: float) -> tuple[list[dict], int]:
    """Place tokens left to right with wrapping (points from the top left). Returns the boxes and the number of rows."""
    space = renderer.get_text_width_height_descent("a a", font, ismath=False)[0] \
        - renderer.get_text_width_height_descent("aa", font, ismath=False)[0]
    space *= 72 / dpi
    boxes, x, row = [], 0.0, 0
    for token in tokens:
        raw = token["token"]
        breaks = raw.count("\n")
        if breaks:
            row += min(breaks, 2)  # a blank line at most
            x = 0.0
        label = raw.strip()
        gap = space if raw[:1] == " " and x > 0 else 0.0
        if not label:
            x += gap
            continue
        w = renderer.get_text_width_height_descent(label, font, ismath=False)[0] * 72 / dpi + 2 * PAD
        if x + gap + w > width and x > 0:
            row, x, gap = row + 1, 0.0, 0.0
        boxes.append({"label": label, "x": x + gap, "w": w, "row": row, "joint": token["joint"]})
        x += gap + w
    return boxes, row + 1


def draw(ax, boxes: list[dict], rows: int, width: float, norm: Normalize, font: FontProperties) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(rows * LINE_HEIGHT, 0)
    ax.axis("off")
    for box in boxes:
        y = box["row"] * LINE_HEIGHT
        if box["joint"] is None:
            face, ink, edge = NO_SCORE, INK, dict(edgecolor="#8C959F", linestyle=(0, (2, 1.5)), linewidth=0.6)
        else:
            face = CMAP(norm(box["joint"]))
            luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
            ink, edge = (INK_ON_DARK if luminance < 0.5 else INK), dict(linewidth=0)
        ax.add_patch(Rectangle((box["x"] + 0.3, y + 1.5), box["w"] - 0.6, LINE_HEIGHT - 3, facecolor=face, **edge))
        ax.text(box["x"] + box["w"] / 2, y + LINE_HEIGHT / 2, box["label"], ha="center", va="center",
                color=ink, fontproperties=font, parse_math=False)


def main() -> None:
    args = parse_args()
    configure_plot_style()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    pair = next(p for p in data["top_pairs"] if p["rank"] == args.rank)
    layer = int(data["selected_layer"].removeprefix("layer_"))

    values = np.asarray([t["joint"] for side in ("human", "machine") for t in pair[side]["tokens"]
                         if t["joint"] is not None])
    norm = Normalize(*np.percentile(values, [2, 98]), clip=True)

    font = FontProperties(family=plt.rcParams["font.family"], size=FONT_SIZE)
    width = PANEL_WIDTH * 72
    probe = plt.figure()
    renderer = probe.canvas.get_renderer()
    laid = {side: layout(pair[side]["tokens"], font, renderer, probe.dpi, width) for side in ("human", "machine")}
    plt.close(probe)

    rows = max(r for _, r in laid.values())
    text_h = rows * LINE_HEIGHT / 72
    top, bottom, side_margin = 0.85, 1.25, 0.15  # inches: titles above, colour bar below
    fig_w = 2 * PANEL_WIDTH + PANEL_GAP + 2 * side_margin
    fig_h = top + text_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, (side, name) in enumerate((("human", "Human"), ("machine", "Machine"))):
        boxes, own_rows = laid[side]
        left = side_margin + i * (PANEL_WIDTH + PANEL_GAP)
        h = own_rows * LINE_HEIGHT / 72
        ax = fig.add_axes((left / fig_w, (fig_h - top - h) / fig_h, PANEL_WIDTH / fig_w, h / fig_h))
        draw(ax, boxes, own_rows, width, norm, font)
        fig.text((left + PANEL_WIDTH / 2) / fig_w, (fig_h - top + 0.12) / fig_h,
                 f"{name}: joint = {pair[side]['joint']:.3f} rad ({pair[side]['n_tokens']} tokens)",
                 ha="center", va="bottom", fontsize=plt.rcParams["axes.titlesize"])

    fig.suptitle(f"{data['model']}, {data['dataset']}, pair {pair['pair_id']} (rank {pair['rank']}): "
                 f"joint divergence {pair['divergence']:+.3f} rad, transformer layer {layer + 1}",
                 y=1 - 0.12 / fig_h, va="top")
    bar_w = 4.0
    cax = fig.add_axes(((fig_w - bar_w) / 2 / fig_w, 0.75 / fig_h, bar_w / fig_w, 0.14 / fig_h))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal",
                       extend="both")
    bar.outline.set_linewidth(0)
    bar.set_label(r"Per-token joint $\sqrt{\theta\,\phi}$ (rad); grey: no score (first token)",
                  fontsize=plt.rcParams["legend.fontsize"])
    bar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"] - 2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"f_inter_sample_{pair['pair_id']}.pdf"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="inter_sample json")
    parser.add_argument("--rank", type=int, default=1, help="pair to plot (1 = largest |divergence|)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
