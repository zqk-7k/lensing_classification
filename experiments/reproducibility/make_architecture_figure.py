#!/usr/bin/env python3
"""Draw the PI-ResNet architecture figure directly from src/classifier/pi_resnet.py.

The previous architecture figure was a legacy raster with no generating source, and it
disagreed with the released code in three ways: stage 3 was drawn with two leading
residual blocks it does not have, the head was labelled a "classification head", and
its output was labelled "output probability" rather than the ranking score the paper
actually defines.

Every label below is taken from the module: `base = int(32 * width_scale) = 128`, so
the stage widths are 128 -> 256 -> 512 -> 1024, the kernel sizes are 15/11/9/7, each
stride-2 convolution halves the time axis for a total factor of 32, the branch
embedding is `base * 8 * 2 = 2048` from concatenated average and max pooling, the
fusion is `[f1 * f2, |f1 - f2|]` giving 4096, and the head is
Linear(4096, 256) -> ReLU -> Dropout(0.7) -> Linear(256, 1). The script asserts the
parameter count against the instantiated model so the caption cannot drift.

Output: results/figures/manuscript/fig_architecture.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

FS = 1.0   # keep text and box geometry in proportion; legibility is bought
           # by giving panel (a) the full page width instead of two thirds

C_STEM = "#dbe7f3"
C_STAGE = "#c7dcef"
C_BLOCK = "#eef3f8"
C_POOL = "#e6eef6"
C_FUSE = "#f7e3dd"
C_HEAD = "#f3e6f2"
C_EDGE = "#7f97ad"
C_ARROW = "#54606d"
C_NOTE = "#4a5260"


def box(ax, x, y, w, h, text, fc=C_BLOCK, ec=C_EDGE, fs=8.4, weight="normal", head=None, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.30,rounding_size=0.9",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if head is None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3, weight=weight, linespacing=1.5)
    else:
        ax.text(x + w / 2, y + h * 0.74, head, ha="center", va="center",
                fontsize=fs + 0.5, weight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.32, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, lw=1.25, style="-|>", color=C_ARROW, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                                 lw=lw, color=color, zorder=4, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def draw_encoder(ax):
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(21.5, 45.5)

    # ---- Siamese inputs -------------------------------------------------
    ax.text(0.5, 43.6, "(a)", fontsize=11, weight="bold", va="center")
    ax.text(4.6, 43.6, "Siamese encoder, weights shared between branches",
            fontsize=9.6, weight="bold", va="center")

    box(ax, 1.5, 33.0, 15.0, 5.2, "whitened segment $d_1$\n$1\\times4096$  (2 s, 2048 Hz)", fc="#ffffff")
    box(ax, 1.5, 25.0, 15.0, 5.2, "whitened segment $d_2$\n$1\\times4096$  (2 s, 2048 Hz)", fc="#ffffff")

    # ---- shared backbone strip -----------------------------------------
    stages = [
        (20.0, "Stem", "Conv1d $1{\\to}128$, k15, s2\nBN1d, ReLU\nMaxPool k3, s2", C_STEM, "$128\\times1024$"),
        (37.5, "Stage 1", "ResBlock k15 $\\times2$\nConv1d $128{\\to}256$, k11, s2\nBN1d, ReLU\nResBlock k11", C_STAGE, "$256\\times512$"),
        (55.0, "Stage 2", "ResBlock k11 $\\times2$\nConv1d $256{\\to}512$, k9, s2\nBN1d, ReLU\nResBlock k9", C_STAGE, "$512\\times256$"),
        (72.5, "Stage 3", "Conv1d $512{\\to}1024$, k9, s2\nBN1d, ReLU\nResBlock k7", C_STAGE, "$1024\\times128$"),
    ]
    for x, title, body, colour, shape in stages:
        box(ax, x, 25.4, 15.0, 12.4, body, fc=colour, fs=7.9, head=title)
        ax.text(x + 7.5, 23.9, shape, ha="center", va="center", fontsize=7.6, color=C_NOTE)

    # bracket showing the shared trunk
    ax.add_patch(Rectangle((19.2, 22.6), 68.6, 16.4, fill=False, ec="#b08d57",
                           lw=1.2, ls=(0, (4, 3)), zorder=1))
    ax.text(53.5, 40.1, "shared weights  —  each branch is encoded independently by this trunk",
            ha="center", fontsize=8.0, style="italic", color="#8a6d3b")

    for y in (35.6, 27.6):
        arrow(ax, 16.5, y, 20.0, y)
    for x in (35.0, 52.5, 70.0):
        arrow(ax, x, 31.6, x + 2.5, 31.6)

    # ---- pooling to the embedding --------------------------------------
    box(ax, 89.0, 25.4, 9.5, 12.4, "global\navg-pool\n$\\oplus$\nmax-pool", fc=C_POOL, fs=7.9)
    arrow(ax, 87.5, 31.6, 89.0, 31.6)



def draw_head(ax):
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 22)
    ax.text(0.5, 19.6, "(b)", fontsize=11, weight="bold", va="center")
    ax.text(4.6, 19.6, "Pairwise-interaction layer and scoring head",
            fontsize=9.6, weight="bold", va="center")

    box(ax, 20.0, 5.4, 24.0, 8.6,
        "$[\\,\\mathbf{f}_1\\odot\\mathbf{f}_2,\\;|\\mathbf{f}_1-\\mathbf{f}_2|\\,]$\n$\\rightarrow\\;4096$",
        fc=C_FUSE, fs=9.0, head="Pairwise interaction")
    box(ax, 50.0, 5.4, 24.0, 8.6,
        "Linear $4096{\\to}256$\nReLU,  Dropout $0.7$\nLinear $256{\\to}1$",
        fc=C_HEAD, fs=8.2, head="Scoring head")
    box(ax, 80.0, 6.6, 18.5, 6.2, "pair score\n$s(d_1,d_2)$", fc="#ffffff", fs=9.4, weight="bold")

    arrow(ax, 44.0, 9.7, 50.0, 9.7)
    arrow(ax, 74.0, 9.7, 80.0, 9.7)

    # embeddings feed down into the fusion box
    ax.plot([93.75, 93.75], [21.0, 17.0], color=C_ARROW, lw=1.25, zorder=1)
    ax.plot([93.75, 32.0], [17.0, 17.0], color=C_ARROW, lw=1.25, zorder=1)
    arrow(ax, 32.0, 17.0, 32.0, 14.0)
    ax.text(63.0, 17.8, "branch embeddings $\\mathbf{f}_1,\\mathbf{f}_2$  (2048-d each)",
            ha="center", fontsize=8.0, color=C_NOTE)

    ax.text(89.25, 4.6, "a ranking statistic, not a calibrated probability;\n"
                        "thresholds are set on the empirical background",
            ha="center", va="top", fontsize=7.2, style="italic", color=C_NOTE, linespacing=1.5)




def draw_residual_block(ax):
    ax.axis("off"); ax.set_xlim(0, 40); ax.set_ylim(-3.2, 45.5)
    ax.text(0.5, 43.6, "(c)", fontsize=11, weight="bold", va="center")
    ax.text(4.6, 43.6, "Residual block", fontsize=9.6, weight="bold", va="center")

    x, w = 9.0, 22.0
    rows = [
        (38.2, "Conv1d $C{\\to}C$, kernel $k$"),
        (34.2, "BatchNorm1d"),
        (30.2, "ReLU"),
        (26.2, "Conv1d $C{\\to}C$, kernel $k$"),
        (22.2, "BatchNorm1d"),
    ]
    for y, text in rows:
        box(ax, x, y, w, 3.0, text, fc=C_BLOCK, fs=8.0)
    box(ax, x, 17.4, w, 3.4, "SE (channel gate)", fc="#e2eddf", fs=8.0)
    box(ax, x, 10.4, w, 3.4, "$\\oplus$   residual add", fc="#ffffff", fs=8.6)
    box(ax, x, 5.4, w, 3.0, "ReLU", fc=C_BLOCK, fs=8.0)

    for y0, y1 in [(38.2, 37.2), (34.2, 33.2), (30.2, 29.2), (26.2, 25.2),
                   (22.2, 20.8), (17.4, 13.8), (10.4, 8.4)]:
        arrow(ax, x + w / 2, y0, x + w / 2, y1)

    # identity skip
    ax.plot([5.4, 5.4], [41.2, 12.1], color=C_ARROW, lw=1.25, ls=(0, (4, 3)), zorder=1)
    ax.plot([5.4, 9.0], [41.2, 41.2], color=C_ARROW, lw=1.25, ls=(0, (4, 3)), zorder=1)
    arrow(ax, 5.4, 12.1, 9.0, 12.1, ls=(0, (4, 3)))
    ax.text(4.6, 27.0, "identity", rotation=90, ha="center", va="center",
            fontsize=7.8, color=C_NOTE, style="italic")

    ax.text(20.0, -2.8, "SE:  GAP$\\rightarrow$Linear $C{\\to}C/4\\rightarrow$ReLU\n"
                       "$\\rightarrow$Linear $C/4{\\to}C\\rightarrow$Sigmoid$\\rightarrow$scale",
            ha="center", va="bottom", fontsize=7.2, color=C_NOTE, linespacing=1.5)


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from classifier.pi_resnet import PIResNet
    model = PIResNet(in_channels=1, d_model=256, width_scale=4.0,
                     use_snake=False, use_se=True, use_pairwise_fusion=True)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == 32_807_937, f"parameter count drifted: {total}"
    print(f"trainable parameters: {total:,}")

    fig = plt.figure(figsize=(9.4, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[24, 24], width_ratios=[100, 42],
                          hspace=0.06, wspace=0.05,
                          left=0.012, right=0.988, top=0.985, bottom=0.015)
    draw_encoder(fig.add_subplot(gs[0, :]))
    draw_head(fig.add_subplot(gs[1, 0]))
    draw_residual_block(fig.add_subplot(gs[1, 1]))
    fig.savefig(OUT / "fig_architecture.pdf")
    fig.savefig(OUT / "fig_architecture.png", dpi=300)
    plt.close(fig)
    print(f"saved {(OUT / 'fig_architecture.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
