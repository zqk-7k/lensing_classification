#!/usr/bin/env python3
"""Architecture figure of PI-ResNet v3, drawn from the module it describes (v3lib.PairNet).

Every number is taken from v3lib: stream A encoder = the submitted backbone at width_scale 4
(channels 128 -> 1024, 2048-d embedding), stream B encoder at width_scale 2 (64 -> 512, 1024-d),
detectors as input channels, per-event embedding 3072, pairwise interaction [f1*f2, |f1-f2|] -> 6144,
head Linear(6144, 256) -> ReLU -> Dropout(0.7) -> Linear(256, 1). Parameter counts of the PI-ResNet
and early-fusion variants are asserted against the instantiated models and written next to the figure.

Run on a machine with torch:  python make_v3_architecture_figure.py --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

C_STEM, C_STAGE, C_BLOCK, C_POOL = "#dbe7f3", "#c7dcef", "#eef3f8", "#e6eef6"
C_FUSE, C_HEAD, C_EDGE, C_ARROW, C_NOTE = "#f7e3dd", "#f3e6f2", "#7f97ad", "#54606d", "#4a5260"
FS = 7.0  # body text, points


def box(ax, x, y, w, h, text, fc=C_BLOCK, ec=C_EDGE, fs=FS, weight="normal", head=None, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if head is None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
                weight=weight, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h * 0.74, head, ha="center", va="center", fontsize=fs + 0.5,
                weight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.34, text, ha="center", va="center", fontsize=fs, zorder=3,
                linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=lw,
                                 color=C_ARROW, zorder=4, shrinkA=0.5, shrinkB=0.5))


def encoder_row(ax, y, h, label, inp, widths, out_shape, emb):
    box(ax, 0.5, y, 13.0, h, inp, fc="#ffffff", head=label)
    xs = [16.5, 29.5, 42.5, 55.5]
    w = 10.0
    for k, (x, name) in enumerate(zip(xs, ["Stem", "Stage 1", "Stage 2", "Stage 3"])):
        box(ax, x, y, w, h, f"${widths[k]}{{\\to}}{widths[k + 1]}$", fc=C_STEM if k == 0 else C_STAGE, head=name)
        arrow(ax, (xs[k - 1] + w) if k else 13.5, y + h / 2, x, y + h / 2)
    ax.text(xs[-1] + w / 2, y - 1.6, out_shape, ha="center", va="center", fontsize=FS - 0.5, color=C_NOTE)
    box(ax, 69.5, y, 12.5, h, f"avg $\\oplus$ max\npool $\\rightarrow$ {emb}-d", fc=C_POOL, fs=FS - 0.3)
    arrow(ax, xs[-1] + w, y + h / 2, 69.5, y + h / 2)
    return 82.0, y + h / 2


def draw_encoder(ax):
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 33)
    ax.text(0.3, 31.6, "(a)", fontsize=8.5, weight="bold", va="center")
    ax.text(5.0, 31.6, "Per-event encoder, the same weights for both events", fontsize=8.0, weight="bold", va="center")
    h = 7.0
    xa, ya = encoder_row(ax, 19.5, h, "stream A", "$3\\times4096$\n2 s, 2048 Hz", [3, 128, 256, 512, 1024], "$1024\\times128$", 2048)
    xb, yb = encoder_row(ax, 6.0, h, "stream B", "$3\\times8192$\n64 s, <64 Hz", [3, 64, 128, 256, 512], "$512\\times256$", 1024)
    ax.add_patch(Rectangle((15.3, 3.0), 67.9, 25.0, fill=False, ec="#b08d57", lw=1.0, ls=(0, (4, 3)), zorder=1))
    ax.text(49.8, 28.9, "stem + three residual stages, kernels 15 / 11 / 9 / 7, total stride 32; stream B at half width",
            ha="center", va="center", fontsize=FS - 0.3, style="italic", color="#8a6d3b")
    box(ax, 86.5, 11.5, 13.0, 10.0, "concat\n$\\mathbf{f}_i$, 3072-d", fc=C_POOL)
    arrow(ax, xa, ya, 86.5, 19.0)
    arrow(ax, xb, yb, 86.5, 14.0)
    ax.text(49.8, 0.9, "detectors are the input channels (ET: three ET-D interferometers; O4: H1, L1, V1), "
                       "aligned on the geocentric merger time", ha="center", va="center", fontsize=FS - 0.3, color=C_NOTE)


def draw_head(ax, n_pi):
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    ax.text(0.3, 27.8, "(b)", fontsize=8.5, weight="bold", va="center")
    ax.text(7.5, 27.8, "Pairwise interaction and scoring head", fontsize=8.0, weight="bold", va="center")
    y, h = 11.0, 11.0
    box(ax, 0.5, y, 14.5, h, "$\\mathbf{f}_1,\\ \\mathbf{f}_2$\n3072-d each", fc="#ffffff")
    box(ax, 20.0, y, 29.5, h, r"$[\,\mathbf{f}_1\odot\mathbf{f}_2,\ |\mathbf{f}_1-\mathbf{f}_2|\,]$" "\n$\\rightarrow$ 6144-d",
        fc=C_FUSE, head="Pairwise interaction")
    box(ax, 54.5, y, 29.5, h, "Linear $6144{\\to}256$, ReLU\nDropout 0.7, Linear $256{\\to}1$", fc=C_HEAD, head="Scoring head")
    box(ax, 89.5, y + 1.5, 10.0, h - 3.0, "logit\n$z$", fc="#ffffff", weight="bold")
    for x0, x1 in ((15.0, 20.0), (49.5, 54.5), (84.0, 89.5)):
        arrow(ax, x0, y + h / 2, x1, y + h / 2)
    ax.text(50.0, 5.5, f"{n_pi / 1e6:.1f} M parameters; symmetric in the two events; each event encoded once per catalog",
            ha="center", va="center", fontsize=FS - 0.6, color=C_NOTE)
    ax.text(50.0, 2.0, "$z$ is a ranking statistic; thresholds are set on the empirical background",
            ha="center", va="center", fontsize=FS - 0.6, color=C_NOTE, style="italic")


def draw_residual_block(ax):
    ax.axis("off"); ax.set_xlim(0, 60); ax.set_ylim(-1.5, 30)
    ax.text(0.5, 27.8, "(c)", fontsize=8.5, weight="bold", va="center")
    ax.text(12.0, 27.8, "Residual block", fontsize=8.0, weight="bold", va="center")
    x, w, h = 14.0, 45.0, 3.8
    ys = [21.2, 14.2, 7.2, 0.2]
    texts = ["Conv1d $C{\\to}C$ ($k$);  BN;  ReLU", "Conv1d $C{\\to}C$ ($k$);  BN",
             "SE channel gate", "$\\oplus$ identity;  ReLU"]
    fcs = [C_BLOCK, C_BLOCK, "#e2eddf", "#ffffff"]
    for yy, t, fc in zip(ys, texts, fcs):
        box(ax, x, yy, w, h, t, fc=fc)
    for k in range(3):
        arrow(ax, x + w / 2, ys[k], x + w / 2, ys[k + 1] + h)
    # identity path: from the block input, round the left side, into the add box
    xin, ytop = x + w / 2, ys[0] + h
    ax.plot([xin, xin, 6.0, 6.0], [ytop + 1.2, ytop + 0.4, ytop + 0.4, ys[3] + h / 2],
            color=C_ARROW, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.plot([xin, xin], [26.0, ytop], color=C_ARROW, lw=1.0, zorder=1)
    arrow(ax, 6.0, ys[3] + h / 2, x, ys[3] + h / 2)
    ax.text(4.3, 13.0, "identity", rotation=90, ha="center", va="center", fontsize=FS, color=C_NOTE, style="italic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from v3lib import PairNet
    n_pi = sum(p.numel() for p in PairNet("pi", 3, True).parameters())
    n_early = sum(p.numel() for p in PairNet("early", 3, True).parameters())
    m = PairNet("pi", 3, True)
    assert m.encA.out_dim == 2048 and m.encB.out_dim == 1024 and m.head[0].in_features == 6144
    print(f"PI-ResNet v3: {n_pi:,} parameters; early fusion: {n_early:,}")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.06, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[33, 30], width_ratios=[62, 38], hspace=0.10, wspace=0.06,
                          left=0.01, right=0.99, top=0.99, bottom=0.01)
    draw_encoder(fig.add_subplot(gs[0, :]))
    draw_head(fig.add_subplot(gs[1, 0]), n_pi)
    draw_residual_block(fig.add_subplot(gs[1, 1]))
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "fig_v3_architecture.pdf")
    fig.savefig(a.out / "fig_v3_architecture.png", dpi=300)
    (a.out / "fig_v3_architecture.txt").write_text(f"pi {n_pi}\nearly {n_early}\n")


if __name__ == "__main__":
    main()
