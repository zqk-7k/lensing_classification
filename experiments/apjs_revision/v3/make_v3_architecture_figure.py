#!/usr/bin/env python3
"""Architecture figure of PI-ResNet v3, drawn from the module it describes (v3lib.PairNet).

Every number is taken from v3lib: stream A encoder = the submitted backbone at width_scale 4
(channels 128 -> 1024, 2048-d embedding), stream B encoder at width_scale 2 (64 -> 512, 1024-d),
detectors as input channels, per-event embedding 3072, pairwise interaction [f1*f2, |f1-f2|] -> 6144,
head Linear(6144, 256) -> ReLU -> Dropout(0.7) -> Linear(256, 1). Parameter counts of the PI-ResNet
and early-fusion variants are asserted against the instantiated models.

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


def box(ax, x, y, w, h, text, fc=C_BLOCK, ec=C_EDGE, fs=6.3, weight="normal", head=None, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if head is None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
                weight=weight, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h * 0.72, head, ha="center", va="center", fontsize=fs + 0.4,
                weight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.32, text, ha="center", va="center", fontsize=fs, zorder=3,
                linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=lw,
                                 color=C_ARROW, zorder=4, linestyle=ls, shrinkA=0, shrinkB=0))


def encoder_row(ax, y, label, inp, widths, out_shape, emb):
    box(ax, 1.0, y, 14.0, 6.0, inp, fc="#ffffff", fs=6.0, head=label)
    xs = [18.5, 32.0, 45.5, 59.0]
    names = ["Stem", "Stage 1", "Stage 2", "Stage 3"]
    for k, (x, name) in enumerate(zip(xs, names)):
        a, b = widths[k], widths[k + 1]
        box(ax, x, y, 11.5, 6.0, f"${a}{{\\to}}{b}$", fc=C_STEM if k == 0 else C_STAGE, fs=6.1, head=name)
        if k:
            arrow(ax, x - 2.0, y + 3.0, x, y + 3.0)
    ax.text(64.75, y - 1.3, out_shape, ha="center", fontsize=5.6, color=C_NOTE)
    arrow(ax, 15.0, y + 3.0, 18.5, y + 3.0)
    box(ax, 73.5, y, 9.0, 6.0, f"avg $\\oplus$ max\npool\n$\\rightarrow${emb}", fc=C_POOL, fs=5.9)
    arrow(ax, 70.5, y + 3.0, 73.5, y + 3.0)


def draw_encoder(ax):
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(17.0, 46.0)
    ax.text(0.5, 44.3, "(a)", fontsize=8.0, weight="bold", va="center")
    ax.text(5.5, 44.3, "Per-event encoder: applied to each event with the same weights",
            fontsize=7.4, weight="bold", va="center")
    encoder_row(ax, 33.0, "stream A", "$3\\times4096$\n2 s, 2048 Hz", [3, 128, 256, 512, 1024],
                "$1024\\times128$", 2048)
    encoder_row(ax, 21.5, "stream B", "$3\\times8192$\n64 s, 128 Hz, <64 Hz", [3, 64, 128, 256, 512],
                "$512\\times256$", 1024)
    ax.add_patch(Rectangle((17.6, 19.0), 66.0, 22.6, fill=False, ec="#b08d57", lw=1.0,
                           ls=(0, (4, 3)), zorder=1))
    ax.text(50.5, 40.0, "stem + three residual stages, kernels 15/11/9/7, total stride 32 "
                        "(the submitted backbone; stream B at half width)",
            ha="center", fontsize=5.7, style="italic", color="#8a6d3b")
    box(ax, 87.0, 26.8, 11.5, 8.4, "concat\n$\\mathbf{f}_i$\n3072-d", fc=C_POOL, fs=6.1)
    arrow(ax, 82.5, 36.0, 87.0, 32.5)
    arrow(ax, 82.5, 24.5, 87.0, 29.5)
    ax.text(50.5, 17.6, "detectors are input channels (ET: three ET-D interferometers; O4: H1, L1, V1), "
                        "aligned on the geocentric merger time", ha="center", fontsize=5.7, color=C_NOTE)


def draw_head(ax, n_pi, n_early):
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 24)
    ax.text(0.5, 22.3, "(b)", fontsize=8.0, weight="bold", va="center")
    ax.text(5.5, 22.3, "Pairwise interaction and scoring head", fontsize=7.4, weight="bold", va="center")
    box(ax, 1.0, 12.0, 15.0, 6.0, "$\\mathbf{f}_1,\\ \\mathbf{f}_2$\n(events 1 and 2)", fc="#ffffff", fs=6.1)
    box(ax, 20.0, 11.5, 25.0, 7.0, "$[\\,\\mathbf{f}_1\\odot\\mathbf{f}_2,\;|\\mathbf{f}_1-\\mathbf{f}_2|\\,]$"
        "$\\rightarrow6144$", fc=C_FUSE, fs=6.3, head="Pairwise interaction")
    box(ax, 49.0, 11.5, 25.0, 7.0, "Linear $6144{\\to}256$, ReLU\nDropout 0.7, Linear $256{\\to}1$",
        fc=C_HEAD, fs=6.0, head="Scoring head")
    box(ax, 78.0, 12.5, 20.5, 5.0, "pair logit $z(d_1,d_2)$", fc="#ffffff", fs=6.4, weight="bold")
    arrow(ax, 16.0, 15.0, 20.0, 15.0); arrow(ax, 45.0, 15.0, 49.0, 15.0); arrow(ax, 74.0, 15.0, 78.0, 15.0)
    ax.text(50.0, 9.0, f"PI-ResNet: {n_pi / 1e6:.1f} M parameters; symmetric in the two events; "
                       "each event is encoded once per catalogue", ha="center", fontsize=5.8, color=C_NOTE)
    box(ax, 1.0, 0.6, 97.5, 5.6,
        "Early fusion (comparison): the channels of both events are stacked ($6\\times4096$, $6\\times8192$) and "
        "passed through the same two encoders;\nthe head acts on the joint 3072-d embedding, with no pairwise layer "
        f"({n_early / 1e6:.1f} M parameters). Every pair must be encoded in full.",
        fc="#f4f4f4", ec="#b5b5b5", fs=5.8)


def draw_residual_block(ax):
    ax.axis("off"); ax.set_xlim(0, 40); ax.set_ylim(-3.2, 46.0)
    ax.text(0.5, 44.3, "(c)", fontsize=8.0, weight="bold", va="center")
    ax.text(6.5, 44.3, "Residual block", fontsize=7.4, weight="bold", va="center")
    x, w = 9.0, 22.0
    for y, t in [(38.2, "Conv1d $C{\\to}C$, kernel $k$"), (34.2, "BatchNorm1d"), (30.2, "ReLU"),
                 (26.2, "Conv1d $C{\\to}C$, kernel $k$"), (22.2, "BatchNorm1d")]:
        box(ax, x, y, w, 3.0, t, fs=6.0)
    box(ax, x, 17.4, w, 3.4, "SE (channel gate)", fc="#e2eddf", fs=6.0)
    box(ax, x, 10.4, w, 3.4, "$\\oplus$  residual add", fc="#ffffff", fs=6.2)
    box(ax, x, 5.4, w, 3.0, "ReLU", fs=6.0)
    for y0, y1 in [(38.2, 37.2), (34.2, 33.2), (30.2, 29.2), (26.2, 25.2), (22.2, 20.8), (17.4, 13.8), (10.4, 8.4)]:
        arrow(ax, x + w / 2, y0, x + w / 2, y1)
    ax.plot([5.4, 5.4], [41.2, 12.1], color=C_ARROW, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.plot([5.4, 9.0], [41.2, 41.2], color=C_ARROW, lw=1.1, ls=(0, (4, 3)), zorder=1)
    arrow(ax, 5.4, 12.1, 9.0, 12.1, ls=(0, (4, 3)))
    ax.text(4.4, 27.0, "identity", rotation=90, ha="center", va="center", fontsize=6.6, color=C_NOTE, style="italic")
    ax.text(20.0, -3.0, "SE: GAP$\\rightarrow$Linear $C{\\to}C/4\\rightarrow$ReLU\n"
                        "$\\rightarrow$Linear $C/4{\\to}C\\rightarrow$Sigmoid$\\rightarrow$scale",
            ha="center", va="bottom", fontsize=5.9, color=C_NOTE, linespacing=1.4)


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
    fig = plt.figure(figsize=(7.06, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[29, 24], width_ratios=[100, 40], hspace=0.04, wspace=0.04,
                          left=0.01, right=0.99, top=0.99, bottom=0.01)
    draw_encoder(fig.add_subplot(gs[0, :]))
    draw_head(fig.add_subplot(gs[1, 0]), n_pi, n_early)
    draw_residual_block(fig.add_subplot(gs[1, 1]))
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "fig_v3_architecture.pdf")
    fig.savefig(a.out / "fig_v3_architecture.png", dpi=300)
    (a.out / "fig_v3_architecture.txt").write_text(f"pi {n_pi}\nearly {n_early}\n")


if __name__ == "__main__":
    main()
