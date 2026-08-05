#!/usr/bin/env python3
"""Regenerate Figure 1: 'from strain pairs to a calibrated ranking statistic'.

Design intent: the scoring chain (row a) is established by prior work and is
compressed to a schematic strip; the statistical characterization -- the three
calibration objects defined in Section 3.4 -- is shown with the released data
(rows b-e), so the figure states the paper's contribution at a glance.

Panels (c), (d), (e) correspond one-to-one to the three calibration objects:
  (c) P_f(s*)              background/foreground separation, frozen thresholds
  (d) eps(P_f)             efficiency at the operating points
  (e) eps(theta | P_f)     selection function in the fainter-image S/N

Data: results/core/{unified_predictions_0228_sis.csv.gz,
selection_functions_0228_sis.csv, core_results.json} of the public repository.
Thresholds and efficiencies annotated in the figure are the published,
protocol-frozen values from core_results.json (not recomputed), so the figure
cannot drift from Table 4.
"""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = str(ROOT / "results" / "core")
OUT = str(ROOT / "results" / "figures" / "manuscript")
Path(OUT).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thresholds import kth_threshold  # noqa: E402

C_PI, C_BG, C_ACC = "#1f6fb2", "#8a8f98", "#c0392b"
C_BOX, C_BOXE = "#eef2f7", "#9fb3c8"
C_PROT = "#fdf3e3"

core = json.load(open(f"{BASE}/core_results.json"))
ops = core["results"]["sis"]["models"]["pi_resnet"]["operating_points"]
THR = {t: ops[k]["threshold"] for t, k in [(1e-2, "0.01"), (1e-3, "0.001"), (1e-4, "0.0001")]}
EFF = {t: ops[k]["efficiency"] for t, k in [(1e-2, "0.01"), (1e-3, "0.001"), (1e-4, "0.0001")]}

df = pd.read_csv(f"{BASE}/unified_predictions_0228_sis.csv.gz")
cal_bg = df[(df.calibration_or_evaluation == "calibration") & (df.label == 0)]
ev_pos = df[(df.calibration_or_evaluation == "evaluation") & (df.label == 1)]
sel = pd.read_csv(f"{BASE}/selection_functions_0228_sis.csv")

def to_logit(s):
    s = np.clip(s, 1e-16, 1 - 1e-16)
    return np.log(s / (1 - s))

# thresholds in logit space; the 1e-4 score threshold saturates at 1.0, so take
# the corresponding upper-tail quantile of the calibration logits directly.
lg_thr = {}
for t, s in THR.items():
    lg_thr[t] = kth_threshold(cal_bg.pi_logit, t) if s >= 1.0 else np.log(s / (1 - s))

fig = plt.figure(figsize=(13.2, 7.4))
gs = GridSpec(3, 3, height_ratios=[0.80, 0.52, 1.55], hspace=0.55, wspace=0.28,
              left=0.055, right=0.985, top=0.95, bottom=0.075)

# ---------------------------------------------------------------- row (a)
axa = fig.add_subplot(gs[0, :]); axa.axis("off")
axa.set_xlim(0, 100); axa.set_ylim(0, 10)

def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_BOXE, fs=8.8, weight="normal", head=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.1",
                                fc=fc, ec=ec, lw=1.1, zorder=2))
    if head is None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3, weight=weight, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h * 0.76, head, ha="center", va="center", fontsize=fs + 0.4,
                zorder=3, weight="bold")
        ax.text(x + w / 2, y + h * 0.34, text, ha="center", va="center", fontsize=fs,
                zorder=3, linespacing=1.45)

def arrow(ax, x0, y0, x1, y1, lw=1.3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 lw=lw, color="#5a6472", zorder=1))

axa.text(0.5, 8.9, "(a)", fontsize=11, weight="bold")
axa.text(3.6, 8.9, "Pair scoring", fontsize=10.2, weight="bold", va="center")
axa.text(18.0, 8.9, "— time-domain pair representation established by earlier work",
         fontsize=8.6, style="italic", color="#4a5260", va="center")

box(axa, 1.5, 2.4, 15.5, 4.4, "Source and lens priors\n(Table 1)")
box(axa, 20.0, 2.4, 16.5, 4.4, "Two images\n$\\sqrt{|\\mu_\\pm|}$, $\\Delta t_d$, Morse phase")
box(axa, 39.5, 2.4, 17.0, 4.4, "ET-D noise, whitening,\npeak alignment, 2 s input")
box(axa, 59.5, 4.7, 17.5, 2.6, "PI-ResNet (1D)", fc="#e3eef8")
box(axa, 59.5, 1.4, 17.5, 2.6, "CQT--DeiT (2D baseline)", fc="#f8e6e4")
axa.text(68.25, 0.35, "shared source-level splits; identical evaluation manifests", fontsize=7.0,
         ha="center", style="italic", color="#4a5260")
box(axa, 80.5, 2.4, 18.0, 4.4, "pair score\n$s(d_1,d_2)$", fc="#eef2f7", weight="bold")

for x0, x1 in [(17.0, 20.0), (36.5, 39.5), (56.5, 59.5), (77.0, 80.5)]:
    arrow(axa, x0, 4.6, x1, 4.6)

# ---------------------------------------------------------------- row (b)
axb = fig.add_subplot(gs[1, :]); axb.axis("off")
axb.set_xlim(0, 100); axb.set_ylim(0, 10)
axb.text(0.5, 8.6, "(b)", fontsize=11, weight="bold")
axb.text(3.6, 8.6, "Blind evaluation protocol", fontsize=10.2, weight="bold", va="center")
axb.text(30.5, 8.6, "— archived and version-tagged before the evaluation catalog was unblinded",
         fontsize=8.6, style="italic", color="#4a5260", va="center")

box(axb, 1.5, 0.6, 27.0, 6.2, "training / validation split\n$\\Rightarrow$ network weights",
    fc=C_PROT, fs=8.4, head="Development catalog")
box(axb, 33.0, 0.6, 30.0, 6.2, "background pairs only\n$\\Rightarrow$ thresholds $s_*$ frozen",
    fc=C_PROT, fs=8.4, head="Calibration partition")
box(axb, 67.5, 0.6, 31.0, 6.2, "never seen before unblinding\n$\\Rightarrow$ efficiency, selection function",
    fc=C_PROT, fs=8.4, head="Final-evaluation partition")
arrow(axb, 28.5, 3.7, 33.0, 3.7)
arrow(axb, 63.0, 3.7, 67.5, 3.7)
axb.plot([30.7, 30.7], [-0.2, 7.6], ls=(0, (3, 3)), color="#b08d57", lw=1.4)
axb.plot([65.2, 65.2], [-0.2, 7.6], ls=(0, (3, 3)), color="#b08d57", lw=1.4)

# ---------------------------------------------------------------- row (c,d,e)
lab = fig.add_subplot(gs[2, :]); lab.axis("off")
lab.text(-0.045, 1.14, "(c)--(e)", fontsize=11, weight="bold", transform=lab.transAxes)
lab.text(0.030, 1.14, "Statistical characterization of the score: the three calibration objects of Section 3.4",
         fontsize=10.2, weight="bold", transform=lab.transAxes)

# (c) background / foreground separation with frozen thresholds
axc = fig.add_subplot(gs[2, 0])
bins = np.linspace(-40, 22, 70)
axc.hist(cal_bg.pi_logit, bins=bins, density=True, color=C_BG, alpha=0.55,
         label="background pairs\n(calibration, $10^5$)")
axc.hist(ev_pos.pi_logit, bins=bins, density=True, histtype="step", lw=1.9, color=C_PI,
         label="true pairs\n(evaluation, 1750)")
for t, ls in [(1e-2, ":"), (1e-3, "--"), (1e-4, "-")]:
    axc.axvline(lg_thr[t], color=C_ACC, ls=ls, lw=1.5)
axc.text(lg_thr[1e-4] + 1.0, axc.get_ylim()[1] * 0.55, "$s_*$ at\n$P_f=10^{-4}$",
         fontsize=7.6, color=C_ACC, ha="left")
axc.text(lg_thr[1e-2] - 1.2, axc.get_ylim()[1] * 0.80, "$10^{-2}$", fontsize=7.6,
         color=C_ACC, ha="right")
axc.set_yscale("log"); axc.set_ylim(1e-4, 1.0)
axc.set_xlabel("score in logit space,  $\\ln[s/(1-s)]$")
axc.set_ylabel("normalized count")
axc.set_title("(c)  $P_f(s_*)$: thresholds from the\nempirical background", fontsize=9.4)
axc.legend(fontsize=7.0, frameon=False, loc="upper left")
axc.grid(alpha=0.2)

# (d) efficiency at the operating points
axd = fig.add_subplot(gs[2, 1])
grid = np.geomspace(1e-4, 1e-2, 40)
curve = [(ev_pos.pi_score >= kth_threshold(cal_bg.pi_score, t)).mean() for t in grid]
axd.plot(grid, curve, color=C_PI, lw=2.0)
for t in [1e-2, 1e-3, 1e-4]:
    axd.plot(t, EFF[t], "o", ms=8, mfc="white", mec=C_ACC, mew=1.9, zorder=5)
axd.annotate(f"$\\varepsilon={100*EFF[1e-3]:.1f}\\%$ at the\nprimary point $10^{{-3}}$",
             xy=(1e-3, EFF[1e-3]), xytext=(1.45e-4, 0.80), fontsize=8.0, color=C_ACC,
             arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1.2))
axd.set_xscale("log"); axd.set_ylim(0, 1.0)
axd.set_xlabel("target per-pair $P_f$")
axd.set_ylabel("detection efficiency $\\varepsilon$")
axd.set_title("(d)  $\\varepsilon(P_f)$: efficiency at the\nfrozen operating points", fontsize=9.4)
axd.grid(alpha=0.2)

# (e) selection function
axe = fig.add_subplot(gs[2, 2])
d = sel[(sel.variable == "rho_min") & (sel.model == "pi_resnet")].sort_values("bin_left")
x = 0.5 * (d.bin_left + d.bin_right)
axe.errorbar(x, d.efficiency, yerr=[d.efficiency - d.ci_low, d.ci_high - d.efficiency],
             fmt="o-", color=C_PI, ms=5, lw=1.8, capsize=3)
axe.set_ylim(0, 1.0)
axe.set_xlabel("fainter-image S/N  $\\rho_{\\rm min}$")
axe.set_ylabel("$\\varepsilon$ at $P_f=10^{-3}$")
axe.set_title("(e)  $\\varepsilon(\\boldsymbol{\\theta}\\mid P_f)$: which systems\nsurvive the cut", fontsize=9.4)
axe.grid(alpha=0.2)
axe.text(0.96, 0.08, "prior-marginalized projection;\njoint reweighting needs the\nreleased event-level products",
         transform=axe.transAxes, fontsize=7.4, ha="right", va="bottom", style="italic",
         color="#4a5260")

# connector from row (b) to the panels
for xx in [0.19, 0.52, 0.85]:
    fig.add_artist(FancyArrowPatch((xx, 0.398), (xx, 0.368), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#5a6472"))

fig.savefig(f"{OUT}/pipeline.pdf")
fig.savefig(f"{OUT}/pipeline.png", dpi=200)
plt.close(fig)
print("thresholds (score):", {f"{t:g}": round(v, 5) for t, v in THR.items()})
print("thresholds (logit):", {f"{t:g}": round(v, 2) for t, v in lg_thr.items()})
print("efficiencies:", {f"{t:g}": round(v, 4) for t, v in EFF.items()})
print("saved pipeline.pdf/.png")
