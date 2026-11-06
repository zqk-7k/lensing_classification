#!/usr/bin/env python3
"""Regenerate the S/N-matched SIS-PM figure with $P_f$ axis labels (data unchanged:
results/core/snr_matched_sis_pm.csv from the released analysis)."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Drawn at the width it is placed at (\columnwidth = 3.35in) so LaTeX does
# not scale it down; the two panels stack, since side by side in one column
# leaves each panel too narrow to label.
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "axes.linewidth": 0.7,
                     "grid.linewidth": 0.5, "xtick.major.width": 0.7,
                     "ytick.major.width": 0.7})
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = str(ROOT / "results" / "core")
OUT = str(ROOT / "results" / "figures" / "manuscript")
Path(OUT).mkdir(parents=True, exist_ok=True)

d=pd.read_csv(f"{BASE}/snr_matched_sis_pm.csv")
C={"pi_resnet":("#1f6fb2","PI-ResNet"),"cqt_deit":("#c0392b","CQT-DeiT")}
fig,axes=plt.subplots(2,1,figsize=(3.35,3.5),sharex=True,sharey=True)
for ax,(mod,(col,name)) in zip(axes,C.items()):
    x=np.arange(2); w=0.36
    un=[d[(d.lens==L)&(d.model==mod)].unweighted_efficiency_common_support.iloc[0] for L in ["SIS","PM"]]
    wt=[d[(d.lens==L)&(d.model==mod)].weighted_efficiency.iloc[0] for L in ["SIS","PM"]]
    ax.bar(x-w/2,un,w,color="0.65",label="Unmatched")
    ax.bar(x+w/2,wt,w,color=col,label="SNR/$y$ matched")
    ax.set_xticks(x); ax.set_xticklabels(["SIS","PM"]); ax.set_title(name)
    ax.set_ylim(0,0.60); ax.grid(axis="y",alpha=0.25); ax.legend(frameon=False)
for ax in axes:
    ax.set_ylabel(r"Efficiency at $P_f=10^{-3}$")
fig.tight_layout()
fig.savefig(f"{OUT}/snr_matched_sis_pm.pdf"); fig.savefig(f"{OUT}/snr_matched_sis_pm.png",dpi=200)
print("regenerated; values:", [round(v,4) for v in un+wt])
