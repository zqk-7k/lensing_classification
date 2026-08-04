#!/usr/bin/env python3
"""Regenerate the two selection-function figures from the released binned CSVs
(results/core/selection_functions_0228_{sis,pm}.csv) with $P_f$ axis labels."""
import pandas as pd, matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = str(ROOT / "results" / "core")
OUT = str(ROOT / "results" / "figures" / "manuscript")
Path(OUT).mkdir(parents=True, exist_ok=True)

C = {"pi_resnet": ("#1f6fb2", "PI-ResNet"), "cqt_deit": ("#c0392b", "CQT-DeiT")}
XL = {"y": "Impact parameter $y$", "flux_ratio": "Absolute magnification ratio",
      "rho_min": "Weaker-image SNR"}

for fam in ["sis", "pm"]:
    df = pd.read_csv(f"{BASE}/selection_functions_0228_{fam}.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    for ax, var in zip(axes, ["y", "flux_ratio", "rho_min"]):
        for model, (c, name) in C.items():
            d = df[(df.variable == var) & (df.model == model)].sort_values("bin_left")
            x = 0.5 * (d.bin_left + d.bin_right)
            ax.errorbar(x, d.efficiency,
                        yerr=[d.efficiency - d.ci_low, d.ci_high - d.efficiency],
                        fmt="o-", color=c, ms=5, lw=1.6, capsize=3,
                        label=name if var == "rho_min" else None)
        ax.set_xlabel(XL[var]); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"Efficiency at target $P_f=10^{-3}$")
    axes[2].legend(frameon=True, fontsize=9,
                   loc="lower right" if fam == "sis" else "upper right")
    fig.suptitle(fam.upper(), y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/selection_functions_{fam}.pdf")
    fig.savefig(f"{OUT}/selection_functions_{fam}.png", dpi=200)
    plt.close(fig)
    print(f"{fam}: rows used {len(df)}")
print("done")
