#!/usr/bin/env python3
"""Plot the paired E7 score shifts from frozen result files."""

from pathlib import Path
import hashlib
import matplotlib.pyplot as plt
import pandas as pd

# Drawn at \columnwidth (3.35in), the width it is placed at, so the
# lettering is not scaled down on the page. The panels stack for the
# same reason as in regen_snr_matched.py.
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "axes.linewidth": 0.7,
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7})

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/diagnostics/type_ii"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.4), sharex=True)
for axis, lens in zip(axes, ("sis", "pm")):
    data = pd.read_csv(BASE / f"e7_scores_{lens}.csv.gz")
    delta = data.score_difference_physical_minus_control
    axis.hist(delta, bins=30, color="#1764ab", alpha=.8)
    axis.axvline(0, color="black", linewidth=1)
    axis.axvline(delta.mean(), color="#d1495b", linestyle="--", label=f"mean={delta.mean():.3f}")
    axis.set_title(lens.upper()); axis.set_xlabel("Physical minus no-Morse score"); axis.legend(frameon=False)
for axis in axes:
    axis.set_ylabel("Sources")
fig.tight_layout()
png, pdf = BASE / "e7_score_shift.png", BASE / "e7_score_shift.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)
(BASE / "e7_figure.sha256").write_text(f"{sha(png)}  {png.name}\n{sha(pdf)}  {pdf.name}\n")
