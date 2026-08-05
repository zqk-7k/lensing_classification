#!/usr/bin/env python3
"""Regenerate three manuscript figures from released artifacts (commit 130dfeb).

Outputs (PDF+PNG) into FIGDIR:
  snr_dist                    -- Fig. 3 replacement: unlensed vs lensed per-image S/N,
                                 common log bins, dashed rho=8 reference (referee G3).
  fixed_fpp_efficiency_bars   -- Fig. 5 replacement: grouped bars at the three operating
                                 points, strict FPP on the LEFT (1e-4 -> 1e-2), 95% CIs,
                                 paired-difference annotations (visual form of Table 4).
  fpp_selection_shift         -- companion to the selection functions: efficiency (left
                                 axis) and median impact parameter y of surviving pairs
                                 (right axis) vs target FPP, normal axis (1e-4 left).
Also prints Appendix numbers (detected-pair subset, jackknife) from posthoc_diagnostics.json.
"""
import sys
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = str(ROOT / "results" / "core")
FIGDIR = str(ROOT / "results" / "figures" / "manuscript")
Path(FIGDIR).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thresholds import kth_threshold  # noqa: E402

SEED, REPS = 20260711, 10000
C_PI, C_CQT, C_Y = "#1f6fb2", "#c0392b", "#c0392b"

core = json.load(open(f"{BASE}/core_results.json"))
DF = {f: pd.read_csv(f"{BASE}/unified_predictions_0228_{f}.csv.gz") for f in ["sis", "pm"]}

def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.pdf"); fig.savefig(f"{FIGDIR}/{name}.png", dpi=200); plt.close(fig)

# ---------------- Figure: snr_dist (regenerated, G3-compliant) ----------------
unl = pd.concat([DF[f][DF[f].negative_type == "easy"][["right_event_id", "rho_2"]] for f in ["sis", "pm"]])
unl = unl.drop_duplicates("right_event_id").rho_2.to_numpy()          # distinct unlensed events pooled (n=5,000)
lens = {f: np.concatenate([DF[f][DF[f].label == 1].rho_1, DF[f][DF[f].label == 1].rho_2]) for f in ["sis", "pm"]}
lo = min(unl.min(), lens["sis"].min(), lens["pm"].min())
hi = max(unl.max(), lens["sis"].max(), lens["pm"].max())
bins = np.geomspace(max(lo, 0.3), hi * 1.02, 45)                       # identical edges, both panels
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
a1.hist(unl, bins=bins, color="0.55", alpha=0.85, label="Unlensed events (pooled)")
a2.hist(lens["sis"], bins=bins, histtype="step", lw=2.0, color=C_PI, label="Lensed images, SIS")
a2.hist(lens["pm"], bins=bins, histtype="step", lw=2.0, color=C_CQT, ls="--", label="Lensed images, PM")
for ax in (a1, a2):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(8, color="k", ls=":", lw=1.6)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylabel("Events")
a1.text(8, a1.get_ylim()[1] * 0.45, r" $\rho=8$", fontsize=9)
a2.set_xlabel(r"Optimal single-detector S/N $\rho$")
fig.tight_layout(); save(fig, "snr_dist")
for f in ["sis", "pm"]:
    both = DF[f][DF[f].label == 1]
    print(f"[snr_dist] {f}: frac pairs with both rho>=8 = {((both.rho_1>=8)&(both.rho_2>=8)).mean():.4f}")
print(f"[snr_dist] unlensed frac >=8 = {(unl>=8).mean():.4f}, n_unl={len(unl)}")

# ------------- Figure: fixed_fpp_efficiency_bars (Table 4 as bars) -------------
FPPS = ["0.0001", "0.001", "0.01"]                                     # strict on the LEFT
LBL = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"]
# read the paired differences from the locked analysis rather than hard-coding them,
# so the annotations cannot go stale when the analysis is rerun
DELTA = {fam: [f"${100 * core['results'][fam]['paired']['efficiency_difference'][f]['point']:+.1f}$"
               for f in FPPS] for fam in ("sis", "pm")}
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)
for ax, fam, title in zip(axes, ["sis", "pm"], ["SIS", "PM"]):
    x = np.arange(3); w = 0.34
    for k, (m, c, name) in enumerate([("pi_resnet", C_PI, "PI-ResNet"), ("cqt_deit", C_CQT, "CQT--DeiT")]):
        pts = [core["results"][fam]["models"][m]["operating_points"][f] for f in FPPS]
        eff = [p["efficiency"] for p in pts]
        lo_ = [p["efficiency"] - p["efficiency_ci"][0] for p in pts]
        hi_ = [p["efficiency_ci"][1] - p["efficiency"] for p in pts]
        ax.bar(x + (k - 0.5) * w, eff, w, color=c, alpha=0.9, label=name if fam == "sis" else None,
               yerr=[lo_, hi_], capsize=3, error_kw=dict(lw=1.1))
    for i in range(3):
        top = max(core["results"][fam]["models"][m]["operating_points"][FPPS[i]]["efficiency_ci"][1]
                  for m in ["pi_resnet", "cqt_deit"])
        ax.text(x[i], top + 0.035, DELTA[fam][i] + " pp", ha="center", fontsize=8.6)
    ax.set_xticks(x); ax.set_xticklabels(LBL)
    ax.set_xlabel("Target per-pair $P_f$"); ax.set_title(title)
    ax.set_ylim(0, 0.95); ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("Detection efficiency"); axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout(); save(fig, "fixed_fpp_efficiency_bars")

# ------- Figure: fpp_selection_shift (dual axis, NORMAL orientation) -------
fpp_grid = np.geomspace(1e-4, 1e-2, 25)
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
capnum = {}
for ax, fam, title in zip(axes, ["sis", "pm"], ["SIS", "PM"]):
    df = DF[fam]
    cal = df[(df.calibration_or_evaluation == "calibration") & (df.label == 0)]
    pos = df[(df.calibration_or_evaluation == "evaluation") & (df.label == 1)]
    blocks = np.sort(pos.source_block_id.unique())
    pos_by_block = {b: pos[pos.source_block_id == b] for b in blocks}
    # numpy views per block: 10,000 replicates x 25 grid points is far too slow with pd.concat
    score_by_block = [pos_by_block[b].pi_score.to_numpy() for b in blocks]
    y_by_block = [pos_by_block[b].y.to_numpy() for b in blocks]
    draws = rng.integers(0, len(blocks), size=(REPS, len(blocks)))
    E, Elo, Ehi, Y, Ylo, Yhi = ([] for _ in range(6))
    for t in fpp_grid:
        thr = kth_threshold(cal.pi_score, t)
        surv = pos[pos.pi_score >= thr]
        E.append(len(surv) / len(pos)); Y.append(surv.y.median())
        er, yr = [], []
        for d in draws:
            s = np.concatenate([score_by_block[i] for i in d])
            yy = np.concatenate([y_by_block[i] for i in d])
            keep = s >= thr
            er.append(keep.mean()); yr.append(np.median(yy[keep]) if keep.any() else np.nan)
        Elo.append(np.percentile(er, 2.5)); Ehi.append(np.percentile(er, 97.5))
        Ylo.append(np.nanpercentile(yr, 2.5)); Yhi.append(np.nanpercentile(yr, 97.5))
    ax.set_xscale("log")                                   # normal orientation: 1e-4 left
    l1, = ax.plot(fpp_grid, E, color=C_PI, lw=2, marker="o", ms=3)
    ax.fill_between(fpp_grid, Elo, Ehi, color=C_PI, alpha=0.18, lw=0)
    ax.set_ylim(0, 1); ax.set_xlabel("Target per-pair $P_f$")
    ax.set_ylabel("Efficiency", color=C_PI); ax.tick_params(axis="y", labelcolor=C_PI)
    ax.set_title(f"{title}  (PI-ResNet)")
    ax2 = ax.twinx()
    l2, = ax2.plot(fpp_grid, Y, color=C_Y, lw=2, marker="s", ms=3, ls="--")
    ax2.fill_between(fpp_grid, Ylo, Yhi, color=C_Y, alpha=0.15, lw=0)
    ax2.set_ylim(0, 0.30)
    ax2.set_ylabel("Median $y$ of surviving pairs", color=C_Y); ax2.tick_params(axis="y", labelcolor=C_Y)
    if fam == "sis":
        ax.legend([l1, l2], ["Efficiency (left axis)", "Median $y$ of survivors (right axis)"],
                  loc="lower right", fontsize=8.6, frameon=False)
    med_all = pos.y.median()
    caps = {}
    for t in [1e-2, 1e-3, 1e-4]:
        thr = kth_threshold(cal.pi_score, t); s = pos[pos.pi_score >= thr]
        caps[t] = (s.y.median(), s.rho_min.median())
    capnum[fam] = (med_all, pos.rho_min.median(), caps)
fig.tight_layout(); save(fig, "fpp_selection_shift")
for fam, (ya, ra, caps) in capnum.items():
    print(f"[shift] {fam}: all-pos median y={ya:.3f}, rho_min={ra:.1f}; " +
          "; ".join(f"FPP={t:g}: y={v[0]:.3f}, rho_min={v[1]:.1f}" for t, v in caps.items()))

# ---------------- Appendix numbers from posthoc_diagnostics.json ----------------
pj = json.load(open(f"{BASE}/posthoc_diagnostics.json", encoding="utf-8"))
print("\n================ APPENDIX: detected-pair subset table rows ================")
for fam in ["sis", "pm"]:
    fo = pj["families"][fam]
    print(f"-- {fam.upper()}: subset n={fo['detected_subset']['n']} ({fo['detected_subset']['fraction']:.3f})")
    for f in ["0.01", "0.001", "0.0001"]:
        p = fo[f"detected_subset.pi_resnet.eff@{f}"]; c = fo[f"detected_subset.cqt_deit.eff@{f}"]
        d = fo[f"detected_subset.delta_eff@{f}"]
        print(f"  FPP {f}: PI {100*p['point']:.1f} [{100*p['ci'][0]:.1f},{100*p['ci'][1]:.1f}] | "
              f"CQT {100*c['point']:.1f} [{100*c['ci'][0]:.1f},{100*c['ci'][1]:.1f}] | "
              f"dEff {100*d['point']:+.1f} [{100*d['ci'][0]:+.1f},{100*d['ci'][1]:+.1f}] pp")
print("\n-- jackknife leave-one-block Delta-eff@1e-3 (pp):")
for fam in ["sis", "pm"]:
    v = pj["families"][fam]["jackknife_delta_eff_1e3"]["values"]
    print(f"  {fam}: " + ", ".join(f"{100*x:+.2f}" for x in v))
print("DONE")
