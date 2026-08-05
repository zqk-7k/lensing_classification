"""Instance-to-instance variability and seed ensembling at the locked operating points.

Limitation (vi) of the manuscript notes that one trained instance is retained per
architecture and lens family, so the reported intervals capture evaluation and
calibration sampling but not run-to-run training variability. This script quantifies
that missing component: it trains nothing, but scores the locked 0228 pair manifests
with every archived seed, and reports both the per-seed spread and the behaviour of a
score-averaged ensemble.

Thresholds are re-derived per configuration with the locked k-th-largest rule on the
calibration background, exactly as in the primary analysis; the evaluation partition is
untouched. Output: results/seeds/seed_variability_{lens}.json
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/reproducibility")
from thresholds import kth_threshold

import argparse

SEEDS = [42, 43, 44, 45, 46]
MODELS = ["pi", "cqt_deit"]
TARGETS = [(1e-2, "1e-2"), (1e-3, "1e-3"), (1e-4, "1e-4")]

# seed 42 lives in the main prediction dir; 43-46 in results/seeds/predictions
AP = argparse.ArgumentParser(); AP.add_argument("--lens", default="pm", choices=["pm", "sis"])
LENS = AP.parse_args().lens

base = pd.read_csv(f"results/core/unified_predictions_0228_{LENS}.csv.gz",
                   usecols=["pair_id", "label", "source_block_id", "calibration_or_evaluation"])
scores = {}
for m, stem in (("pi", f"pi_predictions_0228_{LENS}"),
                ("cqt_deit", f"cqt_deit_predictions_0228_{LENS}")):
    for s in SEEDS:
        path = (f"results/predictions/{stem}.csv.gz" if s == 42
                else f"results/seeds/predictions/{stem}_seed{s}.csv.gz")
        col = "pi_score" if m == "pi" else "cqt_deit_score"
        d = pd.read_csv(path, usecols=["pair_id", col])
        scores[(m, s)] = d.set_index("pair_id")[col]

idx = base.set_index("pair_id")
cal_mask = (idx.calibration_or_evaluation == "calibration") & (idx.label == 0)
pos_mask = (idx.calibration_or_evaluation == "evaluation") & (idx.label == 1)
blocks = idx.loc[pos_mask, "source_block_id"].to_numpy()
ub = np.unique(blocks)

def eff_and_delta(pi_s, cq_s):
    out = {}
    for t, name in TARGETS:
        thr_pi = kth_threshold(pi_s[cal_mask.values], t)
        thr_cq = kth_threshold(cq_s[cal_mask.values], t)
        e_pi = (pi_s[pos_mask.values] >= thr_pi).mean()
        e_cq = (cq_s[pos_mask.values] >= thr_cq).mean()
        out[name] = (e_pi, e_cq, e_pi - e_cq)
    return out

aligned = {k: v.reindex(idx.index).to_numpy() for k, v in scores.items()}

print(f"=== per seed ({LENS.upper()}) ===")
print(f"  {'seed':>6}{'PI@1e-3':>10}{'CQT@1e-3':>10}{'差':>9}{'PI@1e-2':>10}{'CQT@1e-2':>10}{'差':>9}")
per = {}
for s in SEEDS:
    r = eff_and_delta(aligned[("pi", s)], aligned[("cqt_deit", s)])
    per[s] = r
    print(f"  {s:>6}{100*r['1e-3'][0]:9.1f}%{100*r['1e-3'][1]:9.1f}%{100*r['1e-3'][2]:+8.1f}"
          f"{100*r['1e-2'][0]:9.1f}%{100*r['1e-2'][1]:9.1f}%{100*r['1e-2'][2]:+8.1f}")
for name in ("1e-2", "1e-3", "1e-4"):
    d = np.array([per[s][name][2] for s in SEEDS])
    p = np.array([per[s][name][0] for s in SEEDS]); c = np.array([per[s][name][1] for s in SEEDS])
    print(f"  @{name}: PI {100*p.mean():.1f}+-{100*p.std(ddof=1):.1f}  "
          f"CQT {100*c.mean():.1f}+-{100*c.std(ddof=1):.1f}  "
          f"差 {100*d.mean():+.1f}+-{100*d.std(ddof=1):.1f}  范围[{100*d.min():+.1f},{100*d.max():+.1f}]")

# per-instance threshold-inclusive intervals, so that statements about which
# operating points are resolved can be made for every instance, not only seed 42
print("\n=== per-instance threshold-inclusive intervals (10,000 replicates) ===")
cb_ids = idx.loc[cal_mask, "source_block_id"].to_numpy()
ucb = np.unique(cb_ids)
per_seed_ci = {}
for s in SEEDS:
    calA = {m: [aligned[(m, s)][cal_mask.values][cb_ids == b] for b in ucb] for m in MODELS}
    posA = {m: [aligned[(m, s)][pos_mask.values][blocks == b] for b in ub] for m in MODELS}
    rng = np.random.default_rng(20260711)
    rows = {}
    for t_, name in TARGETS:
        d = []
        for _ in range(10000):
            cd = rng.integers(0, len(ucb), size=len(ucb)); ed = rng.integers(0, len(ub), size=len(ub))
            thr = {m: kth_threshold(np.concatenate([calA[m][i] for i in cd]), t_) for m in MODELS}
            pp = {m: np.concatenate([posA[m][i] for i in ed]) for m in MODELS}
            d.append((pp["pi"] >= thr["pi"]).mean() - (pp["cqt_deit"] >= thr["cqt_deit"]).mean())
        d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
        rows[name] = {"ci": [float(lo), float(hi)], "frac_le_zero": float((d <= 0).mean())}
    per_seed_ci[s] = rows
    print(f"  seed {s}: " + "  ".join(
        f"@{n} [{100*rows[n]['ci'][0]:+.1f},{100*rows[n]['ci'][1]:+.1f}]" for n in ("1e-2", "1e-3", "1e-4")))
for name in ("1e-2", "1e-3", "1e-4"):
    resolved = [s for s in SEEDS if per_seed_ci[s][name]["ci"][0] * per_seed_ci[s][name]["ci"][1] > 0]
    print(f"  @{name}: {len(resolved)}/{len(SEEDS)} instances resolve a difference  {resolved}")

print("\n=== 5 种子集成 (分数平均) ===")
ens_pi = np.mean([aligned[("pi", s)] for s in SEEDS], axis=0)
ens_cq = np.mean([aligned[("cqt_deit", s)] for s in SEEDS], axis=0)
r = eff_and_delta(ens_pi, ens_cq)
for name in ("1e-2", "1e-3", "1e-4"):
    print(f"  @{name}: PI {100*r[name][0]:.1f}%  CQT {100*r[name][1]:.1f}%  差 {100*r[name][2]:+.1f} pp")

# block bootstrap on the ensemble, threshold-inclusive
print("\n=== 集成的 threshold-inclusive 自举 (10,000 次) ===")
cb = idx.loc[cal_mask, "source_block_id"].to_numpy()
ucb = np.unique(cb)
calA = {"pi": [ens_pi[cal_mask.values][cb == b] for b in ucb],
        "cqt": [ens_cq[cal_mask.values][cb == b] for b in ucb]}
posA = {"pi": [ens_pi[pos_mask.values][blocks == b] for b in ub],
        "cqt": [ens_cq[pos_mask.values][blocks == b] for b in ub]}
rng = np.random.default_rng(20260711)
for t, name in TARGETS:
    d = []
    for _ in range(10000):
        cd = rng.integers(0, len(ucb), size=len(ucb)); ed = rng.integers(0, len(ub), size=len(ub))
        tp = kth_threshold(np.concatenate([calA["pi"][i] for i in cd]), t)
        tc = kth_threshold(np.concatenate([calA["cqt"][i] for i in cd]), t)
        pp = np.concatenate([posA["pi"][i] for i in ed]); pc = np.concatenate([posA["cqt"][i] for i in ed])
        d.append((pp >= tp).mean() - (pc >= tc).mean())
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  @{name}: 点 {100*r[name][2]:+.1f} pp  CI [{100*lo:+.1f},{100*hi:+.1f}]  P(<=0)={(d<=0).mean():.4f}")


report = {
    "lens": LENS.upper(), "seeds": SEEDS,
    "per_seed": {str(s): {n: {"pi": per[s][n][0], "cqt_deit": per[s][n][1], "delta": per[s][n][2]}
                          for n in ("1e-2", "1e-3", "1e-4")} for s in SEEDS},
    "ensemble": {n: {"pi": r[n][0], "cqt_deit": r[n][1], "delta": r[n][2]} for n in ("1e-2", "1e-3", "1e-4")},
    "per_seed_threshold_inclusive": {str(s): per_seed_ci[s] for s in SEEDS},
    "note": "Post-hoc instance-variability study. The locked primary analysis uses seed 42 only.",
}
import pathlib as _p
_p.Path("results/seeds").mkdir(parents=True, exist_ok=True)
with open(f"results/seeds/seed_variability_{LENS}.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print(f"\nwrote results/seeds/seed_variability_{LENS}.json")
