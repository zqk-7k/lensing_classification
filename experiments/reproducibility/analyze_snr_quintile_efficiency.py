#!/usr/bin/env python3
"""Efficiency at the frozen thresholds, stratified by fainter-image signal-to-noise.

Section 5.5 of the manuscript reports where along the S/N range the two statistics
differ: on PM the advantage is concentrated in the middle of the range and the ordering
inverts in the brightest quintile, while on SIS it grows monotonically. Those numbers
were quoted without a generating script, which this closes.

The stratification uses the calibration-partition quintiles of the fainter-image optimal
S/N, matching the bin edges of the selection-function tables, and the thresholds are the
frozen ones of the locked analysis. Intervals are source-block bootstrap intervals over
the evaluation partition, so a bin difference can be read as resolved or not rather than
only as a point estimate.

This is a post-hoc, descriptive analysis. It changes no primary result and consumes only
the released per-pair scores.

Output: results/core/snr_quintile_efficiency.{json,csv}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "results" / "core"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thresholds import kth_threshold  # noqa: E402

SEED, REPS, TARGET = 20260711, 10000, 1e-3
MODELS = {"pi": "pi_resnet", "cqt_deit": "cqt_deit"}


def percentile_ci(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    report = {"target_fpp": TARGET, "seed": SEED, "replicates": REPS,
              "bin_variable": "rho_min",
              "bin_edges_from": "calibration-partition quintiles of the true pairs",
              "note": "Post-hoc stratification at the frozen thresholds; no primary "
                      "result depends on it.",
              "families": {}}
    rows = []

    for fam in ("sis", "pm"):
        frame = pd.read_csv(CORE / f"unified_predictions_0228_{fam}.csv.gz")
        cal_bg = frame[(frame.calibration_or_evaluation == "calibration") & (frame.label == 0)]
        cal_pos = frame[(frame.calibration_or_evaluation == "calibration") & (frame.label == 1)]
        ev_pos = frame[(frame.calibration_or_evaluation == "evaluation") & (frame.label == 1)]

        thr = {m: kth_threshold(cal_bg[f"{m}_score"].to_numpy(), TARGET) for m in MODELS}
        edges = np.unique(np.quantile(cal_pos.rho_min, np.linspace(0, 1, 6)))
        bins = pd.cut(ev_pos.rho_min, edges, include_lowest=True)

        blocks = np.sort(ev_pos.source_block_id.unique())
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(blocks), size=(REPS, len(blocks)))

        entries = []
        for category in bins.cat.categories:
            mask = (bins == category).to_numpy()
            subset = ev_pos.loc[mask]
            point = {m: float((subset[f"{m}_score"] >= thr[m]).mean()) for m in MODELS}

            by_block = {m: [subset.loc[subset.source_block_id == b, f"{m}_score"].to_numpy()
                            for b in blocks] for m in MODELS}
            deltas, effs = [], {m: [] for m in MODELS}
            for draw in draws:
                cur = {}
                for m in MODELS:
                    scores = np.concatenate([by_block[m][i] for i in draw])
                    cur[m] = (scores >= thr[m]).mean() if len(scores) else np.nan
                    effs[m].append(cur[m])
                deltas.append(cur["pi"] - cur["cqt_deit"])
            deltas = np.asarray(deltas)
            lo, hi = percentile_ci(deltas[~np.isnan(deltas)])

            entry = {
                "bin_left": float(category.left), "bin_right": float(category.right),
                "n": int(mask.sum()),
                "pi_resnet": point["pi"], "pi_resnet_ci": percentile_ci(
                    np.asarray(effs["pi"])[~np.isnan(effs["pi"])]),
                "cqt_deit": point["cqt_deit"], "cqt_deit_ci": percentile_ci(
                    np.asarray(effs["cqt_deit"])[~np.isnan(effs["cqt_deit"])]),
                "delta": point["pi"] - point["cqt_deit"],
                "delta_ci": [lo, hi],
                "resolved": bool(lo * hi > 0),
            }
            entries.append(entry)
            rows.append({"lens": fam.upper(), **entry})
        report["families"][fam] = {"thresholds": {MODELS[m]: float(thr[m]) for m in MODELS},
                                   "bin_edges": [float(e) for e in edges],
                                   "quintiles": entries}

    CORE.mkdir(parents=True, exist_ok=True)
    with (CORE / "snr_quintile_efficiency.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    pd.DataFrame(rows).to_csv(CORE / "snr_quintile_efficiency.csv", index=False)

    for fam in ("sis", "pm"):
        print(f"=== {fam.upper()}  efficiency at the frozen {TARGET:g} thresholds ===")
        print(f"  {'rho_min bin':>22}{'n':>6}{'PI':>8}{'CQT':>8}{'delta':>9}"
              f"{'95% CI':>18}  resolved")
        for e in report["families"][fam]["quintiles"]:
            print(f"  [{e['bin_left']:8.1f},{e['bin_right']:8.1f}]{e['n']:6}"
                  f"{100 * e['pi_resnet']:7.1f}%{100 * e['cqt_deit']:7.1f}%"
                  f"{100 * e['delta']:+8.1f}"
                  f"  [{100 * e['delta_ci'][0]:+6.1f},{100 * e['delta_ci'][1]:+6.1f}]"
                  f"   {'yes' if e['resolved'] else 'no'}")
        print()
    print(f"wrote {(CORE / 'snr_quintile_efficiency.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
