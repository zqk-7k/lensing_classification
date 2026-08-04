#!/usr/bin/env python3
"""Post-hoc robustness statistics for the appendix, from released per-pair scores only.

No model re-training and no new simulation: every number below is derived from
`results/core/unified_predictions_0228_{sis,pm}.csv.gz` and the frozen operating
points in `results/core/core_results.json`.

(A) Threshold-inclusive ("nested") block bootstrap. The locked analysis freezes
    thresholds on the calibration partition and bootstraps only the evaluation
    partition, so its intervals are conditional on those frozen thresholds. Here
    each replicate ALSO resamples the calibration source blocks and re-derives the
    threshold, propagating the finite-background calibration error. The threshold
    rule is the same empirical-quantile rule as the locked protocol -- the k-th
    largest background score with k = round(target_Pf * N) -- and the script
    asserts that it reproduces the published thresholds before bootstrapping.
    Point estimates remain the locked values; the bootstrap supplies only the
    wider interval.

(B) Fully detected-event analysis. The subset diagnostic in the locked analysis
    filters positives only. Here the background is filtered on the same criterion
    (both constituent events with rho >= 8), thresholds are re-calibrated on the
    filtered calibration background, and efficiency, achieved P_f, and the paired
    difference are recomputed for both networks.

(C) Amortized catalog-scale cost. The measured per-pair timings of
    `results/benchmarks/throughput/throughput.json` are re-expressed under
    caching, in which each event is preprocessed once and reused across the N-1
    pairs it enters. Inputs are read from the measurement file rather than
    hard-coded, so this section cannot drift from the benchmark.

Output: results/core/posthoc_robustness.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "results" / "core"
THROUGHPUT = ROOT / "results" / "benchmarks" / "throughput" / "throughput.json"
OUT = CORE / "posthoc_robustness.json"

SEED, REPS = 20260711, 10000
TARGETS = [(1e-2, "0.01"), (1e-3, "0.001"), (1e-4, "0.0001")]
MODELS = ["pi", "cqt_deit"]
MODEL_NAMES = {"pi": "pi_resnet", "cqt_deit": "cqt_deit"}
RHO_DETECTED = 8.0


def kth_threshold(scores, target):
    """Locked-protocol convention: k-th largest score, k = round(target * N)."""
    k = max(1, int(round(target * len(scores))))
    return np.partition(scores, -k)[-k]


def block_arrays(frame, blocks, column):
    return [frame[frame.source_block_id == b][column].to_numpy() for b in blocks]


def percentile_ci(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def amortized_cost():
    """Cached per-pair cost from the measured benchmark, with the crossover point.

    Both preprocessing figures in throughput.json cover the two segments of a pair,
    so the per-event cost is half of each. With caching, every event is preprocessed
    once and reused across the N-1 pairs it enters, giving

        c_pair(N) = c_gpu + 2 * c_event / (N - 1) = c_gpu + c_preproc_pair / (N - 1).
    """
    with THROUGHPUT.open(encoding="utf-8") as handle:
        bench = json.load(handle)
    pre = {
        "pi": bench["pi_resnet"]["preprocess_ms_per_pair_serial"],
        "cqt_deit": bench["cqt_deit"]["preprocess_ms_per_pair_serial_including_cqt"],
    }
    gpu = {m: bench[MODEL_NAMES[m]]["gpu_inference_ms_per_pair"] for m in MODELS}

    cost = {
        "source": "results/benchmarks/throughput/throughput.json",
        "hardware": bench["hardware"],
        "measured_ms_per_pair": {
            "preprocess_pi": pre["pi"],
            "preprocess_cqt_including_transform": pre["cqt_deit"],
            "forward_pi": gpu["pi"],
            "forward_cqt": gpu["cqt_deit"],
        },
        "derived_ms_per_event": {m: pre[m] / 2.0 for m in MODELS},
        "coldstart_ms_per_pair": {m: pre[m] + gpu[m] for m in MODELS},
        "note": "Preprocessing figures cover the two segments of a pair and are "
                "halved to per-event; with caching each event is preprocessed once "
                "and reused across the N-1 pairs it enters.",
    }
    cost["coldstart_ratio_cqt_over_pi"] = (
        cost["coldstart_ms_per_pair"]["cqt_deit"] / cost["coldstart_ms_per_pair"]["pi"]
    )
    cost["cached"] = {}
    for n_events in (10, 100, 1000, 10000):
        per_pair = {m: gpu[m] + pre[m] / (n_events - 1) for m in MODELS}
        cost["cached"][str(n_events)] = {
            "pairs": n_events * (n_events - 1) // 2,
            "pi_ms_per_pair": per_pair["pi"],
            "cqt_ms_per_pair": per_pair["cqt_deit"],
            "ratio_cqt_over_pi": per_pair["cqt_deit"] / per_pair["pi"],
        }
    # c_gpu_pi + p_pi/(N-1) == c_gpu_cqt + p_cqt/(N-1)
    cost["crossover_n_events"] = 1.0 + (pre["cqt_deit"] - pre["pi"]) / (gpu["pi"] - gpu["cqt_deit"])
    return cost


def main():
    with (CORE / "core_results.json").open(encoding="utf-8") as handle:
        core = json.load(handle)

    result = {
        "seed": SEED,
        "replicates": REPS,
        "rho_detected": RHO_DETECTED,
        "threshold_rule": "k-th largest background score, k = round(target_Pf * N_background)",
        "note": "Post-hoc secondary diagnostics from released per-pair scores; "
                "the locked primary analysis is unchanged.",
        "families": {},
    }

    for fam in ("sis", "pm"):
        frame = pd.read_csv(CORE / f"unified_predictions_0228_{fam}.csv.gz")
        cal_bg = frame[(frame.calibration_or_evaluation == "calibration") & (frame.label == 0)]
        ev_bg = frame[(frame.calibration_or_evaluation == "evaluation") & (frame.label == 0)]
        ev_pos = frame[(frame.calibration_or_evaluation == "evaluation") & (frame.label == 1)]
        ops = core["results"][fam]["models"]

        # Sanity: the k-th-largest rule must reproduce the published thresholds.
        drift = []
        for model in MODELS:
            published = ops[MODEL_NAMES[model]]["operating_points"]
            for target, key in TARGETS:
                if published[key]["threshold"] >= 1.0:
                    continue  # saturated score, no finite check available
                drift.append(abs(kth_threshold(cal_bg[f"{model}_score"].to_numpy(), target)
                                 - published[key]["threshold"]))
        family_out = {"threshold_reproduction_maxdiff": float(max(drift)) if drift else 0.0}

        cal_blocks = np.sort(cal_bg.source_block_id.unique())
        ev_blocks = np.sort(ev_pos.source_block_id.unique())
        cal_by_block = {m: block_arrays(cal_bg, cal_blocks, f"{m}_score") for m in MODELS}
        pos_by_block = {m: block_arrays(ev_pos, ev_blocks, f"{m}_score") for m in MODELS}
        rng = np.random.default_rng(SEED)

        # ---------- (A) threshold-inclusive bootstrap ----------
        reps = {key: [] for _, key in TARGETS}
        for _ in range(REPS):
            cal_draw = rng.integers(0, len(cal_blocks), size=len(cal_blocks))
            ev_draw = rng.integers(0, len(ev_blocks), size=len(ev_blocks))
            cal_rep = {m: np.concatenate([cal_by_block[m][i] for i in cal_draw]) for m in MODELS}
            pos_rep = {m: np.concatenate([pos_by_block[m][i] for i in ev_draw]) for m in MODELS}
            for target, key in TARGETS:
                eff = {m: (pos_rep[m] >= kth_threshold(cal_rep[m], target)).mean() for m in MODELS}
                reps[key].append(eff["pi"] - eff["cqt_deit"])

        nested = {}
        for _, key in TARGETS:
            locked = (ops["pi_resnet"]["operating_points"][key]["efficiency"]
                      - ops["cqt_deit"]["operating_points"][key]["efficiency"])
            draws = np.asarray(reps[key])
            nested[key] = {
                "locked_delta": float(locked),
                "nested_ci": percentile_ci(draws),
                "nested_median": float(np.median(draws)),
                "frac_replicates_le_zero": float((draws <= 0).mean()),
            }
        family_out["nested_bootstrap"] = nested

        # ---------- (B) fully detected-event recalibration ----------
        detected = lambda d: d[(d.rho_1 >= RHO_DETECTED) & (d.rho_2 >= RHO_DETECTED)]
        cal_det, ev_bg_det, ev_pos_det = detected(cal_bg), detected(ev_bg), detected(ev_pos)
        pos_det_by_block = {m: block_arrays(ev_pos_det, ev_blocks, f"{m}_score") for m in MODELS}
        recal = {
            "frac_cal_bg": float(len(cal_det) / len(cal_bg)),
            "frac_ev_bg": float(len(ev_bg_det) / len(ev_bg)),
            "frac_pos": float(len(ev_pos_det) / len(ev_pos)),
            "n_pos": int(len(ev_pos_det)),
            "n_ev_bg": int(len(ev_bg_det)),
        }
        draws = [rng.integers(0, len(ev_blocks), size=len(ev_blocks)) for _ in range(REPS)]
        for target, key in TARGETS:
            thr = {m: kth_threshold(cal_det[f"{m}_score"].to_numpy(), target) for m in MODELS}
            eff_reps = {m: [] for m in MODELS}
            delta_reps = []
            for draw in draws:
                current = {m: (np.concatenate([pos_det_by_block[m][i] for i in draw]) >= thr[m]).mean()
                           for m in MODELS}
                for m in MODELS:
                    eff_reps[m].append(current[m])
                delta_reps.append(current["pi"] - current["cqt_deit"])
            row = {}
            for m in MODELS:
                row[MODEL_NAMES[m]] = {
                    "threshold": float(thr[m]),
                    "efficiency": float((ev_pos_det[f"{m}_score"] >= thr[m]).mean()),
                    "ci": percentile_ci(eff_reps[m]),
                    "achieved_pf": float((ev_bg_det[f"{m}_score"] >= thr[m]).mean()),
                }
            row["delta_eff"] = row["pi_resnet"]["efficiency"] - row["cqt_deit"]["efficiency"]
            row["delta_ci"] = percentile_ci(delta_reps)
            recal[key] = row
        family_out["detected_event_recalibration"] = recal

        result["families"][fam] = family_out
        print(f"[{fam}] threshold reproduction max |diff| = "
              f"{family_out['threshold_reproduction_maxdiff']:.2e}")

    result["amortized_cost"] = amortized_cost()

    with OUT.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)

    print("\n=== (A) threshold-inclusive bootstrap, 10,000 replicates, locked quantile rule ===")
    for fam in ("sis", "pm"):
        for _, key in TARGETS:
            entry = result["families"][fam]["nested_bootstrap"][key]
            lo, hi = entry["nested_ci"]
            flag = "excludes 0" if lo * hi > 0 else "INCLUDES 0"
            print(f"  {fam} {key}: locked d={100 * entry['locked_delta']:+.1f} pp | "
                  f"nested CI [{100 * lo:+.1f},{100 * hi:+.1f}] | "
                  f"P(d<=0)={entry['frac_replicates_le_zero']:.4f}  {flag}")

    print("\n=== (B) fully detected-event, both networks ===")
    for fam in ("sis", "pm"):
        entry = result["families"][fam]["detected_event_recalibration"]
        print(f"  {fam}: cal bg {100 * entry['frac_cal_bg']:.1f}%, "
              f"eval bg {100 * entry['frac_ev_bg']:.1f}% (n={entry['n_ev_bg']}), "
              f"pos {100 * entry['frac_pos']:.1f}% (n={entry['n_pos']})")
        for _, key in TARGETS:
            row = entry[key]
            pi_row, cqt_row = row["pi_resnet"], row["cqt_deit"]
            print(f"    {key}: PI {100 * pi_row['efficiency']:.1f} "
                  f"[{100 * pi_row['ci'][0]:.1f},{100 * pi_row['ci'][1]:.1f}] | "
                  f"CQT {100 * cqt_row['efficiency']:.1f} "
                  f"[{100 * cqt_row['ci'][0]:.1f},{100 * cqt_row['ci'][1]:.1f}] | "
                  f"ach Pf {pi_row['achieved_pf']:.2e}/{cqt_row['achieved_pf']:.2e} | "
                  f"d {100 * row['delta_eff']:+.1f} "
                  f"[{100 * row['delta_ci'][0]:+.1f},{100 * row['delta_ci'][1]:+.1f}]")

    print("\n=== (C) amortized catalog-scale cost (ms per pair) ===")
    cost = result["amortized_cost"]
    print(f"  cold start: PI {cost['coldstart_ms_per_pair']['pi']:.3f} | "
          f"CQT {cost['coldstart_ms_per_pair']['cqt_deit']:.3f} | "
          f"ratio {cost['coldstart_ratio_cqt_over_pi']:.1f}x")
    for n_events, row in cost["cached"].items():
        print(f"  N={n_events:>5} cached: PI {row['pi_ms_per_pair']:.3f} | "
              f"CQT {row['cqt_ms_per_pair']:.3f} | ratio {row['ratio_cqt_over_pi']:.3f}")
    print(f"  crossover at N = {cost['crossover_n_events']:.1f} events")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
