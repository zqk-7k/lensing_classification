#!/usr/bin/env python3
"""Revision analyses on top of the locked core analysis, for one detector configuration.

Reads the unified predictions written by analyze_0228_core.py and the catalog metadata of
the realistic population, and writes to <work>/results/revision/:

  efficiency_vs_fap.csv      efficiency and achieved FAP of both models on a log grid of target
                             per-pair FAPs from 1e-1 to 1e-4 (pre-filter operating range), with
                             source-block bootstrap intervals; thresholds are calibration-background
                             upper-tail quantiles exactly as in the core analysis
  selection_functions_rev.csv  efficiency at the FAP 1e-3 and 1e-2 thresholds in bins of the
                             physical variables of the realistic population (quantile bins fixed on
                             calibration positives): y, flux ratio, fainter-image S/N, source
                             redshift, time delay, plus the image-type pair and the image multiplicity
  pm_robustness.json         if the point-mass robustness predictions exist: efficiency of the
                             galaxy-trained PI-ResNet on point-mass pairs at thresholds set on the
                             point-mass calibration background
  summary.json               headline numbers

Nothing here changes thresholds or checkpoints; it is descriptive, post hoc, and labeled so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"pi_resnet": "pi_score", "cqt_deit": "cqt_deit_score"}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def threshold(cal_bg_scores, fap):
    return float(np.quantile(cal_bg_scores, 1 - fap, method="higher"))


def block_boot(ev, score, thr, replicates, seed):
    blocks = sorted(ev.source_block_id.unique()); rng = np.random.RandomState(seed)
    parts = {b: ev[ev.source_block_id == b] for b in blocks}
    stat = {b: (int((parts[b].label == 1).sum()), int(((parts[b].label == 1) & (parts[b][score] >= thr)).sum()),
                int((parts[b].label == 0).sum()), int(((parts[b].label == 0) & (parts[b][score] >= thr)).sum())) for b in blocks}
    eff, fap = [], []
    for _ in range(replicates):
        s = rng.choice(blocks, len(blocks), replace=True)
        P = sum(stat[b][0] for b in s); D = sum(stat[b][1] for b in s); Nn = sum(stat[b][2] for b in s); F = sum(stat[b][3] for b in s)
        eff.append(D / P); fap.append(F / Nn)
    return [float(np.quantile(eff, 0.025)), float(np.quantile(eff, 0.975))], [float(np.quantile(fap, 0.025)), float(np.quantile(fap, 0.975))]


def efficiency_curve(data, out, replicates, seed):
    cal = data[data.calibration_or_evaluation == "calibration"]; ev = data[data.calibration_or_evaluation == "evaluation"]
    grid = np.logspace(-1, -4, 13)
    rows = []
    for model, score in MODELS.items():
        bg = cal.loc[cal.label == 0, score].to_numpy()
        for i, fap in enumerate(grid):
            thr = threshold(bg, fap)
            pos, neg = ev[ev.label == 1], ev[ev.label == 0]
            eff = float((pos[score] >= thr).mean()); ach = float((neg[score] >= thr).mean())
            eci, fci = block_boot(ev, score, thr, replicates, seed + i)
            rows.append({"model": model, "target_fap": float(fap), "threshold": thr, "efficiency": eff, "efficiency_ci_low": eci[0],
                         "efficiency_ci_high": eci[1], "achieved_fap": ach, "achieved_fap_ci_low": fci[0], "achieved_fap_ci_high": fci[1],
                         "hard_fap": float((neg.loc[neg.negative_type == "hard", score] >= thr).mean()),
                         "easy_fap": float((neg.loc[neg.negative_type == "easy", score] >= thr).mean()),
                         "n_positives": int(len(pos)), "n_background": int(len(neg)), "bootstrap_replicates": replicates})
    df = pd.DataFrame(rows); df.to_csv(out / "efficiency_vs_fap.csv", index=False)
    return df


def quantile_edges(values, count=5):
    inner = np.quantile(values, np.linspace(0, 1, count + 1)[1:-1])
    return np.unique(np.r_[-np.inf, inner, np.inf])


def selection(data, meta, out, faps=(1e-3, 1e-2)):
    cal = data[data.calibration_or_evaluation == "calibration"]; ev = data[data.calibration_or_evaluation == "evaluation"]
    pos_cal = cal[cal.label == 1].merge(meta, left_on="left_source_id", right_on="catalog_index", how="left", suffixes=("", "_meta"))
    pos_ev = ev[ev.label == 1].merge(meta, left_on="left_source_id", right_on="catalog_index", how="left", suffixes=("", "_meta"))
    assert pos_ev.zs.notna().all(), "pair_meta join failed"
    pos_ev["delay_days"] = pos_ev.delay_s.abs() / 86400.0; pos_cal["delay_days"] = pos_cal.delay_s.abs() / 86400.0
    pos_ev["type_pair"] = pos_ev.type_1.astype(int).astype(str) + "-" + pos_ev.type_2.astype(int).astype(str)
    pos_ev["multiplicity"] = np.where(pos_ev.n_images >= 4, "quad", "double")
    continuous = {"y": "y", "flux_ratio": "flux_ratio", "rho_min": "rho_min", "zs": "zs", "delay_days": "delay_days",
                  "mass_1_det": "mass_1"}
    rows = []
    for fap in faps:
        thr = {m: threshold(cal.loc[cal.label == 0, s].to_numpy(), fap) for m, s in MODELS.items()}
        for var, col in continuous.items():
            edges = quantile_edges(pos_cal[col].to_numpy())
            bins = pd.cut(pos_ev[col], edges, include_lowest=True, duplicates="drop")
            for model, score in MODELS.items():
                for cat in bins.cat.categories:
                    mask = bins == cat; n = int(mask.sum()); k = int((pos_ev.loc[mask, score] >= thr[model]).sum())
                    lo, hi = wilson(k, n)
                    rows.append({"target_fap": fap, "model": model, "variable": var, "bin": str(cat), "bin_left": float(cat.left),
                                 "bin_right": float(cat.right), "n": n, "detected": k, "efficiency": k / n if n else np.nan,
                                 "ci_low": lo, "ci_high": hi, "median_in_bin": float(pos_ev.loc[mask, col].median()) if n else np.nan})
        for var in ("type_pair", "multiplicity"):
            for model, score in MODELS.items():
                for cat, part in pos_ev.groupby(var):
                    n = len(part); k = int((part[score] >= thr[model]).sum()); lo, hi = wilson(k, n)
                    rows.append({"target_fap": fap, "model": model, "variable": var, "bin": str(cat), "bin_left": np.nan, "bin_right": np.nan,
                                 "n": n, "detected": k, "efficiency": k / n if n else np.nan, "ci_low": lo, "ci_high": hi, "median_in_bin": np.nan})
    df = pd.DataFrame(rows); df.to_csv(out / "selection_functions_rev.csv", index=False)
    return df


def detected_subset(data, out, replicates, seed, rho_th=8.0):
    """Pairs whose two members both reach single-detector S/N >= rho_th in the stored channel.

    The background is filtered the same way and the thresholds are re-derived on the filtered
    calibration background, as in the fully-detected-event analysis of posthoc_robustness.
    Post hoc: separates pairs the network could not see from pairs it mis-ranked."""
    sub = data[(data.rho_1 >= rho_th) & (data.rho_2 >= rho_th)]
    cal = sub[sub.calibration_or_evaluation == "calibration"]; ev = sub[sub.calibration_or_evaluation == "evaluation"]
    rows = []
    for i, fap in enumerate((1e-2, 1e-3, 1e-4)):
        for model, score in MODELS.items():
            thr = threshold(cal.loc[cal.label == 0, score].to_numpy(), fap)
            pos, neg = ev[ev.label == 1], ev[ev.label == 0]
            eci, fci = block_boot(ev, score, thr, replicates, seed + 900 + i)
            rows.append({"model": model, "target_fap": fap, "rho_threshold": rho_th, "threshold": thr,
                         "efficiency": float((pos[score] >= thr).mean()), "efficiency_ci_low": eci[0], "efficiency_ci_high": eci[1],
                         "achieved_fap": float((neg[score] >= thr).mean()), "achieved_fap_ci_low": fci[0], "achieved_fap_ci_high": fci[1],
                         "n_positives": int(len(pos)), "n_background": int(len(neg)),
                         "positive_fraction_retained": float(len(pos) / (data[(data.calibration_or_evaluation == "evaluation") & (data.label == 1)]).shape[0])})
    df = pd.DataFrame(rows); df.to_csv(out / f"detected_subset_rho{int(rho_th)}.csv", index=False)
    return df


def pm_robustness(pred_dir, manifest_dir, data_sis, out, replicates, seed):
    path = pred_dir / "pi_predictions_0228_pm_checkpoint_sis.csv.gz"
    if not path.exists():
        return {"status": "absent", "expected": str(path)}
    pred = pd.read_csv(path)
    man = pd.concat([pd.read_csv(manifest_dir / f"0228_pm_{p}_pairs.csv.gz") for p in ("calibration", "evaluation")], ignore_index=True)
    pm = pred.merge(man[["pair_id", "y", "mu_plus", "mu_minus", "flux_ratio", "rho_1", "rho_2", "rho_min", "rho_max", "left_source_id"]],
                    on="pair_id", validate="one_to_one")
    result = {"status": "complete", "checkpoint": "pi_resnet trained on the galaxy-lens (SIS-labelled) 0222 catalog",
              "target_population": "point-mass pairs on the same BBH sources, p(y) ~ y", "operating_points": {}}
    cal = pm[pm.calibration_or_evaluation == "calibration"]; ev = pm[pm.calibration_or_evaluation == "evaluation"]
    sis_cal = data_sis[data_sis.calibration_or_evaluation == "calibration"]; sis_ev = data_sis[data_sis.calibration_or_evaluation == "evaluation"]
    for i, fap in enumerate((1e-2, 1e-3, 1e-4)):
        thr_pm = threshold(cal.loc[cal.label == 0, "pi_score"].to_numpy(), fap)
        thr_sis = threshold(sis_cal.loc[sis_cal.label == 0, "pi_score"].to_numpy(), fap)
        eci, fci = block_boot(ev, "pi_score", thr_pm, replicates, seed + 500 + i)
        result["operating_points"][str(fap)] = {
            "pm_threshold": thr_pm, "pm_efficiency": float((ev.loc[ev.label == 1, "pi_score"] >= thr_pm).mean()), "pm_efficiency_ci": eci,
            "pm_achieved_fap": float((ev.loc[ev.label == 0, "pi_score"] >= thr_pm).mean()), "pm_achieved_fap_ci": fci,
            "pm_efficiency_at_galaxy_threshold": float((ev.loc[ev.label == 1, "pi_score"] >= thr_sis).mean()),
            "pm_fap_at_galaxy_threshold": float((ev.loc[ev.label == 0, "pi_score"] >= thr_sis).mean()),
            "galaxy_threshold": thr_sis, "galaxy_efficiency": float((sis_ev.loc[sis_ev.label == 1, "pi_score"] >= thr_sis).mean()),
            "n_pm_positives": int((ev.label == 1).sum()), "n_pm_background": int((ev.label == 0).sum())}
    (out / "pm_robustness.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="revision work root (results/core, results/predictions, manifests)")
    ap.add_argument("--data", required=True, help="revision data root (SIS_data_0228/pair_meta.csv)")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()
    work, data = Path(args.work), Path(args.data)
    out = work / "results" / "revision"; out.mkdir(parents=True, exist_ok=True)
    uni = pd.read_csv(work / "results/core/unified_predictions_0228_sis.csv.gz")
    meta = pd.read_csv(data / "SIS_data_0228/pair_meta.csv")
    curve = efficiency_curve(uni, out, args.replicates, args.seed)
    sel = selection(uni, meta, out)
    det = detected_subset(uni, out, args.replicates, args.seed)
    pm = pm_robustness(work / "results/predictions", work / "manifests/0228_pairs_pm", uni, out, args.replicates, args.seed)
    core = json.loads((work / "results/core/core_results.json").read_text())["results"]["sis"]
    summary = {"core_fixed_fap": {m: {fp: {k: core["models"][m]["operating_points"][fp][k] for k in ("efficiency", "efficiency_ci", "achieved_fpp")}
                                      for fp in ("0.01", "0.001", "0.0001")} for m in MODELS},
               "core_paired": core["paired"],
               "efficiency_at_1e-1": {m: float(curve[(curve.model == m) & (np.isclose(curve.target_fap, 1e-1))].efficiency.iloc[0]) for m in MODELS},
               "detected_subset_rho8": det[det.target_fap == 1e-3].set_index("model")[["efficiency", "efficiency_ci_low", "efficiency_ci_high", "n_positives", "positive_fraction_retained"]].to_dict("index"),
               "pm_robustness": pm, "bootstrap_replicates_extras": args.replicates,
               "n_evaluation_positives": int((uni[uni.calibration_or_evaluation == "evaluation"].label == 1).sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
