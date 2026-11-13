"""Realistic lensed and unlensed BBH populations with ``ler`` (ApJS revision, server run).

Derived from an earlier script, ler_realistic_population.py. Differences:
  * explicit numpy seed, recorded in the meta file;
  * draws one large sample per configuration and writes the full ler JSON; the
    split into the two catalogues (training/"0222" role and
    threshold-setting+measurement/"0228" role) is made later, by system index,
    in generate_strain_rev.py, so both catalogues come from one population draw;
  * work directory on the data disk.

  O4 : H1, L1 (aLIGO O4 high ASD), V1 (AdV ASD)
  ET : triangular Einstein Telescope, ET-D
Detection (population level): ler's pdet per image, network optimal SNR >= snr_th,
at least two detected images for a lensed system.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

CONFIGS = {
    "O4": dict(ifos=["L1", "H1", "V1"],
               psds={"L1": "aLIGO_O4_high_asd.txt", "H1": "aLIGO_O4_high_asd.txt", "V1": "AdV_asd.txt"}),
    "ET": dict(ifos=["ET"], psds={"ET": "ET_D_psd.txt"}),
}


def build_ler(config, work, npool, snr_th, mtot_max):
    from ler.rates import LeR
    det = CONFIGS[config]
    return LeR(
        npool=npool, z_min=0.0, z_max=10.0, event_type="BBH", lens_type="epl_shear_galaxy",
        interpolator_directory=str(work / "interpolator_json"), ler_directory=str(work / f"ler_data_{config}"),
        verbose=False, include_redundant_parameters=True,
        snr_method="interpolation_no_spins", waveform_approximant="IMRPhenomD", minimum_frequency=20.0,
        mtot_min=2.0, mtot_max=mtot_max, mtot_resolution=400, ratio_resolution=40,
        ifos=det["ifos"], psds=det["psds"],
        pdet_kwargs=dict(snr_th=snr_th, snr_th_net=snr_th, pdet_type="boolean", distribution_type="noncentral_chi2",
                         include_optimal_snr=True, include_observed_snr=False, return_only_net_results=True),
    )


def lensed_summary(lp, snr_th):
    mu = np.asarray(lp["magnifications"], float); dt = np.asarray(lp["time_delays"], float)
    itype = np.asarray(lp["image_type"], float); snr = np.asarray(lp["optimal_snr_net"], float)
    pdet = np.asarray(lp["pdet_net"], float) if "pdet_net" in lp else None
    rows = []
    for i in range(mu.shape[0]):
        det = np.where(pdet[i] >= 0.5)[0] if pdet is not None else np.where(np.nan_to_num(snr[i]) >= snr_th)[0]
        if len(det) < 2:
            continue
        order = det[np.argsort(-np.abs(mu[i, det]))]; a, b = order[0], order[1]
        rows.append(dict(system_index=i, zs=lp["zs"][i], zl=lp["zl"][i], sigma=lp["sigma"][i], n_images=lp["n_images"][i],
                         n_detected=len(det), mass_1_source=lp["mass_1_source"][i], mass_2_source=lp["mass_2_source"][i],
                         mu_bright=mu[i, a], mu_faint=mu[i, b], type_bright=itype[i, a], type_faint=itype[i, b],
                         snr_bright=snr[i, a], snr_faint=snr[i, b], delay_s=abs(dt[i, b] - dt[i, a]),
                         flux_ratio=abs(mu[i, a] / mu[i, b])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--n-lensed", type=int, default=6000)
    ap.add_argument("--n-unlensed", type=int, default=10000)
    ap.add_argument("--snr-th", type=float, default=8.0)
    ap.add_argument("--mtot-max", type=float, default=500.0)
    ap.add_argument("--npool", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--skip-rates", action="store_true")
    args = ap.parse_args()
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    np.random.seed(args.seed)
    ler = build_ler(args.config, work, args.npool, args.snr_th, args.mtot_max)
    print(f"[{args.config}] LeR ready after {time.time() - t0:.0f} s", flush=True)
    meta = {"config": args.config, "ifos": CONFIGS[args.config]["ifos"], "psds": CONFIGS[args.config]["psds"],
            "snr_threshold": args.snr_th, "seed": args.seed, "ler_version": __import__("ler").__version__,
            "lens_type": "epl_shear_galaxy", "event_type": "BBH", "z_max": 10.0,
            "detectable_lensed_condition": "two images with network optimal SNR >= threshold (ler pdet_net)",
            "time_window_days": 365.0}
    if not args.skip_rates:
        t1 = time.time(); np.random.seed(args.seed + 1)
        ler.unlensed_cbc_statistics(size=args.batch_size, resume=False, output_jsonfile=f"unlensed_param_{args.config}.json")
        ru = ler.unlensed_rate(output_jsonfile=f"unlensed_param_detectable_{args.config}.json")
        ru = float(ru[0]) if isinstance(ru, (tuple, list)) else float(ru)
        np.random.seed(args.seed + 2)
        ler.lensed_cbc_statistics(size=args.batch_size, resume=False, output_jsonfile=f"lensed_param_{args.config}.json")
        rl = ler.lensed_rate(output_jsonfile=f"lensed_param_detectable_{args.config}.json", pdet_threshold=[0.5, 0.5], num_img=[1, 1])
        rl = float(rl[0]) if isinstance(rl, (tuple, list)) else float(rl)
        meta.update(unlensed_rate_per_yr=ru, lensed_rate_per_yr=rl, lensed_fraction=rl / ru if ru > 0 else None, rate_wall_s=time.time() - t1)
        print(f"[{args.config}] rates: unlensed {ru:.1f}/yr, lensed {rl:.3f}/yr, fraction {rl / ru:.2e}", flush=True)
    t2 = time.time(); np.random.seed(args.seed + 3)
    lp = ler.selecting_n_lensed_detectable_events(size=args.n_lensed, batch_size=args.batch_size, pdet_threshold=[0.5, 0.5],
                                                  num_img=[1, 1], resume=False, output_jsonfile=f"n_lensed_detectable_{args.config}.json",
                                                  meta_data_file=f"meta_lensed_{args.config}.json", trim_to_size=True)
    lp = lp[1] if isinstance(lp, tuple) else lp
    summ = lensed_summary(lp, args.snr_th); summ.to_csv(work / f"{args.config}_lensed_summary.csv", index=False)
    print(f"[{args.config}] {len(summ)} lensed systems with two detected images in {time.time() - t2:.0f} s", flush=True)
    t3 = time.time(); np.random.seed(args.seed + 4)
    up = ler.selecting_n_unlensed_detectable_events(size=args.n_unlensed, batch_size=args.batch_size, pdet_threshold=0.5, resume=False,
                                                    output_jsonfile=f"n_unlensed_detectable_{args.config}.json",
                                                    meta_data_file=f"meta_unlensed_{args.config}.json", trim_to_size=True)
    up = up[1] if isinstance(up, tuple) else up
    print(f"[{args.config}] {len(np.asarray(up['zs']))} unlensed detectable events in {time.time() - t3:.0f} s", flush=True)
    meta.update(n_lensed_systems=int(len(summ)), n_unlensed=int(len(np.asarray(up["zs"]))), total_wall_s=time.time() - t0)
    (work / f"{args.config}_population_meta.json").write_text(json.dumps(meta, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
