#!/usr/bin/env python3
"""v3 point-mass lens-profile robustness set on the evaluation sources of a v3 root.

Same idea as generate_pm_robustness_rev.py (v2), in the v3 layout (all detectors, fmin from the
config, two streams, float16, stored noisy data): each galaxy-lensed source of SIS_data_0228 (same
BBH parameters and spins, sky position, arrival time and lens redshift) is lensed instead by a point mass

  M_L ~ log-uniform [--mlens-min, --mlens-max] Msun,  y ~ p(y) = 2y / y_max^2 on (0, y_max]
  mu+- = 1/2 +- (y^2 + 2) / (2 y sqrt(y^2 + 4)),  t_d = 4 G M_L (1 + z_l) / c^3 [y sqrt(y^2+4)/2 + ln((sqrt(y^2+4)+y)/(sqrt(y^2+4)-y))]
  image 1: mu+, Morse index 0, at the source time; image 2: mu-, Morse index 1/2, t_d later.

A draw is accepted when t_d >= --min-delay-s and both images reach network optimal S/N >= --snr-th
from 20 Hz at their own arrival times (the S/N definition of the ler selection), up to --max-tries
draws per source; otherwise the source is skipped. It is a robustness test of the galaxy-trained
networks, not a target population. Output: <root>/PM_data_0228/ in the layout of SIS_data_0228.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_v3 as G  # noqa: E402

G_SI, C_SI, MSUN = 6.67430e-11, 299792458.0, 1.98847e30
KEYS = ("mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "luminosity_distance",
        "theta_jn", "psi", "phase", "ra", "dec")
PM_SEED_OFFSET = 70_000_000
NAMES = [f"{p}_{s}_{k}" for p in ("sig", "data") for s in "AB" for k in (1, 2)]


def pm_lens(y, m_l, z_l):
    s = np.sqrt(y * y + 4.0)
    mu_p = 0.5 + (y * y + 2.0) / (2.0 * y * s)
    mu_m = 0.5 - (y * y + 2.0) / (2.0 * y * s)
    t_d = 4.0 * G_SI * m_l * MSUN * (1.0 + z_l) / C_SI ** 3 * (0.5 * y * s + np.log((s + y) / (s - y)))
    return float(mu_p), float(mu_m), float(t_d)


def net_snr20(fd, params):
    tot = 0.0
    for j, ifo in enumerate(G.W["ifos"]):
        ifo.strain_data.start_time = params["geocent_time"] - (G.FRAME_S - G.TAIL_S)
        p2 = np.abs(ifo.get_detector_response(fd, params) * G.W["whiten"][j]) ** 2
        tot += 2.0 / G.N * float((p2 * G.W["band20"]).sum())
    return float(np.sqrt(tot))


def work(task):
    start, stop, rows, n_total, out_root, opts = task
    arrs = G.open_arrays(Path(out_root) / "PM_data_0228", NAMES, n_total, len(G.W["ifos"]), "r+")
    recs = []
    for i in range(start, stop):
        row = rows[i - start]
        rec = {"catalog_index": i, "source_catalog_index": int(row["catalog_index"]), "accepted": False}
        base = {k: float(row[k]) for k in KEYS}
        t0 = float(row["geocent_time"])
        try:
            mc = (base["mass_1"] * base["mass_2"]) ** 0.6 / (base["mass_1"] + base["mass_2"]) ** 0.2
            G.W["wfg"].waveform_arguments["minimum_frequency"] = max(G.W["fmin"], float(G.f_at_time(mc, G.MAX_INSPIRAL_S)))
            fd = G.W["wfg"].frequency_domain_strain(dict(base, geocent_time=t0, magnification=1.0, morse_index=0.0))
            rng = np.random.RandomState(opts["seed"] + 1000 * i)
            acc = None
            for attempt in range(opts["max_tries"]):
                y = opts["y_max"] * np.sqrt(rng.uniform())
                m_l = 10 ** rng.uniform(np.log10(opts["mlens_min"]), np.log10(opts["mlens_max"]))
                mu_p, mu_m, t_d = pm_lens(y, m_l, float(row["zl"]))
                if t_d < opts["min_delay_s"]:
                    continue
                s1 = np.sqrt(abs(mu_p)) * net_snr20(fd, dict(base, geocent_time=t0))
                s2 = np.sqrt(abs(mu_m)) * net_snr20(fd, dict(base, geocent_time=t0 + t_d))
                if s1 >= opts["snr_th"] and s2 >= opts["snr_th"]:
                    acc = (y, m_l, mu_p, mu_m, t_d, attempt + 1)
                    break
            if acc is not None:
                y, m_l, mu_p, mu_m, t_d, tries = acc
                rec.update(accepted=True, tries=tries, m_lens_msun=m_l, y=y, mu_1=mu_p, mu_2=mu_m, morse_1=0.0, morse_2=0.5,
                           delay_s=t_d, zl=float(row["zl"]), zs=float(row["zs"]),
                           **{k: float(row[k]) for k in G.SOURCE_COLUMNS}, mass_1=base["mass_1"], mass_2=base["mass_2"])
                for k, (mu, morse, t) in enumerate(((mu_p, 0.0, t0), (mu_m, 0.5, t0 + t_d)), start=1):
                    seed = G.ROLE_SEED["0228"] + PM_SEED_OFFSET + 10 * i + k
                    ev = G.make_event(dict(base, geocent_time=t, magnification=mu, morse_index=morse), seed)
                    for s in "AB":
                        arrs[f"sig_{s}_{k}"][i] = ev[f"sig_{s}"]; arrs[f"data_{s}_{k}"][i] = ev[f"data_{s}"]
                    rec[f"snr_net_{k}"] = float(np.sqrt((ev["snr_det"] ** 2).sum()))
                    rec[f"snr_net20_{k}"] = float(np.sqrt((ev["snr20_det"] ** 2).sum()))
                    rec[f"snr_best_det_{k}"] = float(ev["snr_det"].max())
                    rec[f"best_det_{k}"] = G.W["names"][int(np.argmax(ev["snr_det"]))]
                    rec[f"snr_win_A_{k}"] = ev["snr_win_A"]; rec[f"snr_win_B_{k}"] = ev["snr_win_B"]
                    rec[f"f_start_{k}"] = ev["f_start"]; rec[f"noise_seed_{k}"] = seed
                    for name, v in zip(G.W["names"], ev["snr_det"]):
                        rec[f"snr_{name}_{k}"] = float(v)
        except Exception as err:  # waveform failure: the source is skipped and recorded
            rec["failure"] = f"{type(err).__name__}: {str(err)[:200]}"; rec["accepted"] = False
        recs.append(rec)
    for a in arrs.values():
        a.flush()
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(G.CONFIGS), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mlens-min", type=float, default=1e8)
    ap.add_argument("--mlens-max", type=float, default=1e10)
    ap.add_argument("--y-max", type=float, default=1.0)
    ap.add_argument("--snr-th", type=float, default=8.0)
    ap.add_argument("--min-delay-s", type=float, default=G.FRAME_S)
    ap.add_argument("--max-tries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--approximant", default="IMRPhenomXPHM")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--n-max", type=int, default=None, help="first n sources only (tests)")
    a = ap.parse_args()
    t0 = time.time()
    root = Path(a.root)
    src = pd.read_csv(root / "SIS_data_0228" / "pair_meta.csv")
    if a.n_max:
        src = src.iloc[:a.n_max]
    rows, n = src.to_dict("records"), len(src)
    base = root / "PM_data_0228"; base.mkdir(parents=True, exist_ok=True)
    n_det = len(G.CONFIGS[a.config]["psd"])
    G.open_arrays(base, NAMES, n, n_det, "w+")
    opts = dict(seed=a.seed, mlens_min=a.mlens_min, mlens_max=a.mlens_max, y_max=a.y_max, snr_th=a.snr_th,
                min_delay_s=a.min_delay_s, max_tries=a.max_tries)
    tasks = [(s, min(n, s + a.chunk), rows[s:min(n, s + a.chunk)], n, str(root), opts) for s in range(0, n, a.chunk)]
    recs = []
    with Pool(a.workers, initializer=G.init_worker, initargs=(a.config, a.approximant)) as pool:
        for part in pool.imap_unordered(work, tasks):
            recs += part
            print(f"  {len(recs)}/{n} sources {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(recs).sort_values("catalog_index").reset_index(drop=True)
    df[["catalog_index", "source_catalog_index", "accepted"] + (["failure"] if "failure" in df else [])].to_csv(
        base / "acceptance.csv", index=False)
    keep = df.index[df.accepted].to_numpy()
    G.compact(base, NAMES, keep)
    acc = df[df.accepted].reset_index(drop=True).copy()
    acc["catalog_index"] = np.arange(len(acc))
    acc.to_csv(base / "pair_meta.csv", index=False)
    pd.DataFrame({"mu_0": acc.mu_1, "mu_1": acc.mu_2, "t_d": acc.delay_s}).to_csv(base / "lens.csv", index=False)
    pd.DataFrame({"z_l": acc.zl, "z_s": acc.zs, "M_L(Msun)": acc.m_lens_msun, "y": acc.y}).to_csv(base / "lens_params.csv", index=False)
    acc[G.SOURCE_COLUMNS].to_csv(base / "source_samples.csv", index=False)
    for k in (1, 2):
        np.save(base / f"PM_optimal_SNR_{k}.npy", acc[f"snr_net_{k}"].to_numpy(np.float64))
    import bilby, lal
    meta = {"config": a.config, "family_label": "PM", "purpose": "lens-profile robustness test of the galaxy-trained networks; not a target population",
            "sources": "the BBH sources of SIS_data_0228 (parameters, spins, sky position, arrival time, lens redshift)",
            "m_lens_prior": f"log-uniform [{a.mlens_min:g}, {a.mlens_max:g}] Msun", "y_prior": f"p(y) = 2y/y_max^2 on (0, {a.y_max}]",
            "acceptance": "t_d >= min_delay_s and both images network optimal S/N >= snr_th from 20 Hz at their own arrival times",
            "snr_th": a.snr_th, "min_delay_s": a.min_delay_s, "max_tries": a.max_tries, "seed": a.seed,
            "noise_seed": f"ROLE_SEED['0228'] + {PM_SEED_OFFSET} + 10 i + k",
            "n_sources": int(n), "n_accepted": int(len(acc)), "n_failed": int(df["failure"].notna().sum()) if "failure" in df else 0,
            "mean_tries_accepted": float(acc.tries.mean()) if len(acc) else None, "layout": "as SIS_data_0228 (generate_v3.py)",
            "software": {"bilby": bilby.__version__, "lal": lal.__version__, "numpy": np.__version__}, "wall_s": time.time() - t0}
    (base / "generation_meta_pm_v3.json").write_text(json.dumps(meta, indent=2))
    print(f"accepted {len(acc)}/{n} sources in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
