#!/usr/bin/env python3
"""Point-mass lens-profile robustness catalog on the realistic source population.

Point masses with uniform impact parameters are not an astrophysical lens
population. The point-mass class is kept only as a
*lens-profile robustness test* on the same realistic BBH sources, with p(y) proportional to y,
evaluated by the galaxy-trained network.

This script takes the sources of the galaxy-lensed systems of a revision catalog
(`<root>/SIS_data_<role>/pair_meta.csv`, written by generate_strain_rev.py: same BBH
parameters, sky position, arrival time, and the lens redshift of the galaxy lens) and
replaces the galaxy lens by a point mass:

  M_L  ~ log-uniform in [--mlens-min, --mlens-max] solar masses   (default 1e8 .. 1e10, as in the original PM catalogs)
  y    ~ p(y) = 2 y / y_max^2 on (0, y_max]                       (uniform in the area of the source plane)
  mu+  = 1/2 + (y^2 + 2) / (2 y sqrt(y^2 + 4)),   mu- = 1/2 - (y^2 + 2) / (2 y sqrt(y^2 + 4))
  t_d  = 4 G M_L (1 + z_l) / c^3 * [ y sqrt(y^2 + 4) / 2 + ln((sqrt(y^2 + 4) + y) / (sqrt(y^2 + 4) - y)) ]
  Morse phases 0 (image 1, mu+) and 1/2 (image 2, mu-)

A (M_L, y) draw is accepted when both images reach network optimal S/N >= --snr-th and
the delay exceeds --min-delay-s; up to --max-tries draws per source, otherwise the source
is skipped. Everything else (segments, whitening, peak alignment, stored channel) is the
code of generate_strain_rev.py. Output: `<root>/PM_data_<role>/` in the repository layout.
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
import generate_strain_rev as g  # noqa: E402

G_SI, C_SI, MSUN = 6.67430e-11, 299792458.0, 1.98847e30
PARAM_KEYS = ("mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
              "luminosity_distance", "theta_jn", "psi", "phase", "ra", "dec")


def pm_lens(y: float, m_l_msun: float, z_l: float):
    s = np.sqrt(y * y + 4.0)
    mu_plus = 0.5 + (y * y + 2.0) / (2.0 * y * s)
    mu_minus = 0.5 - (y * y + 2.0) / (2.0 * y * s)
    t_d = 4.0 * G_SI * m_l_msun * MSUN * (1.0 + z_l) / C_SI ** 3 * (0.5 * y * s + np.log((s + y) / (s - y)))
    return float(mu_plus), float(mu_minus), float(t_d)


def optimal_snr(wfg, params: dict) -> tuple[np.ndarray, float]:
    """Per-detector and network optimal S/N without noise generation."""
    ifos, fs, N = g.W["ifos"], g.W["fs"], g.W["N"]
    fd = wfg.frequency_domain_strain(params)
    single = np.empty(len(ifos))
    for j, ifo in enumerate(ifos):
        ifo.set_strain_data_from_zero_noise(sampling_frequency=fs, duration=N / fs,
                                            start_time=params["geocent_time"] - (N / fs - g.PEAK_OFFSET_SAMPLES / fs))
        response = ifo.get_detector_response(fd, params)
        single[j] = np.sqrt(abs(float(np.real(ifo.optimal_snr_squared(response)))))
    return single, float(np.sqrt(np.sum(single ** 2)))


def work(task):
    role, start, stop, rows, n_total, opts = task
    out = g.W["out_root"]; N = g.W["N"]; names = g.W["names"]; ref = g.W["ref_idx"]
    base = out / f"PM_data_{role}"
    results = []
    for i in range(start, stop):
        row = rows[i - start]
        rng = np.random.RandomState(opts["seed"] + 1000 * i)
        params_base = {k: float(row[k]) for k in PARAM_KEYS}
        accepted = None
        for attempt in range(opts["max_tries"]):
            y = opts["y_max"] * np.sqrt(rng.uniform())
            m_l = 10 ** rng.uniform(np.log10(opts["mlens_min"]), np.log10(opts["mlens_max"]))
            mu_p, mu_m, t_d = pm_lens(y, m_l, float(row["zl"]))
            if t_d < opts["min_delay_s"]:
                continue
            snrs = []
            for k, (mu, morse) in enumerate(((mu_p, 0.0), (mu_m, 0.5)), start=1):
                params = dict(params_base, geocent_time=float(row["geocent_time"]) + (t_d if k == 2 else 0.0),
                              magnification=mu, morse_index=morse)
                snrs.append(optimal_snr(g.W["wfg_lensed"], params))
            if all(net >= opts["snr_th"] for _, net in snrs):
                accepted = (y, m_l, mu_p, mu_m, t_d, attempt + 1, snrs)
                break
        if accepted is None:
            results.append({"catalog_index": i, "source_system_index": int(row["system_index"]), "accepted": False})
            continue
        y, m_l, mu_p, mu_m, t_d, tries, _ = accepted
        rec = {"catalog_index": i, "accepted": True, "tries": tries, "source_system_index": int(row["system_index"]),
               **{k: float(row[k]) for k in g.SOURCE_COLUMNS + ["zs", "mass_1", "mass_2", "zl"]},
               "m_lens_msun": m_l, "y": y, "mu_1": mu_p, "mu_2": mu_m, "type_1": 1, "type_2": 2, "morse_1": 0.0, "morse_2": 0.5,
               "delay_s": t_d}
        for k, (mu, morse) in enumerate(((mu_p, 0.0), (mu_m, 0.5)), start=1):
            params = dict(params_base, geocent_time=float(row["geocent_time"]) + (t_d if k == 2 else 0.0),
                          magnification=mu, morse_index=morse)
            seed = g.ROLE_SEED[role] + 70_000_000 + 10 * i + k
            data, sig, snr_single, snr_net = g.inject(g.W["wfg_lensed"], params, seed)
            j = ref[int(np.argmax(snr_single[ref]))]
            g.array(base / f"PM_data_strain_{k}.npy", (n_total, N), np.float32)[i] = data[j]
            g.array(base / f"PM_h_strain_{k}.npy", (n_total, N), np.float32)[i] = sig[j]
            g.array(base / f"PM_optimal_SNR_{k}.npy", (n_total,), np.float64)[i] = snr_single[j]
            rec[f"channel_{k}"] = names[j]; rec[f"snr_channel_{k}"] = float(snr_single[j]); rec[f"snr_net_bilby_{k}"] = snr_net
            for name, value in zip(names, snr_single):
                rec[f"snr_{name}_{k}"] = float(value)
            rec[f"noise_seed_{k}"] = seed
        results.append(rec)
    for a in g.W["arrays"].values():
        a.flush()
    return role, results


def compact(base: Path, prefix: str, n_written: int, keep: np.ndarray, N: int):
    """Drop the rows of skipped sources so the arrays and tables stay aligned."""
    for name, shape_tail, dtype in ((f"{prefix}_data_strain", (N,), np.float32), (f"{prefix}_h_strain", (N,), np.float32),
                                    (f"{prefix}_optimal_SNR", (), np.float64)):
        for k in (1, 2):
            path = base / f"{name}_{k}.npy"
            src = np.load(path, mmap_mode="r")
            tmp = base / f"{name}_{k}.tmp.npy"
            dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=(len(keep),) + shape_tail)
            for new, old in enumerate(keep):
                dst[new] = src[old]
            dst.flush(); del dst, src
            tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(g.DETECTORS), required=True)
    ap.add_argument("--root", required=True, help="revision data root holding SIS_data_<role>/pair_meta.csv")
    ap.add_argument("--role", default="0228")
    ap.add_argument("--mlens-min", type=float, default=1e8)
    ap.add_argument("--mlens-max", type=float, default=1e10)
    ap.add_argument("--y-max", type=float, default=1.0)
    ap.add_argument("--snr-th", type=float, default=8.0)
    ap.add_argument("--min-delay-s", type=float, default=60.0)
    ap.add_argument("--max-tries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--duration", type=float, default=24.0)
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--approximant", default="IMRPhenomXPHM")
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--chunk", type=int, default=25)
    args = ap.parse_args()
    t0 = time.time()
    root = Path(args.root); role = args.role
    src = pd.read_csv(root / f"SIS_data_{role}" / "pair_meta.csv")
    if args.n_max:
        src = src.iloc[:args.n_max]
    rows = src.to_dict("records"); n = len(rows)
    N = int(args.duration * args.fs)
    base = root / f"PM_data_{role}"; base.mkdir(parents=True, exist_ok=True)
    for k in (1, 2):
        g.create(base / f"PM_data_strain_{k}.npy", (n, N), np.float32)
        g.create(base / f"PM_h_strain_{k}.npy", (n, N), np.float32)
        g.create(base / f"PM_optimal_SNR_{k}.npy", (n,), np.float64)
    opts = dict(seed=args.seed, mlens_min=args.mlens_min, mlens_max=args.mlens_max, y_max=args.y_max,
                snr_th=args.snr_th, min_delay_s=args.min_delay_s, max_tries=args.max_tries)
    tasks = [(role, s, min(n, s + args.chunk), rows[s:min(n, s + args.chunk)], n, opts) for s in range(0, n, args.chunk)]
    recs = []
    with Pool(args.workers, initializer=g.init_worker,
              initargs=(args.config, args.fs, args.duration, args.fmin, args.approximant, str(root), "PM", N)) as pool:
        for _, part in pool.imap_unordered(work, tasks):
            recs.extend(part)
            done = len(recs); el = time.time() - t0
            print(f"  {done}/{n} sources  {el / 60:.1f} min elapsed", flush=True)
    df = pd.DataFrame(recs).sort_values("catalog_index").reset_index(drop=True)
    keep = df.index[df.accepted].to_numpy()
    compact(base, "PM", n, keep, N)
    acc = df[df.accepted].reset_index(drop=True).copy()
    acc["pm_catalog_index"] = np.arange(len(acc))
    acc.to_csv(base / "pair_meta.csv", index=False)
    df[["catalog_index", "source_system_index", "accepted"]].to_csv(base / "acceptance.csv", index=False)
    pd.DataFrame({"mu_0": acc.mu_1, "mu_1": acc.mu_2, "t_d": acc.delay_s}).to_csv(base / "lens.csv", index=False)
    pd.DataFrame({"z_l": acc.zl, "z_s": acc.zs, "M_L(Msun)": acc.m_lens_msun, "y": acc.y}).to_csv(base / "lens_params.csv", index=False)
    srcs = acc[g.SOURCE_COLUMNS].copy(); srcs.to_csv(base / "source_samples.csv", index=False)
    second = srcs.copy(); second["geocent_time"] += acc.delay_s.to_numpy()
    pd.concat([srcs, second], ignore_index=True).to_csv(base / "lensed_source_samples.csv", index=False)
    pd.DataFrame({"lensed_index": acc.source_system_index}).to_csv(base / "lensed_index.csv", index=False)
    import bilby, lal
    meta = {"config": args.config, "role": role, "family_label": "PM", "purpose": "lens-profile robustness test; not a target population",
            "sources": f"the BBH sources of SIS_data_{role} (galaxy-lensed systems), same sky position, arrival time and lens redshift",
            "m_lens_prior": f"log-uniform [{args.mlens_min:g}, {args.mlens_max:g}] Msun",
            "y_prior": f"p(y) = 2y/y_max^2 on (0, {args.y_max}]", "acceptance": "both images network optimal S/N >= snr_th and delay >= min_delay_s",
            "snr_th": args.snr_th, "min_delay_s": args.min_delay_s, "max_tries": args.max_tries, "seed": args.seed,
            "n_sources": int(n), "n_accepted": int(len(acc)), "mean_tries_accepted": float(acc.tries.mean()) if len(acc) else None,
            "detectors_simulated": g.DETECTORS[args.config]["ifos"],
            "stored_channel_rule": f"detector with the highest optimal S/N among {g.DETECTORS[args.config]['reference']}",
            "duration_s": args.duration, "fs_hz": args.fs, "N": N, "peak_index": N - g.PEAK_OFFSET_SAMPLES, "fmin_hz": args.fmin,
            "approximant": args.approximant, "software": {"bilby": bilby.__version__, "lal": lal.__version__, "numpy": np.__version__},
            "wall_s": time.time() - t0}
    (base / "generation_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"accepted {len(acc)}/{n} sources in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
