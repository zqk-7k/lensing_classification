#!/usr/bin/env python3
"""Whitened single-detector strain catalogs from a ``ler`` population, in the repository's layout.

ApJS revision of the PI-ResNet paper. Replaces the uniform-prior generators
(src/generation/generate_*_catalog.py) with the realistic population drawn by
``ler_population_rev.py`` and writes catalogs that the frozen reproducibility
workflow (experiments/reproducibility/*.py) reads unchanged:

    <out-root>/<FAMILY>_data_<ROLE>/<FAMILY>_data_strain_{1,2}.npy   (n, N) float32 whitened data
                                   <FAMILY>_h_strain_{1,2}.npy      whitened signal only
                                   <FAMILY>_optimal_SNR_{1,2}.npy   (n,) single-detector optimal S/N
                                   lens.csv, lens_params.csv, source_samples.csv,
                                   lensed_source_samples.csv, pair_meta.csv
    <out-root>/Unlensed_data_<ROLE>/unlensed_data_strain.npy, unlensed_h_strain.npy,
                                   unlensed_optimal_SNR.npy, source_samples.csv, event_meta.csv

ROLE is 0222 (training and checkpoint selection) or 0228 (threshold setting and
measurement), as in docs/DATASETS.md. Both roles are disjoint halves of one
population draw, assigned by a seeded permutation of the system index.

FAMILY is the directory label the workflow expects; with the galaxy-lens
population it is SIS by default (the frozen scripts accept only SIS or PM) and
stands for "elliptical power-law + external shear galaxy lenses". LABEL_NOTE.md in
the output root records this.

Physics, identical to the original generators unless stated:
  * 24 s segments at 4096 Hz, IMRPhenomXPHM from 20 Hz, Gaussian stationary noise,
    whitening with the detector PSD, signal peak rolled to sample N - 1500, Hann
    fade 0.05 s after the peak (so the network sees the last 2 s: inspiral,
    merger and ringdown);
  * each image is an independent injection into fresh noise: amplitude sqrt(|mu|),
    Morse phase from the image type (I: 0, II: 1/2, III: 1);
  * image 1 is the brighter of the two brightest *detected* images, image 2 the
    fainter, so |mu_0/mu_1| >= 1 as in the original SIS catalogs; the arrival order
    and the signed delay are recorded in pair_meta.csv;
  * the second image is injected at geocent_time + delay, so the Earth has rotated;
  * the stored channel is one detector: the one with the highest optimal S/N
    in the network (O4: H1, L1 or V1; ET: one of the three ET-D arms). The paper's
    PI-ResNet is a single-detector method and this is the detector a pipeline would
    hand it; at O4 the whitening therefore mixes the aLIGO and AdV spectra, which
    is a condition of the real network. All per-detector and network S/N values
    are kept in the meta tables.
  * pairs whose images arrive within --min-delay-s of each other are excluded
    (they overlap in one segment and belong to an overlapping-signal search);
    the excluded fraction is recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

MORSE = {1: 0.0, 2: 0.5, 3: 1.0}
PEAK_OFFSET_SAMPLES = 1500
FADE_AFTER_PEAK_S = 0.05
ARCSEC = 180.0 / np.pi * 3600.0
SOURCE_COLUMNS = ["luminosity_distance", "mass_1_source", "mass_2_source", "a_1", "a_2", "tilt_1", "tilt_2",
                  "phi_12", "phi_jl", "ra", "dec", "theta_jn", "psi", "phase", "geocent_time"]
DETECTORS = {
    "O4": dict(ifos=["H1", "L1", "V1"], psd={"H1": ("asd", "aLIGO_O4_high_asd.txt"), "L1": ("asd", "aLIGO_O4_high_asd.txt"),
                                             "V1": ("asd", "AdV_asd.txt")}, reference=["H1", "L1", "V1"]),
    "ET": dict(ifos=["ET"], psd={"ET1": ("psd", "ET_D_psd.txt"), "ET2": ("psd", "ET_D_psd.txt"), "ET3": ("psd", "ET_D_psd.txt")},
               reference=["ET1", "ET2", "ET3"]),
}
ROLE_SEED = {"0222": 100_000_000, "0228": 200_000_000}
KIND_SEED = {"lensed": 0, "unlensed": 50_000_000}

W = {}  # per-worker state


# ----------------------------------------------------------------------------- population tables
def load_json(path: Path) -> dict:
    return {k: np.asarray(v) for k, v in json.loads(path.read_text()).items()}


def spin_params(d: dict, i: int) -> dict:
    """ler draws aligned spin components (may be negative); bilby wants magnitude and tilt."""
    out = {}
    for k in ("1", "2"):
        a = float(d[f"a_{k}"][i]) if f"a_{k}" in d else 0.0
        tilt = float(d[f"tilt_{k}"][i]) if f"tilt_{k}" in d else 0.0
        if a < 0:
            a, tilt = -a, np.pi - tilt
        out[f"a_{k}"], out[f"tilt_{k}"] = a, tilt
    out["phi_12"] = float(d["phi_12"][i]) if "phi_12" in d else 0.0
    out["phi_jl"] = float(d["phi_jl"][i]) if "phi_jl" in d else 0.0
    return out


def source_params(d: dict, i: int) -> dict:
    zs = float(d["zs"][i])
    return {
        "luminosity_distance": float(d["luminosity_distance"][i]),
        "mass_1_source": float(d["mass_1_source"][i]), "mass_2_source": float(d["mass_2_source"][i]),
        "zs": zs, "mass_1": float(d["mass_1_source"][i]) * (1 + zs), "mass_2": float(d["mass_2_source"][i]) * (1 + zs),
        **spin_params(d, i),
        "ra": float(d["ra"][i]), "dec": float(d["dec"][i]), "theta_jn": float(d["theta_jn"][i]),
        "psi": float(d["psi"][i]), "phase": float(d["phase"][i]), "geocent_time": float(d["geocent_time"][i]),
    }


def pair_rows(lp: dict, snr_th: float, min_delay_s: float) -> tuple[list[dict], dict]:
    mu = lp["magnifications"].astype(float)
    dt = lp["time_delays"].astype(float)
    itype = lp["image_type"].astype(float)
    snr = lp["optimal_snr_net"].astype(float)
    pdet = lp["pdet_net"].astype(float) if "pdet_net" in lp else None
    n_sys = mu.shape[0]
    rows, n_lt2, n_overlap = [], 0, 0
    for i in range(n_sys):
        det = np.where(pdet[i] >= 0.5)[0] if pdet is not None else np.where(np.nan_to_num(snr[i]) >= snr_th)[0]
        if len(det) < 2:
            n_lt2 += 1
            continue
        order = det[np.argsort(-np.abs(mu[i, det]))]
        a, b = int(order[0]), int(order[1])  # a: brighter image -> image 1
        delay = float(dt[i, b] - dt[i, a])    # signed: >0 if the brighter image arrives first
        if abs(delay) < min_delay_s:
            n_overlap += 1
            continue
        theta_E = float(lp["theta_E"][i]) if "theta_E" in lp else np.nan
        beta_x = float(lp["x_source"][i]) if "x_source" in lp else np.nan
        beta_y = float(lp["y_source"][i]) if "y_source" in lp else np.nan
        row = {
            "system_index": i, **source_params(lp, i),
            "zl": float(lp["zl"][i]), "sigma": float(lp["sigma"][i]), "q": float(lp["q"][i]), "gamma": float(lp["gamma"][i]),
            "gamma1": float(lp["gamma1"][i]), "gamma2": float(lp["gamma2"][i]), "theta_E_rad": theta_E,
            "beta_x_rad": beta_x, "beta_y_rad": beta_y,
            "y": float(np.hypot(beta_x, beta_y) / theta_E) if theta_E and np.isfinite(theta_E) else np.nan,
            "n_images": int(lp["n_images"][i]), "n_detected_images": int(len(det)),
            "mu_1": float(mu[i, a]), "mu_2": float(mu[i, b]), "type_1": int(itype[i, a]), "type_2": int(itype[i, b]),
            "morse_1": MORSE[int(itype[i, a])], "morse_2": MORSE[int(itype[i, b])],
            "delay_s": delay, "brighter_arrives_first": bool(delay > 0),
            "snr_net_ler_1": float(snr[i, a]), "snr_net_ler_2": float(snr[i, b]),
            # ler's arrival time of image 1 is the source time plus this offset (time_delays are
            # relative to the first-arriving image); v2 places image 1 at the source time.
            "t_offset_1": float(dt[i, a]),
        }
        rows.append(row)
    stats = {"systems_in_json": int(n_sys), "fewer_than_two_detected_images": n_lt2,
             "excluded_overlapping_delay": n_overlap, "min_delay_s": min_delay_s, "usable_pairs": len(rows)}
    return rows, stats


# ----------------------------------------------------------------------------- waveform and injection
def lensed_fd_model(frequency_array, mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                    luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec,
                    magnification, morse_index, **kwargs):
    import bilby
    h = bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1=mass_1, mass_2=mass_2, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, phi_jl=phi_jl, luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase, **kwargs)
    if h is None:
        return None
    F = np.sqrt(abs(magnification)) * np.exp(-1j * np.pi * morse_index * np.sign(frequency_array))
    return {"plus": h["plus"] * F, "cross": h["cross"] * F}


def init_worker(config, fs, duration, fmin, approximant, out_root, family, N):
    import bilby
    bilby.core.utils.logger.setLevel("ERROR")
    spec = DETECTORS[config]
    ifos = bilby.gw.detector.InterferometerList(spec["ifos"])
    for ifo in ifos:
        kind, name = spec["psd"][ifo.name]
        ifo.power_spectral_density = (bilby.gw.detector.PowerSpectralDensity(asd_file=name) if kind == "asd"
                                      else bilby.gw.detector.PowerSpectralDensity(psd_file=name))
        ifo.minimum_frequency = fmin
    wf_args = dict(waveform_approximant=approximant, reference_frequency=10.0, minimum_frequency=fmin)
    W["ifos"] = ifos
    W["names"] = [ifo.name for ifo in ifos]
    W["ref_idx"] = [W["names"].index(n) for n in spec["reference"]]
    W["wfg_lensed"] = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=fs, duration=duration, frequency_domain_source_model=lensed_fd_model, waveform_arguments=wf_args)
    W["wfg_plain"] = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=fs, duration=duration, frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=wf_args)
    W.update(fs=fs, N=N, out_root=Path(out_root), family=family, arrays={})


def inject(wfg, params: dict, seed: int):
    """One signal into fresh Gaussian noise in every detector; returns per-detector products."""
    import bilby
    ifos, fs, N = W["ifos"], W["fs"], W["N"]
    bilby.core.utils.random.seed(seed)
    target = N - PEAK_OFFSET_SAMPLES
    data = np.empty((len(ifos), N), np.float32)
    sig = np.empty((len(ifos), N), np.float32)
    snr_single = np.empty(len(ifos))
    fd = wfg.frequency_domain_strain(params)
    for j, ifo in enumerate(ifos):
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=fs, duration=N / fs, start_time=params["geocent_time"] - (N / fs - PEAK_OFFSET_SAMPLES / fs))
        n_t_white = np.fft.irfft(ifo.whitened_frequency_domain_strain, n=N)
        response = ifo.get_detector_response(fd, params)
        snr_single[j] = np.sqrt(abs(float(np.real(ifo.optimal_snr_squared(response)))))
        ifo.inject_signal(waveform_generator=wfg, parameters=params, raise_error=False)
        d_t_white = np.fft.irfft(ifo.whitened_frequency_domain_strain, n=N)
        h_white = d_t_white - n_t_white
        shift = target - int(np.argmax(np.abs(h_white)))
        h_white = np.roll(h_white, shift)
        index = min(target + int(FADE_AFTER_PEAK_S * fs), N)
        fade = N - index
        if fade > 0:
            h_white[index:] *= np.hanning(2 * fade)[fade:]
        data[j] = h_white + n_t_white
        sig[j] = h_white
    return data, sig, snr_single, float(np.sqrt(np.sum(snr_single ** 2)))


def array(path, shape, dtype):
    key = str(path)
    if key not in W["arrays"]:
        W["arrays"][key] = np.lib.format.open_memmap(path, mode="r+", dtype=dtype, shape=shape)
    return W["arrays"][key]


def work(task):
    role, kind, start, stop, rows, n_total = task
    out = W["out_root"]; fam = W["family"]; N = W["N"]; names = W["names"]; ref = W["ref_idx"]
    results = []
    if kind == "lensed":
        base = out / f"{fam}_data_{role}"
        for i in range(start, stop):
            row = rows[i - start]
            params_base = {k: row[k] for k in ("mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
                                               "luminosity_distance", "theta_jn", "psi", "phase", "ra", "dec")}
            rec = {"catalog_index": i, "failed": False, **row}
            for k in (1, 2):
                params = dict(params_base, geocent_time=row["geocent_time"] + (row["delay_s"] if k == 2 else 0.0),
                              magnification=row[f"mu_{k}"], morse_index=row[f"morse_{k}"])
                seed = ROLE_SEED[role] + KIND_SEED[kind] + 10 * i + k
                try:
                    data, sig, snr_single, snr_net = inject(W["wfg_lensed"], params, seed)
                except RuntimeError as err:  # waveform generator refused these parameters
                    rec["failed"] = True; rec["failure"] = str(err)[:200]
                    break
                j = ref[int(np.argmax(snr_single[ref]))]
                array(base / f"{fam}_data_strain_{k}.npy", (n_total, N), np.float32)[i] = data[j]
                array(base / f"{fam}_h_strain_{k}.npy", (n_total, N), np.float32)[i] = sig[j]
                array(base / f"{fam}_optimal_SNR_{k}.npy", (n_total,), np.float64)[i] = snr_single[j]
                rec[f"channel_{k}"] = names[j]; rec[f"snr_channel_{k}"] = float(snr_single[j]); rec[f"snr_net_bilby_{k}"] = snr_net
                for name, value in zip(names, snr_single):
                    rec[f"snr_{name}_{k}"] = float(value)
                rec[f"segment_start_gps_{k}"] = params["geocent_time"] - (N / W["fs"] - PEAK_OFFSET_SAMPLES / W["fs"])
                rec[f"noise_seed_{k}"] = seed
            results.append(rec)
    else:
        base = out / f"Unlensed_data_{role}"
        for i in range(start, stop):
            row = rows[i - start]
            params = {k: row[k] for k in ("mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
                                          "luminosity_distance", "theta_jn", "psi", "phase", "ra", "dec", "geocent_time")}
            seed = ROLE_SEED[role] + KIND_SEED[kind] + 10 * i
            try:
                data, sig, snr_single, snr_net = inject(W["wfg_plain"], params, seed)
            except RuntimeError as err:
                results.append({"catalog_index": i, "failed": True, "failure": str(err)[:200], **row})
                continue
            j = ref[int(np.argmax(snr_single[ref]))]
            array(base / "unlensed_data_strain.npy", (n_total, N), np.float32)[i] = data[j]
            array(base / "unlensed_h_strain.npy", (n_total, N), np.float32)[i] = sig[j]
            array(base / "unlensed_optimal_SNR.npy", (n_total,), np.float64)[i] = snr_single[j]
            rec = {"catalog_index": i, "failed": False, **row, "channel": names[j], "snr_channel": float(snr_single[j]), "snr_net_bilby": snr_net,
                   "segment_start_gps": params["geocent_time"] - (N / W["fs"] - PEAK_OFFSET_SAMPLES / W["fs"]), "noise_seed": seed}
            for name, value in zip(names, snr_single):
                rec[f"snr_{name}"] = float(value)
            results.append(rec)
    for a in W["arrays"].values():
        a.flush()
    return role, kind, results


# ----------------------------------------------------------------------------- driver
def create(path, shape, dtype):
    np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape).flush()


def compact(base: Path, names: list[str], n_written: int, keep: np.ndarray):
    """Remove the rows of failed injections so arrays and tables stay aligned."""
    if len(keep) == n_written:
        return
    for name in names:
        path = base / name
        src = np.load(path, mmap_mode="r")
        tmp = base / (name + ".tmp")
        dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=src.dtype, shape=(len(keep),) + src.shape[1:])
        for new, old in enumerate(keep):
            dst[new] = src[old]
        dst.flush(); del dst, src
        tmp.replace(path)


def write_lensed_tables(base: Path, fam: str, recs: list[dict]):
    df = pd.DataFrame(recs).sort_values("catalog_index").reset_index(drop=True)
    failed = df[df.failed]
    if len(failed):
        failed[["catalog_index", "system_index", "failure"]].to_csv(base / "failed_injections.csv", index=False)
        compact(base, [f"{fam}_{s}_{k}.npy" for s in ("data_strain", "h_strain", "optimal_SNR") for k in (1, 2)],
                len(df), df.index[~df.failed].to_numpy())
        df = df[~df.failed].reset_index(drop=True); df["catalog_index"] = np.arange(len(df))
    df.to_csv(base / "pair_meta.csv", index=False)
    pd.DataFrame({"mu_0": df.mu_1, "mu_1": df.mu_2, "t_d": df.delay_s}).to_csv(base / "lens.csv", index=False)
    pd.DataFrame({"z_l": df.zl, "z_s": df.zs, "sigma_v": df.sigma, "theta_E(arcsec)": df.theta_E_rad * ARCSEC, "y": df.y,
                  "beta_x": df.beta_x_rad * ARCSEC, "beta_y": df.beta_y_rad * ARCSEC}).to_csv(base / "lens_params.csv", index=False)
    src = df[SOURCE_COLUMNS].copy()
    src.to_csv(base / "source_samples.csv", index=False)
    second = src.copy(); second["geocent_time"] = second["geocent_time"] + df.delay_s.to_numpy()
    pd.concat([src, second], ignore_index=True).to_csv(base / "lensed_source_samples.csv", index=False)
    pd.DataFrame({"lensed_index": df.system_index}).to_csv(base / "lensed_index.csv", index=False)
    return df


def write_unlensed_tables(base: Path, recs: list[dict]):
    df = pd.DataFrame(recs).sort_values("catalog_index").reset_index(drop=True)
    failed = df[df.failed]
    if len(failed):
        failed[["catalog_index", "ler_index", "failure"]].to_csv(base / "failed_injections.csv", index=False)
        compact(base, ["unlensed_data_strain.npy", "unlensed_h_strain.npy", "unlensed_optimal_SNR.npy"],
                len(df), df.index[~df.failed].to_numpy())
        df = df[~df.failed].reset_index(drop=True); df["catalog_index"] = np.arange(len(df))
    df.to_csv(base / "event_meta.csv", index=False)
    df[SOURCE_COLUMNS].to_csv(base / "source_samples.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(DETECTORS), required=True)
    ap.add_argument("--ler-dir", required=True, help="ler_data_<config> directory with n_{lensed,unlensed}_detectable_<config>.json")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--family-label", default="SIS", help="directory label the frozen workflow expects (SIS or PM)")
    ap.add_argument("--roles", default="0222,0228")
    ap.add_argument("--n-lensed", type=int, default=2500, help="lensed systems per role")
    ap.add_argument("--n-unlensed", type=int, default=5000, help="unlensed events per role")
    ap.add_argument("--snr-th", type=float, default=8.0)
    ap.add_argument("--min-delay-s", type=float, default=60.0)
    ap.add_argument("--mtot-det-max", type=float, default=800.0,
                    help="drop systems whose detector-frame total mass exceeds this: their IMRPhenomXPHM cutoff "
                         "frequency falls below the 20 Hz minimum frequency, i.e. no in-band signal")
    ap.add_argument("--duration", type=float, default=24.0)
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--approximant", default="IMRPhenomXPHM")
    ap.add_argument("--seed", type=int, default=20261107, help="catalog assignment permutation seed")
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--chunk", type=int, default=25)
    args = ap.parse_args()

    t0 = time.time()
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    fam = args.family_label
    N = int(args.duration * args.fs)
    ler_dir = Path(args.ler_dir)
    lp = load_json(ler_dir / f"n_lensed_detectable_{args.config}.json")
    up = load_json(ler_dir / f"n_unlensed_detectable_{args.config}.json")
    rows, stats = pair_rows(lp, args.snr_th, args.min_delay_s)
    heavy = [r for r in rows if r["mass_1"] + r["mass_2"] > args.mtot_det_max]
    rows = [r for r in rows if r["mass_1"] + r["mass_2"] <= args.mtot_det_max]
    stats.update(excluded_heavy_detector_frame=len(heavy), mtot_det_max=args.mtot_det_max, usable_pairs=len(rows))
    unl_mtot = (np.asarray(up["mass_1_source"], float) + np.asarray(up["mass_2_source"], float)) * (1 + np.asarray(up["zs"], float))
    unl_keep = np.where(unl_mtot <= args.mtot_det_max)[0]
    stats["excluded_heavy_unlensed"] = int(len(unl_mtot) - len(unl_keep))
    n_unl_avail = len(unl_keep)
    need_l, need_u = args.n_lensed * len(roles), args.n_unlensed * len(roles)
    if len(rows) < need_l or n_unl_avail < need_u:
        raise SystemExit(f"population too small: {len(rows)} usable pairs (need {need_l}), {n_unl_avail} unlensed (need {need_u})")
    rng = np.random.RandomState(args.seed)
    perm_l = rng.permutation(len(rows)); perm_u = unl_keep[rng.permutation(n_unl_avail)]
    assignment = {}
    for r, role in enumerate(roles):
        sel_l = np.sort(perm_l[r * args.n_lensed:(r + 1) * args.n_lensed])
        sel_u = np.sort(perm_u[r * args.n_unlensed:(r + 1) * args.n_unlensed])
        assignment[role] = {"lensed": [rows[k] for k in sel_l], "unlensed": [source_params(up, int(k)) | {"ler_index": int(k),
                            "snr_net_ler": float(up["optimal_snr_net"][k]) if "optimal_snr_net" in up else np.nan} for k in sel_u]}
        lb, ub = out / f"{fam}_data_{role}", out / f"Unlensed_data_{role}"
        lb.mkdir(parents=True, exist_ok=True); ub.mkdir(parents=True, exist_ok=True)
        for k in (1, 2):
            create(lb / f"{fam}_data_strain_{k}.npy", (args.n_lensed, N), np.float32)
            create(lb / f"{fam}_h_strain_{k}.npy", (args.n_lensed, N), np.float32)
            create(lb / f"{fam}_optimal_SNR_{k}.npy", (args.n_lensed,), np.float64)
        create(ub / "unlensed_data_strain.npy", (args.n_unlensed, N), np.float32)
        create(ub / "unlensed_h_strain.npy", (args.n_unlensed, N), np.float32)
        create(ub / "unlensed_optimal_SNR.npy", (args.n_unlensed,), np.float64)

    tasks = []
    for role in roles:
        for kind, n in (("lensed", args.n_lensed), ("unlensed", args.n_unlensed)):
            items = assignment[role][kind]
            for s in range(0, n, args.chunk):
                e = min(n, s + args.chunk)
                tasks.append((role, kind, s, e, items[s:e], n))
    total_events = sum(((e - s) * (2 if kind == "lensed" else 1)) for _, kind, s, e, _, _ in tasks)
    print(f"[{args.config}] {len(tasks)} tasks, {total_events} injections, {args.workers} workers; pair stats {stats}", flush=True)

    collected = {role: {"lensed": [], "unlensed": []} for role in roles}
    done_events = 0
    with Pool(args.workers, initializer=init_worker,
              initargs=(args.config, args.fs, args.duration, args.fmin, args.approximant, str(out), fam, N)) as pool:
        for role, kind, recs in pool.imap_unordered(work, tasks):
            collected[role][kind].extend(recs)
            done_events += len(recs) * (2 if kind == "lensed" else 1)
            el = time.time() - t0
            print(f"  {done_events}/{total_events} injections  {el / 60:.1f} min elapsed, "
                  f"~{el / done_events * (total_events - done_events) / 60:.1f} min left", flush=True)

    import bilby, lal
    for role in roles:
        lb, ub = out / f"{fam}_data_{role}", out / f"Unlensed_data_{role}"
        ldf = write_lensed_tables(lb, fam, collected[role]["lensed"])
        udf = write_unlensed_tables(ub, collected[role]["unlensed"])
        meta = {
            "config": args.config, "role": role, "family_label": fam,
            "lens_population": "ler epl_shear_galaxy (EPL + external shear galaxy lenses)",
            "detectors_simulated": DETECTORS[args.config]["ifos"], "psd": {k: v[1] for k, v in DETECTORS[args.config]["psd"].items()},
            "stored_channel_rule": f"detector with the highest optimal S/N among {DETECTORS[args.config]['reference']}",
            "duration_s": args.duration, "fs_hz": args.fs, "N": N, "peak_index": N - PEAK_OFFSET_SAMPLES, "fmin_hz": args.fmin,
            "approximant": args.approximant, "reference_frequency_hz": 10.0, "snr_threshold_population": args.snr_th,
            "min_delay_s": args.min_delay_s, "pair_selection_stats": stats, "assignment_seed": args.seed,
            "n_lensed": int(len(ldf)), "n_unlensed": int(len(udf)),
            "failed_injections_dropped": {"lensed": int(args.n_lensed - len(ldf)), "unlensed": int(args.n_unlensed - len(udf))},
            "image_order": "image 1 = brighter of the two brightest detected images; image 2 = fainter",
            "channel_counts_lensed": {k: {str(c): int(v) for c, v in ldf[f"channel_{k}"].value_counts().items()} for k in (1, 2)},
            "channel_counts_unlensed": {str(c): int(v) for c, v in udf["channel"].value_counts().items()},
            "software": {"python": platform.python_version(), "numpy": np.__version__, "bilby": bilby.__version__,
                         "lal": lal.__version__, "ler_json": str(ler_dir)},
            "wall_s": time.time() - t0,
        }
        (lb / "generation_meta.json").write_text(json.dumps(meta, indent=2))
        (ub / "generation_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "LABEL_NOTE.md").write_text(
        f"# Catalog labels\n\nThe frozen workflow accepts the family labels SIS and PM only. In this root the label\n"
        f"`{fam}` holds the realistic galaxy-lens population (elliptical power-law + external shear, drawn with\n"
        f"`ler`), not singular isothermal spheres. `0222` is the training/validation catalog and `0228` the\n"
        f"threshold-setting/measurement catalog, both disjoint halves of one population draw (seed {args.seed}).\n"
        f"Detector configuration: {args.config}. See `*/generation_meta.json` and `pair_meta.csv`.\n")
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
