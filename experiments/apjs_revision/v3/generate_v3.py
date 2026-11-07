#!/usr/bin/env python3
"""v3 catalogs: every detector, low-frequency start, two sample rates, signal-only storage.

What changes with respect to generate_strain_rev.py (v2) and why
-----------------------------------------------------------------
* All detectors are kept as channels (ET: the three ET-D arms; O4: H1, L1, V1), each whitened
  with its own PSD, aligned on the geocentric merger time so inter-detector delays and amplitude
  ratios (sky information) survive. v2 kept only the loudest channel and rolled it to its own peak.
* ET waveforms start at 5 Hz (O4 at 20 Hz): most of the ET signal-to-noise ratio of a BBH lies
  between 5 and 20 Hz.
* Two streams per event and detector, both cut from one 128 s frame:
      A: full band, 2048 Hz, the last 2.25 s  (merger at 1.75 s)      shape (3, 4608)
      B: band below 64 Hz, 128 Hz, the last 64.25 s                   shape (3, 8224)
  A carries merger and ringdown, B the long inspiral. The model input is a 2 s / 64 s window of
  each (offsets 256 and 16 samples for evaluation; training jitters them by up to +/-0.125 s).
* Storage is the whitened noiseless signal in float16. Whitened stationary Gaussian noise is white,
  so it is generated when needed (training: on the GPU, fresh every epoch; evaluation: stored).
  The evaluation catalog ("0228") also stores its noisy data so the test set is fixed on disk.
* Spins are redrawn small and precessing (magnitudes Beta(2, 6), mean 0.25; tilts: 66 % from a
  Gaussian in cos(tilt) about 1 with width 1.5, truncated to [-1, 1], 34 % isotropic; azimuths
  uniform), approximating the GWTC-3 Default spin model; ler's population is aligned-spin only.
  Detection selection (ler, network S/N >= 8 from 20 Hz per image, no spins) is unchanged.

Whitening convention: for a detector response h(f) (bilby convention, h = dt * FFT) and one-sided
PSD S(f), X_k = h(f_k) sqrt(2 fs / S(f_k)) on [f_min, fs/2), zero elsewhere; x = irfft(X). Then
sum_t x^2 equals the optimal S/N^2 and whitened noise has X_k = sqrt(N/2)(a + ib), a, b ~ N(0, 1),
i.e. unit variance per sample over the full band. Stream B uses X_B = X[:N_B/2+1] sqrt(N_B/N), the
same convention at 128 Hz.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_strain_rev import lensed_fd_model, pair_rows, source_params, load_json, SOURCE_COLUMNS, ARCSEC  # noqa: E402

FS, FRAME_S, TAIL_S = 2048, 128, 0.5
N = FS * FRAME_S
FS_B = 128
N_B = FS_B * FRAME_S
LEN_A, LEN_B = 4608, 8224          # stored samples (2.25 s, 64.25 s)
WIN_A, WIN_B = 4096, 8192          # model window (2 s, 64 s)
EVAL_OFF_A, EVAL_OFF_B = 256, 16   # evaluation window offsets; training offset in [0, 512] step 16
CONFIGS = {
    "ET": dict(ifos=["ET"], psd={"ET1": ("psd", "ET_D_psd.txt"), "ET2": ("psd", "ET_D_psd.txt"), "ET3": ("psd", "ET_D_psd.txt")},
               fmin=5.0),
    "O4": dict(ifos=["H1", "L1", "V1"], psd={"H1": ("asd", "aLIGO_O4_high_asd.txt"), "L1": ("asd", "aLIGO_O4_high_asd.txt"),
                                             "V1": ("asd", "AdV_asd.txt")}, fmin=20.0),
}
ROLE_SEED = {"0222": 300_000_000, "0228": 400_000_000}
KIND_SEED = {"lensed": 0, "unlensed": 50_000_000}
F_REF = 20.0
MAX_INSPIRAL_S = 120.0
W = {}


def chirp_time(mc_det, f):
    """Newtonian time to merger from frequency f [s] for detector-frame chirp mass mc_det [Msun]."""
    tm = mc_det * 4.925490947e-6
    return 5.0 / 256.0 * tm ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def f_at_time(mc_det, tau):
    tm = mc_det * 4.925490947e-6
    return (5.0 / 256.0 * tm ** (-5.0 / 3.0) / tau) ** (3.0 / 8.0) / np.pi


def draw_spins(rng):
    out = {}
    for k in ("1", "2"):
        a = rng.beta(2.0, 6.0)
        if rng.random() < 0.66:
            while True:
                c = rng.normal(1.0, 1.5)
                if -1.0 <= c <= 1.0:
                    break
        else:
            c = rng.uniform(-1.0, 1.0)
        out[f"a_{k}"], out[f"tilt_{k}"] = float(a), float(np.arccos(c))
    out["phi_12"], out["phi_jl"] = float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 2 * np.pi))
    return out


def noise_fd(rng, n_det, band_mask):
    """Whitened noise rfft coefficients (n_det, N/2+1), unit variance per sample over the full band."""
    re = rng.standard_normal((n_det, N // 2 + 1)); im = rng.standard_normal((n_det, N // 2 + 1))
    return (np.sqrt(N / 2.0) * (re + 1j * im)) * band_mask


def streams(X):
    """Stored A and B streams from whitened rfft coefficients X (n_det, N/2+1)."""
    a = np.fft.irfft(X, n=N, axis=-1)[:, -LEN_A:]
    b = np.fft.irfft(X[:, : N_B // 2 + 1] * np.sqrt(N_B / N), n=N_B, axis=-1)[:, -LEN_B:]
    return a, b


def init_worker(config, approximant):
    import bilby
    bilby.core.utils.logger.setLevel("ERROR")
    spec = CONFIGS[config]
    ifos = bilby.gw.detector.InterferometerList(spec["ifos"])
    for ifo in ifos:
        kind, name = spec["psd"][ifo.name]
        ifo.power_spectral_density = (bilby.gw.detector.PowerSpectralDensity(asd_file=name) if kind == "asd"
                                      else bilby.gw.detector.PowerSpectralDensity(psd_file=name))
        ifo.minimum_frequency = spec["fmin"]
        ifo.maximum_frequency = FS / 2
        ifo.set_strain_data_from_zero_noise(sampling_frequency=FS, duration=FRAME_S, start_time=0.0)
    freqs = ifos[0].frequency_array
    band = (freqs >= spec["fmin"]) & (freqs < FS / 2)
    whiten = np.zeros((len(ifos), len(freqs)))
    for j, ifo in enumerate(ifos):
        psd = ifo.power_spectral_density_array
        ok = band & np.isfinite(psd) & (psd > 0)
        whiten[j, ok] = np.sqrt(2.0 * FS / psd[ok])
    W.update(ifos=ifos, names=[i.name for i in ifos], freqs=freqs, band=band.astype(float), whiten=whiten,
             fmin=spec["fmin"], approximant=approximant, df=1.0 / FRAME_S,
             band20=((freqs >= 20.0) & (freqs < FS / 2)).astype(float))
    W["wfg"] = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=FS, duration=FRAME_S, frequency_domain_source_model=lensed_fd_model,
        waveform_arguments=dict(waveform_approximant=approximant, reference_frequency=F_REF, minimum_frequency=spec["fmin"]))


def make_event(params, noise_seed=None):
    """Whitened streams for one image or unlensed event; params include magnification and morse_index."""
    ifos = W["ifos"]
    mc = (params["mass_1"] * params["mass_2"]) ** 0.6 / (params["mass_1"] + params["mass_2"]) ** 0.2
    f_start = max(W["fmin"], float(f_at_time(mc, MAX_INSPIRAL_S)))
    W["wfg"].waveform_arguments["minimum_frequency"] = f_start
    fd = W["wfg"].frequency_domain_strain(params)
    if fd is None:
        raise RuntimeError("waveform generator returned None")
    start = params["geocent_time"] - (FRAME_S - TAIL_S)
    X = np.zeros((len(ifos), len(W["freqs"])), complex)
    snr_det, snr20 = np.zeros(len(ifos)), np.zeros(len(ifos))
    for j, ifo in enumerate(ifos):
        ifo.strain_data.start_time = start
        resp = ifo.get_detector_response(fd, params)
        X[j] = resp * W["whiten"][j]
        p2 = np.abs(X[j]) ** 2
        snr_det[j] = np.sqrt(2.0 / N * p2.sum())
        snr20[j] = np.sqrt(2.0 / N * (p2 * W["band20"]).sum())
    sa, sb = streams(X)
    out = {"sig_A": sa, "sig_B": sb, "snr_det": snr_det, "snr20_det": snr20, "f_start": f_start,
           "snr_win_A": float(np.sqrt((sa[:, EVAL_OFF_A:EVAL_OFF_A + WIN_A] ** 2).sum())),
           "snr_win_B": float(np.sqrt((sb[:, EVAL_OFF_B:EVAL_OFF_B + WIN_B] ** 2).sum()))}
    if noise_seed is not None:
        na, nb = streams(noise_fd(np.random.Generator(np.random.PCG64(noise_seed)), len(ifos), W["band"]))
        out["data_A"], out["data_B"] = sa + na, sb + nb
    return out


def open_arrays(base: Path, names, n, n_det, mode):
    arrs = {}
    for name in names:
        shape = (n, n_det, LEN_A if name.endswith("A") or "_A_" in name else LEN_B)
        arrs[name] = np.lib.format.open_memmap(base / f"{name}.npy", mode=mode, dtype=np.float16, shape=shape)
    return arrs


def work(task):
    role, kind, start, stop, rows, n_total, out_root, fam, store_noisy = task
    n_det = len(W["ifos"])
    base = Path(out_root) / (f"{fam}_data_{role}" if kind == "lensed" else f"Unlensed_data_{role}")
    if kind == "lensed":
        names = [f"sig_{s}_{k}" for s in "AB" for k in (1, 2)] + ([f"data_{s}_{k}" for s in "AB" for k in (1, 2)] if store_noisy else [])
    else:
        names = ["sig_A", "sig_B"] + (["data_A", "data_B"] if store_noisy else [])
    arrs = open_arrays(base, names, n_total, n_det, "r+")
    recs = []
    for i in range(start, stop):
        row = dict(rows[i - start])
        rec = {"catalog_index": i, "failed": False}
        spin_rng = np.random.Generator(np.random.PCG64(ROLE_SEED[role] + KIND_SEED[kind] + 7 * i + 3))
        row.update(draw_spins(spin_rng))
        rec.update(row)
        keys = ("mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "luminosity_distance",
                "theta_jn", "psi", "phase", "ra", "dec")
        images = (1, 2) if kind == "lensed" else (0,)
        try:
            for k in images:
                p = {key: row[key] for key in keys}
                if kind == "lensed":
                    p.update(geocent_time=row["geocent_time"] + (row["delay_s"] if k == 2 else 0.0),
                             magnification=row[f"mu_{k}"], morse_index=row[f"morse_{k}"])
                else:
                    p.update(geocent_time=row["geocent_time"], magnification=1.0, morse_index=0.0)
                seed = ROLE_SEED[role] + KIND_SEED[kind] + 10 * i + k
                ev = make_event(p, seed if store_noisy else None)
                sfx = f"_{k}" if kind == "lensed" else ""
                arrs[f"sig_A{sfx}"][i] = ev["sig_A"]; arrs[f"sig_B{sfx}"][i] = ev["sig_B"]
                if store_noisy:
                    arrs[f"data_A{sfx}"][i] = ev["data_A"]; arrs[f"data_B{sfx}"][i] = ev["data_B"]
                tag = sfx
                rec[f"snr_net{tag}"] = float(np.sqrt((ev["snr_det"] ** 2).sum()))
                rec[f"snr_net20{tag}"] = float(np.sqrt((ev["snr20_det"] ** 2).sum()))
                rec[f"snr_best_det{tag}"] = float(ev["snr_det"].max())
                rec[f"best_det{tag}"] = W["names"][int(np.argmax(ev["snr_det"]))]
                rec[f"snr_win_A{tag}"] = ev["snr_win_A"]; rec[f"snr_win_B{tag}"] = ev["snr_win_B"]
                rec[f"f_start{tag}"] = ev["f_start"]; rec[f"noise_seed{tag}"] = seed if store_noisy else -1
                for name, v in zip(W["names"], ev["snr_det"]):
                    rec[f"snr_{name}{tag}"] = float(v)
        except Exception as err:  # waveform failures: dropped and recorded
            rec["failed"] = True; rec["failure"] = f"{type(err).__name__}: {str(err)[:200]}"
        recs.append(rec)
    for a in arrs.values():
        a.flush()
    return role, kind, recs


def compact(base: Path, names, keep):
    for name in names:
        path = base / f"{name}.npy"
        src = np.load(path, mmap_mode="r")
        tmp = base / f"{name}.tmp.npy"
        dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=src.dtype, shape=(len(keep),) + src.shape[1:])
        for new, old in enumerate(keep):
            dst[new] = src[old]
        dst.flush(); del dst, src
        tmp.replace(path)


def finish_role(out, fam, role, kind, recs, names, n_det):
    base = out / (f"{fam}_data_{role}" if kind == "lensed" else f"Unlensed_data_{role}")
    df = pd.DataFrame(recs).sort_values("catalog_index").reset_index(drop=True)
    failed = df[df.failed]
    if len(failed):
        failed.to_csv(base / "failed_injections.csv", index=False)
        compact(base, names, df.index[~df.failed].to_numpy())
        df = df[~df.failed].reset_index(drop=True)
    df["catalog_index"] = np.arange(len(df))
    if kind == "lensed":
        df.to_csv(base / "pair_meta.csv", index=False)
        pd.DataFrame({"mu_0": df.mu_1, "mu_1": df.mu_2, "t_d": df.delay_s}).to_csv(base / "lens.csv", index=False)
        pd.DataFrame({"z_l": df.zl, "z_s": df.zs, "sigma_v": df.sigma, "theta_E(arcsec)": df.theta_E_rad * ARCSEC, "y": df.y,
                      "beta_x": df.beta_x_rad * ARCSEC, "beta_y": df.beta_y_rad * ARCSEC}).to_csv(base / "lens_params.csv", index=False)
        df[SOURCE_COLUMNS].to_csv(base / "source_samples.csv", index=False)
        for k in (1, 2):
            np.save(base / f"{fam}_optimal_SNR_{k}.npy", df[f"snr_net_{k}"].to_numpy(np.float64))
    else:
        df.to_csv(base / "event_meta.csv", index=False)
        df[SOURCE_COLUMNS].to_csv(base / "source_samples.csv", index=False)
        np.save(base / "unlensed_optimal_SNR.npy", df["snr_net"].to_numpy(np.float64))
    return df


def selftest(config):
    import bilby
    init_worker(config, "IMRPhenomXPHM")
    p = dict(mass_1=40.0, mass_2=30.0, a_1=0.3, a_2=0.2, tilt_1=0.7, tilt_2=1.9, phi_12=1.0, phi_jl=2.0,
             luminosity_distance=8000.0, theta_jn=0.8, psi=0.4, phase=1.1, ra=1.0, dec=-0.3,
             geocent_time=1.3e9, magnification=1.0, morse_index=0.0)
    t = time.time(); ev = make_event(p, 12345); dt = time.time() - t
    fd = W["wfg"].frequency_domain_strain(p)
    bil = []
    for ifo in W["ifos"]:
        ifo.strain_data.start_time = p["geocent_time"] - (FRAME_S - TAIL_S)
        bil.append(np.sqrt(np.real(ifo.optimal_snr_squared(ifo.get_detector_response(fd, p)))))
    X = noise_fd(np.random.Generator(np.random.PCG64(1)), len(W["ifos"]), W["band"])
    na, nb = streams(X)
    s16 = ev["sig_A"].astype(np.float16).astype(np.float64)
    print(json.dumps({
        "config": config, "seconds_per_event": round(dt, 3), "f_start": ev["f_start"],
        "snr_mine": ev["snr_det"].round(4).tolist(), "snr_bilby": np.round(bil, 4).tolist(),
        "snr_net": float(np.sqrt((ev["snr_det"] ** 2).sum())), "snr_net_20Hz": float(np.sqrt((ev["snr20_det"] ** 2).sum())),
        "snr_in_window_A": ev["snr_win_A"], "snr_in_window_B": ev["snr_win_B"],
        "noise_var_A": float(na.var()), "noise_var_B": float(nb.var()),
        "band_fraction_A": float(W["band"].mean()), "band_fraction_B": float(W["band"][: N_B // 2 + 1].mean()),
        "float16_rel_err_A": float(np.abs(s16 - ev["sig_A"]).max() / np.abs(ev["sig_A"]).max()),
    }, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ler-dir", help="ler_data_<config> directory")
    ap.add_argument("--lensed-json", default="n_lensed_detectable_{c}.json,n_lensed_detectable_{c}_v3extra.json")
    ap.add_argument("--unlensed-json", default="n_unlensed_detectable_{c}.json,n_unlensed_detectable_{c}_v3extra.json")
    ap.add_argument("--out-root")
    ap.add_argument("--family-label", default="SIS")
    ap.add_argument("--n-lensed-0228", type=int, default=5000)
    ap.add_argument("--n-unlensed-0228", type=int, default=10000)
    ap.add_argument("--max-lensed-0222", type=int, default=10 ** 9)
    ap.add_argument("--max-unlensed-0222", type=int, default=10 ** 9)
    ap.add_argument("--snr-th", type=float, default=8.0)
    ap.add_argument("--min-delay-s", type=float, default=FRAME_S, help="drop pairs whose images fall in one frame")
    ap.add_argument("--mtot-det-max", type=float, default=800.0)
    ap.add_argument("--approximant", default="IMRPhenomXPHM")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()
    if args.selftest:
        selftest(args.config); return
    t0 = time.time()
    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    fam, c = args.family_label, args.config
    ler_dir = Path(args.ler_dir)
    rows, stats = [], {}
    for d, name in enumerate(args.lensed_json.format(c=c).split(",")):
        r, s = pair_rows(load_json(ler_dir / name), args.snr_th, args.min_delay_s)
        for x in r:
            x["ler_draw"], x["ler_file"] = d, name
        rows += r; stats[name] = s
    heavy = [r for r in rows if r["mass_1"] + r["mass_2"] > args.mtot_det_max]
    rows = [r for r in rows if r["mass_1"] + r["mass_2"] <= args.mtot_det_max]
    stats["excluded_heavy_detector_frame"] = len(heavy)
    unl = []
    for d, name in enumerate(args.unlensed_json.format(c=c).split(",")):
        up = load_json(ler_dir / name)
        for k in range(len(up["zs"])):
            sp = source_params(up, k)
            if sp["mass_1"] + sp["mass_2"] <= args.mtot_det_max:
                unl.append(sp | {"ler_draw": d, "ler_file": name, "ler_index": k,
                                 "snr_net_ler": float(up["optimal_snr_net"][k]) if "optimal_snr_net" in up else np.nan})
    rng = np.random.RandomState(args.seed)
    pl, pu = rng.permutation(len(rows)), rng.permutation(len(unl))
    n28l, n28u = args.n_lensed_0228, args.n_unlensed_0228
    sel = {"0228": ([rows[k] for k in np.sort(pl[:n28l])], [unl[k] for k in np.sort(pu[:n28u])]),
           "0222": ([rows[k] for k in np.sort(pl[n28l:n28l + args.max_lensed_0222])],
                    [unl[k] for k in np.sort(pu[n28u:n28u + args.max_unlensed_0222])])}
    n_det = len(CONFIGS[c]["psd"])
    tasks, names_of = [], {}
    for role, (lr, ur) in sel.items():
        store = role == "0228"
        for kind, items in (("lensed", lr), ("unlensed", ur)):
            base = out / (f"{fam}_data_{role}" if kind == "lensed" else f"Unlensed_data_{role}")
            base.mkdir(parents=True, exist_ok=True)
            if kind == "lensed":
                names = [f"sig_{s}_{k}" for s in "AB" for k in (1, 2)] + ([f"data_{s}_{k}" for s in "AB" for k in (1, 2)] if store else [])
            else:
                names = ["sig_A", "sig_B"] + (["data_A", "data_B"] if store else [])
            names_of[(role, kind)] = names
            open_arrays(base, names, len(items), n_det, "w+")
            for s in range(0, len(items), args.chunk):
                e = min(len(items), s + args.chunk)
                tasks.append((role, kind, s, e, items[s:e], len(items), str(out), fam, store))
    total = sum(len(t[4]) * (2 if t[1] == "lensed" else 1) for t in tasks)
    print(f"[{c}] pairs available {len(rows)}, unlensed available {len(unl)}; "
          f"0228 {n28l}/{n28u}, 0222 {len(sel['0222'][0])}/{len(sel['0222'][1])}; {total} injections; {stats}", flush=True)
    got = {key: [] for key in names_of}
    done = 0
    with Pool(args.workers, initializer=init_worker, initargs=(c, args.approximant)) as pool:
        for role, kind, recs in pool.imap_unordered(work, tasks):
            got[(role, kind)] += recs
            done += len(recs) * (2 if kind == "lensed" else 1)
            el = time.time() - t0
            print(f"  {done}/{total} injections {el / 60:.1f} min, ~{el / done * (total - done) / 60:.1f} min left", flush=True)
    import bilby, lal
    for (role, kind), recs in got.items():
        df = finish_role(out, fam, role, kind, recs, names_of[(role, kind)], n_det)
        print(f"  {role} {kind}: {len(df)} kept, {len(recs) - len(df)} failed", flush=True)
    meta = {"config": c, "detectors": list(CONFIGS[c]["psd"]), "psd": {k: v[1] for k, v in CONFIGS[c]["psd"].items()},
            "fmin_hz": CONFIGS[c]["fmin"], "fs_A_hz": FS, "fs_B_hz": FS_B, "band_B_hz": [CONFIGS[c]["fmin"], FS_B / 2],
            "frame_s": FRAME_S, "tail_after_merger_s": TAIL_S, "stored_A": LEN_A, "stored_B": LEN_B,
            "window_A": WIN_A, "window_B": WIN_B, "eval_offset_A": EVAL_OFF_A, "eval_offset_B": EVAL_OFF_B,
            "approximant": args.approximant, "reference_frequency_hz": F_REF, "max_inspiral_s": MAX_INSPIRAL_S,
            "spins": "a ~ Beta(2,6); cos tilt: 0.66 N(1,1.5) truncated to [-1,1] + 0.34 U(-1,1); phi_12, phi_jl ~ U(0,2pi)",
            "whitening": "X_k = h(f_k) sqrt(2 fs / S(f_k)) on [fmin, fs/2); sum x^2 = optimal S/N^2; noise unit variance per sample",
            "storage": "float16 whitened noiseless signal (sig_*); 0228 also float16 noisy data (data_*) with noise seeds in the tables",
            "optimal_SNR_arrays": "network optimal S/N over all detectors from fmin",
            "selection": f"ler detection (network S/N >= {args.snr_th} from 20 Hz per image, two images); min delay {args.min_delay_s} s; "
                         f"detector-frame total mass <= {args.mtot_det_max}",
            "pair_stats": stats, "assignment_seed": args.seed,
            "software": {"python": platform.python_version(), "numpy": np.__version__, "bilby": bilby.__version__, "lal": lal.__version__},
            "wall_s": time.time() - t0}
    (out / "generation_meta_v3.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
