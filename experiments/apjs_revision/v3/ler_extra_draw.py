#!/usr/bin/env python3
"""Draw additional detectable lensed systems and unlensed events with the same LeR settings
as ler_population_rev.py (same detection definition: network optimal S/N >= 8 from 20 Hz per
image, two detected images for a lensed system), new seeds, separate output files.

The v3 catalogs (generate_v3.py) pool the first draw and these extra draws; nothing in the first
draw is modified.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ler_population_rev import build_ler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=["ET", "O4"])
ap.add_argument("--work-dir", required=True)
ap.add_argument("--n-lensed", type=int, required=True)
ap.add_argument("--n-unlensed", type=int, required=True)
ap.add_argument("--seed", type=int, default=20261108)
ap.add_argument("--npool", type=int, default=24)
ap.add_argument("--tag", default="v3extra")
a = ap.parse_args()
work = Path(a.work_dir)
t0 = time.time()
np.random.seed(a.seed)
ler = build_ler(a.config, work, a.npool, 8.0, 500.0)
print(f"[{a.config}] LeR ready after {time.time() - t0:.0f} s", flush=True)
meta = {"config": a.config, "seed": a.seed, "tag": a.tag, "ler_version": __import__("ler").__version__}
if a.n_unlensed > 0:
    np.random.seed(a.seed + 1); t = time.time()
    up = ler.selecting_n_unlensed_detectable_events(size=a.n_unlensed, batch_size=50000, pdet_threshold=0.5, resume=False,
                                                    output_jsonfile=f"n_unlensed_detectable_{a.config}_{a.tag}.json",
                                                    meta_data_file=f"meta_unlensed_{a.config}_{a.tag}.json", trim_to_size=True)
    up = up[1] if isinstance(up, tuple) else up
    meta["n_unlensed"] = int(len(np.asarray(up["zs"]))); meta["unlensed_wall_s"] = time.time() - t
    print(f"[{a.config}] {meta['n_unlensed']} unlensed in {meta['unlensed_wall_s']:.0f} s", flush=True)
if a.n_lensed > 0:
    np.random.seed(a.seed + 2); t = time.time()
    lp = ler.selecting_n_lensed_detectable_events(size=a.n_lensed, batch_size=50000, pdet_threshold=[0.5, 0.5], num_img=[1, 1],
                                                  resume=False, output_jsonfile=f"n_lensed_detectable_{a.config}_{a.tag}.json",
                                                  meta_data_file=f"meta_lensed_{a.config}_{a.tag}.json", trim_to_size=True)
    lp = lp[1] if isinstance(lp, tuple) else lp
    meta["n_lensed"] = int(len(np.asarray(lp["zs"]))); meta["lensed_wall_s"] = time.time() - t
    print(f"[{a.config}] {meta['n_lensed']} lensed systems in {meta['lensed_wall_s']:.0f} s", flush=True)
(work / f"ler_data_{a.config}" / f"{a.tag}_meta.json").write_text(json.dumps(meta, indent=2))
print("done", flush=True)
