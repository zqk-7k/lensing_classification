#!/usr/bin/env python3
"""Lossless compaction of the 24 s catalogs for everything the analysis reads.

Every script in experiments/reproducibility and experiments/apjs_revision reads only the last
8192 samples of each strain row (pad_or_trim / [-8192::2]). This writes, for each catalog
directory, the same file names with only that window kept, in the original dtype, so the
frozen scripts run unchanged with --data-root pointed at the compact root and see bit-identical
inputs. Time arrays are replaced by the GPS time of the first kept sample (time_window_start.npy)
plus the sampling step (time_window_meta.json). The unused "test_*" twin catalogs are not copied.
A manifest records, per file, the source SHA-256 (optional, slow), the source shape and a SHA-256 of
the kept window, so the derivation is checkable while the originals exist.
"""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np

KEEP = 8192


def sha(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True); ap.add_argument("--dst-root", required=True)
    ap.add_argument("--dirs", required=True, help="comma-separated catalog directory names")
    ap.add_argument("--hash-sources", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src_root), Path(a.dst_root)
    report = {}
    for d in a.dirs.split(","):
        s, t = src / d, dst / d
        t.mkdir(parents=True, exist_ok=True)
        entries = {}
        for p in sorted(s.iterdir()):
            if p.name.startswith("test_"):
                entries[p.name] = "skipped (unused twin catalog)"; continue
            if p.suffix == ".npy":
                x = np.load(p, mmap_mode="r")
                if "time_array" in p.name:
                    start = np.asarray(x[:, -KEEP], dtype=np.float64)
                    dt = float(x[0, 1] - x[0, 0])
                    stem = p.stem.replace("time_array", "time_window_start")
                    np.save(t / f"{stem}.npy", start)
                    (t / f"{stem}.json").write_text(json.dumps({"dt_s": dt, "samples": KEEP, "source_shape": list(x.shape)}))
                    entries[p.name] = {"replaced_by": f"{stem}.npy", "source_shape": list(x.shape)}
                elif x.ndim == 2 and x.shape[1] > KEEP:
                    out = np.lib.format.open_memmap(t / p.name, mode="w+", dtype=x.dtype, shape=(x.shape[0], KEEP))
                    for i in range(0, x.shape[0], 250):
                        out[i:i + 250] = x[i:i + 250, -KEEP:]
                    out.flush(); del out
                    entries[p.name] = {"source_shape": list(x.shape), "kept": f"[:, -{KEEP}:]", "dtype": str(x.dtype),
                                       "kept_sha256": sha(t / p.name)}
                else:
                    shutil.copy2(p, t / p.name); entries[p.name] = {"copied": True, "sha256": sha(p)}
                if a.hash_sources and isinstance(entries[p.name], dict):
                    entries[p.name]["source_sha256"] = sha(p)
            else:
                shutil.copy2(p, t / p.name); entries[p.name] = {"copied": True}
        report[d] = entries
        print(d, "done", flush=True)
    (dst / f"COMPACT_MANIFEST_{'_'.join(a.dirs.split(','))[:80]}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
