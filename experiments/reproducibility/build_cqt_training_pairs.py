#!/usr/bin/env python3
"""Rebuild the 0222 CQT--DeiT training pair images with split-respecting negatives.

Why this replaces the original set
----------------------------------
`audit_cqt_training_pairs.py` recovered the provenance of the original pair images
and showed that their negatives drew the second member from the FULL 0222 pools,
ignoring the source-level train/validation split (see docs/CQT_PAIR_PROVENANCE.md).
482 (SIS) and 484 (PM) lensed sources therefore appeared on one side in training and
the other side in validation. This script rebuilds the set so that both members of
every pair stay inside their own split, and writes the manifest at generation time so
the provenance never has to be recovered forensically again.

Negative construction mirrors `PairDataset._build_fixed_pairs` in
`src/classifier/pair_dataset.py`, the rule PI-ResNet already used, so the two
pipelines now construct negatives identically:

    for each source s in the split:
        positive: (image 1 of s, image 2 of s)
        negative: with probability 0.7  -> (image 1 of s, image 2 of another source
                                            drawn from the SAME split)
                  otherwise             -> (image 1 of s, an unlensed event drawn from
                                            the SAME split's unlensed pool)

The rendering is byte-for-byte the transform of `prepare_cqt_cache_0228.py`: each
segment is truncated to the last 8192 samples, decimated by 2, standardized to zero
mean and unit variance, transformed with `librosa.cqt(sr=2048, fmin=20, n_bins=112,
bins_per_octave=24, hop_length=16)`, converted to dB against its own maximum, resized
to 112x224, concatenated along the frequency axis, and written with
`plt.imsave(..., cmap="viridis")`.

Outputs
-------
  <out-root>/dataset_images_{LENS}_noisy_cqt/lensed/pos_SSSS.png
  <out-root>/dataset_images_{LENS}_noisy_cqt/unlensed/neg_SSSS.png
  <out-root>/dataset_images_{LENS}_noisy_cqt/pair_manifest.csv
  results/audit/cqt_training_pairs_v2_manifest_{sis,pm}.csv.gz
  results/audit/cqt_training_pairs_v2_audit.json

`SSSS` is the left source ID, so `train_cqt_deit.py` continues to resolve the split
from the filename without modification. Point it at the new tree with `--image-root`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import config as cfg  # noqa: E402
from prepare_cqt_cache_0228 import spectrum  # noqa: E402  (identical transform)

SPLIT = ROOT / "experiments" / "reproducibility" / "manifests" / "split_0222_seed42.npz"
OUTDIR = ROOT / "results" / "audit"
HARD_P = cfg.NEG_RATIO["diff_event"] / (cfg.NEG_RATIO["diff_event"] + cfg.NEG_RATIO["noise"])


def build_pair_plan(split, lens, seed):
    """Mirror PairDataset._build_fixed_pairs, per split, with a per-split RandomState."""
    prefix = lens.lower()
    plan = []
    for part in ("train", "val"):
        sources = np.asarray(split[f"{prefix}_{part}_source_ids"])
        unlensed = np.asarray(split[f"unlensed_{part}_source_ids"])
        rng = np.random.RandomState(seed)
        for source in sources:
            source = int(source)
            plan.append({"source": source, "label": 1, "negative_type": "",
                         "right_kind": "img2", "right_id": source, "split": part})
            if rng.rand() < HARD_P:
                other = int(rng.choice(sources))
                while other == source:
                    other = int(rng.choice(sources))
                plan.append({"source": source, "label": 0, "negative_type": "hard",
                             "right_kind": "img2", "right_id": other, "split": part})
            else:
                other = int(unlensed[rng.randint(len(unlensed))])
                plan.append({"source": source, "label": 0, "negative_type": "easy",
                             "right_kind": "unlensed", "right_id": other, "split": part})
    return plan


def _spectrum_task(task):
    kind, index, path = task
    array = np.load(path, mmap_mode="r")
    return (kind, index), spectrum(array[index])


def render(args):
    out_path, left, right = args
    plt.imsave(out_path, np.concatenate([left, right], axis=0), cmap="viridis")
    return out_path


def build_family(data_root: Path, out_root: Path, lens: str, seed: int, workers: int):
    split = np.load(SPLIT, allow_pickle=False)
    plan = build_pair_plan(split, lens, seed)

    source_dir = data_root / f"{lens}_data_0222"
    paths = {
        "img1": source_dir / f"{lens}_data_strain_1.npy",
        "img2": source_dir / f"{lens}_data_strain_2.npy",
        "unlensed": data_root / "Unlensed_data_0222" / "unlensed_data_strain.npy",
    }
    needed = {("img1", row["source"]) for row in plan}
    needed |= {(row["right_kind"], row["right_id"]) for row in plan}
    tasks = [(kind, index, paths[kind]) for kind, index in sorted(needed)]
    print(f"  [{lens}] computing {len(tasks)} distinct event spectra")
    with Pool(workers) as pool:
        spectra = dict(pool.map(_spectrum_task, tasks, chunksize=16))

    image_root = out_root / f"dataset_images_{lens}_noisy_cqt"
    (image_root / "lensed").mkdir(parents=True, exist_ok=True)
    (image_root / "unlensed").mkdir(parents=True, exist_ok=True)

    jobs, records = [], []
    for row in plan:
        source = row["source"]
        sub, stem = ("lensed", "pos") if row["label"] == 1 else ("unlensed", "neg")
        out_path = image_root / sub / f"{stem}_{source:04d}.png"
        jobs.append((str(out_path), spectra[("img1", source)],
                     spectra[(row["right_kind"], row["right_id"])]))
        records.append({
            "pair_image_path": str(out_path.relative_to(out_root)),
            "label": row["label"], "negative_type": row["negative_type"],
            "left_event_id": f"{lens}_img1_{source:04d}", "left_source_id": source,
            "left_kind": f"{lens}_img1",
            "right_event_id": (f"{lens}_img2_{row['right_id']:04d}" if row["right_kind"] == "img2"
                               else f"unlensed_{row['right_id']:04d}"),
            "right_source_id": row["right_id"],
            "right_kind": f"{lens}_img2" if row["right_kind"] == "img2" else "unlensed",
            "unlensed_event_id": row["right_id"] if row["right_kind"] == "unlensed" else -1,
            "split": row["split"],
        })
    print(f"  [{lens}] rendering {len(jobs)} pair images")
    with Pool(workers) as pool:
        pool.map(render, jobs, chunksize=16)

    import pandas as pd
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(image_root / "pair_manifest.csv", index=False)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTDIR / f"cqt_training_pairs_v2_manifest_{lens.lower()}.csv.gz",
                 index=False, compression={"method": "gzip", "mtime": 0})

    # ---------------- the audit that the original set failed ----------------
    train, val = frame[frame.split == "train"], frame[frame.split == "val"]

    def lensed_sources(sub):
        ids = set(sub.left_source_id.astype(int))
        mask = sub.right_kind == f"{lens}_img2"
        return ids | set(sub.loc[mask, "right_source_id"].astype(int))

    def unlensed_ids(sub):
        mask = sub.right_kind == "unlensed"
        return set(sub.loc[mask, "right_source_id"].astype(int))

    train_sources, val_sources = lensed_sources(train), lensed_sources(val)
    train_unl, val_unl = unlensed_ids(train), unlensed_ids(val)
    declared_unl_train = set(map(int, split["unlensed_train_source_ids"]))
    declared_unl_val = set(map(int, split["unlensed_val_source_ids"]))
    negatives = frame[frame.label == 0]

    audit = {
        "lens": lens, "seed": seed,
        "n_images": int(len(frame)), "n_positive": int((frame.label == 1).sum()),
        "n_negative": int(len(negatives)),
        "negative_type_counts": {k: int(v) for k, v in negatives.negative_type.value_counts().items()},
        "hard_fraction": float((negatives.negative_type == "hard").mean()),
        "train_val_lensed_sources_disjoint_both_sides": bool(train_sources.isdisjoint(val_sources)),
        "n_lensed_sources_crossing_split": len(train_sources & val_sources),
        "train_val_unlensed_disjoint": bool(train_unl.isdisjoint(val_unl)),
        "n_unlensed_crossing_split": len(train_unl & val_unl),
        "train_unlensed_outside_declared_pool": len(train_unl - declared_unl_train),
        "val_unlensed_outside_declared_pool": len(val_unl - declared_unl_val),
        "image_root": f"${{GW_CQT_V2_ROOT}}/dataset_images_{lens}_noisy_cqt",
    }
    audit["PASS"] = bool(
        audit["train_val_lensed_sources_disjoint_both_sides"]
        and audit["train_val_unlensed_disjoint"]
        and audit["train_unlensed_outside_declared_pool"] == 0
        and audit["val_unlensed_outside_declared_pool"] == 0)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("GW_DATA_ROOT", str(ROOT / "data")))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    args = parser.parse_args()

    report = {"seed": args.seed, "hard_probability": HARD_P,
              "negative_rule": "mirrors PairDataset._build_fixed_pairs, restricted to each split",
              "families": {}}
    for lens in ("SIS", "PM"):
        report["families"][lens] = build_family(Path(args.data_root), Path(args.out_root),
                                                lens, args.seed, args.workers)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "cqt_training_pairs_v2_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n============ REBUILT CQT TRAINING PAIR SET ============")
    for lens, entry in report["families"].items():
        print(f"\n[{lens}]  {'PASS' if entry['PASS'] else 'FAIL'}")
        print(f"  images                            {entry['n_images']} "
              f"({entry['n_positive']} positive, {entry['n_negative']} negative)")
        print(f"  negative types                    {entry['negative_type_counts']} "
              f"(hard fraction {entry['hard_fraction']:.3f})")
        print(f"  lensed sources disjoint (2 sides) {entry['train_val_lensed_sources_disjoint_both_sides']} "
              f"(crossing: {entry['n_lensed_sources_crossing_split']})")
        print(f"  unlensed train/val disjoint       {entry['train_val_unlensed_disjoint']} "
              f"(crossing: {entry['n_unlensed_crossing_split']})")
        print(f"  unlensed outside declared pools   train {entry['train_unlensed_outside_declared_pool']}, "
              f"val {entry['val_unlensed_outside_declared_pool']}")


if __name__ == "__main__":
    main()
