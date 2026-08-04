#!/usr/bin/env python3
"""Recover the provenance of the CQT--DeiT training pair images and audit the split.

Why this is needed
------------------
`train_cqt_deit.py` consumes precomputed pair images, `lensed/pos_XXXX.png` and
`unlensed/neg_XXXX.png`, and parses a single trailing integer from each filename as
the source ID that decides train/validation membership. The released split manifest
therefore records only that one ID per image. From the released products alone one
cannot check who the *second* member of each pair is, and so cannot verify that no
source or unlensed event crosses the train/validation boundary through the second
member. This script closes that gap by reconstructing both members forensically.

Method
------
Each pair image is `plt.imsave(concat([spectrum(left), spectrum(right)], axis=0),
cmap="viridis")` at 112x224 per event, i.e. a 224x224 RGB image whose top half is the
left event and bottom half the right event. The rendering is deterministic, so:

1. Invert the viridis colormap to recover the normalized scalar field of each half.
   `plt.imsave` normalizes over the *concatenated* matrix, so each recovered half is a
   shared affine rescaling of the underlying spectrum.
2. Recompute `spectrum()` for every candidate event in the 0222 catalog: both lensed
   images of all sources, and the whole unlensed pool.
3. Identify each half by Pearson correlation against every candidate. Correlation is
   invariant to the affine rescaling, so the true member scores ~1.0 while unrelated
   chirp spectrograms score ~0.5-0.7.

The positive images are their own control: `pos_i` must resolve to image 1 and image 2
of source `i`, which is exactly what the pipeline is documented to build.

Audit
-----
With both members known, the script checks the properties the manuscript claims:

  * train and validation source IDs are disjoint on the LEFT member;
  * train and validation source IDs are disjoint on the RIGHT member;
  * no lensed source appears on one side in training and the other side in validation;
  * training and validation unlensed event IDs are disjoint;
  * the realized hard/easy negative proportions.

Outputs
-------
  results/audit/cqt_training_pair_manifest_{sis,pm}.csv.gz
  results/audit/cqt_training_pair_audit.json

Requires the 0222 catalogs and the rendered training images, so it must run on the
machine that holds them (`--data-root`, or `GW_DATA_ROOT`).
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use("Agg")
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "experiments" / "reproducibility" / "manifests" / "split_0222_seed42.npz"
OUTDIR = ROOT / "results" / "audit"

TARGET_LEN, STRIDE = 8192, 2
N_BINS, BINS_PER_OCTAVE, HOP, SR, FMIN = 112, 24, 16, 2048, 20.0
IMG_H, IMG_W = 112, 224
SUBSAMPLE = 2          # decimate both axes before correlating; matching is unaffected
MATCH_MIN_R = 0.99     # a genuine member correlates ~1.0; unrelated chirps ~0.5-0.7
LUT = matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255.0


# ----------------------------------------------------------------- rendering side

def recover_field(path: Path) -> np.ndarray:
    """Invert the viridis colormap to the normalized scalar field of a pair image."""
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32).reshape(-1, 3)
    colors, inverse = np.unique(rgb, axis=0, return_inverse=True)
    nearest = np.argmin(((colors[:, None, :] - LUT[None]) ** 2).sum(-1), axis=1)
    return (nearest[inverse] / 255.0).reshape(2 * IMG_H, IMG_W)


def spectrum(x: np.ndarray) -> np.ndarray:
    """The exact transform used to build the images (see prepare_cqt_cache_0228.py)."""
    x = np.asarray(x, dtype=np.float32)
    x = x[..., -TARGET_LEN:] if x.shape[-1] >= TARGET_LEN else np.pad(x, (TARGET_LEN - x.shape[-1], 0))
    x = x[..., ::STRIDE]
    x = (x - x.mean()) / (x.std() + 1e-8)
    values = librosa.cqt(x, sr=SR, fmin=FMIN, n_bins=N_BINS,
                         bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP)
    mag = librosa.amplitude_to_db(np.abs(values), ref=np.max)
    tensor = torch.tensor(mag, dtype=torch.float32)[None, None]
    return F.interpolate(tensor, size=(IMG_H, IMG_W), mode="bilinear",
                         align_corners=False).squeeze().numpy()


# ----------------------------------------------------------------- matching

def flatten(block: np.ndarray) -> np.ndarray:
    return block[::SUBSAMPLE, ::SUBSAMPLE].ravel()


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    return centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12)


def _spectrum_worker(index, path):
    array = np.load(path, mmap_mode="r")
    return flatten(spectrum(array[index]))


def build_candidates(data_root: Path, lens: str, workers: int):
    """All candidate events: both lensed images of every source, and the unlensed pool."""
    sources = [
        (f"{lens}_img1", data_root / f"{lens}_data_0222" / f"{lens}_data_strain_1.npy"),
        (f"{lens}_img2", data_root / f"{lens}_data_0222" / f"{lens}_data_strain_2.npy"),
        ("unlensed", data_root / "Unlensed_data_0222" / "unlensed_data_strain.npy"),
    ]
    labels, rows = [], []
    for kind, path in sources:
        count = np.load(path, mmap_mode="r").shape[0]
        with Pool(workers) as pool:
            chunk = pool.map(partial(_spectrum_worker, path=path), range(count), chunksize=16)
        rows.extend(chunk)
        labels.extend((kind, i) for i in range(count))
        print(f"    candidates {kind:12} {count}")
    return labels, zscore_rows(np.asarray(rows, dtype=np.float32))


def audit_family(data_root: Path, lens: str, workers: int):
    print(f"  [{lens}] recovering pair images")
    image_root = data_root / f"dataset_images_{lens}_noisy_cqt"
    pos_paths = sorted((image_root / "lensed").glob("pos_*.png"))
    neg_paths = sorted((image_root / "unlensed").glob("neg_*.png"))
    with Pool(workers) as pool:
        pos_fields = pool.map(recover_field, pos_paths, chunksize=8)
        neg_fields = pool.map(recover_field, neg_paths, chunksize=8)

    print(f"  [{lens}] recomputing candidate spectra")
    labels, candidates = build_candidates(data_root, lens, workers)

    def identify(fields, half):
        block = np.asarray([flatten(f[:IMG_H] if half == "top" else f[IMG_H:]) for f in fields],
                           dtype=np.float32)
        scores = zscore_rows(block) @ candidates.T
        best = scores.argmax(axis=1)
        return [labels[b] for b in best], scores[np.arange(len(best)), best]

    print(f"  [{lens}] matching halves")
    pos_left, pos_left_r = identify(pos_fields, "top")
    pos_right, pos_right_r = identify(pos_fields, "bottom")
    neg_left, neg_left_r = identify(neg_fields, "top")
    neg_right, neg_right_r = identify(neg_fields, "bottom")

    split = np.load(SPLIT, allow_pickle=False)
    prefix = lens.lower()
    train_ids = set(map(int, split[f"{prefix}_train_source_ids"]))
    val_ids = set(map(int, split[f"{prefix}_val_source_ids"]))
    unl_train = set(map(int, split["unlensed_train_source_ids"]))
    unl_val = set(map(int, split["unlensed_val_source_ids"]))

    records = []
    for group, paths, left, left_r, right, right_r in (
        ("positive", pos_paths, pos_left, pos_left_r, pos_right, pos_right_r),
        ("negative", neg_paths, neg_left, neg_left_r, neg_right, neg_right_r),
    ):
        for path, (lk, li), lr, (rk, ri), rr in zip(paths, left, left_r, right, right_r):
            declared = int(path.stem.split("_")[-1])
            if group == "positive":
                negative_type = ""
            else:
                negative_type = "easy" if rk == "unlensed" else "hard"
            records.append({
                "pair_image_path": str(path.relative_to(data_root)),
                "label": 1 if group == "positive" else 0,
                "negative_type": negative_type,
                "declared_source_id": declared,
                "left_event_id": f"{lk}_{li:04d}", "left_source_id": li, "left_kind": lk,
                "right_event_id": f"{rk}_{ri:04d}", "right_source_id": ri, "right_kind": rk,
                "unlensed_event_id": ri if rk == "unlensed" else -1,
                "split": "train" if declared in train_ids else ("val" if declared in val_ids else "UNASSIGNED"),
                "match_r_left": float(lr), "match_r_right": float(rr),
            })

    import pandas as pd
    frame = pd.DataFrame.from_records(records)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / f"cqt_training_pair_manifest_{prefix}.csv.gz"
    frame.to_csv(out_csv, index=False, compression={"method": "gzip", "mtime": 0})

    # ---------------- assertions the manuscript relies on ----------------
    weak = frame[(frame.match_r_left < MATCH_MIN_R) | (frame.match_r_right < MATCH_MIN_R)]
    pos = frame[frame.label == 1]
    neg = frame[frame.label == 0]
    train, val = frame[frame.split == "train"], frame[frame.split == "val"]

    def lensed_sources(sub, side):
        mask = sub[f"{side}_kind"].str.startswith(lens)
        return set(sub.loc[mask, f"{side}_source_id"].astype(int))

    train_sources = lensed_sources(train, "left") | lensed_sources(train, "right")
    val_sources = lensed_sources(val, "left") | lensed_sources(val, "right")
    train_unlensed = set(train.loc[train.right_kind == "unlensed", "right_source_id"].astype(int))
    val_unlensed = set(val.loc[val.right_kind == "unlensed", "right_source_id"].astype(int))

    audit = {
        "lens": lens,
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "min_match_r": float(frame[["match_r_left", "match_r_right"]].min().min()),
        "n_images_below_match_threshold": int(len(weak)),
        "match_threshold": MATCH_MIN_R,
        "positive_is_img1_img2": bool(
            (pos.left_kind == f"{lens}_img1").all() and (pos.right_kind == f"{lens}_img2").all()
            and (pos.left_source_id == pos.declared_source_id).all()
            and (pos.right_source_id == pos.declared_source_id).all()),
        "negative_left_is_declared_source": bool(
            (neg.left_kind == f"{lens}_img1").all()
            and (neg.left_source_id == neg.declared_source_id).all()),
        "negative_type_counts": {k: int(v) for k, v in neg.negative_type.value_counts().items()},
        "negative_hard_fraction": float((neg.negative_type == "hard").mean()),
        "n_unassigned_split": int((frame.split == "UNASSIGNED").sum()),
        "train_val_lensed_sources_disjoint_both_sides": bool(train_sources.isdisjoint(val_sources)),
        "n_lensed_sources_crossing_split": len(train_sources & val_sources),
        "lensed_sources_crossing_split": sorted(train_sources & val_sources)[:50],
        "train_val_unlensed_disjoint": bool(train_unlensed.isdisjoint(val_unlensed)),
        "n_unlensed_crossing_split": len(train_unlensed & val_unlensed),
        "unlensed_crossing_split": sorted(train_unlensed & val_unlensed)[:50],
        "train_unlensed_outside_declared_train_pool": len(train_unlensed - unl_train),
        "val_unlensed_outside_declared_val_pool": len(val_unlensed - unl_val),
        "manifest": str(out_csv.relative_to(ROOT)),
    }
    audit["PASS"] = bool(
        audit["n_images_below_match_threshold"] == 0
        and audit["positive_is_img1_img2"]
        and audit["negative_left_is_declared_source"]
        and audit["n_unassigned_split"] == 0
        and audit["train_val_lensed_sources_disjoint_both_sides"]
        and audit["train_val_unlensed_disjoint"])
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("GW_DATA_ROOT", str(ROOT / "data")))
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    args = parser.parse_args()
    data_root = Path(args.data_root)

    report = {"data_root": "${GW_DATA_ROOT}", "subsample": SUBSAMPLE, "families": {}}
    for lens in ("SIS", "PM"):
        report["families"][lens] = audit_family(data_root, lens, args.workers)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "cqt_training_pair_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n================ CQT TRAINING PAIR PROVENANCE AUDIT ================")
    for lens, entry in report["families"].items():
        print(f"\n[{lens}]  {'PASS' if entry['PASS'] else 'FAIL'}")
        print(f"  worst half-match correlation      {entry['min_match_r']:.6f} "
              f"({entry['n_images_below_match_threshold']} below {MATCH_MIN_R})")
        print(f"  pos_i == (img1[i], img2[i])       {entry['positive_is_img1_img2']}")
        print(f"  neg_i left == img1[i]             {entry['negative_left_is_declared_source']}")
        print(f"  negative types                    {entry['negative_type_counts']} "
              f"(hard fraction {entry['negative_hard_fraction']:.3f})")
        print(f"  lensed sources disjoint (2 sides) {entry['train_val_lensed_sources_disjoint_both_sides']} "
              f"(crossing: {entry['n_lensed_sources_crossing_split']})")
        print(f"  unlensed train/val disjoint       {entry['train_val_unlensed_disjoint']} "
              f"(crossing: {entry['n_unlensed_crossing_split']})")
        print(f"  unlensed outside declared pools   train {entry['train_unlensed_outside_declared_train_pool']}, "
              f"val {entry['val_unlensed_outside_declared_val_pool']}")
    print(f"\nwrote {(OUTDIR / 'cqt_training_pair_audit.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
