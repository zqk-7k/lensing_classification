#!/usr/bin/env python3
"""Score the 0228 pair manifests of a v3 root with a trained v3 model (stored noisy data, evaluation
offsets, no augmentation). Order-dependent models (early fusion, CQT images) are scored in both
orders and the logits averaged, so every model is symmetric in the two events.

Output: <out>/<variant>_predictions_0228_<fam>.csv.gz with the manifest keys and logit, score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3lib as L  # noqa: E402

KEYS = ["pair_id", "label", "source_block_id", "negative_type", "calibration_or_evaluation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--family", default="SIS")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    dev = torch.device(a.device)
    root, run = Path(a.root), Path(a.run_dir)
    ck = torch.load(run / a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    meta = L.load_meta(root)
    lb, ub = root / f"{a.family}_data_0228", root / "Unlensed_data_0228"
    arr = {("lensed_img1", s): np.load(lb / f"{s}_1.npy", mmap_mode="r") for s in ("data_A", "data_B", "sig_A")}
    arr |= {("lensed_img2", s): np.load(lb / f"{s}_2.npy", mmap_mode="r") for s in ("data_A", "data_B", "sig_A")}
    arr |= {("unlensed", s): np.load(ub / f"{s}.npy", mmap_mode="r") for s in ("data_A", "data_B", "sig_A")}
    n_det = arr[("unlensed", "data_A")].shape[1]
    model, v, _ = L.build_model(cfg["variant"], n_det, float(meta["fmin_hz"]), dev, Path(cfg["pretrained"]))
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    symmetric = v["kind"] == "pi"
    fam = a.family.lower()
    man = pd.concat([pd.read_csv(Path(a.manifest_dir) / f"0228_{fam}_{p}_pairs.csv.gz") for p in ("calibration", "evaluation")],
                    ignore_index=True)

    def load(types, idx):
        out = {}
        for s, length, off in (("data_A", L.WIN_A, L.EVAL_OFF_A), ("data_B", L.WIN_B, L.EVAL_OFF_B), ("sig_A", L.WIN_A, L.EVAL_OFF_A)):
            x = np.empty((len(idx), n_det, length), np.float32)
            for t in np.unique(types):
                m = types == t
                rows = idx[m]
                order = np.argsort(rows)
                block = arr[(t, s)][rows[order]][:, :, off:off + length]
                tmp = np.empty_like(block); tmp[order] = block
                x[m] = tmp
            out[s] = torch.from_numpy(x).to(dev)
        return out

    logits = np.empty(len(man), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(man), a.batch):
            e = min(len(man), s + a.batch)
            m = man.iloc[s:e]
            lt, li = m.left_event_type.to_numpy(), m.left_event_index.to_numpy()
            rt, ri = m.right_event_type.to_numpy(), m.right_event_index.to_numpy()
            l, r = load(lt, li), load(rt, ri)
            a1, b1 = L.prepare_inputs(v, n_det, l["sig_A"], None, l["data_A"], l["data_B"])
            a2, b2 = L.prepare_inputs(v, n_det, r["sig_A"], None, r["data_A"], r["data_B"])
            z = L.logit_of(model(a1, b1, a2, b2)).float()
            if not symmetric:
                z = 0.5 * (z + L.logit_of(model(a2, b2, a1, b1)).float())
            logits[s:e] = z.cpu().numpy()
            if (s // a.batch) % 50 == 0:
                print(f"  {e}/{len(man)}", flush=True)
    out = man[KEYS].copy()
    out["logit"] = logits.astype(np.float64)
    out["score"] = 1.0 / (1.0 + np.exp(-out["logit"]))
    od = Path(a.output_dir); od.mkdir(parents=True, exist_ok=True)
    path = od / f"{cfg['variant']}_predictions_0228_{fam}.csv.gz"
    out.to_csv(path, index=False, compression="gzip")
    (od / f"{cfg['variant']}_predictions_0228_{fam}.metadata.json").write_text(json.dumps({
        "run_dir": str(run), "checkpoint": a.checkpoint, "checkpoint_step": ck.get("step"), "val_auc": ck.get("val_auc"),
        "checkpoint_sha256": hashlib.sha256((run / a.checkpoint).read_bytes()).hexdigest(),
        "manifest_dir": a.manifest_dir, "symmetrized": not symmetric, "rows": len(out)}, indent=2))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
