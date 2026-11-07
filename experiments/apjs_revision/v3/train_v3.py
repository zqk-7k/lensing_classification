#!/usr/bin/env python3
"""Train one v3 model variant on the 0222 catalog of a v3 root (see v3lib.py for the variants).

Recipe, identical for every variant except the optimizer settings of the image model (kept at the
values of the original CQT-DeiT script):
  * signals live on the GPU; every step draws fresh whitened Gaussian noise (no fixed noise, so the
    network cannot memorize realizations, which is what the v2 runs did: train acc 0.94, val AUC 0.76);
  * pairs per step: half positives (image 1, image 2 of one system), half negatives, of which 70 %
    "hard" (images of two different systems, image slots drawn at random as in the 0228 manifests)
    and 30 % "easy" (an image and an unlensed event); left/right order shuffled;
  * jitter: each event's window shifted by up to +/-0.125 s; a global sign flip per event;
  * curriculum: the signal amplitude is multiplied by kappa, decaying linearly from --kappa0 to 1 over
    the first --curriculum-frac of the steps, then 1 (evaluation never scales signals);
  * selection: the checkpoint with the best validation AUC on a fixed validation pair set (0222
    validation split, fixed noise, evaluation offsets). 0228 is not read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3lib as L  # noqa: E402

ROOT = L.ROOT


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 23), b""):
            h.update(c)
    return h.hexdigest()


def split_ids(n, frac, seed):
    ids = np.random.RandomState(seed).permutation(n)
    k = int(frac * n)
    return np.sort(ids[:k]), np.sort(ids[k:])


class Bank:
    """All 0222 signals on the GPU: rows [img1 of lensed | img2 of lensed | unlensed]."""

    def __init__(self, root: Path, fam: str, device):
        lb, ub = root / f"{fam}_data_0222", root / "Unlensed_data_0222"
        parts_a = [np.load(lb / "sig_A_1.npy", mmap_mode="r"), np.load(lb / "sig_A_2.npy", mmap_mode="r"), np.load(ub / "sig_A.npy", mmap_mode="r")]
        parts_b = [np.load(lb / "sig_B_1.npy", mmap_mode="r"), np.load(lb / "sig_B_2.npy", mmap_mode="r"), np.load(ub / "sig_B.npy", mmap_mode="r")]
        self.nL, self.nU = parts_a[0].shape[0], parts_a[2].shape[0]
        self.n_det = parts_a[0].shape[1]
        self.A = torch.empty((2 * self.nL + self.nU,) + parts_a[0].shape[1:], dtype=torch.float16, device=device)
        self.B = torch.empty((2 * self.nL + self.nU,) + parts_b[0].shape[1:], dtype=torch.float16, device=device)
        o = 0
        for pa, pb in zip(parts_a, parts_b):
            for s in range(0, pa.shape[0], 2000):
                e = min(pa.shape[0], s + 2000)
                self.A[o + s:o + e] = torch.from_numpy(np.ascontiguousarray(pa[s:e])).to(device)
                self.B[o + s:o + e] = torch.from_numpy(np.ascontiguousarray(pb[s:e])).to(device)
            o += pa.shape[0]

    def row(self, kind, idx):  # kind 1, 2 (lensed image) or 0 (unlensed)
        return np.where(kind == 1, idx, np.where(kind == 2, self.nL + idx, 2 * self.nL + idx))


def sample_pairs(rng, n, lens_ids, unl_ids):
    """Returns (left_row_kind, left_idx, right_kind, right_idx, label) arrays of length n."""
    npos = n // 2
    lk, li, rk, ri, y = [], [], [], [], []
    src = rng.choice(lens_ids, npos)
    lk += [1] * npos; li += list(src); rk += [2] * npos; ri += list(src); y += [1] * npos
    for _ in range(n - npos):
        i = int(rng.choice(lens_ids))
        if rng.rand() < 0.7:
            j = int(rng.choice(lens_ids))
            while j == i:
                j = int(rng.choice(lens_ids))
            lk.append(int(rng.randint(1, 3))); li.append(i); rk.append(int(rng.randint(1, 3))); ri.append(j)
        else:
            lk.append(int(rng.randint(1, 3))); li.append(i); rk.append(0); ri.append(int(rng.choice(unl_ids)))
        y.append(0)
    lk, li, rk, ri, y = map(np.asarray, (lk, li, rk, ri, y))
    swap = rng.rand(n) < 0.5
    lk2, li2 = np.where(swap, rk, lk), np.where(swap, ri, li)
    rk2, ri2 = np.where(swap, lk, rk), np.where(swap, li, ri)
    return lk2, li2, rk2, ri2, y


def fixed_val_pairs(lens_ids, unl_ids, seed=42):
    rng = np.random.RandomState(seed)
    out = []
    for i in lens_ids:
        out.append((1, int(i), 2, int(i), 1))
        if rng.rand() < 0.7:
            j = int(rng.choice(lens_ids))
            while j == i:
                j = int(rng.choice(lens_ids))
            out.append((int(rng.randint(1, 3)), int(i), int(rng.randint(1, 3)), j, 0))
        else:
            out.append((int(rng.randint(1, 3)), int(i), 0, int(rng.choice(unl_ids)), 0))
    return [np.asarray(c) for c in zip(*out)]


def build_batch(bank, noise, v, rows_l, rows_r, off_l, off_r, kappa, sign_l, sign_r, gen=None):
    dev = bank.A.device
    outs = []
    for rows, off, sign in ((rows_l, off_l, sign_l), (rows_r, off_r, sign_r)):
        r = torch.as_tensor(rows, device=dev)
        o = torch.as_tensor(off, device=dev)
        sa = L.window(bank.A[r].float(), 16 * o, L.WIN_A) * (kappa * sign).view(-1, 1, 1)
        sb = L.window(bank.B[r].float(), o, L.WIN_B) * (kappa * sign).view(-1, 1, 1)
        na, nb = noise(len(rows) * bank.n_det, gen)
        da = sa + na.view(len(rows), bank.n_det, -1)
        db = sb + nb.view(len(rows), bank.n_det, -1)
        outs.append(L.prepare_inputs(v, bank.n_det, sa, sb, da, db))
    (a1, b1), (a2, b2) = outs
    return a1, b1, a2, b2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--family", default="SIS")
    ap.add_argument("--variant", required=True, choices=list(L.VARIANTS))
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--kappa0", type=float, default=2.0)
    ap.add_argument("--curriculum-frac", type=float, default=0.25)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--train-fraction", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pretrained", default=str(ROOT / "artifacts" / "pretrained" / "deit_tiny_distilled_patch16_224-b40b3cf7.pth"))
    ap.add_argument("--benchmark", type=int, default=0, help="time this many steps and exit")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device(args.device)
    root = Path(args.root)
    meta = L.load_meta(root)
    fmin = float(meta["fmin_hz"])
    bank = Bank(root, args.family, dev)
    print(f"bank: {bank.nL} lensed, {bank.nU} unlensed, {bank.n_det} detectors, "
          f"{(bank.A.numel() + bank.B.numel()) * 2 / 1e9:.2f} GB on {dev}", flush=True)
    tr_l, va_l = split_ids(bank.nL, args.train_fraction, args.seed + 101)
    tr_u, va_u = split_ids(bank.nU, args.train_fraction, args.seed + 303)
    model, v, n_det = build_model_and_check(args, bank, fmin, dev)
    is_cqt = v["kind"] == "cqt"
    lr = args.lr if args.lr is not None else (5e-5 if is_cqt else 1e-4)
    wd = args.weight_decay if args.weight_decay is not None else (5e-4 if is_cqt else 1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              max(1e-2, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1) if is_cqt else nn.BCEWithLogitsLoss()
    noise = L.Noise(fmin, dev)
    rng = np.random.RandomState(args.seed)
    vl = fixed_val_pairs(va_l, va_u)
    run = Path(args.output_root) / f"{args.variant}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    cfg = vars(args) | {"lr": lr, "weight_decay": wd, "variant_spec": v, "n_det_used": n_det, "fmin_hz": fmin,
                        "n_train_lensed": len(tr_l), "n_val_lensed": len(va_l), "n_train_unlensed": len(tr_u), "n_val_unlensed": len(va_u),
                        "n_val_pairs": len(vl[0]), "parameters": sum(p.numel() for p in model.parameters()),
                        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                        "torch": torch.__version__, "python": platform.python_version(), "generation_meta": meta}
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    np.savez_compressed(run / "split.npz", train_lensed=tr_l, val_lensed=va_l, train_unlensed=tr_u, val_unlensed=va_u)

    def evaluate():
        model.eval()
        g = torch.Generator(device=dev)
        scores = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for bi, s in enumerate(range(0, len(vl[0]), 256)):
                e = min(len(vl[0]), s + 256)
                g.manual_seed(10_000 + bi)
                n = e - s
                off = np.full(n, 16)
                a1, b1, a2, b2 = build_batch(bank, noise, v, bank.row(vl[0][s:e], vl[1][s:e]), bank.row(vl[2][s:e], vl[3][s:e]),
                                             off, off, 1.0, torch.ones(n, device=dev), torch.ones(n, device=dev), g)
                scores.append(L.logit_of(model(a1, b1, a2, b2)).float().cpu().numpy())
        model.train()
        sc = np.concatenate(scores)
        y = vl[4]
        loss = float(np.mean(np.logaddexp(0, sc) - y * sc))
        return float(roc_auc_score(y, sc)), loss

    best, t0 = -1.0, time.time()
    hist = open(run / "history.csv", "w", newline="")
    w = csv.DictWriter(hist, fieldnames=["step", "train_loss", "val_loss", "val_auc", "lr", "kappa", "minutes"])
    w.writeheader()
    model.train()
    run_loss, nrun = 0.0, 0
    for step in range(1, args.steps + 1):
        kappa = 1.0 + (args.kappa0 - 1.0) * max(0.0, 1.0 - step / (args.curriculum_frac * args.steps))
        lk, li, rk, ri, y = sample_pairs(rng, args.batch, tr_l, tr_u)
        n = args.batch
        offl, offr = rng.randint(0, L.MAX_OFF_UNITS + 1, n), rng.randint(0, L.MAX_OFF_UNITS + 1, n)
        sl = torch.as_tensor(np.where(rng.rand(n) < 0.5, -1.0, 1.0), device=dev, dtype=torch.float32)
        sr = torch.as_tensor(np.where(rng.rand(n) < 0.5, -1.0, 1.0), device=dev, dtype=torch.float32)
        with torch.no_grad():
            a1, b1, a2, b2 = build_batch(bank, noise, v, bank.row(lk, li), bank.row(rk, ri), offl, offr, kappa, sl, sr)
        yt = torch.as_tensor(y, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(a1, b1, a2, b2)
            loss = crit(out.float(), yt.long()) if is_cqt else crit(out.float(), yt.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run_loss += float(loss); nrun += 1
        if args.benchmark and step == args.benchmark:
            torch.cuda.synchronize()
            print(json.dumps({"variant": args.variant, "steps": step, "seconds": time.time() - t0,
                              "pairs_per_s": step * n / (time.time() - t0), "max_mem_GB": torch.cuda.max_memory_allocated(dev) / 1e9}))
            return
        if step % args.val_every == 0 or step == args.steps:
            auc, vloss = evaluate()
            row = {"step": step, "train_loss": run_loss / nrun, "val_loss": vloss, "val_auc": auc,
                   "lr": opt.param_groups[0]["lr"], "kappa": kappa, "minutes": (time.time() - t0) / 60}
            w.writerow(row); hist.flush(); run_loss, nrun = 0.0, 0
            if auc > best:
                best = auc
                torch.save({"model_state_dict": model.state_dict(), "step": step, "val_auc": auc, "config": cfg}, run / "best.pt")
            print(json.dumps(row), flush=True)
    torch.save({"model_state_dict": model.state_dict(), "step": args.steps, "config": cfg}, run / "last.pt")
    (run / "summary.json").write_text(json.dumps({"best_val_auc": best, "minutes": (time.time() - t0) / 60,
                                                  "best_sha256": sha256(run / "best.pt")}, indent=2))
    print("done", flush=True)


def build_model_and_check(args, bank, fmin, dev):
    model, v, n_det = L.build_model(args.variant, bank.n_det, fmin, dev, Path(args.pretrained))
    return model, v, n_det


if __name__ == "__main__":
    main()
