#!/usr/bin/env python3
"""Compute cost of the v3 models, for the cost model.

PI-ResNet v3 scores a pair from two per-event feature vectors, so in a catalogue search each event is
encoded once and every pair costs only the head:  cost(N events) = N * c_event + N(N-1)/2 * c_head.
The early-fusion and CQT models must process every pair in full.  Reported per device: seconds per
event encoding, per pair head, per full pair forward (pi, early, cqt), with the batch sizes used.
Run on an idle machine for numbers that go into the cost model.
"""
import argparse, json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3lib as L  # noqa: E402


def timeit(fn, reps, sync):
    fn(); sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=None); ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--n-det", type=int, default=3); ap.add_argument("--fmin", type=float, default=5.0)
    ap.add_argument("--pretrained", default=str(L.ROOT / "artifacts/pretrained/deit_tiny_distilled_patch16_224-b40b3cf7.pth"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); gpu = dev.type == "cuda"
    torch.set_num_threads(a.threads)
    bs = a.batch or (512 if gpu else 16); reps = a.reps or (20 if gpu else 5)
    sync = (lambda: torch.cuda.synchronize(dev)) if gpu else (lambda: None)
    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if gpu else (lambda: torch.autocast("cpu", enabled=False))
    res = {"device": str(dev), "threads": a.threads, "batch": bs,
           "gpu_name": torch.cuda.get_device_name(dev) if gpu else None}
    A = torch.randn(bs, a.n_det, L.WIN_A, device=dev); B = torch.randn(bs, a.n_det, L.WIN_B, device=dev)
    with torch.no_grad():
        for variant in ("pi", "early", "cqt"):
            model, v, n_det = L.build_model(variant, a.n_det, a.fmin, dev, Path(a.pretrained))
            model.eval()
            if variant == "pi":
                with ctx():
                    f = model.features(A, B)
                    pair = torch.cat([f * f, torch.abs(f - f)], 1)
                    res["pi_s_per_event_encoding"] = timeit(lambda: model.features(A, B), reps, sync) / bs
                    big = pair.repeat(16, 1)
                    res["pi_s_per_pair_head"] = timeit(lambda: model.head(big), reps, sync) / big.shape[0]
                    res["pi_s_per_pair_full"] = timeit(lambda: model(A, B, A, B), reps, sync) / bs
            else:
                with ctx():
                    res[f"{variant}_s_per_pair_full"] = timeit(lambda: model(A, B, A, B), reps, sync) / bs
            res[f"{variant}_parameters"] = sum(p.numel() for p in model.parameters())
            print(variant, {k: v for k, v in res.items() if k.startswith(variant)}, flush=True)
    for n in (200, 1000, 100000):
        pairs = n * (n - 1) / 2
        res[f"pi_catalogue_s_N{n}"] = n * res["pi_s_per_event_encoding"] + pairs * res["pi_s_per_pair_head"]
        res[f"pi_catalogue_s_per_pair_N{n}"] = res[f"pi_catalogue_s_N{n}"] / pairs
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
