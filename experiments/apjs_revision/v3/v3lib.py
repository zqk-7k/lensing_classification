"""Shared pieces of the v3 training and inference: GPU noise, input windows, models, GPU constant-Q images.

Model variants (all trained with the identical recipe in train_v3.py):
  pi        PI-ResNet v3: Siamese encoders on all detectors, stream A (2 s, 2048 Hz) and stream B
            (64 s, 128 Hz, < 64 Hz), pairwise interaction [f1*f2, |f1-f2|], the original head.
  pi3A      as pi, stream A only (ablation: what the 64 s low-frequency stream adds).
  v2info    as pi, loudest detector only, stream A only, band above 20 Hz: the information content
            of the v2 / submitted setting, trained with the v3 recipe (ablation).
  early     early-fusion time-domain CNN (Offermans-style class: the two events' channels are stacked
            and processed by one encoder), same encoders and head size, no pairwise interaction.
  cqt       CQT-DeiT v3: constant-Q images of both streams of both events, detectors as the three
            image channels, DeiT-tiny (ImageNet weights) with the original two-head classifier.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "classifier"))
sys.path.insert(0, str(ROOT / "src"))
from pi_resnet import PIResNet  # noqa: E402

FS, FRAME_S = 2048, 128
N = FS * FRAME_S
FS_B = 128
N_B = FS_B * FRAME_S
LEN_A, LEN_B, WIN_A, WIN_B = 4608, 8224, 4096, 8192
EVAL_OFF_A, EVAL_OFF_B = 256, 16
MAX_OFF_UNITS = 32  # training offsets o in [0, 32]: A shifted by 16*o samples, B by o samples (+/-0.125 s)


def load_meta(root: Path) -> dict:
    meta = json.loads((Path(root) / "generation_meta_v3.json").read_text())
    for k, v in dict(stored_A=LEN_A, stored_B=LEN_B, window_A=WIN_A, window_B=WIN_B, frame_s=FRAME_S,
                     fs_A_hz=FS, fs_B_hz=FS_B, eval_offset_A=EVAL_OFF_A, eval_offset_B=EVAL_OFF_B).items():
        assert meta[k] == v, (k, meta[k], v)
    return meta


class Noise:
    """Whitened stationary Gaussian noise in the v3 convention, generated on the GPU."""

    def __init__(self, fmin: float, device):
        f = torch.arange(N // 2 + 1, device=device, dtype=torch.float64) / FRAME_S
        self.band = ((f >= fmin) & (f < FS / 2)).float() * math.sqrt(N / 2.0)
        self.device = device

    def __call__(self, rows: int, gen: torch.Generator | None = None):
        re = torch.randn(rows, N // 2 + 1, device=self.device, generator=gen)
        im = torch.randn(rows, N // 2 + 1, device=self.device, generator=gen)
        X = torch.complex(re * self.band, im * self.band)
        del re, im
        a = torch.fft.irfft(X, n=N)[:, -LEN_A:][:, EVAL_OFF_A:EVAL_OFF_A + WIN_A]
        b = torch.fft.irfft(X[:, : N_B // 2 + 1] * math.sqrt(N_B / N), n=N_B)[:, -LEN_B:][:, EVAL_OFF_B:EVAL_OFF_B + WIN_B]
        return a.contiguous(), b.contiguous()


def window(x: torch.Tensor, off: torch.Tensor, length: int):
    """x (n, C, L), off (n,) start indices -> (n, C, length)."""
    idx = off.view(-1, 1, 1) + torch.arange(length, device=x.device).view(1, 1, -1)
    return torch.gather(x, 2, idx.expand(x.shape[0], x.shape[1], length))


def highpass(x: torch.Tensor, f_lo: float, fs: float):
    X = torch.fft.rfft(x, dim=-1)
    f = torch.fft.rfftfreq(x.shape[-1], 1.0 / fs).to(x.device)
    return torch.fft.irfft(X * (f >= f_lo), n=x.shape[-1], dim=-1)


def standardize(x: torch.Tensor):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-8)


def best_channel(sig_a: torch.Tensor, sig_b: torch.Tensor):
    """Index of the detector with the largest whitened signal energy (optimal S/N) per event."""
    return sig_a.float().pow(2).sum(-1).argmax(1)  # stream A carries the band above 20 Hz that v2 selected on


class Encoder(nn.Module):
    def __init__(self, in_ch: int, width: float):
        super().__init__()
        p = PIResNet(in_channels=in_ch, d_model=256, width_scale=width, use_snake=False, use_se=True, use_pairwise_fusion=True)
        self.stem, self.stage1, self.stage2, self.stage3 = p.stem, p.stage1, p.stage2, p.stage3
        self.avg_pool, self.max_pool = p.avg_pool, p.max_pool
        self.out_dim = int(32 * width) * 8 * 2

    def forward(self, x):
        h = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return torch.cat([self.avg_pool(h).squeeze(-1), self.max_pool(h).squeeze(-1)], 1)


VARIANTS = {
    "pi": dict(kind="pi", n_det="all", use_B=True, f_lo=None),
    "pi3A": dict(kind="pi", n_det="all", use_B=False, f_lo=None),
    "v2info": dict(kind="pi", n_det=1, use_B=False, f_lo=20.0),
    "early": dict(kind="early", n_det="all", use_B=True, f_lo=None),
    "cqt": dict(kind="cqt", n_det="all", use_B=True, f_lo=None),
}


class PairNet(nn.Module):
    def __init__(self, kind: str, n_det: int, use_B: bool, width_A: float = 4.0, width_B: float = 2.0, d_model: int = 256):
        super().__init__()
        self.kind, self.use_B = kind, use_B
        c = n_det if kind == "pi" else 2 * n_det
        self.encA = Encoder(c, width_A)
        self.encB = Encoder(c, width_B) if use_B else None
        d = self.encA.out_dim + (self.encB.out_dim if use_B else 0)
        self.head = nn.Sequential(nn.Linear(2 * d if kind == "pi" else d, d_model), nn.ReLU(), nn.Dropout(0.7), nn.Linear(d_model, 1))

    def features(self, a, b):
        f = self.encA(a)
        return torch.cat([f, self.encB(b)], 1) if self.use_B else f

    def forward(self, a1, b1, a2, b2):
        if self.kind == "pi":
            f1, f2 = self.features(a1, b1), self.features(a2, b2)
            return self.head(torch.cat([f1 * f2, torch.abs(f1 - f2)], 1)).squeeze(1)
        return self.head(self.features(torch.cat([a1, a2], 1), torch.cat([b1, b2], 1) if self.use_B else None)).squeeze(1)


class ConstantQ(nn.Module):
    """Constant-Q Gabor filterbank magnitudes (24 bins per octave, Q = 34.1) computed with FFTs on the GPU."""

    def __init__(self, n: int, fs: float, f_lo: float, f_hi: float, width_bins: int, device):
        super().__init__()
        q = 1.0 / (2 ** (1 / 24) - 1)
        nb = int(math.floor(24 * math.log2(f_hi / f_lo))) + 1
        fk = f_lo * 2 ** (np.arange(nb) / 24)
        df = fs / n
        sigma = fk / q / 2.3548
        centre = np.round(fk / df).astype(int)
        offs = np.arange(width_bins) - width_bins // 2
        idx = centre[:, None] + offs[None, :]
        g = np.exp(-0.5 * ((idx * df - fk[:, None]) / sigma[:, None]) ** 2)
        valid = (idx > 0) & (idx < n // 2)
        g[~valid] = 0.0
        idx = np.clip(idx, 0, n // 2)
        # baseband placement: bin offset j -> position j mod width_bins
        self.register_buffer("idx", torch.as_tensor(idx, device=device))
        self.register_buffer("g", torch.as_tensor(g, dtype=torch.float32, device=device))
        self.register_buffer("perm", torch.as_tensor(np.argsort(offs % width_bins), device=device))
        self.n_bins = nb

    def forward(self, x):  # x (rows, n) -> (rows, n_bins, width_bins) magnitudes
        X = torch.fft.rfft(x.float(), dim=-1)
        Y = X[:, self.idx] * self.g  # (rows, nb, W)
        Y = Y[:, :, self.perm]
        return torch.fft.ifft(Y, dim=-1).abs()


class CQTImages(nn.Module):
    """Pair image: [event 1 (A rows; B rows); event 2 (A rows; B rows)] in dB, detectors as RGB, 224 x 224."""

    def __init__(self, fmin_b: float, device):
        super().__init__()
        self.cq_a = ConstantQ(WIN_A, FS, 20.0, 497.0, 128, device)
        self.cq_b = ConstantQ(WIN_B, FS_B, max(fmin_b, 5.0), 60.0, 512, device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @staticmethod
    def db(m):
        d = 20 * torch.log10(m + 1e-8)
        d = d - d.amax(dim=(-1, -2), keepdim=True)
        return (d.clamp(min=-80.0) + 80.0) / 80.0

    def event(self, a, b):  # a (n, 3, WIN_A), b (n, 3, WIN_B) -> (n, 3, rows, 224)
        n, c = a.shape[:2]
        ia = self.db(self.cq_a(a.reshape(n * c, -1)))
        ib = self.db(self.cq_b(b.reshape(n * c, -1)))
        ia = F.interpolate(ia.unsqueeze(1), size=(ia.shape[1], 224), mode="bilinear", align_corners=False).squeeze(1)
        ib = F.interpolate(ib.unsqueeze(1), size=(ib.shape[1], 224), mode="bilinear", align_corners=False).squeeze(1)
        return torch.cat([ia, ib], 1).view(n, c, -1, 224)

    def forward(self, a1, b1, a2, b2):
        img = torch.cat([self.event(a1, b1), self.event(a2, b2)], 2)
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        return (img - self.mean) / self.std


class CQTNet(nn.Module):
    def __init__(self, fmin_b: float, pretrained: Path, device):
        super().__init__()
        from cqt_deit.model import build_cqt_deit
        self.images = CQTImages(fmin_b, device)
        self.deit = build_cqt_deit(num_classes=2, pretrained=False, hidden_dim=512, dropout_rate=0.5, freeze_backbone=False)
        state = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        bad = [k for k in self.deit.load_state_dict(state, strict=False).missing_keys if not k.startswith(("head.", "head_dist."))]
        if bad:
            raise RuntimeError(f"missing DeiT keys {bad}")

    def forward(self, a1, b1, a2, b2):
        with torch.no_grad():
            img = self.images(a1.float(), b1.float(), a2.float(), b2.float())
        out = self.deit(img)
        if isinstance(out, tuple):  # distilled model in train mode returns (cls, dist)
            out = (out[0] + out[1]) / 2
        return out  # (n, 2) logits


def build_model(variant: str, n_det_total: int, fmin_b: float, device, pretrained: Path | None = None):
    v = VARIANTS[variant]
    n_det = n_det_total if v["n_det"] == "all" else v["n_det"]
    if v["kind"] == "cqt":
        return CQTNet(fmin_b, pretrained, device).to(device), v, n_det
    return PairNet(v["kind"], n_det, v["use_B"]).to(device), v, n_det


def logit_of(model_out):
    return model_out[:, 1] - model_out[:, 0] if model_out.ndim == 2 else model_out


def prepare_inputs(v, n_det_total, sig_a, sig_b, data_a=None, data_b=None):
    """Select channels/band for a variant and standardize. sig_* are signals (used for the channel choice);
    data_* the noisy windows (training: signal + generated noise). Returns (a, b)."""
    a, b = data_a, data_b
    if v["n_det"] == 1:
        ch = best_channel(sig_a, sig_b)
        sel = ch.view(-1, 1, 1)
        a = torch.gather(a, 1, sel.expand(-1, 1, a.shape[-1]))
        b = torch.gather(b, 1, sel.expand(-1, 1, b.shape[-1]))
    if v["f_lo"]:
        a = highpass(a, v["f_lo"], FS)
    a = standardize(a)
    b = standardize(b) if v["use_B"] else b[:, :, :8]
    return a, b
