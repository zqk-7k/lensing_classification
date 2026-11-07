#!/usr/bin/env python3
"""Figures for the ApJS revision from the revision result tables (post hoc, descriptive).

  fig_rev_efficiency_vs_fap.{pdf,png}   efficiency against target per-pair FAP, 1e-1 .. 1e-4, both models,
                                         one panel per detector configuration
  fig_rev_selection.{pdf,png}            efficiency at FAP 1e-3 in bins of fainter-image S/N, flux ratio and
                                         source redshift, one row per configuration
  fig_rev_population.{pdf,png}           fainter-image single-detector S/N, flux ratio and time delay of the
                                         realistic catalogs (0228), per configuration

All values are read from CSV tables written by analyze_revision_extras.py and generate_strain_rev.py;
nothing is recomputed here. Column width 3.3 in, full width 6.9 in (ApJ two-column).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "pdf.fonttype": 42, "figure.dpi": 150})
COLORS = {"pi_resnet": "#1f5fa8", "cqt_deit": "#d1495b"}
NAMES = {"pi_resnet": "PI-ResNet", "cqt_deit": "CQT--DeiT"}
LABEL = {"rho_min": r"fainter-image single-detector S/N $\rho_{\min}$", "flux_ratio": r"flux ratio $|\mu_1/\mu_2|$",
         "zs": r"source redshift $z_s$", "y": r"impact parameter $y=\beta/\theta_E$", "delay_days": "time delay [d]"}


def load(rev, cfg, name):
    p = rev / f"work_{cfg}" / "results" / "revision" / name
    return pd.read_csv(p) if p.exists() else None


def fig_curve(rev, cfgs, out):
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.3 * len(cfgs), 2.6), sharey=True, squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        df = load(rev, cfg, "efficiency_vs_fap.csv")
        if df is None:
            ax.set_title(f"{cfg}: not available"); continue
        for model, c in COLORS.items():
            d = df[df.model == model].sort_values("target_fap")
            ax.plot(d.target_fap, d.efficiency, "o-", ms=3, lw=1.1, color=c, label=NAMES[model])
            ax.fill_between(d.target_fap, d.efficiency_ci_low, d.efficiency_ci_high, color=c, alpha=0.18, lw=0)
        ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel(r"target per-pair false-alarm probability $P_f$")
        ax.set_title(f"{cfg} sensitivity, realistic population"); ax.grid(alpha=0.25); ax.set_ylim(0, 1)
        ax.axvline(1e-3, color="gray", lw=0.8, ls=":")
    axes[0, 0].set_ylabel("efficiency on lensed pairs"); axes[0, 0].legend(frameon=True, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "fig_rev_efficiency_vs_fap.pdf"); fig.savefig(out / "fig_rev_efficiency_vs_fap.png", dpi=200); plt.close(fig)


def fig_selection(rev, cfgs, out, variables=("rho_min", "flux_ratio", "zs"), fap=1e-3):
    fig, axes = plt.subplots(len(cfgs), len(variables), figsize=(6.9, 2.3 * len(cfgs)), sharey=True, squeeze=False)
    for r, cfg in enumerate(cfgs):
        df = load(rev, cfg, "selection_functions_rev.csv")
        for c, var in enumerate(variables):
            ax = axes[r, c]
            if df is None:
                ax.set_title(f"{cfg}: not available"); continue
            for model, col in COLORS.items():
                d = df[(df.model == model) & (df.variable == var) & np.isclose(df.target_fap, fap)].sort_values("bin_left")
                x = d.median_in_bin.to_numpy()
                ax.errorbar(x, d.efficiency, yerr=[d.efficiency - d.ci_low, d.ci_high - d.efficiency], fmt="o-", ms=3, lw=1.1,
                            capsize=2, color=col, label=NAMES[model] if (r == 0 and c == len(variables) - 1) else None)
            if var in ("flux_ratio",):
                ax.set_xscale("log")
            if r == len(cfgs) - 1:
                ax.set_xlabel(LABEL[var])
            ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
        axes[r, 0].set_ylabel(f"{cfg}\nefficiency at $P_f=10^{{-3}}$")
    axes[0, -1].legend(frameon=True, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig_rev_selection.pdf"); fig.savefig(out / "fig_rev_selection.png", dpi=200); plt.close(fig)


def fig_population(rev, cfgs, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
    for cfg, c in zip(cfgs, ("#1f5fa8", "#e08e2b")):
        p = rev / f"data_{cfg}" / "SIS_data_0228" / "pair_meta.csv"
        if not p.exists():
            continue
        m = pd.read_csv(p)
        rho_min = m[["snr_channel_1", "snr_channel_2"]].min(axis=1)
        axes[0].hist(rho_min, bins=np.logspace(np.log10(2), np.log10(60), 30), histtype="step", lw=1.2, color=c, label=cfg)
        axes[1].hist((m.mu_1 / m.mu_2).abs(), bins=np.logspace(0, 1.5, 30), histtype="step", lw=1.2, color=c)
        axes[2].hist(m.delay_s.abs() / 86400, bins=np.logspace(-3, 2.6, 30), histtype="step", lw=1.2, color=c)
    axes[0].axvline(8, color="gray", ls=":", lw=0.8)
    for ax, lab in zip(axes, (LABEL["rho_min"], LABEL["flux_ratio"], LABEL["delay_days"])):
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.grid(alpha=0.25)
    axes[0].set_ylabel("lensed pairs (0228 catalog)"); axes[0].legend(frameon=True)
    fig.tight_layout(); fig.savefig(out / "fig_rev_population.pdf"); fig.savefig(out / "fig_rev_population.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="/root/autodl-tmp/apjs_rev")
    ap.add_argument("--configs", default="O4,ET")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rev = Path(args.rev); cfgs = [c for c in args.configs.split(",") if c]
    out = Path(args.out) if args.out else rev / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig_curve(rev, cfgs, out); fig_selection(rev, cfgs, out); fig_population(rev, cfgs, out)
    print("figures in", out)


if __name__ == "__main__":
    main()
