#!/usr/bin/env python3
"""Draw the two analytic lensing figures of the manuscript from their equations.

These were the only two manuscript figures without released generating code. They do
not depend on any trained network or on the simulated catalogs, so they cannot go stale
with the results, but the paper states that its figures are reproducible from released
code and these were the exception.

Both are computed directly from the geometrical-optics relations quoted in Section 2:

  point mass   mu_pm(y) = 1/2 +- (y^2 + 2) / (2 y sqrt(y^2 + 4))
               dt_d     = (4 G M_L^z / c^3) [ y sqrt(y^2+4)/2
                                              + ln( (sqrt(y^2+4)+y) / (sqrt(y^2+4)-y) ) ]
  SIS          mu_pm(y) = 1 +- 1/y
               dt_d     = 8 (G M_L^z / c^3) y

with M_L^z = M_L (1 + z_L), and for the SIS the redshifted mass inside the Einstein
radius M_L^z = (4 pi^2 sigma_v^4 / (G c^2)) (D_L D_LS / D_S) (1 + z_L).

Only the *shape* of dt_d(z_L) matters for the second figure, so it is plotted
normalized to the fiducial choice z_L = z_S / 2 used throughout the paper. The
cosmology is the flat LambdaCDM of Planck 2018 as implemented in Astropy, matching
Section 1.

Outputs: results/figures/manuscript/fig_lens_magnification.{pdf,png}
         results/figures/manuscript/fig_zl_dependence.{pdf,png}
         results/figures/manuscript/analytic_figure_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

COSMO = FlatLambdaCDM(H0=67.66, Om0=0.30966)   # Planck 2018, as in Section 1
Y_PRIOR = (0.01, 0.30)                          # simulated impact-parameter prior
C_SIS, C_PM = "#1f6fb2", "#c0392b"


def mu_pm(y):
    """Signed point-mass image magnifications."""
    root = np.sqrt(y ** 2 + 4.0)
    common = (y ** 2 + 2.0) / (2.0 * y * root)
    return 0.5 + common, 0.5 - common


def mu_sis(y):
    """SIS image magnifications, valid for y < 1."""
    return 1.0 + 1.0 / y, 1.0 - 1.0 / y


def figure_magnification():
    y = np.linspace(Y_PRIOR[0] * 0.5, 0.5, 2000)
    sis_p, sis_m = mu_sis(y)
    pm_p, pm_m = mu_pm(y)
    inside = (y >= Y_PRIOR[0]) & (y <= Y_PRIOR[1])

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    ax = axes[0]
    ax.plot(y, np.abs(sis_p), color=C_SIS, lw=1.9, label=r"SIS $|\mu_+|$")
    ax.plot(y, np.abs(sis_m), color=C_SIS, lw=1.9, ls="--", label=r"SIS $|\mu_-|$")
    ax.plot(y, np.abs(pm_p), color=C_PM, lw=1.9, label=r"PM $|\mu_+|$")
    ax.plot(y, np.abs(pm_m), color=C_PM, lw=1.9, ls="--", label=r"PM $|\mu_-|$")
    ax.axvspan(*Y_PRIOR, color="0.85", zorder=0)
    ax.set_yscale("log"); ax.set_xlim(0, 0.5)
    ax.set_xlabel(r"Impact parameter $y$"); ax.set_ylabel(r"$|\mu_\pm|$")
    ax.set_title("(a)  Image magnifications", fontsize=10)
    ax.legend(fontsize=7.6, frameon=False, ncol=2); ax.grid(alpha=0.25)

    ax = axes[1]
    ratio_p = np.sqrt(np.abs(sis_p) / np.abs(pm_p))
    ratio_m = np.sqrt(np.abs(sis_m) / np.abs(pm_m))
    ax.plot(y, ratio_p, color=C_SIS, lw=1.9, label="type-I image")
    ax.plot(y, ratio_m, color=C_PM, lw=1.9, ls="--", label="type-II image")
    ax.axhline(np.sqrt(2.0), color="0.4", lw=1.2, ls=":", label=r"$\sqrt{2}$")
    ax.axvspan(*Y_PRIOR, color="0.85", zorder=0)
    ax.set_xlim(0, 0.5); ax.set_ylim(1.30, 1.50)
    ax.set_xlabel(r"Impact parameter $y$")
    ax.set_ylabel(r"$\sqrt{|\mu^{\rm SIS}|/|\mu^{\rm PM}|}$")
    ax.set_title("(b)  Per-image amplitude ratio", fontsize=10)
    ax.legend(fontsize=7.6, frameon=False); ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(y, np.abs(sis_p / sis_m), color=C_SIS, lw=1.9, label="SIS")
    ax.plot(y, np.abs(pm_p / pm_m), color=C_PM, lw=1.9, ls="--", label="PM")
    ax.axvspan(*Y_PRIOR, color="0.85", zorder=0)
    ax.set_yscale("log"); ax.set_xlim(0, 0.5)
    ax.set_xlabel(r"Impact parameter $y$"); ax.set_ylabel(r"$|\mu_+/\mu_-|$")
    ax.set_title("(c)  Pair flux ratio", fontsize=10)
    ax.legend(fontsize=7.6, frameon=False); ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "fig_lens_magnification.pdf")
    fig.savefig(OUT / "fig_lens_magnification.png", dpi=300)
    plt.close(fig)

    prior = (y >= Y_PRIOR[0]) & (y <= Y_PRIOR[1])
    return {
        "median_amplitude_ratio_type_I": float(np.median(ratio_p[prior])),
        "median_amplitude_ratio_type_II": float(np.median(ratio_m[prior])),
        "max_relative_deviation_from_sqrt2": float(
            np.max(np.abs(np.concatenate([ratio_p[prior], ratio_m[prior]]) / np.sqrt(2.0) - 1.0))),
        "median_magnification_ratio_sis_over_pm": float(
            np.median(np.abs(sis_p[prior]) / np.abs(pm_p[prior]))),
    }


def delay_shape(z_lens, z_source, profile):
    """dt_d as a function of z_L at fixed y, up to constants that cancel on ratio.

    SIS:  dt_d ~ (1 + z_L) * M_L^z-independent prefactor * D_L D_LS / D_S
          since M_L^z itself carries (1+z_L) D_L D_LS / D_S at fixed sigma_v.
    PM:   dt_d ~ (1 + z_L) at fixed M_L.
    """
    if profile == "PM":
        return 1.0 + z_lens
    d_l = COSMO.angular_diameter_distance(z_lens).to_value(u.Mpc)
    d_s = COSMO.angular_diameter_distance(z_source).to_value(u.Mpc)
    d_ls = COSMO.angular_diameter_distance_z1z2(z_lens, z_source).to_value(u.Mpc)
    return (1.0 + z_lens) * d_l * d_ls / d_s


def figure_zl_dependence():
    fractions = np.linspace(0.05, 0.95, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=False)
    summary = {}
    for ax, profile, title in zip(axes, ("SIS", "PM"),
                                  (r"(a)  SIS, fixed $(y,\sigma_v)$", r"(b)  PM, fixed $(y,M_L)$")):
        peaks = []
        for z_s, colour in zip((0.3, 0.6, 1.0), ("#7fb3d5", "#1f6fb2", "#154360")):
            z_l = fractions * z_s
            shape = np.asarray([delay_shape(z, z_s, profile) for z in z_l])
            shape = shape / delay_shape(z_s / 2.0, z_s, profile)
            ax.plot(fractions, shape, color=colour, lw=1.9, label=rf"$z_S={z_s}$")
            peaks.append(float(fractions[int(np.argmax(shape))]))
            summary[f"{profile}_zs{z_s}_range"] = [float(shape.min()), float(shape.max())]
        summary[f"{profile}_peak_fraction"] = peaks
        ax.axvline(0.5, color="0.4", lw=1.2, ls=":")
        ax.set_xlabel(r"$z_L / z_S$")
        ax.set_ylabel(r"$\Delta t_d(z_L) \,/\, \Delta t_d(z_L=z_S/2)$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.8, frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_zl_dependence.pdf")
    fig.savefig(OUT / "fig_zl_dependence.png", dpi=300)
    plt.close(fig)
    return summary


def sis_delay_days(z_source, sigma_v_km_s, y, z_lens=None):
    """Absolute SIS inter-image delay, for the reference value quoted in the caption.

    dt_d = 8 (G M_L^z / c^3) y  with  M_L^z = (4 pi^2 sigma_v^4 / (G c^2)) (D_L D_LS / D_S)(1+z_L),
    so  dt_d = 8 y (1 + z_L) (4 pi^2 sigma_v^4 / c^5) (D_L D_LS / D_S).
    """
    from astropy.constants import c as c_light
    z_lens = z_source / 2.0 if z_lens is None else z_lens
    sigma = (sigma_v_km_s * u.km / u.s).to(u.m / u.s)
    d_l = COSMO.angular_diameter_distance(z_lens)
    d_s = COSMO.angular_diameter_distance(z_source)
    d_ls = COSMO.angular_diameter_distance_z1z2(z_lens, z_source)
    delay = (8.0 * y * (1.0 + z_lens) * 4.0 * np.pi ** 2 * sigma ** 4 / c_light ** 5
             * (d_l * d_ls / d_s).to(u.m))
    return float(delay.to_value(u.day))


def main():
    reference = sis_delay_days(z_source=0.6, sigma_v_km_s=200.0, y=0.15)
    meta = {
        "sis_reference_delay_days": {"z_S": 0.6, "sigma_v_km_s": 200.0, "y": 0.15,
                                     "z_L": 0.3, "delay_days": reference},
        "cosmology": {"name": "FlatLambdaCDM", "H0_km_s_Mpc": 67.66, "Om0": 0.30966},
        "impact_parameter_prior": list(Y_PRIOR),
        "magnification": figure_magnification(),
        "delay_shape": figure_zl_dependence(),
        "note": "Purely analytic figures; no trained network or simulated catalog is used.",
    }
    (OUT / "analytic_figure_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    m = meta["magnification"]
    print(f"amplitude ratio over the prior: type-I median {m['median_amplitude_ratio_type_I']:.3f}, "
          f"type-II {m['median_amplitude_ratio_type_II']:.3f}, "
          f"max deviation from sqrt(2) {100 * m['max_relative_deviation_from_sqrt2']:.1f}%")
    print(f"SIS delay peaks at z_L/z_S = {meta['delay_shape']['SIS_peak_fraction']}")
    print(f"SIS reference delay (z_S=0.6, sigma_v=200 km/s, y=0.15): {reference:.2f} days")
    print(f"wrote {(OUT / 'fig_lens_magnification.pdf').relative_to(ROOT)} and "
          f"{(OUT / 'fig_zl_dependence.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
