#!/usr/bin/env python3
"""Figures and machine-readable tables of the v3 rerun for the revised manuscript.

Reads only the registered v3 products on the analysis server (frozen-analysis inputs: the 0228 pair
manifests and the per-pair predictions; the evaluation-catalog metadata), never a training file.

Outputs (in --out):
  fig_v3_population.pdf     evaluation-catalog distributions, ET and O4
  fig_v3_example.pdf        one ET lensed pair, whitened data and signal, streams A and B
  fig_v3_efficiency.pdf     efficiency versus per-pair false-positive probability, ET and O4
  fig_v3_selection.pdf      lensing selection function at P_f = 1e-3 (fainter-image S/N, flux ratio)
  v3_efficiency_vs_fpp.csv  the curves of fig_v3_efficiency
  v3_selection_functions.csv  the binned values of fig_v3_selection (Wilson 95% intervals)
  v3_selection_rho1_rho2.csv  5x5 efficiency map over the two images' network S/N, P_f = 1e-3
  v3_operating_points.csv   efficiency at 1e-2, 1e-3, 1e-4 recomputed here, checked against collected*.json

Threshold rule: the frozen one (thresholds.kth_threshold, k = round(P_f N) largest calibration-background
score, a pair is kept when score >= threshold), on the same `score` column the frozen analysis uses.
Bins: quintiles of the calibration positives with open outer edges (open_ended_quantile_edges of
analyze_0228_core.py); efficiencies on the evaluation positives.

Run on the server:
  python make_v3_figures.py --base /root/autodl-tmp/apjs_rev --out figures_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
    "savefig.dpi": 300, "font.family": "serif", "mathtext.fontset": "dejavuserif",
})
COL, TXT = 3.39, 7.06          # AASTeX twocolumn column and text widths, inches

NETS = [  # variant, label, colour
    ("pi", "PI-ResNet", "#1f5fa8"),
    ("early", "Early fusion", "#d9822b"),
    ("cqt", "CQT--DeiT, same input", "#3a8f5c"),
    ("v2info", "PI-ResNet, submitted information", "#8a6bb0"),
]
CFG = {  # work dir, data dir, time-delay suffix used in the manuscript tables, label
    "ET": ("work_v3_ET2", "data_v3_ET2", "td", "ET (3 $\\times$ ET-D, from 5 Hz)"),
    "O4": ("work_v3_O4", "data_v3_O4", "tdc", "O4 (H1, L1, V1, from 20 Hz)"),
}
TARGETS = (1e-2, 1e-3, 1e-4)
PO2 = dict(fpp=2e-6, eff=0.65)   # Barsode et al. (2024), published, O4, its own pairs


def kth_threshold(scores, target):
    scores = np.asarray(scores)
    k = max(1, int(round(target * len(scores))))
    return float(np.partition(scores, -k)[-k])


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def edges(values, count=5):
    inner = np.quantile(values, np.linspace(0, 1, count + 1)[1:-1])
    return np.unique(np.r_[-np.inf, inner, np.inf])


def load(base: Path, cfg: str):
    work, data, td, _ = CFG[cfg]
    w = base / work
    man = pd.concat([pd.read_csv(w / "manifests" / f"0228_sis_{p}_pairs.csv.gz")
                     for p in ("calibration", "evaluation")], ignore_index=True)
    man = man.set_index("pair_id")
    for v, _, _ in NETS:
        for suf in ("", "_" + td):
            p = pd.read_csv(w / "predictions" / f"{v}{suf}_predictions_0228_sis.csv.gz",
                            usecols=["pair_id", "score"]).set_index("pair_id")
            assert len(p) == len(man) and p.index.isin(man.index).all(), (cfg, v, suf)
            man[f"{v}{suf}"] = p.score
    meta = pd.read_csv(base / data / "SIS_data_0228" / "pair_meta.csv")
    return man.reset_index(), meta, td


def operating_points(cfg, df, td):
    cal, ev = df[df.calibration_or_evaluation == "calibration"], df[df.calibration_or_evaluation == "evaluation"]
    rows = []
    for v, label, _ in NETS:
        for suf in ("", "_" + td):
            col = f"{v}{suf}"
            bg_c, pos_e, bg_e = cal[cal.label == 0][col], ev[ev.label == 1][col], ev[ev.label == 0][col]
            for t in TARGETS:
                thr = kth_threshold(bg_c, t)
                rows.append(dict(config=cfg, model=col, target_fpp=t, threshold=thr,
                                 efficiency=float((pos_e >= thr).mean()),
                                 achieved_fpp=float((bg_e >= thr).mean())))
    return pd.DataFrame(rows)


def check_against_collected(base, cfg, op, td):
    work = base / CFG[cfg][0]
    got = {}
    for name in ("collected.json", "collected_tdc.json"):
        p = work / name
        if p.exists():
            got.update(json.loads(p.read_text())["models"])
    worst = 0.0
    for r in op.itertuples():
        m = got.get(r.model)
        if m is None or str(r.target_fpp) not in m:
            continue
        ref = m[str(r.target_fpp)]["efficiency"]
        worst = max(worst, abs(ref - r.efficiency))
        assert abs(ref - r.efficiency) < 1e-9, (cfg, r.model, r.target_fpp, ref, r.efficiency)
    print(f"{cfg}: operating points reproduce collected*.json (max |diff| {worst:.1e})")


def curves(cfg, df, td, grid):
    cal, ev = df[df.calibration_or_evaluation == "calibration"], df[df.calibration_or_evaluation == "evaluation"]
    rows = []
    for v, label, _ in NETS:
        for suf in ("", "_" + td):
            col = f"{v}{suf}"
            bg_c, pos_e, bg_e = np.sort(cal[cal.label == 0][col].values), ev[ev.label == 1][col].values, ev[ev.label == 0][col].values
            for t in grid:
                thr = kth_threshold(bg_c, t)
                rows.append(dict(config=cfg, model=col, time_delay=bool(suf), target_fpp=t, threshold=thr,
                                 efficiency=float((pos_e >= thr).mean()), achieved_fpp=float((bg_e >= thr).mean())))
    return pd.DataFrame(rows)


def selection(cfg, df, td):
    cal, ev = df[df.calibration_or_evaluation == "calibration"], df[df.calibration_or_evaluation == "evaluation"]
    cal_pos, ev_pos = cal[cal.label == 1], ev[ev.label == 1]
    rows, grid = [], []
    for v, label, _ in NETS:
        for suf in ("", "_" + td):
            col = f"{v}{suf}"
            thr = kth_threshold(cal[cal.label == 0][col], 1e-3)
            for var in ("rho_min", "flux_ratio"):
                e = edges(cal_pos[var])
                b = pd.cut(ev_pos[var], e, include_lowest=True)
                for cat in b.cat.categories:
                    m = (b == cat).values
                    n, k = int(m.sum()), int((ev_pos[col].values[m] >= thr).sum())
                    lo, hi = wilson(k, n)
                    rows.append(dict(config=cfg, model=col, time_delay=bool(suf), variable=var,
                                     bin_left=float(cat.left), bin_right=float(cat.right),
                                     median_in_bin=float(np.median(ev_pos[var].values[m])),
                                     n=n, detected=k, efficiency=k / n, ci_low=lo, ci_high=hi,
                                     target_fpp=1e-3, threshold=thr))
            e1, e2 = edges(cal_pos.rho_1), edges(cal_pos.rho_2)
            b1, b2 = pd.cut(ev_pos.rho_1, e1, include_lowest=True), pd.cut(ev_pos.rho_2, e2, include_lowest=True)
            for i in b1.cat.categories:
                for j in b2.cat.categories:
                    m = ((b1 == i) & (b2 == j)).values
                    n, k = int(m.sum()), int((ev_pos[col].values[m] >= thr).sum())
                    lo, hi = wilson(k, n)
                    grid.append(dict(config=cfg, model=col, time_delay=bool(suf),
                                     rho1_left=float(i.left), rho1_right=float(i.right),
                                     rho2_left=float(j.left), rho2_right=float(j.right), n=n, detected=k,
                                     efficiency=k / n if n else np.nan, ci_low=lo, ci_high=hi, target_fpp=1e-3))
    t = pd.DataFrame(rows)
    for (m, var), g in t.groupby(["model", "variable"]):
        assert g.n.sum() == len(ev_pos), (cfg, m, var)
    return t, pd.DataFrame(grid)


# ------------------------------------------------------------------ figures
def fig_efficiency(cur, op, out):
    fig, axes = plt.subplots(1, 2, figsize=(TXT, 2.55), sharey=True)
    for ax, cfg in zip(axes, ("ET", "O4")):
        td = CFG[cfg][2]
        c = cur[cur.config == cfg]
        for v, label, colour in NETS:
            for suf, ls, lw in (("", "-", 1.3), ("_" + td, "--", 1.1)):
                g = c[c.model == v + suf]
                ax.plot(g.target_fpp, 100 * g.efficiency, ls=ls, lw=lw, color=colour,
                        label=label if not suf else None)
            o = op[(op.config == cfg) & (op.model == v)]
            ax.plot(o.target_fpp, 100 * o.efficiency, "o", ms=2.6, color=colour, zorder=5)
        if cfg == "O4":
            ax.plot(PO2["fpp"], 100 * PO2["eff"], marker="*", ms=8, color="k", ls="none", zorder=6)
            ax.annotate("PO2.0, published\n(O4, its own pairs)", (PO2["fpp"], 100 * PO2["eff"]),
                        xytext=(3.2e-6, 80), fontsize=6.3, ha="left",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
        ax.axvspan(1e-6, 1e-4, color="0.93", zorder=0, lw=0)
        ax.text(1.15e-6, 3, "below the $10^{5}$-pair\nbackground", fontsize=6, color="0.4", va="bottom")
        for t in TARGETS:
            ax.axvline(t, color="0.85", lw=0.5, zorder=0)
        ax.set_xscale("log"); ax.set_xlim(1e-6, 1e-1); ax.set_ylim(0, 100)
        ax.set_xlabel("per-pair false-positive probability $P_f$")
        ax.set_title(CFG[cfg][3], fontsize=7.5)
        ax.grid(False)
    axes[0].set_ylabel("lensed pairs recovered [%]")
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="0.3", ls="-", lw=1.2), plt.Line2D([], [], color="0.3", ls="--", lw=1.1)]
    l += ["network alone", "with the time-delay statistic"]
    axes[0].legend(h, l, loc="upper left", frameon=False, fontsize=6.2, handlelength=2.2)
    fig.tight_layout(w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_v3_efficiency.{ext}")
    plt.close(fig)


def fig_selection(sel, out):
    fig, axes = plt.subplots(2, 2, figsize=(TXT, 4.3), sharey=True)
    xl = {"rho_min": "fainter image, network optimal S/N", "flux_ratio": "flux ratio of the two images"}
    for j, cfg in enumerate(("ET", "O4")):
        for i, var in enumerate(("rho_min", "flux_ratio")):
            ax = axes[i, j]
            s = sel[(sel.config == cfg) & (sel.variable == var)]
            for k, (v, label, colour) in enumerate(NETS):
                g = s[s.model == v].sort_values("bin_left")
                x = g.median_in_bin.values * (1 + 0.025 * (k - 1.5))
                ax.errorbar(x, 100 * g.efficiency, yerr=[100 * np.clip(g.efficiency - g.ci_low, 0, None), 100 * np.clip(g.ci_high - g.efficiency, 0, None)],
                            fmt="o-", ms=3, lw=1.1, elinewidth=0.7, capsize=1.5, color=colour, label=label)
            g = s[s.model == "pi_" + CFG[cfg][2]].sort_values("bin_left")
            ax.plot(g.median_in_bin, 100 * g.efficiency, "--", lw=1.0, color=NETS[0][2],
                    label="PI-ResNet with the time-delay statistic")
            ax.set_xscale("log"); ax.set_ylim(-2, 102)
            ax.set_xlabel(xl[var] + " (bin median)")
            if j == 0:
                ax.set_ylabel("recovered at $P_f=10^{-3}$ [%]")
            if i == 0:
                ax.set_title(CFG[cfg][3], fontsize=7.5)
            ticks = {("ET", "rho_min"): [10, 15, 20, 30, 40], ("O4", "rho_min"): [7, 8, 10, 12, 15, 20],
                     ("ET", "flux_ratio"): [1, 1.5, 2, 3, 5, 8], ("O4", "flux_ratio"): [1, 1.2, 1.5, 2, 2.5]}[(cfg, var)]
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1, 0].legend(loc="upper right", frameon=False, fontsize=6.0)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_v3_selection.{ext}")
    plt.close(fig)


def positives_with_meta(df, meta):
    pos = df[df.label == 1].copy()
    m = meta.set_index("catalog_index")
    pos = pos.join(m[["zs", "delay_s", "n_images", "type_1", "type_2"]], on="left_source_id")
    return pos


def fig_population(data, out):
    fig, axes = plt.subplots(1, 4, figsize=(TXT, 1.85))
    colours = {"ET": "#1f3f75", "O4": "#c4541b"}
    specs = [
        ("rho_min", "fainter image, network S/N", np.logspace(np.log10(4), np.log10(400), 36), True),
        ("flux_ratio", "flux ratio", np.logspace(0, 2, 30), True),
        ("abs_delay_d", "$|\\Delta t|$ [days]", np.logspace(-3, 3.2, 36), True),
        ("zs", "source redshift $z_s$", np.linspace(0, 10, 41), False),
    ]
    for cfg, (df, meta, unl) in data.items():
        pos = positives_with_meta(df, meta)
        pos["abs_delay_d"] = np.abs(pos.delay_s) / 86400
        for ax, (col, lab, bins, logx) in zip(axes, specs):
            # fraction of the catalogue per bin (log-spaced bins: per unit log x)
            x = pos[col].dropna().values
            ax.hist(x, bins=bins, histtype="step", lw=1.2, color=colours[cfg],
                    weights=np.full(len(x), 1 / len(x)), label=f"{cfg} lensed")
            if col == "rho_min" and unl is not None:
                ax.hist(unl, bins=bins, histtype="step", lw=0.9, ls=":", color=colours[cfg],
                        weights=np.full(len(unl), 1 / len(unl)), label=f"{cfg} unlensed")
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(lab)
            ax.set_yticks([])
    axes[0].axvline(8, color="0.5", lw=0.6, ls="--")
    axes[0].legend(frameon=False, fontsize=5.8, loc="upper right")
    axes[0].set_ylabel("fraction per bin")
    fig.tight_layout(w_pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_v3_population.{ext}")
    plt.close(fig)


def fig_example(base, df, out):
    d = base / CFG["ET"][1] / "SIS_data_0228"
    meta = pd.read_csv(d / "pair_meta.csv")
    pos = df[(df.label == 1) & (df.calibration_or_evaluation == "evaluation")]
    # a typical pair: fainter image near the median network S/N, a type-I and a type-II image
    m = meta.set_index("catalog_index").loc[pos.left_source_id]
    cand = pos.assign(t1=m.type_1.values, t2=m.type_2.values)
    cand = cand[(cand.t1 != cand.t2) & (cand.rho_min.between(18, 22)) & (cand.flux_ratio.between(1.5, 3))]
    sid = int(cand.sort_values("left_source_id").left_source_id.iloc[0])
    row = meta.set_index("catalog_index").loc[sid]
    det = int(str(row.best_det_1)[-1]) - 1 if str(row.best_det_1).startswith("ET") else 0
    EVAL_OFF_A, EVAL_OFF_B, WIN_A, WIN_B, FS_A, FS_B = 256, 16, 4096, 8192, 2048, 128
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    for i in (1, 2):
        ya = np.load(d / f"data_A_{i}.npy", mmap_mode="r")[sid, det, EVAL_OFF_A:EVAL_OFF_A + WIN_A].astype(float)
        sa = np.load(d / f"sig_A_{i}.npy", mmap_mode="r")[sid, det, EVAL_OFF_A:EVAL_OFF_A + WIN_A].astype(float)
        yb = np.load(d / f"data_B_{i}.npy", mmap_mode="r")[sid, det, EVAL_OFF_B:EVAL_OFF_B + WIN_B].astype(float)
        sb = np.load(d / f"sig_B_{i}.npy", mmap_mode="r")[sid, det, EVAL_OFF_B:EVAL_OFF_B + WIN_B].astype(float)
        ka, kb = int(np.argmax(np.abs(sa))), int(np.argmax(np.abs(sb)))
        ta, tb = (np.arange(WIN_A) - ka) / FS_A, (np.arange(WIN_B) - kb) / FS_B
        snr = row[f"snr_net_{i}"]
        typ = int(row[f"type_{i}"])
        for ax, t, y, s in ((axes[i - 1, 0], ta, ya, sa), (axes[i - 1, 1], tb, yb, sb)):
            ax.plot(t, y, color="0.72", lw=0.35)
            ax.plot(t, s, color="#1f5fa8" if i == 1 else "#c4541b", lw=0.7)
            ax.set_yticks([])
            ax.tick_params(labelsize=6.2)
        axes[i - 1, 0].text(0.02, 0.95, f"image {i}, type {'I' * typ}, network S/N {snr:.0f}",
                            transform=axes[i - 1, 0].transAxes, fontsize=5.9, va="top")
    axes[0, 0].set_title("stream A: 2 s, full band", fontsize=6.8)
    axes[0, 1].set_title("stream B: 64 s, below 64 Hz", fontsize=6.8)
    for ax in axes[1]:
        ax.set_xlabel("time from merger [s]", fontsize=6.6)
    for ax in axes[0]:
        ax.set_xticklabels([])
    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_v3_example.{ext}")
    plt.close(fig)
    return dict(source=sid, detector=f"ET{det + 1}", delay_days=float(row.delay_s) / 86400,
                types=[int(row.type_1), int(row.type_2)], snr_net=[float(row.snr_net_1), float(row.snr_net_2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("/root/autodl-tmp/apjs_rev"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    grid = np.unique(np.r_[np.logspace(-4, -1, 31), TARGETS])
    ops, curs, sels, grids, data = [], [], [], [], {}
    for cfg in ("ET", "O4"):
        df, meta, td = load(a.base, cfg)
        op = operating_points(cfg, df, td)
        check_against_collected(a.base, cfg, op, td)
        ops.append(op); curs.append(curves(cfg, df, td, grid))
        s, g = selection(cfg, df, td); sels.append(s); grids.append(g)
        unl = None
        try:
            unl = np.load(a.base / CFG[cfg][1] / "Unlensed_data_0228" / "unlensed_optimal_SNR.npy")
        except FileNotFoundError:
            pass
        data[cfg] = (df, meta, unl)
    op, cur, sel, grd = (pd.concat(x, ignore_index=True) for x in (ops, curs, sels, grids))
    op.to_csv(a.out / "v3_operating_points.csv", index=False)
    cur.to_csv(a.out / "v3_efficiency_vs_fpp.csv", index=False)
    sel.to_csv(a.out / "v3_selection_functions.csv", index=False)
    grd.to_csv(a.out / "v3_selection_rho1_rho2.csv", index=False)
    fig_efficiency(cur, op, a.out)
    fig_selection(sel, a.out)
    fig_population(data, a.out)
    ex = fig_example(a.base, data["ET"][0], a.out)
    (a.out / "fig_v3_example.json").write_text(json.dumps(ex, indent=1))
    print("example pair:", ex)
    for cfg in ("ET", "O4"):
        s = sel[(sel.config == cfg) & (sel.model == "pi")]
        print(cfg, s[["variable", "bin_left", "bin_right", "n", "efficiency"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
