#!/usr/bin/env python3
"""Collect the frozen-analysis outputs of every v3 comparison into one markdown table and one JSON.

Reads <work>/compare/<A>_vs_<B>/core/core_results.json (written by analyze_0228_core.py through
compare_v3.py, model A in the pi_resnet slot, model B in the cqt_deit slot).
"""
import argparse, json
from pathlib import Path

PF = ["0.01", "0.001", "0.0001"]


def pct(x):
    return f"{100 * x:.1f}"


def ci(c):
    return f"[{pct(c[0])}, {pct(c[1])}]"


ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--out", default=None)
ap.add_argument("--compare-dir", default="compare"); ap.add_argument("--family", default="sis")
a = ap.parse_args()
work = Path(a.work)
models, pairs = {}, []
for d in sorted((work / a.compare_dir).glob("*_vs_*")):
    f = d / "core" / "core_results.json"
    if not f.exists():
        continue
    A, B = d.name.split("_vs_")
    r = json.loads(f.read_text())["results"][a.family]
    for name, slot in ((A, "pi_resnet"), (B, "cqt_deit")):
        m = r["models"][slot]
        models.setdefault(name, {"auc": m["auc"], "ap": m["average_precision"],
                                 **{p: {k: m["operating_points"][p][k] for k in ("efficiency", "efficiency_ci", "achieved_fpp", "fpp_ci")} for p in PF}})
    p = r["paired"]
    pairs.append({"a": A, "b": B, "auc_diff": p["auc_difference_pi_minus_cqt"], "auc_diff_ci": p["block_auc_difference_ci"],
                  **{pf: p["efficiency_difference"][pf] for pf in PF}})
lines = [f"### {work.name} / {a.compare_dir} ({a.family})", "", "| Model | AUC | eff @ 1e-2 | eff @ 1e-3 | eff @ 1e-4 |", "|---|---:|---:|---:|---:|"]
for n, m in sorted(models.items(), key=lambda kv: -kv[1]["0.001"]["efficiency"]):
    lines.append(f"| {n} | {m['auc']:.4f} | " + " | ".join(f"{pct(m[p]['efficiency'])} {ci(m[p]['efficiency_ci'])}" for p in PF) + " |")
lines += ["", "| A − B | ΔAUC | Δeff @ 1e-2 | Δeff @ 1e-3 | Δeff @ 1e-4 |", "|---|---:|---:|---:|---:|"]
for q in pairs:
    lines.append(f"| {q['a']} − {q['b']} | {q['auc_diff']:+.4f} [{q['auc_diff_ci'][0]:+.4f}, {q['auc_diff_ci'][1]:+.4f}] | "
                 + " | ".join(f"{100 * q[p]['point']:+.1f} {ci(q[p]['ci'])}" for p in PF) + " |")
txt = "\n".join(lines) + "\n"
print(txt)
if a.out:
    Path(a.out).write_text(json.dumps({"models": models, "pairs": pairs}, indent=1))
