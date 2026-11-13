"""Cost-benefit model for a staged lensed-pair search.

A per-pair method must be judged by the
combination of efficiency, false-alarm probability per pair, per-pair cost and
the cost of the downstream analysis it feeds, not by speed or efficiency alone.

Three ways of handling the N_pair candidate pairs of a catalogue are compared:

  A. posterior-overlap statistic only   (PO2.0, Barsode et al. 2024: eff e_B at FAP p_B, cost c_B per pair,
                                         plus parameter estimation once per event)
  B. PI-ResNet only                     (eff e_P at FAP p_P, cost c_P per pair, strain only)
  C. PI-ResNet as a pre-filter in front of A

Every pair that survives the last fast stage is assumed to go to a full
Bayesian joint analysis at cost C_PE per pair.  With c_pe_event the per-event
parameter-estimation cost amortised over the (N-1)/2 pairs each event enters,
the expected compute per input pair is

  A:  c_pe_event + c_B + p_B * C_PE
  B:  c_P + p_P * C_PE
  C:  c_P + f_ev * c_pe_event + p_P * c_B + p_P * p_B * C_PE
      f_ev = 1 - (1 - p_P)^(N - 1) is the fraction of events that enter at least one surviving pair and
      therefore still need parameter estimation.  (The first version charged PE only to the fraction p_P
      of pairs, which undercounts it whenever p_P * N is not small: at ET, p_P = 1e-3 and N = 1e5 leave
      every event about 100 surviving partners, so every event still needs PE.)

Efficiencies are e_B, e_P and e_P * e_B (independence; a lower bound, since
both statistics respond mostly to the fainter image's S/N).  The cost that
matters to a search is the compute per RECOVERED lensed pair, which is the
per-pair cost divided by the efficiency; it is reported alongside.

All inputs come from cost_model_inputs.json (sources documented there).
PI-ResNet v3 configs (ET_v3, ET_v3_td, O4_v3, O4_v3_td) use the measured CPU cost per pair amortised over
the catalogue, 2 c_event / (N - 1) + c_head, since each event is encoded once and only the pairwise head
runs per pair; the older ET / O4 configs use the submitted end-to-end per-pair cost C_P_s.

Run:  python experiments/apjs_revision/v3/cost_model/cost_model.py [--config ET|O4|ET_v3|ET_v3_td|O4_v3|O4_v3_td]
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

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 150})
C_PE_GRID = np.logspace(0, 6, 400)  # downstream joint-PE cost per pair, 1 s .. 1e6 s


def load_inputs():
    return json.loads((OUT / "cost_model_inputs.json").read_text())


def pi_cost(pi, inp, n):
    """PI-ResNet CPU cost per pair in a catalogue of n events: each event is encoded once and only the
    pairwise head runs per pair (v3 configs, measured), else the submitted end-to-end per-pair cost."""
    pc = pi.get("cost")
    if pc is None:
        return inp["C_P_s"]
    return 2.0 * pc["event_encoding_cpu_s"] / (n - 1) + pc["pair_head_cpu_s"]


def event_coverage(p_p, n):
    """Fraction of the n events that appear in at least one pair passing a per-pair FAP p_p."""
    return 1.0 - (1.0 - p_p) ** (n - 1)


def cost_per_pair(p_p, c_pe, c_p, c_b, p_b, c_pe_event, n):
    return {
        "A_overlap_only": c_pe_event + c_b + p_b * c_pe,
        "B_piresnet_only": c_p + p_p * c_pe,
        "C_prefilter_then_overlap": c_p + event_coverage(p_p, n) * c_pe_event + p_p * c_b + p_p * p_b * c_pe,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ET", help="which PI-ResNet efficiencies to use (key of PI in the inputs)")
    args = ap.parse_args()
    inp = load_inputs()
    pi = inp["PI"][args.config]
    ov = inp["overlap"]
    e_b, p_b = ov["eff"], ov["fap"]
    c_pe_ref = inp["C_PE_ref_s"]
    pe_event_amortised = {name: ov["pe_per_event_s"] * 2.0 / (n - 1) for name, n in inp["catalogues"].items()}
    cat = pi.get("catalogue", "O4-like year (~200 BBH)")  # catalogue the per-pair table and figure amortise over
    c_p = pi_cost(pi, inp, inp["catalogues"][cat])
    fap_keys = [k for k in pi if k not in ("source", "cost", "catalogue")]

    rows = []
    for p_p_str in fap_keys:
        d = pi[p_p_str]
        p_p = float(p_p_str)
        e_p = d["eff"]
        for c_b in ov["cost_s_grid"]:
            c = cost_per_pair(p_p, c_pe_ref, c_p, c_b, p_b, pe_event_amortised[cat], inp["catalogues"][cat])
            e_c = (e_p * e_b) if e_p is not None else None
            rows.append({
                "config": args.config, "catalogue": cat, "piresnet_cost_per_pair_s": c_p,
                "piresnet_fap": p_p, "piresnet_eff": e_p, "overlap_cost_s": c_b,
                "cost_A_overlap_only_s": c["A_overlap_only"],
                "cost_B_piresnet_only_s": c["B_piresnet_only"],
                "cost_C_prefilter_s": c["C_prefilter_then_overlap"],
                "eff_A": e_b, "eff_B": e_p, "eff_C_independent": e_c,
                "eff_C_nested_upper": min(e_p, e_b) if e_p is not None else None,
                "cost_per_recovered_A_s": c["A_overlap_only"] / e_b,
                "cost_per_recovered_B_s": (c["B_piresnet_only"] / e_p) if e_p else None,
                "cost_per_recovered_C_s": (c["C_prefilter_then_overlap"] / e_c) if e_c else None,
                "downstream_A": p_b, "downstream_B": p_p, "downstream_C": p_p * p_b,
                "break_even_C_PE_B_vs_A_s": (c_b + pe_event_amortised[cat] - c_p) / (p_p - p_b),
                "speedup_C_vs_A_at_1_CPU_day": c["A_overlap_only"] / c["C_prefilter_then_overlap"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"cost_model_table_{args.config}.csv", index=False)

    # catalogue-scale totals at the primary PI-ResNet point, nominal overlap cost, one CPU-day PE
    scale_rows = []
    e_p1 = pi["0.001"]["eff"]
    for name, n in inp["catalogues"].items():
        npairs = n * (n - 1) / 2
        c = cost_per_pair(1e-3, c_pe_ref, pi_cost(pi, inp, n), ov["cost_s_nominal"], p_b, pe_event_amortised[name], n)
        n_lensed = npairs * 4.5e-4  # lensed fraction of detected events (ler, O4 and ET agree to 10%)
        scale_rows.append({
            "catalogue": name, "n_events": n, "n_pairs": npairs, "piresnet_cost_per_pair_s": pi_cost(pi, inp, n),
            "A_overlap_only_CPU_years": npairs * c["A_overlap_only"] / 3.156e7,
            "B_piresnet_only_CPU_years": npairs * c["B_piresnet_only"] / 3.156e7,
            "C_prefilter_CPU_years": npairs * c["C_prefilter_then_overlap"] / 3.156e7,
            "C_events_needing_PE": n * event_coverage(1e-3, n),
            "A_pairs_to_PE": npairs * p_b, "B_pairs_to_PE": npairs * 1e-3, "C_pairs_to_PE": npairs * 1e-3 * p_b,
            "eff_A": e_b, "eff_B": e_p1, "eff_C_independent": (e_p1 * e_b) if e_p1 else None,
        })
    scale = pd.DataFrame(scale_rows)
    scale.to_csv(OUT / f"cost_model_catalogue_scale_{args.config}.csv", index=False)

    # figure: total cost per pair versus downstream cost, nominal overlap cost
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    c = cost_per_pair(1e-3, C_PE_GRID, c_p, ov["cost_s_nominal"], p_b, pe_event_amortised[cat], inp["catalogues"][cat])
    ax.plot(C_PE_GRID, c["A_overlap_only"], color="#9467bd", lw=1.3, label=f"posterior overlap only (FAP {p_b:.0e})")
    ax.plot(C_PE_GRID, c["B_piresnet_only"], color="#1f77b4", lw=1.3, label="PI-ResNet only (FAP $10^{-3}$)")
    ax.plot(C_PE_GRID, c["C_prefilter_then_overlap"], color="#d62728", lw=1.3, label="PI-ResNet pre-filter, then overlap")
    be = (ov["cost_s_nominal"] + pe_event_amortised[cat] - c_p) / (1e-3 - p_b)
    ax.axvline(be, color="0.5", lw=0.7, ls=":")
    ax.text(be * 0.75, 3e-2, f"break-even: {be/3600:.1f} h", fontsize=6.5, color="0.3", ha="right")
    for x, lab in ((3600, "1 hour"), (86400, "1 day")):
        ax.axvline(x, color="0.8", lw=0.6, ls="--")
        ax.text(x * 1.15, 3e-3, lab, fontsize=6.5, color="0.4", rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cost of full Bayesian follow-up per pair [CPU s]")
    ax.set_ylabel("Expected compute per candidate pair [CPU s]")
    ax.legend(frameon=False, loc="upper left", fontsize=6.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_cost_model_{args.config}.{ext}", bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    print()
    print(scale.to_string(index=False))


if __name__ == "__main__":
    main()
