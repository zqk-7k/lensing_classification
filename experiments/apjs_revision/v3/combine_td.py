#!/usr/bin/env python3
"""Add the time-delay log likelihood ratio to a v3 model's 0228 logits (see PROTOCOL_V3.md)."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--family", default="SIS")
ap.add_argument("--manifest-dir", required=True); ap.add_argument("--pred-dir", required=True)
ap.add_argument("--variant", required=True)
ap.add_argument("--span-from", choices=("0222", "0228"), default="0222",
                help="observing span T from the training catalogue (registered) or from the searched 0228 catalogue's "
                     "event times (protocol addendum 5: guarantees dt <= T for every searched pair)")
ap.add_argument("--suffix", default="td", help="output name: <variant>_<suffix>_predictions_...")
a = ap.parse_args()
root, fam = Path(a.root), a.family
p22 = pd.read_csv(root / f"{fam}_data_0222/pair_meta.csv"); u22 = pd.read_csv(root / "Unlensed_data_0222/event_meta.csv")
p28 = pd.read_csv(root / f"{fam}_data_0228/pair_meta.csv"); u28 = pd.read_csv(root / "Unlensed_data_0228/event_meta.csv")
span_p, span_u = (p22, u22) if a.span_from == "0222" else (p28, u28)
t_span = np.r_[span_p.geocent_time, span_p.geocent_time + span_p.delay_s, span_u.geocent_time]
T = float(t_span.max() - t_span.min())
kde = gaussian_kde(np.log(np.abs(p22.delay_s.to_numpy())))


def log_ratio(dt):
    dt = np.clip(np.abs(dt), 1.0, T * (1 - 1e-9))
    log_pl = np.log(kde(np.log(dt))) - np.log(dt)           # density in dt
    log_pu = np.log(2.0 * (T - dt) / T ** 2)
    return log_pl - log_pu


times = {"lensed_img1": p28.geocent_time.to_numpy(), "lensed_img2": (p28.geocent_time + p28.delay_s).to_numpy(),
         "unlensed": u28.geocent_time.to_numpy()}
man = pd.concat([pd.read_csv(Path(a.manifest_dir) / f"0228_{fam.lower()}_{p}_pairs.csv.gz") for p in ("calibration", "evaluation")])
tl = np.array([times[t][i] for t, i in zip(man.left_event_type, man.left_event_index)])
tr = np.array([times[t][i] for t, i in zip(man.right_event_type, man.right_event_index)])
lr = pd.DataFrame({"pair_id": man.pair_id.to_numpy(), "td_log_ratio": log_ratio(tr - tl), "dt_s": np.abs(tr - tl)})
pred = pd.read_csv(Path(a.pred_dir) / f"{a.variant}_predictions_0228_{fam.lower()}.csv.gz").merge(lr, on="pair_id")
pred["logit"] = pred["logit"] + pred["td_log_ratio"]
pred["score"] = 1 / (1 + np.exp(-pred["logit"]))
out = Path(a.pred_dir) / f"{a.variant}_{a.suffix}_predictions_0228_{fam.lower()}.csv.gz"
pred.drop(columns=["td_log_ratio", "dt_s"]).to_csv(out, index=False, compression="gzip")
lr.to_csv(Path(a.pred_dir) / f"{a.suffix}_log_ratio_0228_{fam.lower()}.csv.gz", index=False, compression="gzip")
(Path(a.pred_dir) / f"{a.variant}_{a.suffix}_predictions_0228_{fam.lower()}.metadata.json").write_text(json.dumps(
    {"observing_span_s": T, "span_from": a.span_from, "n_pairs_dt_ge_T": int((lr.dt_s >= T).sum()),
     "n_kde": len(p22), "kde_bandwidth": float(kde.factor), "base_variant": a.variant}, indent=2))
print("wrote", out, "span days", T / 86400)
