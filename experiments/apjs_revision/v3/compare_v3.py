#!/usr/bin/env python3
"""Run the unchanged frozen analysis (analyze_0228_core.py) on a pair of v3 models: model A takes the
PI-ResNet slot and model B the CQT-DeiT slot of that script's paired comparison."""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pred-dir", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--manifest-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--family", default="sis")
ap.add_argument("--bootstrap-replicates", default="10000")
x = ap.parse_args()
pd_, out = Path(x.pred_dir), Path(x.out)
slot = out / "predictions"; slot.mkdir(parents=True, exist_ok=True)
A = pd.read_csv(pd_ / f"{x.a}_predictions_0228_{x.family}.csv.gz")
B = pd.read_csv(pd_ / f"{x.b}_predictions_0228_{x.family}.csv.gz")
A.rename(columns={"logit": "pi_logit", "score": "pi_score"}).to_csv(slot / f"pi_predictions_0228_{x.family}.csv.gz", index=False, compression="gzip")
B = B.rename(columns={"logit": "cqt_deit_logit_1", "score": "cqt_deit_score"}); B.insert(5, "cqt_deit_logit_0", 0.0)
B.to_csv(slot / f"cqt_deit_predictions_0228_{x.family}.csv.gz", index=False, compression="gzip")
(out / "SLOTS.md").write_text(f"pi_resnet slot = v3 variant `{x.a}`; cqt_deit slot = v3 variant `{x.b}`.\n")
script = Path(__file__).resolve().parents[2] / "reproducibility" / "analyze_0228_core.py"
sys.exit(subprocess.call([sys.executable, str(script), "--prediction-dir", str(slot), "--manifest-dir", x.manifest_dir,
                          "--output-dir", str(out / "core"), "--families", x.family, "--bootstrap-replicates", x.bootstrap_replicates]))
