#!/usr/bin/env bash
# ET with the corrected evaluation set: the 0228 lensed images placed at their own ler arrival times
# (generate_v3.py --image-time arrival, same role assignment, spins and noise seeds). Training data
# (0222) and the point-mass set are those of data_v3_ET (linked). Every ET model is scored on it as its
# run finishes; comparisons and the time-delay statistic as in PROTOCOL_V3.md.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
R=$V3/run_v3.sh
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev
L=$REV/logs
D=$REV/data_v3_ET2; W=$REV/work_v3_ET2; W1=$REV/work_v3_ET; D1=$REV/data_v3_ET
GPU=${SCORE_GPU:-0}
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl
cd $REPO
OMP_NUM_THREADS=1 /root/autodl-tmp/ler-venv/bin/python $V3/generate_v3.py --config ET --ler-dir $REV/ler_ET/ler_data_ET --out-root $D \
    --roles 0228 --image-time arrival --workers ${GEN_WORKERS:-16}
export OMP_NUM_THREADS=4
for d in SIS_data_0222 Unlensed_data_0222 PM_data_0228; do ln -sfn $D1/$d $D/$d; done
$PY - "$D" "$D1" <<'PYEOF'
import sys, numpy as np, pandas as pd
d, d1 = sys.argv[1:]
a, b = pd.read_csv(f"{d}/SIS_data_0228/pair_meta.csv"), pd.read_csv(f"{d1}/SIS_data_0228/pair_meta.csv")
assert (a.system_index.values == b.system_index.values).all() and (a.ler_file.values == b.ler_file.values).all()
assert np.allclose(a.delay_s, b.delay_s) and np.allclose(a.mu_1, b.mu_1) and (a.noise_seed_1.values == b.noise_seed_1.values).all()
assert np.allclose(a.geocent_time - b.geocent_time, a.t_offset_1, atol=1e-3)
for f in ("sig_A", "data_A", "data_B"):
    assert (np.load(f"{d}/Unlensed_data_0228/{f}.npy", mmap_mode="r") == np.load(f"{d1}/Unlensed_data_0228/{f}.npy", mmap_mode="r")).all(), f
print("ET2 0228: same systems, delays, magnifications and noise seeds as data_v3_ET; unlensed set identical")
for k in (1, 2):
    r = a[f"snr_net20_{k}"] / a[f"snr_net_ler_{k}"]
    print(f"image {k}: injected/ler S/N from 20 Hz, 5/50/95 %:", np.round(np.quantile(r, [.05, .5, .95]), 3),
          f"below 8: {float((a[f'snr_net20_{k}'] < 8).mean()):.3f} (ler {float((a[f'snr_net_ler_{k}'] < 8).mean()):.3f})")
PYEOF
bash $R manifests ET2
mkdir -p $W; ln -sfn $W1/runs $W/runs
score() {  # VARIANT SEED TAG
  until [ -f $W1/runs/${1}_seed$2/summary.json ]; do sleep 60; done
  $PY $V3/infer_v3.py --root $D --run-dir $W1/runs/${1}_seed$2 --manifest-dir $W/manifests --output-dir $W/predictions \
      --device cuda:$GPU --tag $3
  echo "[$(date +%H:%M:%S)] scored $3"
}
cmp_() { bash $R compare ET2 $1 $2 > $L/v3_ET2_compare_$1_$2.log 2>&1; echo "[$(date +%H:%M:%S)] compared $1 $2"; }
for v in pi early cqt v2info; do score $v 42 $v; done
for b in v2info cqt early; do cmp_ pi $b; done
for v in pi cqt v2info early; do $PY $V3/combine_td.py --root $D --manifest-dir $W/manifests --pred-dir $W/predictions --variant $v; done
for p in "pi_td cqt_td" "pi_td v2info_td" "pi_td early_td" "pi_td pi"; do set -- $p; cmp_ $1 $2; done
score cqt 43 cqt_s43; cmp_ pi cqt_s43
score pi3A 42 pi3A; cmp_ pi pi3A
score pi 43 pi_s43; cmp_ pi_s43 cqt_s43; cmp_ pi_s43 cqt
echo CHAIN_ET2_DONE
