#!/usr/bin/env bash
# Tail of chain_v3_ET2.sh run in parallel: analyze_0228_core.py takes ~25 min per comparison on one
# core (10 000 bootstrap replicates), so the comparisons run side by side. Same commands as the serial
# chain; started after its first comparison (pi vs v2info), which runs on unchanged.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
R=$V3/run_v3.sh
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev
L=$REV/logs
D=$REV/data_v3_ET2; W=$REV/work_v3_ET2; W1=$REV/work_v3_ET
GPU=${SCORE_GPU:-0}
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=2
cd $REPO
score() {  # VARIANT SEED TAG
  until [ -f $W1/runs/${1}_seed$2/summary.json ]; do sleep 60; done
  $PY $V3/infer_v3.py --root $D --run-dir $W1/runs/${1}_seed$2 --manifest-dir $W/manifests --output-dir $W/predictions \
      --device cuda:$GPU --tag $3 > $L/v3_ET2_score_$3.log 2>&1
  echo "[$(date +%H:%M:%S)] scored $3"
}
cmp_() { bash $R compare ET2 $1 $2 > $L/v3_ET2_compare_$1_$2.log 2>&1; echo "[$(date +%H:%M:%S)] compared $1 $2"; }
cmp_ pi cqt &
cmp_ pi early &
(for v in pi cqt v2info early; do $PY $V3/combine_td.py --root $D --manifest-dir $W/manifests --pred-dir $W/predictions --variant $v; done
 for p in "pi_td cqt_td" "pi_td v2info_td" "pi_td early_td" "pi_td pi"; do set -- $p; cmp_ $1 $2 & done; wait) &
(score cqt 43 cqt_s43; cmp_ pi cqt_s43) &
(score pi3A 42 pi3A; cmp_ pi pi3A) &
(score pi 43 pi_s43; cmp_ pi_s43 cqt_s43 & cmp_ pi_s43 cqt & wait) &
wait
echo CHAIN_ET2_DONE
