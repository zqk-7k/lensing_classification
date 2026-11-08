#!/usr/bin/env bash
# O4 additions of protocol addendum 3: early-fusion baseline on GPU 2, then pi vs early and the
# time-delay versions once chain_v3_O4.sh has scored pi, cqt and v2info.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
R=$V3/run_v3.sh
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev; L=$REV/logs; D=$REV/data_v3_O4; W=$REV/work_v3_O4
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=2
GPU=${GPU:-2}
cd $REPO
cmp_() { bash $R compare O4 $1 $2 > $L/v3_O4_compare_$1_$2.log 2>&1; echo "[$(date +%H:%M:%S)] compared $1 $2"; }
have() { [ -f $W/predictions/$1_predictions_0228_sis.csv.gz ]; }
bash $R train O4 early $GPU > $L/v3_O4_train_early.log 2>&1
bash $R infer O4 early $GPU > $L/v3_O4_infer_early.log 2>&1
echo "[$(date +%H:%M:%S)] scored early"
until have pi; do sleep 60; done
cmp_ pi early &
until have v2info && have cqt; do sleep 60; done
for v in pi cqt v2info early; do $PY $V3/combine_td.py --root $D --manifest-dir $W/manifests --pred-dir $W/predictions --variant $v; done
for p in "pi_td cqt_td" "pi_td v2info_td" "pi_td early_td" "pi_td pi"; do set -- $p; cmp_ $1 $2 & done
wait
echo CHAIN_O4C_DONE
