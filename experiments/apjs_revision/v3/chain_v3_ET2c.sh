#!/usr/bin/env bash
# Protocol addendum 4 (ET): time-delay version of the second PI-ResNet seed, then fp32 rescoring of
# every ET model on GPU ${GPU:-1} once the O4 v2info run has released it, and the same comparisons.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev; L=$REV/logs
D=$REV/data_v3_ET2; W=$REV/work_v3_ET2; RUNS=$REV/work_v3_ET/runs; M=$W/manifests
P32=$W/predictions_fp32; C32=$W/compare_fp32
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=2
GPU=${GPU:-1}
cd $REPO
cmp_() {  # PRED_DIR OUT_DIR A B
  $PY $V3/compare_v3.py --pred-dir $1 --a $3 --b $4 --manifest-dir $M --out $2/$3_vs_$4 > $L/v3_ET2c_compare_$(basename $2)_$3_$4.log 2>&1
  echo "[$(date +%H:%M:%S)] compared $(basename $2) $3 $4"
}
$PY $V3/combine_td.py --root $D --manifest-dir $M --pred-dir $W/predictions --variant pi_s43 > $L/v3_ET2c_td_pi_s43.log 2>&1
cmp_ $W/predictions $W/compare pi_s43_td pi_s43 &
cmp_ $W/predictions $W/compare pi_s43_td cqt_td &
until grep -q "infer O4 v2info .* finished" $L/v3_O4_infer_v2info.log 2>/dev/null; do sleep 60; done
mkdir -p $P32 $C32
for vt in pi:42:pi pi:43:pi_s43 early:42:early cqt:42:cqt cqt:43:cqt_s43 v2info:42:v2info pi3A:42:pi3A; do
  IFS=: read v s t <<< "$vt"
  $PY $V3/infer_v3.py --root $D --run-dir $RUNS/${v}_seed$s --manifest-dir $M --output-dir $P32 --device cuda:$GPU \
      --tag $t --precision fp32 > $L/v3_ET2c_fp32_$t.log 2>&1
  echo "[$(date +%H:%M:%S)] fp32 scored $t"
done
for v in pi pi_s43 cqt v2info early; do $PY $V3/combine_td.py --root $D --manifest-dir $M --pred-dir $P32 --variant $v >> $L/v3_ET2c_td_fp32.log 2>&1; done
for p in "pi v2info" "pi cqt" "pi early" "pi pi3A" "pi cqt_s43" "pi_s43 cqt_s43" "pi_s43 cqt" \
         "pi_td pi" "pi_td cqt_td" "pi_td early_td" "pi_td v2info_td" "pi_s43_td cqt_td"; do
  set -- $p; cmp_ $P32 $C32 $1 $2 &
done
wait
echo CHAIN_ET2C_DONE
