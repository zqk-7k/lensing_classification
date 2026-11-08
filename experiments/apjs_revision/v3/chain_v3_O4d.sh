#!/usr/bin/env bash
# Protocol addendum 4 (O4): fp32 rescoring of every O4 model on GPU ${GPU:-1} once the ET fp32 rescoring
# has finished and each run is trained, the same comparisons in compare_fp32, and the time-delay
# version of the second PI-ResNet seed (registered bf16 scores) after chain_v3_O4c has finished.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev; L=$REV/logs
D=$REV/data_v3_O4; W=$REV/work_v3_O4; M=$W/manifests; P=$W/predictions
P32=$W/predictions_fp32; C32=$W/compare_fp32
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=2
GPU=${GPU:-1}
cd $REPO
cmp_() {  # PRED_DIR OUT_DIR A B
  $PY $V3/compare_v3.py --pred-dir $1 --a $3 --b $4 --manifest-dir $M --out $2/$3_vs_$4 > $L/v3_O4d_compare_$(basename $2)_$3_$4.log 2>&1
  echo "[$(date +%H:%M:%S)] compared $(basename $2) $3 $4"
}
(until grep -q CHAIN_O4C_DONE $L/chain_v3_O4c.log 2>/dev/null && [ -f $P/pi_s43_predictions_0228_sis.csv.gz ]; do sleep 60; done
 $PY $V3/combine_td.py --root $D --manifest-dir $M --pred-dir $P --variant pi_s43 > $L/v3_O4d_td_pi_s43.log 2>&1
 cmp_ $P $W/compare pi_s43_td pi_s43 & cmp_ $P $W/compare pi_s43_td cqt_td & wait) &
until grep -q "fp32 scored pi3A" $L/chain_v3_ET2c.log 2>/dev/null; do sleep 60; done
mkdir -p $P32 $C32
for vt in cqt:42:cqt v2info:42:v2info early:42:early pi:43:pi_s43 pi:42:pi; do
  IFS=: read v s t <<< "$vt"
  until [ -f $W/runs/${v}_seed$s/summary.json ]; do sleep 60; done
  $PY $V3/infer_v3.py --root $D --run-dir $W/runs/${v}_seed$s --manifest-dir $M --output-dir $P32 --device cuda:$GPU \
      --tag $t --precision fp32 > $L/v3_O4d_fp32_$t.log 2>&1
  echo "[$(date +%H:%M:%S)] fp32 scored $t"
done
for v in pi pi_s43 cqt v2info early; do $PY $V3/combine_td.py --root $D --manifest-dir $M --pred-dir $P32 --variant $v >> $L/v3_O4d_td_fp32.log 2>&1; done
for p in "pi v2info" "pi cqt" "pi early" "pi_s43 cqt" "pi_s43 v2info" \
         "pi_td pi" "pi_td cqt_td" "pi_td early_td" "pi_td v2info_td" "pi_s43_td cqt_td"; do
  set -- $p; cmp_ $P32 $C32 $1 $2 &
done
wait
echo CHAIN_O4D_DONE
