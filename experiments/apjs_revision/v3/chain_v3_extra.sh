#!/usr/bin/env bash
# Extra ET runs, queued on GPUs as the main chains free them:
#   GPU 0 (after ET pi):  pi3A ablation (stream A only), then a second CQT seed (43)
#   GPU 2 (after O4 cqt): a second PI-ResNet seed (43)
# plus the pre-registered time-delay statistic once the ET chain has scored every variant.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
R=$V3/run_v3.sh
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev
L=$REV/logs
D=$REV/data_v3_ET; W=$REV/work_v3_ET
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=4
cd $REPO
seedinfer() {  # VARIANT GPU SEED
  $PY $V3/infer_v3.py --root $D --run-dir $W/runs/${1}_seed$3 --manifest-dir $W/manifests --output-dir $W/predictions \
      --device cuda:$2 --tag ${1}_s$3
  echo "[$(date +%H:%M:%S)] infer ET $1 seed $3 finished"
}
(until grep -q "CHAIN_ET_DONE" $L/v3_ET_manifests.log 2>/dev/null; do sleep 60; done
 for v in pi cqt v2info early; do
   $PY $V3/combine_td.py --root $D --manifest-dir $W/manifests --pred-dir $W/predictions --variant $v
 done
 for p in "pi_td cqt_td" "pi_td v2info_td" "pi_td early_td" "pi_td pi"; do
   set -- $p; bash $R compare ET $1 $2 > $L/v3_ET_compare_$1_$2.log 2>&1
 done
 echo TD_DONE) > $L/v3_ET_td.log 2>&1 &
(until grep -q "infer ET pi .* finished" $L/v3_ET_infer_pi.log 2>/dev/null; do sleep 60; done
 bash $R train ET pi3A 0 > $L/v3_ET_train_pi3A.log 2>&1; bash $R infer ET pi3A 0 > $L/v3_ET_infer_pi3A.log 2>&1
 until grep -q "CHAIN_ET_DONE" $L/v3_ET_manifests.log 2>/dev/null; do sleep 60; done
 bash $R compare ET pi pi3A > $L/v3_ET_compare_pi_pi3A.log 2>&1
 bash $R train ET cqt 0 43 > $L/v3_ET_train_cqt_s43.log 2>&1; seedinfer cqt 0 43 > $L/v3_ET_infer_cqt_s43.log 2>&1) &
(until grep -q "infer O4 cqt .* finished" $L/v3_O4_infer_cqt.log 2>/dev/null; do sleep 60; done
 bash $R train ET pi 2 43 > $L/v3_ET_train_pi_s43.log 2>&1; seedinfer pi 2 43 > $L/v3_ET_infer_pi_s43.log 2>&1) &
wait
until [ -f $W/predictions/cqt_s43_predictions_0228_sis.csv.gz ] && [ -f $W/predictions/pi_s43_predictions_0228_sis.csv.gz ]; do sleep 60; done
for p in "pi_s43 cqt_s43" "pi_s43 cqt" "pi cqt_s43"; do
  set -- $p; bash $R compare ET $1 $2 > $L/v3_ET_compare_$1_$2.log 2>&1
done
echo CHAIN_EXTRA_DONE >> $L/v3_ET_td.log
