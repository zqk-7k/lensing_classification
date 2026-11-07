#!/usr/bin/env bash
# O4 chain: wait for the ler draw, generate, build manifests; train pi / v2info / cqt as the ET
# runs free their GPUs (ET cqt -> GPU 3, ET early -> GPU 2, ET v2info -> GPU 1); then score and compare.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
R=$REPO/experiments/apjs_revision/v3/run_v3.sh
L=/root/autodl-tmp/apjs_rev/logs
cd /root/autodl-tmp/apjs_rev
until grep -q "^done" $L/ler_v3_O4.log; do sleep 60; done
until grep -q "^done in" $L/gen_v3_ET.log; do sleep 60; done
TMPDIR=/root/autodl-tmp/tmp OMP_NUM_THREADS=1 /root/autodl-tmp/ler-venv/bin/python $REPO/experiments/apjs_revision/v3/generate_v3.py \
  --config O4 --ler-dir ler_O4/ler_data_O4 --out-root data_v3_O4 --n-lensed-0228 4000 --n-unlensed-0228 8000 --workers 56 > $L/gen_v3_O4.log 2>&1
bash $R manifests O4 > $L/v3_O4_manifests.log 2>&1
waitfor() { until grep -q "infer ET $1 .* finished" $L/v3_ET_infer_$1.log 2>/dev/null; do sleep 60; done; }
(waitfor cqt;    bash $R train O4 pi 3     > $L/v3_O4_train_pi.log 2>&1;     bash $R infer O4 pi 3     > $L/v3_O4_infer_pi.log 2>&1) &
(waitfor early;  bash $R train O4 cqt 2    > $L/v3_O4_train_cqt.log 2>&1;    bash $R infer O4 cqt 2    > $L/v3_O4_infer_cqt.log 2>&1) &
(waitfor v2info; bash $R train O4 v2info 1 > $L/v3_O4_train_v2info.log 2>&1; bash $R infer O4 v2info 1 > $L/v3_O4_infer_v2info.log 2>&1) &
wait
for b in v2info cqt; do bash $R compare O4 pi $b > $L/v3_O4_compare_pi_$b.log 2>&1; done
echo CHAIN_O4_DONE >> $L/v3_O4_manifests.log
