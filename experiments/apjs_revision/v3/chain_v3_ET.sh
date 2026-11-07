#!/usr/bin/env bash
# ET chain: wait for the catalog, build manifests, train four variants in parallel, then score and compare.
set -uo pipefail
R=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/experiments/apjs_revision/v3/run_v3.sh
L=/root/autodl-tmp/apjs_rev/logs
until grep -q "^done in" $L/gen_v3_ET.log; do sleep 60; done
bash $R manifests ET > $L/v3_ET_manifests.log 2>&1
i=0
for v in pi v2info early cqt; do
  (bash $R train ET $v $i > $L/v3_ET_train_$v.log 2>&1; bash $R infer ET $v $i > $L/v3_ET_infer_$v.log 2>&1) &
  i=$((i+1))
done
wait
for b in v2info cqt early; do bash $R compare ET pi $b > $L/v3_ET_compare_pi_$b.log 2>&1; done
echo CHAIN_ET_DONE >> $L/v3_ET_manifests.log
