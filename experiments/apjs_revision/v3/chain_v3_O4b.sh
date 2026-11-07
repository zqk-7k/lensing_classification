#!/usr/bin/env bash
# Second O4 PI-ResNet seed (43) on GPU 0 (free after the ET runs), scored and compared with the
# seed-42 baselines once chain_v3_O4.sh has written them. The v2 O4 cold starts collapsed; a second
# v3 seed checks that this one does not.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
R=$V3/run_v3.sh
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev; L=$REV/logs; D=$REV/data_v3_O4; W=$REV/work_v3_O4
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=4
cd $REPO
bash $R train O4 pi ${GPU:-0} 43 > $L/v3_O4_train_pi_s43.log 2>&1
$PY $V3/infer_v3.py --root $D --run-dir $W/runs/pi_seed43 --manifest-dir $W/manifests --output-dir $W/predictions \
    --device cuda:${GPU:-0} --tag pi_s43 > $L/v3_O4_infer_pi_s43.log 2>&1
for b in cqt v2info; do
  until [ -f $W/predictions/${b}_predictions_0228_sis.csv.gz ]; do sleep 60; done
  bash $R compare O4 pi_s43 $b > $L/v3_O4_compare_pi_s43_$b.log 2>&1 &
done
wait
echo CHAIN_O4B_DONE
