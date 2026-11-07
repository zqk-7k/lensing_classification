#!/usr/bin/env bash
# Scoring and comparison half of chain_v3_pm.sh (generation and manifests done), started as soon as the
# four seed-42 ET runs are final instead of after the first-build ET chain; comparisons in parallel.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev
D=$REV/data_v3_ET; W=$REV/work_v3_ET
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=2
cd $REPO
for v in pi cqt v2info early; do
  until [ -f $W/runs/${v}_seed42/summary.json ]; do sleep 60; done
  $PY $V3/infer_v3.py --root $D --family PM --run-dir $W/runs/${v}_seed42 --manifest-dir $W/manifests_pm \
      --output-dir $W/predictions --device cuda:${PM_GPU:-1}
done
for b in cqt v2info early; do
  $PY $V3/compare_v3.py --pred-dir $W/predictions --a pi --b $b --manifest-dir $W/manifests_pm --family pm --out $W/compare_pm/pi_vs_$b &
done
wait
echo PM_DONE
