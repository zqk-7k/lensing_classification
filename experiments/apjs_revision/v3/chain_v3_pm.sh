#!/usr/bin/env bash
# ET point-mass lens-profile robustness set: generate on the 0228 sources, build the PM manifests,
# score with the galaxy-trained ET models once the ET chain is done, compare with the frozen analysis.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev
L=$REV/logs
D=$REV/data_v3_ET; W=$REV/work_v3_ET
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl
cd $REPO
OMP_NUM_THREADS=1 /root/autodl-tmp/ler-venv/bin/python $V3/generate_pm_v3.py --config ET --root $D --workers ${PM_WORKERS:-24}
export OMP_NUM_THREADS=4
$PY experiments/reproducibility/build_0228_pair_manifests.py --data-root $D --output-dir $W/manifests_pm --families SIS,PM
until grep -q "CHAIN_ET_DONE" $L/v3_ET_manifests.log 2>/dev/null; do sleep 60; done
for v in pi cqt v2info early; do
  $PY $V3/infer_v3.py --root $D --family PM --run-dir $W/runs/${v}_seed42 --manifest-dir $W/manifests_pm \
      --output-dir $W/predictions --device cuda:${PM_GPU:-0}
done
for b in cqt v2info early; do
  $PY $V3/compare_v3.py --pred-dir $W/predictions --a pi --b $b --manifest-dir $W/manifests_pm --family pm --out $W/compare_pm/pi_vs_$b
done
echo PM_DONE
