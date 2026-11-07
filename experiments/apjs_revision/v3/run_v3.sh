#!/usr/bin/env bash
# v3 workflow on the analysis server.
#   run_v3.sh manifests CFG
#   run_v3.sh train     CFG VARIANT GPU [SEED]
#   run_v3.sh infer     CFG VARIANT GPU [SEED]
#   run_v3.sh compare   CFG A B
set -euo pipefail
STAGE=$1; CFG=$2
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev
DATA=$REV/data_v3_$CFG
WORK=$REV/work_v3_$CFG
V3=$REPO/experiments/apjs_revision/v3
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=4
mkdir -p "$WORK"/{runs,predictions,compare,logs}
cd "$REPO"
case $STAGE in
  manifests)
    $PY experiments/reproducibility/build_0228_pair_manifests.py --data-root "$DATA" --output-dir "$WORK/manifests" --families SIS ;;
  train)
    V=$3; GPU=$4; SEED=${5:-42}
    $PY $V3/train_v3.py --root "$DATA" --variant "$V" --output-root "$WORK/runs" --device cuda:$GPU --seed "$SEED" ;;
  infer)
    V=$3; GPU=$4; SEED=${5:-42}
    $PY $V3/infer_v3.py --root "$DATA" --run-dir "$WORK/runs/${V}_seed${SEED}" --manifest-dir "$WORK/manifests" \
        --output-dir "$WORK/predictions" --device cuda:$GPU ;;
  compare)
    A=$3; B=$4
    $PY $V3/compare_v3.py --pred-dir "$WORK/predictions" --a "$A" --b "$B" --manifest-dir "$WORK/manifests" --out "$WORK/compare/${A}_vs_${B}" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date +%H:%M:%S)] $STAGE $CFG ${3:-} ${4:-} finished"
