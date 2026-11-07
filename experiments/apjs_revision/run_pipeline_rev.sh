#!/usr/bin/env bash
# Frozen reproducibility workflow on a realistic (ler) catalog root.
#   run_pipeline_rev.sh <CFG> <STAGE> [GPU]
#   CFG   : O4 | ET            (data root  $REV/data_<CFG>, work root $REV/work_<CFG>)
#   STAGE : prep | train_pi | train_cqt | eval | pm | extras | all
# Every step is the unchanged repository script with its --data-root/--output arguments
# pointed at the revision roots; the only protocol-level additions are --families SIS.
set -euo pipefail
CFG=$1; STAGE=$2; GPU=${3:-0}
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev
DATA=$REV/data_$CFG
WORK=$REV/work_$CFG
E=$REPO/experiments/reproducibility
SPLIT=$WORK/manifests/split_0222_seed42.npz
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=4
mkdir -p "$WORK"/{manifests,artifacts,results,logs} "$TMPDIR/mpl"
cd "$REPO"
log() { echo "[$(date +%H:%M:%S)] [$CFG/$STAGE] $*"; }

prep() {
  log "build_split_0222";            $PY $E/build_split_0222.py --data-root "$DATA" --output-dir "$WORK/manifests" --families SIS
  log "build_cqt_training_pairs";    $PY $E/build_cqt_training_pairs.py --data-root "$DATA" --out-root "$WORK/cqt_images" --lens SIS \
                                        --split-manifest "$SPLIT" --audit-dir "$WORK/results/audit" --workers 40
  log "build_0228_pair_manifests";   $PY $E/build_0228_pair_manifests.py --data-root "$DATA" --output-dir "$WORK/manifests/0228_pairs" --families SIS
  log "prepare_cqt_cache_0228";      $PY $E/prepare_cqt_cache_0228.py --data-root "$DATA" --output-dir "$WORK/artifacts/cqt_cache" \
                                        --families SIS --reference-root "$WORK/cqt_images" --workers 40
}
train_pi() {
  log "train_pi_resnet on cuda:$GPU"
  $PY $E/train_pi_resnet.py --lens SIS --data-root "$DATA" --output-root "$WORK/artifacts/training" --split-manifest "$SPLIT" --device cuda:$GPU
}
train_cqt() {
  log "train_cqt_deit on cuda:$GPU"
  $PY $E/train_cqt_deit.py --lens SIS --image-root "$WORK/cqt_images" --output-root "$WORK/artifacts/training" --split-manifest "$SPLIT" --device cuda:$GPU
}
eval_() {
  log "infer_pi_0228";   $PY $E/infer_pi_0228.py  --lens SIS --data-root "$DATA" --manifest-dir "$WORK/manifests/0228_pairs" \
                            --training-root "$WORK/artifacts/training" --output-dir "$WORK/results/predictions" --device cuda:$GPU
  log "infer_cqt_0228";  $PY $E/infer_cqt_0228.py --lens SIS --manifest-dir "$WORK/manifests/0228_pairs" --cache-dir "$WORK/artifacts/cqt_cache" \
                            --training-root "$WORK/artifacts/training" --output-dir "$WORK/results/predictions" --device cuda:$GPU
  log "analyze_0228_core"; $PY $E/analyze_0228_core.py --prediction-dir "$WORK/results/predictions" --manifest-dir "$WORK/manifests/0228_pairs" \
                            --output-dir "$WORK/results/core" --families sis
}
pm() {
  # Point-mass lens-profile robustness set on the 0228 sources, scored by the galaxy-trained PI-ResNet.
  log "generate_pm_robustness_rev"
  OMP_NUM_THREADS=1 /root/autodl-tmp/ler-venv/bin/python "$(dirname "$0")/generate_pm_robustness_rev.py" --config "$CFG" --root "$DATA" --workers 48
  log "build_0228_pair_manifests (SIS,PM) into a separate manifest dir"
  # The hard background is sampled without replacement inside source blocks; with few accepted
  # point-mass sources (741 at O4) the 100k default cannot be met and the sampler never returns,
  # so PM_BACKGROUND (default 100000) is exposed here. O4 used 10000.
  $PY $E/build_0228_pair_manifests.py --data-root "$DATA" --output-dir "$WORK/manifests/0228_pairs_pm" --families SIS,PM \
      --background-count "${PM_BACKGROUND:-100000}"
  log "infer_pi_0228 PM with the SIS checkpoint"
  $PY $E/infer_pi_0228.py --lens PM --checkpoint-lens SIS --data-root "$DATA" --manifest-dir "$WORK/manifests/0228_pairs_pm" \
      --training-root "$WORK/artifacts/training" --output-dir "$WORK/results/predictions" --device cuda:$GPU
}
extras() {
  log "analyze_revision_extras"; $PY "$(dirname "$0")/analyze_revision_extras.py" --work "$WORK" --data "$DATA"
}
case $STAGE in
  prep) prep ;; train_pi) train_pi ;; train_cqt) train_cqt ;; eval) eval_ ;; pm) pm ;; extras) extras ;;
  all) prep; train_pi; train_cqt; eval_ ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
log "finished"
