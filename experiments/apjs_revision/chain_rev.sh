#!/usr/bin/env bash
# chain_rev.sh <CFG> <GPU_PI> <GPU_CQT>
# Generate the realistic catalog root (if absent), then run the frozen workflow on it:
# prep -> PI-ResNet and CQT-DeiT training in parallel on two GPUs -> inference and core analysis.
set -euo pipefail
CFG=$1; GPU_PI=$2; GPU_CQT=$3
REV=/root/autodl-tmp/apjs_rev
S=$(cd "$(dirname "$0")" && pwd)
LOG=$REV/logs; mkdir -p "$LOG"
export TMPDIR=/root/autodl-tmp/tmp
echo "[$(date +%F' '%T)] chain $CFG start"
if [ ! -f "$REV/data_$CFG/LABEL_NOTE.md" ]; then
  OMP_NUM_THREADS=1 /root/autodl-tmp/ler-venv/bin/python "$S/generate_strain_rev.py" --config "$CFG" \
      --ler-dir "$REV/ler_$CFG/ler_data_$CFG" --out-root "$REV/data_$CFG" --workers 56 > "$LOG/gen_$CFG.log" 2>&1
  echo "[$(date +%F' '%T)] catalog generated"
fi
"$S/run_pipeline_rev.sh" "$CFG" prep > "$LOG/prep_$CFG.log" 2>&1
echo "[$(date +%F' '%T)] prep done"
"$S/run_pipeline_rev.sh" "$CFG" train_pi "$GPU_PI" > "$LOG/train_pi_$CFG.log" 2>&1 & P1=$!
"$S/run_pipeline_rev.sh" "$CFG" train_cqt "$GPU_CQT" > "$LOG/train_cqt_$CFG.log" 2>&1 & P2=$!
wait $P1; wait $P2
echo "[$(date +%F' '%T)] training done"
"$S/run_pipeline_rev.sh" "$CFG" eval "$GPU_PI" > "$LOG/eval_$CFG.log" 2>&1
echo "[$(date +%F' '%T)] CHAIN_DONE $CFG"
