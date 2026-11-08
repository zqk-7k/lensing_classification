#!/usr/bin/env bash
# Protocol addendum 5: time-delay statistic with the observing span taken from the searched 0228
# catalogue (suffix tdc), on the registered bf16 scores, ET (corrected evaluation set) and O4.
set -uo pipefail
REPO=/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
V3=$REPO/experiments/apjs_revision/v3
PY=/root/miniconda3/bin/python
REV=/root/autodl-tmp/apjs_rev; L=$REV/logs
export TMPDIR=/root/autodl-tmp/tmp MPLCONFIGDIR=/root/autodl-tmp/tmp/mpl OMP_NUM_THREADS=2
cd $REPO
for C in ET2 O4; do
  D=$REV/data_v3_$C; W=$REV/work_v3_$C
  for v in pi pi_s43 cqt v2info early; do
    $PY $V3/combine_td.py --root $D --manifest-dir $W/manifests --pred-dir $W/predictions --variant $v \
        --span-from 0228 --suffix tdc >> $L/v3_tdc_$C.log 2>&1
  done
  for p in "pi_tdc pi" "pi_tdc cqt_tdc" "pi_tdc v2info_tdc" "pi_tdc early_tdc" "pi_s43_tdc cqt_tdc"; do
    set -- $p
    ($PY $V3/compare_v3.py --pred-dir $W/predictions --a $1 --b $2 --manifest-dir $W/manifests --out $W/compare_tdc/$1_vs_$2 \
        > $L/v3_tdc_compare_${C}_$1_$2.log 2>&1; echo "[$(date +%H:%M:%S)] compared $C $1 $2") &
  done
done
wait
echo CHAIN_TDC_DONE
