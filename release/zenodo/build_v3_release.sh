#!/usr/bin/env bash
# Build the v3 (ApJS revision) Zenodo payloads on the analysis server.
#
#   bash release/zenodo/build_v3_release.sh analysis   # v3 analysis products -> $OUT/software/
#   bash release/zenodo/build_v3_release.sh source     # git archive of HEAD   -> $OUT/software/
#   bash release/zenodo/build_v3_release.sh catalogs   # v3 strain catalogs    -> $OUT/dataset/
#
# Nothing here is uploaded; release/zenodo/deposit_v3.py does that and stops at a draft.
set -euo pipefail
BASE=${BASE:-/root/autodl-tmp/apjs_rev}
OUT=${OUT:-$BASE/zenodo_v3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
GZ=$(command -v pigz >/dev/null && echo "pigz -p 16" || echo gzip)

manifest() {  # sha256 of every regular file under $1, paths relative to $1
  (cd "$1" && find -L . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
}

case "${1:-}" in
analysis)
  S=$OUT/stage_analysis/pi-resnet-v3-analysis; rm -rf "$OUT/stage_analysis"; mkdir -p "$S" "$OUT/software"
  E2=$BASE/work_v3_ET2; E1=$BASE/work_v3_ET; O=$BASE/work_v3_O4
  mkdir -p "$S/ET/evaluation" "$S/ET/evaluation_first_build" "$S/ET/point_mass" "$S/O4/evaluation" "$S/generation"
  for d in manifests predictions predictions_fp32 compare compare_tdc compare_fp32; do ln -s "$E2/$d" "$S/ET/evaluation/$d"; done
  for f in collected.json collected_fp32.json collected_tdc.json; do ln -s "$E2/$f" "$S/ET/evaluation/$f"; done
  for d in manifests predictions compare; do ln -s "$E1/$d" "$S/ET/evaluation_first_build/$d"; done
  ln -s "$E1/collected.json" "$S/ET/evaluation_first_build/collected.json"
  for d in manifests_pm compare_pm; do ln -s "$E1/$d" "$S/ET/point_mass/$d"; done
  ln -s "$E1/collected_pm.json" "$S/ET/point_mass/collected_pm.json"
  for d in manifests predictions predictions_fp32 compare compare_tdc compare_fp32; do ln -s "$O/$d" "$S/O4/evaluation/$d"; done
  for f in collected.json collected_fp32.json collected_tdc.json; do ln -s "$O/$f" "$S/O4/evaluation/$f"; done
  for cfg in ET O4; do
    W=$([ $cfg = ET ] && echo "$E1" || echo "$O")
    for r in "$W"/runs/*/; do
      n=$(basename "$r"); mkdir -p "$S/$cfg/runs/$n"
      for f in best.pt config.json history.csv split.npz summary.json; do [ -e "$r/$f" ] && ln -s "$r/$f" "$S/$cfg/runs/$n/$f"; done
    done
  done
  ln -s "$BASE/data_v3_ET/generation_meta_v3.json" "$S/generation/ET_generation_meta_v3.json"
  ln -s "$BASE/data_v3_ET2/generation_meta_v3.json" "$S/generation/ET2_generation_meta_v3.json"
  ln -s "$BASE/data_v3_O4/generation_meta_v3.json" "$S/generation/O4_generation_meta_v3.json"
  ln -s "$BASE/data_v3_ET/PM_data_0228/generation_meta_pm_v3.json" "$S/generation/ET_PM_generation_meta_pm_v3.json"
  ln -s "$BASE/data_v3_ET/PM_data_0228/acceptance.csv" "$S/generation/ET_PM_acceptance.csv"
  ln -s "$BASE/figures_v3" "$S/figures"
  ln -s "$BASE/cost_model" "$S/cost_model"
  ln -s "$BASE/bench" "$S/bench"
  manifest "$S"
  (cd "$OUT/stage_analysis" && tar -chf - pi-resnet-v3-analysis | $GZ > "$OUT/software/pi-resnet-v3-analysis.tar.gz")
  (cd "$OUT/software" && sha256sum pi-resnet-v3-analysis.tar.gz > pi-resnet-v3-analysis.tar.gz.sha256)
  (cd "$OUT/stage_analysis" && find -L pi-resnet-v3-analysis -type f | sort > "$OUT/software/pi-resnet-v3-analysis.filelist.txt")
  ;;
source)
  mkdir -p "$OUT/software"
  sha=$(git -C "$REPO" rev-parse --short=12 HEAD)
  git -C "$REPO" archive --format=tar.gz --prefix="lensing_classification-$sha/" -o "$OUT/software/lensing_classification-src.tar.gz" HEAD
  (cd "$OUT/software" && sha256sum lensing_classification-src.tar.gz > lensing_classification-src.tar.gz.sha256)
  echo "$sha" > "$OUT/software/lensing_classification-src.commit.txt"
  ;;
catalogs)
  mkdir -p "$OUT/dataset"
  for cfg in ET O4; do
    for d in "$BASE/data_v3_$cfg"/*/; do
      n=$(basename "$d"); f="$OUT/dataset/v3_${cfg}_${n}.tar"
      [ -s "$f" ] || tar -C "$BASE/data_v3_$cfg" -cf "$f" "$n"
    done
    cp "$BASE/data_v3_$cfg/generation_meta_v3.json" "$OUT/dataset/v3_${cfg}_generation_meta_v3.json"
  done
  # corrected ET evaluation catalogue (image 1 at its own arrival time); its other members are links into data_v3_ET
  [ -s "$OUT/dataset/v3_ET2_SIS_data_0228.tar" ] || tar -C "$BASE/data_v3_ET2" -cf "$OUT/dataset/v3_ET2_SIS_data_0228.tar" SIS_data_0228
  cp "$BASE/data_v3_ET2/generation_meta_v3.json" "$OUT/dataset/v3_ET2_generation_meta_v3.json"
  (cd "$OUT/dataset" && sha256sum *.tar *.json > SHA256SUMS)
  ;;
*) echo "usage: $0 analysis|source|catalogs"; exit 2 ;;
esac
ls -la "$OUT"/*/ 2>/dev/null
