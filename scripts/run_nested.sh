#!/usr/bin/env bash
# Nested isolation for any trainer built on scripts/qc_cnn_common.py (docs/PREREG_NEW_MODELS.md):
# 5 base models (one per outer fold f, trained without f) + 20 inner ones (trained without f and j),
# the same scheme as scripts/run_hip_nested_local.sh for the hip CNN.
# Usage: [DEVICE=mps] scripts/run_nested.sh scripts/train_spine_cnn.py outputs/spine_cnn [trainer args]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"; export TORCH_HOME="$PWD/models/torch_home" PYTHONUNBUFFERED=1
TRAINER=$1; ROOTDIR=$2; shift 2
DEV=${DEVICE:-mps}; LOGS=outputs/nested_logs; mkdir -p "$LOGS"
name=$(basename "$ROOTDIR")
train() { local tag=$1; shift
  [ -f "$ROOTDIR/$tag/oof.csv" ] && { echo "skip $name/$tag"; return 0; }
  [ -d "$ROOTDIR/$tag" ] && mv "$ROOTDIR/$tag" "$ROOTDIR/${tag}_partial_$(date +%s)"
  echo "[$(date +%T)] $name/$tag"
  "$PY" "$TRAINER" --device "$DEV" --out-root "$ROOTDIR" --tag "$tag" "$@" > "$LOGS/${name}_$tag.log" 2>&1 \
    || { echo "FAILED $name/$tag (см. $LOGS/${name}_$tag.log)"; return 1; }; }
train nested_base --folds 0,1,2,3,4 "$@"
for f in 0 1 2 3 4; do for j in 0 1 2 3 4; do
  [ "$f" = "$j" ] && continue
  train "nested_f${f}_j${j}" --folds "$j" --exclude-folds "$f" "$@"
done; done
echo "[$(date +%T)] $name: готово"
