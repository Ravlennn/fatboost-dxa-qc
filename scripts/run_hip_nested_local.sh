#!/usr/bin/env bash
# Fold isolation for the hip CNN: 5 base models (without the outer fold) + 20 inner ones (without both).
# convnext_tiny at 384 px takes ~3 min per fold on MPS, so 25 models is ~1.5 h — no H100 needed.
# Usage: [AUG=photometric] [LOSS=weighted_bce] [EPOCHS=30] [DEVICE=mps] [SUF=_tag] scripts/run_hip_nested_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"; export TORCH_HOME="$PWD/models/torch_home" PYTHONUNBUFFERED=1
AUG=${AUG:-photometric}; EP=${EPOCHS:-30}; DEV=${DEVICE:-mps}; RES=${RES:-384}; LOSS=${LOSS:-weighted_bce}
SUF=${SUF:-}                      # tag suffix, e.g. SUF=_full for a second recipe
LOGS=outputs/aug_logs; mkdir -p "$LOGS"
train() { local tag=$1; shift
  [ -f "outputs/hip_cnn/$tag/oof.csv" ] && { echo "skip $tag"; return 0; }
  echo "[$(date +%T)] $tag"
  "$PY" scripts/train_hip_cnn.py --arch convnext_tiny --res $RES --epochs $EP --bs 16 --lr 1e-4 --seed 0 \
    --aug $AUG --loss $LOSS --device $DEV --batch-pause 0 --cpu-threads 8 --tag "$tag" "$@" > "$LOGS/$tag.log" 2>&1 \
    || { echo "FAILED $tag (см. $LOGS/$tag.log)"; return 1; }; }
train "nested_base$SUF" --folds 0,1,2,3,4
for f in 0 1 2 3 4; do for j in 0 1 2 3 4; do
  [ "$f" = "$j" ] && continue
  train "nested_f${f}_j${j}$SUF" --folds "$j" --exclude-folds "$f"
done; done
"$PY" scripts/eval_hip_nested.py ${SUF:+--suffix "$SUF"}
