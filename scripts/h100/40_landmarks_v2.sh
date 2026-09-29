#!/usr/bin/env bash
# Landmarks v2 (lesser trochanter, Th12, iliac crests). Uses the export of labeler_v2.html
#   outputs/landmark_labeling/landmarks_manual_v2.json if present, else automatic LT labels.
. "$(dirname "$0")/common.sh"
G=${1:-0}; L=${LABELS:-outputs/landmark_labeling/landmarks_manual_v2.json}
if [ ! -f "$L" ]; then  # fall back to label-free automatic lesser-trochanter points (hips only)
  L=outputs/lt_auto_v1/landmarks_auto_v2.json
  [ -f "$L" ] || run_logged auto_lt $PY scripts/auto_label_lt.py
  echo "No manual labels: using automatic LT labels $L"
fi
CUDA_VISIBLE_DEVICES=$G run_logged landmarks_v2 $PY scripts/train_landmarks_v2.py --labels "$L" --device cuda --batch 32 \
  --workers 8 --folds 0,1,2,3,4,all --out outputs/landmarks_v2
run_logged eval_landmarks_v2 $PY scripts/eval_landmarks_v2.py --oof outputs/landmarks_v2/oof_points.json --labels "$L"
cat outputs/landmarks_v2/eval/table.md
