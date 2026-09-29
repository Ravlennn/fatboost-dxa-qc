#!/usr/bin/env bash
# Artifact segmenter on synthetic metal over Arak spines; zero-shot AUC on our 99 spines.
# Usage: scripts/h100/30_artifacts.sh [gpu=1]
#   INIT=outputs/ssl_v1/encoder_last.pt  start from the DXA encoder
#   BG_OURS=1                            add our spine images as label-blind backgrounds
#   OUT=outputs/artifact_seg_h100_bgours  separate output dir per variant
. "$(dirname "$0")/common.sh"
G=${1:-1}; ARCH=${ARCH:-convnext_small}
EXTRA=(); [ -n "${INIT:-}" ] && EXTRA=(--init-encoder "$INIT"); [ -n "${BG_OURS:-}" ] && EXTRA+=(--bg-ours)
CUDA_VISIBLE_DEVICES=$G run_logged artifact_seg $PY scripts/train_artifact_seg.py --arch $ARCH --epochs ${EPOCHS:-60} \
  --per-epoch 4000 --batch 32 --device cuda --workers 12 --out "${OUT:-outputs/artifact_seg_h100}" "${EXTRA[@]}"
cat "${OUT:-outputs/artifact_seg_h100}/metrics.json"
