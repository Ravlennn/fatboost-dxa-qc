#!/usr/bin/env bash
# Whole plan on 2xH100. GPU0: SSL -> SSL-init hip + landmarks v2; GPU1: hip ensemble (S and B run concurrently on GPU1, ~50 GB) -> artifacts.
# Total ~10-14 h. Safe to re-run: finished runs (oof.csv / checkpoints) are skipped.
. "$(dirname "$0")/common.sh"
bash scripts/h100/00_setup.sh
( bash scripts/h100/20_ssl.sh 0; bash scripts/h100/40_landmarks_v2.sh 0 || true ) \
  > "$LOGS/gpu0.log" 2>&1 &
( bash scripts/h100/10_hip_ensemble.sh 1 1; bash scripts/h100/30_artifacts.sh 1; BG_OURS=1 OUT=outputs/artifact_seg_h100_bgours bash scripts/h100/30_artifacts.sh 1 ) > "$LOGS/gpu1.log" 2>&1 &
wait
echo "ALL DONE. Summaries:"; for f in outputs/hip_ensemble_eval/table.md outputs/hip_ensemble_eval_ssl/table.md outputs/ssl_probe/table.md \
  outputs/artifact_seg_h100/metrics.json outputs/artifact_seg_h100_bgours/metrics.json outputs/landmarks_v2/eval/table.md; do [ -f "$f" ] && { echo "== $f"; cat "$f"; }; done
