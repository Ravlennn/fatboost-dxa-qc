#!/usr/bin/env bash
# Run on the server after the training: packs results (tables, OOF, logs, small encoders) to send back.
#   bash h100_kit/collect_results.sh            -> without large checkpoints (~tens of MB)
#   bash h100_kit/collect_results.sh --weights  -> also *.pt checkpoints (several GB)
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date +%Y%m%d_%H%M); OUT="h100_results_$STAMP.tar.gz"
EXC=(--exclude='*.pt'); [ "${1:-}" = "--weights" ] && EXC=()
PATHS=()
for p in outputs/h100_logs outputs/hip_cnn/h100_* outputs/hip_ensemble_eval outputs/hip_ensemble_eval_ssl \
         outputs/ssl_v1 outputs/ssl_probe outputs/artifact_seg_h100 outputs/artifact_seg_h100_bgours outputs/landmarks_v2; do
  [ -e "$p" ] && PATHS+=("$p")
done
[ ${#PATHS[@]} -gt 0 ] || { echo "nothing to collect yet"; exit 2; }
# keep the final DINO encoder even without --weights: it is the main SSL artifact
extra=(); [ -f outputs/ssl_v1/encoder_last.pt ] && [ "${1:-}" != "--weights" ] && extra=(outputs/ssl_v1/encoder_last.pt)
TAR="${OUT%.gz}"
tar -cf "$TAR" "${EXC[@]}" "${PATHS[@]}"
[ ${#extra[@]} -gt 0 ] && tar -rf "$TAR" "${extra[@]}"  # appended after the *.pt exclusion
gzip -f "$TAR"
ls -lh "$OUT"; echo "send $OUT back (scp/rsync) and unpack in the repo root on the Mac"
