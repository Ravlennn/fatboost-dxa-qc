#!/usr/bin/env bash
# DINO self-supervised pretraining on ~4.1k DXA -> probe gate -> hip fine-tuning with the DXA encoder.
# Usage: scripts/h100/20_ssl.sh [gpu=0]
. "$(dirname "$0")/common.sh"
G=${1:-0}; ARCH=${ARCH:-convnext_small}; EP=${SSL_EPOCHS:-200}
[ -f outputs/ssl_v1/encoder_last.pt ] || CUDA_VISIBLE_DEVICES=$G run_logged ssl_dino $PY scripts/train_ssl_dxa.py \
  --arch $ARCH --epochs $EP --batch 128 --device cuda --workers 12 --out outputs/ssl_v1
# Gate: frozen-probe comparison of ImageNet vs DINO checkpoints (minutes).
CKS=$(ls outputs/ssl_v1/encoder_ep*.pt | awk 'NR%2==0' | tr '\n' ' ')
CUDA_VISIBLE_DEVICES=$G run_logged ssl_probe $PY scripts/eval_ssl_probe.py --ckpt imagenet:$ARCH $CKS outputs/ssl_v1/encoder_last.pt --device cuda
cat outputs/ssl_probe/table.md
# Full fine-tune with the DXA encoder, same seeds/epochs as 10_hip_ensemble for a paired comparison.
for s in 0 1 2 3 4; do
  [ -f "outputs/hip_cnn/h100_cnxs512ssl_s$s/oof.csv" ] && continue
  CUDA_VISIBLE_DEVICES=$G run_logged "h100_cnxs512ssl_s$s" $PY scripts/train_hip_cnn.py --arch $ARCH --res 512 --epochs ${EPOCHS:-40} \
    --bs 16 --lr 1e-4 --seed $s --tag "h100_cnxs512ssl_s$s" --device cuda --batch-pause 0 --cpu-threads 8 --workers 8 \
    --init-encoder outputs/ssl_v1/encoder_last.pt || true
done
run_logged eval_hip_ssl $PY scripts/eval_hip_ensemble.py outputs/hip_cnn/h100_cnxs512_s* outputs/hip_cnn/h100_cnxs512ssl_s* \
  --out outputs/hip_ensemble_eval_ssl
cat outputs/hip_ensemble_eval_ssl/table.md
