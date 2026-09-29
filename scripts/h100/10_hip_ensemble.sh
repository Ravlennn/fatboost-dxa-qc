#!/usr/bin/env bash
# Hip CNN: 5 seeds x 5 folds for ConvNeXt-S and ConvNeXt-B at 512 px (one GPU each), then paired comparison.
# Usage: scripts/h100/10_hip_ensemble.sh [small_gpu=0] [base_gpu=1]
. "$(dirname "$0")/common.sh"
G_S=${1:-0}; G_B=${2:-1}
EP=${EPOCHS:-40}
seeds() { local arch=$1 tag=$2 gpu=$3 bs=$4
  for s in 0 1 2 3 4; do
    [ -f "outputs/hip_cnn/${tag}_s$s/oof.csv" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu run_logged "${tag}_s$s" $PY scripts/train_hip_cnn.py --arch $arch --res 512 --epochs $EP --bs $bs \
      --lr 1e-4 --seed $s --tag "${tag}_s$s" --device cuda --batch-pause 0 --cpu-threads 8 --workers 8 || true
  done; }
seeds convnext_small h100_cnxs512 $G_S 16 &
seeds convnext_base h100_cnxb512 $G_B 12 &
wait
run_logged eval_hip_ensemble $PY scripts/eval_hip_ensemble.py outputs/hip_cnn/h100_cnxs512_s* outputs/hip_cnn/h100_cnxb512_s* \
  --out outputs/hip_ensemble_eval
cat outputs/hip_ensemble_eval/table.md
