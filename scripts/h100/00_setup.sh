#!/usr/bin/env bash
# Environment + ImageNet weights + data preparation. Needs internet once.
. "$(dirname "$0")/common.sh"
command -v uv >/dev/null || { echo "install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 2; }
uv sync --frozen
$PY - <<'PY'
import torch, torchvision
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())
assert torch.cuda.is_available(), 'CUDA not available: check driver / wheel'
for fn, w in [(torchvision.models.convnext_tiny, 'ConvNeXt_Tiny_Weights'), (torchvision.models.convnext_small, 'ConvNeXt_Small_Weights'),
              (torchvision.models.convnext_base, 'ConvNeXt_Base_Weights')]:
    fn(weights=getattr(torchvision.models, w).IMAGENET1K_V1)  # caches into $TORCH_HOME
print('ImageNet weights cached in', torch.hub.get_dir())
PY
# Required inputs (not in git): see scripts/h100/README.md
for p in data/interim/image_labels.csv data/interim/folds.csv "data/interim/train/Исследования" data/external/arak_dxa/source.zip; do
  [ -e "$p" ] || { echo "MISSING $p — copy it from the Mac (README)"; exit 2; }
done
# Cleaned Arak PNGs are not shipped in the data archive (they are derived): rebuild if absent.
n_clean=$(ls data/external/arak_dxa/clean 2>/dev/null | wc -l)
[ "$n_clean" -ge 3700 ] || run_logged parse_arak $PY scripts/parse_arak_overlays.py
[ -f outputs/landmarks_v1/best.pt ] || run_logged landmarks_v1 $PY scripts/train_landmarks.py --device cuda --batch 32
[ -f outputs/landmarks_v1/ours/predictions.json ] || run_logged predict_v1 $PY scripts/predict_landmarks_ours.py --device cuda
run_logged ssl_manifest $PY scripts/build_ssl_manifest.py
echo "setup OK"
