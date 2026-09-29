#!/usr/bin/env bash
# Measurements on the server (no training): environment, bundle, tests, the main metric table,
# overfitting checks and a timed CLI run over all 499 DICOMs on GPU and CPU.
#   bash h100_kit/measure.sh            -> everything (~20-30 min)
#   bash h100_kit/measure.sh --quick    -> skip the CPU CLI run
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export TORCH_HOME="$PWD/models/torch_home" PYTHONUNBUFFERED=1
OUT=outputs/measure_$(date +%Y%m%d_%H%M); mkdir -p "$OUT"
step() { echo; echo "=== $1"; }
step "1. Environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not found"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'gpus',torch.cuda.device_count())"
step "2. Model bundle integrity (SHA-256)"
.venv/bin/dxaqc doctor --model-dir models/release | tee "$OUT/doctor.json"
step "3. Unit tests"
$PY -m unittest discover -s tests 2>&1 | tail -3 | tee "$OUT/tests.txt"
step "4. Main metric table (nested protocol, reproduces docs/FINAL_VALIDATION.md)"
$PY scripts/evaluate_final.py 2>/dev/null | tee "$OUT/final_table.txt" | head -14
step "5. Overfitting / leakage checks (~3 min)"
$PY scripts/check_overfit.py 2>/dev/null | tee "$OUT/overfit.txt"
run_cli() {  # $1 = device, $2 = threads
  local d="$OUT/cli_$1"; mkdir -p "$d"
  /usr/bin/time -p .venv/bin/dxaqc -i "data/interim/train/Исследования" -o "$d/result.csv" --device "$1" --threads "$2" \
    --details --scores --overlays "$d/overlays.zip" > "$d/stdout.txt" 2> "$d/time.txt"
  $PY - "$d" <<'PY'
import json, sys, pandas as pd
d = sys.argv[1]; s = json.load(open(f'{d}/result.summary.json')); r = pd.read_csv(f'{d}/result.csv')
per = r[r.processing_status.eq('Success')].groupby('study_uid').time_of_processing.sum()
print(f"files {s['files']}  success {s['success']} ({s['success_rate']:.0%})  failure {s['failure']}  overlays {s['overlays']}")
print(f"per study: median {per.median():.2f} s, p95 {per.quantile(.95):.2f} s, max {per.max():.2f} s (TZ limit 180 s)")
print(f"total incl. model loading: {s['wall_seconds_including_setup']:.0f} s; model {s['model_id']}")
PY
}
step "6. CLI on GPU, all 499 DICOMs"
run_cli cuda 8 | tee "$OUT/cli_cuda.txt"
if [ "${1:-}" != "--quick" ]; then
  step "7. CLI on CPU (8 threads), all 499 DICOMs"
  run_cli cpu 8 | tee "$OUT/cli_cpu.txt"
fi
echo; echo "Saved to $OUT (send it back together with the training results)."
