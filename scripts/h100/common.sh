#!/usr/bin/env bash
# Shared settings for the H100 runners. Source it: . scripts/h100/common.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export ROOT="$PWD"
export PY="$ROOT/.venv/bin/python"
export TORCH_HOME="$ROOT/models/torch_home"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export LOGS="$ROOT/outputs/h100_logs"
mkdir -p "$LOGS"
# run_logged NAME cmd... : tee to outputs/h100_logs/NAME.log, keep going on failure of other jobs
run_logged() { local name=$1; shift; echo "[$(date +%T)] start $name: $*"; "$@" > "$LOGS/$name.log" 2>&1 && echo "[$(date +%T)] done $name" || { echo "[$(date +%T)] FAILED $name (see $LOGS/$name.log)"; return 1; }; }
