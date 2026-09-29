"""Run a fixed two-model comparison sequentially on local reconstructed folds."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.nice(10)
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
out = ROOT / 'outputs/hip_cnn/local_pair_lowload_v2'
out.mkdir(parents=True, exist_ok=True)
state = {'pid': os.getpid(), 'started': time.time(), 'runs': []}
for arch, tag in [('convnext_tiny', 'local_convnext384_s0_lowload_v2'), ('densenet121', 'local_densenet384_s0_lowload_v2')]:
    command = [sys.executable, '-u', 'scripts/train_hip_cnn.py', '--arch', arch,
               '--tag', tag, '--res', '384', '--epochs', '30', '--seed', '0',
               '--cpu-threads', '2', '--batch-pause', '1.0']
    if arch == 'convnext_tiny':
        command += ['--resume-from', str(ROOT / 'outputs/hip_cnn/local_convnext384_s0_v1')]
    run = dict(tag=tag, command=command, status='RUNNING', started=time.time())
    state['runs'].append(run)
    (out / 'status.json').write_text(json.dumps(state, indent=2))
    with (out / (tag + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    run.update(status='COMPLETE' if result.returncode == 0 else 'FAILED', returncode=result.returncode, finished=time.time())
    (out / 'status.json').write_text(json.dumps(state, indent=2))
    with (ROOT / 'docs/EXPERIMENTS.md').open('a') as journal:
        journal.write(f"\n### {tag}: {run['status']}\n\nSeed: 0. Артефакты: outputs/hip_cnn/{tag}/.\n")
        metrics = ROOT / 'outputs/hip_cnn' / tag / 'metrics.md'
        if metrics.exists(): journal.write('\n' + metrics.read_text() + '\n')
        else: journal.write(f"Код завершения: {result.returncode}; см. {out.relative_to(ROOT)}/{tag}.log.\n")
    if result.returncode:
        sys.exit(result.returncode)
