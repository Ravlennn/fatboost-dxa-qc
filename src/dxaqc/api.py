"""Public offline batch API shared by CLI and HTTP adapter."""
from __future__ import annotations

import os
import math
import re
import logging
import time
from pathlib import Path

from .inputs import input_directory
from .pipeline import process_folder
from .report import make_row, check_report

log = logging.getLogger(__name__)


def default_model_dir() -> Path:
    return Path(os.environ.get('DXAQC_MODEL_DIR', Path(__file__).resolve().parents[2] / 'models' / 'release'))


def create_predictor(model_dir=None, *, device='cpu', threads=2):
    if not 1 <= threads <= 8:
        raise ValueError('threads must be between 1 and 8')
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # A caller may already have initialized Torch's shared thread pool.
    try:
        # Where NNPACK cannot run (ARM hosts emulating amd64) Torch retries it on every convolution and
        # logs «Could not initialize NNPACK» each time. It is not used there anyway, so switching it off
        # changes no result and keeps the log readable; where it is available it stays on.
        if not torch.backends.nnpack.is_available():
            torch.backends.nnpack.set_flags(False)
    except (AttributeError, RuntimeError):
        pass
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Unsupported device')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable')
    from .release_predictor import ReleasePredictor
    return ReleasePredictor(Path(model_dir) if model_dir else default_model_dir(), device=device, threads=threads)


def analyze(input_path, *, predictor=None, model_dir=None, device='cpu', threads=2,
            path_mode='file', exclude=(), overlay_dir=None, on_result=None, on_start=None, sr_dir=None) -> tuple[list[dict], dict]:
    """One row per input file, including corrupt files; no data leaves this process.

    Raises for batch/configuration failures. Per-file failures are returned as rows.
    Reuse a predictor across calls to amortize model initialization.
    """
    if path_mode not in ('file', 'dir'):
        raise ValueError('Invalid path_mode')
    started = time.perf_counter()
    skipped: list[dict] = []
    with input_directory(Path(input_path), skipped=skipped) as root:
        if predictor is None:
            predictor = create_predictor(model_dir, device=device, threads=threads)
        rows, stats = process_folder(root, predictor, path_mode=path_mode, exclude=tuple(map(Path, exclude)),
                                     overlay_dir=overlay_dir, on_result=on_result, on_start=on_start, sr_dir=sr_dir)
    # archive members that could not be extracted are inputs we failed to process: report them
    for item in skipped:
        if item.get('kept'):
            continue
        rows.append(make_row(path_to_study=item['path'], status='Failure', seconds=0.0,
                             error=f"ZIP member not processed: {item['reason']}"))
        stats['failure'] += 1
    stats['zip_skipped'] = skipped
    errors = check_report(rows, path_mode=path_mode)
    if errors:
        # never lose the whole batch over one row: a row that breaks the contract becomes a
        # well-formed Failure row carrying the reason, and the report is still written
        log.error('report contract violations, rows turned into Failure: %s', '; '.join(errors[:20]))
        bad = {int(m.group(1)) - 1 for e in errors if (m := re.match(r'строка (\d+):', e))}
        seen: dict[str, int] = {}
        for i, r in enumerate(rows):
            path = r.get('path_to_study') or f'unknown_{i + 1}'
            seen[path] = seen.get(path, 0) + 1
            if seen[path] > 1:                          # duplicate logical path: make it unique
                path = f'{path} #{seen[path]}'
            if i in bad or path != r.get('path_to_study'):
                t = r.get('time_of_processing')
                rows[i] = make_row(path_to_study=path, study_uid=str(r.get('study_uid') or ''),
                                   image_uid=str(r.get('image_uid') or ''), status='Failure',
                                   seconds=float(t) if isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 else 0.0,
                                   error='report contract: ' + '; '.join(e for e in errors if e.startswith(f'строка {i + 1}:'))[:500])
        stats['contract_errors'] = errors[:50]
        stats['success'] = sum(r.get('processing_status') == 'Success' for r in rows)
        stats['failure'] = sum(r.get('processing_status') == 'Failure' for r in rows)
        remaining = check_report(rows, path_mode=path_mode)
        if remaining:
            raise RuntimeError('Report contract failed: ' + '; '.join(remaining[:5]))
    stats['wall_seconds_including_setup'] = round(time.perf_counter() - started, 4)
    stats['predictor'] = predictor.name
    stats['model_id'] = getattr(predictor, 'model_id', None)
    stats['status'] = 'empty' if not rows else ('partial_failure' if stats['failure'] else 'complete')
    return rows, stats
