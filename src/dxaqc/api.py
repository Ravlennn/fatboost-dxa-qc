"""Public offline batch API shared by CLI and HTTP adapter."""
from __future__ import annotations

import os
import time
from pathlib import Path

from .inputs import input_directory
from .pipeline import process_folder
from .report import check_report


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
            path_mode='file', exclude=(), overlay_dir=None) -> tuple[list[dict], dict]:
    """One row per input file, including corrupt files; no data leaves this process.

    Raises for batch/configuration failures. Per-file failures are returned as rows.
    Reuse a predictor across calls to amortize model initialization.
    """
    if path_mode not in ('file', 'dir'):
        raise ValueError('Invalid path_mode')
    started = time.perf_counter()
    with input_directory(Path(input_path)) as root:
        if predictor is None:
            predictor = create_predictor(model_dir, device=device, threads=threads)
        rows, stats = process_folder(root, predictor, path_mode=path_mode, exclude=tuple(map(Path, exclude)),
                                     overlay_dir=overlay_dir)
    errors = check_report(rows, path_mode=path_mode)
    if errors:
        raise RuntimeError('Report contract failed: ' + '; '.join(errors[:5]))
    stats['wall_seconds_including_setup'] = round(time.perf_counter() - started, 4)
    stats['predictor'] = predictor.name
    stats['model_id'] = getattr(predictor, 'model_id', None)
    stats['status'] = 'empty' if not rows else ('partial_failure' if stats['failure'] else 'complete')
    return rows, stats
