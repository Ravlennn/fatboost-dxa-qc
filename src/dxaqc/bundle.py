"""Portable, integrity-checked model bundles; inference never downloads weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f'Invalid bundle path: {relative}')
    return path


def verify_bundle(root: Path) -> dict:
    root = Path(root)
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Model bundle missing: {manifest_path}. See README: prepare_release.py')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported model manifest schema')
    files = manifest.get('files', {})
    if not files:
        raise ValueError('Empty model bundle')
    for relative, expected in files.items():
        path = bundle_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Model file missing or SHA-256 mismatch: {relative}')
    references = [manifest['encoder'], manifest['heads']]
    runs = manifest.get('hip_runs', [])
    if not runs:
        raise ValueError('No hip models in bundle')
    for run in runs:
        if run['arch'] not in ('convnext_tiny', 'convnext_small', 'convnext_base', 'densenet121', 'resnet50'):
            raise ValueError('Unknown hip architecture')
        if len(run['checkpoints']) != 5 or len(set(run['checkpoints'])) != 5:
            raise ValueError('Each hip run must contain five distinct fold checkpoints')
        if run.get('crop', 0) or run.get('roi_file'):
            raise ValueError('ROI preprocessing is not supported by this release')
        references.extend(run['checkpoints'])
    if 'view_gate' in manifest:
        references.append(manifest['view_gate'])
    if 'landmarks' in manifest:
        references.extend([manifest['landmarks']['checkpoint'], manifest['landmarks']['heads']])
    if set(references) - files.keys():
        raise ValueError('Unhashed model reference in manifest')
    for name in ('quality', 'v_rotation', 'v_roi'):
        value = manifest['hip_thresholds'][name]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f'Invalid threshold: {name}')
    return manifest
