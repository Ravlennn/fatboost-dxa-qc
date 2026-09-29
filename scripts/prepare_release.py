"""Export verified local experiments into a portable, self-contained model directory."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.bundle import sha256, verify_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'models/release')
    parser.add_argument('--selection', type=Path, default=ROOT / 'outputs/release_ensemble_v1/selection.json')
    parser.add_argument('--heads', type=Path, default=ROOT / 'outputs/release_loss_v1/final_heads.json')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection['folds'] != [0, 1, 2, 3, 4] or len(set(selection['weights'])) != 1:
        raise ValueError('Release supports complete, uniformly weighted runs only')
    if args.output.exists():
        raise ValueError('Output already exists; use a new destination to preserve provenance')
    # Validate sources before creating the bundle.
    sources = {'encoder/densenet121.pth': ROOT / 'models/torch_home/hub/checkpoints/densenet121-a639ec97.pth', 'heads.json': args.heads}
    runs = []
    for tag in selection['tags']:
        source = ROOT / 'outputs/hip_cnn' / tag
        config = json.loads((source / 'config.json').read_text())
        if config.get('crop') or config.get('roi_file'):
            raise ValueError('Unsupported historical preprocessing')
        checkpoints = []
        for fold in selection['folds']:
            name = f'hip/{tag}/fold{fold}.pt'
            sources[name] = source / f'fold{fold}.pt'
            checkpoints.append(name)
        runs.append({'tag': tag, 'arch': config['arch'], 'res': config['res'], 'checkpoints': checkpoints})
    if any(not path.is_file() for path in sources.values()):
        raise FileNotFoundError('Incomplete source model artifacts')
    heads = json.loads(args.heads.read_text())
    if sha256(sources['encoder/densenet121.pth']) != heads['checkpoint_sha256']:
        raise ValueError('Encoder/head checksum mismatch')
    for relative, source in sources.items():
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        'schema_version': 1, 'model_id': 'dxaqc-development-2026-09-19',
        'encoder': 'encoder/densenet121.pth', 'heads': 'heads.json', 'hip_runs': runs,
        'hip_thresholds': selection.get('stable_thresholds', selection['thresholds']),
        'hip_threshold_policy': selection.get('stable_threshold_policy', 'Exact OOF threshold'),
        'preprocessing': {'router_spine': heads['preprocess'], 'hip': 'normalized uint8 restored to [0,252]; mirror left; square pad; PIL bilinear384; RGB; ImageNet normalization'},
        'files': {name: sha256(args.output / name) for name in sources},
        'provenance': {'selection_sha256': sha256(args.selection), 'heads_sha256': sha256(args.heads),
                       'labels_sha256': sha256(ROOT / 'data/interim/image_labels.csv'), 'folds_sha256': sha256(ROOT / 'data/interim/folds.csv')},
        'limitations': ['Development selection on repeatedly used local study folds; no independent test',
                       'Hip thresholds tuned on pooled OOF; F1 at these thresholds is optimistic',
                       'Rare cause heads are exploratory; quality_unspecified means no confident cause',
                       'No physical margin measurement or projection classifier is claimed'],
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    verify_bundle(args.output)
    print(f'Prepared {args.output}: {len(sources)} verified files')


if __name__ == '__main__':
    main()
