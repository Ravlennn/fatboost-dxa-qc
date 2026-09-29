"""Add the landmark model and landmark reason heads to a release bundle.

Heads: spine_axis_tilt and spine_scan_range (landmark-only variant selected in
docs/METHODS_AND_MODELS.md). Refit on all labelled spine images; threshold = midpoint of
adjacent pooled-OOF scores at the F1-optimal partition (optimistic, like the
release hip thresholds).

  .venv/bin/python scripts/export_landmark_bundle.py --bundle models/release
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from dxaqc.bundle import sha256, verify_bundle  # noqa: E402
import run_landmark_fusion as F  # noqa: E402

# reason code -> (label, fusion task whose landmark-only variant is exported)
HEADS = {'spine_axis_tilt': ('v_axis', 'spine_axis_tilt'), 'spine_scan_range': ('v_layout', 'spine_scan_range_iliac')}


def midpoint_threshold(y, s):
    order = np.sort(np.unique(s))
    t = F.best_threshold(y, s)
    below = order[order < t]
    return float((t + below.max()) / 2) if len(below) else float(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, default=ROOT / 'models/release')
    ap.add_argument('--ckpt', type=Path, default=ROOT / 'outputs/landmarks_v1/best.pt')
    args = ap.parse_args()
    manifest = verify_bundle(args.bundle)

    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    pred = pd.DataFrame(json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')))
    df = oof.merge(pred[['image_uid', 'study', 'measurements']], on=['study', 'image_uid'], validate='1:1')
    fus = pd.read_csv(ROOT / 'outputs/landmark_fusion_v1/oof.csv')
    spine = df[df['region'].eq('spine')]
    X = F.features(spine)
    heads = {}
    for code, (lab, task) in HEADS.items():
        cols = F.TASKS[task][3]
        d = spine[spine[lab].notna()]
        y = d[lab].astype(int).values
        m = F.model().fit(X.loc[d.index, cols].values, y)
        sc, lr = m.named_steps['standardscaler'], m.named_steps['logisticregression']
        o = fus[fus['task'].eq(task)]
        heads[code] = {'task': task, 'features': cols, 'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
                       'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
                       'threshold': midpoint_threshold(o['y'].values, o['s_landmarks'].values),
                       'n': int(len(y)), 'pos': int(y.sum()), 'C': 0.3, 'class_weight': 'balanced'}
    dst = args.bundle / 'landmarks'
    dst.mkdir(exist_ok=True)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    torch.save({'model': ck['model'], 'points': ck['points'], 'epoch': ck['epoch']}, dst / 'landmarks_v1.pt')
    old = json.loads((dst / 'heads.json').read_text()) if (dst / 'heads.json').is_file() else {}
    (dst / 'heads.json').write_text(json.dumps({
        **{k: old[k] for k in ('hip_heads', 'spine_artifact_head') if k in old},  # keep export_lt_bundle.py heads
        'heads': heads, 'spine_quality_policy': old.get('spine_quality_policy', 'or'),
        'source': 'scripts/run_landmark_fusion.py; docs/METHODS_AND_MODELS.md',
        'checkpoint_sha256': sha256(dst / 'landmarks_v1.pt')}, indent=2))
    backup = args.bundle / 'manifest.pre_landmarks.json'
    if not backup.exists():
        shutil.copy(args.bundle / 'manifest.json', backup)
    manifest['landmarks'] = {'checkpoint': 'landmarks/landmarks_v1.pt', 'heads': 'landmarks/heads.json',
                             'input': 256, 'arch': 'convnext_tiny_fpn_heatmap'}
    for rel in ('landmarks/landmarks_v1.pt', 'landmarks/heads.json'):
        manifest['files'][rel] = sha256(args.bundle / rel)
    manifest['model_id'] = manifest['model_id'].split('+')[0] + '+landmarks_v1'
    manifest['limitations'] = [x for x in manifest['limitations']
                               if 'physical margin' not in x and not x.startswith('Landmark model')] + [
        'Landmark model trained on Arak GE Lunar overlays (weak labels, CC BY-NC 4.0 mirror); '
        'spine axis/scan-range heads fitted on 99 local images, thresholds tuned on pooled OOF (optimistic)']
    (args.bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    verify_bundle(args.bundle)
    print(json.dumps({k: {'threshold': v['threshold'], 'pos': v['pos']} for k, v in heads.items()}))


if __name__ == '__main__':
    main()
