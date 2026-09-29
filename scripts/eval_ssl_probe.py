"""Cheap gate for an SSL encoder: frozen embeddings + logistic probe on the release folds.

Compares one or more encoders (ImageNet init vs DINO checkpoints) for quality and
each reason; C and threshold chosen on inner study-grouped folds. Run this before
spending GPU hours on full fine-tuning.

  python scripts/eval_ssl_probe.py --ckpt imagenet:convnext_small outputs/ssl_v1/encoder_ep0200.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'models/torch_home'))
from dxaqc.hip_backbones import build_hip_model  # noqa: E402

TASKS = {'quality_spine': ('spine', 'quality'), 'quality_hip': ('hip', 'quality'),
         'spine_scan_range': ('spine', 'v_layout'), 'spine_axis_tilt': ('spine', 'v_axis'),
         'spine_artifact': ('spine', 'v_artifact'), 'hip_positioning': ('hip', 'v_rotation'),
         'hip_roi_margins': ('hip', 'v_roi')}
CS = (0.001, 0.01, 0.1)


def load_encoder(spec, device):
    if spec.startswith('imagenet:'):
        arch = spec.split(':', 1)[1]
        m = build_hip_model(arch, 1, pretrained=True)
    else:
        ck = torch.load(spec, map_location='cpu', weights_only=False)
        m = build_hip_model(ck['arch'], 1, pretrained=False)
        m.features.load_state_dict(ck['features'])
    return m.features.eval().to(device)


@torch.no_grad()
def embed(features, imgs, device, size):
    out = []
    for img in imgs:
        views = []
        for flip in (False, True):
            a = img[:, ::-1] if flip else img
            h, w = a.shape
            s = max(h, w)
            c = np.zeros((s, s), np.float32)
            c[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a / 255.0
            c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
            views.append((c - 0.449) / 0.226)
        x = torch.from_numpy(np.stack(views))[:, None].repeat(1, 3, 1, 1).to(device)
        out.append(features(x).mean((-2, -1)).mean(0).cpu().numpy())
    return np.stack(out)


def nested(X, y, groups, folds):
    oof, dec = np.zeros(len(y)), np.zeros(len(y), bool)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        best = None
        for C in CS:
            inner = np.zeros(tr.sum())
            for a, b in GroupKFold(3).split(X[tr], y[tr], groups[tr]):
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=5000))
                inner[b] = mdl.fit(X[tr][a], y[tr][a]).predict_proba(X[tr][b])[:, 1] if y[tr][a].any() else 0
            ap = average_precision_score(y[tr], inner)
            if best is None or ap > best[0]:
                best = (ap, C, inner)
        _, C, inner = best
        thr = max(np.unique(inner), key=lambda t: f1_score(y[tr], inner >= t, zero_division=0))
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=5000)).fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
        dec[te] = oof[te] >= thr
    return oof, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    ap.add_argument('--out', default=str(ROOT / 'outputs/ssl_probe'))
    args = ap.parse_args()
    lab = pd.read_csv(ROOT / 'data/interim/image_labels.csv').merge(
        pd.read_csv(ROOT / 'data/interim/folds.csv'), on=['study', 'image_uid'])
    imgs = []
    for r in lab.itertuples():
        a = cv2.imread(str(ROOT / 'data/interim/ssl_png' / f'{hashlib.sha1(r.image_uid.encode()).hexdigest()[:16]}.png'), 0)
        imgs.append(a[:, ::-1].copy() if r.region == 'hip_left' else a)  # hip: canonical side, as in hip CNN
    results = {}
    for spec in args.ckpt:
        X = embed(load_encoder(spec, args.device), imgs, args.device, args.size)
        res = {}
        for task, (sub, col) in TASKS.items():
            m = (lab.region.eq('spine') if sub == 'spine' else lab.region.str.startswith('hip')) & lab[col].notna()
            y = lab.loc[m, col].astype(int).values
            s, d = nested(X[m.values], y, lab.loc[m, 'study'].values, lab.loc[m, 'fold'].values)
            res[task] = {'auc': round(roc_auc_score(y, s), 4), 'ap': round(average_precision_score(y, s), 4),
                         'f1': round(f1_score(y, d), 4), 'pos': int(y.sum()), 'n': int(len(y))}
        results[spec] = res
        print(spec, json.dumps({k: (v['auc'], v['ap']) for k, v in res.items()}), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    lines = ['| encoder | ' + ' | '.join(TASKS) + ' |', '|---' * (len(TASKS) + 1) + '|']
    for spec, res in results.items():
        lines.append(f'| {Path(spec).name} | ' + ' | '.join(f"{res[t]['auc']:.3f}/{res[t]['ap']:.3f}" for t in TASKS) + ' |')
    (out / 'table.md').write_text('AUC/AP, frozen embeddings, nested C + threshold\n\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
