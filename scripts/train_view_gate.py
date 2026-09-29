"""Out-of-scope gate: is this image an AP lumbar spine / proximal femur at all?

The router has exactly three classes and must assign one of them to anything it sees.
A forearm, a lateral VFA or a whole-body scan in the closed test would therefore get a
confident but meaningless verdict. This gate is a logistic head on the same frozen
DenseNet121 features, trained label-free on external Arak data:
  in scope      = our DICOMs (no quality labels used) + Arak hips and AP spines
  out of scope  = Arak forearm and lateral/VFA images
The threshold is set so that NO image of ours is rejected, with a margin.

  python scripts/train_view_gate.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.dicom_io import read_dicom  # noqa: E402
from dxaqc.release_features import FrozenDenseNetEncoder  # noqa: E402

DATA = ROOT / 'data/interim/train/Исследования'
OUT = ROOT / 'outputs/view_gate_v1'
ARCHIVE = ROOT / 'data/external/arak_dxa/source.zip'


def arak_raw(z, path):
    """Lateral/VFA images are not written as cleaned PNGs; read and normalise them here."""
    a = np.array(Image.open(io.BytesIO(z.read(path))).convert('L'))
    return 255 - a if np.median(a) > 127 else a


def collect():
    rows = []
    lab = pd.read_csv(ROOT / 'data/interim/image_labels.csv')
    for r in lab.itertuples():  # ours: in scope, labels are not read
        rows.append({'src': 'ours', 'key': f'ours:{r.study}', 'in_scope': 1, 'path': str(DATA / r.path), 'kind': 'dicom'})
    for line in open(ROOT / 'outputs/arak_overlays_v1/annotations.jsonl'):
        a = json.loads(line)
        if a.get('duplicate_of'):
            continue
        view = a['view']
        if view in ('hip', 'spine_ap') and a.get('clean_path'):
            rows.append({'src': 'arak_' + view, 'key': 'arak:' + a['patient_key'], 'in_scope': 1,
                         'path': str(ROOT / a['clean_path']), 'kind': 'png'})
        elif view == 'forearm' and a.get('clean_path'):
            rows.append({'src': 'arak_forearm', 'key': 'arak:' + a['patient_key'], 'in_scope': 0,
                         'path': str(ROOT / a['clean_path']), 'kind': 'png'})
        elif view == 'lateral_vfa':
            rows.append({'src': 'arak_lateral', 'key': 'arak:' + a['patient_key'], 'in_scope': 0,
                         'path': a['archive_path'], 'kind': 'zip'})
    return pd.DataFrame(rows)


def load(row, z):
    if row.kind == 'dicom':
        return read_dicom(Path(row.path)).pixels
    if row.kind == 'zip':
        return arak_raw(z, row.path)
    return cv2.imread(row.path, cv2.IMREAD_GRAYSCALE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--bundle', type=Path, default=ROOT / 'models/release')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    print(df.groupby(['src', 'in_scope']).size().to_string())
    enc = FrozenDenseNetEncoder(args.bundle / 'encoder/densenet121.pth', threads=args.threads, device=args.device)
    z = zipfile.ZipFile(ARCHIVE)
    X = np.stack([enc.embed(load(r, z))[0] for r in df.itertuples()])
    y, g = df.in_scope.values, df.key.values
    # Grouped OOF to see honest separation, then refit on everything for the bundle.
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight='balanced', max_iter=5000))
        oof[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    ours = df.src.eq('ours').values
    # Threshold: below the minimum OOF score of our images, with a safety margin.
    thr = float(min(oof[ours].min() * 0.5, np.percentile(oof[~ours & (y == 0)], 99)))
    res = {'auc_oof': float(auc), 'threshold': thr,
           'our_images_rejected': int((oof[ours] < thr).sum()), 'our_images': int(ours.sum()),
           'our_min_score': float(oof[ours].min()),
           'rejected_out_of_scope': {s: float((oof[(df.src == s).values] < thr).mean())
                                     for s in ('arak_forearm', 'arak_lateral')},
           'kept_in_scope_arak': {s: float((oof[(df.src == s).values] >= thr).mean())
                                  for s in ('arak_hip', 'arak_spine_ap')},
           'n': {s: int((df.src == s).sum()) for s in df.src.unique()}}
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight='balanced', max_iter=5000)).fit(X, y)
    sc, lr = model.named_steps['standardscaler'], model.named_steps['logisticregression']
    head = {'features': 'frozen_densenet121_1024', 'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
            'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]), 'threshold': thr,
            'source': 'scripts/train_view_gate.py', 'auc_oof': float(auc),
            'training': {'in_scope': int(y.sum()), 'out_of_scope': int((y == 0).sum())}}
    (OUT / 'gate_head.json').write_text(json.dumps(head))
    (OUT / 'metrics.json').write_text(json.dumps(res, indent=2))
    pd.DataFrame({'src': df.src, 'in_scope': y, 'oof': oof}).to_csv(OUT / 'oof.csv', index=False)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
