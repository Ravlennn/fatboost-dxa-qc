"""Calibration of the geometric measurements: apply a known rotation / crop, measure it back.

No labels are used. For each image the landmark model measures the spine tilt and the margins;
we then rotate the image by a known angle and crop a known number of millimetres and check what
the model reports. Gives the accuracy of the numbers we print on the overlay and in DICOM SR.

  python scripts/calibrate_geometry.py --device mps --n 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.dicom_io import read_dicom  # noqa: E402
from dxaqc.landmarks import LandmarkPredictor, measurements  # noqa: E402

DATA = ROOT / 'data/interim/train/Исследования'
OUT = ROOT / 'outputs/calibration_v1'
MM = 0.6
ANGLES = (-8, -5, -3, 0, 3, 5, 8)
CROPS_MM = (0, 6, 12, 18)


def rotate(img, deg):
    h, w = img.shape
    # Our tilt is positive when the column leans right (y grows down), which matches +deg here.
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--ckpt', default=str(ROOT / 'models/release/landmarks/landmarks_v1.pt'))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pred = LandmarkPredictor(args.ckpt, args.device)
    lab = pd.read_csv(ROOT / 'data/interim/image_labels.csv')
    rng = np.random.default_rng(0)
    rows = []
    for region, tag in (('spine', 'spine'), ('hip_right', 'hip'), ('hip_left', 'hip')):
        sub = lab[lab.region.eq(region)]
        take = sub.iloc[rng.choice(len(sub), min(args.n // (2 if tag == 'hip' else 1), len(sub)), replace=False)]
        for r in take.itertuples():
            img = read_dicom(DATA / r.path).pixels
            base = measurements(pred.predict(img, r.region)['points'], r.region, img.shape, img)
            if tag == 'spine':
                for a in ANGLES:  # known rotation -> measured tilt
                    im = rotate(img, a)
                    m = measurements(pred.predict(im, r.region)['points'], r.region, im.shape, im)
                    rows.append({'kind': 'tilt', 'uid': r.image_uid, 'applied': a,
                                 'measured': m['spine_tilt_deg'] - base['spine_tilt_deg'],
                                 'absolute': m['spine_tilt_deg']})
                for c in CROPS_MM:  # known crop from the top -> measured margin above L1
                    px = int(round(c / MM))
                    im = img[px:] if px else img
                    m = measurements(pred.predict(im, r.region)['points'], r.region, im.shape, im)
                    rows.append({'kind': 'top_margin_mm', 'uid': r.image_uid, 'applied': -c,
                                 'measured': m['top_margin_mm'] - base['top_margin_mm'], 'absolute': m['top_margin_mm']})
            else:
                for c in CROPS_MM:  # known crop from the bottom -> measured ROI margin below
                    px = int(round(c / MM))
                    im = img[:img.shape[0] - px] if px else img
                    m = measurements(pred.predict(im, r.region)['points'], r.region, im.shape, im)
                    rows.append({'kind': 'roi_bottom_mm', 'uid': r.image_uid, 'applied': -c,
                                 'measured': m['roi_bottom_mm'] - base['roi_bottom_mm'], 'absolute': m['roi_bottom_mm']})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'measurements.csv', index=False)
    res, L = {}, ['| Измерение | n | Единицы | Смещение (bias) | Средняя ошибка | 95% ошибок в пределах | Наклон регрессии |',
                  '|---|---:|---|---:|---:|---:|---:|']
    for kind, g in df.groupby('kind'):
        err = g.measured - g.applied
        slope = np.polyfit(g.applied, g.measured, 1)[0]
        unit = 'град.' if kind == 'tilt' else 'мм'
        res[kind] = {'n': len(g), 'bias': float(err.mean()), 'mae': float(err.abs().mean()),
                     'p95_abs_err': float(err.abs().quantile(0.95)), 'slope': float(slope),
                     'r2': float(np.corrcoef(g.applied, g.measured)[0, 1] ** 2)}
        L.append(f"| {kind} | {len(g)} | {unit} | {err.mean():+.2f} | {err.abs().mean():.2f} | "
                 f"{err.abs().quantile(0.95):.2f} | {slope:.3f} |")
    (OUT / 'calibration.json').write_text(json.dumps(res, indent=2))
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
