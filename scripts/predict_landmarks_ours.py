"""Run the Arak-trained landmark model on our labelled DICOMs.

Writes per-image points, confidences and TZ-style measurements, plus montages
split by quality label for a visual transfer check.

  .venv/bin/python scripts/predict_landmarks_ours.py --ckpt outputs/landmarks_v1/best.pt
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
def draw(img, points, region, text):
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    p = {k: tuple(int(round(c)) for c in xy) for k, xy in points.items()}
    if region == 'spine':
        seq = [p[k] for k in ('L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom')]
        for a, b in zip(seq[:-1], seq[1:]):
            cv2.line(v, a, b, (0, 255, 255), 1)
        for q in seq:
            cv2.line(v, (q[0] - 35, q[1]), (q[0] + 35, q[1]), (0, 0, 255), 1)
    else:
        cv2.rectangle(v, (p['roi_left'][0], p['roi_top'][1]), (p['roi_right'][0], p['roi_bottom'][1]), (0, 200, 255), 1)
        cv2.line(v, p['neck_head_side'], p['neck_troch_side'], (0, 0, 255), 2)
        cv2.circle(v, p['neck_center'], 3, (0, 255, 0), -1)
    cv2.putText(v, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
    return v


def sheet(tiles, path, cols=8, size=(200, 210)):
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    W = np.zeros((rows * size[1], cols * size[0], 3), np.uint8)
    for i, t in enumerate(tiles):
        W[(i // cols) * size[1]:(i // cols + 1) * size[1], (i % cols) * size[0]:(i % cols + 1) * size[0]] = cv2.resize(t, size)
    cv2.imwrite(str(path), W, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(ROOT / 'outputs/landmarks_v1/best.pt'))
    ap.add_argument('--out', default=str(ROOT / 'outputs/landmarks_v1/ours'))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred = LandmarkPredictor(args.ckpt, args.device)
    lab = pd.read_csv(ROOT / 'data/interim/image_labels.csv')
    rows, tiles = [], {}
    for r in lab.itertuples():
        dcm = read_dicom(DATA / r.path)
        img = dcm.pixels
        res = pred.predict(img, r.region)
        m = measurements(res['points'], r.region, img.shape, img, mm_per_px=dcm.meta.get('spacing_mm'))
        viol = [c for c in ('v_layout', 'v_axis', 'v_artifact', 'v_rotation', 'v_roi') if getattr(r, c) == 1]
        rows.append({'image_uid': r.image_uid, 'study': r.study, 'path': r.path, 'region': r.region,
                     'quality': None if pd.isna(r.quality) else int(r.quality), 'violations': viol,
                     'width': img.shape[1], 'height': img.shape[0], **res, 'measurements': m})
        key = ('spine' if r.region == 'spine' else 'hip') + ('_bad' if r.quality == 1 else '_good')
        if r.region == 'spine':
            txt = f"tilt {m['spine_tilt_deg']:+.1f} top {m['top_margin_vert']:.2f}"
        else:
            txt = f"{r.region[4]} neck {m['neck_axis_deg']:.0f}"
        txt += ' ' + ','.join(v[2:5] for v in viol)
        tiles.setdefault(key, []).append(draw(img, res['points'], r.region, txt))
    (out / 'predictions.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for k, t in tiles.items():
        sheet(t[:48], out / f'montage_{k}.jpg')
    print(f'{len(rows)} images -> {out}')


if __name__ == '__main__':
    main()
