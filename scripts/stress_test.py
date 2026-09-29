"""Domain-shift stress test: how the measurements and decisions move under image perturbations.

The closed test may come from another densitometer. We perturb our own images (no labels used)
and report how far the measurements drift and how often the spine decisions flip:
  gamma / contrast      - different detector response
  noise / blur          - different dose and reconstruction
  downscale-upscale     - lower resolution
  rescale + correct mm  - different pixel size, spacing taken from DICOM (what should happen)
  rescale, spacing 0.6  - same, but the scanner does not report the spacing (worst case)

  python scripts/stress_test.py --device mps --hips 60
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
from dxaqc.artifacts import detect as art_detect  # noqa: E402
from dxaqc.dicom_io import read_dicom  # noqa: E402
from dxaqc.landmarks import LandmarkPredictor, apply_head, head_features, measurements  # noqa: E402
from dxaqc.lesser_trochanter import detect as lt_detect  # noqa: E402

DATA = ROOT / 'data/interim/train/Исследования'
OUT = ROOT / 'outputs/stress_test_v1'
BASE_MM = 0.6


def perturb(img, kind, rng):
    f = img.astype(np.float32) / 255
    if kind == 'gamma_0.7':
        f = f ** 0.7
    elif kind == 'gamma_1.4':
        f = f ** 1.4
    elif kind == 'contrast_-25%':
        f = 0.5 + (f - 0.5) * 0.75
    elif kind == 'contrast_+25%':
        f = 0.5 + (f - 0.5) * 1.25
    elif kind == 'noise':
        f = f + rng.normal(0, 5 / 255, f.shape)
    elif kind == 'blur':
        f = cv2.GaussianBlur(f, (0, 0), 1.0)
    elif kind == 'downscale_0.8':
        small = cv2.resize(f, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        f = cv2.resize(small, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
    elif kind.startswith('rescale_1.25'):
        f = cv2.resize(f, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_LINEAR)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def spine_decisions(m, heads, art):
    feats = {**head_features(m, 'spine'), **{'art_wire': art['art_wire'], 'art_metal': art['art_metal']}}
    out = {}
    for code, head in heads['heads'].items():
        out[code] = apply_head(head, feats) >= head['threshold']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--hips', type=int, default=60)
    ap.add_argument('--bundle', type=Path, default=ROOT / 'models/release')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pred = LandmarkPredictor(str(args.bundle / 'landmarks/landmarks_v1.pt'), args.device)
    heads = json.loads((args.bundle / 'landmarks/heads.json').read_text())
    lab = pd.read_csv(ROOT / 'data/interim/image_labels.csv')
    rng = np.random.default_rng(0)
    spine = lab[lab.region.eq('spine')]
    hips = lab[lab.region.ne('spine')].sample(min(args.hips, (lab.region != 'spine').sum()), random_state=0)
    kinds = ['gamma_0.7', 'gamma_1.4', 'contrast_-25%', 'contrast_+25%', 'noise', 'blur', 'downscale_0.8',
             'rescale_1.25_mm_known', 'rescale_1.25_mm_missing']
    rows = []
    for r in pd.concat([spine, hips]).itertuples():
        img = read_dicom(DATA / r.path).pixels
        region = 'spine' if r.region == 'spine' else r.region
        base_m = measurements(pred.predict(img, region)['points'], region, img.shape, img, mm_per_px=BASE_MM)
        base_extra = (art_detect(img, pred.predict(img, region)['points'], mm=BASE_MM) if region == 'spine'
                      else lt_detect(img, pred.predict(img, region)['points'], mm=BASE_MM)[0])
        for kind in kinds:
            im = perturb(img, kind, rng)
            # A different pixel size is only harmless if the spacing is known.
            mm = BASE_MM / 1.25 if kind == 'rescale_1.25_mm_known' else BASE_MM
            pts = pred.predict(im, region)['points']
            m = measurements(pts, region, im.shape, im, mm_per_px=mm)
            row = {'uid': r.image_uid, 'region': region, 'kind': kind}
            if region == 'spine':
                a = art_detect(im, pts, mm=mm)
                row.update({'d_tilt_deg': m['spine_tilt_deg'] - base_m['spine_tilt_deg'],
                            'd_top_margin_mm': m['top_margin_mm'] - base_m['top_margin_mm'],
                            'd_wire_vert': a['wire_len_vert'] - base_extra['wire_len_vert']})
                b = spine_decisions(base_m, heads, {'art_wire': np.log1p(base_extra['wire_len_vert']),
                                                    'art_metal': np.log1p(base_extra['metal_area_mm2'])})
                c = spine_decisions(m, heads, {'art_wire': np.log1p(a['wire_len_vert']),
                                               'art_metal': np.log1p(a['metal_area_mm2'])})
                row['decision_flips'] = int(sum(b[k] != c[k] for k in b))
            else:
                lt, _ = lt_detect(im, pts, mm=mm)
                row.update({'d_roi_bottom_mm': m['roi_bottom_mm'] - base_m['roi_bottom_mm'],
                            'd_neck_deg': m['neck_axis_deg'] - base_m['neck_axis_deg'],
                            'd_lt_mm': (lt['lt_prominence_mm'] if lt else 0) - (base_extra['lt_prominence_mm'] if base_extra else 0)})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'raw.csv', index=False)
    agg = df.groupby('kind').agg(
        tilt_mae=('d_tilt_deg', lambda x: x.abs().mean()), top_margin_mae=('d_top_margin_mm', lambda x: x.abs().mean()),
        wire_mae=('d_wire_vert', lambda x: x.abs().mean()), flips=('decision_flips', 'mean'),
        roi_bottom_mae=('d_roi_bottom_mm', lambda x: x.abs().mean()), neck_mae=('d_neck_deg', lambda x: x.abs().mean()),
        lt_mae=('d_lt_mm', lambda x: x.abs().mean())).round(2)
    (OUT / 'summary.json').write_text(agg.to_json(indent=2))
    L = ['| Искажение | Угол, ° | Запас над L1, мм | Проволока | Смена решений (из 2) | ROI снизу, мм | Ось шейки, ° | Вертел, мм |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k, r in agg.iterrows():
        L.append(f"| {k} | {r.tilt_mae} | {r.top_margin_mae} | {r.wire_mae} | {r.flips} | {r.roi_bottom_mae} | {r.neck_mae} | {r.lt_mae} |")
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
