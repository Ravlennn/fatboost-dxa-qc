"""Error analysis of the final build: montages of false negatives / false positives with measurements.

Reads the nested decisions of scripts/evaluate_final.py and draws, for each error group,
the image with landmark geometry and the numbers the decision was based on.

  python scripts/error_analysis.py
  -> outputs/error_analysis/{group}.jpg, summary.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.dicom_io import read_dicom  # noqa: E402

DATA = ROOT / 'data/interim/train/Исследования'
OUT = ROOT / 'outputs/error_analysis'
SEQ = ('L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom')


def draw(r, pred, lt, art, note):
    img = read_dicom(DATA / pred['path']).pixels
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    p = {k: tuple(int(round(c)) for c in q) for k, q in pred['points'].items()}
    m = pred['measurements']
    if pred['region'] == 'spine':
        for a, b in zip([p[k] for k in SEQ][:-1], [p[k] for k in SEQ][1:]):
            cv2.line(v, a, b, (0, 200, 255), 1)
        txt = [f"tilt {m['spine_tilt_deg']:+.1f} top {m['top_margin_vert']:.2f} bot {m['bottom_margin_vert']:.2f}",
               f"iliac {m.get('iliac_frac_max', 0):.2f} wire {art['wire_len_vert']:.1f} metal {art['metal_area_mm2']:.0f}"]
    else:
        cv2.rectangle(v, (p['roi_left'][0], p['roi_top'][1]), (p['roi_right'][0], p['roi_bottom'][1]), (0, 200, 255), 1)
        cv2.line(v, p['neck_head_side'], p['neck_troch_side'], (0, 0, 255), 1)
        txt = [f"ROI t{m['roi_top_mm']:.0f} b{m['roi_bottom_mm']:.0f} l{m['roi_left_mm']:.0f} r{m['roi_right_mm']:.0f}",
               f"LT {lt['lt_prominence_mm']:.1f}mm {'vis' if lt['visible'] else 'none'} neck {m['neck_axis_deg']:.0f}"]
    v = cv2.resize(v, (300, 320))
    pad = np.zeros((len(txt) * 16 + 20, 300, 3), np.uint8)
    cv2.putText(pad, note, (3, 14), 0, 0.42, (0, 255, 255), 1)
    for i, t in enumerate(txt):
        cv2.putText(pad, t, (3, 32 + i * 15), 0, 0.38, (230, 230, 230), 1)
    return np.vstack([v, pad])


def sheet(tiles, path, cols=6):
    if not tiles:
        return
    tiles = tiles + [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(ROOT / 'outputs/final_validation_v1/decisions_after_v2.csv')
    preds = {r['image_uid']: r for r in json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json'))}
    lt = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))
    art = json.load(open(ROOT / 'outputs/artifacts_lm_v1/measurements.json'))
    groups, rows = {}, []
    for _, r in t.iterrows():
        u = r.image_uid
        pred = preds[u]
        viol = preds[u]['violations']
        if r.y_quality == 1 and r.quality_class == 0:
            g, note = 'quality_FN', 'MISSED: ' + ','.join(v[2:] for v in viol)
        elif r.y_quality == 0 and r.quality_class == 1:
            g, note = 'quality_FP', 'FALSE ALARM: ' + ','.join(c[6:] for c in t.columns
                                                               if c.startswith('reported_') and r[c])
        else:
            continue
        groups.setdefault(g + ('_spine' if r.region == 'spine' else '_hip'), []).append(
            draw(r, pred, lt.get(u, {'lt_prominence_mm': 0, 'visible': False}), art.get(u, {'wire_len_vert': 0, 'metal_area_mm2': 0}), note))
        rows.append({'group': g, 'region': r.region, 'image_uid': u, 'labels': ';'.join(viol),
                     'score_quality': round(r.score_quality, 2)})
    for g, tiles in groups.items():
        sheet(tiles, OUT / f'{g}.jpg')
        print(f'{g}: {len(tiles)}')
    pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
    print(pd.DataFrame(rows).groupby(['group', 'region']).size().to_string())


if __name__ == '__main__':
    main()
