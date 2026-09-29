"""Pick the images where the model most confidently disagrees with the label, for expert review.

Three groups, all from the nested out-of-fold decisions (no image sees its own fold):
  A. labelled defect, no reason flagged in the annotation at all (2 images) — what was wrong?
  B. measured tilt above the TZ limit but not labelled as an axis violation;
  C. the most confident remaining disagreements (both directions), balanced by region.
Produces a sheet with the measurements printed on the images and a CSV for the reply.

  python scripts/label_review_list.py
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
OUT = ROOT / 'outputs/label_review'
SEQ = ('L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom')
TILT_SLOPE = 0.921


def tile(pred, note, numbers):
    img = read_dicom(DATA / pred['path']).pixels
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    p = {k: tuple(int(round(c)) for c in q) for k, q in pred['points'].items()}
    if pred['region'] == 'spine':
        seq = [p[k] for k in SEQ]
        cv2.line(v, seq[0], seq[-1], (0, 200, 255), 2)
        for q in seq:
            cv2.line(v, (q[0] - 25, q[1]), (q[0] + 25, q[1]), (0, 140, 255), 1)
    else:
        cv2.rectangle(v, (p['roi_left'][0], p['roi_top'][1]), (p['roi_right'][0], p['roi_bottom'][1]), (0, 200, 255), 1)
        cv2.line(v, p['neck_head_side'], p['neck_troch_side'], (0, 0, 255), 2)
    v = cv2.resize(v, (320, 340))
    pad = np.zeros((54, 320, 3), np.uint8)
    cv2.putText(pad, note[:46], (4, 16), 0, 0.45, (0, 255, 255), 1)
    cv2.putText(pad, numbers[:52], (4, 34), 0, 0.4, (230, 230, 230), 1)
    cv2.putText(pad, 'согласны? да / нет / спорно', (4, 50), 0, 0.38, (150, 150, 150), 1)
    return np.vstack([v, pad])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(ROOT / 'outputs/final_validation_v1/decisions_after_v2.csv')
    preds = {r['image_uid']: r for r in json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json'))}
    rows = []
    for _, r in t.iterrows():
        p = preds[r.image_uid]
        m, viol = p['measurements'], p['violations']
        if p['region'] == 'spine':
            tilt = abs(m['spine_tilt_deg']) / TILT_SLOPE
            numbers = (f"наклон {tilt:.1f}°, запас сверху {m['top_margin_vert']:.2f}, "
                       f"снизу {m['bottom_margin_vert']:.2f}")
        else:
            tilt = 0.0
            numbers = (f"ROI сверху {m['roi_top_mm']:.0f} мм, снизу {m['roi_bottom_mm']:.0f} мм, "
                       f"ось шейки {m['neck_axis_deg']:.0f}°")
        group = None
        if r.y_quality == 1 and not viol:
            group = 'A. брак без указанной причины'
        elif p['region'] == 'spine' and tilt > 5 and 'v_axis' not in viol:
            group = 'B. угол выше 5°, но метки оси нет'
        elif r.y_quality != r.quality_class:
            group = 'C. расхождение модели и разметки'
        if group:
            rows.append({'group': group, 'image_uid': r.image_uid, 'region': p['region'], 'path': p['path'],
                         'label_quality': int(r.y_quality), 'label_reasons': ';'.join(viol),
                         'model_quality': int(r.quality_class), 'model_score': round(float(r.score_quality), 2),
                         'numbers': numbers, 'confidence': abs(float(r.score_quality) - 1)})
    df = pd.DataFrame(rows).sort_values(['group', 'confidence'], ascending=[True, False])
    pick = pd.concat([g if k.startswith(('A', 'B')) else g.head(max(0, 20 - len(df[df.group.str.startswith(("A", "B"))])))
                      for k, g in df.groupby('group')]).head(24)
    tiles = []
    for r in pick.itertuples():
        note = f"{r.group[:2]} метка: {'брак' if r.label_quality else 'норма'}" + (f" ({r.label_reasons})" if r.label_reasons else '')
        note += f" | модель: {'брак' if r.model_quality else 'норма'}"
        tiles.append(tile(preds[r.image_uid], note, r.numbers))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
    cv2.imwrite(str(OUT / 'review_sheet.jpg'), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    pick.drop(columns=['confidence']).to_csv(OUT / 'review_list.csv', index=False)
    print(pick.groupby('group').size().to_string())
    print(f'-> {OUT}/review_sheet.jpg, review_list.csv')


if __name__ == '__main__':
    main()
