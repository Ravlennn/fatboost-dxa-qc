"""Automatic lesser-trochanter (LT) landmarks from the medial femoral contour. No labels used.

Anchored on landmark-v1 points (neck centre, neck axis => medial side). In a canonical
orientation (medial = +x) the medial cortex edge of the shaft is traced upward from the
lower shaft with a narrow gradient search window (so it does not jump to the ischium
across a dark gap). The LT is the
medium-density bone protruding medially beyond the bright cortical edge (in DXA the LT
is less dense than the cortex); the normal neck flare does not create such a protrusion
because the cortex itself turns medially there.

Outputs landmarks in labeler-v2 format (status 'set' / 'invisible', source 'auto') and
per-image measurements; montage sheets for visual QA are written WITHOUT quality labels.

  python scripts/auto_label_lt.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.dicom_io import read_dicom  # noqa: E402
from dxaqc.lesser_trochanter import MM, detect  # noqa: E402,F401

DATA = ROOT / 'data/interim/train/Исследования'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default=str(ROOT / 'outputs/landmarks_v1/ours/predictions.json'))
    ap.add_argument('--out', default=str(ROOT / 'outputs/lt_auto_v1'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.load(open(args.pred)) if r['region'] != 'spine']
    labels, meas, tiles = [], {}, []
    for i, r in enumerate(rows):
        dcm = read_dicom(DATA / r['path'])
        img = dcm.pixels
        res, err = detect(img, r['points'], mm=dcm.meta.get('spacing_mm'))
        status = {n: 'pred' for n in r['points']}
        pts = dict(r['points'])
        if res:
            for n, q in res['points'].items():
                pts[n] = q
                status[n] = 'set' if res['visible'] else 'invisible'
            meas[r['image_uid']] = {k: res[k] for k in ('visible', 'lt_prominence_mm', 'lt_area_mm2', 'lt_length_mm', 'lt_edge_bump_mm')}
        labels.append({'id': r['image_uid'], 'study': r['study'], 'region': r['region'], 'points': pts,
                       'status': status, 'source': 'auto_lt_v1', 'error': err})
        # QA tile: no quality labels shown.
        v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if res:
            for x, y in res['edge']:
                cv2.circle(v, (int(x), int(y)), 1, (255, 200, 0), -1)
            col = (0, 255, 0) if res['visible'] else (0, 0, 255)
            for n in ('lt_base_top', 'lt_apex', 'lt_base_bottom'):
                q = res['points'][n]
                cv2.circle(v, (int(q[0]), int(q[1])), 3, col, -1)
            txt = f"#{i} {'LT' if res['visible'] else 'none'} {res['lt_prominence_mm']:.1f}mm"
        else:
            txt = f'#{i} ERR {err}'
        cv2.putText(v, txt, (3, 12), 0, 0.4, (0, 255, 255), 1)
        tiles.append(cv2.resize(v, (240, 240)))
    for s in range(0, len(tiles), 24):
        chunk = tiles[s:s + 24] + [np.zeros_like(tiles[0])] * (-len(tiles[s:s + 24]) % 8)
        cv2.imwrite(str(out / f'qa_{s // 24:02d}.jpg'), np.vstack([np.hstack(chunk[j:j + 8]) for j in range(0, len(chunk), 8)]))
    (out / 'landmarks_auto_v2.json').write_text(json.dumps(labels, indent=1))
    (out / 'measurements.json').write_text(json.dumps(meas, indent=1))
    vis = sum(m['visible'] for m in meas.values())
    print(f'{len(rows)} hips, detected {len(meas)}, LT visible {vis}, errors {sum(1 for l in labels if l["error"])}')


if __name__ == '__main__':
    main()
