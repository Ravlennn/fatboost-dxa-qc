"""Collect every DXA image we may use for label-free self-supervised pretraining.

Sources (no quality labels are read):
  * Arak cleaned PNG (overlays inpainted, bone-bright)  -> data/external/arak_dxa/clean
  * Pakistan DXA spine PNG (CC BY 4.0)                  -> data/external/pakistan_dxa_v1
  * our unique organizer DICOM frames, exported as PNG  -> data/interim/ssl_png
Our frames are used without labels (transductive SSL); downstream evaluation
still uses the fixed study folds, so no label leaks into pretraining.

  .venv/bin/python scripts/build_ssl_manifest.py [--pakistan /path/to/pakistan_dxa_v1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.dicom_io import read_dicom  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pakistan', type=Path, default=ROOT / 'data/external/pakistan_dxa_v1')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/ssl_v1/manifest.csv')
    args = ap.parse_args()
    rows = []
    for line in open(ROOT / 'outputs/arak_overlays_v1/annotations.jsonl'):
        r = json.loads(line)
        if r.get('clean_path') and not r.get('duplicate_of'):
            rows.append({'path': r['clean_path'], 'source': 'arak', 'view': r['view'], 'group': 'arak:' + r['patient_key']})
    dst = ROOT / 'data/external/pakistan_dxa_v1'
    if args.pakistan.resolve() != dst.resolve() and args.pakistan.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for f in args.pakistan.glob('*.png'):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    for f in sorted(dst.glob('*.png')):
        rows.append({'path': str(f.relative_to(ROOT)), 'source': 'pakistan', 'view': 'spine_ap', 'group': 'pak:' + f.stem})
    lab = pd.read_csv(ROOT / 'data/interim/image_labels.csv')
    png_dir = ROOT / 'data/interim/ssl_png'
    png_dir.mkdir(parents=True, exist_ok=True)
    for r in lab.itertuples():
        out = png_dir / f'{hashlib.sha1(r.image_uid.encode()).hexdigest()[:16]}.png'
        if not out.exists():
            cv2.imwrite(str(out), read_dicom(ROOT / 'data/interim/train/Исследования' / r.path).pixels)
        view = 'spine_ap' if r.region == 'spine' else 'hip'
        rows.append({'path': str(out.relative_to(ROOT)), 'source': 'ours', 'view': view, 'group': 'ours:' + r.study})
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(['source', 'view']).size().to_string())
    print('total', len(df), '->', args.out)


if __name__ == '__main__':
    main()
