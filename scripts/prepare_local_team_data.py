"""Reconstruct main-schema tables from reviewed organizer annotations, never claim canonical folds."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.folds import make_folds


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f).exists() for f in ('image_labels.csv', 'folds.csv')):
        raise ValueError('Refusing to overwrite existing labels or folds')
    records = []
    sources = [source / 'labels_candidate.jsonl', source / 'quarantine.jsonl']
    for file in sources:
        records.extend(json.loads(line) for line in file.read_text().splitlines())
    rows = []
    for r in sorted(records, key=lambda r: r['path_to_study']):
        assert r['region_mapping'] == 'visually_verified'
        ds = pydicom.dcmread(ROOT / 'data/interim/train/Исследования' / r['path_to_study'])
        a = ds.pixel_array
        digest = hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes()).hexdigest()
        assert digest == r['pixel_sha256']
        assert str(ds.SOPInstanceUID) == r['image_uid']
        row = dict(study=r['group_id'], image_uid=r['image_uid'], study_uid=r['study_uid'],
                   path=r['path_to_study'], region=r['anatomical_region'].replace('lumbar_spine', 'spine'),
                   quality=r['quality_class'], pixel_sha256=r['pixel_sha256'])
        for col, key in [('v_layout','spine_scan_range'), ('v_axis','spine_axis_tilt'),
                         ('v_artifact','spine_artifact'), ('v_rotation','hip_positioning'), ('v_roi','hip_roi_margins')]:
            row[col] = r['violations'][key]
        rows.append(row)
    df = pd.DataFrame(rows)
    assert not df.image_uid.duplicated().any()
    assert not df.pixel_sha256.duplicated().any()
    folds = make_folds(df)
    merged = df.merge(folds, on=['study', 'image_uid'], validate='one_to_one')
    for col in ['study', 'study_uid', 'pixel_sha256', 'image_uid']:
        assert merged.groupby(col).fold.nunique().max() == 1, col
    df.to_csv(out / 'image_labels.csv', index=False)
    df[['study','image_uid','study_uid','path','region','pixel_sha256']].to_csv(out / 'dicom_index.csv', index=False)
    folds.to_csv(out / 'folds.csv', index=False)
    metadata = dict(status='LOCAL_RECONSTRUCTION_NOT_CANONICAL', seed=42,
                    algorithm='main make_folds; sorted relative paths; all 252 unique images',
                    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    tables={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')},
                    limitations=['Not comparable to team AUC until fold hashes match',
                                 'Original quality/reason disagreements retained; missing labels remain missing',
                                 'Prior gamos holdout participates in CV and is no longer an independent test',
                                 'Patient linkage across studies unavailable'])
    (out / 'local_provenance.json').write_text(json.dumps(metadata, indent=2))
    hip = merged[(merged.region != 'spine') & merged.quality.notna()]
    print(hip.groupby('fold').agg(n=('quality','size'), positive=('quality','sum')).to_string())
    print('All unique images:', len(df), 'labelled hips:', len(hip))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    prepare(args.source, ROOT / 'data/interim')
