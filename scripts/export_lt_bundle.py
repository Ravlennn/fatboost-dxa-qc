"""Add hip fusion heads (CNN + lesser trochanter), the spine artifact head and the spine
policy to the release bundle.

hip_positioning and hip quality become standardized logistic heads on
[logit(CNN ensemble score), lt_prom, lt_prom^2, log1p(lt_area), lt_edge_bump]
(pre-specified set of scripts/eval_lt_auto.py), refit on all labelled hips.
Threshold = midpoint of adjacent pooled-OOF scores at the F1-optimal partition
(optimistic, like other shipped thresholds; honest numbers: scripts/evaluate_final.py).

  python scripts/export_lt_bundle.py --bundle models/release
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402
from dxaqc.bundle import sha256, verify_bundle  # noqa: E402
from dxaqc.lesser_trochanter import head_features  # noqa: E402

FEATS = ['cnn', 'lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump']
HEADS = {'hip_positioning': ('v_rotation', 'p_hip_positioning'), 'hip_quality': ('quality', 'p_quality')}


def hip_frame():
    meas = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    d = oof[oof.region.str.startswith('hip')].copy()
    X = pd.DataFrame([head_features({'lt_prominence_mm': meas[u]['lt_prominence_mm'], 'lt_area_mm2': meas[u]['lt_area_mm2'],
                                     'lt_edge_bump_mm': meas[u]['lt_edge_bump_mm']}) for u in d.image_uid], index=d.index)
    return d, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, default=ROOT / 'models/release')
    args = ap.parse_args()
    manifest = verify_bundle(args.bundle)
    if 'landmarks' not in manifest:
        raise SystemExit('Run scripts/export_landmark_bundle.py first')
    d, X = hip_frame()
    heads_path = args.bundle / manifest['landmarks']['heads']
    spec = json.loads(heads_path.read_text())
    spec['hip_heads'] = {}
    for code, (lab, pcol) in HEADS.items():
        dd = d[d[lab].notna()]
        Xd = X.loc[dd.index].copy()
        Xd['cnn'] = F.logit(dd[pcol].values)
        y, g, fo = dd[lab].astype(int).values, dd.study.values, dd.fold.values
        oof_s, _ = F.fit_predict(Xd, y, g, fo, FEATS)
        order = np.sort(np.unique(oof_s))
        t = F.best_threshold(y, oof_s)
        below = order[order < t]
        thr = float((t + below.max()) / 2) if len(below) else float(t)
        m = F.model().fit(Xd[FEATS].values, y)
        sc, lr = m.named_steps['standardscaler'], m.named_steps['logisticregression']
        spec['hip_heads'][code] = {'features': FEATS, 'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
                                   'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]), 'threshold': thr,
                                   'cnn_input': pcol, 'n': int(len(y)), 'pos': int(y.sum()), 'C': 0.3,
                                   'source': 'scripts/export_lt_bundle.py; docs/METHODS_AND_MODELS.md'}
    # Spine artifact: fusion of CNN score and the landmark-anchored detector (docs/METHODS_AND_MODELS.md).
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    d = oof[oof.region.eq('spine') & oof.v_artifact.notna()]
    art = json.load(open(ROOT / 'outputs/artifacts_lm_v1/measurements.json'))
    Xa = pd.DataFrame([art[u] for u in d.image_uid], index=d.index)
    Xa['cnn'] = F.logit(d.p_spine_artifact.values)
    y, g, fo = d.v_artifact.astype(int).values, d.study.values, d.fold.values
    cols = ['cnn', 'art_wire', 'art_metal']
    oof_s, _ = F.fit_predict(Xa, y, g, fo, cols)
    order = np.sort(np.unique(oof_s))
    t = F.best_threshold(y, oof_s)
    below = order[order < t]
    m = F.model().fit(Xa[cols].values, y)
    sc, lr = m.named_steps['standardscaler'], m.named_steps['logisticregression']
    spec['spine_artifact_head'] = {'features': cols, 'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
                                   'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
                                   'threshold': float((t + below.max()) / 2) if len(below) else float(t),
                                   'cnn_input': 'p_spine_artifact', 'n': int(len(y)), 'pos': int(y.sum()), 'C': 0.3}
    # Chosen by nested per-fold selection in scripts/evaluate_final.py (5/5 folds): a spine
    # image is a defect iff at least one reason is found.
    spec['spine_quality_policy'] = 'reasons'
    heads_path.write_text(json.dumps(spec, indent=2))
    manifest['files'][manifest['landmarks']['heads']] = sha256(heads_path)
    manifest['model_id'] = manifest['model_id'].split('+')[0] + '+landmarks_v1+lt_v1+art_v1'
    (args.bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    verify_bundle(args.bundle)
    print(json.dumps({k: v['threshold'] for k, v in spec['hip_heads'].items()}))


if __name__ == '__main__':
    main()
