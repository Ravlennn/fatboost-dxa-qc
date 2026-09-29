"""Spine artifact reason from the landmark-anchored detector (src/dxaqc/artifacts.py) vs release CNN.

Detector frozen after visual QA of all 99 spine images with labels hidden. Pre-specified
feature sets: detector-only [art_wire, art_metal]; fusion [logit CNN, art_wire, art_metal].
Protocol of run_landmark_fusion.py (release folds, inner-fold threshold, paired study bootstrap).

  python scripts/eval_artifacts_lm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402
from dxaqc.artifacts import detect, head_features  # noqa: E402
from dxaqc.dicom_io import read_dicom  # noqa: E402

OUT = ROOT / 'outputs/artifacts_lm_v1'
DATA = ROOT / 'data/interim/train/Исследования'


def measure():
    rows = {}
    for r in json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')):
        if r['region'] != 'spine':
            continue
        dcm = read_dicom(DATA / r['path'])
        res = detect(dcm.pixels, r['points'], mm=dcm.meta.get('spacing_mm', 0.6))
        rows[r['image_uid']] = {**head_features(res), 'wire_len_vert': res['wire_len_vert'],
                                'metal_area_mm2': res['metal_area_mm2'], 'n_wires': res['n_wires'], 'n_metal': res['n_metal']}
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meas = measure()
    (OUT / 'measurements.json').write_text(json.dumps(meas, indent=1))
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    d = oof[oof.region.eq('spine') & oof.v_artifact.notna()].copy()
    X = pd.DataFrame([meas[u] for u in d.image_uid], index=d.index)
    X['cnn'] = F.logit(d.p_spine_artifact.values)
    y, g, fo = d.v_artifact.astype(int).values, d.study.values, d.fold.values
    v = {'cnn': (d.p_spine_artifact.values, F.cnn_decisions(d.p_spine_artifact.values, y, g, fo)),
         'detector': F.fit_predict(X, y, g, fo, ['art_wire', 'art_metal']),
         'fusion': F.fit_predict(X, y, g, fo, ['cnn', 'art_wire', 'art_metal'])}
    res, L = {}, ['| Вариант | AUC | AP | F1 (внутр. порог) | Sens | Spec | ΔAUC vs CNN [95%] | ΔAP vs CNN [95%] |',
                  '|---|---:|---:|---:|---:|---:|---|---|']
    for name, (s, dec) in v.items():
        r = {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s), 'f1': f1_score(y, dec),
             'sens': float(dec[y == 1].mean()), 'spec': float((~dec[y == 0]).mean())}
        if name != 'cnn':
            r['d_auc'], r['d_ap'] = F.paired_boot(y, v['cnn'][0], s, g)
        res[name] = r
        ci = lambda k: f"[{r[k][0]:+.3f}; {r[k][1]:+.3f}]" if k in r else ''
        L.append(f"| {name} | {r['auc']:.3f} | {r['ap']:.3f} | {r['f1']:.3f} | {r['sens']:.2f} | {r['spec']:.2f} | {ci('d_auc')} | {ci('d_ap')} |")
    uni = {k: roc_auc_score(y, X[k]) for k in ('art_wire', 'art_metal', 'art_any')}
    hit = {k: int(((X[k] > 0).values & (y == 1)).sum()) for k in ('art_wire', 'art_metal')}
    fp = {k: int(((X[k] > 0).values & (y == 0)).sum()) for k in ('art_wire', 'art_metal')}
    (OUT / 'eval.json').write_text(json.dumps({'results': res, 'univariate_auc': uni, 'detected_pos': hit,
                                               'detected_neg': fp, 'n': int(len(y)), 'pos': int(y.sum())}, indent=2, default=float))
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('univariate AUC', {k: round(x, 3) for k, x in uni.items()})
    print(f'detector>0 among {int(y.sum())} artifacts:', hit, '| among', int((y == 0).sum()), 'clean:', fp)


if __name__ == '__main__':
    main()
