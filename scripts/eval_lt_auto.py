"""Rotation reason from automatic lesser-trochanter features vs the release CNN.

TZ: correct = contour slightly deformed by the LT; over-rotated = smooth contour (no LT);
under-rotated = LT too large. The relation is non-monotonic, so the pre-specified feature
set includes the squared prominence. Protocol = run_landmark_fusion.py (release folds,
inner-fold threshold, paired study bootstrap). Detector frozen before labels were viewed.

  python scripts/eval_lt_auto.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402

FEATS = ['lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump']


def main():
    meas = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    d = oof[oof.region.str.startswith('hip')].copy()
    d = d[d.image_uid.isin(meas)]
    m = pd.DataFrame([meas[u] for u in d.image_uid], index=d.index)
    X = pd.DataFrame({'lt_prom': m.lt_prominence_mm, 'lt_prom_sq': m.lt_prominence_mm ** 2,
                      'lt_area': np.log1p(m.lt_area_mm2), 'lt_edge_bump': m.lt_edge_bump_mm}, index=d.index)
    out, lines = {}, ['| Задача | n/pos | Вариант | AUC | AP | F1 (внутр. порог) | ΔAUC vs CNN [95%] | ΔAP vs CNN [95%] |',
                      '|---|---|---|---:|---:|---:|---|---|']
    for task, lab, pcol in [('hip_positioning', 'v_rotation', 'p_hip_positioning'), ('quality_hip', 'quality', 'p_quality')]:
        dd = d[d[lab].notna()]
        y, g, fo = dd[lab].astype(int).values, dd.study.values, dd.fold.values
        Xd = X.loc[dd.index].copy()
        Xd['cnn'] = F.logit(dd[pcol].values)
        v = {'cnn': (dd[pcol].values, F.cnn_decisions(dd[pcol].values, y, g, fo)),
             'lt_auto': F.fit_predict(Xd, y, g, fo, FEATS),
             'fusion': F.fit_predict(Xd, y, g, fo, ['cnn'] + FEATS)}
        out[task] = {}
        for name, (s, dec) in v.items():
            r = {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s), 'f1': f1_score(y, dec)}
            if name != 'cnn':
                r['d_auc'], r['d_ap'] = F.paired_boot(y, v['cnn'][0], s, g)
            out[task][name] = r
            ci = lambda k: f"[{r[k][0]:+.3f}; {r[k][1]:+.3f}]" if k in r else ''
            lines.append(f"| {task} | {len(y)}/{y.sum()} | {name} | {r['auc']:.3f} | {r['ap']:.3f} | {r['f1']:.3f} | {ci('d_auc')} | {ci('d_ap')} |")
    # Descriptive: rotation rate by LT prominence tertile (non-monotonic pattern check).
    dd = d[d.v_rotation.notna()]
    q = pd.qcut(X.loc[dd.index, 'lt_prom'].rank(method='first'), 3, labels=['low', 'mid', 'high'])
    rate = dd.groupby(q, observed=True).v_rotation.agg(['mean', 'size']).round(3)
    (ROOT / 'outputs/lt_auto_v1/eval.json').write_text(json.dumps({'results': out, 'rotation_rate_by_prominence_tertile':
                                                                    rate.to_dict()}, indent=2, default=float))
    (ROOT / 'outputs/lt_auto_v1/eval_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(rate)


if __name__ == '__main__':
    main()
