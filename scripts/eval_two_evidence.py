"""Rotation: require two independent evidences (CNN and the lesser trochanter) instead of a fusion head.

Most of our hip errors sit at the rotation threshold, and we have two independent sources.
Pre-specified decision rules, all with the nested protocol (thresholds on inner folds):
  fusion  - current: one head on [CNN, LT features]
  AND     - flag only when both the CNN head and the LT-only head flag (higher precision)
  OR      - flag when either flags (higher recall)
  2of3    - CNN, LT prominence and LT visibility vote

  python scripts/eval_two_evidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402
from dxaqc.lesser_trochanter import head_features as lt_features  # noqa: E402

OUT = ROOT / 'outputs/two_evidence_v1'
FUSION = ['cnn', 'lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump']
LT_ONLY = ['lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump']


def paired_f1(y, a, b, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    us = np.unique(g)
    idx = {s: np.where(g == s)[0] for s in us}
    out = []
    for _ in range(n):
        ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
        if y[ii].sum():
            out.append(f1_score(y[ii], b[ii], zero_division=0) - f1_score(y[ii], a[ii], zero_division=0))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    lt = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))
    d = oof[oof.region.str.startswith('hip')].copy()
    X = pd.DataFrame([lt_features({'lt_prominence_mm': lt[u]['lt_prominence_mm'], 'lt_area_mm2': lt[u]['lt_area_mm2'],
                                   'lt_edge_bump_mm': lt[u]['lt_edge_bump_mm']}) for u in d.image_uid], index=d.index)
    X['lt_visible'] = [1.0 if lt[u]['visible'] else 0.0 for u in d.image_uid]
    res, L = {}, ['| Задача | Правило | AUC | AP | F1 | Sens | Spec | ΔF1 vs fusion [95%] |',
                  '|---|---|---:|---:|---:|---:|---:|---|']
    for task, lab, pcol in (('hip_positioning', 'v_rotation', 'p_hip_positioning'),
                            ('hip_quality', 'quality', 'p_quality')):
        dd = d[d[lab].notna()]
        Xd = X.loc[dd.index].copy()
        Xd['cnn'] = F.logit(dd[pcol].values)
        y, g, fo = dd[lab].astype(int).values, dd.study.values, dd.fold.values
        s_fus, d_fus = F.fit_predict(Xd, y, g, fo, FUSION)
        s_cnn, d_cnn = dd[pcol].values, F.cnn_decisions(dd[pcol].values, y, g, fo)
        s_lt, d_lt = F.fit_predict(Xd, y, g, fo, LT_ONLY)
        s_vis, d_vis = F.fit_predict(Xd, y, g, fo, ['lt_visible'])
        rules = {'fusion (текущее)': (s_fus, d_fus), 'CNN и вертел (AND)': (np.minimum(s_cnn, s_lt), d_cnn & d_lt),
                 'CNN или вертел (OR)': (np.maximum(s_cnn, s_lt), d_cnn | d_lt),
                 '2 из 3 голосов': ((s_cnn + s_lt + s_vis) / 3, (d_cnn.astype(int) + d_lt + d_vis) >= 2)}
        res[task] = {}
        for name, (s, dec) in rules.items():
            r = {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s), 'f1': f1_score(y, dec),
                 'sens': float(dec[y == 1].mean()), 'spec': float((~dec[y == 0]).mean())}
            if name != 'fusion (текущее)':
                r['d_f1'] = paired_f1(y, d_fus, dec, g)
            res[task][name] = r
            ci = f"[{r['d_f1'][0]:+.3f}; {r['d_f1'][1]:+.3f}]" if 'd_f1' in r else ''
            L.append(f"| {task} | {name} | {r['auc']:.3f} | {r['ap']:.3f} | {r['f1']:.3f} | {r['sens']:.2f} | {r['spec']:.2f} | {ci} |")
    (OUT / 'results.json').write_text(json.dumps(res, indent=2, default=float))
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
