"""Contralateral comparison for hip rotation: this hip vs the other hip of the same study.

Orthopaedic practice judges femoral rotation by matching the lesser-trochanter profile with
the uninjured side (PMC11415034: ICC > 0.99, rotational error ~1-1.7 deg). Both hips of a
study are scanned in one session with the same anatomy, so a difference in LT prominence
between sides is evidence of a positioning error rather than of anatomy.

Pre-specified features: lt_prom_diff (this - other), lt_prom_ratio, neck_angle_diff,
plus the existing CNN score and LT features. No labels are used to build the features;
both images of a study live in the same fold, so nothing leaks across folds.

  python scripts/eval_contralateral.py
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

OUT = ROOT / 'outputs/contralateral_v1'
BASE = ['cnn', 'lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump']
CONTRA = ['lt_prom_diff', 'lt_prom_ratio', 'neck_angle_diff', 'has_contra']
# Errors are a property of the session: the rotation labels of the two hips of a study
# correlate at 0.585, so the neighbour's evidence itself (not the difference) may help.
NEIGHBOUR = ['contra_cnn', 'contra_lt_prom', 'has_contra']


def paired_f1(y, dec_a, dec_b, groups, n=2000, seed=0):
    """95% CI of the F1 difference (TZ priority metric), paired bootstrap over studies."""
    rng = np.random.default_rng(seed)
    us = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in us}
    diffs = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(us, len(us))])
        if y[ii].sum() == 0:
            continue
        diffs.append(f1_score(y[ii], dec_b[ii], zero_division=0) - f1_score(y[ii], dec_a[ii], zero_division=0))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def build():
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    v1 = {r['image_uid']: r for r in json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json'))}
    lt = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))
    d = oof[oof.region.str.startswith('hip')].copy()
    d['lt_prom'] = [lt[u]['lt_prominence_mm'] for u in d.image_uid]
    d['neck'] = [v1[u]['measurements']['neck_axis_deg'] for u in d.image_uid]
    # Contralateral partner: the other hip of the same study (if present).
    other = {}
    for study, g in d.groupby('study'):
        for side in ('hip_left', 'hip_right'):
            me = g[g.region.eq(side)]
            it = g[g.region.eq('hip_right' if side == 'hip_left' else 'hip_left')]
            if len(me) == 1 and len(it) == 1:
                other[me.image_uid.iloc[0]] = it.image_uid.iloc[0]
    X = pd.DataFrame([lt_features({'lt_prominence_mm': lt[u]['lt_prominence_mm'], 'lt_area_mm2': lt[u]['lt_area_mm2'],
                                   'lt_edge_bump_mm': lt[u]['lt_edge_bump_mm']}) for u in d.image_uid], index=d.index)
    prom = dict(zip(d.image_uid, d.lt_prom))
    neck = dict(zip(d.image_uid, d.neck))
    has = np.array([u in other for u in d.image_uid], float)
    X['has_contra'] = has
    X['lt_prom_diff'] = [prom[u] - prom[other[u]] if u in other else 0.0 for u in d.image_uid]
    X['lt_prom_ratio'] = [np.log1p(prom[u]) - np.log1p(prom[other[u]]) if u in other else 0.0 for u in d.image_uid]
    X['neck_angle_diff'] = [neck[u] - neck[other[u]] if u in other else 0.0 for u in d.image_uid]
    X['contra_lt_prom'] = [prom[other[u]] if u in other else float(np.median(list(prom.values()))) for u in d.image_uid]
    X['_other'] = [other.get(u) for u in d.image_uid]
    return d, X, has


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, X, has = build()
    res, L = {'pairs': int(has.sum()), 'images': int(len(d))}, \
        ['| Задача | Вариант | AUC | AP | F1 | ΔAUC vs fusion [95%] | ΔF1 [95%] |', '|---|---|---:|---:|---:|---|---|']
    for task, lab, pcol in (('hip_positioning', 'v_rotation', 'p_hip_positioning'), ('hip_quality', 'quality', 'p_quality')):
        dd = d[d[lab].notna()]
        Xd = X.loc[dd.index].copy()
        Xd['cnn'] = F.logit(dd[pcol].values)
        y, g, fo = dd[lab].astype(int).values, dd.study.values, dd.fold.values
        pc = dict(zip(dd.image_uid, F.logit(dd[pcol].values)))
        med = float(np.median(list(pc.values())))
        Xd['contra_cnn'] = [pc.get(o, med) if isinstance(o, str) else med for o in Xd['_other']]
        v = {'fusion (release)': F.fit_predict(Xd, y, g, fo, BASE),
             'fusion + contralateral diff': F.fit_predict(Xd, y, g, fo, BASE + CONTRA),
             'fusion + neighbour evidence': F.fit_predict(Xd, y, g, fo, BASE + NEIGHBOUR)}
        res[task] = {}
        for name, (s, dec) in v.items():
            r = {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s), 'f1': f1_score(y, dec)}
            if name != 'fusion (release)':
                r['d_auc'], r['d_ap'] = F.paired_boot(y, v['fusion (release)'][0], s, g)
                r['d_f1'] = paired_f1(y, v['fusion (release)'][1], dec, g)
            res[task][name] = r
            ci = (f"[{r['d_auc'][0]:+.3f}; {r['d_auc'][1]:+.3f}] | [{r['d_f1'][0]:+.3f}; {r['d_f1'][1]:+.3f}]"
                  if 'd_auc' in r else ' | ')
            L.append(f"| {task} | {name} | {r['auc']:.3f} | {r['ap']:.3f} | {r['f1']:.3f} | {ci} |")
    # Descriptive: does the difference separate the labels at all?
    dd = d[d.v_rotation.notna()]
    diff = X.loc[dd.index, 'lt_prom_diff'].abs()
    res['abs_diff_auc'] = float(roc_auc_score(dd.v_rotation.astype(int), diff))
    res['median_abs_diff'] = {'rotation=1': float(diff[dd.v_rotation.eq(1).values].median()),
                              'rotation=0': float(diff[dd.v_rotation.eq(0).values].median())}
    (OUT / 'eval.json').write_text(json.dumps(res, indent=2, default=float))
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f"pairs with both hips: {int(has.sum())} of {len(d)} images")
    print('|LT difference| AUC for rotation:', round(res['abs_diff_auc'], 3), res['median_abs_diff'])


if __name__ == '__main__':
    main()
