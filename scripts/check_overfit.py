"""Overfitting / leakage checks for the final build (outputs/final_validation_v1).

1. Per-fold stability of the nested decisions (large spread = fitted to some folds).
2. Label-permutation test: study-level shuffled labels through the SAME nested fit/threshold
   code; honest pipelines give AUC ~0.5 and F1 ~ prevalence baseline (no leakage).
3. Optimism of shipped thresholds: pooled-OOF-tuned threshold vs nested thresholds.

  python scripts/check_overfit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402
from dxaqc.lesser_trochanter import head_features as lt_features  # noqa: E402

OUT = ROOT / 'outputs/final_validation_v1'
N_PERM = 200


def per_fold(t):
    rows = []
    for f, u in t.groupby('fold'):
        for name, m in (('quality_all', slice(None)), ('quality_spine', u.region.eq('spine')),
                        ('quality_hip', u.region.str.startswith('hip'))):
            v = u[m]
            rows.append({'fold': int(f), 'slice': name, 'n': len(v), 'pos': int(v.y_quality.sum()),
                         'f1': f1_score(v.y_quality, v.quality_class), 'auc': roc_auc_score(v.y_quality, v.score_quality)})
        for c in ('spine_artifact', 'spine_axis_tilt', 'hip_positioning'):
            v = u[u['score_' + c].notna() & u['y_' + c].notna()]
            if v['y_' + c].nunique() == 2:
                rows.append({'fold': int(f), 'slice': c, 'n': len(v), 'pos': int(v['y_' + c].sum()),
                             'f1': f1_score(v['y_' + c].astype(int), v['reported_' + c]),
                             'auc': roc_auc_score(v['y_' + c].astype(int), v['score_' + c])})
    return pd.DataFrame(rows)


def heads_data():
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    v1 = pd.DataFrame(json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')))
    df = oof.merge(v1[['image_uid', 'study', 'measurements']], on=['study', 'image_uid'])
    art = json.load(open(ROOT / 'outputs/artifacts_lm_v1/measurements.json'))
    lt = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))
    tasks = {}
    d = df[df.region.eq('spine') & df.v_artifact.notna()].copy()
    X = pd.DataFrame([art[u] for u in d.image_uid], index=d.index)
    X['cnn'] = F.logit(d.p_spine_artifact.values)
    tasks['spine_artifact (cnn+detector)'] = (X, ['cnn', 'art_wire', 'art_metal'], d.v_artifact.astype(int).values, d)
    d = df[df.region.eq('spine')].copy()
    X = F.features(d)
    tasks['spine_axis_tilt (landmarks)'] = (X, F.TASKS['spine_axis_tilt'][3], d.v_axis.astype(int).values, d)
    d = df[df.region.str.startswith('hip') & df.v_rotation.notna()].copy()
    X = pd.DataFrame([lt_features({'lt_prominence_mm': lt[u]['lt_prominence_mm'], 'lt_area_mm2': lt[u]['lt_area_mm2'],
                                   'lt_edge_bump_mm': lt[u]['lt_edge_bump_mm']}) for u in d.image_uid], index=d.index)
    X['cnn'] = F.logit(d.p_hip_positioning.values)
    tasks['hip_positioning (cnn+LT)'] = (X, ['cnn', 'lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump'],
                                         d.v_rotation.astype(int).values, d)
    return tasks


def permutation(tasks, rng):
    res = {}
    for name, (X, cols, y, d) in tasks.items():
        g, fo = d.study.values, d.fold.values
        s, dec = F.fit_predict(X, y, g, fo, cols)
        real = {'auc': roc_auc_score(y, s), 'f1': f1_score(y, dec)}
        studies = np.unique(g)
        lab = pd.Series(y, index=g).groupby(level=0).max()  # study-level shuffle keeps clustering
        aucs, f1s = [], []
        for _ in range(N_PERM):
            perm = dict(zip(studies, rng.permutation(lab.reindex(studies).values)))
            yp = np.array([perm[s_] for s_ in g])
            if yp.sum() == 0:
                continue
            sp, dp = F.fit_predict(X, yp, g, fo, cols)
            aucs.append(roc_auc_score(yp, sp))
            f1s.append(f1_score(yp, dp))
        res[name] = {'real_auc': real['auc'], 'real_f1': real['f1'], 'perm_auc_mean': float(np.mean(aucs)),
                     'perm_auc_p95': float(np.percentile(aucs, 95)), 'perm_f1_mean': float(np.mean(f1s)),
                     'p_value_auc': float((1 + sum(a >= real['auc'] for a in aucs)) / (1 + len(aucs))),
                     'n_fitted_params': len(cols) + 2, 'n': int(len(y)), 'pos': int(y.sum())}
    return res


def main():
    t = pd.read_csv(OUT / 'decisions_after_v2.csv')
    pf = per_fold(t)
    pf.to_csv(OUT / 'overfit_per_fold.csv', index=False)
    spread = pf.groupby('slice').agg(f1_min=('f1', 'min'), f1_max=('f1', 'max'), f1_mean=('f1', 'mean'),
                                     auc_min=('auc', 'min'), auc_max=('auc', 'max'), auc_mean=('auc', 'mean')).round(3)
    rng = np.random.default_rng(0)
    perm = permutation(heads_data(), rng)
    # optimism of pooled-OOF thresholds (what ships) vs nested
    shipped = {}
    for c, col in (('quality_all', 'score_quality'),):
        y, s = t.y_quality.values, t[col].values
        thr = F.best_threshold(y, s)
        shipped[c] = {'f1_nested': f1_score(y, s >= 1), 'f1_pooled_threshold': f1_score(y, s >= thr)}
    json.dump({'per_fold_spread': spread.reset_index().to_dict('records'), 'permutation': perm, 'threshold_optimism': shipped},
              open(OUT / 'overfit_checks.json', 'w'), indent=2, default=float)
    print('Per-fold spread (5 outer folds):')
    print(spread.to_string())
    print('\nPermutation test (study-level label shuffle, same nested code, %d permutations):' % N_PERM)
    for k, v in perm.items():
        print(f"  {k:32s} real AUC {v['real_auc']:.3f} F1 {v['real_f1']:.3f} | shuffled AUC {v['perm_auc_mean']:.3f} "
              f"(p95 {v['perm_auc_p95']:.3f}) F1 {v['perm_f1_mean']:.3f} | p={v['p_value_auc']:.3f} | params {v['n_fitted_params']}")
    print('\nThreshold optimism:', {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in shipped.items()})


if __name__ == '__main__':
    main()
