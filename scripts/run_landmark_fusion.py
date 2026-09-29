"""Fuse landmark measurements with release CNN OOF scores, per reason and quality.

Landmark model was trained only on Arak (no labels of ours), so its measurements
are label-free features. Feature sets are fixed before running; no C search.
Outer folds = release folds; F1 threshold chosen on inner 3 study-grouped folds.

  .venv/bin/python scripts/run_landmark_fusion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/landmark_fusion_v1'
EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def features(df):
    m = pd.json_normalize(df['measurements']).set_index(df.index)
    f = pd.DataFrame(index=df.index)
    if 'spine_tilt_deg' in m:
        f['abs_tilt'] = m['spine_tilt_deg'].abs()
        f['abs_tilt_fit'] = m['spine_tilt_fit_deg'].abs()
        f['lateral_dev'] = m['spine_lateral_dev_px'] / m['vertebra_height_px']
        f['top_margin'] = m['top_margin_vert']
        f['bottom_margin'] = m['bottom_margin_vert']
        f['min_margin'] = np.minimum(m['top_margin_vert'], m['bottom_margin_vert'])
        if 'iliac_frac_max' in m:
            f['iliac_max'] = m['iliac_frac_max']
            f['iliac_min'] = m['iliac_frac_min']
        if 'iliac_top_rel' in m:  # bottom-anchored (docs/METHODS_AND_MODELS.md)
            f['iliac_top_rel'] = m['iliac_top_rel']
            f['frame_height_vert'] = m['frame_height_vert']
            f['above_L4_vert'] = m['above_L4_vert']
    if 'roi_top_mm' in m:
        right = df['region'].eq('hip_right')
        # Right hip on screen: shaft/greater trochanter on the left => lateral = left edge.
        f['roi_top'] = m['roi_top_mm']
        f['roi_bottom'] = m['roi_bottom_mm']
        f['roi_lateral'] = np.where(right, m['roi_left_mm'], m['roi_right_mm'])
        f['roi_medial'] = np.where(right, m['roi_right_mm'], m['roi_left_mm'])
        f['roi_min'] = f[['roi_top', 'roi_bottom', 'roi_lateral', 'roi_medial']].min(axis=1)
        f['neck_angle'] = m['neck_axis_deg']
        f['neck_y'] = m['neck_center_y_rel']
    return f


# Pre-specified: (subset, label, CNN score column, landmark features)
TASKS = {
    'spine_axis_tilt': ('spine', 'v_axis', 'p_spine_axis_tilt', ['abs_tilt', 'abs_tilt_fit', 'lateral_dev']),
    'spine_scan_range': ('spine', 'v_layout', 'p_spine_scan_range', ['top_margin', 'bottom_margin', 'min_margin']),
    # v2 (added after landmarks_v1): iliac crest visibility, windows fixed a priori.
    'spine_scan_range_iliac': ('spine', 'v_layout', 'p_spine_scan_range',
                               ['top_margin', 'bottom_margin', 'min_margin', 'iliac_max', 'iliac_min']),
    # v3: the margin above L1 is unreliable (the model re-anchors L1 when the frame is cropped),
    # so the scan-range head is anchored at the bottom: crest position, frame height, extent above L4.
    'spine_scan_range_v3': ('spine', 'v_layout', 'p_spine_scan_range',
                            ['min_margin', 'iliac_max', 'iliac_top_rel', 'frame_height_vert', 'above_L4_vert']),
    'spine_artifact': ('spine', 'v_artifact', 'p_spine_artifact', []),
    'hip_positioning': ('hip', 'v_rotation', 'p_hip_positioning', ['neck_angle', 'neck_y']),
    'hip_roi_margins': ('hip', 'v_roi', 'p_hip_roi_margins', ['roi_top', 'roi_bottom', 'roi_lateral', 'roi_medial', 'roi_min']),
    'quality_spine': ('spine', 'quality', 'p_quality', ['abs_tilt', 'min_margin', 'bottom_margin']),
    'quality_hip': ('hip', 'quality', 'p_quality', ['roi_min', 'roi_bottom']),
}


def model(C=0.3):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=5000))


def best_threshold(y, s):
    cands = np.unique(s)
    best = (-1, 0.5)
    for t in cands:
        f = f1_score(y, s >= t, zero_division=0)
        if f > best[0]:
            best = (f, t)
    return best[1]


def fit_predict(X, y, groups, folds, variant_cols):
    """Outer-fold OOF scores and inner-selected-threshold decisions."""
    oof = np.full(len(y), np.nan)
    dec = np.zeros(len(y), bool)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        Xtr, ytr = X.loc[tr, variant_cols].values, y[tr]
        # Inner OOF for the threshold.
        inner = np.full(tr.sum(), np.nan)
        gkf = GroupKFold(n_splits=3)
        for itr, ite in gkf.split(Xtr, ytr, groups[tr]):
            if ytr[itr].sum() == 0:
                inner[ite] = 0
                continue
            inner[ite] = model().fit(Xtr[itr], ytr[itr]).predict_proba(Xtr[ite])[:, 1]
        thr = best_threshold(ytr, inner) if ytr.sum() else 0.5
        m = model().fit(Xtr, ytr)
        oof[te] = m.predict_proba(X.loc[te, variant_cols].values)[:, 1]
        dec[te] = oof[te] >= thr
    return oof, dec


def cnn_decisions(s, y, groups, folds):
    """Threshold for the raw CNN score chosen on inner folds as well (fair F1)."""
    dec = np.zeros(len(y), bool)
    for f in np.unique(folds):
        tr = folds != f
        thr = best_threshold(y[tr], s[tr]) if y[tr].sum() else 0.5
        dec[folds == f] = s[folds == f] >= thr
    return dec


def paired_boot(y, a, b, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    us = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in us}
    d_auc, d_ap = [], []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(us, len(us))])
        if 0 < y[ii].sum() < len(ii):
            d_auc.append(roc_auc_score(y[ii], b[ii]) - roc_auc_score(y[ii], a[ii]))
            d_ap.append(average_precision_score(y[ii], b[ii]) - average_precision_score(y[ii], a[ii]))
    return [float(np.percentile(d_auc, q)) for q in (2.5, 97.5)], [float(np.percentile(d_ap, q)) for q in (2.5, 97.5)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    pred = pd.DataFrame(json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')))
    df = oof.merge(pred[['image_uid', 'study', 'measurements']], on=['study', 'image_uid'], how='left', validate='1:1')
    assert df['measurements'].notna().all()
    results, rows_out = {}, []
    for name, (subset, lab, pcol, lm) in TASKS.items():
        d = df[df['region'].eq('spine')] if subset == 'spine' else df[df['region'].str.startswith('hip')]
        d = d[d[lab].notna() & d[pcol].notna()].copy()
        y = d[lab].astype(int).values
        groups, folds = d['study'].values, d['fold'].values
        X = features(d)
        X['cnn'] = logit(d[pcol].values)
        s_cnn = d[pcol].values
        res = {'n': int(len(y)), 'pos': int(y.sum()), 'landmark_features': lm}
        variants = {'cnn': None}
        if lm:
            variants['landmarks'] = lm
            variants['fusion'] = ['cnn'] + lm
        scores = {}
        for v, cols in variants.items():
            if cols is None:
                s, dec = s_cnn, cnn_decisions(s_cnn, y, groups, folds)
            else:
                s, dec = fit_predict(X, y, groups, folds, cols)
            scores[v] = s
            res[v] = {'auc': float(roc_auc_score(y, s)), 'ap': float(average_precision_score(y, s)),
                      'f1_inner_thr': float(f1_score(y, dec)),
                      'sens': float(dec[y == 1].mean()), 'spec': float((~dec[y == 0]).mean())}
        for v in ('landmarks', 'fusion'):
            if v in scores:
                ci_auc, ci_ap = paired_boot(y, scores['cnn'], scores[v], groups)
                res[v]['delta_auc_vs_cnn_ci95'] = ci_auc
                res[v]['delta_ap_vs_cnn_ci95'] = ci_ap
        results[name] = res
        for i, idx in enumerate(d.index):
            rows_out.append({'task': name, 'study': d.at[idx, 'study'], 'image_uid': d.at[idx, 'image_uid'],
                             'fold': int(folds[i]), 'y': int(y[i]), **{f's_{k}': float(v[i]) for k, v in scores.items()}})
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    pd.DataFrame(rows_out).to_csv(OUT / 'oof.csv', index=False)
    lines = ['| Задача | n/pos | Вариант | AUC | AP | F1 (внутр. порог) | Sens | Spec | ΔAUC vs CNN [95%] | ΔAP vs CNN [95%] |',
             '|---|---|---|---:|---:|---:|---:|---:|---|---|']
    for name, r in results.items():
        for v in ('cnn', 'landmarks', 'fusion'):
            if v not in r:
                continue
            x = r[v]
            dA = f"[{x['delta_auc_vs_cnn_ci95'][0]:+.3f}; {x['delta_auc_vs_cnn_ci95'][1]:+.3f}]" if 'delta_auc_vs_cnn_ci95' in x else ''
            dP = f"[{x['delta_ap_vs_cnn_ci95'][0]:+.3f}; {x['delta_ap_vs_cnn_ci95'][1]:+.3f}]" if 'delta_ap_vs_cnn_ci95' in x else ''
            lines.append(f"| {name} | {r['n']}/{r['pos']} | {v} | {x['auc']:.3f} | {x['ap']:.3f} | {x['f1_inner_thr']:.3f} | "
                         f"{x['sens']:.2f} | {x['spec']:.2f} | {dA} | {dP} |")
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
