"""Final development validation of the shipped decision rule (TZ 8.4 / 4), one protocol.

Every decision is made by components fitted WITHOUT the evaluated fold:
  * outer folds = release study folds (5); CNN scores are the release OOF;
  * landmark/LT heads: logistic regression refit on the outer-train part;
  * every threshold (CNN or head) is the F1-optimal threshold on outer-train
    predictions (inner 3 study-grouped folds for heads, OOF of other folds for CNN).
Rule reproduced exactly as in the CLI:
  spine: defect = CNN quality OR landmark axis OR landmark scan-range; artifact from CNN
  hip:   defect = fusion(CNN, lesser trochanter) quality; rotation = fusion; ROI = CNN
  reasons are reported only for defects; defect without reason -> quality_unspecified.
Continuous score of a composite decision = max over its components of score/threshold
(>= 1 exactly when the decision is "defect"), used for ROC-AUC / PR-AUC.
Also evaluates the previous CNN-only release with the same protocol ("before").
95% CI: 2000 bootstrap resamples of studies (spine and hip of a study stay together).

  python scripts/evaluate_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402
from dxaqc.lesser_trochanter import head_features  # noqa: E402

OUT = ROOT / 'outputs/final_validation_v1'
# Threshold strategy used by the nested heads; scripts/eval_threshold_policy.py swaps it.
THRESHOLD = {'fn': lambda y, sc: F.best_threshold(y, sc), 'name': 'argmax_f1'}
REASONS = {'spine_scan_range': 'v_layout', 'spine_axis_tilt': 'v_axis', 'spine_artifact': 'v_artifact',
           'hip_positioning': 'v_rotation', 'hip_roi_margins': 'v_roi'}
B = 2000


def nested_head(X, cols, y, g, fo):
    """Outer-fold scores and per-row thresholds for a logistic head."""
    s, thr = np.zeros(len(y)), np.zeros(len(y))
    for f in np.unique(fo):
        tr, te = fo != f, fo == f
        Xtr, ytr = X.loc[tr, cols].values, y[tr]
        inner = np.zeros(tr.sum())
        for a, b in GroupKFold(3).split(Xtr, ytr, g[tr]):
            inner[b] = F.model().fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1] if ytr[a].any() else 0
        thr[te] = THRESHOLD['fn'](ytr, inner)
        s[te] = F.model().fit(Xtr, ytr).predict_proba(X.loc[te, cols].values)[:, 1]
    return s, thr


def nested_raw(score, y, fo):
    thr = np.zeros(len(y))
    for f in np.unique(fo):
        tr = fo != f
        thr[fo == f] = THRESHOLD['fn'](y[tr], score[tr])
    return score, thr


def build():
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    v1 = pd.DataFrame(json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')))
    df = oof.merge(v1[['image_uid', 'study', 'measurements']], on=['study', 'image_uid'], validate='1:1')
    df = df[df.quality.notna()].reset_index(drop=True)
    lt = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))
    comp = {}  # name -> (score, thr) arrays aligned to df (NaN where not applicable)
    nan = lambda: (np.full(len(df), np.nan), np.full(len(df), np.nan))

    def put(name, mask, s, t):
        a, b = comp.setdefault(name, nan())
        a[mask], b[mask] = s, t

    # ---- spine
    sp = df.region.eq('spine').values
    d = df[sp]
    y = lambda col: d[col].astype(int).values
    g, fo = d.study.values, d.fold.values
    X = F.features(d)
    put('spine_quality_cnn', sp, *nested_raw(d.p_quality.values, y('quality'), fo))
    for code in ('spine_scan_range', 'spine_axis_tilt', 'spine_artifact'):
        put(code + '_cnn', sp, *nested_raw(d['p_' + code].values, y(REASONS[code]), fo))
    put('spine_axis_tilt_lm', sp, *nested_head(X, F.TASKS['spine_axis_tilt'][3], y('v_axis'), g, fo))
    put('spine_scan_range_lm', sp, *nested_head(X, F.TASKS['spine_scan_range_iliac'][3], y('v_layout'), g, fo))
    art = json.load(open(ROOT / 'outputs/artifacts_lm_v1/measurements.json'))
    Xa = pd.DataFrame([art[u] for u in d.image_uid], index=d.index)
    Xa['cnn'] = F.logit(d.p_spine_artifact.values)
    put('spine_artifact_fusion', sp, *nested_head(Xa, ['cnn', 'art_wire', 'art_metal'], y('v_artifact'), g, fo))
    # ---- hip
    hp = df.region.str.startswith('hip').values
    d = df[hp]
    g, fo = d.study.values, d.fold.values
    Xh = pd.DataFrame([head_features({'lt_prominence_mm': lt[u]['lt_prominence_mm'], 'lt_area_mm2': lt[u]['lt_area_mm2'],
                                      'lt_edge_bump_mm': lt[u]['lt_edge_bump_mm']}) for u in d.image_uid], index=d.index)
    feats = ['cnn', 'lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump']
    for name, lab, pcol in (('hip_quality', 'quality', 'p_quality'), ('hip_positioning', 'v_rotation', 'p_hip_positioning')):
        yy = d[lab].astype(int).values
        put(name + '_cnn', hp, *nested_raw(d[pcol].values, yy, fo))
        Xd = Xh.copy()
        Xd['cnn'] = F.logit(d[pcol].values)
        put(name + '_fusion', hp, *nested_head(Xd, feats, yy, g, fo))
    put('hip_roi_margins_cnn', hp, *nested_raw(d.p_hip_roi_margins.values, d.v_roi.astype(int).values, fo))
    return df, comp


def spine_policy_nested(df, comp, r):
    """Spine defect policy chosen INSIDE each outer fold by F1 on the outer-train rows:
    'or'      = CNN quality OR landmark axis OR scan range (OR artifact fusion);
    'reasons' = defect only if a reason is found (axis | scan | artifact fusion)."""
    sp = df.region.eq('spine').values
    reasons = np.fmax.reduce([r('spine_axis_tilt_lm'), r('spine_scan_range_lm'), r('spine_artifact_fusion')])
    cand = {'or': np.fmax(r('spine_quality_cnn'), reasons), 'reasons': reasons}
    y, fo = df.quality.values, df.fold.values
    q, chosen = np.full(len(df), np.nan), {}
    for f in np.unique(fo[sp]):
        tr, te = sp & (fo != f), sp & (fo == f)
        best = max(cand, key=lambda k: f1_score(y[tr].astype(int), cand[k][tr] >= 1))
        chosen[int(f)] = best
        q[te] = cand[best][te]
    return q, chosen


def decide(df, comp, version):
    """Per-image composite score (max s/thr), defect decision and reported reasons."""
    r = lambda n: comp[n][0] / np.maximum(comp[n][1], 1e-9)
    sp = df.region.eq('spine').values
    decide.chosen = None
    if version == 'after_v2':
        qs, decide.chosen = spine_policy_nested(df, comp, r)
        q = np.where(sp, qs, r('hip_quality_fusion'))
        reason = {'spine_scan_range': r('spine_scan_range_lm'), 'spine_axis_tilt': r('spine_axis_tilt_lm'),
                  'spine_artifact': r('spine_artifact_fusion'), 'hip_positioning': r('hip_positioning_fusion'),
                  'hip_roi_margins': r('hip_roi_margins_cnn')}
    elif version == 'after':
        q = np.where(sp, np.fmax.reduce([r('spine_quality_cnn'), r('spine_axis_tilt_lm'), r('spine_scan_range_lm')]),
                     r('hip_quality_fusion'))
        reason = {'spine_scan_range': r('spine_scan_range_lm'), 'spine_axis_tilt': r('spine_axis_tilt_lm'),
                  'spine_artifact': r('spine_artifact_cnn'), 'hip_positioning': r('hip_positioning_fusion'),
                  'hip_roi_margins': r('hip_roi_margins_cnn')}
    else:
        q = np.where(sp, r('spine_quality_cnn'), r('hip_quality_cnn'))
        reason = {c: r(c + '_cnn') for c in REASONS if not c.startswith('hip_positioning')}
        reason['hip_positioning'] = r('hip_positioning_cnn')
    bad = q >= 1
    out = pd.DataFrame({'study': df.study, 'image_uid': df.image_uid, 'region': df.region, 'fold': df.fold,
                        'y_quality': df.quality.astype(int), 'score_quality': q, 'quality_class': bad.astype(int)})
    for c, s in reason.items():
        applicable = sp if c.startswith('spine') else ~sp
        out['score_' + c] = np.where(applicable, s, np.nan)
        out['reported_' + c] = applicable & bad & (s >= 1)
        out['y_' + c] = df[REASONS[c]]
    rep = out[[f'reported_{c}' for c in REASONS]].any(axis=1)
    out['quality_unspecified'] = bad & ~rep
    return out


def binary(y, d, s):
    y, d = np.asarray(y, int), np.asarray(d, bool)
    tp, fn = int((d & (y == 1)).sum()), int((~d & (y == 1)).sum())
    tn, fp = int((~d & (y == 0)).sum()), int((d & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    m = {'n': len(y), 'pos': int(y.sum()), 'sensitivity': sens, 'specificity': spec, 'balanced_acc': (sens + spec) / 2,
         'f1': f1_score(y, d, zero_division=0), 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp}
    if 0 < y.sum() < len(y):
        m['roc_auc'] = roc_auc_score(y, s)
        m['pr_auc'] = average_precision_score(y, s)
    return m


def all_metrics(t):
    res = {}
    for name, m in (('quality_all', slice(None)), ('quality_spine', t.region.eq('spine')), ('quality_hip', t.region.str.startswith('hip'))):
        u = t[m]
        res[name] = binary(u.y_quality, u.quality_class == 1, u.score_quality)
    f1s = []
    for c in REASONS:
        u = t[t['score_' + c].notna() & t['y_' + c].notna()]
        res[c] = binary(u['y_' + c], u['reported_' + c], u['score_' + c])
        f1s.append(res[c]['f1'])
    res['reasons_macro_f1'] = {'value': float(np.mean(f1s))}
    res['quality_unspecified_among_defects'] = {'value': float(t.quality_unspecified.sum() / max(t.quality_class.sum(), 1))}
    return res


def with_ci(t, rng):
    point = all_metrics(t)
    studies = t.study.unique()
    idx = {s: np.where(t.study.values == s)[0] for s in studies}
    reps = []
    for _ in range(B):
        ii = np.concatenate([idx[s] for s in rng.choice(studies, len(studies))])
        reps.append(all_metrics(t.iloc[ii]))
    for k, m in point.items():
        for key in list(m):
            if key in ('n', 'pos', 'tp', 'fn', 'tn', 'fp'):
                continue
            vals = [r[k][key] for r in reps if key in r[k]]
            m[key] = {'value': float(m[key]), 'ci95': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
    return point


def fmt(m, key):
    if key not in m:
        return '—'
    v = m[key]
    return f"{v['value']:.3f} [{v['ci95'][0]:.2f}–{v['ci95'][1]:.2f}]"


def processing_stats():
    csv = OUT / 'cli_run.csv'
    if not csv.is_file():
        return None
    r = pd.read_csv(csv)
    ok = r.processing_status.eq('Success')
    per_study = r[ok].groupby('study_uid').time_of_processing.sum()
    summ = json.loads((OUT / 'cli_run.summary.json').read_text()) if (OUT / 'cli_run.summary.json').is_file() else {}
    return {'files': int(len(r)), 'success': int(ok.sum()), 'success_rate': float(ok.mean()),
            'studies': int(per_study.size), 'study_seconds_median': float(per_study.median()),
            'study_seconds_p95': float(per_study.quantile(0.95)), 'study_seconds_max': float(per_study.max()),
            'image_seconds_median_unique': float(r[ok & (r.time_of_processing > 0.05)].time_of_processing.median()),
            'wall_seconds_total': summ.get('wall_seconds_including_setup'), 'duplicates': summ.get('duplicates'),
            'router_regions': r[ok].anatomical_region.value_counts().to_dict()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, comp = build()
    rng = np.random.default_rng(0)
    res, tabs = {}, {}
    policies = {}
    for version in ('before', 'after', 'after_v2'):
        t = decide(df, comp, version)
        t.to_csv(OUT / f'decisions_{version}.csv', index=False)
        res[version] = with_ci(t, rng)
        tabs[version] = t
        if decide.chosen:
            policies[version] = decide.chosen
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    lab = oof[oof.quality.notna()]
    router = {'correct': int((lab.region == lab.region_prediction).sum()), 'n': int(len(lab))}
    proc = processing_stats()
    final = 'after_v2'
    (OUT / 'metrics.json').write_text(json.dumps({'protocol': __doc__, 'metrics': res, 'final_version': final,
                                                  'spine_policy_per_fold': policies, 'router_labelled': router,
                                                  'processing_cli_run': proc}, indent=2, ensure_ascii=False))
    A = res[final]
    Bm = res['before']
    A1 = res['after']
    rows = [('Качество, все области', 'quality_all'), ('Качество, позвоночник', 'quality_spine'), ('Качество, бедро', 'quality_hip')] + \
           [(f'Причина: {c}', c) for c in REASONS]
    L = ['| Срез | n / нарушений | Чувствительность | Специфичность | F1 | ROC-AUC | PR-AUC |', '|---|---|---|---|---|---|---|']
    for title, k in rows:
        m = A[k]
        L.append(f"| {title} | {m['n']} / {m['pos']} | {fmt(m, 'sensitivity')} | {fmt(m, 'specificity')} | {fmt(m, 'f1')} | "
                 f"{fmt(m, 'roc_auc')} | {fmt(m, 'pr_auc')} |")
    L += ['', f"Macro-F1 пяти причин: **{fmt(A['reasons_macro_f1'], 'value')}** (до: {fmt(Bm['reasons_macro_f1'], 'value')}). "
          f"Доля нарушений без установленной причины: {A['quality_unspecified_among_defects']['value']['value']:.1%} "
          f"(до: {Bm['quality_unspecified_among_defects']['value']['value']:.1%})."]
    C = ['| Срез | F1 до → после | ROC-AUC до → после | Чувствительность до → после | Специфичность до → после |', '|---|---|---|---|---|']
    for title, k in rows:
        def g(key):
            if key not in A[k]:
                return '—'
            a, b = Bm[k][key]['value'], A[k][key]['value']
            after = f'**{b:.3f}** ↑' if b > a + 0.0005 else (f'{b:.3f} ↓' if b < a - 0.0005 else f'{b:.3f} =')
            return f'{a:.3f} → {after}'
        C.append(f"| {title} | {g('f1')} | {g('roc_auc')} | {g('sensitivity')} | {g('specificity')} |")
    (OUT / 'table_main.md').write_text('\n'.join(L) + '\n')
    (OUT / 'table_before_after.md').write_text('\n'.join(C) + '\n')
    print('\n'.join(L))
    print()
    print('\n'.join(C))
    print('router', router, '| spine policy per fold:', policies)
    for k in ('quality_all', 'quality_spine', 'quality_hip', 'spine_artifact'):
        print('v1 (prev table):', k, 'F1', round(A1[k]['f1']['value'], 3), 'AUC', round(A1[k]['roc_auc']['value'], 3))
    print('processing', json.dumps(proc, ensure_ascii=False))


if __name__ == '__main__':
    main()
