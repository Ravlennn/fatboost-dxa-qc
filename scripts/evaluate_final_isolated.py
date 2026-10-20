"""Final validation with fully isolated outer folds (fixes the OOF-stacking dependence).

Problem with scripts/evaluate_final.py: the CNN scores are a single global 5-fold OOF, so the
feature of a TRAINING row was produced by a model that had seen the evaluated fold. The second
level therefore depends on the outer fold indirectly.

Here every level is refit inside the outer fold:
  * spine — fully isolated. The DenseNet121 encoder is frozen and never saw labels, so only its
    linear heads are refit on the outer-train rows; the CNN feature of the training rows is an
    inner out-of-fold prediction, so the second level never sees its own row's score.
  * hip — the CNN is fine-tuned on labels, so full isolation would need 20 retrainings (each
    outer fold needs models trained without that fold AND without the inner fold). Instead the
    hip runs here on label-free geometry only (landmarks + lesser trochanter): fully isolated,
    and therefore a lower bound on what the hip branch can do.

  python scripts/evaluate_final_isolated.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_final as EF  # noqa: E402
import run_landmark_fusion as F  # noqa: E402
from dxaqc.lesser_trochanter import head_features as lt_features  # noqa: E402

OUT = ROOT / 'outputs/final_validation_isolated'
FROZEN_C = 0.001  # as in the release frozen-feature heads
REASONS = EF.REASONS


def frozen_head():
    return make_pipeline(StandardScaler(), LogisticRegression(C=FROZEN_C, class_weight='balanced',
                                                              max_iter=2000, random_state=0))


def inner_oof(X, y, g, fit_fn, k=3):
    """Out-of-fold scores within the training part (so the next level never sees its own score)."""
    out = np.zeros(len(y))
    for a, b in GroupKFold(k).split(X, y, g):
        out[b] = fit_fn(X[a], y[a]).predict_proba(X[b])[:, 1] if y[a].any() else 0.0
    return out


def level2(Xtr, ytr, gtr, Xte, cols):
    """Fit a small logistic head on the training part; threshold from inner folds."""
    inner = np.zeros(len(ytr))
    for a, b in GroupKFold(3).split(Xtr[cols].values, ytr, gtr):
        inner[b] = F.model().fit(Xtr[cols].values[a], ytr[a]).predict_proba(Xtr[cols].values[b])[:, 1] if ytr[a].any() else 0.0
    thr = F.best_threshold(ytr, inner) if ytr.any() else 0.5
    m = F.model().fit(Xtr[cols].values, ytr)
    return m.predict_proba(Xte[cols].values)[:, 1], thr


def evaluate(permute_test: bool = False, seed: int = 0) -> pd.DataFrame:
    """Nested decisions with isolated outer folds.

    permute_test=True shuffles the labels of the outer-test rows before every fold. Nothing in
    the pipeline may read them, so the returned scores must be bit-identical — that is the
    isolation test (tests/test_isolated_folds.py).
    """
    feats = np.load(ROOT / 'outputs/release_loss_v1/features.npy')
    meta = json.load(open(ROOT / 'outputs/release_loss_v1/features_meta.json'))
    fidx = {r['image_uid']: i for i, r in enumerate(meta['identity']['rows'])}
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    v1 = pd.DataFrame(json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')))
    df = oof.merge(v1[['image_uid', 'study', 'measurements']], on=['study', 'image_uid'], validate='1:1')
    df = df[df.quality.notna()].reset_index(drop=True)
    art = json.load(open(ROOT / 'outputs/artifacts_lm_v1/measurements.json'))
    lt = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))

    rows = {c: np.full(len(df), np.nan) for c in
            ['score_quality'] + [f'score_{c}' for c in REASONS] + [f'ratio_{c}' for c in REASONS]}

    truth = df[['quality'] + list(REASONS.values())].copy()  # metrics always use the real labels
    sp = df.region.eq('spine').values
    hp = ~sp
    Xall = feats[[fidx[u] for u in df.image_uid]]
    Xlm = F.features(df)                                   # landmark measurements (label-free)
    Xart = pd.DataFrame([art.get(u, {'art_wire': 0.0, 'art_metal': 0.0}) for u in df.image_uid], index=df.index)
    Xlt = pd.DataFrame([lt_features({'lt_prominence_mm': lt[u]['lt_prominence_mm'], 'lt_area_mm2': lt[u]['lt_area_mm2'],
                                     'lt_edge_bump_mm': lt[u]['lt_edge_bump_mm']}) if u in lt else
                        lt_features(None) for u in df.image_uid], index=df.index)

    rng_p = np.random.default_rng(seed)
    labels = ['quality'] + list(REASONS.values())
    for f in sorted(df.fold.unique()):
        te = (df.fold == f).values
        tr = ~te
        Y = df
        if permute_test:  # outer-test labels must be unreachable from here on
            Y = df.copy()
            for c in labels:
                Y.loc[te, c] = rng_p.permutation(Y.loc[te, c].values)
        # ---------- spine, fully isolated ----------
        s_tr, s_te = tr & sp, te & sp
        gtr = df.study.values[s_tr]
        for code, lab in (('spine_artifact', 'v_artifact'),):
            y = Y[lab].values[s_tr].astype(int)
            cnn_tr = inner_oof(Xall[s_tr], y, gtr, lambda a, b: frozen_head().fit(a, b))
            cnn_te = frozen_head().fit(Xall[s_tr], y).predict_proba(Xall[s_te])[:, 1]
            Xtr = Xart.loc[s_tr].copy()
            Xtr['cnn'] = F.logit(cnn_tr)
            Xte = Xart.loc[s_te].copy()
            Xte['cnn'] = F.logit(cnn_te)
            s, thr = level2(Xtr, y, gtr, Xte, ['cnn', 'art_wire', 'art_metal'])
            rows[f'score_{code}'][s_te] = s
            rows[f'ratio_{code}'][s_te] = s / max(thr, 1e-9)
        for code, lab, cols in (('spine_axis_tilt', 'v_axis', F.TASKS['spine_axis_tilt'][3]),
                                ('spine_scan_range', 'v_layout', F.TASKS['spine_scan_range_iliac'][3])):
            y = Y[lab].values[s_tr].astype(int)
            s, thr = level2(Xlm.loc[s_tr], y, gtr, Xlm.loc[s_te], cols)
            rows[f'score_{code}'][s_te] = s
            rows[f'ratio_{code}'][s_te] = s / max(thr, 1e-9)
        # ---------- hip ----------
        h_tr, h_te = tr & hp, te & hp
        gh = df.study.values[h_tr]
        for code, lab in (('hip_positioning', 'v_rotation'), ('hip_roi_margins', 'v_roi')):
            y = Y[lab].values[h_tr].astype(int)
            if code == 'hip_positioning':
                s, thr = level2(Xlt.loc[h_tr], y, gh, Xlt.loc[h_te], ['lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump'])
            else:  # ROI margins from the landmark box (label-free features)
                cols = ['roi_top', 'roi_bottom', 'roi_lateral', 'roi_medial', 'roi_min']
                s, thr = level2(Xlm.loc[h_tr], y, gh, Xlm.loc[h_te], cols)
            rows[f'score_{code}'][h_te] = s
            rows[f'ratio_{code}'][h_te] = s / max(thr, 1e-9)
        y = Y['quality'].values[h_tr].astype(int)
        s, thr = level2(Xlt.loc[h_tr], y, gh, Xlt.loc[h_te], ['lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump'])
        rows['score_quality'][h_te] = s / max(thr, 1e-9)

    # spine defect = any reason found (the policy selected earlier), hip = its own head
    ratio_spine = np.fmax.reduce([rows[f'ratio_{c}'] for c in ('spine_axis_tilt', 'spine_scan_range', 'spine_artifact')])
    q = np.where(sp, ratio_spine, rows['score_quality'])
    t = pd.DataFrame({'study': df.study, 'image_uid': df.image_uid, 'region': df.region, 'fold': df.fold,
                      'y_quality': truth.quality.astype(int), 'score_quality': q, 'quality_class': (q >= 1).astype(int)})
    for c in REASONS:
        applicable = sp if c.startswith('spine') else hp
        t['score_' + c] = np.where(applicable, rows[f'ratio_{c}'], np.nan)
        t['reported_' + c] = applicable & (q >= 1) & (rows[f'ratio_{c}'] >= 1)
        t['y_' + c] = truth[REASONS[c]]
    t['quality_unspecified'] = (q >= 1) & ~t[[f'reported_{c}' for c in REASONS]].any(axis=1)
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = evaluate()
    t.to_csv(OUT / 'decisions_isolated.csv', index=False)
    m = EF.with_ci(t, np.random.default_rng(0))
    (OUT / 'metrics.json').write_text(json.dumps({'isolation': 'spine: full; hip: label-free geometry only',
                                                  'metrics': m}, indent=2, default=float))
    L = ['| Срез | n / нарушений | Чувствительность | Специфичность | F1 | ROC-AUC |', '|---|---|---|---|---|---|']
    for name in ('quality_all', 'quality_spine', 'quality_hip', *REASONS):
        r = m[name]
        L.append(f"| {name} | {r['n']} / {r['pos']} | {EF.fmt(r, 'sensitivity')} | {EF.fmt(r, 'specificity')} | "
                 f"{EF.fmt(r, 'f1')} | {EF.fmt(r, 'roc_auc')} |")
    L.append('')
    L.append(f"Macro-F1 причин: {EF.fmt(m['reasons_macro_f1'], 'value')}")
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
