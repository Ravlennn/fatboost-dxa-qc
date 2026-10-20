"""Paired comparison of the new-model arms with the control (docs/PREREG_NEW_MODELS.md).

Control: spine rows of decisions_isolated.csv + decisions_hip_nested.csv (the published fully
isolated build). An arm replaces the spine and/or hip decisions by its own files. Every metric is
computed on the same 249 frames; the interval is a bootstrap over studies with a Bonferroni level
over the arms evaluated together. An arm is accepted only if its primary endpoint's interval lies
entirely above zero.

  python scripts/compare_arms.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parents[1] / 'outputs/final_validation_isolated'
CAUSES = ['spine_scan_range', 'spine_axis_tilt', 'spine_artifact', 'hip_positioning', 'hip_roi_margins']
# name: (spine decisions, hip decisions, primary endpoint)
ARMS = {
    'A  CNN позвоночника': ('decisions_spine_nested_A.csv', None, ('auc', 'spine')),
    'B  общая сеть': ('decisions_spine_nested_B.csv', 'decisions_hip_nested_B.csv', ('auc', 'all')),
    'C  вырез вертела': (None, 'decisions_hip_nested_C.csv', ('auc', 'hip')),
    'D  RAD-DINO + LoRA': (None, 'decisions_hip_nested_D.csv', ('auc', 'hip')),
}


def assemble(spine_file, hip_file) -> pd.DataFrame:
    sp = pd.read_csv(OUT / (spine_file or 'decisions_isolated.csv'))
    hp = pd.read_csv(OUT / (hip_file or 'decisions_hip_nested.csv'))
    t = pd.concat([sp[sp.region.eq('spine')], hp[~hp.region.eq('spine')]], ignore_index=True)
    return t.set_index('image_uid').sort_index()


def metric(t: pd.DataFrame, kind: str, region: str, idx=None) -> float:
    d = t if idx is None else t.iloc[idx]
    if region in CAUSES:
        d = d[d['y_' + region].notna()]
        y, s, c = d['y_' + region].astype(int), d['score_' + region], d['reported_' + region].astype(int)
    else:
        if region != 'all':
            d = d[d.region.eq('spine')] if region == 'spine' else d[~d.region.eq('spine')]
        y, s, c = d.y_quality.astype(int), d.score_quality, d.quality_class.astype(int)
    if y.nunique() < 2:
        return np.nan
    return roc_auc_score(y, s) if kind == 'auc' else f1_score(y, c, zero_division=0)


def main(n: int = 4000):
    ctrl = assemble(None, None)
    avail = {k: v for k, v in ARMS.items() if all(f is None or (OUT / f).exists() for f in v[:2])}
    if not avail:
        raise SystemExit('Нет готовых вариантов')
    alpha = 0.05 / len(ARMS)            # Bonferroni over all four pre-registered arms, even if fewer are ready
    studies = ctrl.study.unique()
    pos = {s: np.where(ctrl.study.values == s)[0] for s in studies}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([pos[s] for s in rng.choice(studies, len(studies))]) for _ in range(n)]
    rows = [('auc', 'all'), ('f1', 'all'), ('auc', 'spine'), ('f1', 'spine'), ('auc', 'hip'), ('f1', 'hip')] + \
           [(k, c) for c in CAUSES for k in ('f1', 'auc')]
    names = {'all': 'качество, все', 'spine': 'качество, позвоночник', 'hip': 'качество, бедро'}
    print(f'Интервал {100 * (1 - alpha):.2f}% (Бонферрони на {len(ARMS)} варианта), бутстрэп по исследованиям, {n} повторов.')
    for arm, (sf, hf, primary) in avail.items():
        t = assemble(sf, hf).loc[ctrl.index]
        print(f'\n=== {arm}   (главная метрика: {primary[0].upper()} {names[primary[1]]})')
        for kind, region in rows:
            c0, c1 = metric(ctrl, kind, region), metric(t, kind, region)
            if np.isnan(c0) or abs(c1 - c0) < 1e-12 and (kind, region) != primary:
                continue
            d = np.array([metric(t, kind, region, b) - metric(ctrl, kind, region, b) for b in boots])
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            mark = ''
            if (kind, region) == primary:
                mark = '  ← ГЛАВНАЯ: ' + ('ПРИНЯТЬ' if lo > 0 else 'отклонить')
            print(f'   {kind.upper():3s} {names.get(region, region):24s} {c0:.3f} → {c1:.3f}  Δ {c1 - c0:+.3f} [{lo:+.3f}; {hi:+.3f}]{mark}')


if __name__ == '__main__':
    main()
