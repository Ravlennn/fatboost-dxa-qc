"""Prevalence-matched thresholds for the two rare causes (scan range: 6/99, ROI margins: 7/150).

The release picks each cause threshold by maximising F1 inside the training folds, where only 4-5
positives are left, so the threshold jumps between folds. The alternative flags, in the held-out
fold, the same share of frames as the share of violations in the training folds: the threshold is
the (1 - prevalence) quantile of the training folds' out-of-fold scores. Nothing uses the held-out labels.

The quality class is rebuilt the way the release builds it: spine = any reported reason; hip causes
are reported only on frames the hip quality head calls defective (the hip quality class is unchanged).

Criterion fixed before the run: a cause switches only if its ΔF1 interval (bootstrap over studies,
97.5 % = Bonferroni over the two causes) lies above zero. Spine quality F1 is reported alongside.

  python scripts/eval_rare_cause_threshold.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
DEC = ROOT / 'outputs/final_validation_isolated/decisions_combined.csv'


def prevalence_flags(d: pd.DataFrame, score: str, label: str) -> np.ndarray:
    out = np.zeros(len(d), dtype=bool)
    for f in sorted(d.fold.unique()):
        tr, te = (d.fold != f).values, (d.fold == f).values
        prev = d.loc[tr, label].mean()
        thr = np.quantile(d.loc[tr, score], 1 - prev)
        out[te] = d.loc[te, score].values >= thr
    return out


def boot(d, y, a, b, n=4000, alpha=0.025):
    rng = np.random.default_rng(0)
    studies = d.study.unique()
    idx = {s: np.where(d.study.values == s)[0] for s in studies}
    diffs = []
    for _ in range(n):
        t = np.concatenate([idx[s] for s in rng.choice(studies, len(studies))])
        if y[t].sum() > 0:
            diffs.append(f1_score(y[t], b[t], zero_division=0) - f1_score(y[t], a[t], zero_division=0))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return f1_score(y, a, zero_division=0), f1_score(y, b, zero_division=0), float(np.mean(diffs)), lo, hi


def counts(y, p):
    return f'TP {int((y & p).sum())}, FP {int((~y & p).sum())}, FN {int((y & ~p).sum())}'


def main():
    d = pd.read_csv(DEC)
    s = d[d.region == 'spine'].reset_index(drop=True)
    h = d[d.region != 'spine'].reset_index(drop=True)

    y = s.y_spine_scan_range.astype(bool).values
    old = s.reported_spine_scan_range.astype(bool).values
    new = prevalence_flags(s, 'score_spine_scan_range', 'y_spine_scan_range')
    r = boot(s, y, old, new)
    print(f'Границы сканирования (n={len(s)}, нарушений {y.sum()}):')
    print(f'   релиз  F1 {r[0]:.3f}  ({counts(y, old)})')
    print(f'   доля   F1 {r[1]:.3f}  ({counts(y, new)})   ΔF1 {r[2]:+.3f} [{r[3]:+.3f}; {r[4]:+.3f}]  → {"ПРИНЯТЬ" if r[3] > 0 else "отклонить"}')
    yq = s.y_quality.astype(bool).values
    q_old = s.quality_class.astype(bool).values
    q_new = new | s.reported_spine_axis_tilt.astype(bool).values | s.reported_spine_artifact.astype(bool).values
    rq = boot(s, yq, q_old, q_new)
    print(f'   качество позвоночника: F1 {rq[0]:.3f} → {rq[1]:.3f}  Δ {rq[2]:+.3f} [{rq[3]:+.3f}; {rq[4]:+.3f}]')

    y = h.y_hip_roi_margins.astype(bool).values
    old = h.reported_hip_roi_margins.astype(bool).values
    new = prevalence_flags(h, 'score_hip_roi_margins', 'y_hip_roi_margins') & h.quality_class.astype(bool).values
    r = boot(h, y, old, new)
    print(f'\nПоля ROI (n={len(h)}, нарушений {y.sum()}):')
    print(f'   релиз  F1 {r[0]:.3f}  ({counts(y, old)})')
    print(f'   доля   F1 {r[1]:.3f}  ({counts(y, new)})   ΔF1 {r[2]:+.3f} [{r[3]:+.3f}; {r[4]:+.3f}]  → {"ПРИНЯТЬ" if r[3] > 0 else "отклонить"}')
    print('   (класс качества бедра не меняется)')


if __name__ == '__main__':
    main()
