"""Paired comparison of fully isolated hip pipelines: control vs arms (docs/PREREG_HIP_LOSS.md).

Reads outputs/final_validation_isolated/decisions_hip_nested{suffix}.csv written by scripts/eval_hip_nested.py.
Primary endpoint: ΔROC-AUC of hip quality. Bootstrap over studies; with k arms the interval is the
two-sided (1 - 0.05/k) one (Bonferroni). An arm is accepted only if that interval lies above zero.

  python scripts/compare_hip_arms.py _gce _asl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parents[1] / 'outputs/final_validation_isolated'


def load(suffix: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f'decisions_hip_nested{suffix}.csv').set_index('image_uid')


def main(arms: list[str], n: int = 4000) -> None:
    alpha = 0.05 / len(arms)
    ctrl = load('')
    rows = []
    for arm in arms:
        a = load(arm).loc[ctrl.index]
        studies = ctrl.study.unique()
        idx = {s: np.where(ctrl.study.values == s)[0] for s in studies}
        rng = np.random.default_rng(0)
        res = {}
        for name, y_col, s_col, c_col in (('AUC качества бедра (главная)', 'y_quality', 'score_quality', None),
                                          ('AUC ротации', 'y_hip_positioning', 'score_hip_positioning', None),
                                          ('F1 качества бедра', 'y_quality', None, 'quality_class')):
            m = ctrl[y_col].notna().values
            y = ctrl[y_col].values[m].astype(int)
            if s_col:
                f = lambda t, d: roc_auc_score(y[t], d[s_col].values[m][t])
            else:
                f = lambda t, d: f1_score(y[t], d[c_col].values[m][t].astype(int))
            pos = np.arange(m.sum())
            sub_idx = {s: np.intersect1d(np.where(ctrl.study.values[m] == s)[0], pos) for s in studies}
            diffs = []
            for _ in range(n):
                t = np.concatenate([sub_idx[s] for s in rng.choice(studies, len(studies))])
                if len(np.unique(y[t])) == 2:
                    diffs.append(f(t, a) - f(t, ctrl))
            lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            full = np.arange(len(y))
            res[name] = (f(full, ctrl), f(full, a), np.mean(diffs), lo, hi)
        rows.append((arm, res))
    level = round(100 * (1 - alpha), 2)
    print(f'Контроль: weighted BCE. Интервал {level}% (Бонферрони на {len(arms)} ветки), бутстрэп по исследованиям.\n')
    for arm, res in rows:
        print(f'Ветка {arm}:')
        for name, (c, v, d, lo, hi) in res.items():
            print(f'   {name:30s} контроль {c:.3f}  ветка {v:.3f}  Δ {d:+.3f} [{lo:+.3f}; {hi:+.3f}]')
        _, _, _, lo, _ = res['AUC качества бедра (главная)']
        print(f"   Решение по заранее заданному критерию: {'ПРИНЯТЬ' if lo > 0 else 'отклонить'}\n")


if __name__ == '__main__':
    main(sys.argv[1:] or ['_gce', '_asl'])
