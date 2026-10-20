"""Metrics for every violation type from the fully isolated validation, image and study level.

Input: outputs/final_validation_isolated/decisions_combined.csv (scripts/combine_isolated.py) and the
label-free landmark measurements outputs/landmarks_v1/ours/predictions.json. The release decision logic is
replayed on the isolated scores, in the order of release_predictor.py:
  1. hip ROI margins: the learned head is replaced by the image-edge rule (roi_edge_score); a box cut off
     by the image edge forces a defect;
  2. a defect without a cause above its threshold gets the closest cause.
Every variant is reported so the effect of each rule is visible. Metrics and 95% CI: evaluate_final.with_ci
(bootstrap over studies). Study level: a study is a defect if any of its images is; score = maximum.

  python scripts/per_type_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_final as EF  # noqa: E402
from dxaqc.release_predictor import roi_edge_score  # noqa: E402

OUT = ROOT / 'outputs/final_validation_isolated'
R = list(EF.REASONS)
ROI = 'hip_roi_margins'
NAMES = {'quality_all': 'Качество, все области', 'quality_spine': 'Качество, позвоночник', 'quality_hip': 'Качество, бедро',
         'spine_scan_range': 'Границы сканирования', 'spine_axis_tilt': 'Наклон оси позвоночника',
         'spine_artifact': 'Посторонние предметы', 'hip_positioning': 'Позиционирование (ротация) бедра',
         'hip_roi_margins': 'Поля области интереса бедра'}


def replay(t: pd.DataFrame, roi_rule: bool, closest: bool, margins: pd.Series) -> pd.DataFrame:
    t = t.copy()
    hip = t.region.str.startswith('hip').values
    if roi_rule:
        score = np.array([roi_edge_score(m)[1] if h else np.nan for m, h in zip(margins, hip)])
        t['score_' + ROI] = np.where(hip, score / 0.5, np.nan)          # ratio to the 0.5 threshold
        cut = hip & (score >= 0.5)
        t.loc[cut, 'quality_class'] = 1                                 # forced, as in the release
        t.loc[cut, 'score_quality'] = np.maximum(t.loc[cut, 'score_quality'], 1.0)
    bad = t.quality_class.eq(1).values
    for c in R:
        t['reported_' + c] = bad & (t['score_' + c].fillna(0).values >= 1)
    if closest:
        sc = t[['score_' + c for c in R]].to_numpy(float)
        best = np.argmax(np.where(np.isnan(sc), -np.inf, sc), axis=1)
        none = bad & ~t[['reported_' + c for c in R]].any(axis=1).values
        for i in np.flatnonzero(none):
            t.loc[t.index[i], 'reported_' + R[best[i]]] = True
    t['quality_unspecified'] = t.quality_class.eq(1) & ~t[['reported_' + c for c in R]].any(axis=1)
    return t


def study_level(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby('study', sort=False)
    s = pd.DataFrame({'study': list(g.groups), 'region': 'study', 'y_quality': g.y_quality.max().values,
                      'score_quality': g.score_quality.max().values, 'quality_class': g.quality_class.max().values})
    for c in R:
        s['score_' + c] = g['score_' + c].max().values
        s['y_' + c] = g['y_' + c].max().values
        s['reported_' + c] = g['reported_' + c].max().values.astype(bool)
    s['quality_unspecified'] = s.quality_class.eq(1) & ~s[['reported_' + c for c in R]].any(axis=1)
    return s


def table(m: dict, title: str) -> list[str]:
    L = [f'### {title}', '', '| Тип | n / брак | TP | FN | FP | TN | Чувствительность | Специфичность | F1 | ROC-AUC | PR-AUC |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for k in NAMES:
        if k in m:
            r = m[k]
            L.append(f"| {NAMES[k]} | {r['n']} / {r['pos']} | {r['tp']} | {r['fn']} | {r['fp']} | {r['tn']} | "
                     + ' | '.join(EF.fmt(r, x) for x in ('sensitivity', 'specificity', 'f1', 'roc_auc', 'pr_auc')) + ' |')
    L += ['', f"Средний F1 по пяти типам: {EF.fmt(m['reasons_macro_f1'], 'value')}; "
              f"брак без указанной причины: {m['quality_unspecified_among_defects']['value']['value']:.1%}", '']
    return L


def main() -> None:
    t0 = pd.read_csv(OUT / 'decisions_combined.csv')
    pred = {r['image_uid']: r['measurements'] for r in json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json'))}
    margins = pd.Series([pred.get(u) for u in t0.image_uid], index=t0.index)
    variants = {'isolated': (False, False), 'closest': (False, True), 'release': (True, True)}
    res, md = {}, []
    for name, (roi, closest) in variants.items():
        t = replay(t0, roi, closest, margins)
        res[name] = {'image': EF.with_ci(t, np.random.default_rng(0)), 'study': EF.with_ci(study_level(t), np.random.default_rng(0))}
        if name == 'release':
            t.to_csv(OUT / 'decisions_release_rules.csv', index=False)
    md += table(res['release']['image'], 'По снимкам: логика релиза (рамка за краем снимка + ближайшая причина)')
    md += table(res['release']['study'], 'По исследованиям: логика релиза')
    md += table(res['closest']['image'], 'По снимкам: без правила рамки, с ближайшей причиной (как до 18.10)')
    md += table(res['isolated']['image'], 'По снимкам: без обоих правил (combined_table.md)')
    (OUT / 'per_type_metrics.json').write_text(json.dumps(res, indent=1, default=float))
    (OUT / 'per_type_metrics.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
