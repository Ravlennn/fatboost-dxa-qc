"""hip_roi_margins as the direct TZ rule on landmark ROI margins, vs the release CNN.

TZ 2.3: the region of interest must show >= 3 cm above and below it and >= 2 cm laterally.
Features (pre-specified): the four margins in mm from the landmark ROI box, their minimum,
and the TZ deficits max(0, 30 - top/bottom) and max(0, 20 - lateral/medial).
Protocol of run_landmark_fusion.py (release folds, inner-fold threshold, paired bootstrap).

  python scripts/eval_roi_rule.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402

OUT = ROOT / 'outputs/roi_rule_v1'
NEED_AXIAL, NEED_LATERAL = 30.0, 20.0  # mm, TZ 2.3


def features(d):
    m = pd.json_normalize(d['measurements']).set_index(d.index)
    right = d['region'].eq('hip_right')
    lat = np.where(right, m['roi_left_mm'], m['roi_right_mm'])    # trochanter side
    med = np.where(right, m['roi_right_mm'], m['roi_left_mm'])
    f = pd.DataFrame({'roi_top': m['roi_top_mm'], 'roi_bottom': m['roi_bottom_mm'],
                      'roi_lateral': lat, 'roi_medial': med}, index=d.index)
    f['roi_min'] = f.min(axis=1)
    # TZ deficits: how many mm are missing relative to the requirement (0 = requirement met).
    f['deficit_axial'] = np.maximum(0, NEED_AXIAL - np.minimum(f.roi_top, f.roi_bottom))
    f['deficit_lateral'] = np.maximum(0, NEED_LATERAL - np.minimum(lat, med))
    f['deficit_max'] = f[['deficit_axial', 'deficit_lateral']].max(axis=1)
    return f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    v1 = pd.DataFrame(json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')))
    d = oof.merge(v1[['image_uid', 'study', 'measurements']], on=['study', 'image_uid'], validate='1:1')
    d = d[d.region.str.startswith('hip') & d.v_roi.notna()].copy()
    X = features(d)
    X['cnn'] = F.logit(d.p_hip_roi_margins.values)
    y, g, fo = d.v_roi.astype(int).values, d.study.values, d.fold.values
    rule = ['roi_top', 'roi_bottom', 'roi_lateral', 'roi_medial', 'roi_min', 'deficit_axial', 'deficit_lateral']
    # 'axial' is EXPLORATORY: chosen after seeing that the lateral margins carry no signal
    # (univariate AUC 0.52/0.58) while the axial ones do (0.88/0.87). Marked as such in the report.
    axial = ['roi_top', 'roi_bottom', 'deficit_axial']
    v = {'cnn': (d.p_hip_roi_margins.values, F.cnn_decisions(d.p_hip_roi_margins.values, y, g, fo)),
         'tz_rule_head': F.fit_predict(X, y, g, fo, rule),
         'axial_head (exploratory)': F.fit_predict(X, y, g, fo, axial),
         'fusion': F.fit_predict(X, y, g, fo, ['cnn'] + rule),
         'fusion_axial (exploratory)': F.fit_predict(X, y, g, fo, ['cnn'] + axial)}
    # Literal TZ threshold, no fitting at all.
    lit = (X.deficit_max > 0).values
    res, L = {}, ['| Вариант | AUC | AP | F1 | Sens | Spec | ΔAUC vs CNN [95%] |', '|---|---:|---:|---:|---:|---:|---|']
    for name, (s, dec) in v.items():
        r = {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s), 'f1': f1_score(y, dec),
             'sens': float(dec[y == 1].mean()), 'spec': float((~dec[y == 0]).mean())}
        if name != 'cnn':
            r['d_auc'], r['d_ap'] = F.paired_boot(y, v['cnn'][0], s, g)
        res[name] = r
        ci = f"[{r['d_auc'][0]:+.3f}; {r['d_auc'][1]:+.3f}]" if 'd_auc' in r else ''
        L.append(f"| {name} | {r['auc']:.3f} | {r['ap']:.3f} | {r['f1']:.3f} | {r['sens']:.2f} | {r['spec']:.2f} | {ci} |")
    res['literal_tz_threshold'] = {'f1': f1_score(y, lit), 'sens': float(lit[y == 1].mean()),
                                   'spec': float((~lit[y == 0]).mean()), 'flagged': int(lit.sum())}
    L.append(f"| буквальное правило ТЗ (без обучения) | — | — | {res['literal_tz_threshold']['f1']:.3f} | "
             f"{res['literal_tz_threshold']['sens']:.2f} | {res['literal_tz_threshold']['spec']:.2f} | |")
    res['margins_by_label'] = {k: X.loc[y == int(k), ['roi_top', 'roi_bottom', 'roi_lateral', 'roi_medial']].median().round(1).to_dict()
                               for k in ('0', '1')}
    (OUT / 'eval.json').write_text(json.dumps(res, indent=2, default=float))
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('median margins (mm) by label:', json.dumps(res['margins_by_label']))
    print('positives with any TZ deficit:', int(lit[y == 1].sum()), 'of', int(y.sum()))


if __name__ == '__main__':
    main()
