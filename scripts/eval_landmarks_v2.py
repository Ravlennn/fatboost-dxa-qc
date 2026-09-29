"""Evaluate landmarks v2 OOF predictions: point error vs manual labels and reason AUC/F1.

Features (all from held-out-fold models, so no image sees its own labels):
  hip   : lt_visibility (mean heatmap peak), lt_prominence_mm (apex to base line), lt_base_mm
  spine : th12_visibility, th12_margin_vert, iliac_visibility, iliac_top_rel (crest vs L4 bottom)
Compared with release CNN OOF with the protocol of run_landmark_fusion.py
(pre-specified feature sets, inner-fold threshold, paired study bootstrap).

  python scripts/eval_landmarks_v2.py --oof outputs/landmarks_v2/oof_points.json --labels outputs/landmark_labeling/landmarks_manual_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402

MM = 0.6
SPINE_SEQ = ('L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom')


def v2_features(o, region):
    p = {k: np.array(v) for k, v in o['points'].items()}
    pk = o['peak']
    if region == 'spine':
        ys = [p[k][1] for k in SPINE_SEQ]
        vh = max(float(np.median(np.diff(ys))), 1.0)
        il = [pk['iliac_left_top'], pk['iliac_right_top']]
        crest_y = min(p['iliac_left_top'][1], p['iliac_right_top'][1])
        return {'th12_vis': pk['th12_mid'], 'th12_margin': p['th12_mid'][1] / vh,
                'iliac_vis': max(il), 'iliac_vis_min': min(il), 'iliac_top_rel': (crest_y - p['L4_bottom'][1]) / vh}
    a, b, t = p['lt_apex'], p['lt_base_top'], p['lt_base_bottom']
    base = b - t
    v = a - t
    prom = abs(base[0] * v[1] - base[1] * v[0]) / max(np.linalg.norm(base), 1e-6)  # 2-D cross product
    return {'lt_vis': float(np.mean([pk['lt_apex'], pk['lt_base_top'], pk['lt_base_bottom']])),
            'lt_prominence_mm': float(prom * MM), 'lt_base_mm': float(np.linalg.norm(base) * MM)}


TASKS = {  # name: (subset, label, cnn column, v2 features, also include v1 landmark features)
    'hip_positioning_lt': ('hip', 'v_rotation', 'p_hip_positioning', ['lt_vis', 'lt_prominence_mm', 'lt_base_mm']),
    'spine_scan_range_v2': ('spine', 'v_layout', 'p_spine_scan_range',
                            ['th12_vis', 'th12_margin', 'iliac_vis', 'iliac_top_rel', 'min_margin', 'iliac_max']),
    'quality_hip_lt': ('hip', 'quality', 'p_quality', ['lt_vis', 'lt_prominence_mm']),
}


def point_errors(oof, labels):
    rows = []
    for r in labels:
        o = oof.get(r['id'])
        if not o:
            continue
        for n, stt in r['status'].items():
            if stt == 'set' and n in o['points']:
                e = np.linalg.norm(np.array(o['points'][n]) - np.array(r['points'][n])) * MM
                rows.append({'point': n, 'err_mm': e, 'peak': o['peak'][n], 'visible': 1})
            elif stt == 'invisible' and n in o['peak']:
                rows.append({'point': n, 'err_mm': np.nan, 'peak': o['peak'][n], 'visible': 0})
    df = pd.DataFrame(rows)
    out = {}
    for n, g in df.groupby('point'):
        d = {'n_set': int(g.visible.sum()), 'n_invisible': int((g.visible == 0).sum()),
             'median_err_mm': float(g.err_mm.median()) if g.visible.any() else None}
        if g.visible.nunique() == 2:
            d['visibility_auc'] = float(roc_auc_score(g.visible, g.peak))
        out[n] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', type=Path, default=ROOT / 'outputs/landmarks_v2/oof_points.json')
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/landmarks_v2/eval')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    oof = json.loads(args.oof.read_text())
    labels = json.loads(args.labels.read_text())
    res = {'point_errors_oof': point_errors(oof, labels)}

    base = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    v1 = pd.DataFrame(json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')))
    df = base.merge(v1[['image_uid', 'study', 'measurements']], on=['study', 'image_uid'], validate='1:1')
    df = df[df.image_uid.isin(oof)]
    lines = ['| Задача | n/pos | Вариант | AUC | AP | F1 | ΔAUC vs CNN [95%] |', '|---|---|---|---:|---:|---:|---|']
    for name, (sub, lab, pcol, feats) in TASKS.items():
        d = df[df.region.eq('spine')] if sub == 'spine' else df[df.region.str.startswith('hip')]
        d = d[d[lab].notna() & d[pcol].notna()].copy()
        X = F.features(d)
        v2 = pd.DataFrame([v2_features(oof[u], 'spine' if sub == 'spine' else 'hip') for u in d.image_uid], index=d.index)
        X = pd.concat([X, v2], axis=1)
        X['cnn'] = F.logit(d[pcol].values)
        y, g, fo = d[lab].astype(int).values, d.study.values, d.fold.values
        scores = {'cnn': (d[pcol].values, F.cnn_decisions(d[pcol].values, y, g, fo))}
        scores['landmarks_v2'] = F.fit_predict(X, y, g, fo, feats)
        scores['fusion_v2'] = F.fit_predict(X, y, g, fo, ['cnn'] + feats)
        r = {}
        for v, (s, dec) in scores.items():
            r[v] = {'auc': float(roc_auc_score(y, s)), 'ap': float(average_precision_score(y, s)), 'f1': float(f1_score(y, dec))}
            ci = F.paired_boot(y, scores['cnn'][0], s, g)[0] if v != 'cnn' else None
            r[v]['delta_auc_ci95'] = ci
            lines.append(f"| {name} | {len(y)}/{y.sum()} | {v} | {r[v]['auc']:.3f} | {r[v]['ap']:.3f} | {r[v]['f1']:.3f} | "
                         + (f'[{ci[0]:+.3f}; {ci[1]:+.3f}]' if ci else '') + ' |')
        res[name] = r
    (args.out / 'results.json').write_text(json.dumps(res, indent=2))
    (args.out / 'table.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(res['point_errors_oof'], indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
