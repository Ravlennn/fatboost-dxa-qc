"""Hip branch with a fully isolated CNN feature (consumes scripts/run_hip_nested_local.sh).

For outer fold f the second level is fitted on the outer-train rows whose CNN score comes from
outputs/hip_cnn/nested_f{f}_j{j}/oof.csv — a model trained without {f, j}. The test rows are scored
by outputs/hip_cnn/nested_base (trained without f, same recipe and GPU as the inner runs; the
release OOF is the fallback). Nothing at any level has seen fold f.

  python scripts/eval_hip_nested.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_final as EF  # noqa: E402
import run_landmark_fusion as F  # noqa: E402
from evaluate_final_isolated import level2  # noqa: E402
from dxaqc.lesser_trochanter import head_features as lt_features  # noqa: E402

OUT = ROOT / 'outputs/final_validation_isolated'
NESTED = ROOT / 'outputs/hip_cnn'
# code: (label column, column in the release OOF, column in the nested runs)
HEADS = {'hip_positioning': ('v_rotation', 'p_hip_positioning', 'p_v_rotation'),
         'hip_roi_margins': ('v_roi', 'p_hip_roi_margins', 'p_v_roi'),
         'quality': ('quality', 'p_quality', 'p_quality')}


def nested_scores(folds, suffix='', root=NESTED):
    """{(outer f, image_uid): row of nested CNN scores} for every outer-train row."""
    out = {}
    missing = []
    for f in folds:
        for j in folds:
            if f == j:
                continue
            p = root / f'nested_f{f}_j{j}{suffix}' / 'oof.csv'
            if not p.exists():
                missing.append(p.parent.name)
                continue
            d = pd.read_csv(p)
            d = d[d.fold == j]
            for r in d.itertuples():
                out[(f, r.image_uid)] = r
    return out, missing


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='', help='tag suffix of the nested runs (e.g. _full for a second recipe)')
    ap.add_argument('--root', type=Path, default=NESTED, help='folder of the main CNN runs (outputs/joint_cnn for arm B)')
    ap.add_argument('--extra', type=Path, action='append', default=[],
                    help='folder of another nested model whose scores join each head as one more feature (arms C, D)')
    ap.add_argument('--tag', default=None, help='output suffix (default: --suffix)')
    args = ap.parse_args(argv)
    out_tag = args.suffix if args.tag is None else args.tag
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    v1 = pd.DataFrame(json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')))
    df = oof.merge(v1[['image_uid', 'study', 'measurements']], on=['study', 'image_uid'], validate='1:1')
    df = df[df.quality.notna() & df.region.str.startswith('hip')].reset_index(drop=True)
    folds = sorted(df.fold.unique())
    nested, missing = nested_scores(folds, args.suffix, args.root)
    extras = []
    for root in args.extra:
        e_nested, e_missing = nested_scores(folds, '', root)
        missing += e_missing
        extras.append((e_nested, pd.read_csv(root / 'nested_base' / 'oof.csv').set_index('image_uid')))
    if missing:
        print(f'Нет {len(missing)} из {len(folds) * (len(folds) - 1)} вложенных прогонов; '
              f'сначала scripts/run_hip_nested_local.sh. Первые: {missing[:4]}')
        return 1
    base_p = args.root / f'nested_base{args.suffix}' / 'oof.csv'
    if base_p.exists():
        base_oof = pd.read_csv(base_p).set_index('image_uid')
        test_src = 'nested_base'
    else:
        base_oof, test_src = None, 'release_oof'
        print('nested_base отсутствует: тестовые строки берутся из релизного OOF (другая машина).')
    lt = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))
    Xlt = pd.DataFrame([lt_features({'lt_prominence_mm': lt[u]['lt_prominence_mm'], 'lt_area_mm2': lt[u]['lt_area_mm2'],
                                     'lt_edge_bump_mm': lt[u]['lt_edge_bump_mm']}) if u in lt else lt_features(None)
                        for u in df.image_uid], index=df.index)
    Xlm = F.features(df)
    res = {c: np.full(len(df), np.nan) for c in HEADS}
    for f in folds:
        te = (df.fold == f).values
        tr = ~te
        g = df.study.values[tr]
        for code, (lab, rel_col, nest_col) in HEADS.items():
            y = df[lab].values[tr].astype(int)
            cnn_tr = np.array([getattr(nested[(f, u)], nest_col) for u in df.image_uid[tr]])
            cnn_te = (base_oof.loc[df.image_uid[te], nest_col].values if base_oof is not None
                      else df[rel_col].values[te])            # trained without f
            feat = Xlt if code != 'hip_roi_margins' else Xlm
            cols = (['cnn', 'lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump'] if code != 'hip_roi_margins'
                    else ['cnn', 'roi_top', 'roi_bottom', 'roi_lateral', 'roi_medial', 'roi_min'])
            Xtr = feat.loc[tr].copy()
            Xtr['cnn'] = F.logit(np.clip(cnn_tr, 1e-6, 1 - 1e-6))
            Xte = feat.loc[te].copy()
            Xte['cnn'] = F.logit(np.clip(cnn_te, 1e-6, 1 - 1e-6))
            for i, (e_nested, e_base) in enumerate(extras):   # arms C, D: one more input where the model has this output
                if nest_col in e_base.columns:
                    Xtr[f'x{i}'] = F.logit(np.clip([getattr(e_nested[(f, u)], nest_col) for u in df.image_uid[tr]], 1e-6, 1 - 1e-6))
                    Xte[f'x{i}'] = F.logit(np.clip(e_base.loc[df.image_uid[te], nest_col].values, 1e-6, 1 - 1e-6))
                    cols = cols + [f'x{i}']
            s, thr = level2(Xtr, y, g, Xte, cols)
            res[code][te] = s / max(thr, 1e-9)
    t = pd.DataFrame({'study': df.study, 'image_uid': df.image_uid, 'region': df.region, 'fold': df.fold,
                      'y_quality': df.quality.astype(int), 'score_quality': res['quality'],
                      'quality_class': (res['quality'] >= 1).astype(int)})
    for code in ('hip_positioning', 'hip_roi_margins'):
        t['score_' + code] = res[code]
        t['reported_' + code] = (res['quality'] >= 1) & (res[code] >= 1)
        t['y_' + code] = df[EF.REASONS[code]]
    t.to_csv(OUT / f'decisions_hip_nested{out_tag}.csv', index=False)
    m = EF.with_ci(t, np.random.default_rng(0))
    (OUT / f'hip_nested_metrics{out_tag}.json').write_text(json.dumps(
        {'test_side_scores': test_src, 'suffix': args.suffix, 'root': str(args.root), 'extra': [str(e) for e in args.extra], 'isolation': 'full: no model saw the outer fold it scores',
         'metrics': m}, indent=2, default=float))
    L = ['| Срез | n / нарушений | Чувствительность | Специфичность | F1 | ROC-AUC |', '|---|---|---|---|---|---|']
    for name in ('quality_hip', 'hip_positioning', 'hip_roi_margins'):
        r = m[name]
        L.append(f"| {name} | {r['n']} / {r['pos']} | {EF.fmt(r, 'sensitivity')} | {EF.fmt(r, 'specificity')} | "
                 f"{EF.fmt(r, 'f1')} | {EF.fmt(r, 'roc_auc')} |")
    (OUT / f'hip_nested_table{out_tag}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nСравнить с: бедро 0.634 (прежняя, зависимая) и 0.580 (без CNN, изолированная).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
