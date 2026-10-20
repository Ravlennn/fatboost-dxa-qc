"""Spine branch with a fine-tuned, fully isolated CNN feature (docs/PREREG_NEW_MODELS.md, arms A and B).

The spine branch of scripts/evaluate_final_isolated.py, with one extra input in each cause head:
'scnn' = logit of the nested CNN's probability for that cause (scan range ← p_v_layout, axis tilt ←
p_v_axis, foreign objects ← p_v_artifact). For outer fold f the heads are fitted on outer-train rows
scored by <root>/nested_f{f}_j{j} (trained without f and j); the outer-test rows are scored by
<root>/nested_base (trained without f). The spine policy is unchanged: defect = any cause found.

Without --root the CNN feature is left out and the control (decisions_isolated.csv, spine rows) must be
reproduced exactly — that is checked by --check-control.

  python scripts/eval_spine_nested.py --root outputs/spine_cnn --tag _spinecnn
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
from evaluate_final_isolated import frozen_head, inner_oof, level2  # noqa: E402

OUT = ROOT / 'outputs/final_validation_isolated'
CAUSES = {  # code: (label column, CNN column, base feature columns)
    'spine_artifact': ('v_artifact', 'p_v_artifact', ['cnn', 'art_wire', 'art_metal']),
    'spine_axis_tilt': ('v_axis', 'p_v_axis', list(F.TASKS['spine_axis_tilt'][3])),
    'spine_scan_range': ('v_layout', 'p_v_layout', list(F.TASKS['spine_scan_range_iliac'][3])),
}


def nested_scores(root: Path, folds):
    out, missing = {}, []
    for f in folds:
        for j in folds:
            if f == j:
                continue
            p = root / f'nested_f{f}_j{j}' / 'oof.csv'
            if not p.exists():
                missing.append(p.parent.name)
                continue
            d = pd.read_csv(p)
            for r in d[d.fold == j].itertuples():
                out[(f, r.image_uid)] = r
    return out, missing


def evaluate(root: Path | None) -> pd.DataFrame:
    feats = np.load(ROOT / 'outputs/release_loss_v1/features.npy')
    meta = json.load(open(ROOT / 'outputs/release_loss_v1/features_meta.json'))
    fidx = {r['image_uid']: i for i, r in enumerate(meta['identity']['rows'])}
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    v1 = pd.DataFrame(json.load(open(ROOT / 'outputs/landmarks_v1/ours/predictions.json')))
    df = oof.merge(v1[['image_uid', 'study', 'measurements']], on=['study', 'image_uid'], validate='1:1')
    df = df[df.quality.notna() & df.region.eq('spine')].reset_index(drop=True)
    art = json.load(open(ROOT / 'outputs/artifacts_lm_v1/measurements.json'))
    Xall = feats[[fidx[u] for u in df.image_uid]]
    Xlm = F.features(df)
    Xart = pd.DataFrame([art.get(u, {'art_wire': 0.0, 'art_metal': 0.0}) for u in df.image_uid], index=df.index)
    folds = sorted(df.fold.unique())
    nested = base = None
    if root is not None:
        nested, missing = nested_scores(root, folds)
        if missing:
            raise SystemExit(f'Нет {len(missing)} вложенных прогонов в {root}: {missing[:4]}')
        base = pd.read_csv(root / 'nested_base' / 'oof.csv').set_index('image_uid')

    ratio = {c: np.full(len(df), np.nan) for c in CAUSES}
    for f in folds:
        te = (df.fold == f).values
        tr = ~te
        g = df.study.values[tr]
        for code, (lab, cnn_col, cols) in CAUSES.items():
            y = df[lab].values[tr].astype(int)
            if code == 'spine_artifact':   # frozen DenseNet head, inner OOF on the train part (as in the control)
                Xtr, Xte = Xart.loc[tr].copy(), Xart.loc[te].copy()
                Xtr['cnn'] = F.logit(inner_oof(Xall[tr], y, g, lambda a, b: frozen_head().fit(a, b)))
                Xte['cnn'] = F.logit(frozen_head().fit(Xall[tr], y).predict_proba(Xall[te])[:, 1])
            else:
                Xtr, Xte = Xlm.loc[tr].copy(), Xlm.loc[te].copy()
            use = list(cols)
            if root is not None:
                Xtr['scnn'] = F.logit(np.clip([getattr(nested[(f, u)], cnn_col) for u in df.image_uid[tr]], 1e-6, 1 - 1e-6))
                Xte['scnn'] = F.logit(np.clip(base.loc[df.image_uid[te], cnn_col].values, 1e-6, 1 - 1e-6))
                use.append('scnn')
            s, thr = level2(Xtr, y, g, Xte, use)
            ratio[code][te] = s / max(thr, 1e-9)

    q = np.fmax.reduce([ratio[c] for c in CAUSES])
    t = pd.DataFrame({'study': df.study, 'image_uid': df.image_uid, 'region': df.region, 'fold': df.fold,
                      'y_quality': df.quality.astype(int), 'score_quality': q, 'quality_class': (q >= 1).astype(int)})
    for code, (lab, _, _) in CAUSES.items():
        t['score_' + code] = ratio[code]
        t['reported_' + code] = (q >= 1) & (ratio[code] >= 1)
        t['y_' + code] = df[lab]
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, help='folder with nested_base and nested_f*_j* runs; omit for the control')
    ap.add_argument('--tag', default='')
    ap.add_argument('--check-control', action='store_true', help='without --root: compare with decisions_isolated.csv')
    args = ap.parse_args(argv)
    t = evaluate(args.root)
    if args.check_control:
        ctrl = pd.read_csv(OUT / 'decisions_isolated.csv')
        ctrl = ctrl[ctrl.region.eq('spine')].set_index('image_uid').loc[t.image_uid]
        same = all(np.allclose(ctrl['score_' + c].values, t['score_' + c].values, equal_nan=True) for c in CAUSES)
        print('контроль воспроизведён в точности:', same and (ctrl.quality_class.values == t.quality_class.values).all())
        return 0
    t.to_csv(OUT / f'decisions_spine_nested{args.tag}.csv', index=False)
    print('->', OUT / f'decisions_spine_nested{args.tag}.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
