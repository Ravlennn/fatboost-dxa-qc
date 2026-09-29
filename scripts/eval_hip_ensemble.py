"""Compare hip CNN runs / seed ensembles against the release hip ensemble on the same folds.

Each run dir has oof.csv from scripts/train_hip_cnn.py. Runs are grouped by the
text before '_s<seed>' in the tag; a group's OOF is the mean over its seeds.
Reports AUC/AP, inner-fold-threshold F1 and paired study-bootstrap ΔAUC/ΔAP vs baseline.

  python scripts/eval_hip_ensemble.py outputs/hip_cnn/h100_*  --baseline outputs/release_ensemble_v1/selected_oof.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402

TARGETS = ['quality', 'v_rotation', 'v_roi']
KEY = ['study', 'image_uid']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--baseline', type=Path, default=ROOT / 'outputs/release_ensemble_v1/selected_oof.csv')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/hip_ensemble_eval')
    args = ap.parse_args()
    base = pd.read_csv(args.baseline).set_index(KEY)
    groups = {}
    for run in args.runs:
        f = run / 'oof.csv'
        if not f.is_file():
            print('skip (no oof.csv):', run)
            continue
        name = re.sub(r'_s\d+$', '', run.name)
        groups.setdefault(name, []).append(pd.read_csv(f).set_index(KEY))
    args.out.mkdir(parents=True, exist_ok=True)
    res, lines = {}, ['| model | seeds | target | AUC | AP | F1 (inner thr) | ΔAUC [95%] | ΔAP [95%] |', '|---|---:|---|---:|---:|---:|---|---|']
    variants = {'release': (base, 1)}
    for name, dfs in groups.items():
        idx = dfs[0].index
        for d in dfs[1:]:
            assert set(d.index) == set(idx), f'{name}: runs cover different images'
        m = dfs[0].copy()
        for t in TARGETS:
            m['p_' + t] = np.mean([d.loc[idx, 'p_' + t].values for d in dfs], axis=0)
        variants[name] = (m, len(dfs))
    for name, (d, n_seeds) in variants.items():
        d = d.loc[base.index]
        res[name] = {}
        for t in TARGETS:
            ok = base[t].notna().values
            y, s, s0 = base[t].values[ok].astype(int), d['p_' + t].values[ok], base['p_' + t].values[ok]
            g, fo = np.array([k[0] for k in base.index[ok]]), base['fold'].values[ok]
            dec = F.cnn_decisions(s, y, g, fo)
            r = {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s), 'f1': f1_score(y, dec)}
            if name != 'release':
                r['d_auc'], r['d_ap'] = F.paired_boot(y, s0, s, g)
            res[name][t] = r
            lines.append(f"| {name} | {n_seeds} | {t} | {r['auc']:.3f} | {r['ap']:.3f} | {r['f1']:.3f} | "
                         + (f"[{r['d_auc'][0]:+.3f}; {r['d_auc'][1]:+.3f}] | [{r['d_ap'][0]:+.3f}; {r['d_ap'][1]:+.3f}]" if 'd_auc' in r else ' | ')
                         + ' |')
            if name != 'release':
                d.reset_index()[KEY + ['fold'] + ['p_' + x for x in TARGETS]].to_csv(args.out / f'{name}_oof.csv', index=False)
    (args.out / 'results.json').write_text(json.dumps(res, indent=2, default=float))
    (args.out / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
