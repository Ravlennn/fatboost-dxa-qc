"""Final table with both branches isolated: spine from evaluate_final_isolated, hip from the
25 nested CNN runs (eval_hip_nested). No model in either branch saw the fold it is scored on.

  python scripts/combine_isolated.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_final as EF  # noqa: E402

OUT = ROOT / 'outputs/final_validation_isolated'


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--hip-tag', default='', help='suffix of decisions_hip_nested<tag>.csv')
    args = ap.parse_args(argv)
    sp = pd.read_csv(OUT / 'decisions_isolated.csv')
    sp = sp[sp.region.eq('spine')]
    hp = pd.read_csv(OUT / f'decisions_hip_nested{args.hip_tag}.csv')
    t = pd.concat([sp, hp], ignore_index=True)
    for c in EF.REASONS:
        for pre, fill in (('score_', np.nan), ('reported_', False), ('y_', np.nan)):
            t[pre + c] = t[pre + c].fillna(fill) if pre + c in t else fill
        t['reported_' + c] = t['reported_' + c].astype(bool)
    t['quality_unspecified'] = (t.quality_class == 1) & ~t[[f'reported_{c}' for c in EF.REASONS]].any(axis=1)
    t.to_csv(OUT / f'decisions_combined{args.hip_tag}.csv', index=False)
    m = EF.with_ci(t, np.random.default_rng(0))
    (OUT / f'combined_metrics{args.hip_tag}.json').write_text(json.dumps(
        {'isolation': 'spine: frozen encoder refit per fold; hip: 25 nested CNN runs', 'metrics': m},
        indent=2, default=float))
    L = ['| Срез | n / нарушений | Чувствительность | Специфичность | F1 | ROC-AUC |', '|---|---|---|---|---|---|']
    for name in ('quality_all', 'quality_spine', 'quality_hip', *EF.REASONS):
        if name not in m:
            continue
        r = m[name]
        L.append(f"| {name} | {r['n']} / {r['pos']} | {EF.fmt(r, 'sensitivity')} | {EF.fmt(r, 'specificity')} | "
                 f"{EF.fmt(r, 'f1')} | {EF.fmt(r, 'roc_auc')} |")
    L += ['', f"Macro-F1 причин: {EF.fmt(m['reasons_macro_f1'], 'value')}",
          f"Брак без указанной причины: {m['quality_unspecified_among_defects']['value']['value']:.1%}"]
    (OUT / f'combined_table{args.hip_tag}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
