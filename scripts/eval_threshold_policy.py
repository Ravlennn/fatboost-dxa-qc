"""Does a better way of picking the decision threshold raise F1? (model unchanged)

F1 is an operating-point metric and our thresholds are chosen by argmax-F1 on a handful
of positives, which is noisy. Five pre-specified strategies are compared with the usual
nested protocol; the choice between them is itself made inside each outer fold, so the
reported numbers are not the maximum over strategies.

Strategies (input: labels and scores of the training part):
  argmax_f1     current: the threshold maximising F1 on the inner out-of-fold predictions
  mean_folds    average of the per-inner-fold argmax-F1 thresholds
  bootstrap     average of argmax-F1 thresholds over 200 bootstrap resamples
  calibrated    Platt calibration, then the threshold maximising expected F1
  prevalence    the quantile that flags as many images as the training prevalence

  python scripts/eval_threshold_policy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_final as EF  # noqa: E402
import run_landmark_fusion as F  # noqa: E402

OUT = ROOT / 'outputs/threshold_policy_v1'


def t_argmax(y, s):
    return F.best_threshold(y, s)


def t_mean_folds(y, s, k=3):
    ts = []
    for tr, _ in KFold(k, shuffle=True, random_state=0).split(s):
        if y[tr].sum():
            ts.append(F.best_threshold(y[tr], s[tr]))
    return float(np.mean(ts)) if ts else F.best_threshold(y, s)


def t_bootstrap(y, s, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            ts.append(F.best_threshold(y[i], s[i]))
    return float(np.mean(ts)) if ts else F.best_threshold(y, s)


def t_calibrated(y, s):
    """Platt-calibrate, then take the threshold with the highest expected F1 under the calibration."""
    z = F.logit(np.clip(s, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return F.best_threshold(y, s)
    lr = LogisticRegression(max_iter=1000).fit(z, y)
    p = lr.predict_proba(z)[:, 1]
    order = np.argsort(-p)
    ps = p[order]
    tp = np.cumsum(ps)                     # expected true positives when flagging the top k
    k = np.arange(1, len(ps) + 1)
    exp_f1 = 2 * tp / (k + p.sum())
    best = int(np.argmax(exp_f1))
    return float(s[order][best])


def t_prevalence(y, s):
    rate = max(y.mean(), 1 / max(len(y), 1))
    return float(np.quantile(s, 1 - rate))


STRATEGIES = {'argmax_f1': t_argmax, 'mean_folds': t_mean_folds, 'bootstrap': t_bootstrap,
              'calibrated': t_calibrated, 'prevalence': t_prevalence}


def nested_choice(y, s):
    """Pick the strategy inside the training part: threshold on 2 inner folds, F1 on the 3rd."""
    scores = {}
    for name, fn in STRATEGIES.items():
        f1s = []
        for tr, te in KFold(3, shuffle=True, random_state=1).split(s):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            f1s.append(f1_score(y[te], s[te] >= fn(y[tr], s[tr]), zero_division=0))
        scores[name] = float(np.mean(f1s)) if f1s else 0.0
    best = max(scores, key=scores.get)
    nested_choice.counts[best] = nested_choice.counts.get(best, 0) + 1
    return STRATEGIES[best](y, s)


nested_choice.counts = {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res, rows = {}, []
    variants = list(STRATEGIES.items()) + [('nested_choice', nested_choice)]
    for name, fn in variants:
        EF.THRESHOLD = {'fn': fn, 'name': name}
        nested_choice.counts = {}
        df, comp = EF.build()
        t = EF.decide(df, comp, 'after_v2')
        m = EF.all_metrics(t)
        res[name] = {k: {kk: vv for kk, vv in v.items()} for k, v in m.items()}
        rows.append((name, m, t))
        print(f"{name:14s} quality F1 {m['quality_all']['f1']:.3f} (spine {m['quality_spine']['f1']:.3f}, "
              f"hip {m['quality_hip']['f1']:.3f})  macro-F1 причин {m['reasons_macro_f1']['value']:.3f}"
              + (f"  выбор: {nested_choice.counts}" if name == 'nested_choice' else ''), flush=True)
    # CI of the difference against the current strategy, paired over studies
    base = dict(rows)['argmax_f1'] if False else None
    t0 = [t for n, m, t in rows if n == 'argmax_f1'][0]
    L = ['| Стратегия порога | F1 качества | F1 позвоночник | F1 бедро | macro-F1 причин | ΔF1 качества [95%] |',
         '|---|---:|---:|---:|---:|---|']
    for name, m, t in rows:
        d = ''
        if name != 'argmax_f1':
            ci = EF_paired_f1(t0, t, rng)
            d = f'[{ci[0]:+.3f}; {ci[1]:+.3f}]'
            res[name]['delta_f1_quality_ci'] = ci
        L.append(f"| {name} | {m['quality_all']['f1']:.3f} | {m['quality_spine']['f1']:.3f} | "
                 f"{m['quality_hip']['f1']:.3f} | {m['reasons_macro_f1']['value']:.3f} | {d} |")
    (OUT / 'results.json').write_text(json.dumps(res, indent=2, default=float))
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print()
    print('\n'.join(L))


def EF_paired_f1(t_base, t_new, rng, n=2000):
    y = t_base.y_quality.values
    g = t_base.study.values
    a, b = (t_base.quality_class == 1).values, (t_new.quality_class == 1).values
    us = np.unique(g)
    idx = {s: np.where(g == s)[0] for s in us}
    out = []
    for _ in range(n):
        ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
        if y[ii].sum():
            out.append(f1_score(y[ii], b[ii], zero_division=0) - f1_score(y[ii], a[ii], zero_division=0))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


if __name__ == '__main__':
    main()
