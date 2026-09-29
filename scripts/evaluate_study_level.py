"""Метрики на уровне исследования (ТЗ 8.4).

ТЗ требует «чувствительность выявления **исследований** с нарушениями качества» и
«специфичность для качественно выполненных **исследований**». Единица измерения —
исследование, а не кадр; вся прежняя валидация считалась по кадрам.

Агрегация:
  * эталон: исследование бракованное, если бракован хотя бы один его размеченный кадр;
  * решение: исследование помечено, если помечен хотя бы один кадр;
  * непрерывный скор: максимум по кадрам от score/threshold, где threshold —
    F1-оптимальный порог по кадрам, подобранный на ДРУГИХ фолдах (вложенно, по области).
    Такая нормировка даёт score >= 1 ровно тогда, когда кадр помечен текущим правилом.

Дополнительно проверяется гипотеза: правило, настроенное под F1 по кадрам, не оптимально
под F1 по исследованиям, потому что ложные срабатывания перемножаются по 1–3 кадрам.
Множитель m к нормированному скору подбирается вложенно (для фолда k — по исследованиям
остальных фолдов) и сравнивается с текущим m = 1.

Неразмеченные кадры (эндопротезы) исключаются из эталона и из решения.
95% ДИ: bootstrap по исследованиям.

  python scripts/evaluate_study_level.py --pred outputs/mvp/oof_mvp.csv
  python scripts/evaluate_study_level.py --pred outputs/final_validation_v1/decisions_after_v2.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CODES = ('spine_scan_range', 'spine_axis_tilt', 'spine_artifact', 'hip_positioning', 'hip_roi_margins')
GRID = np.round(np.arange(0.40, 3.01, 0.02), 2)


def best_threshold(y, s):
    """Порог, максимизирующий F1; при равенстве берётся больший (консервативнее)."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float(np.nanmax(s)) + 1e-9
    cand = np.unique(np.concatenate([s, [s.min() - 1e-9]]))
    best, arg = -1.0, cand[0]
    for t in cand:
        f = f1_score(y, s >= t, zero_division=0)
        if f > best + 1e-12 or (abs(f - best) <= 1e-12 and t > arg):
            best, arg = f, t
    return float(arg)


def load(path: Path) -> pd.DataFrame:
    """Приводит формат релиза (decisions_*.csv) и формат MVP (oof_mvp.csv) к общим колонкам."""
    d = pd.read_csv(path)
    out = pd.DataFrame({'study': d.study, 'region': np.where(d.region.astype(str).str.startswith('hip'), 'hip', 'spine')})
    out['image_uid'] = d.get('image_uid', pd.Series(range(len(d))))
    if 'fold' in d:
        out['fold'] = d.fold
    else:  # фолды из канонической таблицы
        f = pd.read_csv(ROOT / 'data/interim/folds.csv').set_index('image_uid').fold
        out['fold'] = out.image_uid.map(f)
    out['y_quality'] = d['y_quality'] if 'y_quality' in d else d['quality']
    out['pred_quality'] = d['quality_class'] if 'quality_class' in d else d['yhat']
    out['score_quality'] = d['score_quality'] if 'score_quality' in d else d['p_quality']
    for c in CODES:
        if f'y_{c}' in d:
            out[f'y_{c}'] = d[f'y_{c}']
            out[f'pred_{c}'] = d[f'reported_{c}'] if f'reported_{c}' in d else d.get(f'f_{c}')
            out[f'score_{c}'] = d[f'score_{c}'] if f'score_{c}' in d else d.get(f'p_{c}')
    return out[out.y_quality.notna() & out.fold.notna()].reset_index(drop=True)


def normalize(df: pd.DataFrame) -> np.ndarray:
    """score/thr; thr — F1-оптимальный порог по кадрам своей области на других фолдах."""
    out = np.full(len(df), np.nan)
    for g in ('spine', 'hip'):
        for f in sorted(df.fold.unique()):
            tr = (df.region == g) & (df.fold != f)
            te = ((df.region == g) & (df.fold == f)).values
            if not te.any() or tr.sum() < 5:
                continue
            thr = best_threshold(df.y_quality[tr], df.score_quality[tr])
            out[te] = df.score_quality.values[te] / max(abs(thr), 1e-9)
    return out


def to_study(df: pd.DataFrame, flag: np.ndarray) -> pd.DataFrame:
    """Кадры → исследования: эталон и решение по правилу «хотя бы один», скор — максимум."""
    d = df.assign(_flag=flag.astype(bool))
    rows = {
        'y': d.groupby('study').y_quality.max().astype(int),
        'pred': d.groupby('study')._flag.max().astype(int),
        'score': d.groupby('study').score_norm.max(),
        'fold': d.groupby('study').fold.first().astype(int),
        'n_images': d.groupby('study').size(),
    }
    hip, spine = d[d.region == 'hip'], d[d.region == 'spine']
    for name, part in (('hip', hip), ('spine', spine)):
        rows[f'y_{name}'] = part.groupby('study').y_quality.max()
        rows[f'pred_{name}'] = part.groupby('study')._flag.max()
        rows[f'score_{name}'] = part.groupby('study').score_norm.max()
    for c in CODES:
        if f'y_{c}' in d:
            sub = d[d[f'y_{c}'].notna()]
            rows[f'y_{c}'] = sub.groupby('study')[f'y_{c}'].max()
            rows[f'pred_{c}'] = sub.groupby('study')[f'pred_{c}'].max()
            rows[f'score_{c}'] = sub.groupby('study')[f'score_{c}'].max()
    return pd.DataFrame(rows).reset_index()


def binary(y, pred, score=None) -> dict:
    y = np.asarray(y, float)
    keep = ~np.isnan(y)
    y, pred = y[keep].astype(int), np.asarray(pred, float)[keep].astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    m = {'n': len(y), 'pos': int(y.sum()), 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
         'sensitivity': sens, 'specificity': spec, 'balanced_acc': (sens + spec) / 2,
         'f1': f1_score(y, pred, zero_division=0)}
    if score is not None and 0 < y.sum() < len(y):
        s = np.asarray(score, float)[keep]
        if np.isfinite(s).all():
            m['roc_auc'] = roc_auc_score(y, s)
            m['pr_auc'] = average_precision_score(y, s)
    return m


def all_metrics(t: pd.DataFrame) -> dict:
    res = {'study_all': binary(t.y, t.pred, t.score)}
    for name in ('spine', 'hip'):
        res[f'study_{name}'] = binary(t[f'y_{name}'], t[f'pred_{name}'].fillna(0), t[f'score_{name}'])
    f1s = []
    for c in CODES:
        if f'y_{c}' in t:
            res[c] = binary(t[f'y_{c}'], t[f'pred_{c}'].fillna(0), t[f'score_{c}'])
            f1s.append(res[c]['f1'])
    if f1s:
        res['reasons_macro_f1'] = {'n': len(f1s), 'pos': 0, 'f1': float(np.mean(f1s))}
    return res


def with_ci(t: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    point = all_metrics(t)
    rng = np.random.default_rng(seed)
    reps = [all_metrics(t.iloc[rng.integers(len(t), size=len(t))]) for _ in range(n_boot)]
    for k, m in point.items():
        for key in list(m):
            if key in ('n', 'pos', 'tp', 'fn', 'tn', 'fp'):
                continue
            vals = [r[k][key] for r in reps if k in r and key in r[k]]
            lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
            m[key] = {'value': float(m[key]), 'ci95': [float(lo), float(hi)]}
    return point


def nested_multiplier(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Для фолда k множитель m выбирается по F1 на уровне ИССЛЕДОВАНИЙ остальных фолдов."""
    flag = np.zeros(len(df), bool)
    chosen = {}
    for f in sorted(df.fold.unique()):
        tr, te = df.fold != f, (df.fold == f).values
        best, arg = -1.0, 1.0
        for m in GRID:
            s = to_study(df[tr], (df.score_norm[tr] >= m).values)
            v = f1_score(s.y, s.pred, zero_division=0)
            if v > best + 1e-12:
                best, arg = v, float(m)
        chosen[int(f)] = arg
        flag[te] = df.score_norm.values[te] >= arg
    return flag, chosen


def baselines(t: pd.DataFrame, seed: int = 0) -> dict:
    """Тривиальные точки отсчёта. На уровне исследований выборка сбалансирована,
    поэтому «всё бракованное» даёт высокий F1 и без него метрику читать нельзя."""
    y = t.y.values.astype(int)
    rng = np.random.default_rng(seed)
    return {'всё бракованное': binary(y, np.ones(len(y))),
            'ничего не помечено': binary(y, np.zeros(len(y))),
            'случайно 50/50': binary(y, rng.integers(0, 2, len(y)), rng.random(len(y)))}


def size_bias(t: pd.DataFrame) -> list[str]:
    """Правило «хотя бы один» даёт исследованию с 3 кадрами больше шансов быть помеченным."""
    L = ['| Кадров в исследовании | n | доля с нарушением | доля помеченных | ложных срабатываний |', '|---|---|---|---|---|']
    for k, g in t.groupby('n_images'):
        neg = g[g.y == 0]
        fp = float(neg.pred.mean()) if len(neg) else float('nan')
        L.append(f'| {int(k)} | {len(g)} | {g.y.mean():.2f} | {g.pred.mean():.2f} | {fp:.2f} |')
    return L


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int = 1) -> dict:
    """95% ДИ разницы b−a на одних и тех же исследованиях (парный bootstrap)."""
    rng = np.random.default_rng(seed)
    keys = ['study_all', 'study_spine', 'study_hip']
    point = {k: all_metrics(b)[k]['f1'] - all_metrics(a)[k]['f1'] for k in keys}
    reps = {k: [] for k in keys}
    for _ in range(n_boot):
        ii = rng.integers(len(a), size=len(a))
        ma, mb = all_metrics(a.iloc[ii]), all_metrics(b.iloc[ii])
        for k in keys:
            reps[k].append(mb[k]['f1'] - ma[k]['f1'])
    return {k: {'delta': float(point[k]), 'ci95': [float(np.percentile(reps[k], 2.5)), float(np.percentile(reps[k], 97.5))]}
            for k in keys}


def fmt(m, key) -> str:
    if key not in m:
        return '—'
    v = m[key]
    return f"{v['value']:.3f} [{v['ci95'][0]:.2f}–{v['ci95'][1]:.2f}]" if isinstance(v, dict) else f'{v:.3f}'


def main() -> None:
    ap = argparse.ArgumentParser(description='Метрики по исследованиям (ТЗ 8.4)')
    ap.add_argument('--pred', default='outputs/mvp/oof_mvp.csv', help='CSV с решениями по кадрам')
    ap.add_argument('--out', default='outputs/study_level_v1')
    ap.add_argument('--bootstrap', type=int, default=2000)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    df = load(ROOT / a.pred)
    df['score_norm'] = normalize(df)
    df = df[df.score_norm.notna()].reset_index(drop=True)

    image = binary(df.y_quality, df.pred_quality, df.score_norm)
    cur = to_study(df, df.pred_quality.values.astype(bool))
    res_cur = with_ci(cur, a.bootstrap)
    tuned_flag, mult = nested_multiplier(df)
    tuned = to_study(df, tuned_flag)
    res_tuned = with_ci(tuned, a.bootstrap)
    cur.to_csv(out / 'studies_current.csv', index=False)
    tuned.to_csv(out / 'studies_retuned.csv', index=False)

    rows = [('Все области', 'study_all'), ('Позвоночник', 'study_spine'), ('Бедро', 'study_hip')] + \
           [(f'Причина: {c}', c) for c in CODES if c in res_cur]
    L = [f'# Метрики на уровне исследования — `{a.pred}`', '',
         f'Кадров: {len(df)}, из них нарушений {int(df.y_quality.sum())}. '
         f'Исследований: {len(cur)}, из них с нарушением {int(cur.y.sum())}.', '',
         'Для сравнения, та же сборка **по кадрам**: '
         f"F1 {image['f1']:.3f}, чувствительность {image['sensitivity']:.3f}, специфичность {image['specificity']:.3f}"
         + (f", ROC-AUC {image['roc_auc']:.3f}." if 'roc_auc' in image else '.'), '',
         '## Текущее правило, агрегированное по исследованиям', '',
         '| Срез | n / с нарушением | Чувствительность | Специфичность | F1 | ROC-AUC | PR-AUC |', '|---|---|---|---|---|---|---|']
    for title, k in rows:
        m = res_cur[k]
        L.append(f"| {title} | {m['n']} / {m['pos']} | {fmt(m, 'sensitivity')} | {fmt(m, 'specificity')} | "
                 f"{fmt(m, 'f1')} | {fmt(m, 'roc_auc')} | {fmt(m, 'pr_auc')} |")
    if 'reasons_macro_f1' in res_cur:
        L += ['', f"Macro-F1 пяти причин по исследованиям: **{fmt(res_cur['reasons_macro_f1'], 'f1')}**."]

    base = baselines(cur)
    L += ['', '## Точки отсчёта на уровне исследования', '',
          'Исследований с нарушением ровно половина, поэтому тривиальные правила здесь сильные. '
          'Сравнивать F1 по исследованиям с F1 по кадрам напрямую нельзя: у них разная доля положительного класса.', '',
          '| Правило | Чувствительность | Специфичность | Сбалансированная точность | F1 |', '|---|---|---|---|---|']
    for name, m in base.items():
        L.append(f"| {name} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['balanced_acc']:.3f} | {m['f1']:.3f} |")
    m = res_cur['study_all']
    L.append(f"| **наше решение** | **{m['sensitivity']['value']:.3f}** | **{m['specificity']['value']:.3f}** | "
             f"**{m['balanced_acc']['value']:.3f}** | **{m['f1']['value']:.3f}** |")
    L += ['', f"Прирост F1 над «всё бракованное»: **{m['f1']['value'] - base['всё бракованное']['f1']:+.3f}**. "
          f"Сбалансированная точность и ROC-AUC от доли класса не зависят и остаются честным ориентиром: "
          f"{fmt(m, 'balanced_acc')} и {fmt(m, 'roc_auc')}.", '',
          '## Смещение от правила «хотя бы один»', ''] + size_bias(cur)

    delta = paired_delta(cur, tuned, min(a.bootstrap, 1000))
    L += ['', '## Порог, пересобранный под F1 по исследованиям', '',
          f'Множитель к нормированному скору, выбранный вложенно по фолдам: {mult}. '
          'Значение меньше 1 означает более смелое решение по кадру: на уровне исследований '
          'доля нарушений вдвое выше, чем на уровне кадров, и оптимум F1 смещается в сторону чувствительности.', '',
          '| Срез | F1 текущее → пересобранное | ΔF1 [95% ДИ, парный] | Чувствительность | Специфичность |',
          '|---|---|---|---|---|']
    for title, k in rows:
        c, t = res_cur[k], res_tuned[k]
        arrow = '↑' if t['f1']['value'] > c['f1']['value'] + 5e-4 else ('↓' if t['f1']['value'] < c['f1']['value'] - 5e-4 else '=')
        dd = delta.get(k)
        ds = f"{dd['delta']:+.3f} [{dd['ci95'][0]:+.2f}; {dd['ci95'][1]:+.2f}]" if dd else '—'
        L.append(f"| {title} | {c['f1']['value']:.3f} → **{t['f1']['value']:.3f}** {arrow} | {ds} | "
                 f"{c['sensitivity']['value']:.3f} → {t['sensitivity']['value']:.3f} | "
                 f"{c['specificity']['value']:.3f} → {t['specificity']['value']:.3f} |")
    L += ['', 'Интервал разницы, включающий ноль, означает, что пересборка порога не доказана как улучшение '
          'на этих данных. Важен сам факт: оптимум операционной точки зависит от того, в каких единицах '
          'организатор считает метрику.']

    (out / 'metrics.json').write_text(json.dumps(
        {'protocol': __doc__, 'pred': a.pred, 'image_level': image, 'image_level_baseline_all_positive':
         binary(df.y_quality, np.ones(len(df))), 'study_current': res_cur, 'study_baselines': baselines(cur),
         'study_retuned': res_tuned, 'multiplier_per_fold': mult, 'paired_delta_f1': delta}, indent=2, ensure_ascii=False, default=float))
    (out / 'table.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
