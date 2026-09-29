"""Three follow-up checks on the H100 artefacts, with our nested protocol.

The H100 series were all rejected on their own. What was not tested there:
  1. artifact segmenter FUSED with the deterministic detector and the CNN;
  2. frozen DINO embeddings as an extra feature for hip rotation / quality
     (the probe was better than ImageNet, the fine-tune was worse);
  3. averaging different architectures together with the release ensemble.
Everything is evaluated out-of-fold on the release study folds, thresholds on
inner folds, 95% CI by paired study bootstrap (AUC, AP and F1).

  python scripts/eval_h100_followups.py --h100 <dir with outputs/ from the server>
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
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402
from dxaqc.lesser_trochanter import head_features as lt_features  # noqa: E402

OUT = ROOT / 'outputs/h100_followups'
HIP_BASE = ['cnn', 'lt_prom', 'lt_prom_sq', 'lt_area', 'lt_edge_bump']


def paired_f1(y, dec_a, dec_b, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    us = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in us}
    out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(us, len(us))])
        if y[ii].sum():
            out.append(f1_score(y[ii], dec_b[ii], zero_division=0) - f1_score(y[ii], dec_a[ii], zero_division=0))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def row(name, y, s, dec, base=None, g=None):
    r = {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s), 'f1': f1_score(y, dec)}
    if base is not None:
        r['d_auc'], r['d_ap'] = F.paired_boot(y, base[0], s, g)
        r['d_f1'] = paired_f1(y, base[1], dec, g)
    return r


def fmt(name, r):
    ci = (f"[{r['d_auc'][0]:+.3f}; {r['d_auc'][1]:+.3f}] | [{r['d_f1'][0]:+.3f}; {r['d_f1'][1]:+.3f}]"
          if 'd_auc' in r else ' | ')
    return f"| {name} | {r['auc']:.3f} | {r['ap']:.3f} | {r['f1']:.3f} | {ci} |"


def artifacts(h100, oof, res, L):
    d = oof[oof.region.eq('spine') & oof.v_artifact.notna()].copy()
    det = json.load(open(ROOT / 'outputs/artifacts_lm_v1/measurements.json'))
    X = pd.DataFrame([det[u] for u in d.image_uid], index=d.index)
    X['cnn'] = F.logit(d.p_spine_artifact.values)
    seg_cols = []
    for tag in ('artifact_seg_h100', 'artifact_seg_h100_bgours'):
        f = h100 / 'outputs' / tag / 'ours_scores.csv'
        if not f.is_file():
            continue
        s = pd.read_csv(f).set_index('image_uid')
        key = 'bgours' if 'bgours' in tag else 'ext'
        X[f'seg_topk_{key}'] = [F.logit(np.clip(s.topk.get(u, 0.0), 1e-4, 1 - 1e-4)) for u in d.image_uid]
        X[f'seg_area_{key}'] = [np.log1p(s.area.get(u, 0.0)) for u in d.image_uid]
        seg_cols += [f'seg_topk_{key}', f'seg_area_{key}']
    y, g, fo = d.v_artifact.astype(int).values, d.study.values, d.fold.values
    cur = F.fit_predict(X, y, g, fo, ['cnn', 'art_wire', 'art_metal'])
    res['spine_artifact'] = {'current (CNN + detector)': row('cur', y, *cur)}
    L += ['', '### 1. Артефакты: сегментатор с H100 в объединении', '',
          '| Вариант | AUC | AP | F1 | ΔAUC vs текущий [95%] | ΔF1 [95%] |', '|---|---:|---:|---:|---|---|',
          fmt('текущий (CNN + детектор)', res['spine_artifact']['current (CNN + detector)'])]
    for name, cols in (('+ сегментатор (внешние фоны)', ['seg_topk_ext', 'seg_area_ext']),
                       ('+ сегментатор (наши фоны)', ['seg_topk_bgours', 'seg_area_bgours']),
                       ('+ оба сегментатора', seg_cols)):
        use = [c for c in cols if c in X]
        if not use:
            continue
        s, dec = F.fit_predict(X, y, g, fo, ['cnn', 'art_wire', 'art_metal'] + use)
        r = row(name, y, s, dec, cur, g)
        res['spine_artifact'][name] = r
        L.append(fmt(name, r))
    # segmenter alone, for reference
    if seg_cols:
        s, dec = F.fit_predict(X, y, g, fo, [c for c in seg_cols if c.startswith('seg_topk')])
        res['spine_artifact']['segmenter only'] = row('seg', y, s, dec, cur, g)
        L.append(fmt('только сегментатор', res['spine_artifact']['segmenter only']))


def dino(h100, oof, res, L):
    emb_f = h100 / 'outputs/ssl_v1/ours_embeddings.npy'
    idx_f = h100 / 'outputs/ssl_v1/ours_embeddings_index.csv'
    if not emb_f.is_file():
        return
    E = np.load(emb_f)
    index = pd.read_csv(idx_f)
    emb = {u: E[i] for i, u in enumerate(index.image_uid)}
    lt = json.load(open(ROOT / 'outputs/lt_auto_v1/measurements.json'))
    d = oof[oof.region.str.startswith('hip')].copy()
    Xb = pd.DataFrame([lt_features({'lt_prominence_mm': lt[u]['lt_prominence_mm'], 'lt_area_mm2': lt[u]['lt_area_mm2'],
                                    'lt_edge_bump_mm': lt[u]['lt_edge_bump_mm']}) for u in d.image_uid], index=d.index)
    Xe = pd.DataFrame(np.stack([emb[u] for u in d.image_uid]), index=d.index,
                      columns=[f'e{i}' for i in range(E.shape[1])])
    L += ['', f'### 2. Замороженные признаки DINO ({E.shape[1]} измерений) в головах бедра', '',
          '| Задача | Вариант | AUC | AP | F1 | ΔAUC vs текущий [95%] | ΔF1 [95%] |', '|---|---|---:|---:|---:|---|---|']
    for task, lab, pcol in (('hip_positioning', 'v_rotation', 'p_hip_positioning'),
                            ('hip_quality', 'quality', 'p_quality')):
        dd = d[d[lab].notna()]
        y, g, fo = dd[lab].astype(int).values, dd.study.values, dd.fold.values
        X = Xb.loc[dd.index].copy()
        X['cnn'] = F.logit(dd[pcol].values)
        # DINO score is produced out-of-fold first, then used as one feature (as the CNN OOF is).
        dino_s, _ = F.fit_predict(Xe.loc[dd.index], y, g, fo, list(Xe.columns))
        X['dino'] = F.logit(dino_s)
        cur = F.fit_predict(X, y, g, fo, HIP_BASE)
        res.setdefault(task, {})['current fusion'] = row('cur', y, *cur)
        L.append(f"| {task} | текущий fusion | {res[task]['current fusion']['auc']:.3f} | "
                 f"{res[task]['current fusion']['ap']:.3f} | {res[task]['current fusion']['f1']:.3f} |  |  |")
        for name, cols in (('+ DINO', HIP_BASE + ['dino']), ('только DINO', ['dino'])):
            s, dec = F.fit_predict(X, y, g, fo, cols)
            r = row(name, y, s, dec, cur, g)
            res[task][name] = r
            L.append(f"| {task} | {name} | {r['auc']:.3f} | {r['ap']:.3f} | {r['f1']:.3f} | "
                     f"[{r['d_auc'][0]:+.3f}; {r['d_auc'][1]:+.3f}] | [{r['d_f1'][0]:+.3f}; {r['d_f1'][1]:+.3f}] |")


def ensembles(h100, oof, res, L):
    base = oof[oof.region.str.startswith('hip')].set_index('image_uid')
    runs = {}
    for f in sorted((h100 / 'outputs/hip_cnn').glob('h100_*/oof.csv')):
        runs.setdefault(f.parent.name.rsplit('_s', 1)[0], []).append(pd.read_csv(f).set_index('image_uid'))
    groups = {k: {t: np.mean([df.loc[base.index, 'p_' + t].values for df in v], axis=0)
                  for t in ('quality', 'v_rotation', 'v_roi')} for k, v in runs.items()}
    L += ['', '### 3. Смеси архитектур с релизом', '',
          '| Цель | Смесь | AUC | AP | F1 | ΔAUC vs release [95%] | ΔF1 [95%] |', '|---|---|---:|---:|---:|---|---|']
    cols = {'quality': 'p_quality', 'v_rotation': 'p_hip_positioning', 'v_roi': 'p_hip_roi_margins'}
    for t, pcol in cols.items():
        keep = base[t].notna().values
        y = base.loc[keep, t].astype(int).values
        g, fo = base.loc[keep, 'study'].values, base.loc[keep, 'fold'].values
        rel = base.loc[keep, pcol].values
        cur = (rel, F.cnn_decisions(rel, y, g, fo))
        res.setdefault('ensemble_' + t, {})['release'] = row('rel', y, *cur)
        L.append(f"| {t} | release | {res['ensemble_' + t]['release']['auc']:.3f} | "
                 f"{res['ensemble_' + t]['release']['ap']:.3f} | {res['ensemble_' + t]['release']['f1']:.3f} |  |  |")
        for name, mix in (('release + S', ['h100_cnxs512']), ('release + B', ['h100_cnxb512']),
                          ('release + S + B', ['h100_cnxs512', 'h100_cnxb512']),
                          ('release + S + B + SSL', ['h100_cnxs512', 'h100_cnxb512', 'h100_cnxs512ssl'])):
            have = [m for m in mix if m in groups]
            if len(have) != len(mix):
                continue
            s = np.mean([rel] + [groups[m][t][keep] for m in have], axis=0)
            dec = F.cnn_decisions(s, y, g, fo)
            r = row(name, y, s, dec, cur, g)
            res['ensemble_' + t][name] = r
            L.append(f"| {t} | {name} | {r['auc']:.3f} | {r['ap']:.3f} | {r['f1']:.3f} | "
                     f"[{r['d_auc'][0]:+.3f}; {r['d_auc'][1]:+.3f}] | [{r['d_f1'][0]:+.3f}; {r['d_f1'][1]:+.3f}] |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h100', type=Path, required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    res, L = {}, ['# Проверки поверх результатов H100 (вложенный протокол)']
    artifacts(args.h100, oof, res, L)
    dino(args.h100, oof, res, L)
    ensembles(args.h100, oof, res, L)
    (OUT / 'results.json').write_text(json.dumps(res, indent=2, default=float))
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
