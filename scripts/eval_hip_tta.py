"""Test-time augmentation for the hip CNN: recompute the OOF with small shifts/zooms and compare.

For every fold, the released fold checkpoint predicts its held-out images under N
pre-specified geometric variants (identity, +-3 px shift, 0.97/1.03 zoom); scores are averaged.
Compared with the current single-view OOF by the usual protocol (inner-fold threshold,
paired study bootstrap). Inference cost grows ~N times; the TZ budget is 180 s per study.

  python scripts/eval_hip_tta.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_landmark_fusion as F  # noqa: E402
from dxaqc.bundle import bundle_path, verify_bundle  # noqa: E402
from dxaqc.dicom_io import read_dicom  # noqa: E402
from dxaqc.hip_backbones import build_hip_model  # noqa: E402
from dxaqc.release_predictor import hip_tensor  # noqa: E402

DATA = ROOT / 'data/interim/train/Исследования'
OUT = ROOT / 'outputs/hip_tta_v1'
TARGETS = ['quality', 'v_rotation', 'v_roi']
# Pre-specified variants: identity + small shifts and zooms (the scanner framing varies by a few mm).
VARIANTS = [(0, 0, 1.0), (3, 0, 1.0), (-3, 0, 1.0), (0, 3, 1.0), (0, -3, 1.0), (0, 0, 0.97), (0, 0, 1.03)]


def warp(img, dx, dy, s):
    if (dx, dy, s) == (0, 0, 1.0):
        return img
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--n-variants', type=int, default=len(VARIANTS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = verify_bundle(ROOT / 'models/release')
    run = manifest['hip_runs'][0]
    oof = pd.read_csv(ROOT / 'outputs/release_validation_v1/assembled_oof.csv')
    lab = pd.read_csv(ROOT / 'data/interim/image_labels.csv')[['image_uid', 'path']].rename(columns={'path': 'dcm_path'})
    d = oof[oof.region.str.startswith('hip')].merge(lab, on='image_uid', validate='1:1').reset_index(drop=True)
    variants = VARIANTS[:args.n_variants]
    scores = np.zeros((len(d), len(TARGETS)))
    t0 = time.time()
    with torch.inference_mode():
        for fold, ck in enumerate(run['checkpoints']):
            m = build_hip_model(run['arch'], outputs=3, pretrained=False)
            m.load_state_dict(torch.load(bundle_path(ROOT / 'models/release', ck), map_location='cpu', weights_only=True))
            m.eval().to(args.device)
            idx = np.where(d.fold.values == fold)[0]
            for i in idx:
                img = read_dicom(DATA / d.dcm_path[i]).pixels
                acc = []
                for dx, dy, s in variants:
                    x = hip_tensor(warp(img, dx, dy, s), d.region[i], run['res']).to(args.device)
                    acc.append(torch.sigmoid(m(x)).cpu().numpy()[0])
                scores[i] = np.mean(acc, axis=0)
    sec = time.time() - t0
    res, L = {'variants': len(variants), 'seconds_total': sec, 'seconds_per_image': sec / len(d)}, \
        ['| Цель | Вариант | AUC | AP | F1 (внутр. порог) | ΔAUC [95%] | ΔAP [95%] |', '|---|---|---:|---:|---:|---|---|']
    for j, t in enumerate(TARGETS):
        col = {'quality': 'p_quality', 'v_rotation': 'p_hip_positioning', 'v_roi': 'p_hip_roi_margins'}[t]
        keep = d[t].notna().values
        y, g, fo = d.loc[keep, t].astype(int).values, d.loc[keep, 'study'].values, d.loc[keep, 'fold'].values
        base, tta = d.loc[keep, col].values, scores[keep, j]
        r = {}
        for name, s in (('single (release)', base), ('TTA', tta)):
            dec = F.cnn_decisions(s, y, g, fo)
            r[name] = {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s), 'f1': f1_score(y, dec)}
        ci_auc, ci_ap = F.paired_boot(y, base, tta, g)
        r['TTA']['d_auc'], r['TTA']['d_ap'] = ci_auc, ci_ap
        res[t] = r
        for name in ('single (release)', 'TTA'):
            x = r[name]
            ci = f"[{ci_auc[0]:+.3f}; {ci_auc[1]:+.3f}] | [{ci_ap[0]:+.3f}; {ci_ap[1]:+.3f}]" if name == 'TTA' else ' | '
            L.append(f"| {t} | {name} | {x['auc']:.3f} | {x['ap']:.3f} | {x['f1']:.3f} | {ci} |")
    pd.DataFrame({'image_uid': d.image_uid, 'study': d.study, 'fold': d.fold,
                  **{f'p_{t}_tta': scores[:, j] for j, t in enumerate(TARGETS)}}).to_csv(OUT / 'oof_tta.csv', index=False)
    (OUT / 'eval.json').write_text(json.dumps(res, indent=2, default=float))
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f"{len(variants)} variants, {sec:.0f} s total, {sec / len(d):.2f} s per image ({args.device})")


if __name__ == '__main__':
    main()
