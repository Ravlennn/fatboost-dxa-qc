"""Landmarks v2: v1 points + lesser trochanter / Th12 / iliac crests from manual labels.

Starts from the v1 checkpoint (12 channels) and adds 6 channels. Batches mix
  * Arak cleaned images (weak Lunar labels, v1 channels only), and
  * our images labelled in labeler_v2.html (new channels; v1 channels only where
    the annotator moved them). Status 'invisible' is trained as an all-zero heatmap,
    so the heatmap peak doubles as a visibility score.
Our labelled images are split by the release study folds: --folds 0,1,2,3,4 gives
honest OOF predictions for our images; 'all' trains the deployable model.

H100:  python scripts/train_landmarks_v2.py --labels outputs/landmark_labeling/landmarks_manual_v2.json --device cuda --folds 0,1,2,3,4,all
Mac smoke: add --epochs 1 --arak-per-epoch 64
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'models/torch_home'))

import train_landmarks as V1  # noqa: E402
from dxaqc.dicom_io import read_dicom  # noqa: E402
from dxaqc.landmarks import INPUT, POINTS, POINTS_V2, STRIDE, LandmarkNet, decode, flip_perm, letterbox  # noqa: E402

K = len(POINTS_V2)
PERM = flip_perm(POINTS_V2)
HM = INPUT // STRIDE
DATA = ROOT / 'data/interim/train/Исследования'


def augment(img, pts, rng):
    h, w = img.shape
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        pts = pts.copy()
        pts[:, 0] = w - 1 - pts[:, 0]
        pts = pts[PERM]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.85, 1.2))
    M[:, 2] += (rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.08, 0.08) * h)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    pts = pts @ M[:, :2].T + M[:, 2]
    f = (img.astype(np.float32) / 255) ** rng.uniform(0.6, 1.6) * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.4:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.3))
    f = f + rng.normal(0, rng.uniform(0, 0.05), f.shape).astype(np.float32)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8), pts


def heatmaps(pts, sup, present, sigma=1.5):
    ys, xs = np.mgrid[0:HM, 0:HM].astype(np.float32)
    out = np.zeros((K, HM, HM), np.float32)
    for k in np.where(sup & present)[0]:
        cx, cy = (pts[k] - (STRIDE - 1) / 2) / STRIDE
        out[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return out


class Items(torch.utils.data.Dataset):
    """items: list of (image uint8, pts (K,2), supervised mask (K,), present mask (K,))."""

    def __init__(self, items, train, seed=0):
        self.items, self.train, self.seed = items, train, seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, pts, sup, present = self.items[i]
        if self.train:
            # Arak masks are flip-symmetric (all v1 points or none), so only points need permuting.
            img, pts = augment(img, pts, np.random.default_rng((self.seed, i, time.time_ns() % 1_000_003)))
        canvas, s = letterbox(img)
        p = pts * s
        inside = (p[:, 0] >= 0) & (p[:, 0] < INPUT) & (p[:, 1] >= 0) & (p[:, 1] < INPUT)
        # A present point pushed out of frame by augmentation is no longer supervised.
        sup2 = sup & (inside | ~present)
        return (torch.from_numpy(canvas).float().div(255)[None], torch.from_numpy(heatmaps(p, sup2, present & inside)),
                torch.from_numpy(sup2))


class Flipping(Items):
    """Items where masks must follow the flip permutation (our labels have per-point status)."""

    def __getitem__(self, i):
        img, pts, sup, present = self.items[i]
        if self.train:
            rng = np.random.default_rng((self.seed, i, time.time_ns() % 1_000_003))
            h, w = img.shape
            if rng.random() < 0.5:
                img, pts = img[:, ::-1].copy(), pts.copy()
                pts[:, 0] = w - 1 - pts[:, 0]
                pts, sup, present = pts[PERM], sup[PERM], present[PERM]
            # remaining (non-flip) augmentation
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(0.9, 1.15))
            M[:, 2] += (rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.06, 0.06) * h)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
            pts = pts @ M[:, :2].T + M[:, 2]
            f = (img.astype(np.float32) / 255) ** rng.uniform(0.7, 1.4) * rng.uniform(0.85, 1.15)
            img = (np.clip(f + rng.normal(0, 0.02, f.shape), 0, 1) * 255).astype(np.uint8)
        canvas, s = letterbox(img)
        p = pts * s
        inside = (p[:, 0] >= 0) & (p[:, 0] < INPUT) & (p[:, 1] >= 0) & (p[:, 1] < INPUT)
        sup2 = sup & (inside | ~present)
        return (torch.from_numpy(canvas).float().div(255)[None], torch.from_numpy(heatmaps(p, sup2, present & inside)),
                torch.from_numpy(sup2))


def arak_items(split):
    recs = [r for r in V1.load_records() if r['split'] == split]
    items = []
    for r in recs:
        img = cv2.imread(str(ROOT / r['clean_path']), cv2.IMREAD_GRAYSCALE)
        p12, ok12 = V1.targets(r, img)
        pts = np.zeros((K, 2), np.float32)
        pts[:12] = p12
        sup = np.zeros(K, bool)
        sup[:12] = ok12
        items.append((img, pts, sup, sup.copy()))
    return items


def our_items(labels, uids=None):
    items, meta = [], []
    for r in labels:
        if uids is not None and r['id'] not in uids:
            continue
        img = read_dicom(DATA / r['path']).pixels
        pts = np.zeros((K, 2), np.float32)
        sup = np.zeros(K, bool)
        present = np.zeros(K, bool)
        for k, n in enumerate(POINTS_V2):
            stt = r['status'].get(n)
            if stt in ('set', 'invisible'):
                sup[k] = True
                present[k] = stt == 'set'
                pts[k] = r['points'][n] if stt == 'set' else (0, 0)
        if sup.any():
            items.append((img, pts, sup, present))
            meta.append(r)
    return items, meta


@torch.no_grad()
def predict(model, device, labels_all):
    model.eval()
    out = {}
    for r in labels_all:
        img = read_dicom(DATA / r['path']).pixels
        canvas, s = letterbox(img)
        x = torch.from_numpy(canvas).float().div(255)[None, None].to(device)
        heat = (model(x) + torch.flip(model(torch.flip(x, [-1])), [-1])[:, PERM]) / 2
        c, peak = decode(heat)
        out[r['id']] = {'points': {n: (c[0, k] / s).tolist() for k, n in enumerate(POINTS_V2)},
                        'peak': {n: float(peak[0, k]) for k, n in enumerate(POINTS_V2)}}
    model.train()
    return out


def build_from_v1(v1_ckpt, device):
    model = LandmarkNet(n_points=K)
    ck = torch.load(v1_ckpt, map_location='cpu', weights_only=False)
    state = ck['model']
    own = model.state_dict()
    for k, v in state.items():
        if k.startswith('head.'):
            own[k][:v.shape[0]] = v
        else:
            own[k] = v
    model.load_state_dict(own)
    return model.to(device)


def run_fold(fold, labels, lab_folds, args, out):
    torch.manual_seed(args.seed + (0 if fold == 'all' else int(fold)))
    uid_fold = lab_folds
    train_uids = {r['id'] for r in labels if fold == 'all' or uid_fold.get(r['id']) != int(fold)}
    ours, _ = our_items(labels, train_uids)
    arak = arak_items('train')
    if args.arak_per_epoch:
        arak = arak[:args.arak_per_epoch]
    ds = torch.utils.data.ConcatDataset([Items(arak, True, args.seed), Flipping(ours * args.ours_repeat, True, args.seed)])
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers)
    model = build_from_v1(args.v1, args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    amp = args.device == 'cuda'
    for ep in range(args.epochs):
        tot, t0 = 0.0, time.time()
        for x, hm, sup in dl:
            x, hm, sup = x.to(args.device), hm.to(args.device), sup.to(args.device).float()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                pred = model(x).float()
            per = ((pred - hm) ** 2 * (1 + 10 * hm)).mean((-1, -2))
            loss = (per * sup).sum() / sup.sum().clamp(min=1) * 100
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(json.dumps({'fold': fold, 'epoch': ep + 1, 'loss': round(tot / len(dl), 4), 'sec': round(time.time() - t0, 1),
                          'ours_train': len(ours)}), flush=True)
    torch.save({'model': model.state_dict(), 'points': POINTS_V2, 'fold': fold, 'epoch': args.epochs}, out / f'fold_{fold}.pt')
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, type=Path)
    ap.add_argument('--v1', default=str(ROOT / 'outputs/landmarks_v1/best.pt'))
    ap.add_argument('--folds', default='0,1,2,3,4,all')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--ours-repeat', type=int, default=8, help='oversampling of our labelled images per epoch')
    ap.add_argument('--arak-per-epoch', type=int, default=0, help='limit Arak images (smoke tests)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT / 'outputs/landmarks_v2'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(ROOT / 'data/interim/image_labels.csv').merge(pd.read_csv(ROOT / 'data/interim/folds.csv'), on=['study', 'image_uid'])
    path_of = dict(zip(lab.image_uid, lab.path))
    labels = json.loads(args.labels.read_text())
    for r in labels:
        r['path'] = path_of[r['id']]
    lab_folds = dict(zip(lab.image_uid, lab.fold))
    all_rows = [{'id': u, 'path': p} for u, p in zip(lab.image_uid, lab.path)]
    (out / 'config.json').write_text(json.dumps({**vars(args), 'n_labelled': len(labels), 'points': POINTS_V2}, indent=2, default=str))
    oof = {}
    for fold in args.folds.split(','):
        model = run_fold(fold, labels, lab_folds, args, out)
        if fold != 'all':
            held = [r for r in all_rows if lab_folds.get(r['id']) == int(fold)]
            oof.update(predict(model, args.device, held))
            (out / 'oof_points.json').write_text(json.dumps(oof))
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
