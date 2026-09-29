"""Train the landmark heatmap model on cleaned Arak DXA with Lunar-derived labels.

Split: by patient_key (prefix of file name), 80/10/10 via a stable hash, so the
hip (-1) and spine (-2) scans of one patient never cross splits.

  .venv/bin/python scripts/train_landmarks.py --epochs 30 --device mps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'models/torch_home'))

from dxaqc.landmarks import (FLIP_PERM, HIP_CH, INPUT, NECK_HALF, POINTS, SPINE_CH,  # noqa: E402
                             STRIDE, LandmarkNet, decode, letterbox)

HM = INPUT // STRIDE


def split_of(key: str) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    return 'train' if h < 80 else ('val' if h < 90 else 'test')


def spine_column_x(img, box, ys):
    """Column centre per vertebra band from brightness (bone is bright)."""
    x0, _, x1, _ = box
    xs = []
    for ya, yb in zip(ys[:-1], ys[1:]):
        band = img[int(max(ya, 0)):int(min(yb, img.shape[0])), x0:x1 + 1].astype(np.float32)
        if band.size == 0:
            xs.append(np.nan)
            continue
        prof = cv2.GaussianBlur(band.mean(0)[None], (0, 0), 4).ravel()
        thr = (prof.max() + np.median(prof)) / 2
        k = int(prof.argmax())
        l, r = k, k
        while l > 0 and prof[l - 1] >= thr:
            l -= 1
        while r < len(prof) - 1 and prof[r + 1] >= thr:
            r += 1
        xs.append(x0 + (l + r) / 2)
    return xs


def targets(rec, img):
    """Return (K,2) points in image px and (K,) validity mask."""
    pts = np.zeros((len(POINTS), 2), np.float32)
    ok = np.zeros(len(POINTS), bool)
    if rec['view'] == 'hip':
        x0, y0, x1, y1 = rec['box']
        h = rec['hip']
        c = np.array(h['neck_center'], np.float32)
        th = math.radians(h['neck_axis_angle_deg'])
        u = np.array([math.cos(th), math.sin(th)], np.float32)  # points down: toward trochanter
        pts[HIP_CH] = [[(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1], [x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2],
                       c, c - NECK_HALF * u, c + NECK_HALF * u]
        ok[HIP_CH] = True
    else:
        b = rec['spine']['boundaries']
        x0, _, x1, _ = rec['box']
        cxb = (x0 + x1) / 2
        ys = [l['y_center'] for l in b]
        vx = spine_column_x(img, rec['box'], ys)
        for i, l in enumerate(b):
            near = [v for v in (vx[i - 1] if i > 0 else np.nan, vx[i] if i < len(vx) else np.nan) if not np.isnan(v)]
            x = float(np.mean(near)) if near else cxb
            y = l['y_center'] + (x - cxb) * math.tan(math.radians(l['angle_deg']))
            pts[SPINE_CH[i]] = [x, y]
        ok[SPINE_CH] = True
    return pts, ok


def load_records():
    recs = []
    for line in open(ROOT / 'outputs/arak_overlays_v1/annotations.jsonl'):
        r = json.loads(line)
        if r['parse_ok'] and r.get('clean_path') and not r.get('duplicate_of'):
            r['split'] = split_of(r['patient_key'])
            recs.append(r)
    return recs


def augment(img, pts, rng):
    h, w = img.shape
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        pts = pts.copy()
        pts[:, 0] = w - 1 - pts[:, 0]
        pts = pts[FLIP_PERM]
    ang = rng.uniform(-10, 10)
    sc = rng.uniform(0.85, 1.2)
    tx, ty = rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.08, 0.08) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pts = pts @ M[:, :2].T + M[:, 2]
    f = img.astype(np.float32) / 255
    f = f ** rng.uniform(0.6, 1.6)
    f = f * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.4:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.3))
    if rng.random() < 0.3:
        d = rng.uniform(0.5, 0.85)
        f = cv2.resize(cv2.resize(f, None, fx=d, fy=d, interpolation=cv2.INTER_AREA), (w, h))
    f = f + rng.normal(0, rng.uniform(0, 0.05), f.shape).astype(np.float32)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8), pts


def heatmaps(pts, ok, sigma=1.5):
    ys, xs = np.mgrid[0:HM, 0:HM].astype(np.float32)
    out = np.zeros((len(POINTS), HM, HM), np.float32)
    for k in np.where(ok)[0]:
        cx, cy = (pts[k] - (STRIDE - 1) / 2) / STRIDE
        out[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return out


class DS(torch.utils.data.Dataset):
    def __init__(self, recs, train, seed=0):
        self.recs, self.train, self.seed = recs, train, seed
        self.cache = {}

    def __len__(self):
        return len(self.recs)

    def item(self, i):
        if i not in self.cache:
            r = self.recs[i]
            img = cv2.imread(str(ROOT / r['clean_path']), cv2.IMREAD_GRAYSCALE)
            pts, ok = targets(r, img)
            self.cache[i] = (img, pts, ok)
        return self.cache[i]

    def __getitem__(self, i):
        img, pts, ok = self.item(i)
        if self.train:
            rng = np.random.default_rng((self.seed, i, int(time.time_ns() % 1_000_003)))
            img, pts = augment(img, pts, rng)
        canvas, s = letterbox(img)
        p = pts * s
        vis = ok & (p[:, 0] >= 0) & (p[:, 0] < INPUT) & (p[:, 1] >= 0) & (p[:, 1] < INPUT)
        return (torch.from_numpy(canvas).float().div(255)[None], torch.from_numpy(heatmaps(p, vis)),
                torch.from_numpy(vis), torch.from_numpy(p.astype(np.float32)), torch.tensor(s))


def run_eval(model, loader, device, recs):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, _, vis, p, s in loader:
            x = x.to(device)
            heat = model(x)
            heat_f = torch.flip(model(torch.flip(x, [-1])), [-1])[:, FLIP_PERM]
            coords, _ = decode((heat + heat_f) / 2)
            coords = coords.cpu()
            for b in range(x.shape[0]):
                err = ((coords[b] - p[b]).norm(dim=-1) / s[b]).numpy()  # original px
                rows.append((err, vis[b].numpy(), (coords[b] / s[b]).numpy(), (p[b] / s[b]).numpy()))
    model.train()
    return summarize(rows)


def summarize(rows):
    out = {}
    for k, name in enumerate(POINTS):
        e = [r[0][k] for r in rows if r[1][k]]
        if e:
            out[f'{name}_px_median'] = float(np.median(e))
            out[f'{name}_px_p90'] = float(np.percentile(e, 90))
    hip = [r for r in rows if r[1][HIP_CH].all()]
    if hip:
        def ang(q):
            d = q[6] - q[5]
            return math.degrees(math.atan2(d[1], d[0]))
        out['hip_neck_angle_err_deg_median'] = float(np.median([abs(ang(r[2]) - ang(r[3])) for r in hip]))
    sp = [r for r in rows if r[1][SPINE_CH].all()]
    if sp:
        def tilt(q):
            q = q[SPINE_CH]
            d = q[-1] - q[0]
            return math.degrees(math.atan2(d[0], d[1]))
        out['spine_tilt_err_deg_median'] = float(np.median([abs(tilt(r[2]) - tilt(r[3])) for r in sp]))
        out['spine_L1top_y_err_px_median'] = float(np.median([abs(r[2][SPINE_CH[0], 1] - r[3][SPINE_CH[0], 1]) for r in sp]))
    out['n'] = len(rows)
    out['mean_px_median'] = float(np.mean([v for k, v in out.items() if k.endswith('_px_median') and not k.startswith('spine_L1')]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=str(ROOT / 'outputs/landmarks_v1'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = load_records()
    if args.limit:
        recs = recs[:args.limit]
    parts = {s: [r for r in recs if r['split'] == s] for s in ('train', 'val', 'test')}
    info = {s: {'images': len(v), 'hip': sum(r['view'] == 'hip' for r in v),
                'spine': sum(r['view'] == 'spine_ap' for r in v),
                'patients': len({r['patient_key'] for r in v})} for s, v in parts.items()}
    assert not ({r['patient_key'] for r in parts['train']} & {r['patient_key'] for r in parts['val'] + parts['test']})
    print(json.dumps(info), flush=True)
    (out / 'config.json').write_text(json.dumps({**vars(args), 'splits': info, 'points': POINTS}, indent=2))
    (out / 'split.json').write_text(json.dumps({s: [r['stem'] for r in v] for s, v in parts.items()}))

    dl = lambda rs, tr: torch.utils.data.DataLoader(DS(rs, tr, args.seed), batch_size=args.batch, shuffle=tr,
                                                    num_workers=0, drop_last=tr)
    train_dl, val_dl = dl(parts['train'], True), dl(parts['val'], False)
    model = LandmarkNet().to(args.device)
    enc = list(model.features.parameters())
    enc_ids = {id(p) for p in enc}
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{'params': enc, 'lr': args.lr / 3}, {'params': dec, 'lr': args.lr}], weight_decay=1e-4)
    total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr / 3, args.lr], total_steps=total, pct_start=0.1)

    best, log = 1e9, []
    for ep in range(args.epochs):
        t0, tot = time.time(), 0.0
        for x, hm, vis, _, _ in train_dl:
            x, hm, vis = x.to(args.device), hm.to(args.device), vis.to(args.device).float()
            pred = model(x)
            per = ((pred - hm) ** 2 * (1 + 10 * hm)).mean((-1, -2))
            loss = (per * vis).sum() / vis.sum().clamp(min=1) * 100
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = run_eval(model, val_dl, args.device, parts['val'])
        row = {'epoch': ep + 1, 'train_loss': tot / len(train_dl), 'sec': round(time.time() - t0, 1), **m}
        log.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                          if k in ('epoch', 'train_loss', 'sec', 'mean_px_median', 'hip_neck_angle_err_deg_median',
                                   'spine_tilt_err_deg_median', 'spine_L1top_y_err_px_median')}), flush=True)
        if m['mean_px_median'] < best:
            best = m['mean_px_median']
            torch.save({'model': model.state_dict(), 'epoch': ep + 1, 'val': m, 'points': POINTS}, out / 'best.pt')
        (out / 'log.json').write_text(json.dumps(log, indent=1))

    ck = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    test = run_eval(model, dl(parts['test'], False), args.device, parts['test'])
    (out / 'metrics.json').write_text(json.dumps({'best_epoch': ck['epoch'], 'val': ck['val'], 'test': test}, indent=2))
    print('TEST', json.dumps(test), flush=True)


if __name__ == '__main__':
    main()
