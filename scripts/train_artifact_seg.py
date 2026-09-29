"""Artifact segmenter trained only on synthetic metal over external (Arak) spine backgrounds.

Evaluation on our 99 labelled spine images is zero-shot: our labels are never
used for training, only for the reported AUC (with --bg-ours our images are seen
as label-blind backgrounds, i.e. transductive, still label-free). Score per image = mean of the
top-K pixel probabilities (K=40), plus predicted artifact area.

H100:  python scripts/train_artifact_seg.py --arch convnext_small --epochs 60 --batch 32 --device cuda --workers 12
Mac:   python scripts/train_artifact_seg.py --epochs 8 --batch 12 --device mps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'models/torch_home'))

from artifact_synth import apply_artifacts  # noqa: E402
from dxaqc.hip_backbones import build_hip_model  # noqa: E402

SIZE = 384


class SegNet(nn.Module):
    """ConvNeXt encoder + FPN, logits at full input resolution."""

    def __init__(self, arch='convnext_tiny', pretrained=True, width=96):
        super().__init__()
        self.features = build_hip_model(arch, 1, pretrained=pretrained).features
        chans = {'convnext_tiny': [96, 192, 384, 768], 'convnext_small': [96, 192, 384, 768],
                 'convnext_base': [128, 256, 512, 1024]}[arch]
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in chans)
        self.fuse = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU())
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU())
        self.out = nn.Sequential(nn.Conv2d(width + 32, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):  # x (B,1,H,W) normalized
        h = x.repeat(1, 3, 1, 1)
        feats = []
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in (1, 3, 5, 7):
                feats.append(h)
        p = self.lateral[3](feats[3])
        for k in (2, 1, 0):
            p = F.interpolate(p, size=feats[k].shape[-2:], mode='bilinear', align_corners=False) + self.lateral[k](feats[k])
        p = F.interpolate(self.fuse(p), size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Full-resolution stem keeps 1-3 px wires that stride-4 features blur.
        return self.out(torch.cat([p, self.stem(x)], 1))


def letterbox(img, size=SIZE):
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    c = np.zeros((size, size), img.dtype)
    c[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if img.dtype == np.uint8 else cv2.INTER_NEAREST)
    return c


def norm(u8):
    return (u8.astype(np.float32) / 255 - 0.449) / 0.226


class Synth(torch.utils.data.Dataset):
    def __init__(self, paths, n, train, seed):
        self.paths, self.n, self.train, self.seed, self.epoch = paths, n, train, seed, 0
        self.cache = {}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch if self.train else 0, i))
        p = self.paths[rng.integers(len(self.paths))]
        img = cv2.imread(str(ROOT / p), 0)
        if self.train:
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()
            f = (img.astype(np.float32) / 255) ** rng.uniform(0.7, 1.4)
            img = np.clip(f * 255, 0, 255).astype(np.uint8)
        if rng.random() < 0.6:
            img, mask, _ = apply_artifacts(img, rng)
        else:
            mask = np.zeros_like(img)
        if self.train and rng.random() < 0.5:
            img = np.clip(img + rng.normal(0, rng.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
        return torch.from_numpy(norm(letterbox(img)))[None], torch.from_numpy(letterbox(mask).astype(np.float32))[None]


def loss_fn(logits, m):
    bce = F.binary_cross_entropy_with_logits(logits, m, pos_weight=torch.tensor(5.0, device=m.device))
    p = torch.sigmoid(logits)
    inter = (p * m).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def score_ours(model, device, k=40):
    lab = pd.read_csv(ROOT / 'data/interim/image_labels.csv')
    sp = lab[lab.region.eq('spine') & lab.v_artifact.notna()]
    rows, probs = [], {}
    for r in sp.itertuples():
        img = cv2.imread(str(ROOT / 'data/interim/ssl_png' / f'{hashlib.sha1(r.image_uid.encode()).hexdigest()[:16]}.png'), 0)
        x = torch.from_numpy(norm(letterbox(img)))[None, None].to(device)
        p = torch.sigmoid(model(x)).float()
        p = (p + torch.flip(torch.sigmoid(model(torch.flip(x, [-1]))).float(), [-1])) / 2
        flat = p.flatten()
        rows.append({'image_uid': r.image_uid, 'study': r.study, 'y': int(r.v_artifact),
                     'topk': float(flat.topk(k).values.mean()), 'area': float((flat > 0.5).float().mean() * 1e3)})
        probs[r.image_uid] = (img, p[0, 0].cpu().numpy())
    df = pd.DataFrame(rows)
    res = {s: {'auc': float(roc_auc_score(df.y, df[s])), 'ap': float(average_precision_score(df.y, df[s]))} for s in ('topk', 'area')}
    return res, df, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='convnext_tiny', choices=['convnext_tiny', 'convnext_small', 'convnext_base'])
    ap.add_argument('--init-encoder', type=Path, help='SSL checkpoint for the encoder')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--per-epoch', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--bg-ours', action='store_true',
                    help='also use ALL our spine images as backgrounds (label-blind, transductive domain adaptation)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT / 'outputs/artifact_seg_v1'))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / 'outputs/ssl_v1/manifest.csv')
    bg = man[man.source.eq('arak') & man.view.eq('spine_ap')]  # Pakistan PNGs carry drawn overlays
    keys = bg.group.map(lambda g: int(hashlib.sha256(g.encode()).hexdigest(), 16) % 10)
    tr_paths, va_paths = bg[keys >= 1].path.tolist(), bg[keys < 1].path.tolist()
    if args.bg_ours:
        # Every spine image, regardless of its artifact label: the model sees our domain,
        # never our labels. Unlabelled real artifacts in these backgrounds act as label noise.
        tr_paths += man[man.source.eq('ours') & man.view.eq('spine_ap')].path.tolist()
    model = SegNet(args.arch).to(args.device)
    if args.init_encoder:
        ck = torch.load(args.init_encoder, map_location='cpu', weights_only=False)
        model.features.load_state_dict(ck['features'])
    tr = Synth(tr_paths, args.per_epoch, True, args.seed)
    va = Synth(va_paths, 300, False, args.seed + 1)
    dl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    vl = torch.utils.data.DataLoader(va, batch_size=args.batch, num_workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    amp = args.device == 'cuda'
    (out / 'config.json').write_text(json.dumps({**vars(args), 'train_bg': len(tr_paths), 'val_bg': len(va_paths)}, indent=2, default=str))
    log, best = [], -1
    for ep in range(args.epochs):
        tr.epoch = ep
        model.train()
        t0, tot = time.time(), 0.0
        for x, m in dl:
            x, m = x.to(args.device), m.to(args.device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                logits = model(x).float()
            loss = loss_fn(logits, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        inter = union = 0.0
        with torch.no_grad():
            for x, m in vl:
                p = (torch.sigmoid(model(x.to(args.device)).float()) > 0.5).float().cpu()
                inter += (p * m).sum().item()
                union += (p.sum() + m.sum()).item()
        ours, df, _ = score_ours(model, args.device)
        row = {'epoch': ep + 1, 'loss': tot / len(dl), 'synth_val_dice': 2 * inter / max(union, 1),
               'ours_topk_auc': ours['topk']['auc'], 'ours_topk_ap': ours['topk']['ap'],
               'ours_area_auc': ours['area']['auc'], 'sec': round(time.time() - t0, 1)}
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        (out / 'log.json').write_text(json.dumps(log, indent=1))
        # Checkpoint selection on synthetic validation only (never on our labels).
        if row['synth_val_dice'] > best:
            best = row['synth_val_dice']
            torch.save({'arch': args.arch, 'model': model.state_dict(), 'epoch': ep + 1}, out / 'best.pt')
            df.to_csv(out / 'ours_scores.csv', index=False)
    ck = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    ours, df, probs = score_ours(model, args.device)
    (out / 'metrics.json').write_text(json.dumps({'best_epoch': ck['epoch'], 'synth_val_dice': best, 'ours_zero_shot': ours}, indent=2))
    tiles = []
    for r in df.sort_values(['y', 'topk'], ascending=[False, False]).head(16).itertuples():
        img, p = probs[r.image_uid]
        v = cv2.cvtColor(letterbox(img), cv2.COLOR_GRAY2BGR)
        v[p > 0.5] = (0, 0, 255)
        cv2.putText(v, f"y={r.y} s={r.topk:.2f}", (4, 16), 0, 0.5, (0, 255, 255), 1)
        tiles.append(cv2.resize(v, (240, 240)))
    cv2.imwrite(str(out / 'ours_top.jpg'), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 16, 8)]))
    print('FINAL', json.dumps(ours), flush=True)


if __name__ == '__main__':
    main()
