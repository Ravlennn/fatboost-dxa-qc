"""DINO self-distillation on all available DXA images (label-free), ConvNeXt backbone.

Continues from ImageNet weights so the encoder adapts to DXA texture/anatomy.
Output checkpoints ({'arch', 'features', 'epoch'}) plug into
  scripts/train_hip_cnn.py --arch <arch> --init-encoder <ckpt>
  scripts/eval_ssl_probe.py --ckpt <ckpt>

H100 (one GPU):  python scripts/train_ssl_dxa.py --arch convnext_small --epochs 200 --batch 128 --device cuda --workers 12
Smoke (Mac):     python scripts/train_ssl_dxa.py --arch convnext_tiny --epochs 1 --limit 64 --batch 8 --device mps

Reference: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (DINO), 2021.
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'models/torch_home'))
from dxaqc.hip_backbones import build_hip_model  # noqa: E402

MEAN, STD = 0.449, 0.226  # grey ImageNet stats (x repeated to 3 channels)


# ------------------------------------------------------------------ augment
def rand_resized_crop(img, out, scale, rng):
    h, w = img.shape
    for _ in range(10):
        area = h * w * rng.uniform(*scale)
        ar = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
        cw, ch = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
            return cv2.resize(img[y:y + ch, x:x + cw], (out, out), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (out, out), interpolation=cv2.INTER_AREA)


def photometric(x, rng, blur_p):
    f = x.astype(np.float32) / 255
    if rng.random() < 0.5:
        f = f[:, ::-1]
    ang = rng.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((f.shape[1] / 2, f.shape[0] / 2), ang, 1.0)
    f = cv2.warpAffine(f, M, f.shape[::-1], borderMode=cv2.BORDER_CONSTANT)
    f = np.clip(f, 0, 1) ** rng.uniform(0.6, 1.6)
    f = f * rng.uniform(0.75, 1.25) + rng.uniform(-0.1, 0.1)
    if rng.random() < blur_p:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.5:
        f = f + rng.normal(0, rng.uniform(0.0, 0.06), f.shape).astype(np.float32)
    if rng.random() < 0.2:  # solarization, as in DINO/BYOL
        f = np.where(f > 0.5, 1 - f, f)
    return (np.clip(f, 0, 1) - MEAN) / STD


class MultiCrop(torch.utils.data.Dataset):
    def __init__(self, paths, gsize, lsize, n_local, seed):
        self.paths, self.gsize, self.lsize, self.n_local, self.seed = paths, gsize, lsize, n_local, seed
        self.epoch = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        img = cv2.imread(str(ROOT / self.paths[i]), cv2.IMREAD_GRAYSCALE)
        crops = [photometric(rand_resized_crop(img, self.gsize, (0.35, 1.0), rng), rng, b) for b in (1.0, 0.1)]
        crops += [photometric(rand_resized_crop(img, self.lsize, (0.08, 0.35), rng), rng, 0.5) for _ in range(self.n_local)]
        return [torch.from_numpy(np.ascontiguousarray(c))[None] for c in crops]


# ------------------------------------------------------------------ model
class DINOHead(nn.Module):
    def __init__(self, dim, out=8192, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, out, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class Net(nn.Module):
    def __init__(self, arch, pretrained=True):
        super().__init__()
        base = build_hip_model(arch, outputs=1, pretrained=pretrained)
        self.features = base.features
        self.dim = base.classifier[2].in_features
        self.norm = nn.LayerNorm(self.dim)
        self.head = DINOHead(self.dim)

    def embed(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.norm(self.features(x).mean((-2, -1)))

    def forward(self, crops):
        # Same-resolution crops are batched together.
        out, i = [], 0
        while i < len(crops):
            j = i
            while j < len(crops) and crops[j].shape[-1] == crops[i].shape[-1]:
                j += 1
            out.append(self.head(self.embed(torch.cat(crops[i:j]))))
            i = j
        return torch.cat(out)


def dino_loss(s_out, t_out, center, n_crops, t_temp, s_temp=0.1):
    s = (s_out / s_temp).chunk(n_crops)
    t = F.softmax((t_out - center) / t_temp, dim=-1).detach().chunk(2)
    total, n = 0.0, 0
    for iq, q in enumerate(t):
        for v in range(n_crops):
            if v == iq:
                continue
            total = total + torch.sum(-q * F.log_softmax(s[v], dim=-1), dim=-1).mean()
            n += 1
    return total / n


def cosine(base, final, epochs, steps, warmup=0):
    sched = final + 0.5 * (base - final) * (1 + np.cos(np.pi * np.arange(epochs * steps) / (epochs * steps)))
    if warmup:
        sched[:warmup * steps] = np.linspace(0, base, warmup * steps)
    return sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=str(ROOT / 'outputs/ssl_v1/manifest.csv'))
    ap.add_argument('--arch', default='convnext_small', choices=['convnext_tiny', 'convnext_small', 'convnext_base'])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4, help='per 256 images')
    ap.add_argument('--global-size', type=int, default=320)
    ap.add_argument('--local-size', type=int, default=128)
    ap.add_argument('--n-local', type=int, default=6)
    ap.add_argument('--teacher-temp', type=float, default=0.04)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--out', default=str(ROOT / 'outputs/ssl_v1'))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = pd.read_csv(args.manifest)['path'].tolist()
    if args.limit:
        paths = paths[:args.limit]
    ds = MultiCrop(paths, args.global_size, args.local_size, args.n_local, args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                                     persistent_workers=args.workers > 0, pin_memory=args.device == 'cuda')
    student, teacher = Net(args.arch).to(args.device), Net(args.arch).to(args.device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    params = [{'params': [p for n, p in student.named_parameters() if p.requires_grad and p.ndim > 1]},
              {'params': [p for n, p in student.named_parameters() if p.requires_grad and p.ndim <= 1], 'weight_decay': 0}]
    opt = torch.optim.AdamW(params, lr=0, weight_decay=0.04)
    steps = len(dl)
    lr_s = cosine(args.lr * args.batch / 256, 1e-6, args.epochs, steps, warmup=min(10, max(args.epochs // 10, 0)))
    wd_s = cosine(0.04, 0.4, args.epochs, steps)
    mom_s = cosine(0.996, 1.0, args.epochs, steps)
    center = torch.zeros(1, student.head.last.out_features, device=args.device)
    amp = args.device == 'cuda'
    n_crops = 2 + args.n_local
    (out / 'config.json').write_text(json.dumps({**vars(args), 'images': len(paths), 'method': 'DINO'}, indent=2))
    log = []
    it = 0
    for ep in range(args.epochs):
        ds.epoch = ep
        t0, tot = time.time(), 0.0
        for crops in dl:
            crops = [c.to(args.device, non_blocking=True) for c in crops]
            for k, g in enumerate(opt.param_groups):
                g['lr'] = lr_s[it]
                if k == 0:
                    g['weight_decay'] = wd_s[it]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                t_out = teacher(crops[:2]).float()
                s_out = student(crops).float()
            loss = dino_loss(s_out, t_out, center, n_crops, args.teacher_temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            if ep == 0:  # freeze last layer during the first epoch (DINO recipe)
                for p in student.head.last.parameters():
                    p.grad = None
            opt.step()
            with torch.no_grad():
                m = mom_s[it]
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                center.mul_(0.9).add_(t_out.mean(0, keepdim=True), alpha=0.1)
            tot += loss.item()
            it += 1
        row = {'epoch': ep + 1, 'loss': tot / max(steps, 1), 'sec': round(time.time() - t0, 1), 'lr': float(lr_s[it - 1])}
        log.append(row)
        print(json.dumps(row), flush=True)
        (out / 'log.json').write_text(json.dumps(log, indent=1))
        if (ep + 1) % args.save_every == 0 or ep + 1 == args.epochs:
            ck = {'arch': args.arch, 'epoch': ep + 1, 'features': teacher.features.state_dict(), 'method': 'DINO-teacher'}
            torch.save(ck, out / f'encoder_ep{ep + 1:04d}.pt')
            torch.save(ck, out / 'encoder_last.pt')


if __name__ == '__main__':
    main()
