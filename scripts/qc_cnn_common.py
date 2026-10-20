"""Shared fold loop for the new QC networks (docs/PREREG_NEW_MODELS.md).

The recipe and the isolation are those of scripts/train_hip_cnn.py, so that the new arms differ from
the control only in what they look at and how the network is built: AdamW, OneCycle, 30 epochs,
batch 16, seed + fold, prevalence-weighted BCE on the known labels only, and the nested scheme
`--folds j --exclude-folds f` (trained without both folds) used by scripts/run_nested.sh.

A trainer supplies the rows, the label matrix, one image per row, a per-row train transform,
a test transform and a model builder; this module does the rest and writes the same files as
train_hip_cnn.py (config.json, provenance.json, history.json, oof.csv, fold*.pt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/interim/train/Исследования'
os.environ.setdefault('TORCH_HOME', str(ROOT / 'models/torch_home'))
warnings.filterwarnings('ignore')
T = torchvision.transforms
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def parser(description: str, epochs: int = 30, res: int = 384, lr: float = 1e-4, bs: int = 16) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--epochs', type=int, default=epochs)
    ap.add_argument('--res', type=int, default=res)
    ap.add_argument('--lr', type=float, default=lr)
    ap.add_argument('--bs', type=int, default=bs)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--exclude-folds', default='', help='also kept out of training (nested: --folds j --exclude-folds f)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    ap.add_argument('--out-root', type=Path, default=None)
    ap.add_argument('--cpu-threads', type=int, default=8)
    ap.add_argument('--workers', type=int, default=0)
    return ap


def source_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def setup(args, default_root: Path) -> tuple[Path, str]:
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    out = (args.out_root or default_root) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise ValueError(f'Run already exists: {out}; use a new tag')
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str))
    (out / 'provenance.json').write_text(json.dumps({
        'commit': source_commit(),
        'tables': {n: hashlib.sha256((ROOT / 'data/interim' / n).read_bytes()).hexdigest()
                   for n in ('image_labels.csv', 'folds.csv')},
    }, indent=2))
    dev = args.device if args.device != 'auto' else (
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return out, dev


def labelled() -> pd.DataFrame:
    """Every frame with a quality label, with its study fold."""
    il = pd.read_csv(ROOT / 'data/interim/image_labels.csv').merge(
        pd.read_csv(ROOT / 'data/interim/folds.csv'), on=['study', 'image_uid'])
    return il[il.quality.notna()].reset_index(drop=True)


def square(a: np.ndarray) -> Image.Image:
    """Zero-pad to a centred square, as the hip CNN does."""
    h, w = a.shape
    s = max(h, w)
    c = np.zeros((s, s), np.uint8)
    c[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return Image.fromarray(c)


def photometric(res: int, mean=IMAGENET[0], std=IMAGENET[1]):
    """Brightness/contrast only: geometry carries the label (tilt, scan range, trochanter shape)."""
    return T.Compose([T.Resize((res, res)), T.ColorJitter(brightness=0.25, contrast=0.25), T.Grayscale(3),
                      T.ToTensor(), T.Normalize(mean, std)])


def hip_full(res: int, mean=IMAGENET[0], std=IMAGENET[1]):
    """The current hip recipe ('full' in train_hip_cnn.py)."""
    return T.Compose([T.Resize((res, res)), T.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.9, 1.1), fill=0),
                      T.ColorJitter(brightness=0.25, contrast=0.25), T.Grayscale(3), T.ToTensor(), T.Normalize(mean, std),
                      T.RandomErasing(p=0.3, scale=(0.01, 0.05))])


def plain(res: int, mean=IMAGENET[0], std=IMAGENET[1]):
    return T.Compose([T.Resize((res, res)), T.Grayscale(3), T.ToTensor(), T.Normalize(mean, std)])


class _DS(torch.utils.data.Dataset):
    def __init__(self, idx, images, Y, tf):
        self.idx, self.images, self.Y, self.tf = idx, images, Y, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        tf = self.tf(j) if callable(self.tf) and not isinstance(self.tf, T.Compose) else self.tf
        return tf(self.images[j]), torch.from_numpy(self.Y[j])


def run(args, out: Path, dev: str, rows: pd.DataFrame, Y: np.ndarray, targets: list[str], images: list,
        train_tf: Callable[[int], object], test_tf, build: Callable[[], torch.nn.Module],
        finalize: Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> pd.DataFrame:
    """Train one model per requested fold and write out-of-fold probabilities for those folds."""
    Y = Y.astype(np.float32)
    oof = np.full((len(rows), len(targets)), np.nan)
    hist = {}
    exclude = [int(x) for x in args.exclude_folds.split(',') if x.strip() != '']

    @torch.no_grad()
    def predict(m, idx):
        m.eval()
        out_ = []
        for x, _ in torch.utils.data.DataLoader(_DS(idx, images, Y, test_tf), batch_size=16, num_workers=args.workers):
            out_.append(torch.sigmoid(m(x.to(dev)).float()).cpu().numpy())
        return np.concatenate(out_)

    for k in [int(f) for f in args.folds.split(',')]:
        torch.manual_seed(args.seed + k)
        np.random.seed(args.seed + k)
        tr = np.where(~rows.fold.isin([k] + exclude))[0]
        te = np.where(rows.fold == k)[0]
        pos = np.nan_to_num(Y[tr]).sum(0)
        known = (~np.isnan(Y[tr])).sum(0)
        pw = torch.tensor(np.clip((known - pos) / np.maximum(pos, 1), 1, 10), dtype=torch.float32).to(dev)
        m = build().to(dev)
        groups = m.param_groups(args.lr) if hasattr(m, 'param_groups') else [{'params': [p for p in m.parameters() if p.requires_grad], 'lr': args.lr}]
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.05)
        steps = args.epochs * ((len(tr) + args.bs - 1) // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=steps, pct_start=0.15)
        dl = torch.utils.data.DataLoader(_DS(tr, images, Y, train_tf), batch_size=args.bs, shuffle=True,
                                         num_workers=args.workers)
        t0 = time.time()
        hist[k] = []
        for ep in range(args.epochs):
            m.train()
            tot = 0.0
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                mask = ~torch.isnan(y)
                logits = m(x).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[mask], torch.nan_to_num(y)[mask], pos_weight=pw.expand_as(y)[mask])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(x)
            if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
                p = predict(m, te)
                aucs = {}
                for j, t in enumerate(targets):
                    yy = Y[te, j]
                    mm = ~np.isnan(yy)
                    aucs[t] = round(roc_auc_score(yy[mm], p[mm, j]), 3) if len(set(yy[mm])) > 1 else float('nan')
                hist[k].append({'epoch': ep + 1, 'loss': round(tot / len(tr), 4), **aucs})
                print(f'fold {k} ep {ep + 1} loss {tot / len(tr):.4f} val {aucs} ({time.time() - t0:.0f}s)', flush=True)
        oof[te] = predict(m, te)
        torch.save({k_: v for k_, v in m.state_dict().items()}, out / f'fold{k}.pt')
        (out / 'history.json').write_text(json.dumps(hist, indent=1))

    res = rows[['study', 'image_uid', 'region', 'fold']].copy()
    for j, t in enumerate(targets):
        res['p_' + t] = oof[:, j]
    if finalize is not None:
        res = finalize(res)
    res.to_csv(out / 'oof.csv', index=False)
    print('oof ->', out / 'oof.csv')
    return res
