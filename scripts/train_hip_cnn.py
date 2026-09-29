"""Дообучение convnext_tiny на бёдрах (384 px, мультилейбл: quality, rotation, roi), 5 фолдов → OOF.
Запуск: TORCH_HOME=models/torch_home .venv/bin/python scripts/train_hip_cnn.py [--epochs 30] [--res 384]
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, pydicom, torch, torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch_home")); warnings.filterwarnings("ignore")
from dxaqc.metrics import bootstrap_ci, format_table, best_threshold, binary_metrics

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--res", type=int, default=384)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--tag", default="convnext384")
ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--arch", default="convnext_tiny")
ap.add_argument("--crop", type=int, default=0, help="размер кропа малого вертела в нативных px (0 = весь кадр)")
ap.add_argument('--cpu-threads', type=int, default=2)
ap.add_argument('--batch-pause', type=float, default=1.0)
ap.add_argument('--resume-from', type=Path)
ap.add_argument('--roi-file', type=str, help='External-only segmenter boxes in mirrored native coordinates')
ap.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
ap.add_argument('--init-encoder', type=Path, help='SSL checkpoint (scripts/train_ssl_dxa.py) for model.features')
ap.add_argument('--workers', type=int, default=0, help='DataLoader workers (GPU: 8)')
ap.add_argument('--out-root', type=Path, default=None, help='default outputs/hip_cnn')
args = ap.parse_args()
if args.cpu_threads < 1 or args.batch_pause < 0:
    ap.error('cpu-threads must be positive and batch-pause nonnegative')
if args.resume_from: args.resume_from = str(args.resume_from.resolve())
torch.set_num_threads(args.cpu_threads)
torch.set_num_interop_threads(1)
def source_commit():
    # Portable kit (h100_kit/build_portable_kit.sh) has no .git: the commit is in ./COMMIT.
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        f = ROOT / 'COMMIT'
        return f.read_text().strip() if f.is_file() else 'unknown'


OUT = (args.out_root or ROOT / "outputs/hip_cnn") / args.tag; OUT.mkdir(parents=True, exist_ok=True)
if args.init_encoder: args.init_encoder = str(args.init_encoder.resolve())
if (OUT / 'config.json').exists():
    raise ValueError(f'Run already exists: {OUT}; use a new tag')
(OUT / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str))
(OUT / 'provenance.json').write_text(json.dumps({
    'commit': source_commit(),
    'tables': {name: hashlib.sha256((ROOT / 'data/interim' / name).read_bytes()).hexdigest()
               for name in ['image_labels.csv', 'folds.csv']},
    'split_status': json.loads((ROOT / 'data/interim/local_provenance.json').read_text())['status']
                   if (ROOT / 'data/interim/local_provenance.json').exists() else 'EXTERNAL_UNVERIFIED',
}, indent=2))
DEV = args.device if args.device != 'auto' else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
AMP = DEV == 'cuda'  # bf16 autocast on H100; weights stay fp32
torch.manual_seed(args.seed); np.random.seed(args.seed)
TARGETS = ["quality", "v_rotation", "v_roi"]

il = pd.read_csv(ROOT / "data/interim/image_labels.csv").merge(pd.read_csv(ROOT / "data/interim/folds.csv"), on=["study", "image_uid"])
hip = il[(il.region != "spine") & il.quality.notna()].reset_index(drop=True)
print("hips", len(hip), "device", DEV, "res", args.res)

def load(r):
    a = pydicom.dcmread(str(ROOT / "data/interim/train/Исследования" / r.path)).pixel_array
    if r.region == "hip_left": a = np.ascontiguousarray(a[:, ::-1])
    if args.crop:
        from dxaqc.hip_rules import lt_crop
        a, _ = lt_crop(a, size=args.crop)
    h, w = a.shape; s = max(h, w); c = np.zeros((s, s), np.uint8); c[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = a
    return Image.fromarray(np.ascontiguousarray(c))
IMGS = [load(r) for _, r in hip.iterrows()]
ROI_IMGS = None
if args.roi_file:
    if args.crop or args.resume_from: raise ValueError('ROI comparison does not support crop/resume')
    roi_path = Path(args.roi_file)
    boxes = {r['image_uid']: r['box'] for r in json.loads(roi_path.read_text())}
    ROI_IMGS = []
    for r in hip.itertuples():
        a = pydicom.dcmread(ROOT / 'data/interim/train/Исследования' / r.path).pixel_array
        if r.region == 'hip_left': a = np.ascontiguousarray(a[:, ::-1])
        x0,y0,x1,y1 = boxes[r.image_uid]
        if not (0 <= x0 < x1 <= a.shape[1] and 0 <= y0 < y1 <= a.shape[0]): raise ValueError('Invalid ROI')
        a = a[y0:y1,x0:x1]; h,w = a.shape; s = max(h,w)
        c = np.zeros((s,s),np.uint8); c[(s-h)//2:(s-h)//2+h,(s-w)//2:(s-w)//2+w] = a
        ROI_IMGS.append(Image.fromarray(c))
    provenance = json.loads((OUT / 'provenance.json').read_text())
    provenance['roi_sha256'] = hashlib.sha256(roi_path.read_bytes()).hexdigest()
    (OUT / 'provenance.json').write_text(json.dumps(provenance,indent=2))
Y = hip[TARGETS].values.astype(np.float32)

T = torchvision.transforms
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = T.Compose([T.Resize((args.res, args.res)), T.RandomAffine(degrees=8, translate=(0.12, 0.12) if args.crop else (0.06, 0.06), scale=(0.85, 1.15) if args.crop else (0.9, 1.1), fill=0),
                      T.ColorJitter(brightness=0.25, contrast=0.25), T.Grayscale(3), T.ToTensor(), norm, T.RandomErasing(p=0.3, scale=(0.01, 0.05))])
test_tf = T.Compose([T.Resize((args.res, args.res)), T.Grayscale(3), T.ToTensor(), norm])

class DS(torch.utils.data.Dataset):
    def __init__(self, idx, tf): self.idx, self.tf = idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; x = self.tf(IMGS[j])
        if ROI_IMGS is not None: x = torch.cat([x,self.tf(ROI_IMGS[j])],dim=0)
        return x, torch.from_numpy(Y[j])

def build():
    from dxaqc.hip_backbones import build_hip_model
    m = build_hip_model(args.arch, outputs=len(TARGETS), pretrained=True)
    if args.init_encoder:
        from dxaqc.hip_backbones import load_encoder
        load_encoder(m, args.init_encoder)
    return m.to(DEV)

def rest():
    if args.batch_pause:
        if DEV == 'mps': torch.mps.synchronize()
        elif DEV == 'cuda': torch.cuda.synchronize()
        time.sleep(args.batch_pause)

def forward_views(m,x):
    if x.shape[1] == 6: return (m(x[:,:3]) + m(x[:,3:])) / 2
    return m(x)

@torch.no_grad()
def predict(m, idx):
    m.eval(); out = []
    for x, _ in torch.utils.data.DataLoader(DS(idx, test_tf), batch_size=32, num_workers=args.workers):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=AMP):
            logits = forward_views(m, x.to(DEV))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
        rest()
    return np.concatenate(out)

oof = np.full((len(hip), len(TARGETS)), np.nan); hist = {}
completed = set()
if args.resume_from:
    import shutil
    source = Path(args.resume_from)
    config = json.loads((source / 'config.json').read_text())
    for key in ['arch', 'res', 'epochs', 'seed', 'lr', 'bs', 'crop']:
        if config[key] != getattr(args, key): raise ValueError(f'Resume mismatch: {key}')
    old = json.loads((source / 'provenance.json').read_text())
    current = json.loads((OUT / 'provenance.json').read_text())
    if old['tables'] != current['tables']: raise ValueError('Resume table hash mismatch')
    partial = pd.read_csv(source / 'oof_partial.csv')
    if not partial[['study','image_uid','fold']].equals(hip[['study','image_uid','fold']]):
        raise ValueError('Resume row order mismatch')
    history = json.loads((source / 'history.json').read_text())
    for k in sorted(hip.fold.unique()):
        mask = hip.fold == k
        values = partial.loc[mask, ['p_' + t for t in TARGETS]].values
        if (source / f'fold{k}.pt').exists() and np.isfinite(values).all():
            oof[mask] = values
            hist[int(k)] = history[str(k)]
            shutil.copy2(source / f'fold{k}.pt', OUT / f'fold{k}.pt')
            completed.add(int(k))
    current['resumed_from'] = dict(path=str(source), provenance=old, completed_folds=sorted(completed),
        limitation='Incomplete folds restarted with seed + fold; original RNG state was not saved')
    (OUT / 'provenance.json').write_text(json.dumps(current, indent=2))
    print('Recovered completed folds:', sorted(completed), flush=True)
for k in [int(f) for f in args.folds.split(",")]:
    if k in completed: continue
    torch.manual_seed(args.seed + k); np.random.seed(args.seed + k)
    tr = np.where(hip.fold != k)[0]; te = np.where(hip.fold == k)[0]
    pos = np.nan_to_num(Y[tr]).sum(0); neg = len(tr) - pos; pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 10), dtype=torch.float32).to(DEV)
    m = build(); opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * ((len(tr) + args.bs - 1) // args.bs), pct_start=0.15)
    dl = torch.utils.data.DataLoader(DS(tr, train_tf), batch_size=args.bs, shuffle=True, drop_last=False,
                                     num_workers=args.workers, persistent_workers=args.workers > 0)
    t0 = time.time(); hist[k] = []
    for ep in range(args.epochs):
        m.train(); tot = 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV); mask = ~torch.isnan(y)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=AMP):
                logits = forward_views(m, x).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], torch.nan_to_num(y)[mask], pos_weight=pw.expand_as(y)[mask])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(x)
            rest()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            p = predict(m, te); aucs = {}
            for j, t in enumerate(TARGETS):
                yy = Y[te, j]; mm = ~np.isnan(yy)
                aucs[t] = round(roc_auc_score(yy[mm], p[mm, j]), 3) if len(set(yy[mm])) > 1 else float("nan")
            hist[k].append({"epoch": ep + 1, "loss": round(tot / len(tr), 4), **aucs})
            print(f"fold {k} ep {ep+1} loss {tot/len(tr):.4f} val {aucs} ({time.time()-t0:.0f}s)", flush=True)
    oof[te] = predict(m, te)
    torch.save(m.state_dict(), OUT / f"fold{k}.pt")
    partial = hip[['study', 'image_uid', 'region', 'fold'] + TARGETS].copy()
    for j, target in enumerate(TARGETS): partial['p_' + target] = oof[:, j]
    partial.to_csv(OUT / 'oof_partial.csv', index=False)
    (OUT / 'history.json').write_text(json.dumps(hist, indent=2))

res = hip[["study", "image_uid", "region", "fold"] + TARGETS].copy()
for j, t in enumerate(TARGETS): res["p_" + t] = oof[:, j]
res.to_csv(OUT / "oof.csv", index=False); (OUT / "history.json").write_text(json.dumps(hist, indent=1)); (OUT / "config.json").write_text(json.dumps(vars(args), default=str))
done = res.p_quality.notna()
rep = [f"# hip CNN {args.tag}: arch {args.arch}, res {args.res}, epochs {args.epochs}, lr {args.lr}, seed {args.seed}\n"]
provenance = json.loads((OUT / 'provenance.json').read_text())
rep += [f"Commit: {provenance['commit']}. Split: {provenance['split_status']}.",
        "F1@best is optimistic: threshold selected on the same OOF labels. Use ROC/PR-AUC for this architecture comparison.",
        "Local reconstructed folds are not confirmed identical to the team folds. Prior holdout is reused in CV; no independent test claim.\n"]
r = {"бедро quality @0.5": bootstrap_ci(res.quality[done].astype(int).values, res.p_quality[done].values, res.study[done].values, 0.5)}
t = best_threshold(res.quality[done].astype(int).values, res.p_quality[done].values)
r[f"бедро quality @best={t:.2f}*"] = bootstrap_ci(res.quality[done].astype(int).values, res.p_quality[done].values, res.study[done].values, t)
rep += [format_table(r), ""]
lines = ["| класс | n | pos | ROC-AUC | PR-AUC | F1@best |", "|---|---|---|---|---|---|"]
for tname in TARGETS:
    mm = done & res[tname].notna(); y = res[tname][mm].astype(int).values; p = res["p_" + tname][mm].values
    b = binary_metrics(y, p, 0.5); tb = best_threshold(y, p); bb = binary_metrics(y, p, tb)
    lines.append(f"| {tname} | {b['n']} | {b['pos']} | {b['roc_auc']:.3f} | {b['pr_auc']:.3f} | {bb['f1']:.3f} (t={tb:.2f}) |")
rep += lines; (OUT / "metrics.md").write_text("\n".join(rep), encoding="utf-8"); print("\n".join(rep))
