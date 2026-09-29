"""Бейзлайн: замороженные ImageNet-эмбеддинги + логистическая регрессия, OOF по 5 фолдам.
Запуск: TORCH_HOME=models/torch_home .venv/bin/python scripts/baseline_embed.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch_home"))
warnings.filterwarnings("ignore")
from dxaqc.metrics import best_threshold, binary_metrics, bootstrap_ci, format_table  # noqa: E402

OUT = ROOT / "outputs/baseline"; OUT.mkdir(parents=True, exist_ok=True)
IMG = 224
SPINE_CLASSES = ["v_layout", "v_axis", "v_artifact"]
HIP_CLASSES = ["v_rotation", "v_roi"]
torch.manual_seed(0); np.random.seed(0)
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_pixels(rel: str) -> np.ndarray:
    import pydicom
    return pydicom.dcmread(str(ROOT / "data/interim/train/Исследования" / rel)).pixel_array


def letterbox(a: np.ndarray, size: int = IMG) -> np.ndarray:
    h, w = a.shape; s = max(h, w)
    canvas = np.zeros((s, s), dtype=np.uint8); y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = a
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.BILINEAR))


def to_tensor(a: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(a.astype(np.float32) / 255.0)[None].repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (x - mean) / std


def build(name: str):
    if name == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity()
    elif name == "convnext_tiny":
        m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1); m.classifier[2] = torch.nn.Identity()
    else:
        raise ValueError(name)
    return m.eval().to(DEV)


@torch.no_grad()
def embed(model, imgs: list[np.ndarray], bs: int = 32) -> np.ndarray:
    feats = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([to_tensor(a) for a in imgs[i:i + bs]]).to(DEV)
        feats.append(model(x).float().cpu().numpy())
    return np.concatenate(feats)


def oof_lr(X: np.ndarray, y: np.ndarray, folds: np.ndarray, C: float = 0.1) -> np.ndarray:
    p = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = ~np.isnan(y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
        clf.fit(X[tr & m], y[tr & m].astype(int))
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def main():
    il = pd.read_csv(ROOT / "data/interim/image_labels.csv").merge(pd.read_csv(ROOT / "data/interim/folds.csv"), on=["study", "image_uid"])
    il["is_spine"] = il.region == "spine"
    imgs = []
    for _, r in il.iterrows():
        a = load_pixels(r.path)
        if r.region == "hip_left":
            a = a[:, ::-1]  # все бёдра приводим к «правой» ориентации
        imgs.append(letterbox(a))
    print(f"images {len(imgs)} device {DEV}")

    report = ["# Бейзлайн: ImageNet-эмбеддинги + LogisticRegression (OOF, 5 фолдов по исследованиям)\n"]
    y_all = il.quality.values.astype(float); g_all = il.study.values; f_all = il.fold.values
    # тривиальные бейзлайны
    triv = {}
    m = ~np.isnan(y_all)
    triv["всегда 0"] = bootstrap_ci(y_all[m], np.zeros(m.sum()), g_all[m], thr=0.5, n_boot=300)
    rng = np.random.default_rng(0)
    triv["случайно"] = bootstrap_ci(y_all[m], rng.random(m.sum()), g_all[m], thr=0.5, n_boot=300)
    report += ["## Тривиальные бейзлайны (все изображения, бинарный класс)\n", format_table(triv), ""]

    summary = {}
    for bname in ["resnet50", "convnext_tiny"]:
        X = embed(build(bname), imgs); np.save(OUT / f"emb_{bname}.npy", X)
        print(bname, X.shape)
        P = pd.DataFrame(index=il.index)
        for region_mask, name, classes in [(il.is_spine.values, "spine", SPINE_CLASSES), (~il.is_spine.values, "hip", HIP_CLASSES)]:
            Xr, fr = X[region_mask], f_all[region_mask]
            P.loc[region_mask, "p_quality"] = oof_lr(Xr, il.loc[region_mask, "quality"].values.astype(float), fr)
            for c in classes:
                P.loc[region_mask, "p_" + c] = oof_lr(Xr, il.loc[region_mask, c].values.astype(float), fr)
        P["study"] = il.study; P["region"] = il.region; P["image_uid"] = il.image_uid; P["quality"] = il.quality; P["fold"] = il.fold
        for c in SPINE_CLASSES + HIP_CLASSES: P[c] = il[c]
        P.to_csv(OUT / f"oof_{bname}.csv", index=False)

        res = {}
        for name, mask in [("все", np.ones(len(il), bool)), ("позвоночник", il.is_spine.values), ("бедро", ~il.is_spine.values)]:
            mm = mask & ~np.isnan(y_all)
            y, p, g = y_all[mm], P.p_quality.values[mm], g_all[mm]
            res[f"{name} @0.5"] = bootstrap_ci(y, p, g, thr=0.5)
            t = best_threshold(y, p)
            res[f"{name} @best={t:.2f}*"] = bootstrap_ci(y, p, g, thr=t)
        report += [f"## {bname}: бинарный класс качества\n", format_table(res), "", "\\* порог подобран на тех же OOF → оптимистично.", ""]
        summary[bname] = {k: {m: round(v["value"], 3) for m, v in r.items() if isinstance(v, dict)} for k, r in res.items()}

        # по типам нарушений
        lines = ["| класс | n | pos | ROC-AUC | PR-AUC | F1@0.5 | F1@best |", "|---|---|---|---|---|---|---|"]
        f1s = []
        for c in SPINE_CLASSES + HIP_CLASSES:
            mm = P[c].notna().values & P["p_" + c].notna().values
            y, p = P.loc[mm, c].values.astype(int), P.loc[mm, "p_" + c].values
            b = binary_metrics(y, p, 0.5); t = best_threshold(y, p); bb = binary_metrics(y, p, t)
            f1s.append(bb["f1"])
            lines.append(f"| {c} | {b['n']} | {b['pos']} | {b['roc_auc']:.3f} | {b['pr_auc']:.3f} | {b['f1']:.3f} | {bb['f1']:.3f} (t={t:.2f}) |")
        lines.append(f"| **macro-F1** | | | | | | **{np.mean(f1s):.3f}** |")
        report += [f"## {bname}: типы нарушений (OOF)\n", *lines, ""]
        summary[bname]["macro_f1_best"] = round(float(np.mean(f1s)), 3)

    (OUT / "metrics.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
