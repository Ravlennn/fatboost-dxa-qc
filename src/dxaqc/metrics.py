"""Метрики по ТЗ п. 8.4 с бутстрап 95% ДИ (ресемплинг по исследованиям)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    out = {
        "n": int(len(y)), "pos": int(y.sum()),
        "sensitivity": recall_score(y, yhat, zero_division=0),
        "specificity": recall_score(1 - y, 1 - yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, yhat) if len(set(y)) > 1 else float("nan"),
        "f1": f1_score(y, yhat, zero_division=0),
    }
    if len(set(y)) > 1:
        out["roc_auc"] = roc_auc_score(y, p); out["pr_auc"] = average_precision_score(y, p)
    else:
        out["roc_auc"] = float("nan"); out["pr_auc"] = float("nan")
    return out


def bootstrap_ci(y, p, groups, thr: float = 0.5, n_boot: int = 1000, seed: int = 0,
                 keys=("sensitivity", "specificity", "f1", "balanced_acc", "roc_auc", "pr_auc")) -> dict:
    """Точечная оценка + 95% ДИ. Ресемплинг по группам (исследованиям), чтобы не разрывать пары."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float); groups = np.asarray(groups)
    point = binary_metrics(y, p, thr)
    ug = np.unique(groups); idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        gs = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(set(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], p[idx], thr)
        for k in keys:
            samples[k].append(m[k])
    out = {}
    for k in keys:
        s = np.array([v for v in samples[k] if not np.isnan(v)])
        lo, hi = (np.percentile(s, [2.5, 97.5]) if len(s) else (np.nan, np.nan))
        out[k] = {"value": point[k], "ci_low": float(lo), "ci_high": float(hi)}
    out["n"] = point["n"]; out["pos"] = point["pos"]; out["threshold"] = thr
    return out


def best_threshold(y, p, grid=None) -> float:
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def format_table(results: dict[str, dict]) -> str:
    """results: {название среза: bootstrap_ci(...)} → markdown-таблица."""
    keys = ["sensitivity", "specificity", "balanced_acc", "f1", "roc_auc", "pr_auc"]
    lines = ["| срез | n | pos | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    for name, r in results.items():
        cells = [f"{r[k]['value']:.3f} [{r[k]['ci_low']:.2f}–{r[k]['ci_high']:.2f}]" if not np.isnan(r[k]["value"]) else "—" for k in keys]
        lines.append(f"| {name} | {r['n']} | {r['pos']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def macro_f1_multilabel(Y: pd.DataFrame, P: pd.DataFrame, thr: dict[str, float]) -> dict:
    """Y, P: колонки = коды нарушений. Возвращает f1 по классам и macro."""
    per = {}
    for c in Y.columns:
        m = Y[c].notna()
        if m.sum() == 0 or Y.loc[m, c].sum() == 0:
            continue
        per[c] = f1_score(Y.loc[m, c].astype(int), (P.loc[m, c] >= thr.get(c, 0.5)).astype(int), zero_division=0)
    per["macro_f1"] = float(np.mean(list(per.values()))) if per else float("nan")
    return per
