"""Фиксированные фолды: группировка по исследованию, стратификация по наличию нарушения в исследовании."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5
SEED = 42


def make_folds(image_labels: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    df = image_labels.copy()
    study_y = df.groupby("study")["quality"].max().fillna(0).astype(int)
    df["study_y"] = df["study"].map(study_y)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df["study_y"], groups=df["study"])):
        df.loc[df.index[te], "fold"] = k
    return df[["study", "image_uid", "fold"]]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    il = pd.read_csv(root / "data/interim/image_labels.csv")
    f = make_folds(il)
    f.to_csv(root / "data/interim/folds.csv", index=False)
    m = il.merge(f, on=["study", "image_uid"])
    print(m.groupby("fold").agg(studies=("study", "nunique"), images=("image_uid", "size"), pos=("quality", lambda s: int((s == 1).sum()))))
