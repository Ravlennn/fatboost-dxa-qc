"""Paired study bootstrap for the prespecified full vs full+ROI comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]


def main():
    tags=['cgmh_control_full_s0_v1','cgmh_full_plus_roi_s0_v1']
    tables=[pd.read_csv(ROOT/'outputs/hip_cnn'/t/'oof.csv').sort_values(['study','image_uid']).reset_index(drop=True) for t in tags]
    a,b=tables
    if not a[['study','image_uid','fold','quality']].equals(b[['study','image_uid','fold','quality']]):
        raise ValueError('Comparison cohorts differ')
    if not np.isfinite(a.p_quality).all() or not np.isfinite(b.p_quality).all(): raise ValueError('Incomplete OOF')
    groups=a.study.unique(); indices=[np.flatnonzero(a.study.values==g) for g in groups]
    rng=np.random.default_rng(42); samples=[np.concatenate([indices[j] for j in rng.integers(0,len(groups),len(groups))]) for _ in range(1000)]
    rows=[]
    for name,fn in [('ROC-AUC',roc_auc_score),('PR-AUC (AP)',average_precision_score)]:
        va=float(fn(a.quality,a.p_quality)); vb=float(fn(b.quality,b.p_quality)); delta=[]
        for ix in samples:
            y=a.quality.values[ix]
            if len(np.unique(y))<2: continue
            delta.append(fn(y,b.p_quality.values[ix])-fn(y,a.p_quality.values[ix]))
        lo,hi=np.quantile(delta,[.025,.975]); rows.append(f'| {name} | {va:.4f} | {vb:.4f} | {vb-va:+.4f} [{lo:+.4f}, {hi:+.4f}] |')
    text='\n'.join(['# CGMH ROI: paired DXA comparison','',
        'Seed 0, 5 local reconstructed folds, 30 epochs, ConvNeXt 384. Segmenter trained only on CGMH.',
        '95% paired bootstrap CI by organizer study, 1000 draws, bootstrap seed 42. One training seed; training variance is not covered.',
        'Full+ROI averages two shared-network logits; extra computation and augmentation differ from single full-frame.',
        'No independent test or clinical-performance claim.','',
        '| Metric | Full | Full+ROI | Delta [95% CI] |','|---|---|---|---|',*rows])
    (ROOT/'outputs/cgmh_pipeline_v1/comparison.md').write_text(text)
    with (ROOT/'docs/EXPERIMENTS.md').open('a') as f: f.write('\n'+text+'\n')
    print(text)

if __name__=='__main__': main()
