"""Smoke tests for the nested-hip evaluators on synthetic CNN runs (no GPU needed).

Catches wiring bugs before the 25-model job: every head must read its own CNN column, and
every head must produce finite scores for all 150 hips.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
OOF = ROOT / 'outputs/release_validation_v1/assembled_oof.csv'
COLS = ['study', 'image_uid', 'region', 'fold', 'quality', 'v_rotation', 'v_roi']


@unittest.skipUnless(OOF.exists(), 'needs the local validation artefacts')
class HipNestedSmoke(unittest.TestCase):
    def make_runs(self, tmp):
        """Fake nested runs with a weak but real signal, so the second level has something to fit."""
        hip = pd.read_csv(OOF)
        hip = hip[hip.region.str.startswith('hip') & hip.quality.notna()][COLS].reset_index(drop=True)
        rng = np.random.default_rng(0)
        nested, out = Path(tmp) / 'hip_cnn', Path(tmp) / 'val'
        out.mkdir(parents=True)
        folds = sorted(hip.fold.unique())
        for name, rows in [(f'nested_f{f}_j{j}', hip[hip.fold == j])
                           for f in folds for j in folds if f != j] + [('nested_base', hip)]:
            d = rows.copy()
            for t in ('quality', 'v_rotation', 'v_roi'):
                d['p_' + t] = np.clip(0.3 * d[t].fillna(0) + rng.uniform(0.1, 0.6, len(d)), 0.01, 0.99)
            (nested / name).mkdir(parents=True)
            d.to_csv(nested / name / 'oof.csv', index=False)
        return hip, nested, out

    def test_all_heads_scored(self):
        import eval_hip_nested as EH
        with tempfile.TemporaryDirectory() as tmp:
            hip, nested, out = self.make_runs(tmp)
            with unittest.mock.patch.object(EH, 'NESTED', nested), unittest.mock.patch.object(EH, 'OUT', out):
                self.assertEqual(EH.main([]), 0)
            t = pd.read_csv(out / 'decisions_hip_nested.csv')
        self.assertEqual(len(t), len(hip))
        for col in ('score_quality', 'score_hip_positioning', 'score_hip_roi_margins'):
            self.assertTrue(np.isfinite(t[col]).all(), f'{col} has non-finite scores')


if __name__ == '__main__':
    unittest.main()
