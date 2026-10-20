"""The isolated evaluation must not read the labels of the fold it scores."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


@unittest.skipUnless((ROOT / 'outputs/release_loss_v1/features.npy').exists(),
                     'needs the local validation artefacts')
class IsolatedFolds(unittest.TestCase):
    def test_outer_test_labels_do_not_influence_scores(self):
        import evaluate_final_isolated as EI
        base = EI.evaluate()
        perm = EI.evaluate(permute_test=True, seed=7)
        self.assertEqual(list(base.image_uid), list(perm.image_uid))
        for col in [c for c in base.columns if c.startswith('score_')]:
            np.testing.assert_allclose(base[col].values, perm[col].values, equal_nan=True,
                                       err_msg=f'{col} moved when outer-test labels were shuffled')
        self.assertTrue((base.quality_class.values == perm.quality_class.values).all())


if __name__ == '__main__':
    unittest.main()
