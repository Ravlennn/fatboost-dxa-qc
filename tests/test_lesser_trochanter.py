import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from dxaqc.landmarks import apply_head  # noqa: E402
from dxaqc.lesser_trochanter import detect, head_features, median5  # noqa: E402

PTS = {'neck_center': [60, 50], 'neck_head_side': [80, 35], 'neck_troch_side': [40, 65]}


def synthetic_femur(bump=0):
    """Bright shaft cortex on the left of the medial edge, optional grey medial bump (LT)."""
    img = np.full((260, 160), 20, np.uint8)
    for y in range(40, 260):
        edge = 70 + (y - 40) // 20
        img[y, 40:edge] = 150
        img[y, edge - 6:edge] = 230  # cortex
    if bump:
        for y in range(110, 150):
            w = int(bump * np.sin(np.pi * (y - 110) / 40))
            edge = 70 + (y - 40) // 20
            img[y, edge:edge + w] = 110  # medium-density lesser trochanter
    return img


class LesserTrochanter(unittest.TestCase):
    def test_median5_matches_scipy(self):
        from scipy.ndimage import median_filter
        a = np.random.default_rng(0).normal(size=40)
        np.testing.assert_allclose(median5(a), median_filter(a, size=5, mode='nearest'))

    def test_bump_is_detected_and_smooth_contour_is_not(self):
        with_lt, err = detect(synthetic_femur(bump=12), PTS)
        smooth, err2 = detect(synthetic_femur(bump=0), PTS)
        self.assertIsNone(err)
        self.assertIsNone(err2)
        self.assertTrue(with_lt['visible'])
        self.assertFalse(smooth['visible'])
        self.assertGreater(with_lt['lt_prominence_mm'], smooth['lt_prominence_mm'] + 3)

    def test_mirrored_input_gives_mirrored_points(self):
        img = synthetic_femur(bump=12)
        w = img.shape[1]
        mp = {k: [w - 1 - x, y] for k, (x, y) in PTS.items()}
        a, _ = detect(img, PTS)
        b, _ = detect(img[:, ::-1].copy(), mp)
        self.assertAlmostEqual(a['lt_prominence_mm'], b['lt_prominence_mm'])
        self.assertAlmostEqual(a['points']['lt_apex'][0], w - 1 - b['points']['lt_apex'][0])

    def test_degenerate_image_returns_error_not_exception(self):
        res, err = detect(np.zeros((40, 40), np.uint8), PTS)
        self.assertIsNone(res)
        self.assertTrue(err)
        self.assertEqual(head_features(None)['lt_prom'], 0.0)

    def test_bundle_hip_heads_get_all_features(self):
        path = ROOT / 'models/release/landmarks/heads.json'
        if not path.is_file() or 'hip_heads' not in json.loads(path.read_text()):
            self.skipTest('hip heads not exported')
        res, _ = detect(synthetic_femur(bump=12), PTS)
        feats = {**head_features(res), 'cnn': 0.0}
        for code, head in json.loads(path.read_text())['hip_heads'].items():
            self.assertFalse(set(head['features']) - feats.keys(), code)
            self.assertTrue(0 <= apply_head(head, feats) <= 1)


if __name__ == '__main__':
    unittest.main()
