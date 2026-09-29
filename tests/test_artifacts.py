import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from dxaqc.artifacts import detect, head_features  # noqa: E402
from dxaqc.landmarks import apply_head  # noqa: E402

PTS = {'L1_top': [150, 60], 'L1L2': [150, 110], 'L2L3': [150, 160], 'L3L4': [150, 210], 'L4_bottom': [150, 260]}


def spine(wire=False, hook=False, rib=False):
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(30, 6, (315, 300)), 0, 255).astype(np.float32)
    img[:, 115:185] = 170  # vertebral column
    if rib:  # broad oblique rib: must not count as a wire
        cv2.line(img, (40, 20), (110, 90), 90, 9)
    if wire:  # thin faint shallow arc in the upper third, outside the column (as in our scans)
        t = np.linspace(-1, 1, 80)
        pts = np.stack([60 + t * 45, 40 + 12 * (1 - t ** 2)], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, 110, 2)
    if hook:
        img[150:156, 240:248] = 255
    return np.clip(img, 0, 255).astype(np.uint8)


class Artifacts(unittest.TestCase):
    def test_clean_image_has_no_artifacts(self):
        r = detect(spine(), PTS)
        self.assertEqual(r['n_wires'], 0)
        self.assertEqual(r['n_metal'], 0)

    def test_thin_wire_detected_broad_rib_ignored(self):
        self.assertGreater(detect(spine(wire=True), PTS)['wire_len_vert'], 0.5)
        self.assertEqual(detect(spine(rib=True), PTS)['n_wires'], 0)

    def test_metal_hook_detected(self):
        r = detect(spine(hook=True), PTS)
        self.assertGreater(r['metal_area_mm2'], 5)
        self.assertEqual(r['mask'].shape, (315, 300))

    def test_bundle_artifact_head_features(self):
        path = ROOT / 'models/release/landmarks/heads.json'
        spec = json.loads(path.read_text()) if path.is_file() else {}
        if 'spine_artifact_head' not in spec:
            self.skipTest('artifact head not exported')
        feats = {**head_features(detect(spine(wire=True), PTS)), 'cnn': 0.0}
        head = spec['spine_artifact_head']
        self.assertFalse(set(head['features']) - feats.keys())
        self.assertTrue(0 <= apply_head(head, feats) <= 1)
        self.assertIn(spec['spine_quality_policy'], ('or', 'reasons'))


if __name__ == '__main__':
    unittest.main()
