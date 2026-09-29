"""Architecture compatibility and offline DenseNet checkpoint integrity."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from dxaqc.hip_backbones import build_hip_model
from dxaqc.hip_cnn import HipCNNPredictor

ROOT = Path(__file__).resolve().parents[1]


class HipBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_architectures_have_three_outputs(self):
        for arch in ('convnext_tiny', 'resnet50', 'densenet121'):
            with self.subTest(arch=arch), torch.no_grad():
                model = build_hip_model(arch).eval()
                self.assertEqual(model(torch.zeros(1, 3, 64, 64)).shape, (1, 3))

    def test_unknown_architecture_rejected(self):
        with self.assertRaises(ValueError):
            build_hip_model('unknown')

    def test_bad_checkpoint_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoints/densenet121-a639ec97.pth'
            path.parent.mkdir()
            path.write_bytes(b'corrupt')
            with patch('torch.hub.get_dir', return_value=tmp), patch('torch.load', side_effect=AssertionError):
                with self.assertRaises(ValueError):
                    build_hip_model('densenet121', pretrained=True)

    def test_missing_checkpoint_does_not_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('torch.hub.get_dir', return_value=tmp), patch('torch.hub.download_url_to_file', side_effect=AssertionError):
                with self.assertRaises(FileNotFoundError):
                    build_hip_model('densenet121', pretrained=True)

    def test_official_weights_train_and_inference_roundtrip(self):
        hub = ROOT / 'models/torch_home/hub'
        if not (hub / 'checkpoints/densenet121-a639ec97.pth').exists():
            self.skipTest('Install official weights to run integration test')
        with patch('torch.hub.get_dir', return_value=str(hub)):
            model = build_hip_model('densenet121', pretrained=True)
        torch.manual_seed(42)
        x = torch.rand(2, 3, 64, 64)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), torch.zeros(2, 3))
        loss.backward()
        self.assertGreater(float(model.features.conv0.weight.grad.abs().sum()), 0)
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            torch.save(model.state_dict(), d / 'fold0.pt')
            predictor = HipCNNPredictor.__new__(HipCNNPredictor)
            predictor.runs = [(d, {'arch': 'densenet121', 'res': 64})]
            predictor.device = 'cpu'
            predictor._models = None
            restored, _, _ = predictor._load()[0]
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x))


if __name__ == '__main__':
    unittest.main()
