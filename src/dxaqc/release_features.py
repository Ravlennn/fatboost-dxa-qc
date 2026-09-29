"""Offline feature extractor and portable linear heads shared by training and CLI."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models import densenet121

PREPROCESS = "grayscale uint8; symmetric zero square padding; PIL bilinear 224; RGB repeat; ImageNet normalize; no mirroring"
DENSENET_SHA256 = "a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395"


def image_tensor(pixels: np.ndarray) -> torch.Tensor:
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or min(pixels.shape) < 1 or max(pixels.shape) > 4096:
        raise ValueError("Expected nonempty 2D uint8 pixels")
    h, w = pixels.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    canvas[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = pixels
    resized = np.array(Image.fromarray(canvas).resize((224, 224), Image.Resampling.BILINEAR), copy=True)
    x = torch.from_numpy(resized).float()[None].repeat(3, 1, 1) / 255.0
    return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


class FrozenDenseNetEncoder:
    """Accepts the official local ImageNet checkpoint; never downloads weights."""

    def __init__(self, checkpoint: str | Path, threads: int = 2, device: str = "cpu"):
        if threads < 1:
            raise ValueError("threads must be positive")
        path = Path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != DENSENET_SHA256:
            raise ValueError("DenseNet121 checkpoint checksum mismatch")
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        state = torch.load(path, map_location="cpu", weights_only=True)
        pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.([12]\.(?:weight|bias|running_mean|running_var))$")
        for key in list(state):
            match = pattern.match(key)
            if match:
                state[match.group(1) + match.group(2)] = state.pop(key)
        self.model = densenet121(weights=None)
        self.model.load_state_dict(state)
        self.model.classifier = torch.nn.Identity()
        self.model.requires_grad_(False).eval().to(self.device)
        self.checkpoint_sha256 = digest

    @torch.inference_mode()
    def embed(self, pixels: np.ndarray | list[np.ndarray], batch_size: int = 4) -> np.ndarray:
        """Return (N,1024), including N=1 when a single image is supplied."""
        images = [pixels] if isinstance(pixels, np.ndarray) and pixels.ndim == 2 else list(pixels)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not images:
            return np.empty((0, 1024), dtype=np.float32)
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([image_tensor(a) for a in images[start:start+batch_size]])
            chunks.append(self.model(batch.to(self.device)).cpu().numpy())
        return np.concatenate(chunks)


def predict_linear_head(head: dict, features: np.ndarray) -> np.ndarray:
    """Return class probabilities (N,K), with K in JSON ``classes`` order."""
    x = np.atleast_2d(np.asarray(features, dtype=np.float64))
    if x.shape[1] != len(head["mean"]) or not np.isfinite(x).all():
        raise ValueError("Invalid features for linear head")
    if "constant_probabilities" in head:
        return np.tile(np.asarray(head["constant_probabilities"]), (len(x), 1))
    scale = np.asarray(head["scale"], dtype=np.float64)
    if np.any(scale <= 0):
        raise ValueError("Invalid scaler")
    z = ((x - np.asarray(head["mean"])) / scale) @ np.asarray(head["coef"]).T + np.asarray(head["intercept"])
    if len(head["classes"]) == 2 and z.shape[1] == 1:
        p = np.exp(-np.logaddexp(0., -z[:, 0]))
        return np.column_stack([1 - p, p])
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)
