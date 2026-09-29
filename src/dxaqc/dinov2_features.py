"""Frozen DINOv2 probe with pinned local official code and weights.

No network calls, torch.hub entrypoint, dependency installation or finetuning.
The experiment downloader supplies the reviewed official files separately.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .release_features import image_tensor

COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
ARCHIVE_SHA256 = "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48"
CHECKPOINT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
CODE_SHA256 = "c7c4cf024d30057e99b3b7728a57bd4e56be41fe9ecb5d3dc5281f85a7787069"


def verify_code(code_dir: Path) -> str:
    files = [code_dir / "dinov2/__init__.py",
             *sorted((code_dir / "dinov2/layers").glob("*.py")),
             *sorted((code_dir / "dinov2/models").glob("*.py"))]
    if len(files) != 12:
        raise ValueError("Unexpected DINOv2 official source inventory")
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(code_dir).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    if digest.hexdigest() != CODE_SHA256:
        raise ValueError("DINOv2 official source checksum mismatch")
    return digest.hexdigest()


class FrozenDinoEncoder:
    def __init__(self, code_dir: Path, checkpoint: Path, threads: int = 2):
        if threads < 1:
            raise ValueError("threads must be positive")
        code_dir = Path(code_dir).resolve()
        verify_code(code_dir)
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("DINOv2 checkpoint checksum mismatch")
        # Force torch attention; do not require or load optional xformers kernels.
        os.environ["XFORMERS_DISABLED"] = "1"
        sys.path.insert(0, str(code_dir))
        from dinov2.models.vision_transformer import vit_small

        if Path(sys.modules[vit_small.__module__].__file__).resolve() != code_dir / "dinov2/models/vision_transformer.py":
            raise ValueError("An unverified DINOv2 module was already imported")
        torch.set_num_threads(threads)
        self.model = vit_small(img_size=518, patch_size=14, init_values=1., ffn_layer="mlp", block_chunks=0,
                               num_register_tokens=0, interpolate_antialias=False, interpolate_offset=.1)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def embed(self, pixels: list[np.ndarray]) -> np.ndarray:
        if not pixels:
            return np.empty((0, 384), dtype=np.float32)
        features = self.model(torch.stack([image_tensor(p) for p in pixels])).numpy()
        if features.shape != (len(pixels), 384) or not np.isfinite(features).all():
            raise ValueError("DINOv2 returned invalid features")
        return features
