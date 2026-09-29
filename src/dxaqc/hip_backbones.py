"""Shared training/inference constructors for the hip CNN architectures."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch
from torchvision import models


def build_hip_model(arch: str, outputs: int = 3, pretrained: bool = False):
    if arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = torch.nn.Linear(768, outputs)
    elif arch in ("convnext_small", "convnext_base"):
        builder, w, dim = {"convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights, 768),
                           "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights, 1024)}[arch]
        model = builder(weights=w.IMAGENET1K_V1 if pretrained else None)
        model.classifier[2] = torch.nn.Linear(dim, outputs)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Linear(2048, outputs)
    elif arch == "densenet121":
        model = models.densenet121(weights=None)
        if pretrained:
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints/densenet121-a639ec97.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Download official DenseNet121 weights to {checkpoint}")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != "a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395":
                raise ValueError("DenseNet121 ImageNet checkpoint SHA-256 mismatch")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.([12]\.(?:weight|bias|running_mean|running_var))$")
            for key in list(state):
                match = pattern.match(key)
                if match:
                    state[match.group(1) + match.group(2)] = state.pop(key)
            model.load_state_dict(state)
        model.classifier = torch.nn.Linear(1024, outputs)
    else:
        raise ValueError(f"Unsupported hip architecture: {arch}")
    return model


def load_encoder(model, path: str | Path) -> dict:
    """Initialise ``model.features`` from a self-supervised checkpoint
    (scripts/train_ssl_dxa.py saves {'arch', 'features': state_dict})."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.features.load_state_dict(ck["features"], strict=True)
    return {"arch": ck.get("arch"), "epoch": ck.get("epoch"), "source": str(path)}
