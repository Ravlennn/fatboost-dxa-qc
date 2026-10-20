"""D — RAD-DINO fine-tuned with LoRA for the hip (docs/PREREG_NEW_MODELS.md).

RAD-DINO (microsoft/rad-dino, DINOv2 ViT-B/14 on ~840k radiographs, MIT) is the only large encoder
that moved anything here, and only frozen. This arm adapts it: rank-8 LoRA on the attention
projections of every block plus a linear head on [CLS, mean patch token]; the backbone weights stay
frozen (~1 % of the parameters train). Same rows, mirroring and augmentation as the hip CNN, with
RAD-DINO's own normalisation; 336 px (24×24 patches, above the ~280 px native frame). Outputs: hip quality, rotation, ROI margins.

Weights: models/foundation/rad_dino_backbone.safetensors = backbone_compatible.safetensors of
microsoft/rad-dino at 110cbc18, SHA-256 checked. Code: the official DINOv2 module pinned in
outputs/dinov2_probe_v1/vendor (no new dependencies).

  python scripts/train_hip_raddino_lora.py --tag raddino_base --device mps
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pydicom
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import qc_cnn_common as C  # noqa: E402

TARGETS = ['quality', 'v_rotation', 'v_roi']
WEIGHTS = C.ROOT / 'models/foundation/rad_dino_backbone.safetensors'
SHA256 = '1eac0464b2a00d368aa3eea1dc029964b10320fbabc59a8a4e768c43a23d26f4'
VENDOR = C.ROOT / 'outputs/dinov2_probe_v1/vendor/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8'
MEAN, STD = [0.5307] * 3, [0.2583] * 3   # RAD-DINO preprocessor
RANK, ALPHA = 8, 16
HEAD_LR_FACTOR = 10.0


def read_safetensors(path: Path) -> dict:
    """Minimal safetensors reader (8-byte header size, JSON header, raw little-endian float32)."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for k, v in header.items():
        if k == '__metadata__':
            continue
        if v['dtype'] != 'F32':
            raise ValueError(f'unsupported dtype {v["dtype"]}')
        a, b = v['data_offsets']
        out[k] = torch.from_numpy(np.frombuffer(blob[a:b], dtype=np.float32).reshape(v['shape']).copy())
    return out


class LoRALinear(torch.nn.Module):
    """y = W x + (B A x) * alpha / r, with W frozen and B initialised to zero (starts as the base model)."""

    def __init__(self, base: torch.nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = torch.nn.Parameter(torch.empty(r, base.in_features))
        self.b = torch.nn.Parameter(torch.zeros(base.out_features, r))
        torch.nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.a.t() @ self.b.t()) * self.scale


class RadDinoLoRA(torch.nn.Module):
    def __init__(self, outputs: int):
        super().__init__()
        if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != SHA256:
            raise ValueError(f'{WEIGHTS.name}: checksum mismatch')
        os.environ['XFORMERS_DISABLED'] = '1'
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from dinov2.models.vision_transformer import vit_base
        self.backbone = vit_base(img_size=518, patch_size=14, init_values=1., ffn_layer='mlp', block_chunks=0,
                                 num_register_tokens=0, interpolate_antialias=False, interpolate_offset=.1)
        self.backbone.load_state_dict(read_safetensors(WEIGHTS), strict=True)
        self.backbone.requires_grad_(False)
        for blk in self.backbone.blocks:
            blk.attn.qkv = LoRALinear(blk.attn.qkv, RANK, ALPHA)
            blk.attn.proj = LoRALinear(blk.attn.proj, RANK, ALPHA)
        self.head = torch.nn.Linear(2 * self.backbone.embed_dim, outputs)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        z = torch.cat([f['x_norm_clstoken'], f['x_norm_patchtokens'].mean(1)], dim=1)
        return self.head(z)

    def param_groups(self, lr: float):
        lora = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith('head.')]
        return [{'params': lora, 'lr': lr}, {'params': list(self.head.parameters()), 'lr': lr * HEAD_LR_FACTOR}]


def main():
    args = C.parser(__doc__.splitlines()[0], res=336, lr=2e-4, bs=8).parse_args()
    if args.res % 14:
        raise ValueError('res must be a multiple of the 14 px patch')
    out, dev = C.setup(args, C.ROOT / 'outputs/raddino_lora')
    rows = C.labelled()
    rows = rows[rows.region != 'spine'].reset_index(drop=True)
    images = []
    for r in rows.itertuples():
        a = pydicom.dcmread(str(C.DATA / r.path)).pixel_array
        if r.region == 'hip_left':
            a = np.ascontiguousarray(a[:, ::-1])
        images.append(C.square(a))
    print('hips', len(rows), 'device', dev, 'res', args.res, flush=True)
    train_tf, test_tf = C.hip_full(args.res, MEAN, STD), C.plain(args.res, MEAN, STD)
    C.run(args, out, dev, rows, rows[TARGETS].values, TARGETS, images, train_tf, test_tf, lambda: RadDinoLoRA(len(TARGETS)))


if __name__ == '__main__':
    main()
