"""Бедро: дообученный convnext_tiny (мультилейбл quality / hip_positioning / hip_roi_margins), ансамбль фолдов.
Веса: models/hip_cnn/<tag>/fold*.pt + config.json + thresholds.json (scripts/export_hip_cnn.py)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .predictor import HIP_POSITION, HIP_ROI, MODELS_DIR, BasePredictor, Prediction

TARGETS = ["quality", "v_rotation", "v_roi"]
CODE_OF = {"v_rotation": HIP_POSITION, "v_roi": HIP_ROI}


class HipCNNPredictor(BasePredictor):
    name = "hip_cnn"

    def __init__(self, tags: str | list[str] | None = None, device: str | None = None):
        """tags: список прогонов (models/hip_cnn/<tag>/); None → models/hip_cnn/ensemble.json."""
        import torch
        root = MODELS_DIR / "hip_cnn"
        if tags is None:
            ens = json.loads((root / "ensemble.json").read_text()); tags = ens["tags"]; self.thr = ens["thresholds"]
        else:
            tags = [tags] if isinstance(tags, str) else list(tags)
            self.thr = json.loads((root / tags[0] / "thresholds.json").read_text())
        self.runs = []
        for t in tags:
            d = root / t
            if not (d / "config.json").exists():
                raise FileNotFoundError(f"нет прогона {d}")
            self.runs.append((d, json.loads((d / "config.json").read_text())))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models = None

    def _load(self):
        """[(model, res), ...] по всем фолдам всех прогонов."""
        if self._models is not None:
            return self._models
        import torch, torchvision
        os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch_home"))
        models = []
        for d, cfg in self.runs:
            for pt in sorted(d.glob("fold*.pt")):
                from .hip_backbones import build_hip_model
                m = build_hip_model(cfg.get("arch", "convnext_tiny"), outputs=len(TARGETS))
                m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
                models.append((m.eval().to(self.device), int(cfg.get("res", 384)), int(cfg.get("crop", 0) or 0)))
        if not models:
            raise FileNotFoundError("нет весов fold*.pt")
        self._models = models
        return models

    def _tensor(self, pixels: np.ndarray, region: str, res: int, crop: int = 0):
        import torch
        from PIL import Image
        a = np.ascontiguousarray(pixels[:, ::-1]) if region == "hip_left" else pixels
        if crop:
            from .hip_rules import lt_crop
            a, _ = lt_crop(a, size=crop)
        h, w = a.shape; s = max(h, w)
        c = np.zeros((s, s), np.uint8); c[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
        a = np.asarray(Image.fromarray(np.ascontiguousarray(c)).resize((res, res), Image.BILINEAR)).astype(np.float32) / 255.0
        x = torch.from_numpy(a)[None].repeat(3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return ((x - mean) / std)[None].to(self.device)

    def predict(self, pixels: np.ndarray, region: str) -> Prediction:
        if not region.startswith("hip"):
            return Prediction(0, details={"note": "no model for region"})
        import torch
        models = self._load(); cache = {}
        with torch.no_grad():
            ps = []
            for m, res, crop in models:
                key = (res, crop)
                if key not in cache:
                    cache[key] = self._tensor(pixels, region, res, crop)
                ps.append(torch.sigmoid(m(cache[key])).cpu().numpy()[0])
            # усреднение: сначала по фолдам внутри прогона, потом по прогонам — здесь все прогоны по 5 фолдов, среднее эквивалентно
            p = np.mean(ps, axis=0)
        probs = dict(zip(TARGETS, map(float, p)))
        viol = [CODE_OF[t] for t in ("v_rotation", "v_roi") if probs[t] >= self.thr[t]]
        q = int(probs["quality"] >= self.thr["quality"] or bool(viol))
        if q and not viol:  # контракт: quality_class=1 ⇒ хотя бы один код; берём самый вероятный
            viol = [CODE_OF[max(("v_rotation", "v_roi"), key=lambda t: probs[t] / max(self.thr[t], 1e-6))]]
        return Prediction(q, viol, {HIP_POSITION: probs["v_rotation"], HIP_ROI: probs["v_roi"], "hip_quality": probs["quality"]},
                          {"p_quality": round(probs["quality"], 3), "n_models": len(models)})
