"""Модели качества. Коды нарушений (violation_type), через ';' если несколько."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPINE_LAYOUT = "spine_scan_range"      # не видны гребни подвздошных / половина Th12
SPINE_AXIS = "spine_axis_tilt"         # наклон оси > допустимого
SPINE_ARTIFACT = "spine_artifact"      # посторонние предметы, артефакты
HIP_POSITION = "hip_positioning"       # позиционирование / ротация
HIP_ROI = "hip_roi_margins"            # недостаточные поля вокруг ROI
ALL_CODES = [SPINE_LAYOUT, SPINE_AXIS, SPINE_ARTIFACT, HIP_POSITION, HIP_ROI]

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("DXAQC_MODELS", ROOT / "models"))


@dataclass
class Prediction:
    quality_class: int
    violations: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    details: dict = field(default_factory=dict)


class BasePredictor:
    name = "base"

    def predict(self, pixels: np.ndarray, region: str) -> Prediction:  # noqa: ARG002
        raise NotImplementedError


class DummyPredictor(BasePredictor):
    name = "dummy"

    def predict(self, pixels: np.ndarray, region: str) -> Prediction:  # noqa: ARG002
        return Prediction(0)


class SpineRulesPredictor(BasePredictor):
    """Позвоночник: геометрические правила. Другие области → «качественное» (нет модели)."""
    name = "spine_rules"

    def predict(self, pixels: np.ndarray, region: str) -> Prediction:
        if region != "lumbar_spine":
            return Prediction(0, details={"note": "no model for region"})
        from .spine_rules import measure_spine
        m = measure_spine(pixels)
        flags = m.flags()
        viol = [k for k, v in flags.items() if v]
        details = {"axis_angle_deg": round(m.axis_angle_deg, 2), "curvature_deg": round(m.curvature_deg, 2),
                   "axis_coverage": round(m.axis_coverage, 2), "iliac_ratio": round(m.iliac_ratio, 3),
                   "artifact_px": m.artifact_px}
        return Prediction(int(bool(viol)), viol, m.scores(), details)


class HipEmbedLRPredictor(BasePredictor):
    """Бедро: ImageNet-эмбеддинг (convnext_tiny, заморожен) + логистическая регрессия.
    Левое бедро зеркалится. Модель: models/hip_convnext_lr.joblib (scripts/train_hip_lr.py)."""
    name = "hip_embed_lr"

    def __init__(self, path: Path | None = None):
        import joblib
        self.bundle = joblib.load(path or MODELS_DIR / "hip_convnext_lr.joblib")
        self._model = None

    def _backbone(self):
        if self._model is None:
            import torch, torchvision
            os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch_home"))
            m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            m.classifier[2] = torch.nn.Identity()
            self._model = m.eval()
        return self._model

    def embed(self, pixels: np.ndarray, region: str) -> np.ndarray:
        import torch
        from PIL import Image
        a = pixels[:, ::-1] if region == "hip_left" else pixels
        h, w = a.shape; s = max(h, w)
        canvas = np.zeros((s, s), np.uint8); canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
        a = np.asarray(Image.fromarray(canvas).resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255.0
        x = torch.from_numpy(a)[None].repeat(3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        with torch.no_grad():
            return self._backbone()(((x - mean) / std)[None]).numpy()[0]

    def predict(self, pixels: np.ndarray, region: str) -> Prediction:
        if not region.startswith("hip"):
            return Prediction(0, details={"note": "no model for region"})
        e = self.embed(pixels, region)[None]
        b = self.bundle
        p_q = float(b["quality"]["model"].predict_proba(e)[0, 1])
        scores = {HIP_POSITION: float(b[HIP_POSITION]["model"].predict_proba(e)[0, 1]),
                  HIP_ROI: float(b[HIP_ROI]["model"].predict_proba(e)[0, 1])}
        viol = [c for c in (HIP_POSITION, HIP_ROI) if scores[c] >= b[c]["thr"]]
        q = int(p_q >= b["quality"]["thr"] or bool(viol))
        if q and not viol:  # контракт: quality_class=1 ⇒ хотя бы один код; берём самый вероятный
            viol = [max(scores, key=scores.get)]
        return Prediction(q, viol, {**scores, "hip_quality": p_q}, {"p_quality": round(p_q, 3)})


class MVPPredictor(BasePredictor):
    """Позвоночник — правила, бедро — дообученный CNN (ансамбль фолдов); fallback — эмбеддинг + LR."""
    name = "mvp"

    def __init__(self, hip_tags: list[str] | None = None):
        self.spine = SpineRulesPredictor()
        try:
            from .hip_cnn import HipCNNPredictor
            self.hip = HipCNNPredictor(hip_tags)
        except FileNotFoundError:
            self.hip = HipEmbedLRPredictor()

    def predict(self, pixels: np.ndarray, region: str) -> Prediction:
        return self.spine.predict(pixels, region) if region == "lumbar_spine" else self.hip.predict(pixels, region)


def load_predictor(name: str = "release") -> BasePredictor:
    if name == 'release':
        from .api import create_predictor
        return create_predictor()
    if name == "hip_cnn":
        from .hip_cnn import HipCNNPredictor
        return HipCNNPredictor()
    return {"dummy": DummyPredictor, "spine_rules": SpineRulesPredictor, "hip_embed_lr": HipEmbedLRPredictor, "mvp": MVPPredictor}[name]()
