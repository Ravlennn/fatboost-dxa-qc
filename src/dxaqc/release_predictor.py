"""Offline model-based anatomy router, spine heads and five-fold hip CNN."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .bundle import bundle_path, verify_bundle
from .hip_backbones import build_hip_model
from .predictor import BasePredictor, Prediction, SPINE_LAYOUT, SPINE_AXIS, SPINE_ARTIFACT, HIP_POSITION, HIP_ROI
from .release_features import FrozenDenseNetEncoder, predict_linear_head


def hip_tensor(pixels: np.ndarray, region: str, res: int = 384) -> torch.Tensor:
    """Match historical hip training: native GE uint8 range [0,252], square then bilinear.

    DICOM reader yields [0,255]. All 153 local hip source frames have min=0/max=252;
    this inverse normalization exactly restores them before PIL interpolation.
    """
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or min(pixels.shape) < 1 or max(pixels.shape) > 4096:
        raise ValueError('Expected 2D uint8 image with sides between 1 and 4096')
    a = np.rint(pixels.astype(np.float32) * (252.0 / 255.0)).astype(np.uint8)
    if region == 'hip_left':
        a = a[:, ::-1]
    h, w = a.shape
    size = max(h, w)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = a
    image = np.array(Image.fromarray(canvas).resize((res, res), Image.Resampling.BILINEAR), copy=True)
    x = torch.from_numpy(image).float()[None].repeat(3, 1, 1) / 255
    return ((x - torch.tensor([.485, .456, .406])[:, None, None])
            / torch.tensor([.229, .224, .225])[:, None, None])[None]


class ReleasePredictor(BasePredictor):
    name = 'release'

    def __init__(self, root: Path, *, device='cpu', threads=2):
        self.root = Path(root)
        self.manifest = verify_bundle(self.root)
        self.model_id = self.manifest['model_id']
        self.device = device
        self.encoder = FrozenDenseNetEncoder(bundle_path(self.root, self.manifest['encoder']), threads=threads, device=device)
        heads = json.loads(bundle_path(self.root, self.manifest['heads']).read_text())
        if heads['checkpoint_sha256'] != self.encoder.checkpoint_sha256:
            raise ValueError('Linear heads were fitted on a different feature extractor')
        self.heads = heads['heads']
        expected = {'region', 'spine_quality', SPINE_LAYOUT, SPINE_AXIS, SPINE_ARTIFACT}
        if expected - self.heads.keys():
            raise ValueError('Missing required anatomy/spine head')
        self.hip_models = None
        self.view_gate = json.loads(bundle_path(self.root, self.manifest['view_gate']).read_text()) \
            if 'view_gate' in self.manifest else None
        self.use_view_gate = True
        self.landmarks = None
        self.landmark_heads = {}
        self.hip_heads = {}
        self.artifact_head = None
        self.spine_policy = 'gate'
        if 'landmarks' in self.manifest:
            from .landmarks import LandmarkPredictor
            spec = self.manifest['landmarks']
            lm_heads = json.loads(bundle_path(self.root, spec['heads']).read_text())
            self.landmarks = LandmarkPredictor(str(bundle_path(self.root, spec['checkpoint'])), device=device)
            self.landmark_heads = lm_heads['heads']
            self.hip_heads = lm_heads.get('hip_heads', {})
            self.artifact_head = lm_heads.get('spine_artifact_head')
            self.spine_policy = lm_heads.get('spine_quality_policy', 'gate')

    def _load_hip(self):
        if self.hip_models is None:
            loaded = []
            for run in self.manifest['hip_runs']:
                models = []
                for relative in run['checkpoints']:
                    model = build_hip_model(run['arch'], outputs=3, pretrained=False)
                    model.load_state_dict(torch.load(bundle_path(self.root, relative), map_location='cpu', weights_only=True))
                    models.append(model.requires_grad_(False).eval().to(self.device))
                loaded.append((models, run['res']))
            self.hip_models = loaded
        return self.hip_models

    def _score(self, key, features):
        head = self.heads[key]
        return float(predict_linear_head(head, features)[0, head['classes'].index(1)])

    def predict_image(self, pixels, mm_per_px=None):
        features = self.encoder.embed(pixels)
        if self.view_gate is not None and self.use_view_gate:
            # Out-of-scope images (forearm, lateral/VFA, whole body) must not receive a verdict.
            from .landmarks import apply_head
            score = apply_head(self.view_gate, {f'f{i}': float(v) for i, v in enumerate(features[0])})
            if score < self.view_gate['threshold']:
                raise ValueError(f'unsupported anatomical region: not an AP lumbar spine or proximal femur '
                                 f'(view gate {score:.3f} < {self.view_gate["threshold"]:.3f})')
        head = self.heads['region']
        probabilities = predict_linear_head(head, features)[0]
        region = head['classes'][int(np.argmax(probabilities))]
        if region == 'spine':
            region = 'lumbar_spine'
        if region not in ('lumbar_spine', 'hip_left', 'hip_right'):
            raise ValueError('Unsupported anatomy prediction')
        prediction = self._predict(pixels, region, features, mm_per_px=mm_per_px)
        prediction.details.update(region_method='frozen_densenet_linear', region_confidence=float(probabilities.max()))
        if probabilities.max() < .8:
            prediction.details['warnings'].append('review: low anatomy confidence')
        return region, prediction

    def predict(self, pixels, region, mm_per_px=None):
        return self._predict(pixels, region, self.encoder.embed(pixels) if region == 'lumbar_spine' else None,
                             mm_per_px=mm_per_px)

    def _landmarks(self, pixels, region, mm_per_px=None):
        from .landmarks import head_features, measurements
        res = self.landmarks.predict(pixels, region)
        m = measurements(res['points'], region, pixels.shape, pixels, mm_per_px=mm_per_px)
        return {'points': res['points'], 'confidence': res['confidence'], 'measurements': m,
                'features': head_features(m, region)}

    @torch.inference_mode()
    def _predict(self, pixels, region, features, mm_per_px=None):
        landmarks = self._landmarks(pixels, region, mm_per_px) if self.landmarks is not None else None
        forced = []  # reasons allowed to set quality=1 (TZ: any violation => defect)
        if region == 'lumbar_spine':
            quality = self._score('spine_quality', features)
            threshold = self.heads['spine_quality']['threshold']
            reasons = {code: self._score(code, features) for code in (SPINE_LAYOUT, SPINE_AXIS, SPINE_ARTIFACT)}
            thresholds = {code: self.heads[code]['threshold'] for code in reasons}
            if landmarks is not None:
                from .landmarks import apply_head
                for code, head in self.landmark_heads.items():
                    reasons[code] = apply_head(head, landmarks['features'])
                    thresholds[code] = head['threshold']
                    if self.spine_policy == 'or':
                        forced.append(code)
                if self.artifact_head is not None:
                    from .artifacts import detect as detect_artifacts, head_features as art_features
                    art = detect_artifacts(pixels, landmarks['points'], mm=landmarks['measurements'].get('mm_per_px', 0.6))
                    landmarks['artifacts'] = {k: art[k] for k in ('wires', 'metal', 'wire_len_vert', 'metal_area_mm2')}
                    landmarks['artifact_mask'] = art['mask']
                    c = float(np.clip(reasons[SPINE_ARTIFACT], 1e-4, 1 - 1e-4))
                    reasons[SPINE_ARTIFACT] = apply_head(self.artifact_head,
                                                         {**art_features(art), 'cnn': float(np.log(c / (1 - c)))})
                    thresholds[SPINE_ARTIFACT] = self.artifact_head['threshold']
            codes = [code for code, score in reasons.items() if score >= thresholds[code]]
            if landmarks is not None and self.spine_policy == 'reasons':
                # Defect iff a reason is found (nested selection, docs/FINAL_VALIDATION.md).
                # quality score: 0.5 at the decision boundary, max over reasons of score/threshold.
                ratio = max(reasons[c] / max(thresholds[c], 1e-9) for c in reasons)
                quality, threshold = float(min(1.0, 0.5 * ratio)), 0.5
                if codes:
                    quality = max(quality, 0.5)
                else:
                    quality = min(quality, 0.4999)
        else:
            run_probabilities, tensors = [], {}
            for models, resolution in self._load_hip():
                if resolution not in tensors:
                    tensors[resolution] = hip_tensor(pixels, region, resolution).to(self.device)
                run_probabilities.append(np.mean([torch.sigmoid(model(tensors[resolution])).cpu().numpy()[0] for model in models], axis=0))
            p = np.mean(run_probabilities, axis=0)
            quality = float(p[0])
            thresholds = self.manifest['hip_thresholds']
            threshold = thresholds['quality']
            reasons = {HIP_POSITION: float(p[1]), HIP_ROI: float(p[2])}
            reason_thr = {HIP_POSITION: thresholds['v_rotation'], HIP_ROI: thresholds['v_roi']}
            if landmarks is not None and self.hip_heads:
                # Fusion: CNN score + lesser-trochanter geometry (docs/METHODS_AND_MODELS.md).
                from .landmarks import apply_head
                from .lesser_trochanter import detect, head_features
                lt, lt_err = detect(pixels, landmarks['points'], mm=landmarks['measurements'].get('mm_per_px'))
                landmarks['lesser_trochanter'] = lt
                cnn_in = {'p_quality': quality, 'p_hip_positioning': reasons[HIP_POSITION]}
                for code, head in self.hip_heads.items():
                    c = float(np.clip(cnn_in[head['cnn_input']], 1e-4, 1 - 1e-4))
                    feats = {**head_features(lt), 'cnn': float(np.log(c / (1 - c)))}
                    score = apply_head(head, feats)
                    if code == 'hip_quality':
                        quality, threshold = score, head['threshold']
                    else:
                        reasons[code], reason_thr[code] = score, head['threshold']
                if lt is None:
                    landmarks.setdefault('warnings', []).append(f'lesser trochanter not measured: {lt_err}')
            codes = [code for code, score in reasons.items() if score >= reason_thr[code]]
        bad = int(quality >= threshold)
        warnings = []
        if not bad and any(code in forced for code in codes):
            bad = 1
            warnings.append('landmark measurement violates a TZ criterion; binary head overridden')
        if bad and not codes:
            codes = ['quality_unspecified']
            warnings.append('review: quality head detects a defect; specific cause is uncertain')
        if not bad and codes:
            warnings.append('review: binary quality and cause heads disagree')
            codes = []
        details = {'warnings': warnings, 'quality_threshold': threshold}
        if landmarks is not None:
            details['landmarks'] = landmarks
        return Prediction(bad, codes, {'quality': quality, **reasons}, details)
