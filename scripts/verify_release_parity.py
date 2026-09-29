"""Replay every held-out hip prediction with the release DICOM preprocessing.

CPU-only, one model at a time, two threads, small batches and deliberate pauses.
No patient identifiers or paths are written to the aggregate result JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dxaqc.bundle import sha256
from dxaqc.dicom_io import read_dicom
from dxaqc.hip_backbones import build_hip_model
from dxaqc.release_predictor import hip_tensor

TARGETS = ["quality", "v_rotation", "v_roi"]
KEY = ["study", "image_uid"]


def training_tensor(path: Path, region: str, resolution: int):
    """Historical train_hip_cnn.load/test_tf, reproduced independently."""
    native = pydicom.dcmread(path).pixel_array
    if region == "hip_left":
        native = np.ascontiguousarray(native[:, ::-1])
    h, w = native.shape
    side = max(h, w)
    canvas = np.zeros((side, side), np.uint8)
    canvas[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = native
    transform = T.Compose([
        T.Resize((resolution, resolution)), T.Grayscale(3), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(Image.fromarray(np.ascontiguousarray(canvas)))[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="local_convnext384_s0_lowload_v2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-pause", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/release_checks/parity.json")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8 or args.batch_pause < 0 or args.atol <= 0:
        parser.error("Require batch-size in [1,8], nonnegative pause and positive atol")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    directory = ROOT / "outputs/hip_cnn" / args.tag
    config = json.loads((directory / "config.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    for filename, expected in provenance["tables"].items():
        if sha256(ROOT / "data/interim" / filename) != expected:
            raise ValueError(f"Source table checksum mismatch: {filename}")
    if config.get("crop") or config.get("roi_file"):
        raise ValueError("This parity check supports full-frame models only")
    labels = pd.read_csv(ROOT / "data/interim/image_labels.csv")
    folds = pd.read_csv(ROOT / "data/interim/folds.csv")
    labels = labels.merge(folds, on=KEY, validate="one_to_one")
    labels = labels.loc[labels.region.isin(["hip_left", "hip_right"]) & labels.quality.notna()].sort_values(KEY).reset_index(drop=True)
    original = pd.read_csv(directory / "oof.csv")
    data = labels.merge(original, on=KEY, how="outer", validate="one_to_one", suffixes=("", "_oof"), indicator=True)
    if (data._merge != "both").any():
        raise ValueError("OOF coverage differs from labelled hip data")
    for name in ["region", "fold", *TARGETS]:
        if not ((data[name] == data[name + "_oof"]) | (data[name].isna() & data[name + "_oof"].isna())).all():
            raise ValueError(f"OOF metadata mismatch: {name}")
    if set(data.fold) != set(range(5)):
        raise ValueError("Expected five complete held-out folds")
    reference = data[["p_" + name for name in TARGETS]].to_numpy(float)
    if not np.isfinite(reference).all():
        raise ValueError("Incomplete reference predictions")
    replay = np.full_like(reference, np.nan)
    tensor_max_difference = 0.0
    tensor_mismatches = 0
    checkpoint_hashes = {}
    fold_results = []
    source = ROOT / "data/interim/train/Исследования"
    audit_path = ROOT / "outputs/release_ensemble_v1/audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    with torch.inference_mode():
        for fold in range(5):
            checkpoint = directory / f"fold{fold}.pt"
            digest = sha256(checkpoint)
            checkpoint_hashes[f"fold{fold}.pt"] = digest
            if audit and audit["source_runs"][args.tag]["weights_sha256"][f"fold{fold}.pt"] != digest:
                raise ValueError("Checkpoint changed after ensemble audit")
            model = build_hip_model(config["arch"], outputs=3, pretrained=False)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.requires_grad_(False).eval()
            indices = np.flatnonzero(data.fold.to_numpy() == fold)
            for offset in range(0, len(indices), args.batch_size):
                batch_indices = indices[offset:offset + args.batch_size]
                tensors = []
                for index in batch_indices:
                    row = data.iloc[index]
                    path = source / row.path
                    tensor = hip_tensor(read_dicom(path, source_id=row.path).pixels, row.region, config["res"])
                    original_tensor = training_tensor(path, row.region, config["res"])
                    difference = float((tensor - original_tensor).abs().max())
                    tensor_max_difference = max(tensor_max_difference, difference)
                    tensor_mismatches += int(difference != 0)
                    tensors.append(tensor)
                replay[batch_indices] = torch.sigmoid(model(torch.cat(tensors))).numpy()
                if args.batch_pause:
                    time.sleep(args.batch_pause)
            fold_difference = np.abs(replay[indices] - reference[indices])
            fold_results.append({"fold": fold, "images": len(indices), "max_abs_difference": float(fold_difference.max()),
                                 "probabilities_outside_tolerance": int((fold_difference > args.atol).sum())})
            print(json.dumps(fold_results[-1]), flush=True)
            del model
    if not np.isfinite(replay).all():
        raise ValueError("Incomplete replay")
    absolute = np.abs(replay - reference)
    selection = json.loads((ROOT / "outputs/release_ensemble_v1/selection.json").read_text())
    thresholds = selection.get('stable_thresholds', selection['thresholds'])
    by_target = {}
    for j, target in enumerate(TARGETS):
        changed = (replay[:, j] >= thresholds[target]) != (reference[:, j] >= thresholds[target])
        by_target[target] = {
            "max_abs_difference": float(absolute[:, j].max()),
            "mean_abs_difference": float(absolute[:, j].mean()),
            "probabilities_outside_tolerance": int((absolute[:, j] > args.atol).sum()),
            "decisions_changed_at_0_5": int(((replay[:, j] >= 0.5) != (reference[:, j] >= 0.5)).sum()),
            "decisions_changed_at_tuned_threshold": int(changed.sum()),
            "changed_decisions_with_reference_within_tolerance_of_threshold": int((changed & (np.abs(reference[:, j] - thresholds[target]) <= args.atol)).sum()),
        }
    passed = bool(tensor_mismatches == 0 and not (absolute > args.atol).any())
    result = {
        "status": "PASS" if passed else "FAIL", "tag": args.tag, "images": len(data), "probabilities": int(reference.size),
        "device": "cpu", "dtype": "float32", "threads": 2, "interop_threads": 1,
        "batch_size": args.batch_size, "batch_pause_seconds": args.batch_pause, "absolute_tolerance": args.atol,
        "max_abs_probability_difference": float(absolute.max()), "mean_abs_probability_difference": float(absolute.mean()),
        "probabilities_outside_tolerance": int((absolute > args.atol).sum()),
        "images_outside_tolerance": int((absolute > args.atol).any(axis=1).sum()),
        "preprocessing_max_abs_tensor_difference": tensor_max_difference, "preprocessing_images_not_exact": tensor_mismatches,
        "target_results": by_target, "fold_results": fold_results, "checkpoint_sha256": checkpoint_hashes,
        "reference_oof_sha256": sha256(directory / "oof.csv"), "table_sha256": provenance["tables"],
        "torch_version": torch.__version__, "elapsed_seconds_including_pauses": round(time.perf_counter() - started, 3),
        "interpretation": "Every image replayed with its held-out fold. Checks inference/preprocessing equivalence, not independent-test performance. Small CPU/MPS arithmetic differences may flip a sample exactly at a tuned threshold.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "images": result["images"], "max_abs_difference": result["max_abs_probability_difference"],
                      "preprocessing_not_exact": tensor_mismatches, "seconds": result["elapsed_seconds_including_pauses"]}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
