"""Pinned frozen DINOv2-S/14 vs DenseNet121; identical five study folds and LR.

Run with --prepare to download the explicitly pinned official assets first.
Only cached image embeddings and linear heads are trained; no backbone updates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from dxaqc.dicom_io import read_dicom
from dxaqc.dinov2_features import COMMIT, ARCHIVE_SHA256, CHECKPOINT_SHA256, CODE_SHA256, FrozenDinoEncoder
from dxaqc.release_features import PREPROCESS, predict_linear_head
from run_release_loss_suite import checked_frame, fit, save, digest, binary_metrics, choose_threshold, fixed_heads


def prepare(vendor: Path) -> None:
    vendor.mkdir(parents=True, exist_ok=True)
    sources = {
        "official-code.zip": (f"https://codeload.github.com/facebookresearch/dinov2/zip/{COMMIT}", ARCHIVE_SHA256),
        "dinov2_vits14_pretrain.pth": ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth", CHECKPOINT_SHA256),
    }
    manifest = dict(repository="https://github.com/facebookresearch/dinov2", commit=COMMIT, files={})
    for name, (url, expected) in sources.items():
        path = vendor / name
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "dxaqc-research"})
            partial = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if digest(partial) != expected:
                raise ValueError(f"Downloaded checksum mismatch: {name}")
            partial.rename(path)
        if digest(path) != expected:
            raise ValueError(f"Official asset checksum mismatch: {name}")
        manifest["files"][name] = dict(url=url, sha256=expected, bytes=path.stat().st_size)
    with ZipFile(vendor / "official-code.zip") as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts
            if Path(item.filename).is_absolute() or ".." in parts or not parts or parts[0] != f"dinov2-{COMMIT}":
                raise ValueError("Unsafe official archive member")
            if item.is_dir():
                continue
            target = vendor / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
    save(vendor / "sources.json", manifest)


def extract(frame, args):
    rows = frame[["study", "image_uid", "path", "pixel_sha256"]].to_dict("records")
    identity = dict(rows=rows, preprocessing=PREPROCESS, checkpoint_sha256=CHECKPOINT_SHA256,
                    commit=COMMIT, source_sha256=CODE_SHA256, output="final normalized CLS token; 384 dimensions")
    feature_file, meta_file = args.out / "features.npy", args.out / "features_meta.json"
    if feature_file.exists() and meta_file.exists():
        saved = json.loads(meta_file.read_text())
        if saved["identity"] != identity or saved["sha256"] != digest(feature_file):
            raise ValueError("DINO cache provenance mismatch")
        features = np.load(feature_file, allow_pickle=False)
        if features.shape != (len(frame), 384) or not np.isfinite(features).all():
            raise ValueError("DINO cache invalid")
        print("Verified existing DINO features", flush=True)
        return features
    model = FrozenDinoEncoder(args.out / "vendor" / f"dinov2-{COMMIT}",
                              args.out / "vendor/dinov2_vits14_pretrain.pth")
    start_time, encoder_time = time.perf_counter(), 0.
    chunks = []
    for start in range(0, len(frame), 4):
        images = []
        for row in frame.iloc[start:start+4].itertuples():
            path = args.source / row.path
            ds = pydicom.dcmread(path)
            pixels = ds.pixel_array
            fingerprint = hashlib.sha256(str(pixels.shape).encode() + pixels.dtype.str.encode() + pixels.tobytes()).hexdigest()
            if fingerprint != row.pixel_sha256 or str(ds.SOPInstanceUID) != row.image_uid or str(ds.StudyInstanceUID) != row.study_uid:
                raise ValueError("DICOM identity differs from labels")
            images.append(read_dicom(path).pixels)
        forward_start = time.perf_counter()
        chunks.append(model.embed(images))
        encoder_time += time.perf_counter() - forward_start
        if start % 24 == 0 or start+4 >= len(frame):
            print(f"DINO features {min(start+4, len(frame))}/{len(frame)}", flush=True)
        time.sleep(args.batch_pause)
    features = np.concatenate(chunks)
    np.save(feature_file, features, allow_pickle=False)
    save(meta_file, dict(identity=identity, sha256=digest(feature_file), extraction_seconds=time.perf_counter()-start_time,
                        preprocess_and_forward_seconds=encoder_time, excludes_model_load=True))
    return features


def paired_ci(y, candidate, control, groups, count=2000):
    blocks = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(42)
    differences = {"roc_auc": [], "average_precision": []}
    for _ in range(count):
        index = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
        if len(np.unique(y[index])) < 2:
            continue
        a, b = binary_metrics(y[index], candidate[index]), binary_metrics(y[index], control[index])
        for key in differences:
            differences[key].append(a[key]-b[key])
    a, b = binary_metrics(y, candidate), binary_metrics(y, control)
    return {k: dict(delta=a[k]-b[k], low=float(np.quantile(v, .025)), high=float(np.quantile(v, .975)),
                    valid_replicates=len(v)) for k, v in differences.items()}


def quality(frame, features, anatomy, args, heads):
    selected = (frame.region == "spine" if anatomy == "spine" else frame.region != "spine") & frame.quality.notna()
    rows = frame.loc[selected, ["study", "study_uid", "image_uid", "region", "quality", "fold"]].copy().reset_index(drop=True)
    x, y = features[selected], rows.quality.to_numpy(dtype=int)
    folds, groups = rows.fold.to_numpy(), rows.study.to_numpy()
    p, thresholds = np.full(len(y), np.nan), np.full(len(y), np.nan)
    per_fold = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        xt, yt, gt = x[train], y[train], groups[train]
        inner_p = np.full(len(yt), np.nan)
        inner_split = StratifiedGroupKFold(3, shuffle=True, random_state=42+fold).split(xt, yt, gt)
        for ti, vi in inner_split:
            if set(gt[ti]) & set(gt[vi]):
                raise ValueError("Inner study leakage")
            inner_p[vi] = predict_linear_head(fit(xt[ti], yt[ti]), xt[vi])[:, 1]
        thresholds[valid] = choose_threshold(yt, inner_p)
        p[valid] = predict_linear_head(fit(xt, yt), x[valid])[:, 1]
        per_fold.append(dict(fold=fold, threshold=float(thresholds[valid][0]),
                             metrics=binary_metrics(y[valid], p[valid], thresholds[valid])))
    rows["p_dinov2_balanced_logistic"] = p
    rows["threshold_inner_f1"] = thresholds
    rows.to_csv(args.out / f"{anatomy}_oof.csv", index=False)
    head = fit(x, y)
    head.update(anatomy=anatomy, threshold=choose_threshold(y, p), positive_train=int(y.sum()),
                threshold_origin="development OOF; assessment uses separate inner-fold thresholds")
    heads[f"{anatomy}_quality"] = head
    baseline = pd.read_csv(args.baseline / f"{anatomy}_oof.csv")
    comparison = rows.merge(baseline, on=["study", "study_uid", "image_uid", "region", "quality", "fold"], validate="one_to_one")
    if len(comparison) != len(rows):
        raise ValueError("DenseNet baseline does not match DINO comparison cohort")
    result = dict(pooled_fixed_threshold=binary_metrics(y, p), pooled_inner_f1_threshold=binary_metrics(y, p, thresholds),
                  mean_fold_ap=float(np.mean([f["metrics"]["average_precision"] for f in per_fold])), folds=per_fold,
                  vs_densenet_fixed_balanced_logistic=paired_ci(comparison.quality.to_numpy(dtype=int),
                      comparison.p_dinov2_balanced_logistic.to_numpy(), comparison.p_balanced_logistic.to_numpy(),
                      comparison.study.to_numpy()),
                  densenet_fixed_balanced_logistic=binary_metrics(comparison.quality.to_numpy(dtype=int),
                                                                  comparison.p_balanced_logistic.to_numpy()))
    save(args.out / f"{anatomy}_results.json", result)
    print(anatomy, json.dumps(result["pooled_fixed_threshold"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/dinov2_probe_v1")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/interim/image_labels.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "data/interim/folds.csv")
    parser.add_argument("--source", type=Path, default=ROOT / "data/interim/train/Исследования")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/release_loss_v1")
    parser.add_argument("--batch-pause", type=float, default=.75)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.batch_pause < 0:
        raise ValueError("Pause must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prepare or args.prepare_only:
        prepare(args.out / "vendor")
    if args.prepare_only:
        return
    if (args.out / "run.json").exists():
        raise ValueError("Completed run already exists; use a new output directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    protocol = dict(experiment="dinov2_probe_v1", model="DINOv2 ViT-S/14 no registers; frozen normalized CLS384",
                    commit=COMMIT, code_sha256=CODE_SHA256, checkpoint_sha256=CHECKPOINT_SHA256,
                    labels_sha256=digest(args.labels), folds_sha256=digest(args.folds), preprocess=PREPROCESS,
                    baseline="frozen DenseNet121 with identical preprocessing and fixed balanced LR C=.001",
                    head="train-only StandardScaler + balanced LogisticRegression C=.001 max_iter2000 seed42",
                    validation="original 5 study outer folds; no backbone fitting or hyperparameter search",
                    threshold="three train-only inner group folds select F1 threshold; same helper as DenseNet suite",
                    timing="CPU2 interop1 batch4 pause.75sec; no model compilation; no GPU",
                    ci="2000 paired study bootstrap conditional on predictions; not retraining uncertainty",
                    limitations=["Development comparison, not independent test", "Patient linkage unavailable",
                                 "Folds reconstructed locally; no cross-site evidence", "Anatomy quality metrics use reference region",
                                 "Dataset conflict/missing labels unchanged; binary and reason objectives independent"])
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing protocol differs")
    save(protocol_path, protocol)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        frame = checked_frame(args.labels, args.folds)
        baseline_protocol = json.loads((args.baseline / "protocol.json").read_text())
        for key in ["labels_sha256", "folds_sha256"]:
            if baseline_protocol[key] != protocol[key]:
                raise ValueError("Baseline identity mismatch")
        features = extract(frame, args)
        heads, auxiliary = fixed_heads(frame, features, args.out)
        results = {anatomy: quality(frame, features, anatomy, args, heads) for anatomy in ["spine", "hip"]}
        bundle = dict(schema_version=1, backbone="dinov2_vits14", feature_dimension=384,
                      checkpoint_sha256=CHECKPOINT_SHA256, commit=COMMIT, source_sha256=CODE_SHA256,
                      preprocess=PREPROCESS, heads=heads, features_sha256=digest(args.out / "features.npy"),
                      labels_sha256=protocol["labels_sha256"], folds_sha256=protocol["folds_sha256"])
        save(args.out / "final_heads.json", bundle)
        restored = json.loads((args.out / "final_heads.json").read_text())
        for key, head in heads.items():
            np.testing.assert_array_equal(predict_linear_head(head, features), predict_linear_head(restored["heads"][key], features))
        save(args.out / "run.json", dict(status="COMPLETE", seconds=time.perf_counter()-started,
             n_images=len(frame), n_quality_labels=int(frame.quality.notna().sum()), auxiliary=auxiliary, quality=results,
             source_sha256=digest(__file__), extractor_sha256=digest(ROOT / "src/dxaqc/dinov2_features.py"),
             shared_fit_source_sha256=digest(ROOT / "scripts/run_release_loss_suite.py"),
             final_heads_sha256=digest(args.out / "final_heads.json"), reload_predictions_identical=True,
             versions={p: importlib.metadata.version(p) for p in ["torch", "torchvision", "scikit-learn", "numpy", "pydicom"]}))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
