"""Nested study-CV of numerical geometry, frozen CNN and their learned fusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[name] = "2"

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dxaqc.dicom_io import read_dicom
from dxaqc.release_features import DENSENET_SHA256, PREPROCESS, predict_linear_head
from dxaqc.spine_features import FEATURE_VERSION, FEATURE_NAMES, spine_features
from run_release_loss_suite import checked_frame, digest, save, binary_metrics, choose_threshold
from run_dinov2_probe import paired_ci

CS = [.001, .01, .1]
SETS = ["cnn", "geometry", "fusion"]


def fit_head(x, y, c, feature_set):
    x = np.asarray(x, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    if len(np.unique(y)) != 2:
        raise ValueError("Cannot train a binary head with one observed class")
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42).fit(scaler.transform(x), y)
    head = dict(classes=model.classes_.tolist(), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=model.coef_.tolist(), intercept=model.intercept_.tolist(), C=c, feature_set=feature_set,
                n_train=len(y), positive_train=int(y.sum()), threshold=.5, loss="balanced_logistic")
    np.testing.assert_allclose(predict_linear_head(head, x), model.predict_proba(scaler.transform(x)), atol=1e-10)
    return head


def prepare_features(frame, args):
    meta = json.loads((args.baseline / "features_meta.json").read_text())
    expected = dict(rows=frame[["study", "image_uid", "path", "pixel_sha256"]].to_dict("records"),
                    preprocessing=PREPROCESS, checkpoint_sha256=DENSENET_SHA256)
    if meta["identity"] != expected or meta["sha256"] != digest(args.baseline / "features.npy"):
        raise ValueError("DenseNet cache identity mismatch")
    cnn = np.load(args.baseline / "features.npy", allow_pickle=False)
    if cnn.shape != (len(frame), 1024) or not np.isfinite(cnn).all():
        raise ValueError("Invalid DenseNet feature cache")
    selected = (frame.region == "spine") & frame.quality.notna()
    rows, cnn = frame.loc[selected].reset_index(drop=True), cnn[selected]
    geometry = []
    started = time.perf_counter()
    for i, row in enumerate(rows.itertuples()):
        path = args.source / row.path
        ds = pydicom.dcmread(path)
        raw = ds.pixel_array
        identity = hashlib.sha256(str(raw.shape).encode() + raw.dtype.str.encode() + raw.tobytes()).hexdigest()
        if identity != row.pixel_sha256 or str(ds.SOPInstanceUID) != row.image_uid or str(ds.StudyInstanceUID) != row.study_uid:
            raise ValueError("Source image identity mismatch")
        geometry.append(spine_features(read_dicom(path).pixels))
        if i % 25 == 0:
            print("Spine geometry", i+1, "/", len(rows), flush=True)
        time.sleep(.02)
    geometry = np.stack(geometry)
    np.save(args.out / "geometry.npy", geometry, allow_pickle=False)
    save(args.out / "geometry_meta.json", dict(version=FEATURE_VERSION, names=FEATURE_NAMES,
         rows=rows[["study", "image_uid", "path", "pixel_sha256"]].to_dict("records"),
         sha256=digest(args.out / "geometry.npy"), seconds=time.perf_counter()-started,
         source_sha256=digest(ROOT / "src/dxaqc/spine_features.py"),
         measurement_source_sha256=digest(ROOT / "src/dxaqc/spine_rules.py")))
    return rows, dict(cnn=cnn, geometry=geometry, fusion=np.concatenate([cnn, geometry], axis=1))


def inner_candidates(features, y, groups, seed):
    split = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(features["cnn"], y, groups))
    results = []
    for kind in SETS:
        for c in CS:
            x, p, aps = features[kind], np.full(len(y), np.nan), []
            for ti, vi in split:
                if set(groups[ti]) & set(groups[vi]):
                    raise ValueError("Inner study leakage")
                p[vi] = predict_linear_head(fit_head(x[ti], y[ti], c, kind), x[vi])[:, 1]
                aps.append(binary_metrics(y[vi], p[vi])["average_precision"])
            results.append(dict(feature_set=kind, C=c, mean_fold_ap=float(np.mean(aps)),
                                threshold=choose_threshold(y, p)))
    return results


def evaluate_quality(rows, features, args):
    y, groups, folds = rows.quality.to_numpy(dtype=int), rows.study.to_numpy(), rows.fold.to_numpy()
    keys = SETS + ["nested_selected"]
    probabilities = {k: np.full(len(rows), np.nan) for k in keys}
    thresholds = {k: np.full(len(rows), np.nan) for k in keys}
    outer = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        inner = inner_candidates({k: v[train] for k, v in features.items()}, y[train], groups[train], 42+fold)
        selection = {kind: max((r for r in inner if r["feature_set"] == kind), key=lambda r: r["mean_fold_ap"]) for kind in SETS}
        selection["nested_selected"] = max(inner, key=lambda r: r["mean_fold_ap"])
        fold_summary = dict(fold=fold, inner_candidates=inner, selected={})
        for key, choice in selection.items():
            kind, c = choice["feature_set"], choice["C"]
            x = features[kind]
            probabilities[key][valid] = predict_linear_head(fit_head(x[train], y[train], c, kind), x[valid])[:, 1]
            thresholds[key][valid] = choice["threshold"]
            fold_summary["selected"][key] = dict(**choice,
                  metrics=binary_metrics(y[valid], probabilities[key][valid], thresholds[key][valid]))
        outer.append(fold_summary)
        print("Outer", fold, "selection", selection["nested_selected"], flush=True)
    oof = rows[["study", "study_uid", "image_uid", "region", "quality", "fold"]].copy()
    summary = {}
    for key in keys:
        p, t = probabilities[key], thresholds[key]
        oof[f"p_{key}"], oof[f"threshold_{key}"] = p, t
        summary[key] = dict(pooled=binary_metrics(y, p, t), fixed_05=binary_metrics(y, p),
                            mean_fold_ap=float(np.mean([f["selected"][key]["metrics"]["average_precision"] for f in outer])))
        if key != "cnn":
            summary[key]["paired_vs_nested_cnn"] = paired_ci(y, p, probabilities["cnn"], groups)
    oof.to_csv(args.out / "quality_oof.csv", index=False)
    # Refit selection is development CV. Its training predictions are never reported.
    development = []
    for kind in SETS:
        for c in CS:
            p, aps = np.full(len(y), np.nan), []
            for fold in range(5):
                ti, vi = folds != fold, folds == fold
                p[vi] = predict_linear_head(fit_head(features[kind][ti], y[ti], c, kind), features[kind][vi])[:, 1]
                aps.append(binary_metrics(y[vi], p[vi])["average_precision"])
            development.append(dict(feature_set=kind, C=c, mean_fold_ap=float(np.mean(aps)), threshold=choose_threshold(y, p)))
    choice = max(development, key=lambda r: r["mean_fold_ap"])
    head = fit_head(features[choice["feature_set"]], y, choice["C"], choice["feature_set"])
    head.update(threshold=choice["threshold"], threshold_origin="development OOF; nested outer metrics evaluate selection procedure",
                anatomy="spine")
    baseline = pd.read_csv(args.baseline / "spine_oof.csv")
    # Explicit suffixes avoid accidentally comparing a model with its own scores.
    joined = oof.merge(baseline, on=["study", "study_uid", "image_uid", "region", "quality", "fold"],
                       validate="one_to_one", suffixes=("_fusion", "_baseline"))
    if len(joined) != len(rows):
        raise ValueError("Current DenseNet OOF cohort mismatch")
    comparison = paired_ci(joined.quality.to_numpy(dtype=int), joined.p_nested_selected_fusion.to_numpy(),
                           joined.p_nested_selected_baseline.to_numpy(), joined.study.to_numpy())
    result = dict(n=len(rows), positive=int(y.sum()), candidates=summary, outer_folds=outer,
                  final_development_selection=choice, development_candidates=development,
                  paired_vs_release_nested_loss=comparison,
                  release_nested_loss=binary_metrics(joined.quality.to_numpy(dtype=int), joined.p_nested_selected_baseline.to_numpy(),
                                                     joined.threshold_nested_selected_baseline.to_numpy()))
    save(args.out / "quality_results.json", result)
    return head, result


def reasons(rows, features, args):
    heads, result = {}, {}
    out = rows[["study", "study_uid", "image_uid", "region", "fold"]].copy()
    for key, label in [("spine_scan_range", "v_layout"), ("spine_axis_tilt", "v_axis"), ("spine_artifact", "v_artifact")]:
        selected = rows[label].notna().to_numpy()
        y, x = rows.loc[selected, label].to_numpy(dtype=int), features["fusion"][selected]
        folds = rows.loc[selected, "fold"].to_numpy()
        p = np.full(len(y), np.nan)
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            p[valid] = predict_linear_head(fit_head(x[train], y[train], .01, "fusion"), x[valid])[:, 1]
        head = fit_head(x, y, .01, "fusion")
        head.update(threshold=choose_threshold(y, p), threshold_origin="development OOF; tuned F1 is optimistic", anatomy="spine")
        heads[key] = head
        result[key] = dict(pooled_fixed_threshold=binary_metrics(y, p), deployment_threshold=head["threshold"])
        out[f"p_{key}"] = np.nan
        out.loc[selected, f"p_{key}"] = p
    out.to_csv(args.out / "reason_oof.csv", index=False)
    save(args.out / "reason_results.json", result)
    return heads, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/spine_fusion_v1")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/interim/image_labels.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "data/interim/folds.csv")
    parser.add_argument("--source", type=Path, default=ROOT / "data/interim/train/Исследования")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/release_loss_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "run.json").exists():
        raise ValueError("Completed experiment already exists")
    cv2.setNumThreads(2)
    protocol = dict(name="spine_fusion_v1", feature_sets=SETS, Cs=CS, feature_version=FEATURE_VERSION, feature_names=FEATURE_NAMES,
                    labels_sha256=digest(args.labels), folds_sha256=digest(args.folds),
                    cnn_checkpoint_sha256=DENSENET_SHA256, geometry="native gray normalized; isotropic resize to height400; raw descriptors only",
                    quality="5study outer; 3study inner select features+C by mean AP and threshold by innerOOF F1",
                    fixed_feature_sets="Each of cnn,geometry,fusion independently selects C and threshold inside outer train",
                    reasons="Predefined fusion C=.01, missing labels masked; threshold selected from development OOF",
                    limitations=["Development CV, no independent final test", "Historic geometry extractor was developed on this cohort",
                                 "No ruleflags used; earlier descriptor engineering still introduces selection optimism",
                                 "Clinical landmark correctness unverified; no patient or site-level external validation"],
                    limits="CPU2, no image encoder pass, 20ms pause between images")
    save(args.out / "protocol.json", protocol)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        frame = checked_frame(args.labels, args.folds)
        rows, features = prepare_features(frame, args)
        quality_head, quality_result = evaluate_quality(rows, features, args)
        heads, reason_result = reasons(rows, features, args)
        heads["spine_quality"] = quality_head
        bundle = dict(schema_version=1, backbone="densenet121", checkpoint_sha256=DENSENET_SHA256,
                      preprocess=PREPROCESS, geometry=dict(version=FEATURE_VERSION, names=FEATURE_NAMES, dimension=len(FEATURE_NAMES)),
                      feature_dimensions={k: x.shape[1] for k, x in features.items()}, heads=heads,
                      labels_sha256=protocol["labels_sha256"], folds_sha256=protocol["folds_sha256"])
        save(args.out / "final_heads.json", bundle)
        restored = json.loads((args.out / "final_heads.json").read_text())
        for key, head in heads.items():
            np.testing.assert_array_equal(predict_linear_head(head, features[head["feature_set"]]),
                                          predict_linear_head(restored["heads"][key], features[head["feature_set"]]))
        save(args.out / "run.json", dict(status="COMPLETE", seconds=time.perf_counter()-started, quality=quality_result,
             reasons=reason_result, final_heads_sha256=digest(args.out / "final_heads.json"),
             source_sha256=digest(__file__), reload_predictions_identical=True))
    print("COMPLETE", json.dumps(quality_result["candidates"]), flush=True)


if __name__ == "__main__":
    main()
