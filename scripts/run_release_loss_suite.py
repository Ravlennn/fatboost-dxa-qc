"""Low-load, grouped and nested loss comparison on frozen ImageNet features.

No deep-backbone training. Every scaler, weight and class weight uses training
rows only. Nested outer predictions include inner loss/threshold selection.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dxaqc.dicom_io import read_dicom
from dxaqc.release_features import DENSENET_SHA256, PREPROCESS, FrozenDenseNetEncoder, predict_linear_head

LOSSES = ["balanced_logistic", "bce", "balanced_bce", "focal_gamma2", "smooth_bce_005"]
SEED = 42
C = .001
REASONS = {"spine_scan_range": ("spine", "v_layout"), "spine_axis_tilt": ("spine", "v_axis"),
           "spine_artifact": ("spine", "v_artifact"), "hip_positioning": ("hip", "v_rotation"),
           "hip_roi_margins": ("hip", "v_roi")}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fit(x, y, loss="balanced_logistic", multiclass=False):
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x).astype(np.float64)
    classes = np.unique(y).tolist()
    head = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), classes=classes,
                loss=loss, n_train=len(y), threshold=.5)
    if not multiclass:
        head["classes"] = [0, 1]
    if len(classes) == 1:
        if multiclass:
            raise ValueError("Region training partition has only one class")
        head["constant_probabilities"] = [float(classes[0] == 0), float(classes[0] == 1)]
        head["status"] = "UNTRAINABLE_SINGLE_CLASS"
        return head
    if loss == "balanced_logistic":
        estimator = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED).fit(z, y)
        head.update(coef=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist(),
                    classes=estimator.classes_.tolist(), iterations=int(estimator.n_iter_.max()))
        np.testing.assert_allclose(predict_linear_head(head, x), estimator.predict_proba(z), atol=1e-10)
        return head
    a = torch.from_numpy(z)
    target = torch.from_numpy(y.astype(np.float64))
    coef = torch.zeros(a.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    weights = torch.ones_like(target)
    if loss == "balanced_bce":
        counts = np.bincount(y.astype(int), minlength=2)
        weights = torch.from_numpy(np.asarray([len(y) / (2 * counts[int(v)]) for v in y]))
    smooth_target = target * .95 + .025 if loss == "smooth_bce_005" else target
    optimizer = torch.optim.LBFGS([coef, bias], lr=1., max_iter=150, tolerance_grad=1e-9,
                                 tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        logits = a @ coef + bias
        per_row = torch.nn.functional.binary_cross_entropy_with_logits(logits, smooth_target, reduction="none")
        if loss == "focal_gamma2":
            pt = torch.exp(-per_row)
            per_row = (1 - pt).square() * per_row
        objective = (weights * per_row).mean() + coef.square().sum() / (2 * C * len(y))
        objective.backward()
        return objective

    optimizer.step(closure)
    head.update(coef=[coef.detach().tolist()], intercept=[float(bias.detach())], closure_calls=calls)
    if not np.isfinite(predict_linear_head(head, x)).all():
        raise ValueError("Nonfinite probabilities")
    return head


def binary_metrics(y, p, threshold=.5):
    y = np.asarray(y).astype(int)
    predicted = np.asarray(p) >= threshold
    tp, tn = int(((y == 1) & predicted).sum()), int(((y == 0) & ~predicted).sum())
    fp, fn = int(((y == 0) & predicted).sum()), int(((y == 1) & ~predicted).sum())
    return dict(n=len(y), positive=int(y.sum()), roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, p)) if y.sum() else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                sensitivity=tp/(tp+fn) if tp+fn else None, specificity=tn/(tn+fp) if tn+fp else None,
                tp=tp, tn=tn, fp=fp, fn=fn)


def choose_threshold(y, p):
    grid = np.linspace(.05, .95, 91)
    return float(grid[np.argmax([binary_metrics(y, p, t)["f1"] for t in grid])])


def group_ci(y, p, groups, threshold, count=1000):
    unique = np.unique(groups)
    blocks = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(SEED)
    values = {key: [] for key in ["roc_auc", "average_precision", "f1", "sensitivity", "specificity"]}
    for _ in range(count):
        index = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), size=len(blocks))])
        if len(np.unique(y[index])) < 2:
            continue
        threshold_sample = threshold[index] if isinstance(threshold, np.ndarray) else threshold
        m = binary_metrics(y[index], p[index], threshold_sample)
        for key in values:
            values[key].append(m[key])
    return {key: dict(low=float(np.quantile(v, .025)), high=float(np.quantile(v, .975))) for key, v in values.items()}


def checked_frame(labels, folds):
    frame = pd.read_csv(labels).merge(pd.read_csv(folds), on=["study", "image_uid"], validate="one_to_one")
    if len(frame) != len(pd.read_csv(labels)) or frame["fold"].isna().any():
        raise ValueError("Incomplete fold coverage")
    if set(frame.fold) != set(range(5)):
        raise ValueError("Expected original five folds")
    for column in ["study", "study_uid", "pixel_sha256", "image_uid"]:
        if frame.groupby(column).fold.nunique().max() != 1:
            raise ValueError(f"Leakage between folds: {column}")
    if frame.pixel_sha256.duplicated().any() or frame.image_uid.duplicated().any():
        raise ValueError("Duplicated images")
    return frame


def extract(frame, source, checkpoint, out, pause):
    identities = frame[["study", "image_uid", "path", "pixel_sha256"]].to_dict("records")
    key = dict(rows=identities, preprocessing=PREPROCESS, checkpoint_sha256=DENSENET_SHA256)
    feature_file, meta_file = out / "features.npy", out / "features_meta.json"
    if feature_file.exists() and meta_file.exists():
        saved = json.loads(meta_file.read_text())
        if saved["identity"] != key or saved["sha256"] != digest(feature_file):
            raise ValueError("Feature cache provenance mismatch")
        features = np.load(feature_file, allow_pickle=False)
        if features.shape != (len(frame), 1024) or not np.isfinite(features).all():
            raise ValueError("Feature cache shape/values invalid")
        print("Verified existing frozen features", flush=True)
        return features
    encoder = FrozenDenseNetEncoder(checkpoint, threads=2)
    chunks = []
    for start in range(0, len(frame), 4):
        images = []
        for row in frame.iloc[start:start+4].itertuples():
            path = source / row.path
            ds = pydicom.dcmread(path)
            a = ds.pixel_array
            h = hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes()).hexdigest()
            if h != row.pixel_sha256 or str(ds.SOPInstanceUID) != row.image_uid or str(ds.StudyInstanceUID) != row.study_uid:
                raise ValueError("DICOM does not match manifest")
            images.append(read_dicom(path).pixels)
        chunks.append(encoder.embed(images))
        if start % 24 == 0:
            print(f"Frozen features: {min(start+4, len(frame))}/{len(frame)}", flush=True)
        time.sleep(pause)
    features = np.concatenate(chunks)
    np.save(feature_file, features, allow_pickle=False)
    save(meta_file, dict(identity=key, sha256=digest(feature_file)))
    return features


def quality_suite(frame, x, anatomy, out):
    selected = ((frame.region == "spine") if anatomy == "spine" else (frame.region != "spine")) & frame.quality.notna()
    index = np.flatnonzero(selected)
    y, a = frame.iloc[index].quality.to_numpy(dtype=int), x[index]
    folds, groups = frame.iloc[index].fold.to_numpy(), frame.iloc[index].study.to_numpy()
    rows = frame.iloc[index][["study", "study_uid", "image_uid", "region", "quality", "fold"]].copy()
    summary = {}
    for loss in LOSSES:
        p = np.full(len(y), np.nan)
        fold_results = []
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            p[valid] = predict_linear_head(fit(a[train], y[train], loss), a[valid])[:, 1]
            fold_results.append(dict(fold=fold, **binary_metrics(y[valid], p[valid])))
            time.sleep(.05)
        rows[f"p_{loss}"] = p
        threshold = choose_threshold(y, p)
        summary[loss] = dict(mean_fold_ap=float(np.mean([r["average_precision"] for r in fold_results])),
                             folds=fold_results, pooled_fixed_threshold=binary_metrics(y, p),
                             deployment_threshold=threshold, tuned_threshold_optimistic=binary_metrics(y, p, threshold))
        print(anatomy, loss, "mean fold AP", summary[loss]["mean_fold_ap"], flush=True)
    winner = max(LOSSES, key=lambda loss: summary[loss]["mean_fold_ap"])
    final_head = fit(a, y, winner)
    final_head.update(threshold=summary[winner]["deployment_threshold"], selection_metric="mean_fold_ap",
                      threshold_origin="all development OOF; metrics at this threshold are optimistic",
                      anatomy=anatomy, positive_train=int(y.sum()))
    nested_p, nested_t = np.full(len(y), np.nan), np.full(len(y), np.nan)
    nested_folds = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        xt, yt, gt = a[train], y[train], groups[train]
        split = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED+fold).split(xt, yt, gt))
        inner = {}
        for loss in LOSSES:
            p = np.full(len(yt), np.nan)
            aps = []
            for ti, vi in split:
                assert set(gt[ti]).isdisjoint(gt[vi])
                p[vi] = predict_linear_head(fit(xt[ti], yt[ti], loss), xt[vi])[:, 1]
                aps.append(binary_metrics(yt[vi], p[vi])["average_precision"])
            inner[loss] = dict(mean_ap=float(np.mean(aps)), threshold=choose_threshold(yt, p))
        choice = max(LOSSES, key=lambda loss: inner[loss]["mean_ap"])
        threshold = inner[choice]["threshold"]
        nested_p[valid] = predict_linear_head(fit(xt, yt, choice), a[valid])[:, 1]
        nested_t[valid] = threshold
        nested_folds.append(dict(fold=fold, selected_loss=choice, threshold=threshold, inner=inner,
                                 metrics=binary_metrics(y[valid], nested_p[valid], threshold)))
        print(anatomy, "nested outer", fold, "selected", choice, "threshold", threshold, flush=True)
        time.sleep(.15)
    rows["p_nested_selected"] = nested_p
    rows["threshold_nested_selected"] = nested_t
    rows.to_csv(out / f"{anatomy}_oof.csv", index=False)
    result = dict(anatomy=anatomy, n=len(y), positive=int(y.sum()), candidates=summary, final_selected_loss=winner,
                  nested_selected=dict(folds=nested_folds, pooled=binary_metrics(y, nested_p, nested_t),
                                       ci95=group_ci(y, nested_p, groups, nested_t)))
    save(out / f"{anatomy}_results.json", result)
    return final_head, result


def fixed_heads(frame, x, out):
    heads, reports = {}, {}
    oof = frame[["study", "study_uid", "image_uid", "region", "fold"]].copy()
    folds = frame.fold.to_numpy()
    region_y = frame.region.to_numpy()
    predicted = np.empty(len(frame), dtype=object)
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        head = fit(x[train], region_y[train], multiclass=True)
        predicted[valid] = np.asarray(head["classes"])[predict_linear_head(head, x[valid]).argmax(axis=1)]
    oof["region_prediction"] = predicted
    heads["region"] = fit(x, region_y, multiclass=True)
    reports["region"] = dict(n=len(frame), accuracy=float(accuracy_score(region_y, predicted)),
                             macro_f1=float(f1_score(region_y, predicted, average="macro")))
    for key, (anatomy, column) in REASONS.items():
        region_mask = frame.region == "spine" if anatomy == "spine" else frame.region != "spine"
        valid_label = region_mask & frame[column].notna()
        index = np.flatnonzero(valid_label)
        y, a, f = frame.iloc[index][column].to_numpy(dtype=int), x[index], folds[index]
        p = np.full(len(index), np.nan)
        for fold in range(5):
            train, valid = f != fold, f == fold
            p[valid] = predict_linear_head(fit(a[train], y[train]), a[valid])[:, 1]
        threshold = choose_threshold(y, p)
        heads[key] = fit(a, y)
        heads[key].update(threshold=threshold, positive_train=int(y.sum()), anatomy=anatomy,
                          threshold_origin="all development OOF; metrics at this threshold are optimistic")
        oof[f"p_{key}"] = np.nan
        oof.loc[index, f"p_{key}"] = p
        reports[key] = dict(pooled_fixed_threshold=binary_metrics(y, p), deployment_threshold=threshold,
                            tuned_threshold_optimistic=binary_metrics(y, p, threshold))
    oof.to_csv(out / "auxiliary_oof.csv", index=False)
    save(out / "auxiliary_results.json", reports)
    return heads, reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/release_loss_v1")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/interim/image_labels.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "data/interim/folds.csv")
    parser.add_argument("--source", type=Path, default=ROOT / "data/interim/train/Исследования")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/torch_home/hub/checkpoints/densenet121-a639ec97.pth")
    parser.add_argument("--batch-pause", type=float, default=.75)
    args = parser.parse_args()
    if args.batch_pause < 0:
        raise ValueError("batch pause cannot be negative")
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "final_heads.json").exists():
        raise ValueError("Completed experiment exists; choose a new output directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    protocol = dict(name="release_loss_v1", seed=SEED, losses=LOSSES, C=C,
                    backbone="frozen ImageNet DenseNet121; no trainable backbone parameters", preprocessing=PREPROCESS,
                    feature_checkpoint_sha256=DENSENET_SHA256, labels_sha256=digest(args.labels), folds_sha256=digest(args.folds),
                    optimizer="Torch float64 full-batch LBFGS maxiter150; mean loss + ||w||²/(2*C*n); unpenalized intercept",
                    focal="unweighted gamma2", smooth="target = .95*y+.025", balanced="inverse training prevalence; mean weight1",
                    selection="highest arithmetic mean fold AP; order of losses breaks ties; final selection on original five folds",
                    nested="5 original study outer folds; fresh 3 study inner folds select loss and F1 threshold; no outer-row training",
                    threshold="grid .05..95 step .01 maximizing inner OOF F1; smallest threshold breaks ties",
                    auxiliary="fixed balanced logistic C=.001 for region and reasons; labels missing stay masked",
                    limits="CPU2threads interop1; no GPU; batches4; pause between extraction batches",
                    limitations=["Development CV; historical holdout now included, no independent test",
                                 "Folds locally reconstructed, not confirmed team canonical", "Patient linkage between studies unavailable",
                                 "Nested CI conditional on this dataset and previously explored architecture; no site generalization evidence",
                                 "Quality/reason disagreements retained; binary and reason metrics independent"])
    protocol_file = args.out / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("Existing experiment protocol differs")
    save(protocol_file, protocol)
    with threadpool_limits(limits=2):
        frame = checked_frame(args.labels, args.folds)
        x = extract(frame, args.source, args.checkpoint, args.out, args.batch_pause)
        heads, auxiliary = fixed_heads(frame, x, args.out)
        results = {}
        for anatomy in ["spine", "hip"]:
            heads[f"{anatomy}_quality"], results[anatomy] = quality_suite(frame, x, anatomy, args.out)
        bundle = dict(schema_version=1, backbone="densenet121", checkpoint_sha256=DENSENET_SHA256,
                      preprocess=PREPROCESS, feature_dimension=1024, features_sha256=digest(args.out / "features.npy"),
                      labels_sha256=digest(args.labels), folds_sha256=digest(args.folds), heads=heads,
                      deployment_note="final heads refit on all available development labels; never use their training predictions as validation")
        save(args.out / "final_heads.json", bundle)
        restored = json.loads((args.out / "final_heads.json").read_text())
        for key, head in heads.items():
            np.testing.assert_array_equal(predict_linear_head(head, x), predict_linear_head(restored["heads"][key], x))
        run = dict(status="COMPLETE", elapsed_seconds=time.perf_counter()-started, n_images=len(frame),
                   n_quality_labels=int(frame.quality.notna().sum()), auxiliary=auxiliary, quality=results,
                   versions={p: importlib.metadata.version(p) for p in ["torch", "torchvision", "scikit-learn", "numpy", "pydicom"]},
                   final_heads_sha256=digest(args.out / "final_heads.json"), source_sha256=digest(__file__),
                   features_source_sha256=digest(ROOT / "src/dxaqc/release_features.py"), reload_predictions_identical=True)
        save(args.out / "run.json", run)
        print("COMPLETE", json.dumps({a: results[a]["nested_selected"]["pooled"] for a in results}), flush=True)


if __name__ == "__main__":
    main()
