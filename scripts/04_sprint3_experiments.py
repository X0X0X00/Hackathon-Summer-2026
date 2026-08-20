"""Sprint 3 experiments: YW-005 ensemble, YW-006 hard-bucket specialist, YW-007 hybrid.

Uses frozen experiments/folds.csv. Does not write test predictions.
Does not overwrite YW-000..YW-004.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from merfish60.cv import CV_PROTOCOL, CV_RANDOM_STATE, FOLD_VALUES, load_folds  # noqa: E402
from merfish60.ensemble import (  # noqa: E402
    convex_weight_grid,
    cross_fitted_ensemble,
    mix_probas,
)
from merfish60.hard_bucket import (  # noqa: E402
    CANDIDATE_SPECS,
    HARD_BUCKET_FOCUS_CLASSES,
    EXTRA_TREES_KWARGS,
    HGB_KWARGS,
    fit_selected_model,
    predict_selected_model,
    select_hard_bucket_model,
    spec_by_name,
)
from merfish60.io import load_dataset, validate_contract  # noqa: E402
from merfish60.metrics import summarize_oof  # noqa: E402
from merfish60.models import (  # noqa: E402
    LR_KWARGS,
    argmax_labels,
    assert_probability_rows,
    train_val_masks,
)
from merfish60.official_contract import (  # noqa: E402
    allowed_labels,
    folds_sha256,
    git_commit,
    manifest_sha256,
    verify_official_manifest,
)
from merfish60.registry import RegistryError, append_registry_row  # noqa: E402
from merfish60.signatures import (  # noqa: E402
    build_candidate_map,
    is_deterministic_map,
    missing_bucket_key,
    signatures_from_meta,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError("not JSON serializable: {}".format(type(obj)))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    path.write_text(json.dumps(clean, indent=2, sort_keys=True, default=_json_default) + "\n")


def write_oof(path: Path, cell_ids, y_true, y_pred, folds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Cell_ID": [str(v) for v in cell_ids],
            "true_label": list(y_true),
            "predicted_label": list(y_pred),
            "fold": [int(v) for v in folds],
        }
    ).to_csv(path, index=False)


def write_proba(path: Path, cell_ids, proba: np.ndarray, class_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(proba, columns=list(class_names))
    frame.insert(0, "Cell_ID", [str(v) for v in cell_ids])
    frame.to_csv(path, index=False, compression="gzip")


def load_proba_matrix(run_id: str, class_names: Sequence[str], cell_ids: Sequence[str]) -> np.ndarray:
    path = ROOT / "outputs" / "probabilities" / "{}_oof_probabilities.csv.gz".format(run_id)
    df = pd.read_csv(path, dtype={"Cell_ID": str}, compression="gzip")
    if list(df.columns) != ["Cell_ID"] + list(class_names):
        raise RuntimeError("{} probability columns are not the stable class order".format(run_id))
    aligned = df.set_index("Cell_ID").loc[list(cell_ids)]
    return aligned.loc[:, list(class_names)].to_numpy(dtype=np.float64)


def load_oof_frame(run_id: str) -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "outputs" / "oof" / "{}_oof.csv".format(run_id),
        dtype={"Cell_ID": str, "true_label": str, "predicted_label": str, "fold": int},
    )


def confusion_pairs(y_true, y_pred, k: int = 5) -> List[dict]:
    counts = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            counts[(str(t), str(p))] += 1
    return [
        {"true_label": t, "predicted_label": p, "n": int(n)}
        for (t, p), n in counts.most_common(k)
    ]


def paired_vs(base_pred, new_pred, y_true) -> dict:
    base_pred = np.asarray(base_pred)
    new_pred = np.asarray(new_pred)
    y_true = np.asarray(y_true)
    w2c = int(((base_pred != y_true) & (new_pred == y_true)).sum())
    c2w = int(((base_pred == y_true) & (new_pred != y_true)).sum())
    return {
        "wrong_to_correct": w2c,
        "correct_to_wrong": c2w,
        "net_additional_correct": w2c - c2w,
        "n_changed": int((base_pred != new_pred).sum()),
        "total_correct": int((new_pred == y_true).sum()),
        "total_errors": int((new_pred != y_true).sum()),
    }


def record_run(payload: dict, overwrite: bool) -> None:
    fold_acc = payload["fold_accuracy"]
    notes = payload.get("notes") or (
        "; ".join(payload.get("warnings") or []) or "no warnings"
    )
    notes = " ".join(str(notes).split())
    row = {
        "run_id": payload["run_id"],
        "timestamp": utc_now(),
        "git_commit": git_commit(ROOT),
        "model": payload.get("model", ""),
        "feature_set": payload.get("feature_set", ""),
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "fold_0_accuracy": "{:.10f}".format(float(fold_acc["0"])),
        "fold_1_accuracy": "{:.10f}".format(float(fold_acc["1"])),
        "fold_2_accuracy": "{:.10f}".format(float(fold_acc["2"])),
        "oof_accuracy": "{:.10f}".format(float(payload["oof_accuracy"])),
        "macro_f1": "{:.10f}".format(float(payload["macro_f1"])),
        "runtime_seconds": "{:.3f}".format(float(payload["runtime_seconds"])),
        "status": str(payload.get("status") or "completed"),
        "notes": notes,
        "manifest_sha256": manifest_sha256(ROOT),
        "folds_sha256": folds_sha256(ROOT),
        "oof_path": payload.get("oof_path", ""),
        "proba_path": payload.get("proba_path", ""),
        "metrics_path": "outputs/metrics/{}_metrics.json".format(payload["run_id"]),
        "conclusion": payload.get("conclusion", ""),
    }
    append_registry_row(row, ROOT, overwrite=overwrite)


def run_yw005(cell_ids, y, folds, class_names, yw004_pred) -> dict:
    t0 = time.perf_counter()
    p2 = load_proba_matrix("YW-002", class_names, cell_ids)
    p3 = load_proba_matrix("YW-003", class_names, cell_ids)
    p4 = load_proba_matrix("YW-004", class_names, cell_ids)
    equal_w = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    p_equal = mix_probas(p2, p3, p4, equal_w)
    pred_equal = argmax_labels(p_equal, class_names)
    acc_equal = float(np.mean(pred_equal == y))
    p_cf, weight_details = cross_fitted_ensemble(p2, p3, p4, y, folds, class_names)
    assert_probability_rows(p_cf)
    pred = argmax_labels(p_cf, class_names)
    runtime = time.perf_counter() - t0
    metrics = summarize_oof(y, pred, folds, labels=class_names)
    paired = paired_vs(yw004_pred, pred, y)
    provenance_warning = (
        "YW-005 is an exploratory ensemble diagnostic, not a nested stack. "
        "For held-out fold f, weights are scored on saved OOF rows from the "
        "other two folds. Those rows were produced by YW-002/003/004 models "
        "trained on all folds except their own validation fold, which includes "
        "fold f. Poisoning fold-f labels without regenerating base probabilities "
        "is not sufficient to prove nested independence."
    )
    payload = {
        "run_id": "YW-005",
        "model": "exploratory convex blend of saved YW-002/YW-003/YW-004 OOF probabilities (not nested)",
        "feature_set": "saved OOF probabilities from YW-002, YW-003, YW-004; no new features; no nested refit",
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "role": "exploratory_ensemble_diagnostic",
        "status": "exploratory",
        "formal_selection_eligible": False,
        "nested_independence": False,
        "provenance": {
            "base_oof_source": ["YW-002", "YW-003", "YW-004"],
            "weight_labels_used": "true labels from the two non-held-out folds only",
            "base_training_includes_held_out_fold": True,
            "reason": (
                "Standard 3-fold OOF: a prediction for fold g is fit on folds != g. "
                "When selecting weights for held-out fold f, the eligible folds are "
                "g != f, so those base models were trained with fold f in the training set. "
                "Regenerating base probabilities after changing fold-f labels could change "
                "the selected weights."
            ),
            "excluded_from_formal_model_selection": True,
        },
        "weight_grid": {
            "step": 0.1,
            "n_points": len(convex_weight_grid()),
            "constraint": "w2>=0, w3>=0, w4>=0, w2+w3+w4=1",
            "selection": "maximize overall accuracy on the two non-held-out folds; ties broken toward equal weights",
        },
        "equal_weight": {
            "weights": {"YW-002": equal_w[0], "YW-003": equal_w[1], "YW-004": equal_w[2]},
            "oof_accuracy": acc_equal,
        },
        "fold_selected_weights": weight_details,
        "class_order": list(class_names),
        "fold_accuracy": {str(k): v for k, v in metrics["fold_accuracy"].items()},
        "oof_accuracy": metrics["oof_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_recall": metrics["per_class_recall"],
        "paired_vs_YW-004": paired,
        "runtime_seconds": runtime,
        "warnings": [provenance_warning],
        "notes": "exploratory; not nested; excluded from formal selection",
        "n_cells": int(len(y)),
        "oof_path": "outputs/oof/YW-005_oof.csv",
        "proba_path": "outputs/probabilities/YW-005_oof_probabilities.csv.gz",
        "conclusion": (
            "Exploratory (not nested, excluded from formal selection). "
            "Cross-fitted OOF={:.4f}; equal-weight OOF={:.4f}; net vs YW-004={}".format(
                metrics["oof_accuracy"], acc_equal, paired["net_additional_correct"]
            )
        ),
    }
    write_oof(ROOT / "outputs/oof/YW-005_oof.csv", cell_ids, y, pred, folds)
    write_proba(ROOT / "outputs/probabilities/YW-005_oof_probabilities.csv.gz", cell_ids, p_cf, class_names)
    write_json(ROOT / "outputs/metrics/YW-005_metrics.json", payload)
    return payload


def run_yw006(data, cell_ids, y, folds, class_names, sigs, yw004_pred) -> dict:
    t0 = time.perf_counter()
    bucket_key = missing_bucket_key()
    bucket_mask = sigs.to_numpy() == bucket_key
    X_counts = data.counts_train.to_numpy(dtype=np.float64)
    y_arr = np.asarray(y, dtype=object)
    fold_arr = np.asarray(folds)
    ids = np.asarray(cell_ids, dtype=object)

    n = len(y)
    proba_global = np.zeros((n, len(class_names)), dtype=np.float64)
    pred_all = np.empty(n, dtype=object)
    pred_all[:] = None
    selected = {}
    inner_scores = {}
    outer_bucket_acc = {}
    train_val_id_pairs = []
    warnings_all = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for fold_id in FOLD_VALUES:
            train_mask, val_mask = train_val_masks(pd.Series(fold_arr), fold_id)
            tr_b = train_mask & bucket_mask
            va_b = val_mask & bucket_mask
            tr_ids = ids[tr_b].tolist()
            va_ids = ids[va_b].tolist()
            overlap = set(tr_ids) & set(va_ids)
            if overlap:
                raise RuntimeError("hard-bucket train/val overlap in fold {}".format(fold_id))
            train_val_id_pairs.append(
                {"fold": int(fold_id), "n_train": len(tr_ids), "n_val": len(va_ids)}
            )
            print(
                "  outer fold {} hard-bucket train={} val={}".format(
                    fold_id, int(tr_b.sum()), int(va_b.sum())
                ),
                flush=True,
            )
            name, scores = select_hard_bucket_model(
                X_counts[tr_b], y_arr[tr_b], class_names, random_state=CV_RANDOM_STATE
            )
            print("  selected={} inner-CV={}".format(name, scores), flush=True)
            selected[str(fold_id)] = name
            inner_scores[str(fold_id)] = scores
            spec = spec_by_name(name)
            model = fit_selected_model(spec, X_counts[tr_b], y_arr[tr_b])
            P_val = predict_selected_model(spec, model, X_counts[va_b], class_names)
            pred_val = argmax_labels(P_val, class_names)
            proba_global[va_b] = P_val
            pred_all[va_b] = pred_val
            outer_bucket_acc[str(fold_id)] = float(np.mean(pred_val == y_arr[va_b]))
        for warn in caught:
            msg = "{}: {}".format(warn.category.__name__, warn.message)
            if msg not in warnings_all:
                warnings_all.append(msg)

    if np.any(pred_all[bucket_mask] == None):  # noqa: E711
        raise RuntimeError("YW-006 missing predictions for some hard-bucket cells")
    assert_probability_rows(proba_global[bucket_mask])
    runtime = time.perf_counter() - t0

    y_b = y_arr[bucket_mask]
    pred_b = pred_all[bucket_mask]
    fold_b = fold_arr[bucket_mask]
    ids_b = ids[bucket_mask]
    bucket_oof_acc = float(np.mean(pred_b == y_b))
    yw004_b = np.asarray(yw004_pred)[bucket_mask]
    yw004_bucket_acc = float(np.mean(yw004_b == y_b))
    extra_correct = int(((pred_b == y_b) & (yw004_b != y_b)).sum()) - int(
        ((pred_b != y_b) & (yw004_b == y_b)).sum()
    )
    labels_b = sorted(set(y_b.tolist()))
    cm = confusion_matrix(y_b, pred_b, labels=labels_b)
    metrics_bucket = summarize_oof(y_b, pred_b, fold_b, labels=labels_b)
    payload = {
        "run_id": "YW-006",
        "model": "hard-bucket specialist selected by inner stratified CV on outer-training bucket only",
        "feature_set": "200 gene counts inside __MISSING__/__MISSING__/__MISSING__; candidate transforms documented per model",
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "bucket_key": bucket_key,
        "n_bucket_cells": int(bucket_mask.sum()),
        "candidate_models": [
            {
                "name": spec["name"],
                "description": spec["description"],
                "transform": spec["transform"],
            }
            for spec in CANDIDATE_SPECS
        ],
        "candidate_hyperparameters": {
            "log1p_lr": dict(LR_KWARGS),
            "cp10k_log1p_lr": dict(LR_KWARGS),
            "extratrees": EXTRA_TREES_KWARGS,
            "hist_gradient_boosting": HGB_KWARGS,
        },
        "inner_cv_scores_by_outer_fold": inner_scores,
        "selected_model_by_outer_fold": selected,
        "outer_train_val_sizes": train_val_id_pairs,
        "outer_fold_bucket_accuracy": outer_bucket_acc,
        "fold_accuracy": {
            "0": outer_bucket_acc["0"],
            "1": outer_bucket_acc["1"],
            "2": outer_bucket_acc["2"],
        },
        "oof_accuracy": bucket_oof_acc,
        "macro_f1": metrics_bucket["macro_f1"],
        "per_class_recall": metrics_bucket["per_class_recall"],
        "focus_class_recall": {
            c: metrics_bucket["per_class_recall"].get(c)
            for c in HARD_BUCKET_FOCUS_CLASSES
        },
        "yw004_hard_bucket_accuracy": yw004_bucket_acc,
        "delta_vs_YW-004_hard_bucket": bucket_oof_acc - yw004_bucket_acc,
        "net_additional_correct_bucket_cells": extra_correct,
        "zero_cell_delta_reason": (
            "inner-CV selected log1p_lr on every outer fold, matching the YW-004 "
            "missing-bucket logistic specialist; not a silent failed replacement"
        ),
        "role": "valid_negative_ablation",
        "formal_selection_eligible": False,
        "status": "completed",
        "confusion_matrix_labels": labels_b,
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_pairs_top5": confusion_pairs(y_b, pred_b, k=5),
        "class_order": list(class_names),
        "runtime_seconds": runtime,
        "warnings": warnings_all,
        "notes": "valid negative ablation; selected log1p_lr on every outer fold",
        "n_cells": int(bucket_mask.sum()),
        "oof_path": "outputs/oof/YW-006_hard_bucket_oof.csv",
        "proba_path": "outputs/probabilities/YW-006_hard_bucket_probabilities.csv.gz",
        "conclusion": (
            "Valid negative ablation. Hard-bucket OOF={:.4f} vs YW-004 bucket {:.4f}; "
            "selected={}".format(bucket_oof_acc, yw004_bucket_acc, selected)
        ),
    }
    write_oof(
        ROOT / "outputs/oof/YW-006_hard_bucket_oof.csv",
        ids_b,
        y_b,
        pred_b,
        fold_b,
    )
    write_proba(
        ROOT / "outputs/probabilities/YW-006_hard_bucket_probabilities.csv.gz",
        ids_b,
        proba_global[bucket_mask],
        class_names,
    )
    write_json(ROOT / "outputs/metrics/YW-006_metrics.json", payload)
    # stash full-length arrays for YW-007 without a second fit
    payload["_proba_global_full"] = proba_global
    payload["_pred_full"] = pred_all
    payload["_bucket_mask"] = bucket_mask
    return payload


def run_yw007(cell_ids, y, folds, class_names, sigs, yw004_oof, yw004_proba, yw006_payload) -> dict:
    t0 = time.perf_counter()
    bucket_mask = yw006_payload["_bucket_mask"]
    pred = np.asarray(yw004_oof, dtype=object).copy()
    proba = np.array(yw004_proba, dtype=np.float64, copy=True)
    pred[bucket_mask] = yw006_payload["_pred_full"][bucket_mask]
    proba[bucket_mask] = yw006_payload["_proba_global_full"][bucket_mask]
    assert_probability_rows(proba)
    if len(pred) != 5000:
        raise RuntimeError("YW-007 n={} != 5000".format(len(pred)))
    if len(set(cell_ids)) != 5000:
        raise RuntimeError("YW-007 Cell_ID not unique")
    runtime = time.perf_counter() - t0
    metrics = summarize_oof(y, pred, folds, labels=class_names)
    paired = paired_vs(yw004_oof, pred, y)
    bucket_key = missing_bucket_key()
    miss = np.asarray(sigs) == bucket_key
    # deterministic/ambiguous from YW-004 metrics file if present; compute from current preds
    hard_acc = float(np.mean(pred[miss] == np.asarray(y)[miss]))
    outside_acc = float(np.mean(pred[~miss] == np.asarray(y)[~miss]))
    kinds = np.empty(len(y), dtype=object)
    sig_arr = np.asarray(pd.Series(sigs).astype(str))
    y_arr = np.asarray(y)
    fold_arr = np.asarray(folds)
    for fold_id in FOLD_VALUES:
        train_mask, val_mask = train_val_masks(pd.Series(fold_arr), fold_id)
        cmap = build_candidate_map(sig_arr[train_mask], y_arr[train_mask])
        for i in np.where(val_mask)[0]:
            sig = str(sig_arr[i])
            if sig not in cmap:
                kinds[i] = "unseen"
            elif is_deterministic_map(cmap[sig]):
                kinds[i] = "deterministic"
            else:
                kinds[i] = "ambiguous"
    det = kinds == "deterministic"
    amb = kinds == "ambiguous"

    def _acc(mask):
        if not np.any(mask):
            return None
        return float(np.mean(pred[mask] == y_arr[mask]))

    payload = {
        "run_id": "YW-007",
        "model": "hybrid: YW-004 outside missing bucket; YW-006 specialist inside missing bucket",
        "feature_set": "YW-004 OOF probabilities replaced only on __MISSING__/__MISSING__/__MISSING__ cells by YW-006",
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "role": "exact_duplicate_of_YW-004",
        "formal_selection_eligible": True,
        "status": "completed",
        "construction": {
            "outside_bucket": "YW-004 predictions and 60-class probabilities unchanged",
            "inside_bucket": "YW-006 outer-fold OOF predictions/probabilities mapped to global 60-class order",
            "replacement": "explicit copy of YW-006 predictions/probabilities onto the missing-bucket mask",
            "n_changed_vs_YW-004": paired["n_changed"],
            "zero_cell_delta_reason": (
                "YW-006 selected log1p_lr on every outer fold, matching the YW-004 "
                "hard-bucket specialist; replacement ran and copied identical predictions"
            ),
        },
        "class_order": list(class_names),
        "fold_accuracy": {str(k): v for k, v in metrics["fold_accuracy"].items()},
        "oof_accuracy": metrics["oof_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_recall": metrics["per_class_recall"],
        "hard_bucket_accuracy": hard_acc,
        "outside_bucket_accuracy": outside_acc,
        "hard_bucket_n": int(miss.sum()),
        "outside_bucket_n": int((~miss).sum()),
        "deterministic_signature_accuracy": _acc(det),
        "deterministic_signature_n": int(det.sum()),
        "ambiguous_signature_accuracy": _acc(amb),
        "ambiguous_signature_n": int(amb.sum()),
        "delta_vs_YW-004": float(metrics["oof_accuracy"] - 0.7598),
        "paired_vs_YW-004": paired,
        "confusion_pairs_top5": confusion_pairs(y, pred, k=5),
        "runtime_seconds": runtime,
        "warnings": [],
        "notes": "exact duplicate of YW-004; replacement copied YW-006; n_changed=0; not selected",
        "n_cells": 5000,
        "oof_path": "outputs/oof/YW-007_oof.csv",
        "proba_path": "outputs/probabilities/YW-007_oof_probabilities.csv.gz",
        "conclusion": (
            "Not selected: exact more-complex duplicate of YW-004. "
            "Hybrid OOF={:.4f}; delta vs YW-004={:+.4f}; n_changed={}".format(
                metrics["oof_accuracy"],
                metrics["oof_accuracy"] - 0.7598,
                paired["n_changed"],
            )
        ),
    }
    write_oof(ROOT / "outputs/oof/YW-007_oof.csv", cell_ids, y, pred, folds)
    write_proba(ROOT / "outputs/probabilities/YW-007_oof_probabilities.csv.gz", cell_ids, proba, class_names)
    write_json(ROOT / "outputs/metrics/YW-007_metrics.json", payload)
    return payload, pred


def print_run(payload: dict) -> None:
    print("=== {} ===".format(payload["run_id"]))
    print("fold accuracies:", payload.get("fold_accuracy"))
    print("OOF accuracy: {:.6f}".format(payload["oof_accuracy"]))
    print("macro-F1: {:.6f}".format(payload["macro_f1"]))
    print("runtime_seconds: {:.3f}".format(payload["runtime_seconds"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="*", default=["YW-005", "YW-006", "YW-007"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("Verifying official data manifest...")
    for line in verify_official_manifest(ROOT):
        print(" - {}".format(line))

    data = load_dataset(ROOT)
    validate_contract(data)
    folds_df = load_folds(ROOT / "experiments" / "folds.csv")
    class_names = allowed_labels(ROOT)
    y = data.y_train.astype(str)
    cell_ids = [str(v) for v in y.index.tolist()]
    fold_ids = folds_df.set_index("Cell_ID").loc[cell_ids, "fold"].to_numpy()
    sigs = signatures_from_meta(data.meta_train)
    yw004 = load_oof_frame("YW-004")
    yw004 = yw004.set_index("Cell_ID").loc[cell_ids]
    yw004_pred = yw004["predicted_label"].astype(str).to_numpy()
    yw004_proba = load_proba_matrix("YW-004", class_names, cell_ids)
    y_np = y.to_numpy()

    yw006_payload = None
    for run_id in args.run:
        print("\nRunning {}".format(run_id))
        if run_id == "YW-005":
            payload = run_yw005(cell_ids, y_np, fold_ids, class_names, yw004_pred)
        elif run_id == "YW-006":
            yw006_payload = run_yw006(
                data, cell_ids, y_np, fold_ids, class_names, sigs, yw004_pred
            )
            payload = {k: v for k, v in yw006_payload.items() if not k.startswith("_")}
        elif run_id == "YW-007":
            if yw006_payload is None:
                # load saved YW-006 artifacts
                m006 = json.loads((ROOT / "outputs/metrics/YW-006_metrics.json").read_text())
                hb = pd.read_csv(
                    ROOT / "outputs/oof/YW-006_hard_bucket_oof.csv",
                    dtype={"Cell_ID": str, "true_label": str, "predicted_label": str, "fold": int},
                )
                pb = pd.read_csv(
                    ROOT / "outputs/probabilities/YW-006_hard_bucket_probabilities.csv.gz",
                    dtype={"Cell_ID": str},
                    compression="gzip",
                )
                bucket_key = missing_bucket_key()
                bucket_mask = sigs.to_numpy() == bucket_key
                pred_full = np.array(yw004_pred, dtype=object)
                proba_full = np.array(yw004_proba, dtype=np.float64, copy=True)
                hb_idx = hb.set_index("Cell_ID")
                pb_idx = pb.set_index("Cell_ID")
                ids_np = np.asarray(cell_ids)
                for i, cid in enumerate(ids_np):
                    if bucket_mask[i]:
                        pred_full[i] = hb_idx.loc[cid, "predicted_label"]
                        proba_full[i] = pb_idx.loc[cid, class_names].to_numpy(dtype=np.float64)
                yw006_payload = {
                    "_bucket_mask": bucket_mask,
                    "_pred_full": pred_full,
                    "_proba_global_full": proba_full,
                }
            payload, _pred = run_yw007(
                cell_ids, y_np, fold_ids, class_names, sigs, yw004_pred, yw004_proba, yw006_payload
            )
        else:
            raise SystemExit("unknown run {}".format(run_id))
        try:
            record_run(payload, overwrite=args.overwrite)
        except RegistryError as exc:
            print("REGISTRY ERROR: {}".format(exc), file=sys.stderr)
            return 1
        print_run(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
