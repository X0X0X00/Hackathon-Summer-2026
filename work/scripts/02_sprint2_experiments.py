"""Run leakage-safe Sprint 2 experiments YW-002, YW-003, and YW-004.

Loads experiments/folds.csv; never regenerates folds. Does not write
prediction/prediction.csv or fit on test data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

_WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORK / "src"))
from merfish60.io import repo_root  # noqa: E402
ROOT = repo_root()

from merfish60.cv import CV_PROTOCOL, CV_RANDOM_STATE, FOLD_VALUES, load_folds  # noqa: E402
from merfish60.io import load_dataset, validate_contract  # noqa: E402
from merfish60.metrics import summarize_oof  # noqa: E402
from merfish60.models import (  # noqa: E402
    LR_KWARGS,
    align_predict_proba,
    argmax_labels,
    assert_probability_rows,
    canonicalize_metadata_frame,
    fit_onehot_encoder,
    log1p_counts,
    make_logistic_regression,
    one_hot_class,
    train_val_masks,
    transform_onehot,
)
from merfish60.official_contract import (  # noqa: E402
    YW002_FORBIDDEN_FIELDS,
    YW002_METADATA_FIELDS,
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
    mask_and_renormalize,
    missing_bucket_key,
    signatures_from_meta,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_oof(path: Path, cell_ids: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str], folds: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Cell_ID": [str(v) for v in cell_ids],
            "true_label": list(y_true),
            "predicted_label": list(y_pred),
            "fold": [int(v) for v in folds],
        }
    ).to_csv(path, index=False)


def write_proba(path: Path, cell_ids: Sequence[str], proba: np.ndarray, class_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(proba, columns=list(class_names))
    frame.insert(0, "Cell_ID", [str(v) for v in cell_ids])
    frame.to_csv(path, index=False, compression="gzip")


def per_signature_accuracy(
    signatures: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str]
) -> Dict[str, dict]:
    grouped: Dict[str, list] = defaultdict(list)
    for sig, t, p in zip(signatures, y_true, y_pred):
        grouped[str(sig)].append(int(t == p))
    out = {}
    for sig, flags in sorted(grouped.items()):
        out[sig] = {
            "n": int(len(flags)),
            "accuracy": float(np.mean(flags)) if flags else None,
        }
    return out


def confusion_pairs(y_true: Sequence[str], y_pred: Sequence[str], k: int = 5) -> List[dict]:
    counts = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            counts[(str(t), str(p))] += 1
    top = []
    for (t, p), n in counts.most_common(k):
        top.append({"true_label": t, "predicted_label": p, "n": int(n)})
    return top


def fit_global_gene_model(X_log: np.ndarray, y: np.ndarray, train_mask: np.ndarray):
    model = make_logistic_regression()
    caught = []
    with warnings.catch_warnings(record=True) as caught_list:
        warnings.simplefilter("always")
        model.fit(X_log[train_mask], y[train_mask])
        for item in caught_list:
            caught.append("{}: {}".format(item.category.__name__, str(item.message)))
    n_iter = int(np.max(model.n_iter_)) if np.ndim(model.n_iter_) else int(model.n_iter_)
    return model, caught, n_iter


def run_yw002(data, folds: pd.DataFrame, class_names: List[str]) -> dict:
    t0 = time.perf_counter()
    y = data.y_train.astype(str)
    cell_ids = y.index.astype(str)
    fold_ids = folds.set_index("Cell_ID").loc[cell_ids, "fold"]
    X_log = log1p_counts(data.counts_train)
    meta_cat = canonicalize_metadata_frame(data.meta_train, YW002_METADATA_FIELDS)
    used = set(YW002_METADATA_FIELDS)
    forbidden_present = [c for c in YW002_FORBIDDEN_FIELDS if c in used]
    if forbidden_present:
        raise RuntimeError("YW-002 used forbidden fields: {}".format(forbidden_present))

    n = len(y)
    proba = np.zeros((n, len(class_names)), dtype=np.float64)
    pred = np.empty(n, dtype=object)
    warnings_all: List[str] = []
    n_iter = {}
    encoded_dim = {}
    for fold_id in FOLD_VALUES:
        train_mask, val_mask = train_val_masks(fold_ids, fold_id)
        encoder = fit_onehot_encoder(meta_cat.to_numpy()[train_mask])
        X_train = np.hstack(
            [X_log[train_mask], transform_onehot(encoder, meta_cat.to_numpy()[train_mask])]
        )
        X_val = np.hstack(
            [X_log[val_mask], transform_onehot(encoder, meta_cat.to_numpy()[val_mask])]
        )
        encoded_dim[str(fold_id)] = {
            "n_gene_features": int(X_log.shape[1]),
            "n_meta_features": int(X_train.shape[1] - X_log.shape[1]),
            "n_total_features": int(X_train.shape[1]),
        }
        model = make_logistic_regression()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, y.to_numpy()[train_mask])
            for item in caught:
                warnings_all.append("fold_{}: {}: {}".format(fold_id, item.category.__name__, item.message))
        n_iter[str(fold_id)] = int(np.max(model.n_iter_)) if np.ndim(model.n_iter_) else int(model.n_iter_)
        P = align_predict_proba(model, X_val, class_names)
        proba[val_mask] = P
        pred[val_mask] = argmax_labels(P, class_names)
    runtime = time.perf_counter() - t0
    assert_probability_rows(proba)
    metrics = summarize_oof(y.to_numpy(), pred, fold_ids.to_numpy(), labels=class_names)
    sigs = signatures_from_meta(data.meta_train)
    payload = {
        "run_id": "YW-002",
        "model": "LogisticRegression_l2_lbfgs",
        "feature_set": "log1p(raw gene counts, 200) + fold-safe one-hot Region/Excitatory_vs_Inhibitory/Segment",
        "preprocessing": {
            "genes": "log1p(raw counts); no learned parameters",
            "metadata_fields": YW002_METADATA_FIELDS,
            "forbidden_fields": YW002_FORBIDDEN_FIELDS,
            "missing_token": "__MISSING__",
            "encoder": "OneHotEncoder(handle_unknown='ignore') fit on training fold only",
            "target_encoding": False,
        },
        "encoded_feature_dimensions": encoded_dim,
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "model_hyperparameters": dict(LR_KWARGS),
        "n_iter_per_fold": n_iter,
        "class_order": class_names,
        "fold_accuracy": {str(k): v for k, v in metrics["fold_accuracy"].items()},
        "oof_accuracy": metrics["oof_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_recall": metrics["per_class_recall"],
        "per_signature_accuracy": per_signature_accuracy(sigs, y.to_numpy(), pred),
        "runtime_seconds": runtime,
        "warnings": warnings_all,
        "n_cells": int(n),
        "n_classes": len(class_names),
        "oof_path": "work/outputs/oof/YW-002_oof.csv",
        "proba_path": "work/outputs/probabilities/YW-002_oof_probabilities.csv.gz",
    }
    write_oof(ROOT / "work/outputs/oof/YW-002_oof.csv", cell_ids, y.to_numpy(), pred, fold_ids.to_numpy())
    write_proba(ROOT / "work/outputs/probabilities/YW-002_oof_probabilities.csv.gz", cell_ids, proba, class_names)
    write_json(ROOT / "work/outputs/metrics/YW-002_metrics.json", payload)
    return payload


def run_yw003(data, folds: pd.DataFrame, class_names: List[str], yw001_oof: pd.DataFrame) -> dict:
    t0 = time.perf_counter()
    y = data.y_train.astype(str)
    cell_ids = y.index.astype(str)
    fold_ids = folds.set_index("Cell_ID").loc[cell_ids, "fold"]
    X_log = log1p_counts(data.counts_train)
    sigs = signatures_from_meta(data.meta_train)
    n = len(y)
    proba_unmasked = np.zeros((n, len(class_names)), dtype=np.float64)
    proba_masked = np.zeros((n, len(class_names)), dtype=np.float64)
    pred_unmasked = np.empty(n, dtype=object)
    pred_masked = np.empty(n, dtype=object)
    warnings_all: List[str] = []
    n_iter = {}
    n_unseen = 0
    n_uniform = 0
    fold_sig_kind = np.empty(n, dtype=object)
    for fold_id in FOLD_VALUES:
        train_mask, val_mask = train_val_masks(fold_ids, fold_id)
        cmap = build_candidate_map(sigs.to_numpy()[train_mask], y.to_numpy()[train_mask])
        model, caught, iters = fit_global_gene_model(X_log, y.to_numpy(), train_mask)
        warnings_all.extend("fold_{}: {}".format(fold_id, w) for w in caught)
        n_iter[str(fold_id)] = iters
        P = align_predict_proba(model, X_log[val_mask], class_names)
        proba_unmasked[val_mask] = P
        pred_unmasked[val_mask] = argmax_labels(P, class_names)
        val_idx = np.where(val_mask)[0]
        Pm = np.zeros_like(P)
        kinds = []
        for row_i, cell_i in enumerate(val_idx):
            sig = str(sigs.iloc[cell_i])
            if sig not in cmap:
                new_row, action = mask_and_renormalize(P[row_i], class_names, None)
                n_unseen += 1
                kinds.append("unseen")
            else:
                cands = cmap[sig]
                new_row, action = mask_and_renormalize(P[row_i], class_names, cands)
                kinds.append("deterministic" if is_deterministic_map(cands) else "ambiguous")
            if action == "uniform_candidates":
                n_uniform += 1
            Pm[row_i] = new_row
        proba_masked[val_mask] = Pm
        pred_masked[val_mask] = argmax_labels(Pm, class_names)
        fold_sig_kind[val_mask] = kinds
    runtime = time.perf_counter() - t0
    assert_probability_rows(proba_unmasked)
    assert_probability_rows(proba_masked)
    metrics_u = summarize_oof(y.to_numpy(), pred_unmasked, fold_ids.to_numpy(), labels=class_names)
    metrics_m = summarize_oof(y.to_numpy(), pred_masked, fold_ids.to_numpy(), labels=class_names)

    yw001 = yw001_oof.set_index("Cell_ID").loc[cell_ids]
    y001 = yw001["predicted_label"].astype(str).to_numpy()
    true = y.to_numpy()
    changed = int((pred_masked != y001).sum())
    w2c = int(((y001 != true) & (pred_masked == true)).sum())
    c2w = int(((y001 == true) & (pred_masked != true)).sum())
    net = w2c - c2w

    det_mask = fold_sig_kind == "deterministic"
    amb_mask = fold_sig_kind == "ambiguous"
    miss_key = missing_bucket_key()
    miss_mask = sigs.to_numpy() == miss_key

    def _acc(mask):
        if not np.any(mask):
            return None
        return float(np.mean(pred_masked[mask] == true[mask]))

    payload = {
        "run_id": "YW-003",
        "model": "gene_only_LogisticRegression + fold-safe candidate mask",
        "feature_set": "log1p(raw gene counts, 200); metadata used only for candidate masking, not as classifier features",
        "global_model": "same specification as YW-001 (gene-only multinomial L2 logistic regression)",
        "candidate_map": {
            "fields": YW002_METADATA_FIELDS,
            "missing_token": "__MISSING__",
            "fit": "training fold rows and labels only",
            "unseen_signature_policy": "keep unmasked global probabilities",
        },
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "model_hyperparameters": dict(LR_KWARGS),
        "n_iter_per_fold": n_iter,
        "class_order": class_names,
        "fold_accuracy": {str(k): v for k, v in metrics_m["fold_accuracy"].items()},
        "unmasked_global_oof_accuracy": metrics_u["oof_accuracy"],
        "oof_accuracy": metrics_m["oof_accuracy"],
        "macro_f1": metrics_m["macro_f1"],
        "per_class_recall": metrics_m["per_class_recall"],
        "delta_vs_YW-001": float(metrics_m["oof_accuracy"] - 0.55),
        "n_predictions_changed_vs_YW-001": changed,
        "yw001_wrong_to_yw003_correct": w2c,
        "yw001_correct_to_yw003_wrong": c2w,
        "net_additional_correct_vs_YW-001": net,
        "unseen_signature_fallback_count": int(n_unseen),
        "uniform_candidate_renorm_count": int(n_uniform),
        "deterministic_in_fold_accuracy": _acc(det_mask),
        "deterministic_in_fold_n": int(det_mask.sum()),
        "ambiguous_in_fold_accuracy": _acc(amb_mask),
        "ambiguous_in_fold_n": int(amb_mask.sum()),
        "missing_bucket_key": miss_key,
        "missing_bucket_accuracy": _acc(miss_mask),
        "missing_bucket_n": int(miss_mask.sum()),
        "per_signature_accuracy": per_signature_accuracy(sigs, true, pred_masked),
        "runtime_seconds": runtime,
        "warnings": warnings_all,
        "n_cells": int(n),
        "n_classes": len(class_names),
        "oof_path": "work/outputs/oof/YW-003_oof.csv",
        "proba_path": "work/outputs/probabilities/YW-003_oof_probabilities.csv.gz",
    }
    write_oof(ROOT / "work/outputs/oof/YW-003_oof.csv", cell_ids, true, pred_masked, fold_ids.to_numpy())
    write_proba(ROOT / "work/outputs/probabilities/YW-003_oof_probabilities.csv.gz", cell_ids, proba_masked, class_names)
    write_json(ROOT / "work/outputs/metrics/YW-003_metrics.json", payload)
    return payload


def run_yw004(data, folds: pd.DataFrame, class_names: List[str]) -> dict:
    t0 = time.perf_counter()
    y = data.y_train.astype(str)
    cell_ids = y.index.astype(str)
    fold_ids = folds.set_index("Cell_ID").loc[cell_ids, "fold"]
    X_log = log1p_counts(data.counts_train)
    sigs = signatures_from_meta(data.meta_train)
    n = len(y)
    proba = np.zeros((n, len(class_names)), dtype=np.float64)
    pred = np.empty(n, dtype=object)
    warnings_all: List[str] = []
    specialists_per_fold: Dict[str, int] = {}
    fallbacks = Counter()
    fallback_rows = []
    fold_sig_kind = np.empty(n, dtype=object)
    for fold_id in FOLD_VALUES:
        train_mask, val_mask = train_val_masks(fold_ids, fold_id)
        cmap = build_candidate_map(sigs.to_numpy()[train_mask], y.to_numpy()[train_mask])
        global_model, caught, _iters = fit_global_gene_model(X_log, y.to_numpy(), train_mask)
        warnings_all.extend("fold_{}_global: {}".format(fold_id, w) for w in caught)
        P_global = align_predict_proba(global_model, X_log[val_mask], class_names)
        val_idx = np.where(val_mask)[0]
        P_out = np.zeros_like(P_global)
        kinds = []
        n_specialists = 0
        specialist_cache = {}
        for row_i, cell_i in enumerate(val_idx):
            sig = str(sigs.iloc[cell_i])
            if sig not in cmap:
                masked_row, _action = mask_and_renormalize(P_global[row_i], class_names, None)
                P_out[row_i] = masked_row
                fallbacks["unseen_signature_unmasked_global"] += 1
                fallback_rows.append({"fold": fold_id, "signature": sig, "reason": "unseen_signature"})
                kinds.append("unseen")
                continue
            cands = cmap[sig]
            if is_deterministic_map(cands):
                lab = next(iter(cands))
                P_out[row_i] = one_hot_class(1, lab, class_names)[0]
                kinds.append("deterministic")
                continue
            kinds.append("ambiguous")
            if sig not in specialist_cache:
                local_mask = train_mask & (sigs.to_numpy() == sig)
                y_local = y.to_numpy()[local_mask]
                n_local = int(local_mask.sum())
                n_cls = int(len(set(y_local.tolist())))
                reason = None
                if n_cls < 2:
                    reason = "fewer_than_two_target_classes"
                elif n_local < 2:
                    reason = "insufficient_training_rows"
                elif n_cls < len(cands):
                    reason = "required_class_absent"
                model_local = None
                if reason is None:
                    try:
                        model_local = make_logistic_regression()
                        with warnings.catch_warnings(record=True) as caught_local:
                            warnings.simplefilter("always")
                            model_local.fit(X_log[local_mask], y_local)
                            for item in caught_local:
                                if issubclass(item.category, ConvergenceWarning) or "Convergence" in item.category.__name__:
                                    warnings_all.append(
                                        "fold_{} sig={}: {}: {}".format(
                                            fold_id, sig, item.category.__name__, item.message
                                        )
                                    )
                    except Exception as exc:
                        reason = "specialist_fit_failed: {}".format(exc)
                        model_local = None
                if reason is not None:
                    specialist_cache[sig] = ("fallback", reason)
                else:
                    specialist_cache[sig] = ("model", model_local)
                    n_specialists += 1
            kind, payload_local = specialist_cache[sig]
            if kind == "fallback":
                masked_row, _action = mask_and_renormalize(P_global[row_i], class_names, cands)
                P_out[row_i] = masked_row
                fallbacks[str(payload_local)] += 1
                fallback_rows.append({"fold": fold_id, "signature": sig, "reason": str(payload_local)})
            else:
                local_p = align_predict_proba(payload_local, X_log[[cell_i]], class_names)[0]
                masked_row, _action = mask_and_renormalize(local_p, class_names, cands)
                P_out[row_i] = masked_row
        specialists_per_fold[str(fold_id)] = n_specialists
        proba[val_mask] = P_out
        pred[val_mask] = argmax_labels(P_out, class_names)
        fold_sig_kind[val_mask] = kinds
    runtime = time.perf_counter() - t0
    assert_probability_rows(proba)
    metrics = summarize_oof(y.to_numpy(), pred, fold_ids.to_numpy(), labels=class_names)
    true = y.to_numpy()
    det_mask = fold_sig_kind == "deterministic"
    amb_mask = fold_sig_kind == "ambiguous"
    miss_key = missing_bucket_key()
    miss_mask = sigs.to_numpy() == miss_key

    def _acc(mask):
        if not np.any(mask):
            return None
        return float(np.mean(pred[mask] == true[mask]))

    amb_sig_acc = {}
    for sig, stats in per_signature_accuracy(sigs, true, pred).items():
        # report every signature; callers can subset. Also store ambiguous-only below.
        pass
    per_sig = per_signature_accuracy(sigs, true, pred)
    # Ambiguous signatures: those that were ambiguous in at least one fold for some cell.
    amb_sigs = sorted(set(sigs.to_numpy()[amb_mask].tolist())) if np.any(amb_mask) else []
    for sig in amb_sigs:
        amb_sig_acc[sig] = per_sig.get(sig)

    payload = {
        "run_id": "YW-004",
        "model": "per-signature specialists (gene-only L2 logistic regression) with YW-003 masked-global fallback",
        "feature_set": "log1p(raw gene counts, 200) inside each fold-learned signature group; no extra metadata features in specialists",
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "model_hyperparameters": dict(LR_KWARGS),
        "class_order": class_names,
        "fold_accuracy": {str(k): v for k, v in metrics["fold_accuracy"].items()},
        "oof_accuracy": metrics["oof_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_recall": metrics["per_class_recall"],
        "deterministic_signature_accuracy": _acc(det_mask),
        "deterministic_signature_n": int(det_mask.sum()),
        "ambiguous_signature_accuracy": _acc(amb_mask),
        "ambiguous_signature_n": int(amb_mask.sum()),
        "missing_bucket_key": miss_key,
        "missing_bucket_accuracy": _acc(miss_mask),
        "missing_bucket_n": int(miss_mask.sum()),
        "per_signature_accuracy": per_sig,
        "ambiguous_signature_accuracy_by_signature": amb_sig_acc,
        "n_specialists_trained_per_fold": specialists_per_fold,
        "fallback_counts": dict(fallbacks),
        "fallback_n": int(sum(fallbacks.values())),
        "fallback_examples": fallback_rows[:50],
        "confusion_pairs_top5": confusion_pairs(true, pred, k=5),
        "runtime_seconds": runtime,
        "warnings": warnings_all,
        "n_cells": int(n),
        "n_classes": len(class_names),
        "oof_path": "work/outputs/oof/YW-004_oof.csv",
        "proba_path": "work/outputs/probabilities/YW-004_oof_probabilities.csv.gz",
    }
    write_oof(ROOT / "work/outputs/oof/YW-004_oof.csv", cell_ids, true, pred, fold_ids.to_numpy())
    write_proba(ROOT / "work/outputs/probabilities/YW-004_oof_probabilities.csv.gz", cell_ids, proba, class_names)
    write_json(ROOT / "work/outputs/metrics/YW-004_metrics.json", payload)
    return payload


def conclusion_for(run_id: str, payload: dict) -> str:
    if run_id == "YW-002":
        return "Metadata one-hot + genes OOF accuracy={:.4f} vs YW-001 0.5500".format(
            payload["oof_accuracy"]
        )
    if run_id == "YW-003":
        return "Fold-safe candidate masking OOF={:.4f}; net additional correct vs YW-001={}".format(
            payload["oof_accuracy"], payload["net_additional_correct_vs_YW-001"]
        )
    if run_id == "YW-004":
        return "Per-signature specialists OOF={:.4f}; missing-bucket accuracy={:.4f}; fallbacks={}".format(
            payload["oof_accuracy"],
            payload["missing_bucket_accuracy"],
            payload["fallback_n"],
        )
    return "completed"


def record_run(payload: dict, overwrite: bool) -> None:
    run_id = payload["run_id"]
    fold_acc = payload["fold_accuracy"]
    notes = "; ".join(payload.get("warnings") or []) or "no warnings"
    row = {
        "run_id": run_id,
        "timestamp": utc_now(),
        "git_commit": git_commit(ROOT),
        "model": payload.get("model", ""),
        "feature_set": payload.get("feature_set", ""),
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "fold_0_accuracy": "{:.10f}".format(fold_acc["0"]),
        "fold_1_accuracy": "{:.10f}".format(fold_acc["1"]),
        "fold_2_accuracy": "{:.10f}".format(fold_acc["2"]),
        "oof_accuracy": "{:.10f}".format(payload["oof_accuracy"]),
        "macro_f1": "{:.10f}".format(payload["macro_f1"]),
        "runtime_seconds": "{:.3f}".format(payload["runtime_seconds"]),
        "status": "completed",
        "notes": notes,
        "manifest_sha256": manifest_sha256(ROOT),
        "folds_sha256": folds_sha256(ROOT),
        "oof_path": payload.get("oof_path", ""),
        "proba_path": payload.get("proba_path", ""),
        "metrics_path": "work/outputs/metrics/{}_metrics.json".format(run_id),
        "conclusion": conclusion_for(run_id, payload),
    }
    append_registry_row(row, ROOT, overwrite=overwrite)


def print_run(payload: dict) -> None:
    print("=== {} ===".format(payload["run_id"]))
    print("fold accuracies:", payload["fold_accuracy"])
    print("OOF accuracy: {:.6f}".format(payload["oof_accuracy"]))
    print("macro-F1: {:.6f}".format(payload["macro_f1"]))
    print("runtime_seconds: {:.3f}".format(payload["runtime_seconds"]))
    if payload.get("warnings"):
        print("warnings: n={}".format(len(payload["warnings"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="*", default=["YW-002", "YW-003", "YW-004"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("Verifying official data manifest...")
    for line in verify_official_manifest(ROOT):
        print(" - {}".format(line))

    data = load_dataset(ROOT)
    for line in validate_contract(data):
        print(" - {}".format(line))
    folds = load_folds(ROOT / "work" / "experiments" / "folds.csv")
    class_names = allowed_labels(ROOT)
    yw001_path = ROOT / "work" / "outputs" / "oof" / "YW-001_oof.csv"
    if not yw001_path.is_file():
        raise SystemExit("missing YW-001 OOF file; refusing to overwrite by rerunning it")
    yw001_oof = pd.read_csv(yw001_path, dtype={"Cell_ID": str})

    runners = {
        "YW-002": lambda: run_yw002(data, folds, class_names),
        "YW-003": lambda: run_yw003(data, folds, class_names, yw001_oof),
        "YW-004": lambda: run_yw004(data, folds, class_names),
    }
    for run_id in args.run:
        if run_id not in runners:
            raise SystemExit("unknown run {}".format(run_id))
        print("\nRunning {} (overwrite={})".format(run_id, args.overwrite))
        payload = runners[run_id]()
        try:
            record_run(payload, overwrite=args.overwrite)
        except RegistryError as exc:
            print("REGISTRY ERROR: {}".format(exc), file=sys.stderr)
            return 1
        print_run(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
