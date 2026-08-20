"""Frozen YW-004 architecture evaluated on an arbitrary persisted fold file.

This is not a new model version. Hyperparameters stay exactly LR_KWARGS.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from merfish60.metrics import summarize_oof
from merfish60.models import (
    align_predict_proba,
    argmax_labels,
    assert_probability_rows,
    log1p_counts,
    make_logistic_regression,
    one_hot_class,
    train_val_masks,
)
from merfish60.signatures import (
    build_candidate_map,
    is_deterministic_map,
    mask_and_renormalize,
    missing_bucket_key,
    signatures_from_meta,
)


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


def confusion_pairs(y_true: Sequence[str], y_pred: Sequence[str], k: int = 10) -> List[dict]:
    counts = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            counts[(str(t), str(p))] += 1
    top = []
    for (t, p), n in counts.most_common(k):
        top.append({"true_label": t, "predicted_label": p, "n": int(n)})
    return top


def run_yw004_oof(
    data,
    folds: pd.DataFrame,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Run frozen YW-004 specialists on the provided fold assignments.

    Returns (pred, proba, diagnostics).
    """
    y = data.y_train.astype(str)
    cell_ids = y.index.astype(str)
    fold_ids = folds.set_index("Cell_ID").loc[cell_ids, "fold"]
    fold_values = tuple(sorted(int(v) for v in fold_ids.unique().tolist()))
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

    for fold_id in fold_values:
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
                fallback_rows.append(
                    {"fold": fold_id, "signature": sig, "reason": "unseen_signature"}
                )
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
                                if issubclass(item.category, ConvergenceWarning) or (
                                    "Convergence" in item.category.__name__
                                ):
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
                fallback_rows.append(
                    {"fold": fold_id, "signature": sig, "reason": str(payload_local)}
                )
            else:
                local_p = align_predict_proba(payload_local, X_log[[cell_i]], class_names)[0]
                masked_row, _action = mask_and_renormalize(local_p, class_names, cands)
                P_out[row_i] = masked_row
        specialists_per_fold[str(fold_id)] = n_specialists
        proba[val_mask] = P_out
        pred[val_mask] = argmax_labels(P_out, class_names)
        fold_sig_kind[val_mask] = kinds

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

    per_sig = per_signature_accuracy(sigs, true, pred)
    amb_sigs = sorted(set(sigs.to_numpy()[amb_mask].tolist())) if np.any(amb_mask) else []
    amb_sig_acc = {sig: per_sig.get(sig) for sig in amb_sigs}

    diagnostics = {
        "cell_ids": [str(v) for v in cell_ids],
        "fold_ids": fold_ids.to_numpy().astype(int),
        "y_true": true,
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
        "confusion_pairs_top10": confusion_pairs(true, pred, k=10),
        "warnings": warnings_all,
        "n_cells": int(n),
        "n_classes": len(class_names),
        "fold_values": list(fold_values),
        "signatures": sigs.to_numpy(),
    }
    return pred, proba, diagnostics
