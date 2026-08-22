"""Evaluation metrics for the MERFISH-60 protocol."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def overall_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_recall(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {label: float(value) for label, value in zip(labels, recalls)}


def summarize_oof(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    folds: Sequence[int],
    labels: Optional[Iterable[str]] = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    folds = np.asarray(folds)
    if labels is None:
        labels = sorted(set(y_true.tolist()))
    labels = list(labels)

    fold_acc = {}
    for fold_id in sorted(set(folds.tolist())):
        mask = folds == fold_id
        fold_acc[int(fold_id)] = overall_accuracy(y_true[mask], y_pred[mask])

    return {
        "fold_accuracy": fold_acc,
        "oof_accuracy": overall_accuracy(y_true, y_pred),
        "macro_f1": macro_f1_score(y_true, y_pred),
        "per_class_recall": per_class_recall(y_true, y_pred, labels=labels),
    }
