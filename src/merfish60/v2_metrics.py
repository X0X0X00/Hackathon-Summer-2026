"""Shared slice metrics and artifact writers for the MODEL V2 family."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from merfish60.signatures import missing_bucket_key, signatures_from_meta


def json_default(obj):
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
    path.write_text(json.dumps(clean, indent=2, sort_keys=True, default=json_default) + "\n")


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


def write_confusion(
    path: Path,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(list(y_true), list(y_pred), labels=list(class_names))
    frame = pd.DataFrame(matrix, index=list(class_names), columns=list(class_names))
    frame.to_csv(path)


def confusion_pairs(y_true: Sequence[str], y_pred: Sequence[str], k: int = 10) -> List[dict]:
    counts = Counter()
    for true_lab, pred_lab in zip(y_true, y_pred):
        if true_lab != pred_lab:
            counts[(str(true_lab), str(pred_lab))] += 1
    return [
        {"true_label": t, "predicted_label": p, "n": int(n)}
        for (t, p), n in counts.most_common(k)
    ]


def accuracy_on_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    if not np.any(mask):
        return None
    return float(np.mean(y_true[mask] == y_pred[mask]))


def hard_bucket_mask(meta_train) -> np.ndarray:
    sigs = signatures_from_meta(meta_train)
    return sigs.to_numpy() == missing_bucket_key()


def neuron_glial_masks(y_true: Sequence[str], class_names: Sequence[str], ei_of_label: np.ndarray):
    index = {str(name): i for i, name in enumerate(class_names)}
    codes = np.array([index[str(v)] for v in y_true], dtype=np.int64)
    ei = np.asarray(ei_of_label)
    neuron = np.isin(ei[codes], [0, 1])
    glial = ei[codes] == -1
    return neuron, glial


def slice_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    meta_train,
    class_names: Sequence[str],
    ei_of_label: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    hard = hard_bucket_mask(meta_train)
    neuron, glial = neuron_glial_masks(y_true, class_names, ei_of_label)
    return {
        "hard_bucket_key": missing_bucket_key(),
        "hard_bucket_n": int(hard.sum()),
        "hard_bucket_accuracy": accuracy_on_mask(y_true, y_pred, hard),
        "neuron_n": int(neuron.sum()),
        "neuron_accuracy": accuracy_on_mask(y_true, y_pred, neuron),
        "glial_n": int(glial.sum()),
        "glial_accuracy": accuracy_on_mask(y_true, y_pred, glial),
        "confusion_pairs_top10": confusion_pairs(y_true, y_pred, k=10),
    }


def universe_fold_ids(folds: pd.DataFrame, cell_ids: Iterable[str], is_train: np.ndarray) -> np.ndarray:
    """Map persisted train folds onto the train-then-test universe (test = -1)."""
    mapping = folds.set_index("Cell_ID")["fold"]
    ids = [str(v) for v in cell_ids]
    out = np.full(len(ids), -1, dtype=np.int64)
    train_mask = np.asarray(is_train)
    for i, cid in enumerate(ids):
        if train_mask[i]:
            out[i] = int(mapping.loc[cid])
    return out
