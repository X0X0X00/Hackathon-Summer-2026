"""Shared fold-safe model helpers for Sprint 2 experiments."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from merfish60.cv import CV_RANDOM_STATE
from merfish60.official_contract import YW002_METADATA_FIELDS
from merfish60.signatures import canonicalize_value


LR_KWARGS = dict(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=2000,
    class_weight=None,
    random_state=CV_RANDOM_STATE,
)


def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(**LR_KWARGS)


def log1p_counts(counts: pd.DataFrame) -> np.ndarray:
    return np.log1p(counts.to_numpy(dtype=np.float64))


def canonicalize_metadata_frame(meta: pd.DataFrame, fields: Sequence[str] = YW002_METADATA_FIELDS) -> pd.DataFrame:
    out = pd.DataFrame(index=meta.index)
    for field in fields:
        out[field] = [canonicalize_value(v) for v in meta[field].to_numpy()]
    return out


def fit_onehot_encoder(train_meta) -> OneHotEncoder:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    arr = train_meta.to_numpy() if hasattr(train_meta, "to_numpy") else np.asarray(train_meta)
    encoder.fit(arr)
    return encoder


def transform_onehot(encoder: OneHotEncoder, meta) -> np.ndarray:
    arr = meta.to_numpy() if hasattr(meta, "to_numpy") else np.asarray(meta)
    return encoder.transform(arr)


def align_predict_proba(
    model,
    X: np.ndarray,
    global_classes: Sequence[str],
) -> np.ndarray:
    """Return probabilities in the stable global class order."""
    local = model.predict_proba(X)
    out = np.zeros((X.shape[0], len(global_classes)), dtype=np.float64)
    index = {name: i for i, name in enumerate(global_classes)}
    for j, name in enumerate(model.classes_):
        out[:, index[str(name)]] = local[:, j]
    return out


def one_hot_class(n_rows: int, class_name: str, global_classes: Sequence[str]) -> np.ndarray:
    out = np.zeros((n_rows, len(global_classes)), dtype=np.float64)
    idx = list(global_classes).index(str(class_name))
    out[:, idx] = 1.0
    return out


def argmax_labels(proba: np.ndarray, global_classes: Sequence[str]) -> np.ndarray:
    return np.asarray(global_classes)[np.argmax(proba, axis=1)]


def assert_probability_rows(proba: np.ndarray, atol: float = 1e-6) -> None:
    totals = proba.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=atol):
        bad = int(np.sum(~np.isclose(totals, 1.0, atol=atol)))
        raise ValueError("probability rows not summing to 1: n_bad={}".format(bad))


def train_val_masks(fold_ids: pd.Series, fold_id: int) -> Tuple[np.ndarray, np.ndarray]:
    val = (fold_ids.to_numpy() == fold_id)
    train = ~val
    if (train & val).any():
        raise RuntimeError("train and validation masks overlap")
    if not train.any() or not val.any():
        raise RuntimeError("empty train or validation split for fold {}".format(fold_id))
    return train, val
