"""Leakage-safe inner-CV model selection for the missing-metadata bucket.

Inner comparison uses only the arrays passed in. Callers must pass
outer-training bucket rows; this module never consults outer-validation
labels.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from merfish60.cv import CV_RANDOM_STATE
from merfish60.models import align_predict_proba, log1p_counts, make_logistic_regression
from merfish60.signatures import missing_bucket_key


HARD_BUCKET_FOCUS_CLASSES = [
    "oligodendrocyte_1",
    "oligodendrocyte_2",
    "oligodendrocyte_progenitor_1",
    "oligodendrocyte_progenitor_2",
    "oligodendrocyte_precursor_cell",
    "astrocyte_1",
    "astrocyte_2",
]

EXTRA_TREES_KWARGS = dict(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    n_jobs=1,
    random_state=CV_RANDOM_STATE,
    class_weight=None,
)

HGB_KWARGS = dict(
    max_iter=150,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=10,
    random_state=CV_RANDOM_STATE,
    early_stopping=False,
)


def log1p_cp10k(counts_array: np.ndarray) -> np.ndarray:
    """Per-cell library-size normalization to CP10k, then log1p."""
    X = np.asarray(counts_array, dtype=np.float64)
    lib = X.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1.0)
    return np.log1p(X / lib * 1.0e4)


def _make_hgb():
    kwargs = dict(HGB_KWARGS)
    try:
        return HistGradientBoostingClassifier(class_weight=None, **kwargs)
    except TypeError:
        return HistGradientBoostingClassifier(**kwargs)


CANDIDATE_SPECS = (
    {
        "name": "log1p_lr",
        "description": "log1p(raw counts) + multinomial L2 LogisticRegression",
        "transform": "log1p_raw",
        "factory": make_logistic_regression,
    },
    {
        "name": "cp10k_log1p_lr",
        "description": "per-cell CP10k library-size normalization -> log1p + L2 LogisticRegression",
        "transform": "log1p_cp10k",
        "factory": make_logistic_regression,
    },
    {
        "name": "extratrees",
        "description": "ExtraTreesClassifier on log1p(raw counts)",
        "transform": "log1p_raw",
        "factory": lambda: ExtraTreesClassifier(**EXTRA_TREES_KWARGS),
    },
    {
        "name": "hist_gradient_boosting",
        "description": "HistGradientBoostingClassifier on log1p(raw counts)",
        "transform": "log1p_raw",
        "factory": _make_hgb,
    },
)


def apply_transform(name: str, counts_array: np.ndarray) -> np.ndarray:
    if name == "log1p_raw":
        return np.log1p(np.asarray(counts_array, dtype=np.float64))
    if name == "log1p_cp10k":
        return log1p_cp10k(counts_array)
    raise ValueError("unknown transform {}".format(name))


def choose_inner_n_splits(y: Sequence[str], requested: int = 3) -> int:
    counts = {}
    for lab in y:
        counts[lab] = counts.get(lab, 0) + 1
    min_n = min(counts.values()) if counts else 0
    n_splits = min(int(requested), int(min_n))
    if n_splits < 2:
        raise ValueError(
            "cannot run stratified inner CV: min class count={} in outer-training bucket".format(
                min_n
            )
        )
    return n_splits


def inner_cv_accuracy(
    X_counts: np.ndarray,
    y: np.ndarray,
    spec: dict,
    class_names: Sequence[str],
    random_state: int = CV_RANDOM_STATE,
) -> float:
    """Mean inner-fold accuracy. Uses only the provided X_counts/y rows."""
    y = np.asarray(y, dtype=object)
    n_splits = choose_inner_n_splits(y.tolist(), requested=3)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    dummy = np.zeros((len(y), 1))
    scores = []
    for train_idx, val_idx in splitter.split(dummy, y):
        X_tr = apply_transform(spec["transform"], X_counts[train_idx])
        X_va = apply_transform(spec["transform"], X_counts[val_idx])
        model = spec["factory"]()
        model.fit(X_tr, y[train_idx])
        proba = align_predict_proba(model, X_va, class_names)
        pred = np.asarray(class_names)[np.argmax(proba, axis=1)]
        scores.append(float(np.mean(pred == y[val_idx])))
    return float(np.mean(scores))


def select_hard_bucket_model(
    X_counts_train: np.ndarray,
    y_train: np.ndarray,
    class_names: Sequence[str],
    random_state: int = CV_RANDOM_STATE,
) -> Tuple[str, Dict[str, float]]:
    """Compare candidates on inner CV of OUTER-TRAINING rows only.

    Validation/held-out labels must not be passed in.
    """
    y_train = np.asarray(y_train, dtype=object)
    scores = {}
    best_name = None
    best_score = -1.0
    for spec in CANDIDATE_SPECS:
        acc = inner_cv_accuracy(
            X_counts_train, y_train, spec, class_names, random_state=random_state
        )
        scores[spec["name"]] = acc
        print("    candidate {} inner-CV accuracy={:.6f}".format(spec["name"], acc), flush=True)
        if acc > best_score:
            best_score = acc
            best_name = spec["name"]
    return best_name, scores


def spec_by_name(name: str) -> dict:
    for spec in CANDIDATE_SPECS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def fit_selected_model(
    spec: dict,
    X_counts_train: np.ndarray,
    y_train: np.ndarray,
):
    X_tr = apply_transform(spec["transform"], X_counts_train)
    model = spec["factory"]()
    model.fit(X_tr, np.asarray(y_train, dtype=object))
    return model


def predict_selected_model(
    spec: dict,
    model,
    X_counts: np.ndarray,
    class_names: Sequence[str],
) -> np.ndarray:
    X = apply_transform(spec["transform"], X_counts)
    return align_predict_proba(model, X, class_names)
