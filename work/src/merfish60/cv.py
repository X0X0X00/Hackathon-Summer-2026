"""Frozen 3-fold stratified validation protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from merfish60.io import repo_root


CV_N_SPLITS = 3
CV_SHUFFLE = True
CV_RANDOM_STATE = 20260819
CV_PROTOCOL = "StratifiedKFold(n_splits=3, shuffle=True, random_state=20260819)"
FOLD_VALUES = (0, 1, 2)


class FoldContractError(Exception):
    """Raised when persisted folds violate the frozen protocol."""


def folds_path(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / "work" / "experiments" / "folds.csv"


def make_fold_assignments(cell_ids: Sequence[str], y: Sequence[str]) -> pd.DataFrame:
    """Assign each training Cell_ID to a validation fold using the frozen protocol."""
    cell_ids = pd.Index([str(v) for v in cell_ids], name="Cell_ID")
    y = pd.Series(list(y), index=cell_ids, dtype=object)
    if len(cell_ids) != len(y):
        raise FoldContractError("cell_ids and y have different lengths")
    if cell_ids.has_duplicates:
        raise FoldContractError("duplicate Cell_IDs passed to fold assignment")

    fold = np.empty(len(cell_ids), dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=CV_SHUFFLE,
        random_state=CV_RANDOM_STATE,
    )
    dummy_x = np.zeros((len(y), 1), dtype=np.float64)
    for fold_id, (_train_idx, val_idx) in enumerate(splitter.split(dummy_x, y)):
        fold[val_idx] = fold_id

    out = pd.DataFrame({"Cell_ID": cell_ids.to_numpy(), "fold": fold})
    return out


def write_folds(folds: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or folds_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    folds = folds.loc[:, ["Cell_ID", "fold"]].copy()
    folds["Cell_ID"] = folds["Cell_ID"].astype(str)
    folds["fold"] = folds["fold"].astype(int)
    folds.to_csv(path, index=False)
    return path


def load_folds(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or folds_path()
    if not path.is_file():
        raise FoldContractError("folds file does not exist: {}".format(path))
    folds = pd.read_csv(path, dtype={"Cell_ID": str, "fold": int})
    if list(folds.columns) != ["Cell_ID", "fold"]:
        raise FoldContractError(
            "folds.csv columns {} != ['Cell_ID', 'fold']".format(list(folds.columns))
        )
    folds["Cell_ID"] = folds["Cell_ID"].astype(str)
    if folds["Cell_ID"].str.endswith(".0").any():
        raise FoldContractError("folds.csv Cell_ID appears float-cast")
    return folds


def validate_folds(
    folds: pd.DataFrame,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    y_train: Sequence[str],
) -> list:
    """Validate persisted folds against the frozen protocol and training IDs."""
    messages = []
    train_ids = pd.Index([str(v) for v in train_ids], name="Cell_ID")
    test_ids = pd.Index([str(v) for v in test_ids])
    y_train = pd.Series(list(y_train), index=train_ids, dtype=object)

    if list(folds.columns) != ["Cell_ID", "fold"]:
        raise FoldContractError(
            "folds columns {} != ['Cell_ID', 'fold']".format(list(folds.columns))
        )

    fold_ids = pd.Index(folds["Cell_ID"].astype(str))
    if fold_ids.has_duplicates:
        raise FoldContractError("duplicate Cell_IDs in folds.csv")
    if len(folds) != len(train_ids):
        raise FoldContractError(
            "folds n={} != n_train={}".format(len(folds), len(train_ids))
        )
    if set(fold_ids) != set(train_ids):
        missing = set(train_ids) - set(fold_ids)
        extra = set(fold_ids) - set(train_ids)
        raise FoldContractError(
            "folds Cell_ID set != train Cell_ID set (missing={}, extra={})".format(
                len(missing), len(extra)
            )
        )
    messages.append("every training Cell_ID appears exactly once")

    test_in_folds = set(fold_ids) & set(test_ids)
    if test_in_folds:
        raise FoldContractError(
            "test Cell_IDs present in folds.csv: n={}".format(len(test_in_folds))
        )
    messages.append("no test Cell_IDs in folds.csv")

    unique_folds = tuple(sorted(folds["fold"].unique().tolist()))
    if unique_folds != FOLD_VALUES:
        raise FoldContractError("fold values {} != {}".format(unique_folds, FOLD_VALUES))
    messages.append("folds are 0, 1, 2")

    expected = make_fold_assignments(train_ids, y_train)
    merged = folds.merge(expected, on="Cell_ID", suffixes=("_saved", "_expected"))
    n_mismatch = int((merged["fold_saved"] != merged["fold_expected"]).sum())
    if n_mismatch:
        raise FoldContractError(
            "saved folds do not match frozen protocol; mismatches={}".format(n_mismatch)
        )
    messages.append("saved folds match {}".format(CV_PROTOCOL))

    fold_labels = folds.merge(
        y_train.rename("label").reset_index(),
        on="Cell_ID",
        how="left",
    )
    if fold_labels["label"].isna().any():
        raise FoldContractError("folds contain Cell_IDs without training labels")

    classes = sorted(y_train.unique().tolist())
    n_splits = CV_N_SPLITS
    missing_by_fold = {}
    for fold_id in FOLD_VALUES:
        present = set(fold_labels.loc[fold_labels["fold"] == fold_id, "label"])
        missing = [c for c in classes if c not in present]
        # A class can appear in every fold iff count >= n_splits.
        impossible = [c for c in missing if int((y_train == c).sum()) < n_splits]
        unexpected = [c for c in missing if c not in impossible]
        if unexpected:
            missing_by_fold[fold_id] = unexpected
    if missing_by_fold:
        raise FoldContractError(
            "class missing from a fold despite n >= n_splits: {}".format(missing_by_fold)
        )
    messages.append(
        "each fold contains every class with n >= {} (all {} classes)".format(
            n_splits, len(classes)
        )
    )
    return messages


def load_and_validate_folds(
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    y_train: Sequence[str],
    path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, list]:
    folds = load_folds(path)
    messages = validate_folds(folds, train_ids, test_ids, y_train)
    return folds, messages
