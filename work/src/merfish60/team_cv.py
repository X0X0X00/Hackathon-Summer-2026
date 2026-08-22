"""Team-compatible 5-fold protocol. Does not touch experiments/folds.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from merfish60.io import repo_root
from merfish60.official_contract import sha256_file


TEAM_CV_N_SPLITS = 5
TEAM_CV_SHUFFLE = True
TEAM_CV_RANDOM_STATE = 42
TEAM_CV_PROTOCOL = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
TEAM_FOLD_VALUES = (0, 1, 2, 3, 4)
TEAM_FOLDS_REL = "work/experiments/team_folds_5_seed42.csv"


class TeamFoldContractError(Exception):
    """Raised when the team-compatible 5-fold file is invalid."""


def team_folds_path(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / TEAM_FOLDS_REL


def make_team_fold_assignments(cell_ids: Sequence[str], y: Sequence[str]) -> pd.DataFrame:
    """Assign each training Cell_ID using the team-compatible 5-fold protocol."""
    cell_ids = pd.Index([str(v) for v in cell_ids], name="Cell_ID")
    y = pd.Series(list(y), index=cell_ids, dtype=object)
    if len(cell_ids) != len(y):
        raise TeamFoldContractError("cell_ids and y have different lengths")
    if cell_ids.has_duplicates:
        raise TeamFoldContractError("duplicate Cell_IDs passed to team fold assignment")

    fold = np.empty(len(cell_ids), dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=TEAM_CV_N_SPLITS,
        shuffle=TEAM_CV_SHUFFLE,
        random_state=TEAM_CV_RANDOM_STATE,
    )
    dummy_x = np.zeros((len(y), 1), dtype=np.float64)
    for fold_id, (_train_idx, val_idx) in enumerate(splitter.split(dummy_x, y)):
        fold[val_idx] = fold_id

    return pd.DataFrame({"Cell_ID": cell_ids.to_numpy(), "fold": fold})


def write_team_folds(folds: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or team_folds_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    folds = folds.loc[:, ["Cell_ID", "fold"]].copy()
    folds["Cell_ID"] = folds["Cell_ID"].astype(str)
    folds["fold"] = folds["fold"].astype(int)
    folds.to_csv(path, index=False)
    return path


def load_team_folds(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or team_folds_path()
    if not path.is_file():
        raise TeamFoldContractError("team folds file does not exist: {}".format(path))
    folds = pd.read_csv(path, dtype={"Cell_ID": str, "fold": int})
    if list(folds.columns) != ["Cell_ID", "fold"]:
        raise TeamFoldContractError(
            "team folds columns {} != ['Cell_ID', 'fold']".format(list(folds.columns))
        )
    folds["Cell_ID"] = folds["Cell_ID"].astype(str)
    if folds["Cell_ID"].str.endswith(".0").any():
        raise TeamFoldContractError("team folds Cell_ID appears float-cast")
    return folds


def validate_team_folds(
    folds: pd.DataFrame,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    y_train: Sequence[str],
) -> list:
    messages = []
    train_ids = pd.Index([str(v) for v in train_ids], name="Cell_ID")
    test_ids = pd.Index([str(v) for v in test_ids])
    y_train = pd.Series(list(y_train), index=train_ids, dtype=object)

    if list(folds.columns) != ["Cell_ID", "fold"]:
        raise TeamFoldContractError(
            "folds columns {} != ['Cell_ID', 'fold']".format(list(folds.columns))
        )

    fold_ids = pd.Index(folds["Cell_ID"].astype(str))
    if fold_ids.has_duplicates:
        raise TeamFoldContractError("duplicate Cell_IDs in team folds file")
    if len(folds) != len(train_ids):
        raise TeamFoldContractError(
            "team folds n={} != n_train={}".format(len(folds), len(train_ids))
        )
    if set(fold_ids) != set(train_ids):
        missing = set(train_ids) - set(fold_ids)
        extra = set(fold_ids) - set(train_ids)
        raise TeamFoldContractError(
            "team folds Cell_ID set != train Cell_ID set (missing={}, extra={})".format(
                len(missing), len(extra)
            )
        )
    messages.append("every training Cell_ID appears exactly once")

    test_in_folds = set(fold_ids) & set(test_ids)
    if test_in_folds:
        raise TeamFoldContractError(
            "test Cell_IDs present in team folds: n={}".format(len(test_in_folds))
        )
    messages.append("no test Cell_IDs in team folds file")

    unique_folds = tuple(sorted(folds["fold"].unique().tolist()))
    if unique_folds != TEAM_FOLD_VALUES:
        raise TeamFoldContractError(
            "fold values {} != {}".format(unique_folds, TEAM_FOLD_VALUES)
        )
    messages.append("folds are 0, 1, 2, 3, 4")

    expected = make_team_fold_assignments(train_ids, y_train)
    merged = folds.merge(expected, on="Cell_ID", suffixes=("_saved", "_expected"))
    n_mismatch = int((merged["fold_saved"] != merged["fold_expected"]).sum())
    if n_mismatch:
        raise TeamFoldContractError(
            "saved team folds do not match {}; mismatches={}".format(
                TEAM_CV_PROTOCOL, n_mismatch
            )
        )
    messages.append("saved folds match {}".format(TEAM_CV_PROTOCOL))

    fold_labels = folds.merge(
        y_train.rename("label").reset_index(),
        on="Cell_ID",
        how="left",
    )
    if fold_labels["label"].isna().any():
        raise TeamFoldContractError("team folds contain Cell_IDs without training labels")

    classes = sorted(y_train.unique().tolist())
    missing_by_fold = {}
    for fold_id in TEAM_FOLD_VALUES:
        present = set(fold_labels.loc[fold_labels["fold"] == fold_id, "label"])
        missing = [c for c in classes if c not in present]
        impossible = [c for c in missing if int((y_train == c).sum()) < TEAM_CV_N_SPLITS]
        unexpected = [c for c in missing if c not in impossible]
        if unexpected:
            missing_by_fold[fold_id] = unexpected
    if missing_by_fold:
        raise TeamFoldContractError(
            "class missing from a 5-fold despite n >= n_splits: {}".format(missing_by_fold)
        )
    n_eligible = sum(int((y_train == c).sum()) >= TEAM_CV_N_SPLITS for c in classes)
    messages.append(
        "each fold contains every class with n >= {} ({} of {} classes)".format(
            TEAM_CV_N_SPLITS, n_eligible, len(classes)
        )
    )
    return messages


def load_and_validate_team_folds(
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    y_train: Sequence[str],
    path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, list]:
    folds = load_team_folds(path)
    messages = validate_team_folds(folds, train_ids, test_ids, y_train)
    return folds, messages


def ensure_team_folds(
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    y_train: Sequence[str],
    root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, list]:
    path = team_folds_path(root)
    expected = make_team_fold_assignments(train_ids, y_train)
    if path.is_file():
        folds = load_team_folds(path)
        messages = validate_team_folds(folds, train_ids, test_ids, y_train)
        return folds, messages
    write_team_folds(expected, path)
    folds = load_team_folds(path)
    messages = validate_team_folds(folds, train_ids, test_ids, y_train)
    messages.append("wrote {}".format(path))
    return folds, messages


def team_folds_sha256(root: Optional[Path] = None) -> str:
    return sha256_file(team_folds_path(root))
