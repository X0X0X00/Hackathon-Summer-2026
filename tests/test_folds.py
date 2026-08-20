"""Frozen fold-file integrity tests."""

from pathlib import Path

import pandas as pd
import pytest

from merfish60.cv import (
    CV_N_SPLITS,
    FOLD_VALUES,
    folds_path,
    load_folds,
    make_fold_assignments,
    validate_folds,
)
from merfish60.io import load_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return load_dataset(ROOT)


@pytest.fixture(scope="module")
def folds():
    path = folds_path(ROOT)
    if not path.is_file():
        pytest.fail("experiments/folds.csv is missing; run scripts/01_baseline.py first")
    return load_folds(path)


def test_fold_file_columns(folds):
    assert list(folds.columns) == ["Cell_ID", "fold"]


def test_every_training_id_once(data, folds):
    train_ids = data.counts_train.index.astype(str)
    fold_ids = folds["Cell_ID"].astype(str)
    assert not fold_ids.duplicated().any()
    assert len(folds) == len(train_ids)
    assert set(fold_ids) == set(train_ids)


def test_no_duplicate_fold_assignments(folds):
    assert not folds["Cell_ID"].duplicated().any()
    assert folds.groupby("Cell_ID").size().max() == 1


def test_fold_values(folds):
    assert tuple(sorted(folds["fold"].unique().tolist())) == FOLD_VALUES
    assert set(folds["fold"].tolist()) == set(FOLD_VALUES)


def test_no_test_ids_in_folds(data, folds):
    test_ids = set(data.counts_test.index.astype(str))
    fold_ids = set(folds["Cell_ID"].astype(str))
    assert fold_ids.isdisjoint(test_ids)


def test_folds_match_frozen_protocol(data, folds):
    expected = make_fold_assignments(data.counts_train.index, data.y_train)
    merged = folds.merge(expected, on="Cell_ID", suffixes=("_saved", "_expected"))
    assert (merged["fold_saved"] == merged["fold_expected"]).all()


def test_each_class_in_each_fold_when_possible(data, folds):
    y = data.y_train.astype(str)
    labeled = folds.merge(y.rename("label").reset_index(), on="Cell_ID", how="left")
    classes = sorted(y.unique().tolist())
    for fold_id in FOLD_VALUES:
        present = set(labeled.loc[labeled["fold"] == fold_id, "label"])
        for cls in classes:
            n = int((y == cls).sum())
            if n >= CV_N_SPLITS:
                assert cls in present, "class {} missing from fold {}".format(cls, fold_id)


def test_validate_folds_helper(data, folds):
    messages = validate_folds(
        folds,
        data.counts_train.index,
        data.counts_test.index,
        data.y_train,
    )
    assert any("exactly once" in m for m in messages)
