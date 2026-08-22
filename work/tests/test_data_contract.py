"""Official data-contract tests. These read but never modify source CSVs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merfish60.io import (
    CELL_ID_N_DIGITS,
    N_CLASSES,
    N_GENES,
    N_TEST_CELLS,
    N_TRAIN_CELLS,
    TARGET_COL,
    load_dataset,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def data():
    return load_dataset(ROOT)


def test_validate_contract_passes(data):
    messages = validate_contract(data)
    assert messages


def test_cell_id_integrity(data):
    for name, idx in [
        ("counts_train", data.counts_train.index),
        ("counts_test", data.counts_test.index),
        ("meta_train", data.meta_train.index),
        ("meta_test", data.meta_test.index),
    ]:
        assert idx.dtype == object or str(idx.dtype).startswith("string")
        values = idx.astype(str)
        assert values.str.fullmatch(r"\d{" + str(CELL_ID_N_DIGITS) + r"}").all(), name
        assert not values.str.endswith(".0").any(), name
        assert not values.duplicated().any(), name
        # Held IDs must be the original digit strings, which do not survive
        # float64 round-trip on this dataset.
        original = [int(v) for v in values]
        rounded = [int(float(v)) for v in values]
        assert original != rounded, name
        assert list(pd.Index(values)) == list(values)


def test_train_meta_alignment(data):
    assert data.counts_train.index.equals(data.meta_train.index)
    assert list(data.counts_train.index) == list(data.meta_train.index)
    assert len(data.counts_train) == N_TRAIN_CELLS
    assert len(data.meta_train) == N_TRAIN_CELLS


def test_test_meta_alignment(data):
    assert data.counts_test.index.equals(data.meta_test.index)
    assert list(data.counts_test.index) == list(data.meta_test.index)
    assert len(data.counts_test) == N_TEST_CELLS


def test_train_test_disjointness(data):
    overlap = set(data.counts_train.index) & set(data.counts_test.index)
    assert overlap == set()
    overlap_meta = set(data.meta_train.index) & set(data.meta_test.index)
    assert overlap_meta == set()


def test_gene_column_consistency(data):
    assert list(data.counts_train.columns) == list(data.counts_test.columns)
    assert len(data.genes) == N_GENES
    assert len(set(data.genes)) == N_GENES


def test_target_behavior(data):
    y_train = data.y_train
    assert y_train.notna().all()
    assert y_train.nunique() == N_CLASSES
    y_test = data.meta_test[TARGET_COL]
    assert int(y_test.isna().sum()) == N_TEST_CELLS


def test_nonnegative_integer_counts(data):
    for frame in (data.counts_train, data.counts_test):
        values = frame.to_numpy()
        assert np.issubdtype(values.dtype, np.integer)
        assert values.min() >= 0
