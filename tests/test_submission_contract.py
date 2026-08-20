"""Submission-contract tests. The organizer example is read-only."""

from pathlib import Path

import pandas as pd
import pytest

from merfish60.io import load_dataset
from merfish60.official_contract import SUBMISSION_COLUMNS, allowed_labels, expected_test_cell_ids
from merfish60.validate_submission import SubmissionContractError, validate_submission

ROOT = Path(__file__).resolve().parents[1]


def _write_candidate(path: Path, cell_ids, labels, extra_col=None, header=None):
    header = header or SUBMISSION_COLUMNS
    frame = pd.DataFrame({header[0]: cell_ids, header[1]: labels})
    if extra_col is not None:
        frame["extra"] = extra_col
    frame.to_csv(path, index=False)


def test_organizer_example_passes_readonly():
    messages = validate_submission(ROOT / "prediction" / "prediction.csv", ROOT)
    assert messages
    data = load_dataset(ROOT)
    raw = pd.read_csv(
        ROOT / "prediction" / "prediction.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert list(raw.columns) == SUBMISSION_COLUMNS
    assert len(raw) == len(data.meta_test)
    assert list(raw["Cell_ID"]) == [str(v) for v in data.meta_test.index.tolist()]


def test_exact_row_count(tmp_path):
    ids = expected_test_cell_ids(ROOT)
    labels = allowed_labels(ROOT)
    pred = [labels[0]] * (len(ids) - 1)
    path = tmp_path / "short.csv"
    _write_candidate(path, ids[:-1], pred)
    with pytest.raises(SubmissionContractError) as exc:
        validate_submission(path, ROOT)
    assert any(v.startswith("Rule 4:") for v in exc.value.violations)


def test_wrong_headers(tmp_path):
    ids = expected_test_cell_ids(ROOT)[:3]
    path = tmp_path / "bad_header.csv"
    pd.DataFrame({"Cell_ID": ids, "label": ["astrocyte_1"] * 3}).to_csv(path, index=False)
    with pytest.raises(SubmissionContractError) as exc:
        validate_submission(path, ROOT)
    assert any(v.startswith("Rule 3:") for v in exc.value.violations)


def test_missing_label_rejection(tmp_path):
    ids = expected_test_cell_ids(ROOT)
    labels = [allowed_labels(ROOT)[0]] * len(ids)
    labels[10] = "   "
    path = tmp_path / "blank.csv"
    _write_candidate(path, ids, labels)
    with pytest.raises(SubmissionContractError) as exc:
        validate_submission(path, ROOT)
    assert any(v.startswith("Rule 11:") for v in exc.value.violations)


def test_invalid_label_rejection(tmp_path):
    ids = expected_test_cell_ids(ROOT)
    labels = [allowed_labels(ROOT)[0]] * len(ids)
    labels[0] = "not_a_real_cell_type"
    path = tmp_path / "illegal.csv"
    _write_candidate(path, ids, labels)
    with pytest.raises(SubmissionContractError) as exc:
        validate_submission(path, ROOT)
    assert any(v.startswith("Rule 12:") for v in exc.value.violations)


def test_duplicate_id_rejection(tmp_path):
    ids = expected_test_cell_ids(ROOT)
    ids = list(ids)
    ids[1] = ids[0]
    labels = [allowed_labels(ROOT)[0]] * len(ids)
    path = tmp_path / "dups.csv"
    _write_candidate(path, ids, labels)
    with pytest.raises(SubmissionContractError) as exc:
        validate_submission(path, ROOT)
    assert any(v.startswith("Rule 7:") for v in exc.value.violations)


def test_extra_column_rejection(tmp_path):
    ids = expected_test_cell_ids(ROOT)
    labels = [allowed_labels(ROOT)[0]] * len(ids)
    path = tmp_path / "extra.csv"
    _write_candidate(path, ids, labels, extra_col=["x"] * len(ids))
    with pytest.raises(SubmissionContractError) as exc:
        validate_submission(path, ROOT)
    assert any(v.startswith("Rule 2:") or v.startswith("Rule 14:") or v.startswith("Rule 13:") for v in exc.value.violations)


def test_train_id_rejection(tmp_path):
    data = load_dataset(ROOT)
    test_ids = [str(v) for v in data.meta_test.index.tolist()]
    train_ids = [str(v) for v in data.meta_train.index.tolist()]
    swapped = list(test_ids)
    swapped[0] = train_ids[0]
    labels = [allowed_labels(ROOT)[0]] * len(swapped)
    path = tmp_path / "train_id.csv"
    _write_candidate(path, swapped, labels)
    with pytest.raises(SubmissionContractError) as exc:
        validate_submission(path, ROOT)
    assert any(v.startswith("Rule 10:") for v in exc.value.violations)


def test_order_mismatch_rejection(tmp_path):
    ids = list(expected_test_cell_ids(ROOT))
    ids[0], ids[1] = ids[1], ids[0]
    labels = [allowed_labels(ROOT)[0]] * len(ids)
    path = tmp_path / "order.csv"
    _write_candidate(path, ids, labels)
    with pytest.raises(SubmissionContractError) as exc:
        validate_submission(path, ROOT)
    assert any(v.startswith("Rule 9:") for v in exc.value.violations)
