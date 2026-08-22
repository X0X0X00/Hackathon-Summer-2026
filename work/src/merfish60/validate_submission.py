"""Official submission-file contract validator.

Validates a candidate CSV against the organizer example schema and the
Cell_ID order of official meta_test.csv. Never rewrites the candidate.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Sequence

from merfish60.io import CELL_ID_N_DIGITS, _assert_cell_ids_lossless, load_meta
from merfish60.official_contract import (
    SUBMISSION_CELL_ID_COL,
    SUBMISSION_COLUMNS,
    SUBMISSION_PRED_COL,
    allowed_labels,
    expected_test_cell_ids,
    expected_test_row_count,
    official_meta_train_path,
    repo_root,
)


class SubmissionContractError(Exception):
    """One or more submission-contract rules failed."""

    def __init__(self, violations: Sequence[str]):
        self.violations = list(violations)
        super().__init__("\n".join(self.violations))


def _rule(number: int, message: str) -> str:
    return "Rule {}: {}".format(number, message)


def validate_submission(path: Path, root: Optional[Path] = None) -> List[str]:
    """Return OK messages. Raise SubmissionContractError with numbered rules."""
    root = root or repo_root()
    path = Path(path)
    violations: List[str] = []

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SubmissionContractError([_rule(1, "cannot read file: {}".format(exc))])

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionContractError([_rule(1, "file is not valid UTF-8: {}".format(exc))])

    try:
        rows = list(csv.reader(text.splitlines()))
    except csv.Error as exc:
        raise SubmissionContractError(
            [_rule(1, "file is not valid comma-separated CSV: {}".format(exc))]
        )

    if not rows:
        violations.append(_rule(1, "CSV is empty"))
        raise SubmissionContractError(violations)

    header = rows[0]
    data_rows = rows[1:]

    if len(header) != 2:
        violations.append(
            _rule(2, "expected exactly 2 columns, found {}".format(len(header)))
        )
    if header != SUBMISSION_COLUMNS:
        violations.append(
            _rule(
                3,
                "columns must be exactly {} in that order; found {}".format(
                    SUBMISSION_COLUMNS, header
                ),
            )
        )
    if any(col == "" or str(col).startswith("Unnamed") for col in header):
        violations.append(
            _rule(13, "unnamed or extra index column present in header {}".format(header))
        )

    expected_ids = expected_test_cell_ids(root)
    expected_n = expected_test_row_count(root)
    labels_allowed = set(allowed_labels(root))
    train_ids = set(str(v) for v in load_meta(official_meta_train_path(root)).index.tolist())

    if len(data_rows) != expected_n:
        violations.append(
            _rule(
                4,
                "expected {} data rows matching meta_test.csv, found {}".format(
                    expected_n, len(data_rows)
                ),
            )
        )

    cell_ids: List[str] = []
    preds: List[str] = []
    extra_col_rows = 0
    short_rows = 0
    for i, row in enumerate(data_rows, start=2):
        if len(row) > 2:
            extra_col_rows += 1
        if len(row) < 2:
            short_rows += 1
            continue
        cell_ids.append(str(row[0]))
        preds.append(str(row[1]) if len(row) > 1 else "")

    if extra_col_rows:
        violations.append(
            _rule(
                14,
                "extra columns on {} data row(s); expected exactly 2 fields".format(
                    extra_col_rows
                ),
            )
        )
    if short_rows:
        violations.append(
            _rule(2, "{} data row(s) have fewer than 2 fields".format(short_rows))
        )

    if cell_ids:
        if any(v.endswith(".0") or "e" in v.lower() or "." in v for v in cell_ids):
            violations.append(
                _rule(6, "Cell_ID is not a lossless digit string (float-like values present)")
            )
        try:
            _assert_cell_ids_lossless(cell_ids, source=str(path))
        except Exception as exc:
            violations.append(_rule(6, "Cell_ID not lossless: {}".format(exc)))
        if any(len(v) != CELL_ID_N_DIGITS or not v.isdigit() for v in cell_ids):
            violations.append(
                _rule(
                    5,
                    "Cell_ID must be read as a {}-digit string".format(CELL_ID_N_DIGITS),
                )
            )
        if len(cell_ids) != len(set(cell_ids)):
            violations.append(_rule(7, "duplicate Cell_ID values are present"))
        if set(cell_ids) != set(expected_ids) and len(cell_ids) == expected_n:
            violations.append(
                _rule(8, "Cell_ID set does not exactly match meta_test.csv")
            )
        if cell_ids != expected_ids and len(cell_ids) == expected_n and set(cell_ids) == set(expected_ids):
            violations.append(
                _rule(9, "Cell_ID order does not exactly match meta_test.csv")
            )
        if cell_ids != expected_ids and set(cell_ids) != set(expected_ids) and len(cell_ids) == expected_n:
            # order check is implied by set mismatch; still record order if lengths match
            pass
        train_in_candidate = set(cell_ids) & train_ids
        if train_in_candidate:
            violations.append(
                _rule(
                    10,
                    "train Cell_IDs appear in the candidate (n={})".format(
                        len(train_in_candidate)
                    ),
                )
            )

    n_missing_pred = 0
    n_illegal = 0
    illegal_examples = []
    for pred in preds:
        if pred is None or str(pred).strip() == "":
            n_missing_pred += 1
            continue
        if str(pred) != str(pred).strip():
            n_missing_pred += 1
            continue
        if pred not in labels_allowed:
            n_illegal += 1
            if len(illegal_examples) < 5:
                illegal_examples.append(pred)
    if n_missing_pred:
        violations.append(
            _rule(
                11,
                "{} prediction(s) are missing, blank, or whitespace-only".format(
                    n_missing_pred
                ),
            )
        )
    if n_illegal:
        violations.append(
            _rule(
                12,
                "{} prediction(s) are not in the meta_train label set (examples={})".format(
                    n_illegal, illegal_examples
                ),
            )
        )

    if violations:
        raise SubmissionContractError(violations)

    ok = [
        "UTF-8 CSV with header {}".format(SUBMISSION_COLUMNS),
        "n_data_rows={} matches meta_test.csv".format(len(data_rows)),
        "Cell_ID lossless unique strings in meta_test order",
        "no train Cell_IDs present",
        "all {} predictions in the {}-class train label set".format(
            len(preds), len(labels_allowed)
        ),
    ]
    return ok
