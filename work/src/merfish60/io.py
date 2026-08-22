"""Official data I/O and contract checks.

Cell_ID values are 19-digit integers that do not survive float64 round-trip.
They are always read as strings and never cast through float.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


TARGET_COL = "MERFISH_cell_type_annotation"
N_TRAIN_CELLS = 5000
N_TEST_CELLS = 5000
N_GENES = 200
N_CLASSES = 60
CELL_ID_N_DIGITS = 19
# Official CSVs store the row label as an unnamed first column.
ID_COL_POSITION = 0

META_COLUMNS = [
    "Datasets",
    "volume",
    "center_x",
    "center_y",
    TARGET_COL,
    "Region",
    "Excitatory_vs_Inhibitory",
    "Segment",
    "Gender",
    "Mouse_ID",
    "AP_position",
    "Section_ID",
]


class DataContractError(Exception):
    """Raised when an official data file violates the expected contract."""


@dataclass(frozen=True)
class MerfishData:
    counts_train: pd.DataFrame
    counts_test: pd.DataFrame
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame

    @property
    def genes(self) -> List[str]:
        return list(self.counts_train.columns)

    @property
    def y_train(self) -> pd.Series:
        return self.meta_train[TARGET_COL]


def repo_root() -> Path:
    """Walk upward to the repository root that holds official data/.

    The implementation lives under work/src/, so a fixed parent count is
    fragile. Official competition inputs remain at the repository root.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "data" / "counts_train.csv").is_file() and (
            candidate / "Data.Description.md"
        ).is_file():
            return candidate
    raise DataContractError(
        "could not locate repository root from {}".format(here)
    )


def work_dir(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / "work"


def experiments_dir(root: Optional[Path] = None) -> Path:
    return work_dir(root) / "experiments"


def generated_outputs_dir(root: Optional[Path] = None) -> Path:
    return work_dir(root) / "outputs"


def evidence_dir(root: Optional[Path] = None) -> Path:
    return experiments_dir(root) / "evidence"


def data_dir(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / "data"


def _read_csv_string_index(path: Path) -> pd.DataFrame:
    """Read a CSV whose first column is Cell_ID, preserving it as str."""
    if not path.is_file():
        raise DataContractError("missing file: {}".format(path))

    df = pd.read_csv(
        path,
        index_col=ID_COL_POSITION,
        dtype={ID_COL_POSITION: str},
        float_precision="high",
    )
    df.index = pd.Index(df.index.astype(str), name="Cell_ID")
    _assert_cell_ids_lossless(df.index, source=str(path))
    return df


def _assert_cell_ids_lossless(ids: Iterable[str], source: str) -> None:
    values = pd.Index(list(ids), dtype=object)
    if values.hasnans or any(v in {"", "nan", "NaN", "None"} for v in values):
        raise DataContractError("{}: Cell_ID contains missing values".format(source))
    if any(str(v).endswith(".0") for v in values):
        raise DataContractError(
            "{}: Cell_ID looks float-cast (value ends with .0)".format(source)
        )
    if any("e" in str(v).lower() or "." in str(v) for v in values):
        raise DataContractError(
            "{}: Cell_ID is not a plain digit string".format(source)
        )
    bad_len = [v for v in values if not str(v).isdigit() or len(str(v)) != CELL_ID_N_DIGITS]
    if bad_len:
        raise DataContractError(
            "{}: Cell_ID is not a {}-digit integer string (example={!r})".format(
                source, CELL_ID_N_DIGITS, bad_len[0]
            )
        )
    as_int = [int(v) for v in values]
    as_float = [int(float(v)) for v in values]
    n_lossy = sum(a != b for a, b in zip(as_int, as_float))
    if n_lossy == 0:
        # Still never store them as float; this only documents the hazard.
        pass
    else:
        # Expected: most IDs are float64-unsafe. Confirm we still hold the
        # original digit string rather than the rounded float value.
        pass


def _assert_unique(ids: Sequence[str], source: str) -> None:
    n = len(ids)
    n_unique = len(set(ids))
    if n != n_unique:
        raise DataContractError(
            "{}: duplicate Cell_ID values (n={}, unique={})".format(source, n, n_unique)
        )


def _assert_nonnegative_integer_counts(counts: pd.DataFrame, source: str) -> None:
    values = counts.to_numpy()
    if values.size == 0:
        raise DataContractError("{}: empty count matrix".format(source))
    if np.isnan(values).any():
        raise DataContractError("{}: gene counts contain missing values".format(source))
    if np.issubdtype(values.dtype, np.floating):
        if not np.all(np.isfinite(values)):
            raise DataContractError("{}: gene counts contain non-finite values".format(source))
        if not np.allclose(values, np.round(values)):
            raise DataContractError("{}: gene counts are not integers".format(source))
        values = np.round(values)
    if np.issubdtype(values.dtype, np.signedinteger) or np.issubdtype(
        values.dtype, np.unsignedinteger
    ) or np.issubdtype(values.dtype, np.floating):
        if (values < 0).any():
            raise DataContractError("{}: gene counts contain negatives".format(source))
    else:
        raise DataContractError(
            "{}: unexpected gene-count dtype {}".format(source, values.dtype)
        )


def _assert_aligned(counts: pd.DataFrame, meta: pd.DataFrame, split: str) -> None:
    if not counts.index.equals(meta.index):
        same_set = set(counts.index) == set(meta.index)
        raise DataContractError(
            "{} counts/meta Cell_ID alignment failed (same_set={}, "
            "counts_n={}, meta_n={})".format(
                split, same_set, len(counts), len(meta)
            )
        )


def load_counts(path: Path) -> pd.DataFrame:
    df = _read_csv_string_index(path)
    if df.shape[1] != N_GENES:
        raise DataContractError(
            "{}: expected {} gene columns, found {}".format(path, N_GENES, df.shape[1])
        )
    numeric = df.apply(pd.to_numeric, errors="raise")
    _assert_nonnegative_integer_counts(numeric, str(path))
    # Preserve integer dtype when possible without going through Cell_ID float.
    if np.isfinite(numeric.to_numpy()).all() and np.allclose(
        numeric.to_numpy(), np.round(numeric.to_numpy())
    ):
        numeric = numeric.round().astype(np.int64)
    return numeric


def load_meta(path: Path) -> pd.DataFrame:
    df = _read_csv_string_index(path)
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError("{}: missing meta columns {}".format(path, missing))
    return df.loc[:, META_COLUMNS]


def load_dataset(root: Optional[Path] = None) -> MerfishData:
    d = data_dir(root)
    counts_train = load_counts(d / "counts_train.csv")
    counts_test = load_counts(d / "counts_test.csv")
    meta_train = load_meta(d / "meta_train.csv")
    meta_test = load_meta(d / "meta_test.csv")
    return MerfishData(
        counts_train=counts_train,
        counts_test=counts_test,
        meta_train=meta_train,
        meta_test=meta_test,
    )


def validate_contract(data: MerfishData) -> List[str]:
    """Validate the official data contract. Returns human-readable OK lines.

    Raises DataContractError on the first violation.
    """
    messages: List[str] = []

    if data.counts_train.shape != (N_TRAIN_CELLS, N_GENES):
        raise DataContractError(
            "counts_train shape {} != ({}, {})".format(
                data.counts_train.shape, N_TRAIN_CELLS, N_GENES
            )
        )
    if data.counts_test.shape != (N_TEST_CELLS, N_GENES):
        raise DataContractError(
            "counts_test shape {} != ({}, {})".format(
                data.counts_test.shape, N_TEST_CELLS, N_GENES
            )
        )
    if data.meta_train.shape[0] != N_TRAIN_CELLS:
        raise DataContractError(
            "meta_train n_rows {} != {}".format(data.meta_train.shape[0], N_TRAIN_CELLS)
        )
    if data.meta_test.shape[0] != N_TEST_CELLS:
        raise DataContractError(
            "meta_test n_rows {} != {}".format(data.meta_test.shape[0], N_TEST_CELLS)
        )
    messages.append(
        "shapes: counts_train={}, counts_test={}, meta_train={}, meta_test={}".format(
            data.counts_train.shape,
            data.counts_test.shape,
            data.meta_train.shape,
            data.meta_test.shape,
        )
    )

    if list(data.counts_train.columns) != list(data.counts_test.columns):
        raise DataContractError("train/test gene columns differ in name or order")
    if len(data.genes) != N_GENES:
        raise DataContractError("expected {} genes, found {}".format(N_GENES, len(data.genes)))
    if len(set(data.genes)) != N_GENES:
        raise DataContractError("duplicate gene names")
    messages.append("gene columns identical: n={}".format(N_GENES))

    _assert_unique(data.counts_train.index, "counts_train")
    _assert_unique(data.counts_test.index, "counts_test")
    _assert_unique(data.meta_train.index, "meta_train")
    _assert_unique(data.meta_test.index, "meta_test")
    messages.append("no duplicate Cell_IDs in counts or meta")

    _assert_aligned(data.counts_train, data.meta_train, "train")
    _assert_aligned(data.counts_test, data.meta_test, "test")
    messages.append("train counts/meta Cell_ID order aligned")
    messages.append("test counts/meta Cell_ID order aligned")

    overlap = set(data.counts_train.index) & set(data.counts_test.index)
    if overlap:
        raise DataContractError(
            "train/test Cell_ID overlap: n={}".format(len(overlap))
        )
    messages.append("train/test Cell_IDs disjoint")

    y_train = data.y_train
    if y_train.isna().any() or (y_train.astype(str) == "NA").any():
        # Official missing token is parsed as NA/NaN by pandas; train must have labels.
        n_missing = int(y_train.isna().sum() + (y_train.astype(str) == "NA").sum())
        raise DataContractError("meta_train target has {} missing labels".format(n_missing))
    n_classes = y_train.nunique()
    if n_classes != N_CLASSES:
        raise DataContractError(
            "expected {} training classes, found {}".format(N_CLASSES, n_classes)
        )
    messages.append(
        "train target {}: n_classes={}, missing=0".format(TARGET_COL, n_classes)
    )

    y_test = data.meta_test[TARGET_COL]
    hidden = y_test.isna() | y_test.astype(str).isin(["NA", "nan", "NaN", "<NA>"])
    if int(hidden.sum()) != N_TEST_CELLS:
        raise DataContractError(
            "meta_test target should be fully hidden; n_hidden={}".format(int(hidden.sum()))
        )
    messages.append("test target fully hidden (all missing)")

    _assert_nonnegative_integer_counts(data.counts_train, "counts_train")
    _assert_nonnegative_integer_counts(data.counts_test, "counts_test")
    messages.append("gene counts are nonnegative integers")

    _assert_cell_ids_lossless(data.counts_train.index, "counts_train")
    _assert_cell_ids_lossless(data.counts_test.index, "counts_test")
    messages.append("Cell_ID stored as 19-digit strings (not float)")

    return messages
