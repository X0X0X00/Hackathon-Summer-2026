"""Centralized official competition contract.

Paths, column names, and derived label/ID sets come from the four official
CSV files. Allowed labels are taken only from meta_train. Expected test
Cell_IDs and row count are taken only from meta_test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from merfish60.io import (
    TARGET_COL,
    _assert_cell_ids_lossless,
    load_meta,
    repo_root,
)


SUBMISSION_CELL_ID_COL = "Cell_ID"
SUBMISSION_PRED_COL = "MERFISH_cell_type_annotation.y"
SUBMISSION_COLUMNS = [SUBMISSION_CELL_ID_COL, SUBMISSION_PRED_COL]

OFFICIAL_EXAMPLE_PRED_REL = "prediction/prediction.csv"
OFFICIAL_FINAL_SUBMISSION_REL = "prediction/prediction.csv"
CANDIDATE_SUBMISSION_DIR_REL = "outputs/submissions"

OFFICIAL_COUNTS_TRAIN_REL = "data/counts_train.csv"
OFFICIAL_COUNTS_TEST_REL = "data/counts_test.csv"
OFFICIAL_META_TRAIN_REL = "data/meta_train.csv"
OFFICIAL_META_TEST_REL = "data/meta_test.csv"
OFFICIAL_DATA_RELS = (
    OFFICIAL_COUNTS_TRAIN_REL,
    OFFICIAL_COUNTS_TEST_REL,
    OFFICIAL_META_TRAIN_REL,
    OFFICIAL_META_TEST_REL,
)

YW002_METADATA_FIELDS = ["Region", "Excitatory_vs_Inhibitory", "Segment"]
YW002_FORBIDDEN_FIELDS = [
    "Datasets",
    "volume",
    "center_x",
    "center_y",
    "Gender",
    "Mouse_ID",
    "AP_position",
    "Section_ID",
]

MANIFEST_REL = "experiments/official_data_manifest.json"
FOLDS_REL = "experiments/folds.csv"


class OfficialContractError(Exception):
    """Raised when the official competition contract is violated."""


def official_path(relative: str, root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / relative


def official_counts_train_path(root: Optional[Path] = None) -> Path:
    return official_path(OFFICIAL_COUNTS_TRAIN_REL, root)


def official_counts_test_path(root: Optional[Path] = None) -> Path:
    return official_path(OFFICIAL_COUNTS_TEST_REL, root)


def official_meta_train_path(root: Optional[Path] = None) -> Path:
    return official_path(OFFICIAL_META_TRAIN_REL, root)


def official_meta_test_path(root: Optional[Path] = None) -> Path:
    return official_path(OFFICIAL_META_TEST_REL, root)


def official_example_prediction_path(root: Optional[Path] = None) -> Path:
    return official_path(OFFICIAL_EXAMPLE_PRED_REL, root)


def official_final_submission_path(root: Optional[Path] = None) -> Path:
    return official_path(OFFICIAL_FINAL_SUBMISSION_REL, root)


def candidate_submission_dir(root: Optional[Path] = None) -> Path:
    return official_path(CANDIDATE_SUBMISSION_DIR_REL, root)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Optional[Path] = None) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root or repo_root()),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "UNKNOWN"


def _csv_header_and_nrows(path: Path) -> Tuple[List[str], int]:
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        n_rows = sum(1 for _ in reader)
    return header, n_rows


def allowed_labels(root: Optional[Path] = None) -> List[str]:
    """Target classes derived only from official meta_train.csv, sorted."""
    meta = load_meta(official_meta_train_path(root))
    labels = meta[TARGET_COL].astype(str)
    if labels.isna().any() or labels.isin(["NA", "nan", "NaN"]).any():
        raise OfficialContractError("meta_train target contains missing labels")
    return sorted(labels.unique().tolist())


def expected_test_cell_ids(root: Optional[Path] = None) -> List[str]:
    """Test Cell_IDs in official meta_test.csv row order."""
    meta = load_meta(official_meta_test_path(root))
    ids = [str(v) for v in meta.index.tolist()]
    _assert_cell_ids_lossless(ids, source="meta_test Cell_ID")
    return ids


def expected_test_row_count(root: Optional[Path] = None) -> int:
    return len(expected_test_cell_ids(root))


def build_official_manifest(root: Optional[Path] = None) -> dict:
    root = root or repo_root()
    files: Dict[str, dict] = {}
    for rel in OFFICIAL_DATA_RELS:
        path = official_path(rel, root)
        header, n_rows = _csv_header_and_nrows(path)
        files[rel] = {
            "relative_path": rel,
            "sha256": sha256_file(path),
            "byte_size": path.stat().st_size,
            "n_data_rows": n_rows,
            "column_names": header,
        }
    labels = allowed_labels(root)
    test_ids = expected_test_cell_ids(root)
    train_meta = load_meta(official_meta_train_path(root))
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(root),
        "target_column": TARGET_COL,
        "n_train_labels": len(labels),
        "allowed_labels_sorted": labels,
        "n_train_cell_ids": int(len(train_meta)),
        "n_test_cell_ids": int(len(test_ids)),
        "files": files,
    }
    return payload


def write_official_manifest(root: Optional[Path] = None) -> Path:
    root = root or repo_root()
    payload = build_official_manifest(root)
    path = official_path(MANIFEST_REL, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_official_manifest(root: Optional[Path] = None) -> dict:
    path = official_path(MANIFEST_REL, root)
    if not path.is_file():
        raise OfficialContractError("missing official data manifest: {}".format(path))
    return json.loads(path.read_text())


def verify_official_manifest(root: Optional[Path] = None) -> List[str]:
    """Re-hash official CSVs and fail if they differ from the saved manifest."""
    root = root or repo_root()
    saved = load_official_manifest(root)
    messages = []
    current = build_official_manifest(root)
    for rel in OFFICIAL_DATA_RELS:
        old = saved["files"][rel]
        new = current["files"][rel]
        for key in ("sha256", "byte_size", "n_data_rows", "column_names"):
            if old[key] != new[key]:
                raise OfficialContractError(
                    "official file {} changed ({}: saved={} current={})".format(
                        rel, key, old[key], new[key]
                    )
                )
        messages.append(
            "{} ok sha256={} n_rows={}".format(rel, new["sha256"], new["n_data_rows"])
        )
    if saved.get("n_train_labels") != current["n_train_labels"]:
        raise OfficialContractError("n_train_labels changed in official train meta")
    if saved.get("n_test_cell_ids") != current["n_test_cell_ids"]:
        raise OfficialContractError("n_test_cell_ids changed in official test meta")
    messages.append("official data manifest verified")
    return messages


def manifest_sha256(root: Optional[Path] = None) -> str:
    return sha256_file(official_path(MANIFEST_REL, root))


def folds_sha256(root: Optional[Path] = None) -> str:
    return sha256_file(official_path(FOLDS_REL, root))
