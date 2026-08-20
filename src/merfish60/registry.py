"""Append-only experiment registry. Never silently overwrite an existing run."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from merfish60.io import repo_root


REGISTRY_REL = "experiments/registry.csv"
FROZEN_RUN_IDS = ("YW-000", "YW-001", "YW-002", "YW-003", "YW-004")
BASELINE_RUN_IDS = FROZEN_RUN_IDS

REGISTRY_COLUMNS = [
    "run_id",
    "timestamp",
    "git_commit",
    "model",
    "feature_set",
    "cv_protocol",
    "random_seed",
    "fold_0_accuracy",
    "fold_1_accuracy",
    "fold_2_accuracy",
    "oof_accuracy",
    "macro_f1",
    "runtime_seconds",
    "status",
    "notes",
    "manifest_sha256",
    "folds_sha256",
    "oof_path",
    "proba_path",
    "metrics_path",
    "conclusion",
]


class RegistryError(Exception):
    pass


def registry_path(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / REGISTRY_REL


def load_registry(root: Optional[Path] = None) -> pd.DataFrame:
    path = registry_path(root)
    if not path.is_file():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    df = pd.read_csv(path, dtype=str)
    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df.loc[:, REGISTRY_COLUMNS]


def append_registry_row(
    row: Mapping[str, object],
    root: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    df = load_registry(root)
    run_id = str(row["run_id"])
    exists = bool(len(df) and (df["run_id"] == run_id).any())
    if exists and run_id in FROZEN_RUN_IDS:
        raise RegistryError("refusing to alter frozen run {}".format(run_id))
    if exists and not overwrite:
        raise RegistryError(
            "run_id {} already exists; pass --overwrite to replace that row only".format(
                run_id
            )
        )
    if exists:
        df = df.loc[df["run_id"] != run_id].copy()
    incoming = {col: "" for col in REGISTRY_COLUMNS}
    for key, value in row.items():
        if key in incoming:
            incoming[key] = "" if value is None else str(value)
    df = pd.concat([df, pd.DataFrame([incoming], columns=REGISTRY_COLUMNS)], ignore_index=True)
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
