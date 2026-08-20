"""Separate MODEL V2 registry. Never writes experiments/registry.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from merfish60.io import repo_root


REGISTRY_V2_REL = "experiments/registry_v2.csv"
REGISTRY_V2_COLUMNS = [
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
    "fold_3_accuracy",
    "fold_4_accuracy",
    "oof_accuracy",
    "oof_accuracy_ei",
    "hard_bucket_accuracy",
    "neuron_accuracy",
    "glial_accuracy",
    "macro_f1",
    "runtime_seconds",
    "status",
    "notes",
    "manifest_sha256",
    "team_folds_sha256",
    "oof_path",
    "proba_path",
    "metrics_path",
    "compliance_status",
    "warnings",
]


class RegistryV2Error(Exception):
    pass


def registry_v2_path(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / REGISTRY_V2_REL


def load_registry_v2(root: Optional[Path] = None) -> pd.DataFrame:
    path = registry_v2_path(root)
    if not path.is_file():
        return pd.DataFrame(columns=REGISTRY_V2_COLUMNS)
    df = pd.read_csv(path, dtype=str)
    for col in REGISTRY_V2_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df.loc[:, REGISTRY_V2_COLUMNS]


def append_registry_v2_row(
    row: Mapping[str, object],
    root: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    df = load_registry_v2(root)
    run_id = str(row["run_id"])
    exists = bool(len(df) and (df["run_id"] == run_id).any())
    if exists and not overwrite:
        raise RegistryV2Error(
            "run_id {} already exists; pass --overwrite to replace that row only".format(
                run_id
            )
        )
    if exists:
        df = df.loc[df["run_id"] != run_id].copy()
    incoming = {col: "" for col in REGISTRY_V2_COLUMNS}
    for key, value in row.items():
        if key in incoming:
            incoming[key] = "" if value is None else str(value)
    df = pd.concat([df, pd.DataFrame([incoming], columns=REGISTRY_V2_COLUMNS)], ignore_index=True)
    path = registry_v2_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
