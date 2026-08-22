"""Run leakage-safe YW-000 and YW-001 baselines on frozen folds."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORK / "src"))
from merfish60.io import repo_root  # noqa: E402
ROOT = repo_root()

from merfish60.cv import (  # noqa: E402
    CV_PROTOCOL,
    CV_RANDOM_STATE,
    FOLD_VALUES,
    load_folds,
    make_fold_assignments,
    validate_folds,
    write_folds,
)
from merfish60.io import load_dataset, validate_contract  # noqa: E402
from merfish60.metrics import summarize_oof  # noqa: E402

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
]


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"


def ensure_folds(data) -> pd.DataFrame:
    path = ROOT / "work" / "experiments" / "folds.csv"
    expected = make_fold_assignments(data.counts_train.index, data.y_train)
    if path.is_file():
        folds = load_folds(path)
        messages = validate_folds(
            folds,
            data.counts_train.index,
            data.counts_test.index,
            data.y_train,
        )
        print("Loaded existing folds.csv")
        for line in messages:
            print(" - {}".format(line))
        return folds
    write_folds(expected, path)
    folds = load_folds(path)
    messages = validate_folds(
        folds,
        data.counts_train.index,
        data.counts_test.index,
        data.y_train,
    )
    print("Wrote {}".format(path))
    for line in messages:
        print(" - {}".format(line))
    return folds


def majority_class(labels: pd.Series) -> str:
    counts = labels.value_counts()
    max_n = int(counts.max())
    tied = sorted(counts[counts == max_n].index.astype(str).tolist())
    return tied[0]


def run_yw000(y: pd.Series, folds: pd.DataFrame) -> Tuple[pd.DataFrame, dict, float, List[str]]:
    t0 = time.perf_counter()
    notes: List[str] = []
    aligned_fold = folds.set_index("Cell_ID").loc[y.index, "fold"]
    pred = pd.Series(index=y.index, dtype=object)
    fold_majorities = {}
    for fold_id in FOLD_VALUES:
        train_mask = aligned_fold != fold_id
        val_mask = aligned_fold == fold_id
        maj = majority_class(y.loc[train_mask])
        fold_majorities[fold_id] = maj
        pred.loc[val_mask] = maj
        notes.append("fold_{}_majority={}".format(fold_id, maj))
    runtime = time.perf_counter() - t0
    oof = pd.DataFrame(
        {
            "Cell_ID": y.index.astype(str),
            "true_label": y.to_numpy(),
            "predicted_label": pred.to_numpy(),
            "fold": aligned_fold.to_numpy(),
        }
    )
    labels = sorted(y.unique().tolist())
    metrics = summarize_oof(
        oof["true_label"], oof["predicted_label"], oof["fold"], labels=labels
    )
    metrics["fold_majority_class"] = {str(k): v for k, v in fold_majorities.items()}
    return oof, metrics, runtime, notes


def run_yw001(
    X: pd.DataFrame, y: pd.Series, folds: pd.DataFrame
) -> Tuple[pd.DataFrame, dict, float, List[str], dict]:
    t0 = time.perf_counter()
    notes: List[str] = []
    caught_warnings: List[str] = []
    aligned_fold = folds.set_index("Cell_ID").loc[y.index, "fold"]
    pred = pd.Series(index=y.index, dtype=object)
    X_log = np.log1p(X.to_numpy(dtype=np.float64))
    y_np = y.to_numpy()
    n_iter = {}
    model_kwargs = dict(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight=None,
        random_state=CV_RANDOM_STATE,
    )
    for fold_id in FOLD_VALUES:
        train_mask = (aligned_fold != fold_id).to_numpy()
        val_mask = (aligned_fold == fold_id).to_numpy()
        model = LogisticRegression(**model_kwargs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_log[train_mask], y_np[train_mask])
            for item in caught:
                text = "{}: {}".format(item.category.__name__, str(item.message))
                caught_warnings.append("fold_{}: {}".format(fold_id, text))
        pred.loc[val_mask] = model.predict(X_log[val_mask])
        iters = model.n_iter_
        n_iter[str(fold_id)] = int(np.max(iters)) if np.ndim(iters) else int(iters)
        if hasattr(model, "n_iter_"):
            max_seen = int(np.max(model.n_iter_))
            if max_seen >= model_kwargs["max_iter"]:
                notes.append(
                    "fold_{} reached max_iter={}".format(fold_id, model_kwargs["max_iter"])
                )
    runtime = time.perf_counter() - t0
    notes.extend(caught_warnings)
    oof = pd.DataFrame(
        {
            "Cell_ID": y.index.astype(str),
            "true_label": y.to_numpy(),
            "predicted_label": pred.to_numpy(),
            "fold": aligned_fold.to_numpy(),
        }
    )
    labels = sorted(y.unique().tolist())
    metrics = summarize_oof(
        oof["true_label"], oof["predicted_label"], oof["fold"], labels=labels
    )
    hyperparams = dict(model_kwargs)
    hyperparams["n_iter_per_fold"] = n_iter
    return oof, metrics, runtime, notes, hyperparams


def write_oof(path: Path, oof: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = oof.loc[:, ["Cell_ID", "true_label", "predicted_label", "fold"]].copy()
    out["Cell_ID"] = out["Cell_ID"].astype(str)
    out["fold"] = out["fold"].astype(int)
    out.to_csv(path, index=False)


def write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def registry_row(run_id: str, model: str, feature_set: str, metrics: dict, runtime: float, notes: str) -> dict:
    fold_acc = metrics["fold_accuracy"]
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "model": model,
        "feature_set": feature_set,
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "fold_0_accuracy": "{:.10f}".format(fold_acc[0]),
        "fold_1_accuracy": "{:.10f}".format(fold_acc[1]),
        "fold_2_accuracy": "{:.10f}".format(fold_acc[2]),
        "oof_accuracy": "{:.10f}".format(metrics["oof_accuracy"]),
        "macro_f1": "{:.10f}".format(metrics["macro_f1"]),
        "runtime_seconds": "{:.3f}".format(runtime),
        "status": "completed",
        "notes": notes,
    }


def upsert_registry(rows: List[dict]) -> None:
    path = ROOT / "work" / "experiments" / "registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = pd.read_csv(path, dtype=str)
        for col in REGISTRY_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing.loc[:, REGISTRY_COLUMNS]
    else:
        existing = pd.DataFrame(columns=REGISTRY_COLUMNS)
    incoming = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    if len(existing):
        keep = existing.loc[~existing["run_id"].isin(incoming["run_id"])]
        combined = pd.concat([keep, incoming], ignore_index=True)
    else:
        combined = incoming
    combined.to_csv(path, index=False)


def main() -> int:
    print("Loading and validating data...")
    data = load_dataset(ROOT)
    for line in validate_contract(data):
        print(" - {}".format(line))

    folds = ensure_folds(data)
    y = data.y_train.astype(str)
    labels = sorted(y.unique().tolist())

    print("\n=== YW-000 majority-class baseline ===")
    oof0, metrics0, runtime0, notes0 = run_yw000(y, folds)
    write_oof(ROOT / "work" / "outputs" / "oof" / "YW-000_oof.csv", oof0)
    payload0 = {
        "run_id": "YW-000",
        "model": "majority_class_per_training_fold",
        "feature_set": "none",
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "fold_accuracy": {str(k): v for k, v in metrics0["fold_accuracy"].items()},
        "oof_accuracy": metrics0["oof_accuracy"],
        "macro_f1": metrics0["macro_f1"],
        "per_class_recall": metrics0["per_class_recall"],
        "runtime_seconds": runtime0,
        "fold_majority_class": metrics0["fold_majority_class"],
        "notes": notes0,
        "n_classes": len(labels),
        "n_cells": int(len(y)),
    }
    write_metrics(ROOT / "work" / "outputs" / "metrics" / "YW-000_metrics.json", payload0)
    print("fold accuracies:", metrics0["fold_accuracy"])
    print("OOF accuracy: {:.6f}".format(metrics0["oof_accuracy"]))
    print("macro-F1: {:.6f}".format(metrics0["macro_f1"]))
    print("runtime_seconds: {:.3f}".format(runtime0))

    print("\n=== YW-001 log1p gene counts + L2 logistic regression ===")
    oof1, metrics1, runtime1, notes1, hyperparams = run_yw001(
        data.counts_train, y, folds
    )
    write_oof(ROOT / "work" / "outputs" / "oof" / "YW-001_oof.csv", oof1)
    payload1 = {
        "run_id": "YW-001",
        "model": "LogisticRegression",
        "feature_set": "log1p(raw gene counts); 200 genes; no metadata; no coordinates; no spatial features; no feature selection; no PCA; no class weighting",
        "feature_description": "log1p(raw counts) of the 200 official gene columns, in official column order",
        "cv_protocol": CV_PROTOCOL,
        "random_seed": CV_RANDOM_STATE,
        "fold_accuracy": {str(k): v for k, v in metrics1["fold_accuracy"].items()},
        "oof_accuracy": metrics1["oof_accuracy"],
        "macro_f1": metrics1["macro_f1"],
        "per_class_recall": metrics1["per_class_recall"],
        "runtime_seconds": runtime1,
        "model_hyperparameters": hyperparams,
        "n_features": int(data.counts_train.shape[1]),
        "n_classes": len(labels),
        "n_cells": int(len(y)),
        "warnings": notes1,
        "oof_path": "work/outputs/oof/YW-001_oof.csv",
    }
    write_metrics(ROOT / "work" / "outputs" / "metrics" / "YW-001_metrics.json", payload1)
    print("fold accuracies:", metrics1["fold_accuracy"])
    print("OOF accuracy: {:.6f}".format(metrics1["oof_accuracy"]))
    print("macro-F1: {:.6f}".format(metrics1["macro_f1"]))
    print("runtime_seconds: {:.3f}".format(runtime1))
    if notes1:
        print("warnings:")
        for line in notes1:
            print(" - {}".format(line))

    upsert_registry(
        [
            registry_row(
                "YW-000",
                "majority_class_per_training_fold",
                "none",
                metrics0,
                runtime0,
                "; ".join(notes0),
            ),
            registry_row(
                "YW-001",
                "LogisticRegression_l2_lbfgs",
                "log1p_raw_gene_counts_200",
                metrics1,
                runtime1,
                "; ".join(notes1) if notes1 else "converged",
            ),
        ]
    )
    print("\nUpdated experiments/registry.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
