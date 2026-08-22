"""BRIDGE-YW004-5F: frozen YW-004 architecture on the team-compatible 5-fold file.

This is not a new model version. Does not modify MODEL V1, experiments/folds.csv,
or experiments/registry.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORK / "src"))
from merfish60.io import repo_root  # noqa: E402
ROOT = repo_root()

from merfish60.io import load_dataset, validate_contract  # noqa: E402
from merfish60.models import LR_KWARGS  # noqa: E402
from merfish60.official_contract import (  # noqa: E402
    allowed_labels,
    git_commit,
    manifest_sha256,
    verify_official_manifest,
)
from merfish60.registry_v2 import RegistryV2Error, append_registry_v2_row  # noqa: E402
from merfish60.spatial_features import ei_of_label_from_train  # noqa: E402
from merfish60.team_cv import (  # noqa: E402
    TEAM_CV_PROTOCOL,
    TEAM_CV_RANDOM_STATE,
    ensure_team_folds,
    team_folds_sha256,
)
from merfish60.v2_metrics import (  # noqa: E402
    slice_metrics,
    write_confusion,
    write_json,
    write_oof,
    write_proba,
)
from merfish60.yw004_cv import run_yw004_oof  # noqa: E402


RUN_ID = "BRIDGE-YW004-5F"
COMPLIANCE = "EXTERNAL_REFERENCE_PERMISSION_REQUIRED"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("Verifying official data manifest...", flush=True)
    for line in verify_official_manifest(ROOT):
        print(" - {}".format(line), flush=True)

    data = load_dataset(ROOT)
    for line in validate_contract(data):
        print(" - {}".format(line), flush=True)

    class_names = allowed_labels(ROOT)
    folds, fold_messages = ensure_team_folds(
        data.counts_train.index,
        data.counts_test.index,
        data.y_train,
        root=ROOT,
    )
    print("Team-compatible 5-fold protocol:", TEAM_CV_PROTOCOL, flush=True)
    for line in fold_messages:
        print(" - {}".format(line), flush=True)

    t0 = time.perf_counter()
    pred, proba, diag = run_yw004_oof(data, folds, class_names)
    runtime = time.perf_counter() - t0

    ei_of_label = ei_of_label_from_train(data.meta_train, class_names)
    slices = slice_metrics(diag["y_true"], pred, data.meta_train, class_names, ei_of_label)

    oof_rel = "work/outputs/oof/{}_oof.csv".format(RUN_ID)
    proba_rel = "work/outputs/probabilities/{}_oof_probabilities.csv.gz".format(RUN_ID)
    metrics_rel = "work/outputs/metrics/{}_metrics.json".format(RUN_ID)
    cm_rel = "work/outputs/metrics/{}_confusion.csv".format(RUN_ID)

    write_oof(ROOT / oof_rel, diag["cell_ids"], diag["y_true"], pred, diag["fold_ids"])
    write_proba(ROOT / proba_rel, diag["cell_ids"], proba, class_names)
    write_confusion(ROOT / cm_rel, diag["y_true"], pred, class_names)

    payload = {
        "run_id": RUN_ID,
        "is_model_version": False,
        "model": (
            "frozen YW-004 per-signature gene-only L2 logistic specialists "
            "evaluated on the team-compatible 5-fold protocol"
        ),
        "feature_set": (
            "log1p(raw gene counts, 200) inside each fold-learned signature group; "
            "no extra metadata features in specialists"
        ),
        "cv_protocol": TEAM_CV_PROTOCOL,
        "random_seed": TEAM_CV_RANDOM_STATE,
        "model_hyperparameters": dict(LR_KWARGS),
        "class_order": class_names,
        "fold_accuracy": diag["fold_accuracy"],
        "oof_accuracy": diag["oof_accuracy"],
        "macro_f1": diag["macro_f1"],
        "per_class_recall": diag["per_class_recall"],
        "deterministic_signature_accuracy": diag["deterministic_signature_accuracy"],
        "deterministic_signature_n": diag["deterministic_signature_n"],
        "ambiguous_signature_accuracy": diag["ambiguous_signature_accuracy"],
        "ambiguous_signature_n": diag["ambiguous_signature_n"],
        "n_specialists_trained_per_fold": diag["n_specialists_trained_per_fold"],
        "fallback_counts": diag["fallback_counts"],
        "fallback_n": diag["fallback_n"],
        "fallback_examples": diag["fallback_examples"],
        "warnings": diag["warnings"],
        "n_cells": diag["n_cells"],
        "n_classes": diag["n_classes"],
        "runtime_seconds": runtime,
        "original_yw004_3fold_oof_accuracy": 0.7598,
        "original_yw004_3fold_note": "secondary reference only; not the V2 comparison point",
        "oof_path": oof_rel,
        "proba_path": proba_rel,
        "metrics_path": metrics_rel,
        "confusion_path": cm_rel,
        "compliance_status": COMPLIANCE,
        "git_commit": git_commit(ROOT),
        "manifest_sha256": manifest_sha256(ROOT),
        "team_folds_sha256": team_folds_sha256(ROOT),
        "timestamp": utc_now(),
    }
    payload.update(slices)
    write_json(ROOT / metrics_rel, payload)

    notes = "BRIDGE only; not a new model version"
    try:
        append_registry_v2_row(
            {
                "run_id": RUN_ID,
                "timestamp": payload["timestamp"],
                "git_commit": payload["git_commit"],
                "model": payload["model"],
                "feature_set": payload["feature_set"],
                "cv_protocol": TEAM_CV_PROTOCOL,
                "random_seed": TEAM_CV_RANDOM_STATE,
                "fold_0_accuracy": payload["fold_accuracy"].get("0"),
                "fold_1_accuracy": payload["fold_accuracy"].get("1"),
                "fold_2_accuracy": payload["fold_accuracy"].get("2"),
                "fold_3_accuracy": payload["fold_accuracy"].get("3"),
                "fold_4_accuracy": payload["fold_accuracy"].get("4"),
                "oof_accuracy": payload["oof_accuracy"],
                "oof_accuracy_ei": "",
                "hard_bucket_accuracy": payload["hard_bucket_accuracy"],
                "neuron_accuracy": payload["neuron_accuracy"],
                "glial_accuracy": payload["glial_accuracy"],
                "macro_f1": payload["macro_f1"],
                "runtime_seconds": runtime,
                "status": "completed",
                "notes": notes,
                "manifest_sha256": payload["manifest_sha256"],
                "team_folds_sha256": payload["team_folds_sha256"],
                "oof_path": oof_rel,
                "proba_path": proba_rel,
                "metrics_path": metrics_rel,
                "compliance_status": COMPLIANCE,
                "warnings": "; ".join(payload["warnings"][:8]) if payload["warnings"] else "none",
            },
            root=ROOT,
            overwrite=args.overwrite,
        )
    except RegistryV2Error as exc:
        print("registry_v2: {}".format(exc), flush=True)
        return 1

    print("BRIDGE-YW004-5F OOF accuracy={:.6f}".format(payload["oof_accuracy"]), flush=True)
    for fold_id in sorted(payload["fold_accuracy"], key=int):
        print(
            "  fold {}: {:.6f}".format(fold_id, payload["fold_accuracy"][fold_id]),
            flush=True,
        )
    print(
        "  hard-bucket={:.6f} neuron={:.6f} glial={:.6f}".format(
            payload["hard_bucket_accuracy"],
            payload["neuron_accuracy"],
            payload["glial_accuracy"],
        ),
        flush=True,
    )
    print("wrote {}".format(metrics_rel), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
