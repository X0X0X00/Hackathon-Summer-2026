"""Train MODEL-V1 (frozen YW-004 architecture) and write a personal candidate.

Writes only to outputs/submissions/model_v1.csv.
Does not read hidden test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from merfish60.io import N_GENES, N_TEST_CELLS, N_TRAIN_CELLS, load_dataset, validate_contract  # noqa: E402
from merfish60.model_v1 import (  # noqa: E402
    ARCHITECTURE,
    MODEL_NAME,
    SELECTED_FROM_RUN,
    SELECTED_OOF_ACCURACY,
    SELECTED_OOF_CORRECT,
    predict_model_v1,
    signature_summary_frame,
    train_model_v1,
    write_submission_csv,
)
from merfish60.official_contract import (  # noqa: E402
    SUBMISSION_COLUMNS,
    allowed_labels,
    expected_test_cell_ids,
    folds_sha256,
    git_commit,
    manifest_sha256,
    verify_official_manifest,
)
from merfish60.validate_submission import SubmissionContractError, validate_submission  # noqa: E402


SUBMISSION_REL = "outputs/submissions/model_v1.csv"
PROBA_REL = "outputs/probabilities/model_v1_test_probabilities.csv.gz"
METRICS_REL = "outputs/metrics/model_v1_metrics.json"
CLASS_ORDER_REL = "outputs/metrics/model_v1_class_order.json"
SUMMARY_REL = "outputs/metrics/model_v1_signature_summary.csv"


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError("not JSON serializable: {}".format(type(obj)))


def existing_artifacts() -> list:
    rels = [SUBMISSION_REL, PROBA_REL, METRICS_REL, CLASS_ORDER_REL, SUMMARY_REL]
    return [rel for rel in rels if (ROOT / rel).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing MODEL-V1 artifacts",
    )
    args = parser.parse_args()

    present = existing_artifacts()
    if present and not args.overwrite:
        print(
            "MODEL-V1 artifacts already exist; pass --overwrite to replace:\n  {}".format(
                "\n  ".join(present)
            ),
            file=sys.stderr,
        )
        return 1

    print("Verifying official data manifest...")
    for line in verify_official_manifest(ROOT):
        print(" - {}".format(line))

    data = load_dataset(ROOT)
    for line in validate_contract(data):
        print(" - {}".format(line))

    class_names = allowed_labels(ROOT)
    test_ids = expected_test_cell_ids(ROOT)
    if len(class_names) != 60:
        raise SystemExit("expected 60 training labels, got {}".format(len(class_names)))
    if len(test_ids) != N_TEST_CELLS:
        raise SystemExit("expected {} test ids, got {}".format(N_TEST_CELLS, len(test_ids)))

    print("Training MODEL-V1 on {} labeled training cells...".format(N_TRAIN_CELLS))
    t0 = time.perf_counter()
    fitted = train_model_v1(data, class_names)
    train_runtime = time.perf_counter() - t0
    print(
        " trained signatures={} deterministic={} specialists={} failures={}".format(
            len(fitted.routing_type),
            len(fitted.deterministic_signatures),
            len(fitted.specialists),
            len(fitted.specialist_fit_failures),
        )
    )

    print("Predicting official test Cell_IDs in meta_test order...")
    t1 = time.perf_counter()
    pred, proba, routing = predict_model_v1(fitted, data, test_ids)
    pred_runtime = time.perf_counter() - t1

    sub_path = ROOT / SUBMISSION_REL
    write_submission_csv(sub_path, test_ids, pred)

    proba_path = ROOT / PROBA_REL
    proba_path.parent.mkdir(parents=True, exist_ok=True)
    proba_frame = pd.DataFrame(proba, columns=list(class_names))
    proba_frame.insert(0, "Cell_ID", list(test_ids))
    proba_frame.to_csv(proba_path, index=False, compression="gzip")

    class_order_path = ROOT / CLASS_ORDER_REL
    class_order_path.parent.mkdir(parents=True, exist_ok=True)
    class_order_path.write_text(
        json.dumps(
            {
                "class_order": list(class_names),
                "n_classes": len(class_names),
                "source": "allowed_labels() from meta_train only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    summary = signature_summary_frame(fitted)
    summary_path = ROOT / SUMMARY_REL
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    n_det_sig = int((summary["routing_type"] == "deterministic").sum())
    n_amb_sig = int((summary["routing_type"] == "specialist").sum())
    metrics = {
        "model_name": MODEL_NAME,
        "architecture": ARCHITECTURE,
        "selected_oof_accuracy": SELECTED_OOF_ACCURACY,
        "selected_oof_correct": SELECTED_OOF_CORRECT,
        "selected_from_run": SELECTED_FROM_RUN,
        "current_git_commit": git_commit(ROOT),
        "official_data_manifest_hash": manifest_sha256(ROOT),
        "folds_file_hash": folds_sha256(ROOT),
        "training_rows": fitted.n_train,
        "test_rows": N_TEST_CELLS,
        "gene_count": N_GENES,
        "target_class_count": len(class_names),
        "number_of_full_train_signatures": int(len(fitted.routing_type)),
        "number_of_single-class_signatures": n_det_sig,
        "number_of_ambiguous_signatures": n_amb_sig,
        "number_of_specialists_trained": int(len(fitted.specialists)),
        "test_cells_routed_deterministically": routing["test_cells_routed_deterministically"],
        "test_cells_routed_to_specialists": routing["test_cells_routed_to_specialists"],
        "fallback_count": routing["fallback_count"],
        "specialist_fit_failures": fitted.specialist_fit_failures,
        "training_runtime_seconds": train_runtime,
        "prediction_runtime_seconds": pred_runtime,
        "submission_path": SUBMISSION_REL,
        "probability_path": PROBA_REL,
        "class_order_path": CLASS_ORDER_REL,
        "signature_summary_path": SUMMARY_REL,
        "lr_kwargs": dict(fitted.lr_kwargs),
        "n_features_per_specialist": fitted.n_features,
        "warnings": fitted.warnings,
        "note": "Hidden test labels were not used. Official test accuracy and leaderboard accuracy are unknown.",
    }
    metrics_path = ROOT / METRICS_REL
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=_json_default) + "\n")

    print("Validating {}".format(SUBMISSION_REL))
    try:
        messages = validate_submission(sub_path, ROOT)
    except SubmissionContractError as exc:
        print("SUBMISSION CONTRACT FAILED", file=sys.stderr)
        for line in exc.violations:
            print(" - {}".format(line), file=sys.stderr)
        return 1
    for line in messages:
        print(" - {}".format(line))

    print("MODEL-V1 submission columns: {}".format(SUBMISSION_COLUMNS))
    print("fallback_count={}".format(routing["fallback_count"]))
    print("Wrote {}".format(SUBMISSION_REL))
    print("Wrote {}".format(PROBA_REL))
    print("Wrote {}".format(METRICS_REL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
