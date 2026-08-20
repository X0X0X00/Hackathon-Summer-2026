"""MODEL-V1 full-train YW-004 leakage, contract, and artifact tests."""

from pathlib import Path
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from merfish60.io import N_GENES, N_TEST_CELLS, N_TRAIN_CELLS, TARGET_COL, load_dataset
from merfish60.model_v1 import train_model_v1
from merfish60.models import LR_KWARGS
from merfish60.official_contract import (
    SUBMISSION_COLUMNS,
    allowed_labels,
    expected_test_cell_ids,
    official_example_prediction_path,
    sha256_file,
)
from merfish60.signatures import SIGNATURE_FIELDS, is_deterministic_map

ROOT = Path(__file__).resolve().parents[1]
MODEL_V1_SOURCES = [
    ROOT / "src" / "merfish60" / "model_v1.py",
    ROOT / "scripts" / "06_model_v1.py",
]


@pytest.fixture(scope="module")
def data():
    return load_dataset(ROOT)


@pytest.fixture(scope="module")
def class_names():
    return allowed_labels(ROOT)


@pytest.fixture(scope="module")
def fitted(data, class_names):
    return train_model_v1(data, class_names)


def _require(rel: str) -> Path:
    path = ROOT / rel
    if not path.is_file():
        pytest.fail("missing {}; run .venv/bin/python scripts/06_model_v1.py".format(rel))
    return path


def test_training_uses_all_5000_labeled_training_cells(fitted, data):
    train_ids = [str(v) for v in data.counts_train.index]
    assert fitted.n_train == N_TRAIN_CELLS == 5000
    assert len(fitted.train_cell_ids) == 5000
    assert set(fitted.train_cell_ids) == set(train_ids)
    covered = [cid for ids in fitted.signature_train_ids.values() for cid in ids]
    assert len(covered) == 5000
    assert set(covered) == set(train_ids)


def test_model_v1_sources_never_use_meta_test_target():
    forbidden = (
        'meta_test[TARGET_COL]',
        'meta_test["MERFISH_cell_type_annotation"]',
        "meta_test['MERFISH_cell_type_annotation']",
        "y_test",
        "official_final_submission_path",
        "OFFICIAL_FINAL_SUBMISSION_REL",
    )
    for path in MODEL_V1_SOURCES:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, "{} mentions {}".format(path.name, token)
        assert "data.meta_test[{}]".format(TARGET_COL) not in text
        assert "data.meta_test[TARGET_COL]" not in text


def test_signature_mappings_learned_from_training_labels_only(fitted, data):
    y = data.y_train.astype(str)
    for sig, ids in fitted.signature_train_ids.items():
        observed = set(y.loc[ids].astype(str).tolist())
        assert fitted.candidate_map[sig] == observed
        if is_deterministic_map(observed):
            assert fitted.routing_type[sig] == "deterministic"
        else:
            assert fitted.routing_type[sig] == "specialist"


def test_specialist_training_ids_are_official_train_ids_only(fitted, data):
    train_ids = set(str(v) for v in data.counts_train.index)
    test_ids = set(str(v) for v in data.counts_test.index)
    for sig, ids in fitted.signature_train_ids.items():
        assert set(ids) <= train_ids
        assert set(ids).isdisjoint(test_ids)
        if fitted.routing_type[sig] == "specialist" and sig in fitted.specialists:
            model = fitted.specialists[sig]
            assert int(model.n_features_in_) == N_GENES


def test_no_test_cell_id_used_in_fitting(fitted, data):
    test_ids = set(str(v) for v in data.counts_test.index)
    assert set(fitted.train_cell_ids).isdisjoint(test_ids)
    for ids in fitted.signature_train_ids.values():
        assert set(ids).isdisjoint(test_ids)


def test_specialist_features_are_log1p_200_genes(fitted, data):
    assert fitted.n_features == N_GENES == 200
    assert fitted.gene_names == list(data.counts_train.columns)
    assert len(fitted.gene_names) == 200
    for sig, model in fitted.specialists.items():
        assert int(model.n_features_in_) == 200
        assert list(model.classes_)
        assert set(str(c) for c in model.classes_) <= fitted.candidate_map[sig]


def test_no_metadata_features_in_specialists(fitted):
    assert list(SIGNATURE_FIELDS) == [
        "Region",
        "Excitatory_vs_Inhibitory",
        "Segment",
    ]
    src = (ROOT / "src" / "merfish60" / "model_v1.py").read_text()
    assert "OneHotEncoder" not in src
    assert "canonicalize_metadata_frame" not in src
    for model in fitted.specialists.values():
        assert int(model.n_features_in_) == 200
    assert int(fitted.global_fallback.n_features_in_) == 200


def test_hyperparameters_match_yw004(fitted):
    assert fitted.lr_kwargs == dict(LR_KWARGS)
    assert fitted.lr_kwargs["penalty"] == "l2"
    assert fitted.lr_kwargs["C"] == 1.0
    assert fitted.lr_kwargs["solver"] == "lbfgs"
    assert fitted.lr_kwargs["max_iter"] == 2000
    assert fitted.lr_kwargs["class_weight"] is None
    for model in list(fitted.specialists.values()) + [fitted.global_fallback]:
        assert model.penalty == "l2"
        assert model.C == 1.0
        assert model.solver == "lbfgs"
        assert model.max_iter == 2000
        assert model.class_weight is None


def test_model_v1_csv_row_count_and_headers():
    path = _require("outputs/submissions/model_v1.csv")
    df = pd.read_csv(path, dtype={"Cell_ID": str})
    assert list(df.columns) == SUBMISSION_COLUMNS == [
        "Cell_ID",
        "MERFISH_cell_type_annotation.y",
    ]
    assert len(df) == N_TEST_CELLS == 5000


def test_model_v1_cell_id_order_matches_meta_test(data):
    path = _require("outputs/submissions/model_v1.csv")
    df = pd.read_csv(path, dtype={"Cell_ID": str})
    expected = expected_test_cell_ids(ROOT)
    assert list(df["Cell_ID"]) == expected
    assert list(df["Cell_ID"]) == [str(v) for v in data.meta_test.index.tolist()]


def test_model_v1_predictions_are_complete_official_labels(class_names):
    path = _require("outputs/submissions/model_v1.csv")
    df = pd.read_csv(path, dtype={"Cell_ID": str, "MERFISH_cell_type_annotation.y": str})
    pred = df["MERFISH_cell_type_annotation.y"]
    assert pred.notna().all()
    assert (pred.str.strip() == pred).all()
    assert (pred != "").all()
    assert set(pred) <= set(class_names)
    assert df["Cell_ID"].nunique() == 5000
    assert not df["Cell_ID"].duplicated().any()


def test_model_v1_probability_order_and_row_sums(class_names):
    order = json.loads(_require("outputs/metrics/model_v1_class_order.json").read_text())
    assert order["class_order"] == class_names
    assert order["n_classes"] == 60
    proba = pd.read_csv(
        _require("outputs/probabilities/model_v1_test_probabilities.csv.gz"),
        dtype={"Cell_ID": str},
        compression="gzip",
    )
    sub = pd.read_csv(
        _require("outputs/submissions/model_v1.csv"),
        dtype={"Cell_ID": str},
    )
    assert list(proba.columns) == ["Cell_ID"] + class_names
    assert len(proba) == 5000
    assert list(proba["Cell_ID"]) == list(sub["Cell_ID"])
    mass = proba[class_names].to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(mass, 1.0, atol=1e-6)


def test_example_prediction_csv_unchanged():
    example = official_example_prediction_path(ROOT)
    diff = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    )
    assert diff == b""
    assert example.is_file()
    assert sha256_file(example)


def test_metrics_record_frozen_selection_not_test_accuracy():
    metrics = json.loads(_require("outputs/metrics/model_v1_metrics.json").read_text())
    assert metrics["model_name"] == "MODEL-V1"
    assert metrics["selected_from_run"] == "YW-004"
    assert metrics["selected_oof_accuracy"] == 0.7598
    assert metrics["selected_oof_correct"] == 3799
    assert "test_accuracy" not in metrics
    assert "leaderboard_accuracy" not in metrics
    assert metrics["training_rows"] == 5000
    assert metrics["test_rows"] == 5000
    summary = pd.read_csv(_require("outputs/metrics/model_v1_signature_summary.csv"))
    assert list(summary.columns) == [
        "Region",
        "Excitatory_vs_Inhibitory",
        "Segment",
        "training_cell_count",
        "candidate_class_count",
        "candidate_classes",
        "routing_type",
    ]
    assert int(summary["training_cell_count"].sum()) == 5000
    assert set(summary["routing_type"]) <= {"deterministic", "specialist"}
