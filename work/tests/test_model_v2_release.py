"""MODEL V2 release-contract tests. Does not train models."""

from pathlib import Path
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from merfish60.models import argmax_labels
from merfish60.official_contract import (
    SUBMISSION_COLUMNS,
    SUBMISSION_PRED_COL,
    allowed_labels,
    expected_test_cell_ids,
    verify_official_manifest,
)
from merfish60.reference import EXPECTED_MD5
from merfish60.validate_submission import validate_submission
from merfish60.v2c_blend import EXPERT_C_TEST, load_proba_frame

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "work" / "experiments" / "evidence" / "submissions" / "model_v2_candidate.csv"
METRICS = ROOT / "work" / "experiments" / "evidence" / "metrics" / "model_v2_metrics.json"
DOCS = ROOT / "work" / "docs" / "versions" / "model_v2.md"
README = ROOT / "README.md"
MODEL_V1_PATHS = [
    "work/src/merfish60/model_v1.py",
    "work/scripts/06_model_v1.py",
    "work/experiments/evidence/submissions/model_v1.csv",
]


def test_model_v2_candidate_contract_and_source():
    messages = validate_submission(CANDIDATE, ROOT)
    assert any("5000" in line for line in messages)
    raw = pd.read_csv(CANDIDATE, dtype=str, keep_default_na=False)
    assert list(raw.columns) == SUBMISSION_COLUMNS
    assert list(raw["Cell_ID"]) == expected_test_cell_ids(ROOT)
    class_names = allowed_labels(ROOT)
    assert set(raw[SUBMISSION_PRED_COL]).issubset(set(class_names))
    proba_path = ROOT / EXPERT_C_TEST
    if not proba_path.is_file():
        pytest.skip("regenerable V2-B probability dump not retained in the delivery tree")
    seg = load_proba_frame(proba_path, class_names)
    pred = argmax_labels(seg[class_names].to_numpy(dtype=float), class_names)
    assert list(raw[SUBMISSION_PRED_COL]) == list(pred)
    assert EXPERT_C_TEST.endswith("V2-B-REFONLY_test_probabilities_seg.csv.gz")


def test_model_v2_metrics_and_provenance():
    payload = json.loads(METRICS.read_text())
    assert payload["selected_experiment"] == "V2-B-REFONLY"
    assert payload["selected_blend_id"] == "C0"
    assert payload["oof_accuracy"] == 0.8212
    assert payload["correct"] == 4106
    assert payload["official_leaderboard_score"] == "Not submitted"
    assert payload["candidate_path"] in {
        "outputs/submissions/model_v2_candidate.csv",
        "work/experiments/evidence/submissions/model_v2_candidate.csv",
    }
    assert Path(payload["selected_test_probability_artifact"]).name == Path(EXPERT_C_TEST).name
    assert payload["reference_provenance"]["md5"] == EXPECTED_MD5
    assert payload["reference_provenance"]["n_usable_reference_rows"] == 136574
    assert payload["v2c_rejection"]["weight_search"] is False
    assert payload["three_expert_oracle_diagnostic_only"]["not_a_model_score"] is True
    assert payload["three_expert_oracle_diagnostic_only"]["oracle_correct"] == 4414


def test_readme_and_version_doc_links():
    readme = README.read_text()
    docs = DOCS.read_text()
    assert "work/docs/versions/model_v2.md" in readme
    assert "82.12%" in readme
    assert "Original Hackathon Information" not in readme
    assert readme.startswith("# WYH")
    assert docs.startswith("# MODEL V2 — Reference-Augmented MERFISH Cell-Type Classification")
    assert "ce06f62c0ec4973581dae17bb76f0cd9" in docs
    assert "0.8212" in docs
    assert "Not submitted" in docs


def test_official_prediction_immutable():
    diff = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert diff == "", diff


def test_official_manifest_for_v2_release():
    messages = verify_official_manifest(ROOT)
    assert any("verified" in line for line in messages)
