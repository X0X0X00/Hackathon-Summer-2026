"""Sprint 5 / V2-C fixed-blend selection tests. No model fitting."""

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from merfish60.io import N_CLASSES, TARGET_COL, load_dataset
from merfish60.models import argmax_labels
from merfish60.official_contract import (
    SUBMISSION_COLUMNS,
    allowed_labels,
    expected_test_cell_ids,
    verify_official_manifest,
)
from merfish60.team_cv import load_team_folds, team_folds_path
from merfish60.validate_submission import validate_submission
from merfish60.v2c_blend import (
    BLEND_IDS,
    EXPERT_A_OOF,
    EXPERT_B_OOF,
    EXPERT_C_OOF,
    FIXED_BLENDS,
    V2B_OOF_TARGET,
    assert_oof_alignment,
    load_proba_frame,
    mix_fixed,
)

ROOT = Path(__file__).resolve().parents[1]
V2C_SOURCES = [
    ROOT / "src" / "merfish60" / "v2c_blend.py",
    ROOT / "scripts" / "11_v2c_select.py",
]
MODEL_V1_PATHS = [
    "src/merfish60/model_v1.py",
    "scripts/06_model_v1.py",
    "outputs/submissions/model_v1.csv",
]


@pytest.fixture(scope="module")
def data():
    return load_dataset(ROOT)


@pytest.fixture(scope="module")
def class_names():
    return allowed_labels(ROOT)


def test_fixed_blend_space_is_exactly_c0_to_c4():
    assert BLEND_IDS == ("C0", "C1", "C2", "C3", "C4")
    assert set(FIXED_BLENDS) == set(BLEND_IDS)
    assert FIXED_BLENDS["C0"] == (0.0, 0.0, 1.0)
    assert FIXED_BLENDS["C1"] == (0.0, 0.25, 0.75)
    assert FIXED_BLENDS["C2"] == (0.15, 0.15, 0.70)
    assert FIXED_BLENDS["C3"] == (0.20, 0.20, 0.60)
    assert abs(sum(FIXED_BLENDS["C4"]) - 1.0) < 1e-12
    for w in FIXED_BLENDS["C4"]:
        assert abs(w - 1.0 / 3.0) < 1e-12


def test_v2c_sources_have_no_weight_search_or_training():
    forbidden = (
        "lgb.train",
        "lightgbm",
        "LogisticRegression",
        "Nelder-Mead",
        "nelder",
        "scipy.optimize",
        "convex_weight_grid",
        "select_ensemble_weights",
        "grid_search",
        "GridSearch",
        "scripts/07_bridge",
        "scripts/08_v2a",
        "scripts/09_v2b",
        "y_test",
        "prediction/prediction.csv",
    )
    for path in V2C_SOURCES:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, "{} mentions {}".format(path.name, token)


def test_abc_oof_alignment_and_row_sums(data, class_names):
    frames = {
        "A": load_proba_frame(ROOT / EXPERT_A_OOF, class_names),
        "B": load_proba_frame(ROOT / EXPERT_B_OOF, class_names),
        "C": load_proba_frame(ROOT / EXPERT_C_OOF, class_names),
    }
    folds = load_team_folds(team_folds_path(ROOT))
    messages = assert_oof_alignment(
        frames, data.counts_train.index, folds, class_names
    )
    assert any("5000" in line for line in messages)
    ids = frames["A"]["Cell_ID"].tolist()
    assert ids == frames["B"]["Cell_ID"].tolist() == frames["C"]["Cell_ID"].tolist()
    assert ids == [str(v) for v in data.counts_train.index]
    fold_map = folds.set_index("Cell_ID")["fold"]
    for name in ("A", "B", "C"):
        oof = pd.read_csv(
            ROOT / "outputs/oof/{}_oof.csv".format(
                {
                    "A": "BRIDGE-YW004-5F",
                    "B": "V2-A-SPATIAL-LGBM",
                    "C": "V2-B-REFONLY",
                }[name]
            ),
            dtype={"Cell_ID": str, "fold": int},
        )
        assert list(oof["Cell_ID"]) == ids
        assert list(oof["fold"]) == [int(fold_map.loc[cid]) for cid in ids]
        assert list(frames[name].columns) == ["Cell_ID"] + list(class_names)
        mass = frames[name][class_names].to_numpy(dtype=float).sum(axis=1)
        assert np.allclose(mass, 1.0, atol=1e-5)
        assert frames[name][class_names].shape[1] == N_CLASSES


def test_c0_reproduces_v2b(data, class_names):
    a = load_proba_frame(ROOT / EXPERT_A_OOF, class_names)
    b = load_proba_frame(ROOT / EXPERT_B_OOF, class_names)
    c = load_proba_frame(ROOT / EXPERT_C_OOF, class_names)
    mixed = mix_fixed(
        a[class_names].to_numpy(dtype=float),
        b[class_names].to_numpy(dtype=float),
        c[class_names].to_numpy(dtype=float),
        FIXED_BLENDS["C0"],
    )
    pred = argmax_labels(mixed, class_names)
    true = data.y_train.loc[a["Cell_ID"]].astype(str).to_numpy()
    acc = float(np.mean(pred == true))
    assert abs(acc - V2B_OOF_TARGET) < 1e-6
    np.testing.assert_allclose(mixed, c[class_names].to_numpy(dtype=float), atol=1e-12)


def test_model_v1_folds_and_prediction_immutable():
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--",
            *MODEL_V1_PATHS,
            "experiments/folds.csv",
            "prediction/prediction.csv",
        ],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert diff == "", diff


def test_official_manifest():
    messages = verify_official_manifest(ROOT)
    assert any("verified" in line for line in messages)


def test_model_v2_candidate_contract_if_present():
    path = ROOT / "outputs" / "submissions" / "model_v2_candidate.csv"
    if not path.is_file():
        pytest.skip("MODEL V2 candidate not written yet")
    messages = validate_submission(path, ROOT)
    assert messages
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(raw.columns) == SUBMISSION_COLUMNS
    assert list(raw["Cell_ID"]) == expected_test_cell_ids(ROOT)
    assert set(raw[SUBMISSION_COLUMNS[1]]).issubset(set(allowed_labels(ROOT)))
    official = pd.read_csv(
        ROOT / "prediction" / "prediction.csv", dtype=str, keep_default_na=False
    )
    assert list(raw["Cell_ID"]) == list(official["Cell_ID"])
    assert TARGET_COL not in list(raw.columns)


def test_selected_test_proba_row_sums_if_present(class_names):
    path = ROOT / "outputs" / "probabilities" / "MODEL-V2_test_probabilities.csv.gz"
    if not path.is_file():
        pytest.skip("MODEL V2 test probabilities not written yet")
    frame = pd.read_csv(path, dtype={"Cell_ID": str}, compression="gzip")
    assert list(frame["Cell_ID"]) == expected_test_cell_ids(ROOT)
    assert list(frame.columns) == ["Cell_ID"] + list(class_names)
    mass = frame[class_names].to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(mass, 1.0, atol=1e-5)
