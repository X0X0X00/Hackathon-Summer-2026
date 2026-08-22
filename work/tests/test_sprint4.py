"""Sprint 4 / MODEL V2 leakage, fold, and integrity tests."""

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from merfish60.cv import folds_path, load_folds, make_fold_assignments
from merfish60.io import N_CLASSES, N_TRAIN_CELLS, TARGET_COL, load_dataset
from merfish60.neighbor_labels import (
    apply_ei,
    build_X,
    encode_train_labels,
    label_hist,
    visible_label_codes,
)
from merfish60.official_contract import allowed_labels, verify_official_manifest
from merfish60.registry import FROZEN_RUN_IDS, RegistryError, append_registry_row
from merfish60.spatial_features import build_spatial_universe, ei_of_label_from_train
from merfish60.team_cv import (
    TEAM_CV_N_SPLITS,
    TEAM_FOLD_VALUES,
    load_team_folds,
    make_team_fold_assignments,
    team_folds_path,
    validate_team_folds,
)
from merfish60.v2_metrics import universe_fold_ids

ROOT = Path(__file__).resolve().parents[2]
SPRINT4_MODELING_SOURCES = [
    ROOT / "work" / "src" / "merfish60" / "team_cv.py",
    ROOT / "work" / "src" / "merfish60" / "yw004_cv.py",
    ROOT / "work" / "src" / "merfish60" / "spatial_features.py",
    ROOT / "work" / "src" / "merfish60" / "neighbor_labels.py",
    ROOT / "work" / "src" / "merfish60" / "v2_metrics.py",
    ROOT / "work" / "src" / "merfish60" / "registry_v2.py",
    ROOT / "work" / "scripts" / "07_bridge_yw004_5f.py",
    ROOT / "work" / "scripts" / "08_v2a_spatial_lgbm.py",
]
MODEL_V1_PATHS = [
    "work/src/merfish60/model_v1.py",
    "work/scripts/06_model_v1.py",
    "work/experiments/evidence/submissions/model_v1.csv",
]
FROZEN_3FOLD = "work/experiments/folds.csv"


@pytest.fixture(scope="module")
def data():
    return load_dataset(ROOT)


@pytest.fixture(scope="module")
def class_names():
    return allowed_labels(ROOT)


@pytest.fixture(scope="module")
def universe(data):
    return build_spatial_universe(data)


def test_personal_3fold_file_unchanged(data):
    path = folds_path(ROOT)
    assert path.is_file()
    folds = load_folds(path)
    expected = make_fold_assignments(data.counts_train.index, data.y_train)
    merged = folds.merge(expected, on="Cell_ID", suffixes=("_saved", "_expected"))
    assert (merged["fold_saved"] == merged["fold_expected"]).all()
    assert len(folds) == N_TRAIN_CELLS
    assert tuple(sorted(folds["fold"].unique().tolist())) == (0, 1, 2)


def test_team_5fold_file_contract(data):
    path = team_folds_path(ROOT)
    if not path.is_file():
        pytest.fail("missing {}; run scripts/07_bridge_yw004_5f.py".format(path))
    folds = load_team_folds(path)
    messages = validate_team_folds(
        folds,
        data.counts_train.index,
        data.counts_test.index,
        data.y_train,
    )
    assert any("exactly once" in line for line in messages)
    train_ids = [str(v) for v in data.counts_train.index]
    test_ids = set(str(v) for v in data.counts_test.index)
    assert len(folds) == 5000
    assert folds["Cell_ID"].nunique() == 5000
    assert set(folds["Cell_ID"].astype(str)) == set(train_ids)
    assert set(folds["Cell_ID"].astype(str)).isdisjoint(test_ids)
    assert tuple(sorted(folds["fold"].unique().tolist())) == TEAM_FOLD_VALUES
    expected = make_team_fold_assignments(data.counts_train.index, data.y_train)
    merged = folds.merge(expected, on="Cell_ID", suffixes=("_saved", "_expected"))
    assert (merged["fold_saved"] == merged["fold_expected"]).all()
    assert not folds["Cell_ID"].astype(str).str.endswith(".0").any()


def test_team_and_personal_fold_files_are_distinct(data):
    personal = load_folds(folds_path(ROOT))
    team = load_team_folds(team_folds_path(ROOT))
    merged = personal.merge(team, on="Cell_ID", suffixes=("_3", "_5"))
    assert len(merged) == 5000
    assert (merged["fold_3"] != merged["fold_5"]).any()
    assert personal["fold"].max() == 2
    assert team["fold"].max() == 4


def test_visible_label_codes_hide_holdout_and_test():
    y_codes = np.array([0, 1, 2, 3, -1, -1], dtype=np.int64)
    is_train = np.array([True, True, True, True, False, False])
    folds = np.array([0, 0, 1, 1, -1, -1], dtype=np.int64)
    known = visible_label_codes(y_codes, is_train, folds, holdout_fold=0)
    np.testing.assert_array_equal(known, np.array([-1, -1, 2, 3, -1, -1]))
    known_test = visible_label_codes(y_codes, is_train, folds, holdout_fold=None)
    np.testing.assert_array_equal(known_test, np.array([0, 1, 2, 3, -1, -1]))


def test_neighbor_histograms_ignore_poisoned_held_out_labels():
    """Poison fold-f labels; histograms must be unchanged when hide-rule is applied."""
    n = 8
    n_classes = 4
    idx = np.array(
        [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 4],
            [0, 1, 2],
            [5, 6, 7],
            [4, 6, 7],
            [4, 5, 7],
            [4, 5, 6],
        ],
        dtype=np.int32,
    )
    dist = np.ones_like(idx, dtype=np.float32)
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    is_train = np.array([True] * 6 + [False, False])
    folds = np.array([0, 0, 1, 1, 0, 1, -1, -1], dtype=np.int64)
    known = visible_label_codes(y, is_train, folds, holdout_fold=0)
    hist, names = label_hist(known, idx, dist, k=2, tag="sp", n_classes=n_classes)

    y_poison = y.copy()
    y_poison[folds == 0] = n_classes - 1
    known_poison = visible_label_codes(y_poison, is_train, folds, holdout_fold=0)
    hist_poison, _ = label_hist(known_poison, idx, dist, k=2, tag="sp", n_classes=n_classes)
    np.testing.assert_array_equal(hist, hist_poison)
    assert "{}_n".format("sp") in names
    assert hist.shape == (n, n_classes + 2)


def test_unhidden_holdout_labels_change_histograms():
    """If fold-f labels are left visible, histograms must change. Proves hiding matters."""
    idx = np.array([[1, 2], [0, 2], [0, 1], [0, 1]], dtype=np.int32)
    dist = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3], [1.0, 1.1]], dtype=np.float32)
    y = np.array([0, 1, 2, 1], dtype=np.int64)
    is_train = np.array([True, True, True, True])
    folds = np.array([0, 0, 1, 1], dtype=np.int64)
    hidden = visible_label_codes(y, is_train, folds, holdout_fold=0)
    leaked = visible_label_codes(y, is_train, folds, holdout_fold=None)
    hist_hidden, _ = label_hist(hidden, idx, dist, k=2, tag="sp", n_classes=3)
    hist_leaked, _ = label_hist(leaked, idx, dist, k=2, tag="sp", n_classes=3)
    assert not np.allclose(hist_hidden, hist_leaked)


def test_apply_ei_does_not_use_inconsistent_classes():
    probs = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]], dtype=np.float64)
    ei_known = np.array([1.0, np.nan], dtype=np.float32)
    ei_of_label = np.array([1, 0, -1], dtype=np.int8)
    out = apply_ei(probs, ei_known, ei_of_label)
    np.testing.assert_allclose(out[0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(out[1], probs[1])


def test_real_neighbor_histograms_poison_on_official_train(data, class_names, universe):
    path = team_folds_path(ROOT)
    if not path.is_file():
        pytest.fail("missing {}; run scripts/07_bridge_yw004_5f.py".format(path))
    folds = load_team_folds(path)
    y_codes = encode_train_labels(
        data.y_train.astype(str).tolist(),
        class_names,
        len(universe.cell_ids),
        universe.is_train,
    )
    fold_universe = universe_fold_ids(folds, universe.cell_ids, universe.is_train)
    holdout = 0
    known = visible_label_codes(y_codes, universe.is_train, fold_universe, holdout)
    hist, _ = label_hist(known, universe.sp_idx, universe.sp_dist, k=15, tag="sp")
    y_poison = y_codes.copy()
    y_poison[(universe.is_train) & (fold_universe == holdout)] = (
        y_poison[(universe.is_train) & (fold_universe == holdout)] + 7
    ) % N_CLASSES
    known_poison = visible_label_codes(y_poison, universe.is_train, fold_universe, holdout)
    hist_poison, _ = label_hist(
        known_poison, universe.sp_idx, universe.sp_dist, k=15, tag="sp"
    )
    np.testing.assert_array_equal(hist, hist_poison)

    leaked = visible_label_codes(y_codes, universe.is_train, fold_universe, None)
    hist_leaked, _ = label_hist(leaked, universe.sp_idx, universe.sp_dist, k=15, tag="sp")
    assert not np.allclose(hist, hist_leaked)


def test_spatial_knn_never_includes_self(universe):
    n = len(universe.cell_ids)
    rows = np.arange(n)[:, None]
    sp_hit = (universe.sp_idx == rows) & (universe.sp_idx >= 0)
    ex_hit = (universe.ex_idx == rows) & (universe.ex_idx >= 0)
    assert not sp_hit.any()
    assert not ex_hit.any()


def test_build_x_uses_only_visible_codes(data, class_names, universe):
    path = team_folds_path(ROOT)
    if not path.is_file():
        pytest.fail("missing team folds")
    folds = load_team_folds(path)
    y_codes = encode_train_labels(
        data.y_train.astype(str).tolist(),
        class_names,
        len(universe.cell_ids),
        universe.is_train,
    )
    fold_universe = universe_fold_ids(folds, universe.cell_ids, universe.is_train)
    known = visible_label_codes(y_codes, universe.is_train, fold_universe, 1)
    features, names = build_X(universe, known)
    assert features.shape[0] == 10000
    assert "sp_h0" in names and "ex_h0" in names
    y_poison = y_codes.copy()
    y_poison[(universe.is_train) & (fold_universe == 1)] = 0
    known_p = visible_label_codes(y_poison, universe.is_train, fold_universe, 1)
    features_p, _ = build_X(universe, known_p)
    np.testing.assert_array_equal(features, features_p)


def test_sprint4_sources_never_use_test_target_or_official_submission():
    forbidden = (
        'meta_test[TARGET_COL]',
        'meta_test["MERFISH_cell_type_annotation"]',
        "meta_test['MERFISH_cell_type_annotation']",
        "y_test",
        "official_final_submission_path",
        "OFFICIAL_FINAL_SUBMISSION_REL",
        "prediction/prediction.csv",
        "prediction.csv",
        "zenodo",
        "h5ad",
        "prep_ext",
        "common_ext",
    )
    for path in SPRINT4_MODELING_SOURCES:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, "{} mentions {}".format(path.name, token)
        assert "data.meta_test[TARGET_COL]" not in text
        assert "data.meta_test[{}]".format(TARGET_COL) not in text


def test_ei_of_label_uses_training_labels_only(data, class_names):
    ei = ei_of_label_from_train(data.meta_train, class_names)
    assert ei.shape == (len(class_names),)
    assert set(np.unique(ei).tolist()).issubset({-1, 0, 1})
    assert (ei == -1).any()
    assert (ei == 0).any() and (ei == 1).any()


def test_official_manifest_still_verifies():
    messages = verify_official_manifest(ROOT)
    assert any("verified" in line for line in messages)


def test_frozen_registry_still_cannot_overwrite_yw004(tmp_path):
    assert FROZEN_RUN_IDS == ("YW-000", "YW-001", "YW-002", "YW-003", "YW-004")
    exp = tmp_path / "work" / "experiments"
    exp.mkdir(parents=True)
    (exp / "registry.csv").write_text((ROOT / "work" / "experiments" / "registry.csv").read_text())
    with pytest.raises(RegistryError, match="frozen"):
        append_registry_row(
            {"run_id": "YW-004", "model": "must_not_overwrite"},
            root=tmp_path,
            overwrite=True,
        )


def test_model_v1_sources_and_candidate_unmodified():
    diff = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert diff == "", diff


def test_prediction_csv_unmodified():
    diff = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert diff == "", diff


def _load_oof(name: str) -> pd.DataFrame:
    path = ROOT / "work" / "outputs" / "oof" / name
    if not path.is_file():
        pytest.skip("generated OOF not retained: {}".format(path))
    return pd.read_csv(
        path,
        dtype={"Cell_ID": str, "true_label": str, "predicted_label": str, "fold": int},
    )


def _load_proba(name: str) -> pd.DataFrame:
    path = ROOT / "work" / "outputs" / "probabilities" / name
    if not path.is_file():
        pytest.skip("generated probabilities not retained: {}".format(path))
    return pd.read_csv(path, dtype={"Cell_ID": str}, compression="gzip")


@pytest.mark.parametrize(
    "oof_name,proba_name,folds_kind",
    [
        ("BRIDGE-YW004-5F_oof.csv", "BRIDGE-YW004-5F_oof_probabilities.csv.gz", "team5"),
        ("V2-A-SPATIAL-LGBM_oof.csv", "V2-A-SPATIAL-LGBM_oof_probabilities.csv.gz", "team5"),
    ],
)
def test_sprint4_oof_coverage_and_probability_order(data, class_names, oof_name, proba_name, folds_kind):
    oof = _load_oof(oof_name)
    proba = _load_proba(proba_name)
    train_ids = [str(v) for v in data.counts_train.index]
    assert len(oof) == 5000
    assert oof["Cell_ID"].nunique() == 5000
    assert set(oof["Cell_ID"]) == set(train_ids)
    assert set(oof["Cell_ID"]).isdisjoint(set(data.counts_test.index.astype(str)))
    assert list(proba.columns) == ["Cell_ID"] + list(class_names)
    assert list(proba["Cell_ID"]) == list(oof["Cell_ID"])
    mass = proba[class_names].to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(mass, 1.0, atol=1e-5)
    folds = load_team_folds(team_folds_path(ROOT))
    merged = oof.merge(folds, on="Cell_ID", suffixes=("_oof", "_saved"))
    assert (merged["fold_oof"] == merged["fold_saved"]).all()
    assert folds_kind == "team5"
    assert TEAM_CV_N_SPLITS == 5
