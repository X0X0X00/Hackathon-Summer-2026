"""Sprint 3 leakage, coverage, probability, fold, and official-data tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merfish60.cv import FOLD_VALUES, load_folds
from merfish60.ensemble import (
    convex_weight_grid,
    cross_fitted_ensemble,
    select_ensemble_weights,
)
from merfish60.hard_bucket import CANDIDATE_SPECS, select_hard_bucket_model
from merfish60.io import load_dataset
from merfish60.models import train_val_masks
from merfish60.official_contract import allowed_labels, verify_official_manifest
from merfish60.registry import FROZEN_RUN_IDS, RegistryError, append_registry_row
from merfish60.signatures import missing_bucket_key, signatures_from_meta

ROOT = Path(__file__).resolve().parents[2]
SPRINT3_MODELING_SOURCES = [
    ROOT / "work" / "scripts" / "04_sprint3_experiments.py",
    ROOT / "work" / "scripts" / "05_sprint3_comparison.py",
    ROOT / "work" / "src" / "merfish60" / "ensemble.py",
    ROOT / "work" / "src" / "merfish60" / "hard_bucket.py",
]


def _load_oof(name: str) -> pd.DataFrame:
    path = ROOT / "work" / "outputs" / "oof" / name
    if not path.is_file():
        pytest.skip("generated OOF not retained; rerun work/scripts/04_sprint3_experiments.py to recreate")
    return pd.read_csv(
        path,
        dtype={"Cell_ID": str, "true_label": str, "predicted_label": str, "fold": int},
    )


def _load_proba(name: str) -> pd.DataFrame:
    path = ROOT / "work" / "outputs" / "probabilities" / name
    if not path.is_file():
        pytest.skip("generated probabilities not retained: {}".format(path))
    return pd.read_csv(path, dtype={"Cell_ID": str}, compression="gzip")


def _synthetic_ensemble_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    class_names = ["a", "b", "c", "d"]
    n = 90
    folds = np.array([0] * 30 + [1] * 30 + [2] * 30)
    y = rng.choice(class_names, size=n)
    p2 = rng.dirichlet(np.ones(len(class_names)), size=n)
    p3 = rng.dirichlet(np.ones(len(class_names)), size=n)
    p4 = rng.dirichlet(np.ones(len(class_names)), size=n)
    return p2, p3, p4, y, folds, class_names


def test_convex_weight_grid_is_coarse_and_convex():
    grid = convex_weight_grid(0.1)
    assert len(grid) == 66
    for w2, w3, w4 in grid:
        assert w2 >= 0 and w3 >= 0 and w4 >= 0
        assert abs((w2 + w3 + w4) - 1.0) < 1e-12


def test_ensemble_weight_grid_does_not_read_held_out_labels_directly():
    """Necessary but not sufficient for nested independence.

    This only proves the weight scorer does not index fold-f labels.
    It does not prove the saved OOF probabilities for the other folds
    were produced without fold-f training labels.
    """
    p2, p3, p4, y, folds, class_names = _synthetic_ensemble_problem()
    held = folds == 0
    eligible = ~held
    w_true, acc_true = select_ensemble_weights(p2, p3, p4, y, eligible, class_names)
    y_poison = np.array(y, dtype=object, copy=True)
    y_poison[held] = "POISON_LABEL_MUST_NOT_BE_USED"
    w_poison, acc_poison = select_ensemble_weights(
        p2, p3, p4, y_poison, eligible, class_names
    )
    assert w_true == w_poison
    assert acc_true == acc_poison

    _, details_true = cross_fitted_ensemble(p2, p3, p4, y, folds, class_names)
    _, details_poison = cross_fitted_ensemble(p2, p3, p4, y_poison, folds, class_names)
    assert details_true["0"]["weights"] == details_poison["0"]["weights"]
    assert (
        details_true["0"]["selection_accuracy_on_other_folds"]
        == details_poison["0"]["selection_accuracy_on_other_folds"]
    )


def test_yw005_is_not_a_nested_refit_of_base_models():
    sprint3 = (ROOT / "work" / "scripts" / "04_sprint3_experiments.py").read_text()
    sprint2 = (ROOT / "work" / "scripts" / "02_sprint2_experiments.py").read_text()
    assert 'load_proba_matrix("YW-002"' in sprint3
    assert 'load_proba_matrix("YW-003"' in sprint3
    assert 'load_proba_matrix("YW-004"' in sprint3
    assert "run_yw002" not in sprint3
    assert "run_yw003" not in sprint3
    assert "run_yw004" not in sprint3
    assert "y.to_numpy()[train_mask]" in sprint2
    assert "train_val_masks" in sprint2


def test_yw005_metrics_mark_exploratory_and_ineligible():
    import json

    path = ROOT / "work" / "experiments" / "evidence" / "metrics" / "YW-005_metrics.json"
    metrics = json.loads(path.read_text())
    assert metrics["formal_selection_eligible"] is False
    assert metrics["nested_independence"] is False
    assert metrics["role"] == "exploratory_ensemble_diagnostic"
    assert metrics["provenance"]["base_training_includes_held_out_fold"] is True
    assert metrics["provenance"]["excluded_from_formal_model_selection"] is True


def test_yw006_selected_log1p_lr_matching_yw004_specialist():
    import json

    metrics = json.loads((ROOT / "work" / "experiments" / "evidence" / "metrics" / "YW-006_metrics.json").read_text())
    selected = metrics["selected_model_by_outer_fold"]
    assert selected == {"0": "log1p_lr", "1": "log1p_lr", "2": "log1p_lr"}
    assert metrics["net_additional_correct_bucket_cells"] == 0
    assert abs(metrics["delta_vs_YW-004_hard_bucket"]) < 1e-12
    assert metrics.get("role") == "valid_negative_ablation"


def test_yw007_replacement_copied_yw006_probabilities_not_silent_noop():
    data = load_dataset(ROOT)
    labels = allowed_labels(ROOT)
    sigs = signatures_from_meta(data.meta_train)
    key = missing_bucket_key()
    cell_ids = [str(v) for v in data.y_train.index]
    miss = sigs.loc[cell_ids].to_numpy() == key
    p4 = (
        _load_proba("YW-004_oof_probabilities.csv.gz")
        .set_index("Cell_ID")
        .loc[cell_ids, labels]
        .to_numpy(dtype=float)
    )
    p7 = (
        _load_proba("YW-007_oof_probabilities.csv.gz")
        .set_index("Cell_ID")
        .loc[cell_ids, labels]
        .to_numpy(dtype=float)
    )
    p6 = (
        _load_proba("YW-006_hard_bucket_probabilities.csv.gz")
        .set_index("Cell_ID")
        .loc[np.array(cell_ids)[miss], labels]
        .to_numpy(dtype=float)
    )
    assert np.array_equal(p7[~miss], p4[~miss])
    assert np.allclose(p7[miss], p6, atol=0.0, rtol=0.0)
    import json

    m007 = json.loads((ROOT / "work" / "experiments" / "evidence" / "metrics" / "YW-007_metrics.json").read_text())
    assert m007["paired_vs_YW-004"]["n_changed"] == 0
    assert "identical" in m007["construction"]["zero_cell_delta_reason"]


def test_sprint3_comparison_selects_yw004_and_excludes_yw005():
    import json

    payload = json.loads((ROOT / "work" / "experiments" / "evidence" / "metrics" / "sprint3_comparison.json").read_text())
    decision = payload["formal_model_v1_decision"]
    assert decision["selected_run_id"] == "YW-004"
    assert decision["selected_oof_accuracy"] == 0.7598
    assert decision["model_v1_written"] is False
    assert payload["yw005_formal_selection_eligible"] is False
    rows = {r["run_id"]: r for r in payload["table"]}
    assert rows["YW-005"]["formal_selection_eligible"] is False
    assert rows["YW-004"]["formal_selection_eligible"] is True
    assert rows["YW-007"]["formal_selection_eligible"] is True


def test_inner_hard_bucket_selection_never_sees_outer_validation_labels(monkeypatch):
    rng = np.random.default_rng(1)
    n_train, n_val, n_feat, n_class = 40, 12, 8, 4
    class_names = ["c{}".format(i) for i in range(n_class)]
    y_train = np.array(class_names * (n_train // n_class), dtype=object)
    y_val = np.array(["VAL_ONLY_{}".format(i) for i in range(n_val)], dtype=object)
    X_train = rng.poisson(1.0, size=(n_train, n_feat)).astype(np.float64)
    seen = []

    def fake_inner(X_counts, y, spec, class_names, random_state=0):
        seen.extend([str(v) for v in np.asarray(y).tolist()])
        ranking = {
            "log1p_lr": 0.40,
            "cp10k_log1p_lr": 0.55,
            "extratrees": 0.31,
            "hist_gradient_boosting": 0.22,
        }
        return ranking[spec["name"]]

    monkeypatch.setattr(
        "merfish60.hard_bucket.inner_cv_accuracy", fake_inner
    )
    name, scores = select_hard_bucket_model(X_train, y_train, class_names)
    assert name == "cp10k_log1p_lr"
    assert set(scores) == {spec["name"] for spec in CANDIDATE_SPECS}
    assert all(not s.startswith("VAL_ONLY_") for s in seen)
    assert set(seen).isdisjoint(set(y_val.tolist()))
    assert set(seen) <= set(y_train.tolist())


def test_hard_bucket_validation_cell_ids_never_in_training_set():
    data = load_dataset(ROOT)
    folds = load_folds(ROOT / "work" / "experiments" / "folds.csv")
    y = data.y_train.astype(str)
    cell_ids = np.array([str(v) for v in y.index.tolist()], dtype=object)
    fold_ids = folds.set_index("Cell_ID").loc[list(cell_ids), "fold"]
    sigs = signatures_from_meta(data.meta_train).to_numpy()
    bucket = sigs == missing_bucket_key()
    required = set(cell_ids[bucket].tolist())
    covered = set()
    for fold_id in FOLD_VALUES:
        train_mask, val_mask = train_val_masks(fold_ids, fold_id)
        tr_ids = set(cell_ids[train_mask & bucket].tolist())
        va_ids = set(cell_ids[val_mask & bucket].tolist())
        assert tr_ids.isdisjoint(va_ids)
        assert tr_ids and va_ids
        covered |= va_ids
    assert covered == required


def test_yw006_covers_every_hard_bucket_id_exactly_once():
    data = load_dataset(ROOT)
    folds = load_folds(ROOT / "work" / "experiments" / "folds.csv")
    sigs = signatures_from_meta(data.meta_train)
    required = set(sigs.index.astype(str)[sigs.to_numpy() == missing_bucket_key()])
    oof = _load_oof("YW-006_hard_bucket_oof.csv")
    assert len(oof) == len(required) == 2958
    assert oof["Cell_ID"].nunique() == len(oof)
    assert set(oof["Cell_ID"]) == required
    merged = oof.merge(folds, on="Cell_ID", suffixes=("_oof", "_saved"))
    assert (merged["fold_oof"] == merged["fold_saved"]).all()


def test_yw007_covers_all_training_ids_exactly_once():
    data = load_dataset(ROOT)
    folds = load_folds(ROOT / "work" / "experiments" / "folds.csv")
    oof = _load_oof("YW-007_oof.csv")
    train_ids = set(data.counts_train.index.astype(str))
    assert len(oof) == 5000
    assert oof["Cell_ID"].nunique() == 5000
    assert set(oof["Cell_ID"]) == train_ids
    assert set(oof["Cell_ID"]).isdisjoint(set(data.counts_test.index.astype(str)))
    merged = oof.merge(folds, on="Cell_ID", suffixes=("_oof", "_saved"))
    assert (merged["fold_oof"] == merged["fold_saved"]).all()


@pytest.mark.parametrize(
    "proba_name,oof_name,n_rows",
    [
        ("YW-005_oof_probabilities.csv.gz", "YW-005_oof.csv", 5000),
        ("YW-006_hard_bucket_probabilities.csv.gz", "YW-006_hard_bucket_oof.csv", 2958),
        ("YW-007_oof_probabilities.csv.gz", "YW-007_oof.csv", 5000),
    ],
)
def test_sprint3_probability_files_stable_order_and_row_sums(proba_name, oof_name, n_rows):
    labels = allowed_labels(ROOT)
    proba = _load_proba(proba_name)
    oof = _load_oof(oof_name)
    assert len(proba) == n_rows
    assert list(proba.columns) == ["Cell_ID"] + labels
    assert list(proba["Cell_ID"]) == list(oof["Cell_ID"])
    mass = proba[labels].to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(mass, 1.0, atol=1e-6)


@pytest.mark.parametrize("run_id", ["YW-005", "YW-007"])
def test_sprint3_full_oof_folds_match_frozen_file(run_id):
    folds = load_folds(ROOT / "work" / "experiments" / "folds.csv")
    oof = _load_oof("{}_oof.csv".format(run_id))
    merged = oof.merge(folds, on="Cell_ID", suffixes=("_oof", "_saved"))
    assert len(merged) == 5000
    assert (merged["fold_oof"] == merged["fold_saved"]).all()


def test_yw007_replaces_only_the_hard_bucket():
    data = load_dataset(ROOT)
    sigs = signatures_from_meta(data.meta_train)
    key = missing_bucket_key()
    yw004 = _load_oof("YW-004_oof.csv").set_index("Cell_ID")
    yw006 = _load_oof("YW-006_hard_bucket_oof.csv").set_index("Cell_ID")
    yw007 = _load_oof("YW-007_oof.csv").set_index("Cell_ID")
    for cid, pred in yw007["predicted_label"].items():
        sig = sigs.loc[cid]
        if sig == key:
            assert pred == yw006.loc[cid, "predicted_label"]
        else:
            assert pred == yw004.loc[cid, "predicted_label"]


def test_sprint3_modeling_sources_do_not_read_test_labels():
    forbidden = (
        "meta_test",
        "counts_test",
        "y_test",
        "prediction/prediction.csv",
        "prediction.csv",
    )
    for path in SPRINT3_MODELING_SOURCES:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, "{} mentions {}".format(path.name, token)


def test_official_data_manifest_still_verifies():
    messages = verify_official_manifest(ROOT)
    assert any("verified" in m for m in messages)


def test_frozen_sprint2_registry_rows_cannot_be_overwritten(tmp_path):
    assert FROZEN_RUN_IDS == ("YW-000", "YW-001", "YW-002", "YW-003", "YW-004")
    exp = tmp_path / "work" / "experiments"
    exp.mkdir(parents=True)
    src = ROOT / "work" / "experiments" / "registry.csv"
    (exp / "registry.csv").write_text(src.read_text())
    with pytest.raises(RegistryError, match="frozen"):
        append_registry_row(
            {"run_id": "YW-004", "model": "must_not_overwrite"},
            root=tmp_path,
            overwrite=True,
        )
