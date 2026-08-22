"""V2-B-REFONLY leakage, exclusion, alignment, and artifact tests."""

from pathlib import Path
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from merfish60.io import N_CLASSES, TARGET_COL, load_dataset
from merfish60.neighbor_labels import (
    apply_segment_mask,
    label_hist,
    visible_ext_label_codes,
)
from merfish60.official_contract import allowed_labels, verify_official_manifest
from merfish60.reference import (
    EXPECTED_MD5,
    EXPECTED_N_OBS,
    REFERENCE_H5AD_REL,
    audit_reference,
    reference_h5ad_path,
    verify_reference_md5,
)
from merfish60.team_cv import load_team_folds, team_folds_path

ROOT = Path(__file__).resolve().parents[2]
V2B_SOURCES = [
    ROOT / "work" / "src" / "merfish60" / "reference.py",
    ROOT / "work" / "src" / "merfish60" / "ext_universe.py",
    ROOT / "work" / "src" / "merfish60" / "neighbor_labels.py",
    ROOT / "work" / "scripts" / "09_v2b_refonly.py",
]
MODEL_V1_PATHS = [
    "work/src/merfish60/model_v1.py",
    "work/scripts/06_model_v1.py",
    "work/experiments/evidence/submissions/model_v1.csv",
]


@pytest.fixture(scope="module")
def data():
    return load_dataset(ROOT)


@pytest.fixture(scope="module")
def class_names():
    return allowed_labels(ROOT)


@pytest.fixture(scope="module")
def audit(data, class_names):
    path = reference_h5ad_path(ROOT)
    if not path.is_file():
        pytest.skip("approved reference h5ad is not present")
    return audit_reference(data, class_names, root=ROOT)


def test_reference_md5_matches_approved_constant():
    path = reference_h5ad_path(ROOT)
    if not path.is_file():
        pytest.skip("approved reference h5ad is not present")
    assert path.as_posix().endswith(REFERENCE_H5AD_REL) or path == ROOT / REFERENCE_H5AD_REL
    assert verify_reference_md5(path) == EXPECTED_MD5


def test_reference_audit_exclusion_and_gene_order(audit, data, class_names):
    assert audit["md5"] == EXPECTED_MD5
    assert audit["raw_n_obs"] == EXPECTED_N_OBS
    assert audit["raw_n_vars"] == 500
    assert audit["n_competition_genes"] == 200
    assert audit["gene_order_ok"] is True
    assert audit["competition_genes_aligned"] == list(data.counts_train.columns)
    assert audit["n_train_id_overlaps_removed"] == 5000
    assert audit["n_test_id_overlaps_removed"] == 5000
    assert audit["n_usable_reference_rows"] == (
        audit["raw_n_obs"]
        - audit["n_train_id_overlaps_removed"]
        - audit["n_test_id_overlaps_removed"]
        - audit["n_exact_vector_duplicates_removed"]
    )
    assert not (audit["_is_train"] & audit["_is_ref"]).any()
    assert not (audit["_is_test"] & audit["_is_ref"]).any()
    train_ids = set(str(v) for v in data.counts_train.index)
    test_ids = set(str(v) for v in data.counts_test.index)
    ref_ids = set(audit["_ext_ids"][audit["_is_ref"]].tolist())
    assert ref_ids.isdisjoint(train_ids)
    assert ref_ids.isdisjoint(test_ids)
    assert list(audit["_genes"]) == list(data.counts_train.columns)
    assert audit["test_target_used"] is False
    assert set(audit["unmapped_labels"]) == set()
    assert audit["reference_label_coverage"] == len(class_names)


def test_taxonomy_normalization_maps_hyphens_to_official(audit, class_names):
    raw = set(audit["raw_unique_labels"])
    from merfish60.reference import norm_label

    normalized = {norm_label(v) for v in raw}
    assert normalized <= set(class_names)
    assert audit["n_raw_unique_labels"] == 60


def test_held_out_fold_labels_invisible_with_reference_present():
    y = np.array([0, 1, 2, 3, 4, 5, -1], dtype=np.int64)
    is_train = np.array([True, True, True, True, False, False, False])
    is_ref = np.array([False, False, False, False, True, True, False])
    folds = np.array([0, 0, 1, 1, -1, -1, -1], dtype=np.int64)
    known = visible_ext_label_codes(y, is_train, is_ref, folds, holdout_fold=0)
    np.testing.assert_array_equal(known, np.array([-1, -1, 2, 3, 4, 5, -1]))
    known_test = visible_ext_label_codes(y, is_train, is_ref, folds, holdout_fold=None)
    np.testing.assert_array_equal(known_test, np.array([0, 1, 2, 3, 4, 5, -1]))


def test_poisoned_holdout_labels_do_not_change_ext_histograms():
    idx = np.array([[1, 4], [0, 4], [3, 5], [2, 5], [0, 1], [2, 3]], dtype=np.int32)
    dist = np.ones_like(idx, dtype=np.float32)
    y = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    is_train = np.array([True, True, True, True, False, False])
    is_ref = np.array([False, False, False, False, True, True])
    folds = np.array([0, 0, 1, 1, -1, -1], dtype=np.int64)
    known = visible_ext_label_codes(y, is_train, is_ref, folds, 0)
    hist, _ = label_hist(known, idx, dist, k=2, tag="sp", n_classes=6)
    y_poison = y.copy()
    y_poison[folds == 0] = 5
    known_p = visible_ext_label_codes(y_poison, is_train, is_ref, folds, 0)
    hist_p, _ = label_hist(known_p, idx, dist, k=2, tag="sp", n_classes=6)
    np.testing.assert_array_equal(hist, hist_p)
    y_ref_poison = y.copy()
    y_ref_poison[is_ref] = 0
    known_r = visible_ext_label_codes(y_ref_poison, is_train, is_ref, folds, 0)
    hist_r, _ = label_hist(known_r, idx, dist, k=2, tag="sp", n_classes=6)
    assert not np.allclose(hist, hist_r)


def test_segment_mask_uses_supplied_table_only():
    probs = np.array([[0.1, 0.7, 0.2], [0.6, 0.2, 0.2]], dtype=np.float64)
    seg = np.array([1.0, np.nan], dtype=np.float32)
    tab = {1: [0]}
    out = apply_segment_mask(probs, seg, tab, n_classes=3)
    np.testing.assert_allclose(out[0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(out[1], probs[1])


def test_v2b_sources_never_use_test_target_or_official_submission():
    forbidden = (
        'meta_test[TARGET_COL]',
        'meta_test["MERFISH_cell_type_annotation"]',
        "meta_test['MERFISH_cell_type_annotation']",
        "y_test",
        "prediction/prediction.csv",
        "official_final_submission_path",
    )
    for path in V2B_SOURCES:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, "{} mentions {}".format(path.name, token)
        assert "data.meta_test[TARGET_COL]" not in text
        assert "data.meta_test[{}]".format(TARGET_COL) not in text


def test_official_manifest_and_model_v1_immutable():
    messages = verify_official_manifest(ROOT)
    assert any("verified" in line for line in messages)
    diff = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert diff == "", diff


def _load_oof():
    path = ROOT / "work" / "outputs" / "oof" / "V2-B-REFONLY_oof.csv"
    if not path.is_file():
        pytest.skip("V2-B OOF not written yet")
    return pd.read_csv(
        path,
        dtype={"Cell_ID": str, "true_label": str, "predicted_label": str, "fold": int},
    )


def _load_proba(name: str):
    path = ROOT / "work" / "outputs" / "probabilities" / name
    if not path.is_file():
        pytest.skip("missing {}".format(path))
    return pd.read_csv(path, dtype={"Cell_ID": str}, compression="gzip")


def test_v2b_oof_probability_order_and_row_sums(data, class_names):
    oof = _load_oof()
    proba = _load_proba("V2-B-REFONLY_oof_probabilities.csv.gz")
    train_ids = [str(v) for v in data.counts_train.index]
    assert len(oof) == 5000
    assert set(oof["Cell_ID"]) == set(train_ids)
    assert set(oof["Cell_ID"]).isdisjoint(set(data.counts_test.index.astype(str)))
    assert list(proba.columns) == ["Cell_ID"] + list(class_names)
    assert list(proba["Cell_ID"]) == list(oof["Cell_ID"])
    mass = proba[class_names].to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(mass, 1.0, atol=1e-5)
    folds = load_team_folds(team_folds_path(ROOT))
    merged = oof.merge(folds, on="Cell_ID", suffixes=("_oof", "_saved"))
    assert (merged["fold_oof"] == merged["fold_saved"]).all()
    assert proba[class_names].shape[1] == N_CLASSES


def test_v2b_audit_artifacts_match_live_audit(audit):
    path = ROOT / "work" / "experiments" / "evidence" / "metrics" / "V2-B-REFONLY_exclusion_manifest.json"
    if not path.is_file():
        pytest.skip("exclusion manifest not written yet")
    saved = json.loads(path.read_text())
    assert saved["md5"] == EXPECTED_MD5
    assert saved["train_cell_ids_removed"] == audit["n_train_id_overlaps_removed"]
    assert saved["test_cell_ids_removed"] == audit["n_test_id_overlaps_removed"]
    assert saved["exact_vector_duplicates_removed"] == audit["n_exact_vector_duplicates_removed"]
    assert saved["final_usable_reference_rows"] == audit["n_usable_reference_rows"]
