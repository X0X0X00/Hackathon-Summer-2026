"""Sprint 2 experiment output and leakage-guard tests.

These tests read saved OOF/probability files after the experiments have been
run. They do not refit models or touch official CSVs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merfish60.cv import load_folds
from merfish60.io import load_dataset
from merfish60.official_contract import (
    YW002_FORBIDDEN_FIELDS,
    YW002_METADATA_FIELDS,
    allowed_labels,
)
from merfish60.signatures import missing_bucket_key

ROOT = Path(__file__).resolve().parents[2]


def _load_oof(run_id: str) -> pd.DataFrame:
    path = ROOT / "work" / "outputs" / "oof" / "{}_oof.csv".format(run_id)
    if not path.is_file():
        pytest.skip("generated OOF not retained; rerun work/scripts/02_sprint2_experiments.py to recreate")
    return pd.read_csv(path, dtype={"Cell_ID": str, "true_label": str, "predicted_label": str, "fold": int})


def _load_proba(run_id: str) -> pd.DataFrame:
    path = ROOT / "work" / "outputs" / "probabilities" / "{}_oof_probabilities.csv.gz".format(run_id)
    if not path.is_file():
        pytest.skip("generated probabilities not retained: {}".format(path))
    return pd.read_csv(path, dtype={"Cell_ID": str}, compression="gzip")


def _load_metrics(run_id: str) -> dict:
    import json

    path = ROOT / "work" / "experiments" / "evidence" / "metrics" / "{}_metrics.json".format(run_id)
    return json.loads(path.read_text())


@pytest.mark.parametrize("run_id", ["YW-002", "YW-003", "YW-004"])
def test_oof_unique_ids_and_fold_match(run_id):
    data = load_dataset(ROOT)
    folds = load_folds(ROOT / "work" / "experiments" / "folds.csv")
    oof = _load_oof(run_id)
    assert len(oof) == 5000
    assert oof["Cell_ID"].nunique() == 5000
    assert set(oof["Cell_ID"]) == set(data.counts_train.index.astype(str))
    assert set(oof["Cell_ID"]).isdisjoint(set(data.counts_test.index.astype(str)))
    merged = oof.merge(folds, on="Cell_ID", suffixes=("_oof", "_saved"))
    assert (merged["fold_oof"] == merged["fold_saved"]).all()


@pytest.mark.parametrize("run_id", ["YW-002", "YW-003", "YW-004"])
def test_stable_class_order_and_probability_sums(run_id):
    labels = allowed_labels(ROOT)
    proba = _load_proba(run_id)
    oof = _load_oof(run_id)
    assert list(proba.columns) == ["Cell_ID"] + labels
    assert list(proba["Cell_ID"]) == list(oof["Cell_ID"])
    mass = proba[labels].to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(mass, 1.0, atol=1e-6)
    metrics = _load_metrics(run_id)
    assert metrics["class_order"] == labels


def test_yw002_uses_only_allowed_metadata_fields():
    metrics = _load_metrics("YW-002")
    assert metrics["preprocessing"]["metadata_fields"] == YW002_METADATA_FIELDS
    assert metrics["preprocessing"]["forbidden_fields"] == YW002_FORBIDDEN_FIELDS
    assert metrics["preprocessing"]["target_encoding"] is False
    for fold_id, dims in metrics["encoded_feature_dimensions"].items():
        assert dims["n_gene_features"] == 200
        assert dims["n_total_features"] == dims["n_gene_features"] + dims["n_meta_features"]
        assert dims["n_meta_features"] > 0


def test_yw003_global_model_is_gene_only():
    metrics = _load_metrics("YW-003")
    assert "gene-only" in metrics["global_model"]
    assert "metadata used only for candidate masking" in metrics["feature_set"]
    assert metrics["candidate_map"]["fit"] == "training fold rows and labels only"
    assert "unmasked_global_oof_accuracy" in metrics
    oof001_path = ROOT / "work" / "outputs" / "oof" / "YW-001_oof.csv"
    if not oof001_path.is_file():
        pytest.skip("generated YW-001 OOF not retained")
    oof001 = pd.read_csv(oof001_path, dtype={"Cell_ID": str})
    oof003 = _load_oof("YW-003")
    assert list(oof001["Cell_ID"]) == list(oof003["Cell_ID"])


def test_yw004_fallback_behavior_is_recorded():
    metrics = _load_metrics("YW-004")
    assert "n_specialists_trained_per_fold" in metrics
    assert "fallback_counts" in metrics
    assert "fallback_n" in metrics
    assert metrics["missing_bucket_key"] == missing_bucket_key()
    assert metrics["missing_bucket_n"] > 0
    for fold_id, n_spec in metrics["n_specialists_trained_per_fold"].items():
        assert int(n_spec) >= 0
    oof = _load_oof("YW-004")
    assert oof["predicted_label"].notna().all()
    assert set(oof["predicted_label"]).issubset(set(allowed_labels(ROOT)))
