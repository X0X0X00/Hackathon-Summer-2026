"""Integrity tests for V3-E00T team expert audit artifacts.

Does not train models. Does not read competition test labels for scoring.
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work" / "src"))
sys.path.insert(0, str(ROOT / "work" / "experiments" / "v3"))

from merfish60.io import TARGET_COL, load_dataset  # noqa: E402
from merfish60.official_contract import allowed_labels  # noqa: E402

from v3_e00t_team_expert_audit import (  # noqa: E402
    class_columns_from_header,
    pair_metrics,
    inspect_probabilities,
)

OUT = ROOT / "work" / "experiments" / "evidence" / "v3"
PRED = ROOT / "prediction" / "prediction.csv"


def test_probability_column_alignment_helper():
    classes = ["a", "b", "c"]
    assert class_columns_from_header(["Cell_ID", "a", "b", "c"], classes) == ["a", "b", "c"]
    assert class_columns_from_header(
        ["Cell_ID", "p__a", "p__b", "p__c"], classes
    ) == ["p__a", "p__b", "p__c"]


def test_inspect_probabilities_rejects_wrong_class_count():
    with pytest.raises(ValueError):
        inspect_probabilities(np.ones((3, 2)), ["a", "b", "c"])


def test_pair_oracle_is_union_of_correctness():
    y = np.array(["x", "x", "y", "z"])
    a = np.array(["x", "y", "y", "x"])
    b = np.array(["y", "x", "y", "z"])
    out = pair_metrics("A", a, "B", b, y)
    assert out["both_correct"] == 1
    assert out["a_only_correct"] == 1
    assert out["b_only_correct"] == 2
    assert out["both_wrong"] == 0
    assert out["pair_oracle_correct"] == 4
    assert out["pair_oracle_accuracy"] == 1.0


@pytest.mark.skipif(
    not (OUT / "v3_e00t_team_oof_registry.parquet").is_file(),
    reason="regenerable E00T registry not retained",
)
def test_audit_artifacts_reload_and_cell_identity():
    metrics = json.loads((OUT / "v3_e00t_metrics.json").read_text())
    manifest = json.loads((OUT / "v3_e00t_expert_manifest.json").read_text())
    registry = pd.read_parquet(OUT / "v3_e00t_team_oof_registry.parquet")
    pairwise = pd.read_csv(OUT / "v3_e00t_pairwise_oracle.csv")
    multi = pd.read_csv(OUT / "v3_e00t_multi_expert_oracle.csv")
    recov = pd.read_csv(OUT / "v3_e00t_anchor_recoverability.csv")

    assert metrics["oracle_is_not_deployable_accuracy"] is True
    assert metrics["integrity"]["n_train_cells"] == 5000
    assert metrics["integrity"]["test_labels_used"] is False
    assert metrics["integrity"]["prediction_csv_modified"] is False
    assert len(manifest["experts"]) >= 4
    assert set(registry["Cell_ID"].astype(str)) == set(load_dataset(ROOT).meta_train.index.astype(str))
    assert registry["Cell_ID"].nunique() == 5000
    assert not registry["Cell_ID"].duplicated().any()
    assert registry["Cell_ID"].str.fullmatch(r"\d{19}").all()
    assert not registry["Cell_ID"].str.endswith(".0").any()
    assert registry["true_label"].isna().sum() == 0
    data = load_dataset(ROOT)
    aligned = registry.set_index("Cell_ID")["true_label"]
    official = data.meta_train[TARGET_COL].astype(str)
    assert list(aligned.reindex(official.index.astype(str))) == list(official)
    assert len(pairwise) >= 1
    assert len(multi) >= 1
    assert len(recov) == 1
    assert metrics["multi_expert_oracle"]["best_3_way"] is None
    class_names = allowed_labels(ROOT)
    assert len(class_names) == 60
    assert metrics["class_names"] == class_names


@pytest.mark.skipif(not (OUT / "v3_e00t_metrics.json").is_file(), reason="audit artifacts not generated")
def test_official_prediction_and_frozen_models_untouched():
    dirty = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert dirty == ""
    assert PRED.is_file()
