"""Integrity tests for V3-E03A weak-but-diverse expert rescue audit.

Does not train a gate. Does not retrain S0. Does not read competition test
labels for scoring. Does not modify prediction/prediction.csv.
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "v3"))

from merfish60.io import TARGET_COL, load_dataset  # noqa: E402
from merfish60.models import assert_probability_rows  # noqa: E402
from merfish60.official_contract import allowed_labels  # noqa: E402

from v3_e03a_rescue_audit import (  # noqa: E402
    ALL_THREE_WRONG,
    BOTH_WRONG,
    FOLD_RESCUES,
    LZH_CORRECT,
    LZH_WYH_ORACLE,
    POSITIVE_RESCUES,
    S0_CORRECT,
    THREE_EXPERT_ORACLE,
    WYH_CORRECT,
    family_of,
    family_of_e02d,
    rescue_evidence_score,
    zscore,
)

OUT = ROOT / "outputs" / "v3"
PRED = ROOT / "prediction" / "prediction.csv"
E00T_REGISTRY = OUT / "v3_e00t_team_oof_registry.parquet"
E02D_S0 = OUT / "v3_e02d_s0_validation.csv"
E02D_COMP = OUT / "v3_e02d_complementarity.json"


def test_family_map_splits_microglia_but_preserves_e02d_oligo_vascular():
    assert family_of("oligodendrocyte_1") == "oligodendrocyte_opc"
    assert family_of("endothelial") == "vascular"
    assert family_of("microglia") == "microglia"
    assert family_of("ependymal") == "remaining_glial_non_neuronal"
    assert family_of("DH_ex_Grpr") == "neuronal_or_other"
    assert family_of_e02d("microglia") == "other_glial_non_neuronal"
    assert family_of_e02d("oligodendrocyte_progenitor_2") == "oligodendrocyte_opc"


def test_unique_rescue_and_override_set_algebra():
    lzh_ok = np.array([True, False, False, True, False])
    wyh_ok = np.array([True, False, True, False, False])
    s0_ok = np.array([False, True, False, False, False])
    positive = (~lzh_ok) & (~wyh_ok) & s0_ok
    n1 = (~lzh_ok) & (~wyh_ok) & (~s0_ok)
    n2 = lzh_ok & wyh_ok & (~s0_ok)
    n3 = (lzh_ok | wyh_ok) & (~s0_ok)
    assert positive.tolist() == [False, True, False, False, False]
    assert n1.tolist() == [False, False, False, False, True]
    assert n2.tolist() == [True, False, False, False, False]
    assert n3.tolist() == [True, False, True, True, False]
    assert int((lzh_ok | wyh_ok | s0_ok).sum()) == 4
    assert int(positive.sum()) + int((lzh_ok | wyh_ok).sum()) == int((lzh_ok | wyh_ok | s0_ok).sum())


def test_predeclared_rescue_evidence_score_has_fixed_coefficients():
    s0 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    lzh = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float64)
    wyh = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    score = rescue_evidence_score(s0, lzh, wyh)
    expected = zscore(s0) - 0.5 * zscore(lzh) - 0.5 * zscore(wyh)
    np.testing.assert_allclose(score, expected)
    assert float(np.mean(zscore(s0))) == pytest.approx(0.0, abs=1e-12)


def test_feature_classification_requires_fold_stable_p_vs_n1():
    from v3_e03a_rescue_audit import classify_feature

    n3_only = {
        "feature": "lzh_entropy",
        "available": True,
        "auc_p_vs_n1_folds_0_2": 0.63,
        "auc_p_vs_n3_folds_0_2": 0.84,
        "auc_p_vs_n1_folds_3_4": 0.60,
        "auc_p_vs_n3_folds_3_4": 0.80,
        "folds_0_2_direction": "higher_in_P",
        "folds_3_4_direction": "higher_in_P",
    }
    assert classify_feature(n3_only) == "MODERATE"
    both = dict(n3_only)
    both["auc_p_vs_n1_folds_0_2"] = 0.72
    both["auc_p_vs_n1_folds_3_4"] = 0.70
    assert classify_feature(both) == "STRONG"
    definitional = dict(n3_only)
    definitional["feature"] = "lzh_s0_agree"
    definitional["auc_p_vs_n1_folds_0_2"] = 0.15
    assert classify_feature(definitional) == "WEAK"


def test_fold_rescue_identity():
    assert FOLD_RESCUES == {0: 25, 1: 28, 2: 16, 3: 17, 4: 10}
    assert sum(FOLD_RESCUES[i] for i in range(3)) == 69
    assert sum(FOLD_RESCUES[i] for i in (3, 4)) == 27
    assert sum(FOLD_RESCUES.values()) == POSITIVE_RESCUES
    assert LZH_WYH_ORACLE + POSITIVE_RESCUES == THREE_EXPERT_ORACLE
    assert BOTH_WRONG - POSITIVE_RESCUES == ALL_THREE_WRONG


@pytest.mark.skipif(not (OUT / "v3_e03a_rescue_metrics.json").is_file(), reason="V3-E03A artifacts not generated")
def test_registry_integrity_and_cell_identity():
    metrics = json.loads((OUT / "v3_e03a_rescue_metrics.json").read_text())
    registry = pd.read_parquet(OUT / "v3_e03a_three_expert_registry.parquet")
    data = load_dataset(ROOT)
    class_names = allowed_labels(ROOT)

    assert metrics["oracle_is_not_deployable_accuracy"] is True
    assert len(registry) == 5000
    assert registry["Cell_ID"].nunique() == 5000
    assert not registry["Cell_ID"].duplicated().any()
    assert registry["Cell_ID"].str.fullmatch(r"\d{19}").all()
    assert not registry["Cell_ID"].str.endswith(".0").any()
    assert set(registry["Cell_ID"].astype(str)) == set(data.meta_train.index.astype(str))
    aligned = registry.set_index("Cell_ID")["true_label"]
    official = data.meta_train[TARGET_COL].astype(str)
    assert list(aligned.reindex(official.index.astype(str))) == list(official)
    assert len(class_names) == 60

    integ = metrics["integrity"]
    assert integ["lzh_correct"] == LZH_CORRECT
    assert integ["wyh_correct"] == WYH_CORRECT
    assert integ["s0_correct"] == S0_CORRECT
    assert integ["lzh_wyh_oracle"] == LZH_WYH_ORACLE
    assert integ["both_wrong"] == BOTH_WRONG
    assert integ["positive_rescues"] == POSITIVE_RESCUES
    assert integ["three_expert_oracle"] == THREE_EXPERT_ORACLE
    assert integ["all_three_wrong"] == ALL_THREE_WRONG
    assert {int(k): int(v) for k, v in integ["fold_rescues"].items()} == FOLD_RESCUES

    assert int(registry["lzh_correct"].sum()) == LZH_CORRECT
    assert int(registry["wyh_correct"].sum()) == WYH_CORRECT
    assert int(registry["s0_correct"].sum()) == S0_CORRECT
    assert int(registry["three_expert_oracle_correct"].sum()) == THREE_EXPERT_ORACLE
    assert int(registry["positive_s0_rescue"].sum()) == POSITIVE_RESCUES
    assert int(registry["group_n1"].sum()) == ALL_THREE_WRONG
    assert int((registry["lzh_wyh_both_wrong"] & registry["s0_correct"]).sum()) == POSITIVE_RESCUES


@pytest.mark.skipif(not (OUT / "v3_e03a_rescue_metrics.json").is_file(), reason="V3-E03A artifacts not generated")
def test_group_files_and_probability_normalization():
    metrics = json.loads((OUT / "v3_e03a_rescue_metrics.json").read_text())
    rescues = pd.read_csv(OUT / "v3_e03a_positive_rescues.csv", dtype={"Cell_ID": str})
    danger = pd.read_csv(OUT / "v3_e03a_dangerous_overrides.csv", dtype={"Cell_ID": str})
    family = pd.read_csv(OUT / "v3_e03a_family_rescue_map.csv")
    feat = pd.read_csv(OUT / "v3_e03a_feature_diagnostics.csv")

    assert len(rescues) == POSITIVE_RESCUES
    assert rescues["Cell_ID"].nunique() == POSITIVE_RESCUES
    assert metrics["dangerous_overrides"]["n1"] == ALL_THREE_WRONG
    assert metrics["dangerous_overrides"]["n1"] + metrics["dangerous_overrides"]["n3"] == (
        5000 - S0_CORRECT
    )
    assert set(danger["Cell_ID"]).isdisjoint(set(rescues["Cell_ID"]))
    assert len(family) >= 6
    assert "oligodendrocyte_opc" in set(family["family"])
    assert len(feat) >= 10
    assert metrics["leakage_audit"]["learned_gate_trained"] is False
    assert metrics["leakage_audit"]["s0_retrained"] is False
    assert metrics["leakage_audit"]["threshold_optimized"] is False
    assert metrics["final_classification"]["label"] in {
        "RESCUE SIGNAL STRONG",
        "RESCUE SIGNAL FAMILY-SPECIFIC",
        "RESCUE SIGNAL WEAK",
    }

    proba_path = OUT / "v3_e03a_tables" / "s0_inference_probabilities.csv.gz"
    if metrics["s0_probability"]["available"]:
        assert metrics["s0_probability"]["retrained"] is False
        assert metrics["s0_probability"]["n_label_match"] == 5000
        assert len(metrics["s0_probability"]["checkpoint_sha256"]) == 64
        proba = pd.read_csv(proba_path, dtype={"Cell_ID": str})
        class_names = allowed_labels(ROOT)
        assert list(proba.columns[1:]) == class_names
        assert_probability_rows(proba.loc[:, class_names].to_numpy(), atol=1e-4)
        registry = pd.read_parquet(OUT / "v3_e03a_three_expert_registry.parquet")
        aligned = proba.set_index("Cell_ID").reindex(registry["Cell_ID"])
        pred = aligned.loc[:, class_names].to_numpy().argmax(axis=1)
        pred_labels = np.asarray(class_names)[pred]
        assert list(pred_labels) == list(registry["s0_pred"].astype(str))


@pytest.mark.skipif(not (OUT / "v3_e03a_rescue_metrics.json").is_file(), reason="V3-E03A artifacts not generated")
def test_frozen_artifacts_and_prediction_untouched():
    dirty_pred = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert dirty_pred == ""
    frozen = subprocess.check_output(
        [
            "git",
            "diff",
            "--",
            "outputs/v3/v3_e00t_team_oof_registry.parquet",
            "outputs/v3/v3_e00t_metrics.json",
            "outputs/v3/v3_e02d_complementarity.json",
            "outputs/v3/v3_e02d_s0_validation.csv",
            "outputs/v3/v3_e02d_s1_validation.csv",
            "outputs/v3/v3_e02d_s2_validation.csv",
            "outputs/v3/v3_e02d_student_comparison.csv",
            "outputs/v3/v3_e02d_unique_recoveries.csv",
            "docs/versions/model_v1.md",
            "docs/versions/model_v2.md",
            "outputs/submissions/model_v1.csv",
            "outputs/submissions/model_v2_candidate.csv",
        ],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert frozen == ""
    assert PRED.is_file()
    assert E00T_REGISTRY.is_file()
    assert E02D_S0.is_file()
    assert E02D_COMP.is_file()
    assert not (ROOT / "docs" / "versions" / "model_v3.md").is_file()
