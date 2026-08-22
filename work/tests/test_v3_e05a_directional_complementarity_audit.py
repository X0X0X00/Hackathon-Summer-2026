"""Integrity tests for V3-E05A asymmetric directional complementarity audit.

Does not train a model or router. Does not search thresholds or weights.
Does not read competition test labels for scoring. Does not modify
prediction/prediction.csv.
"""
from pathlib import Path
import inspect
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
from merfish60.official_contract import sha256_file  # noqa: E402

from v3_e05a_directional_complementarity_audit import (  # noqa: E402
    ALL_FOUR_WRONG,
    FOUR_EXPERT_ORACLE,
    LZH_CORRECT,
    MIN_PRECISION_FOLDS_02,
    MIN_PRECISION_FOLDS_34,
    MIN_PRECISION_OVERALL,
    MIN_SUPPORT,
    MIN_WRONG_TO_CORRECT,
    N_TRAIN,
    OLIGO_1,
    OLIGO_2,
    OPC_2,
    S0_CORRECT,
    S0_UNIQUE_RECOVERIES,
    SNI_CORRECT,
    SNI_UNIQUE_RECOVERIES,
    THREE_EXPERT_ORACLE,
    TWO_EXPERT_ORACLE,
    WYH_CORRECT,
    actionable_checks,
    apply_fixed_patches,
    classify_experiment,
    directional_accounting,
    empty_accounting,
    system_metrics,
    trigger_functions_use_predictions_only,
    trigger_h1,
    trigger_h2,
    trigger_h3,
)

OUT = ROOT / "work" / "experiments" / "evidence" / "v3"
PRED = ROOT / "prediction" / "prediction.csv"
REPORT = ROOT / "work" / "reports" / "v3" / "v3_e05a_directional_complementarity_audit.md"


def test_h1_h2_h3_trigger_definitions_use_predictions_only():
    assert trigger_functions_use_predictions_only() is True
    for fn in (trigger_h1, trigger_h2, trigger_h3):
        names = set(inspect.signature(fn).parameters)
        assert "true_label" not in names
        assert "y_true" not in names
        assert "correct" not in names
        assert "oracle" not in names
        assert "rescue" not in names
        src = inspect.getsource(fn)
        assert "true_label" not in src
        assert "y_true" not in src


def test_h1_h2_h3_triggers_are_observable_and_mutually_exclusive():
    lzh = np.array([OLIGO_1, OLIGO_1, OPC_2, OLIGO_2, OLIGO_1, "astrocyte_1"], dtype=object)
    wyh = np.array([OLIGO_1, OPC_2, OPC_2, OLIGO_2, OLIGO_1, "astrocyte_1"], dtype=object)
    s0 = np.array(["x", "x", OLIGO_1, "x", OLIGO_1, "x"], dtype=object)
    sni = np.array([OPC_2, OPC_2, "x", OPC_2, OLIGO_1, OPC_2], dtype=object)

    h1 = trigger_h1(lzh, wyh, sni)
    h2 = trigger_h2(lzh, wyh, s0)
    h3 = trigger_h3(lzh, wyh, sni)

    assert h1.tolist() == [True, False, False, False, False, False]
    assert h2.tolist() == [False, False, True, False, False, False]
    assert h3.tolist() == [False, False, False, True, False, False]
    assert not np.any(h1 & h2)
    assert not np.any(h1 & h3)
    assert not np.any(h2 & h3)
    # LZH/WYH disagreement cannot fire H1 even if SNI proposes progenitor-2.
    assert h1[1] is False or bool(h1[1]) is False


def test_directional_accounting_net_and_precision():
    trigger = np.array([True, True, True, True, False])
    strong = np.array(["a", "a", "a", "a", "a"])
    cand = np.array(["b", "b", "a", "c", "b"])
    y = np.array(["b", "a", "a", "z", "b"])
    acc = directional_accounting(trigger, strong, cand, y)
    # cells 0-3 in trigger:
    # 0 strong wrong, cand correct -> w2c
    # 1 strong correct, cand wrong -> c2w
    # 2 both correct
    # 3 both wrong
    assert acc["support"] == 4
    assert acc["wrong_to_correct"] == 1
    assert acc["correct_to_wrong"] == 1
    assert acc["both_correct"] == 1
    assert acc["both_wrong"] == 1
    assert acc["net"] == 0
    assert acc["patch_precision"] == pytest.approx(0.5)
    assert empty_accounting()["support"] == 0
    assert empty_accounting()["patch_precision"] is None


def test_d0_d3_deterministic_accounting_and_additivity():
    lzh = np.array([OLIGO_1, OPC_2, OLIGO_2, "astrocyte_1"], dtype=object)
    wyh = np.array([OLIGO_1, OPC_2, OLIGO_2, "astrocyte_1"], dtype=object)
    s0 = np.array(["x", OLIGO_1, "x", "x"], dtype=object)
    sni = np.array([OPC_2, "x", OPC_2, "x"], dtype=object)
    y = np.array([OPC_2, OLIGO_1, OPC_2, "astrocyte_1"], dtype=object)
    folds = np.array([0, 1, 3, 4])
    class_names = [OLIGO_1, OLIGO_2, OPC_2, "astrocyte_1", "x"]

    h1 = trigger_h1(lzh, wyh, sni)
    h2 = trigger_h2(lzh, wyh, s0)
    h3 = trigger_h3(lzh, wyh, sni)
    assert h1.tolist() == [True, False, False, False]
    assert h2.tolist() == [False, True, False, False]
    assert h3.tolist() == [False, False, True, False]

    d0 = apply_fixed_patches(lzh, h1, h2, h3, s0, sni, "D0")
    d1 = apply_fixed_patches(lzh, h1, h2, h3, s0, sni, "D1")
    d2 = apply_fixed_patches(lzh, h1, h2, h3, s0, sni, "D2")
    d3 = apply_fixed_patches(lzh, h1, h2, h3, s0, sni, "D3")
    assert list(d0) == list(lzh)
    assert list(d1) == [OPC_2, OPC_2, OLIGO_2, "astrocyte_1"]
    assert list(d2) == [OPC_2, OLIGO_1, OLIGO_2, "astrocyte_1"]
    assert list(d3) == [OPC_2, OLIGO_1, OPC_2, "astrocyte_1"]

    m0 = system_metrics(y, d0, lzh, folds, class_names, "D0")
    m1 = system_metrics(y, d1, lzh, folds, class_names, "D1")
    m2 = system_metrics(y, d2, lzh, folds, class_names, "D2")
    m3 = system_metrics(y, d3, lzh, folds, class_names, "D3")
    assert m0["changed"] == 0
    assert m0["net"] == 0
    assert m0["not_unbiased_final_oof"] is True
    assert m0["not_model_v3"] is True
    assert m1["wrong_to_correct"] == 1
    assert m1["correct_to_wrong"] == 0
    assert m1["net"] == 1
    assert m2["net"] == 2
    assert m3["net"] == 3
    assert m3["correct"] == 4
    h1_acc = directional_accounting(h1, lzh, sni, y)
    h2_acc = directional_accounting(h2, lzh, s0, y)
    h3_acc = directional_accounting(h3, lzh, sni, y)
    assert m1["net"] == h1_acc["net"]
    assert m2["net"] == h1_acc["net"] + h2_acc["net"]
    assert m3["net"] == h1_acc["net"] + h2_acc["net"] + h3_acc["net"]


def test_predeclared_actionable_criteria_are_not_post_hoc():
    good = {
        "support": 20,
        "wrong_to_correct": 12,
        "correct_to_wrong": 3,
        "net": 9,
        "patch_precision": 0.80,
    }
    part = {
        "support": 10,
        "wrong_to_correct": 6,
        "correct_to_wrong": 2,
        "net": 4,
        "patch_precision": 0.75,
    }
    act = actionable_checks(good, part, part)
    assert act["actionable"] is True
    assert MIN_SUPPORT == 15
    assert MIN_WRONG_TO_CORRECT == 10
    assert MIN_PRECISION_OVERALL == 0.65
    assert MIN_PRECISION_FOLDS_02 == 0.60
    assert MIN_PRECISION_FOLDS_34 == 0.55

    weak = dict(good)
    weak["patch_precision"] = 0.50
    weak["net"] = 1
    assert actionable_checks(weak, part, part)["actionable"] is False

    no_holdout = dict(good)
    f34 = dict(part)
    f34["net"] = 0
    assert actionable_checks(good, part, f34)["actionable"] is False


def test_classification_labels_are_predeclared():
    def hyp(actionable, net, w2c, sparse=False, artifact=False, p=0.7, n02=1, n34=1, support=20):
        return {
            "actionable": actionable,
            "overall": {"net": net, "wrong_to_correct": w2c, "patch_precision": p, "support": support},
            "folds_0_2": {"net": n02},
            "folds_3_4": {"net": n34},
            "sparsity": {"sparse_unstable": sparse, "single_section_artifact": artifact},
        }

    strong = classify_experiment({"H1": hyp(True, 12, 14), "H2": hyp(False, 0, 0), "H3": hyp(False, -1, 0)})
    assert strong["label"] == "DIRECTIONAL SIGNAL STRONG"
    assert "E05B" in strong["next_action"]

    limited = classify_experiment(
        {"H1": hyp(False, 4, 6, p=0.70, n02=2, n34=2, support=12), "H2": hyp(False, 0, 0), "H3": hyp(False, 0, 0)}
    )
    assert limited["label"] == "DIRECTIONAL SIGNAL LIMITED"

    weak = classify_experiment(
        {"H1": hyp(False, 0, 0), "H2": hyp(False, -2, 1), "H3": hyp(False, 0, 0)}
    )
    assert weak["label"] == "DIRECTIONAL SIGNAL WEAK"
    assert "router" in weak["next_action"].lower()

    artifact = classify_experiment(
        {"H1": hyp(True, 12, 14, artifact=True, sparse=True), "H2": hyp(False, 0, 0), "H3": hyp(False, 0, 0)}
    )
    assert artifact["label"] != "DIRECTIONAL SIGNAL STRONG"


@pytest.mark.skipif(
    not (OUT / "v3_e05a_four_expert_registry.parquet").is_file(),
    reason="regenerable E05A registry not retained",
)
def test_four_expert_registry_integrity_and_cell_identity():
    metrics = json.loads((OUT / "v3_e05a_directional_metrics.json").read_text())
    registry = pd.read_parquet(OUT / "v3_e05a_four_expert_registry.parquet")
    data = load_dataset(ROOT)

    assert len(registry) == N_TRAIN
    assert registry["Cell_ID"].nunique() == N_TRAIN
    assert not registry["Cell_ID"].duplicated().any()
    assert registry["Cell_ID"].str.fullmatch(r"\d{19}").all()
    assert not registry["Cell_ID"].str.endswith(".0").any()
    assert set(registry["Cell_ID"].astype(str)) == set(data.meta_train.index.astype(str))
    aligned = registry.set_index("Cell_ID")["true_label"]
    official = data.meta_train[TARGET_COL].astype(str)
    assert list(aligned.reindex(official.index.astype(str))) == list(official)

    integ = metrics["integrity"]
    assert integ["lzh_correct"] == LZH_CORRECT
    assert integ["wyh_correct"] == WYH_CORRECT
    assert integ["s0_correct"] == S0_CORRECT
    assert integ["sni_correct"] == SNI_CORRECT
    assert integ["two_expert_oracle"] == TWO_EXPERT_ORACLE
    assert integ["three_expert_oracle"] == THREE_EXPERT_ORACLE
    assert integ["four_expert_oracle"] == FOUR_EXPERT_ORACLE
    assert integ["all_four_wrong"] == ALL_FOUR_WRONG
    assert integ["s0_unique_recoveries"] == S0_UNIQUE_RECOVERIES
    assert integ["sni_unique_recoveries"] == SNI_UNIQUE_RECOVERIES
    assert FOUR_EXPERT_ORACLE == THREE_EXPERT_ORACLE + SNI_UNIQUE_RECOVERIES
    assert THREE_EXPERT_ORACLE == TWO_EXPERT_ORACLE + S0_UNIQUE_RECOVERIES

    assert int(registry["lzh_correct"].sum()) == LZH_CORRECT
    assert int(registry["wyh_correct"].sum()) == WYH_CORRECT
    assert int(registry["s0_correct"].sum()) == S0_CORRECT
    assert int(registry["sni_correct"].sum()) == SNI_CORRECT
    assert int(registry["four_expert_oracle_correct"].sum()) == FOUR_EXPERT_ORACLE
    assert int(registry["all_four_wrong"].sum()) == ALL_FOUR_WRONG
    assert int(registry["s0_unique_recovery"].sum()) == S0_UNIQUE_RECOVERIES
    assert int(registry["sni_unique_recovery"].sum()) == SNI_UNIQUE_RECOVERIES

    h1 = trigger_h1(registry["lzh_pred"], registry["wyh_pred"], registry["sni_pred"])
    h2 = trigger_h2(registry["lzh_pred"], registry["wyh_pred"], registry["s0_pred"])
    h3 = trigger_h3(registry["lzh_pred"], registry["wyh_pred"], registry["sni_pred"])
    assert h1.tolist() == registry["h1_trigger"].astype(bool).tolist()
    assert h2.tolist() == registry["h2_trigger"].astype(bool).tolist()
    assert h3.tolist() == registry["h3_trigger"].astype(bool).tolist()
    assert not np.any(h1 & h2)
    assert metrics["triggers_use_predictions_only"] is True
    assert metrics["oracle_is_not_deployable_accuracy"] is True
    assert metrics["d0_d3_are_not_unbiased_final_oof"] is True
    assert metrics["not_model_v3"] is True


@pytest.mark.skipif(
    not (OUT / "v3_e05a_directional_metrics.json").is_file(),
    reason="V3-E05A artifacts not generated",
)
def test_d0_d3_and_report_contract():
    metrics = json.loads((OUT / "v3_e05a_directional_metrics.json").read_text())
    d = metrics["fixed_patch_diagnostics"]
    h = metrics["hypotheses"]
    assert d["D0"]["correct"] == LZH_CORRECT
    assert d["D0"]["changed"] == 0
    assert d["D1"]["net"] == h["H1"]["overall"]["net"]
    assert d["D2"]["net"] == h["H1"]["overall"]["net"] + h["H2"]["overall"]["net"]
    assert d["D3"]["net"] == (
        h["H1"]["overall"]["net"] + h["H2"]["overall"]["net"] + h["H3"]["overall"]["net"]
    )
    for key in ("D0", "D1", "D2", "D3"):
        assert d[key]["not_unbiased_final_oof"] is True
        assert d[key]["not_model_v3"] is True
        assert d[key]["label"] == "RETROSPECTIVE FIXED-PATCH DIAGNOSTIC"

    assert metrics["final_classification"]["label"] in {
        "DIRECTIONAL SIGNAL STRONG",
        "DIRECTIONAL SIGNAL LIMITED",
        "DIRECTIONAL SIGNAL WEAK",
    }
    leak = metrics["leakage_audit"]
    assert leak["competition_test_labels_used"] is False
    assert leak["true_labels_used_to_define_triggers"] is False
    assert leak["learned_router_trained"] is False
    assert leak["threshold_optimized"] is False
    assert leak["ensemble_weights_optimized"] is False
    assert leak["posthoc_pattern_promoted_to_candidate_rule"] is False
    assert leak["h1_h2_h3_motivated_by_prior_full_data_error_analysis"] is True

    text = REPORT.read_text()
    assert text.startswith("# V3-E05A — Asymmetric Directional Expert Complementarity Audit")
    assert "D0-D3 ARE NOT UNBIASED FINAL OOF RESULTS" in text
    assert metrics["final_classification"]["label"] in text
    assert "docs/versions/model_v3.md" in text
    assert not (ROOT / "work" / "docs" / "versions" / "model_v3.md").is_file()


@pytest.mark.skipif(not PRED.is_file(), reason="prediction.csv missing")
def test_prediction_csv_and_frozen_artifacts_unmodified_by_v3_e05a():
    before = sha256_file(PRED)
    status = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"], cwd=str(ROOT)
    ).decode()
    assert status == ""
    assert sha256_file(PRED) == before
    frozen = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert frozen == ""
    assert not (ROOT / "work" / "docs" / "versions" / "model_v3.md").is_file()
