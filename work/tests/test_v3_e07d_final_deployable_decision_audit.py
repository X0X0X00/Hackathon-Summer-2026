"""Integrity tests for V3-E07D final deployable candidate decision audit.

Does not train a model, router, or stacker. Does not blend experts, search
thresholds, or read competition test labels for scoring. Does not modify
prediction/prediction.csv or create MODEL V3.
"""
from pathlib import Path
import inspect
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work" / "src"))
sys.path.insert(0, str(ROOT / "work" / "experiments" / "v3"))

from merfish60.io import TARGET_COL, load_dataset  # noqa: E402
from merfish60.models import assert_probability_rows  # noqa: E402
from merfish60.official_contract import allowed_labels, sha256_file  # noqa: E402
from merfish60.team_cv import load_and_validate_team_folds  # noqa: E402

from v3_e06m_source_balanced_multireference import (  # noqa: E402
    STRONG_ACCURACY_MIN,
    STRONG_NET_VS_M0,
)
from v3_e07d_final_deployable_decision_audit import (  # noqa: E402
    ALLOWED_PERSONAL_DECISIONS,
    ALLOWED_TEAM_DECISIONS,
    BOOTSTRAP_SEED,
    E06M_FROZEN_LABEL,
    E06M_STRONG_ACCURACY_MIN,
    E06M_STRONG_NET_VS_M0,
    FIVE_EXPERT_ORACLE,
    FOUR_EXPERT_ORACLE,
    LZH_CORRECT,
    M0_CORRECT,
    M2_CORRECT,
    M2_NEW_UNIQUE,
    M2_VS_M0_CORRECT_TO_WRONG,
    M2_VS_M0_NET,
    M2_VS_M0_WRONG_TO_CORRECT,
    N_BOOTSTRAP,
    N_TEST,
    N_TRAIN,
    equal_width_ece,
    js_divergence_vectors,
    mcnemar_exact,
    paired_accuracy_deltas,
    personal_promotion_decision,
    predicted_class_frequency,
    section_cluster_deltas,
    strong_criteria_met,
    team_standalone_decision,
)

OUT = ROOT / "work" / "experiments" / "evidence" / "v3"
PRED = ROOT / "prediction" / "prediction.csv"
REPORT = ROOT / "work" / "reports" / "v3" / "v3_e07d_final_deployable_decision_audit.md"
REGISTRY = OUT / "v3_e07d_deployable_registry.parquet"
DECISION = OUT / "v3_e07d_decision.json"
MODEL_V3_DOC = ROOT / "work" / "docs" / "versions" / "model_v3.md"


def _require_outputs():
    if not REGISTRY.is_file() or not DECISION.is_file():
        pytest.skip("V3-E07D artifacts not generated")


def test_frozen_identity_constants_and_strong_bar_not_lowered():
    assert M0_CORRECT == 4106
    assert M2_CORRECT == 4109
    assert LZH_CORRECT == 4133
    assert M2_VS_M0_WRONG_TO_CORRECT == 105
    assert M2_VS_M0_CORRECT_TO_WRONG == 102
    assert M2_VS_M0_NET == 3
    assert FOUR_EXPERT_ORACLE == 4364
    assert FIVE_EXPERT_ORACLE == 4393
    assert FOUR_EXPERT_ORACLE + M2_NEW_UNIQUE == FIVE_EXPERT_ORACLE
    assert E06M_STRONG_NET_VS_M0 == 25
    assert E06M_STRONG_ACCURACY_MIN == 0.8262
    assert STRONG_NET_VS_M0 == E06M_STRONG_NET_VS_M0
    assert STRONG_ACCURACY_MIN == E06M_STRONG_ACCURACY_MIN
    assert BOOTSTRAP_SEED == 20260819
    assert N_BOOTSTRAP == 10000


def test_plus_three_cells_cannot_promote_model_v3():
    strong_ok, failed = strong_criteria_met(
        net=3,
        acc=0.8218,
        net_02=12,
        net_34=-9,
        macro_f1=0.7955,
        m0_macro_f1=0.7936,
        sni_capture=4,
        new_unique=29,
        m2_correct=4109,
        m1_correct=4086,
    )
    assert strong_ok is False
    assert "net_ge_25" in failed
    assert "acc_ge_0.8262" in failed
    assert "folds_3_4_net_positive" in failed
    decision = personal_promotion_decision(
        e06m_label=E06M_FROZEN_LABEL,
        strong_ok=False,
        m2_correct=4109,
        m0_correct=4106,
        net=3,
        net_34=-9,
    )
    assert decision == "MODEL V3 PROMOTION NOT JUSTIFIED"
    assert decision in ALLOWED_PERSONAL_DECISIONS


def test_team_standalone_ranking_uses_accuracy_not_oracle():
    src = inspect.getsource(team_standalone_decision)
    assert "oracle" not in src
    label = team_standalone_decision(
        lzh_correct=4133,
        m2_correct=4109,
        m0_correct=4106,
        lzh_net_vs_m0=27,
        lzh_net_vs_m2=24,
        m2_net_vs_m0=3,
        lzh_folds_34=0.8210,
        m2_folds_34=0.8160,
        m0_folds_34=0.8205,
    )
    assert label == "LZH REMAINS STRONGEST AUDITABLE STANDALONE EXPERT"
    assert label in ALLOWED_TEAM_DECISIONS


def test_mcnemar_uses_exact_binomial_not_chi_square():
    src = inspect.getsource(mcnemar_exact)
    assert "binomtest" in src
    assert "chi2" not in src.lower()
    assert "chi_square" not in src
    out = mcnemar_exact(105, 102)
    expected = binomtest(k=105, n=207, p=0.5, alternative="two-sided").pvalue
    assert out["n_discordant"] == 207
    assert out["net"] == 3
    assert out["pvalue"] == pytest.approx(expected)
    empty = mcnemar_exact(0, 0)
    assert empty["pvalue"] == 1.0


def test_bootstrap_seed_is_deterministic():
    rng_ok_a = np.array([True, False, True, True, False, False, True, True])
    rng_ok_b = np.array([True, True, True, False, False, True, True, False])
    first = paired_accuracy_deltas(rng_ok_a, rng_ok_b, seed=20260819, n_resamples=32)
    second = paired_accuracy_deltas(rng_ok_a, rng_ok_b, seed=20260819, n_resamples=32)
    other = paired_accuracy_deltas(rng_ok_a, rng_ok_b, seed=0, n_resamples=32)
    assert np.allclose(first, second)
    assert not np.allclose(first, other)
    sections = np.array(["A", "A", "B", "B", "C", "C", "C", "A"], dtype=object)
    s1 = section_cluster_deltas(sections, rng_ok_a, rng_ok_b, seed=20260819, n_resamples=16)
    s2 = section_cluster_deltas(sections, rng_ok_a, rng_ok_b, seed=20260819, n_resamples=16)
    assert np.allclose(s1, s2)


def test_js_divergence_and_ece_are_deterministic():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.5, 0.5, 0.0])
    assert js_divergence_vectors(p, q) == pytest.approx(0.0)
    names = ["a", "b", "c"]
    pred = np.array(["a", "a", "b", "c"])
    freq = predicted_class_frequency(pred, names)
    assert freq.sum() == pytest.approx(1.0)
    assert freq[0] == pytest.approx(0.5)
    correct = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1], dtype=float)
    conf = np.array([0.95, 0.91, 0.12, 0.08, 0.88, 0.72, 0.33, 0.81, 0.99, 0.60])
    ece = equal_width_ece(correct, conf, n_bins=10)
    assert ece["n_bins"] == 10
    assert 0.0 <= ece["ece"] <= 1.0
    assert len(ece["bins"]) == 10


def test_experiment_is_analysis_only():
    src = Path(ROOT / "work/experiments/v3/v3_e07d_final_deployable_decision_audit.py").read_text()
    assert "lgb.train" not in src
    assert "lightgbm" not in src
    assert "model-v3" in src
    assert "Does not train" in src or "does not train" in src.lower()
    assert "MODEL V3 PROMOTION NOT JUSTIFIED" in src


def test_registry_cell_identity_and_frozen_correct_counts():
    _require_outputs()
    registry = pd.read_parquet(REGISTRY)
    data = load_dataset(ROOT)
    assert len(registry) == N_TRAIN
    assert registry["Cell_ID"].nunique() == N_TRAIN
    assert not registry["Cell_ID"].duplicated().any()
    assert registry["Cell_ID"].astype(str).str.fullmatch(r"\d{19}").all()
    assert not registry["Cell_ID"].astype(str).str.endswith(".0").any()
    official = data.meta_train[TARGET_COL].astype(str)
    aligned = registry.set_index("Cell_ID")["true_label"].astype(str)
    assert list(aligned.reindex(official.index.astype(str))) == list(official)
    assert int(registry["m0_correct"].sum()) == M0_CORRECT
    assert int(registry["m2_correct"].sum()) == M2_CORRECT
    assert int(registry["lzh_correct"].sum()) == LZH_CORRECT
    w2c = int((registry["m2_beats_m0"]).sum())
    c2w = int((registry["m0_beats_m2"]).sum())
    assert w2c == M2_VS_M0_WRONG_TO_CORRECT
    assert c2w == M2_VS_M0_CORRECT_TO_WRONG
    assert w2c - c2w == M2_VS_M0_NET
    assert registry["Section_ID"].astype(str).ne("").all()
    assert registry["Section_ID"].astype(str).ne("nan").all()
    assert not registry["Section_ID"].isna().any()


def test_canonical_folds_and_oracle_identities():
    _require_outputs()
    registry = pd.read_parquet(REGISTRY)
    data = load_dataset(ROOT)
    train_ids = [str(v) for v in data.meta_train.index.astype(str)]
    test_ids = [str(v) for v in data.meta_test.index.astype(str)]
    y_true = data.meta_train.loc[train_ids, TARGET_COL].astype(str).to_numpy()
    folds, _ = load_and_validate_team_folds(
        train_ids, test_ids, y_true, ROOT / "work" / "experiments" / "team_folds_5_seed42.csv"
    )
    expected = folds.set_index("Cell_ID").reindex(registry["Cell_ID"].astype(str))["fold"].astype(int)
    assert list(registry["canonical_fold"].astype(int)) == list(expected)
    e05a = pd.read_parquet(OUT / "v3_e05a_four_expert_registry.parquet")
    e05a = e05a.set_index("Cell_ID").reindex(registry["Cell_ID"].astype(str))
    four = (
        e05a["lzh_correct"].to_numpy(dtype=bool)
        | e05a["wyh_correct"].to_numpy(dtype=bool)
        | e05a["s0_correct"].to_numpy(dtype=bool)
        | e05a["sni_correct"].to_numpy(dtype=bool)
    )
    m2_ok = registry["m2_correct"].to_numpy(dtype=bool)
    assert int(four.sum()) == FOUR_EXPERT_ORACLE
    new_unique = m2_ok & ~four
    assert int(new_unique.sum()) == M2_NEW_UNIQUE
    assert int((four | m2_ok).sum()) == FIVE_EXPERT_ORACLE


def test_probability_rows_normalization_and_counts():
    _require_outputs()
    class_names = allowed_labels()
    cases = [
        (N_TRAIN, ROOT / "work/outputs/probabilities/V2-B-REFONLY_oof_probabilities_seg.csv.gz"),
        (N_TRAIN, OUT / "v3_e06m_m2_validation_probabilities.csv.gz"),
        (N_TEST, ROOT / "work/outputs/probabilities/V2-B-REFONLY_test_probabilities_seg.csv.gz"),
        (N_TEST, OUT / "v3_e06m_m2_test_probabilities.csv.gz"),
    ]
    for n, path in cases:
        frame = pd.read_csv(path, dtype={"Cell_ID": str})
        assert len(frame) == n
        assert frame["Cell_ID"].astype(str).str.fullmatch(r"\d{19}").all()
        matrix = frame.loc[:, class_names].to_numpy(dtype=np.float64)
        assert matrix.shape == (n, 60)
        assert_probability_rows(matrix, atol=1e-4)
        assert np.isfinite(matrix).all()
        assert (matrix >= -1e-12).all()


def test_bootstrap_artifact_uses_frozen_seed():
    if not (OUT / "v3_e07d_bootstrap.json").is_file():
        pytest.skip("V3-E07D bootstrap artifact missing")
    boot = json.loads((OUT / "v3_e07d_bootstrap.json").read_text())
    assert boot["seed"] == 20260819
    assert boot["n_resamples"] == 10000
    assert boot["not_used_to_invent_promotion_threshold"] is True
    for key in ("cell_level", "section_cluster"):
        block = boot[key]
        assert "mean_delta" in block
        assert "p2_5" in block
        assert "p97_5" in block
        assert 0.0 <= block["fraction_gt_0"] <= 1.0


def test_decision_and_report_contract():
    if not DECISION.is_file():
        pytest.skip("V3-E07D decision artifact missing")
    metrics = json.loads(DECISION.read_text())
    assert metrics["decisions"]["team_standalone"] in ALLOWED_TEAM_DECISIONS
    assert metrics["decisions"]["personal_version"] == "MODEL V3 PROMOTION NOT JUSTIFIED"
    assert metrics["decisions"]["replace_model_v2_with_m2"] is False
    assert metrics["created_model_v3"] is False
    assert metrics["oracle_is_not_deployable_accuracy"] is True
    assert metrics["leakage_audit"]["competition_test_labels_used"] is False
    assert metrics["leakage_audit"]["trained_model"] is False
    assert metrics["test_shift"]["test_labels_used"] is False
    assert metrics["e06m_strong_audit"]["strong_criteria_met"] is False
    assert metrics["e06m_strong_audit"]["reproduced_label"] == E06M_FROZEN_LABEL
    assert metrics["integrity"]["m0_correct"] == M0_CORRECT
    assert metrics["integrity"]["m2_correct"] == M2_CORRECT
    assert metrics["integrity"]["lzh_correct"] == LZH_CORRECT
    text = REPORT.read_text()
    assert text.startswith("# V3-E07D — Final Deployable Candidate Decision Audit")
    required = [
        "## 1. Decision Question",
        "## 4. Integrity Reproduction",
        "## 8. M2 vs M0 Paired Cell-Level Audit",
        "## 13. Test-Side Descriptive Shift",
        "NO TEST LABELS WERE USED",
        "ORACLE != DEPLOYABLE ACCURACY",
        "TEAM STANDALONE DECISION:",
        "PERSONAL VERSION DECISION:",
        "MODEL V3 PROMOTION NOT JUSTIFIED",
        "LZH REMAINS STRONGEST AUDITABLE STANDALONE EXPERT",
    ]
    for token in required:
        assert token in text
    assert not MODEL_V3_DOC.is_file()


def test_prediction_csv_frozen_artifacts_and_no_model_v3_tag():
    before = sha256_file(PRED) if PRED.is_file() else None
    status = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"], cwd=str(ROOT)
    ).decode()
    assert status == ""
    if before is not None:
        assert sha256_file(PRED) == before
    frozen = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert frozen == ""
    tags = subprocess.check_output(["git", "tag", "--list", "model-v3"], cwd=str(ROOT)).decode().strip()
    assert tags == ""
    assert not MODEL_V3_DOC.is_file()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=str(ROOT)).decode().strip()
    assert (
        branch == "ywan/ml-pipeline"
        or branch == "wyh"
        or branch.startswith("wyh-delivery-closure-")
    ), f"Unexpected branch for V3 integrity audit: {branch}"
