"""Integrity tests for V3-E06M source-balanced multi-reference transfer.

Does not train a router or optimize source weights. Does not read competition
test labels for scoring. Does not modify prediction/prediction.csv.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "v3"))

from merfish60.io import load_dataset  # noqa: E402
from merfish60.models import assert_probability_rows  # noqa: E402
from merfish60.official_contract import allowed_labels, sha256_file  # noqa: E402
from merfish60.reference import EXPECTED_MD5 as MERFISH_MD5, md5_file  # noqa: E402

from v3_e02d_privileged_gene_distillation import gene_list_sha256  # noqa: E402
from v3_e06m_source_balanced_multireference import (  # noqa: E402
    ALL_FOUR_WRONG,
    EXPECTED_COMBINED_BEFORE_EXTRA,
    EXPECTED_SNI_MD5,
    FOUR_EXPERT_ORACLE,
    HISTORICAL_MERFISH_USABLE,
    HISTORICAL_SNI_USABLE,
    M0_ACCURACY,
    N_CLASSES,
    N_TEST,
    N_TRAIN,
    OFFICIAL_GENE_SHA256,
    SNI_UNIQUE_RECOVERIES,
    SOURCE_BALANCE,
    WYH_CORRECT,
    classify_m2,
    combined_row_identity,
    expected_feature_names,
    feature_schema_ok,
    source_balanced_weights,
    weight_summary,
)

OUT = ROOT / "outputs" / "v3"
PRED = ROOT / "prediction" / "prediction.csv"
MERFISH_PATH = ROOT / "work/external/MERFISH_spinal_cord_resolved_0718.h5ad"
SNI_PATH = ROOT / "work/external/SNI_merged_0917.h5ad"


def test_both_external_md5_values_when_files_present():
    if MERFISH_PATH.is_file():
        assert md5_file(MERFISH_PATH) == MERFISH_MD5
    else:
        pytest.skip("MERFISH h5ad is local/gitignored and is not present")
    if SNI_PATH.is_file():
        assert md5_file(SNI_PATH) == EXPECTED_SNI_MD5
    else:
        pytest.skip("SNI h5ad is local/gitignored and is not present")


def test_official_200_gene_order_hash_matches_frozen_v2():
    data = load_dataset(ROOT)
    genes = list(data.counts_train.columns)
    assert len(genes) == 200
    assert gene_list_sha256(genes) == OFFICIAL_GENE_SHA256
    assert gene_list_sha256(list(reversed(genes))) != OFFICIAL_GENE_SHA256
    assert hashlib.sha256(",".join(genes).encode("utf-8")).hexdigest() == OFFICIAL_GENE_SHA256


def test_frozen_counts_and_five_expert_identity():
    assert HISTORICAL_MERFISH_USABLE == 136574
    assert HISTORICAL_SNI_USABLE == 55193
    assert HISTORICAL_MERFISH_USABLE + HISTORICAL_SNI_USABLE == EXPECTED_COMBINED_BEFORE_EXTRA
    assert FOUR_EXPERT_ORACLE == 4364
    assert ALL_FOUR_WRONG == 636
    assert FOUR_EXPERT_ORACLE + ALL_FOUR_WRONG == N_TRAIN
    assert WYH_CORRECT / N_TRAIN == pytest.approx(M0_ACCURACY)
    new_unique = 11
    assert FOUR_EXPERT_ORACLE + new_unique == 4375


def test_m1_uniform_weight_is_v2_base_behavior():
    source = np.array(["MERFISH"] * 4 + ["SNI"] * 2)
    base = np.ones(6, dtype=np.float64)
    summary = weight_summary(source, base, base)
    assert summary["by_source"]["MERFISH"]["total_effective_weight"] == pytest.approx(4.0)
    assert summary["by_source"]["SNI"]["total_effective_weight"] == pytest.approx(2.0)
    assert summary["weight_sum_ratio_merfish_over_sni"] == pytest.approx(2.0)
    assert summary["mean_weight"] == pytest.approx(1.0)


def test_m2_equal_total_effective_source_mass_and_mean_one():
    n_mer, n_sni = 136574, 55193
    source = np.array(["MERFISH"] * n_mer + ["SNI"] * n_sni)
    base = np.ones(n_mer + n_sni, dtype=np.float64)
    weights = source_balanced_weights(source, base)
    mer = float(weights[source == "MERFISH"].sum())
    sni = float(weights[source == "SNI"].sum())
    assert mer == pytest.approx(sni, rel=0, abs=1e-8)
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-10)
    assert SOURCE_BALANCE == 0.5
    assert weights.min() > 0
    again = source_balanced_weights(source, base)
    assert np.allclose(weights, again)
    with pytest.raises(RuntimeError):
        source_balanced_weights(np.array(["MERFISH"] * 10), np.ones(10))


def test_source_column_is_not_a_model_input():
    data = load_dataset(ROOT)
    names = expected_feature_names(list(data.counts_train.columns))
    ok, reason = feature_schema_ok(names)
    assert ok, reason
    assert "reference_source" not in names
    assert "source" not in [n.lower() for n in names]
    assert "Condition" not in names
    bad_ok, _ = feature_schema_ok(names + ["reference_source"])
    assert bad_ok is False


def test_combined_row_identity_is_order_and_source_sensitive():
    ids = np.array(["1" * 19, "2" * 19], dtype=object)
    a = combined_row_identity(np.array(["MERFISH", "SNI"]), ids)
    b = combined_row_identity(np.array(["SNI", "MERFISH"]), ids)
    c = combined_row_identity(np.array(["MERFISH", "SNI"]), ids[::-1])
    assert a != b
    assert a != c
    assert a == combined_row_identity(np.array(["MERFISH", "SNI"]), ids)


def test_predeclared_decision_thresholds_are_not_post_hoc():
    strong, _ = classify_m2(
        net_vs_m0=25,
        acc=0.8262,
        net_02=10,
        net_34=15,
        macro_f1=0.7936,
        m0_macro_f1=0.7936,
        m2_correct=4131,
        m1_correct=4120,
        sni_capture=15,
        new_unique=15,
        integrity_ok=True,
    )
    assert strong == "STRONG SOURCE-BALANCED TRANSFER"
    promising, _ = classify_m2(
        net_vs_m0=8,
        acc=0.8228,
        net_02=6,
        net_34=2,
        macro_f1=0.7930,
        m0_macro_f1=0.7936,
        m2_correct=4114,
        m1_correct=4110,
        sni_capture=10,
        new_unique=4,
        integrity_ok=True,
    )
    assert promising == "PROMISING SOURCE-BALANCED TRANSFER"
    insufficient, _ = classify_m2(
        net_vs_m0=-3,
        acc=0.8206,
        net_02=1,
        net_34=-4,
        macro_f1=0.7900,
        m0_macro_f1=0.7936,
        m2_correct=4103,
        m1_correct=4110,
        sni_capture=2,
        new_unique=1,
        integrity_ok=True,
    )
    assert insufficient == "SOURCE-BALANCED INTEGRATION INSUFFICIENT"
    leak, _ = classify_m2(40, 0.83, 20, 20, 0.80, 0.79, 4150, 4140, 20, 20, False)
    assert leak == "SOURCE-BALANCED INTEGRATION INSUFFICIENT"


def test_no_prediction_csv_modification_contract():
    assert PRED.is_file()
    digest = sha256_file(PRED)
    again = sha256_file(PRED)
    assert digest == again
    status = subprocess.check_output(
        ["git", "status", "--short", "prediction/prediction.csv"], cwd=str(ROOT)
    )
    assert status.decode("utf-8").strip() == ""


def _require_outputs():
    needed = [
        OUT / "v3_e06m_dataset_manifest.json",
        OUT / "v3_e06m_source_weight_audit.json",
        OUT / "v3_e06m_candidate_comparison.csv",
        OUT / "v3_e06m_m1_validation.csv",
        OUT / "v3_e06m_m1_validation_probabilities.csv.gz",
        OUT / "v3_e06m_m1_test_probabilities.csv.gz",
        OUT / "v3_e06m_m2_validation.csv",
        OUT / "v3_e06m_m2_validation_probabilities.csv.gz",
        OUT / "v3_e06m_m2_test_probabilities.csv.gz",
        OUT / "v3_e06m_cell_deltas.csv",
        OUT / "v3_e06m_sni_transfer_capture.csv",
        OUT / "v3_e06m_pairwise_oracle.csv",
        OUT / "v3_e06m_complementarity.json",
        OUT / "v3_e06m_new_unique_recoveries.csv",
        ROOT / "reports/v3/v3_e06m_source_balanced_multireference.md",
    ]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        pytest.skip("V3-E06M artifacts not written yet: {}".format(missing[0]))


def test_cleaned_source_counts_and_combined_identity_from_manifest():
    _require_outputs()
    man = json.loads((OUT / "v3_e06m_dataset_manifest.json").read_text())
    assert man["merfish"]["n_usable"] == HISTORICAL_MERFISH_USABLE
    assert man["sni"]["n_usable_e04s"] == HISTORICAL_SNI_USABLE
    assert man["combined_before_extra"] == EXPECTED_COMBINED_BEFORE_EXTRA
    assert man["gene_order_sha256"] == OFFICIAL_GENE_SHA256
    assert man["merfish"]["n_classes"] == 60
    assert man["sni"]["n_classes"] == 60
    assert man["reference_source_used_as_feature"] is False
    assert man["merfish"]["md5"] == MERFISH_MD5
    assert man["sni"]["md5"] == EXPECTED_SNI_MD5


def test_m2_source_weight_audit_is_exactly_balanced():
    _require_outputs()
    audit = json.loads((OUT / "v3_e06m_source_weight_audit.json").read_text())
    assert audit["M1"]["mean_weight"] == pytest.approx(1.0)
    assert audit["M1"]["by_source"]["MERFISH"]["count"] == HISTORICAL_MERFISH_USABLE
    assert audit["M2"]["mean_weight"] == pytest.approx(1.0, abs=1e-8)
    assert audit["M2"]["weight_sum_ratio_merfish_over_sni"] == pytest.approx(1.0, abs=1e-8)
    assert audit["M2"]["balance_ok"] is True
    mer = audit["M2"]["by_source"]["MERFISH"]["total_effective_weight"]
    sni = audit["M2"]["by_source"]["SNI"]["total_effective_weight"]
    assert mer == pytest.approx(sni, abs=1e-6)


def test_validation_and_test_probability_contracts():
    _require_outputs()
    class_names = allowed_labels(ROOT)
    assert len(class_names) == N_CLASSES
    cases = [
        ("m1_validation", N_TRAIN, OUT / "v3_e06m_m1_validation.csv", OUT / "v3_e06m_m1_validation_probabilities.csv.gz"),
        ("m2_validation", N_TRAIN, OUT / "v3_e06m_m2_validation.csv", OUT / "v3_e06m_m2_validation_probabilities.csv.gz"),
        ("m1_test", N_TEST, None, OUT / "v3_e06m_m1_test_probabilities.csv.gz"),
        ("m2_test", N_TEST, None, OUT / "v3_e06m_m2_test_probabilities.csv.gz"),
    ]
    for _name, n, label_path, proba_path in cases:
        if label_path is not None:
            labels = pd.read_csv(label_path, dtype={"Cell_ID": str})
            assert len(labels) == n
            assert labels["Cell_ID"].astype(str).str.len().eq(19).all()
        proba = pd.read_csv(proba_path, dtype={"Cell_ID": str})
        assert len(proba) == n
        assert list(proba.columns[1:]) == class_names
        matrix = proba.loc[:, class_names].to_numpy(dtype=np.float64)
        assert_probability_rows(matrix, atol=1e-4)
        assert np.isfinite(matrix).all()
        assert (matrix >= 0).all()


def test_frozen_m0_accuracy_and_four_expert_oracle_in_outputs():
    _require_outputs()
    cmp_ = pd.read_csv(OUT / "v3_e06m_candidate_comparison.csv")
    m0 = cmp_.set_index("model").loc["M0"]
    assert int(m0["correct"]) == WYH_CORRECT
    assert float(m0["accuracy"]) == pytest.approx(M0_ACCURACY)
    comp = json.loads((OUT / "v3_e06m_complementarity.json").read_text())
    assert comp["four_expert_oracle"] == FOUR_EXPERT_ORACLE
    assert comp["all_four_wrong"] == ALL_FOUR_WRONG
    assert comp["sni_unique_recoveries"] == SNI_UNIQUE_RECOVERIES
    assert comp["m2_five_expert_oracle_correct"] == FOUR_EXPERT_ORACLE + comp["m2_new_unique_recoveries"]
    rec = pd.read_csv(OUT / "v3_e06m_sni_transfer_capture.csv", dtype={"Cell_ID": str})
    assert len(rec) == SNI_UNIQUE_RECOVERIES
    new = pd.read_csv(OUT / "v3_e06m_new_unique_recoveries.csv", dtype={"Cell_ID": str})
    assert len(new) == comp["m2_new_unique_recoveries"]


def test_report_contains_required_decision_and_oracle_warning():
    _require_outputs()
    text = (ROOT / "reports/v3/v3_e06m_source_balanced_multireference.md").read_text()
    assert text.startswith("# V3-E06M — Source-Balanced Multi-Reference Transfer")
    assert "ORACLE != DEPLOYABLE ACCURACY" in text
    marked = [
        label
        for label in (
            "STRONG SOURCE-BALANCED TRANSFER",
            "PROMISING SOURCE-BALANCED TRANSFER",
            "SOURCE-BALANCED INTEGRATION INSUFFICIENT",
        )
        if "**{}".format(label) in text
    ]
    assert len(marked) == 1
