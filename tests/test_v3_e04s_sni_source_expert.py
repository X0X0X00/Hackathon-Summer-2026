"""Integrity tests for V3-E04S source-diverse SNI expert.

Does not concatenate SNI with the MERFISH reference. Does not train a router.
Does not read competition test labels for scoring. Does not modify
prediction/prediction.csv.
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
from merfish60.official_contract import allowed_labels, sha256_file  # noqa: E402
from merfish60.reference import md5_file, norm_label  # noqa: E402

from v3_e02d_privileged_gene_distillation import gene_list_sha256  # noqa: E402
from v3_e04s_sni_source_expert import (  # noqa: E402
    ALL_THREE_WRONG,
    EXPECTED_SNI_MD5,
    N_CLASSES,
    N_TRAIN,
    OFFICIAL_GENE_SHA256,
    SNI_H5AD_REL,
    SNI_LABEL_COL,
    THREE_EXPERT_ORACLE,
    classify_experiment,
    feasibility_decision,
    pairwise_oracle_row,
    sni_h5ad_path,
)

OUT = ROOT / "outputs" / "v3"
PRED = ROOT / "prediction" / "prediction.csv"
SNI_PATH = ROOT / SNI_H5AD_REL


def test_sni_md5_and_provenance_when_file_present():
    if not SNI_PATH.is_file():
        pytest.skip("SNI h5ad is local/gitignored and is not present")
    assert md5_file(SNI_PATH) == EXPECTED_SNI_MD5
    assert SNI_PATH.name == "SNI_merged_0917.h5ad"
    assert str(sni_h5ad_path(ROOT)) == str(SNI_PATH)


def test_official_200_gene_hash_is_order_sensitive_and_matches_v2_contract():
    data = load_dataset(ROOT)
    genes = list(data.counts_train.columns)
    assert len(genes) == 200
    assert gene_list_sha256(genes) == OFFICIAL_GENE_SHA256
    shuffled = list(reversed(genes))
    assert gene_list_sha256(shuffled) != OFFICIAL_GENE_SHA256
    assert gene_list_sha256(sorted(genes)) != OFFICIAL_GENE_SHA256 or sorted(genes) == genes


def test_existing_norm_label_maps_sni_style_names_without_new_biology():
    class_names = allowed_labels(ROOT)
    assert norm_label("oligodendrocyte progenitor-2") == "oligodendrocyte_progenitor_2"
    assert norm_label("alpha motoneuron") == "alpha_motoneuron"
    assert norm_label("DH-ex-Reln/Nmur2") == "DH_ex_Reln/Nmur2"
    assert norm_label("oligodendrocyte progenitor-2") in class_names
    assert SNI_LABEL_COL == "voting"


def test_competition_cell_id_exclusions_are_set_algebra():
    sni_ids = np.array(["1" * 19, "2" * 19, "3" * 19], dtype=object)
    train_ids = {"1" * 19}
    test_ids = {"2" * 19}
    in_train = np.fromiter((cid in train_ids for cid in sni_ids), dtype=bool, count=3)
    in_test = np.fromiter((cid in test_ids for cid in sni_ids), dtype=bool, count=3)
    keep = ~(in_train | in_test)
    assert in_train.tolist() == [True, False, False]
    assert in_test.tolist() == [False, True, False]
    assert keep.tolist() == [False, False, True]


def test_exact_vector_duplicate_policy_excludes_competition_and_reference_matches():
    dup_train = np.array([True, False, False, False])
    dup_test = np.array([False, True, False, False])
    dup_mer = np.array([False, False, True, False])
    id_overlap = np.array([False, False, False, False])
    keep = ~(id_overlap | dup_train | dup_test | dup_mer)
    assert keep.tolist() == [False, False, False, True]
    assert int(dup_train.sum()) + int(dup_test.sum()) + int(dup_mer.sum()) == 3


def test_four_expert_identity_and_frozen_starting_oracle():
    lzh_ok = np.array([True, True, False, False, False])
    wyh_ok = np.array([True, False, True, False, False])
    s0_ok = np.array([False, False, False, True, False])
    sni_ok = np.array([False, False, False, False, True])
    three = lzh_ok | wyh_ok | s0_ok
    unique = (~lzh_ok) & (~wyh_ok) & (~s0_ok) & sni_ok
    four = three | sni_ok
    assert int(three.sum()) == 4
    assert int((~three).sum()) == 1
    assert int(unique.sum()) == 1
    assert int(four.sum()) == int(three.sum()) + int(unique.sum())
    assert THREE_EXPERT_ORACLE == 4311
    assert ALL_THREE_WRONG == 689
    assert THREE_EXPERT_ORACLE + ALL_THREE_WRONG == N_TRAIN
    assert THREE_EXPERT_ORACLE + 40 == 4351
    assert 4351 / 5000 == pytest.approx(0.8702)


def test_predeclared_decision_thresholds_are_not_post_hoc():
    strong, _ = classify_experiment(40, 24, 16, 0.8702, True)
    assert strong == "STRONG SOURCE-DIVERSE EXPERT"
    promising, _ = classify_experiment(25, 15, 10, 0.866, True)
    assert promising == "PROMISING SOURCE-DIVERSE EXPERT"
    weak, _ = classify_experiment(19, 12, 7, 0.866, True)
    assert weak == "SOURCE DIVERSITY INSUFFICIENT"
    no_holdout, _ = classify_experiment(30, 30, 0, 0.868, True)
    assert no_holdout == "SOURCE DIVERSITY INSUFFICIENT"
    leak, _ = classify_experiment(50, 30, 20, 0.88, False)
    assert leak == "SOURCE DIVERSITY INSUFFICIENT"


def test_pairwise_oracle_is_not_deployable_accuracy():
    ok_a = np.array([True, True, False, False])
    ok_b = np.array([True, False, True, False])
    pred_a = np.array(["a", "a", "x", "x"])
    pred_b = np.array(["a", "b", "a", "y"])
    row = pairwise_oracle_row("A", "B", ok_a, ok_b, pred_a, pred_b)
    assert row["pair_oracle_count"] == 3
    assert row["pair_oracle_accuracy"] == pytest.approx(0.75)
    assert row["A_only_correct"] == 1
    assert row["B_only_correct"] == 1
    assert row["oracle_is_not_deployable_accuracy"] is True


def test_feasibility_requires_md5_genes_and_taxonomy():
    ok, reason = feasibility_decision({"md5_ok": False})
    assert ok is False and reason == "SNI CHECKSUM FAILURE"
    ok, reason = feasibility_decision(
        {
            "md5_ok": True,
            "official_gene_alignment": {"exact_200_of_200": False, "matches_official_gene_contract": False},
        }
    )
    assert ok is False and reason == "SNI GENE ALIGNMENT FAILURE"
    ok, reason = feasibility_decision(
        {
            "md5_ok": True,
            "official_gene_alignment": {"exact_200_of_200": True, "matches_official_gene_contract": True},
            "taxonomy": {"case": "C"},
        }
    )
    assert ok is False and reason == "SNI TAXONOMY MAPPING NEEDS HUMAN REVIEW"


@pytest.mark.skipif(not (OUT / "v3_e04s_complementarity.json").is_file(), reason="V3-E04S artifacts not generated")
def test_validation_cell_identity_probabilities_and_oracle_identity():
    metrics = json.loads((OUT / "v3_e04s_complementarity.json").read_text())
    audit = json.loads((OUT / "v3_e04s_dataset_audit.json").read_text())
    val = pd.read_csv(OUT / "v3_e04s_sni_validation.csv", dtype={"Cell_ID": str})
    proba = pd.read_csv(OUT / "v3_e04s_sni_validation_probabilities.csv.gz", dtype={"Cell_ID": str})
    test_proba = pd.read_csv(OUT / "v3_e04s_sni_test_probabilities.csv.gz", dtype={"Cell_ID": str})
    shared = pd.read_csv(OUT / "v3_e04s_shared_failure_registry.csv", dtype={"Cell_ID": str})
    rec = pd.read_csv(OUT / "v3_e04s_new_unique_recoveries.csv", dtype={"Cell_ID": str})
    data = load_dataset(ROOT)
    class_names = allowed_labels(ROOT)

    assert len(val) == N_TRAIN
    assert val["Cell_ID"].nunique() == N_TRAIN
    assert not val["Cell_ID"].duplicated().any()
    assert val["Cell_ID"].str.fullmatch(r"\d{19}").all()
    assert not val["Cell_ID"].str.endswith(".0").any()
    assert set(val["Cell_ID"].astype(str)) == set(data.meta_train.index.astype(str))
    aligned = val.set_index("Cell_ID")["true_label"]
    official = data.meta_train[TARGET_COL].astype(str)
    assert list(aligned.reindex(official.index.astype(str))) == list(official)

    assert list(proba.columns[1:]) == list(class_names)
    assert len(class_names) == N_CLASSES
    arr = proba[class_names].to_numpy(dtype=np.float64)
    assert_probability_rows(arr, atol=1e-4)
    assert len(proba) == N_TRAIN
    assert list(test_proba.columns[1:]) == list(class_names)
    assert len(test_proba) == 5000
    assert_probability_rows(test_proba[class_names].to_numpy(dtype=np.float64), atol=1e-4)
    test_ids = [str(v) for v in data.meta_test.index.tolist()]
    assert list(test_proba["Cell_ID"].astype(str)) == test_ids

    assert audit["md5"] == EXPECTED_SNI_MD5
    assert audit["official_gene_alignment"]["exact_200_of_200"] is True
    assert audit["official_gene_alignment"]["ordered_200_gene_sha256"] == OFFICIAL_GENE_SHA256
    assert audit["taxonomy"]["mapped_class_count"] == 60
    assert audit["competition_train_cell_id_overlap"] == 0
    assert audit["competition_test_cell_id_overlap"] == 0
    assert len(shared) == ALL_THREE_WRONG
    assert metrics["frozen_three_expert_oracle_correct"] == THREE_EXPERT_ORACLE
    assert metrics["frozen_all_three_wrong"] == ALL_THREE_WRONG
    assert metrics["four_expert_oracle_correct"] == THREE_EXPERT_ORACLE + metrics["sni_new_unique_recoveries"]
    assert metrics["four_expert"]["identity_check"] is True
    assert len(rec) == metrics["sni_new_unique_recoveries"]
    assert metrics["oracle_is_not_deployable_accuracy"] is True
    assert metrics["leakage_audit"]["competition_test_labels_used"] is False
    assert metrics["leakage_audit"]["competition_train_labels_used_for_sni_fitting"] is False
    assert metrics["leakage_audit"]["hyperparameter_search"] is False
    assert metrics["leakage_audit"]["blend_optimization"] is False
    assert metrics["leakage_audit"]["sni_concatenated_with_merfish_reference"] is False


@pytest.mark.skipif(not PRED.is_file(), reason="prediction.csv missing")
def test_prediction_csv_unmodified_by_v3_e04s():
    before = sha256_file(PRED)
    status = subprocess.check_output(["git", "diff", "--", "prediction/prediction.csv"], cwd=str(ROOT)).decode()
    assert status == ""
    assert sha256_file(PRED) == before
    if (OUT / "v3_e04s_complementarity.json").is_file():
        metrics = json.loads((OUT / "v3_e04s_complementarity.json").read_text())
        assert metrics["leakage_audit"]["prediction_csv_modified"] is False
        assert metrics["leakage_audit"]["submission_generated"] is False
