"""Integrity tests for V3-E02D privileged-gene dual-level distillation.

Does not train the full reference models. Does not read competition test labels
for scoring. Does not modify prediction/prediction.csv.
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

from merfish60.io import TARGET_COL, load_dataset  # noqa: E402
from merfish60.models import assert_probability_rows  # noqa: E402
from merfish60.official_contract import allowed_labels  # noqa: E402
from merfish60.reference import (  # noqa: E402
    EXPECTED_MD5,
    EXPECTED_N_OBS,
    EXPECTED_N_VARS,
    HISTORICAL_USABLE_ROWS,
    HISTORICAL_VECTOR_DUPES,
    audit_reference,
    reference_h5ad_path,
)

from v3_e02d_privileged_gene_distillation import (  # noqa: E402
    CLASS_FAMILIES,
    LZH_WYH_ORACLE_CORRECT,
    SHARED_ERROR_N,
    class_prototypes,
    complementarity_block,
    cosine_to_prototypes,
    family_of,
    gene_list_sha256,
    oracle_union,
    privileged_signal_decision,
    softmax_from_logits,
    unique_recovery_mask,
)

OUT = ROOT / "outputs" / "v3"
PRED = ROOT / "prediction" / "prediction.csv"


def test_gene_list_hash_is_order_sensitive():
    a = gene_list_sha256(["Gad1", "Slc17a6"])
    b = gene_list_sha256(["Slc17a6", "Gad1"])
    c = gene_list_sha256(["Gad1", "Slc17a6"])
    assert a != b
    assert a == c
    assert len(a) == 64


def test_official_200_subset_of_teacher_500_helper():
    genes200 = ["a", "c"]
    genes500 = ["a", "b", "c", "d"]
    assert set(genes200).issubset(set(genes500))
    assert gene_list_sha256(genes200) != gene_list_sha256(genes500)


def test_family_map_covers_priority_groups():
    assert family_of("oligodendrocyte_1") == "oligodendrocyte_opc"
    assert family_of("astrocyte_2") == "astrocyte"
    assert family_of("endothelial") == "vascular"
    assert family_of("meninges_1") == "meningeal"
    assert family_of("microglia") == "other_glial_non_neuronal"
    assert family_of("not_a_real_class") == "neuronal_or_other"


def test_unique_recovery_and_oracle_identity():
    y = np.array(["a", "a", "b", "c", "c"])
    lzh = np.array(["a", "x", "x", "c", "x"])
    wyh = np.array(["x", "x", "b", "c", "x"])
    st = np.array(["a", "a", "x", "x", "c"])
    lzh_ok = lzh == y
    wyh_ok = wyh == y
    st_ok = st == y
    rec = unique_recovery_mask(lzh_ok, wyh_ok, st_ok)
    assert rec.tolist() == [False, True, False, False, True]
    assert int(rec.sum()) == 2
    pair = oracle_union(lzh_ok, wyh_ok)
    three = oracle_union(lzh_ok, wyh_ok, st_ok)
    assert int(pair.sum()) + int(rec.sum()) == int(three.sum())
    folds = np.array([0, 1, 2, 3, 4])
    block = complementarity_block(y, lzh, wyh, st, folds)
    assert block["new_unique_recoveries"] == 2
    assert block["oracle_identity_check"] is True
    assert block["all_three_wrong"] == 0
    assert block["lzh_wyh_oracle_correct"] == 3


def test_relation_target_shape_is_n_classes():
    rng = np.random.RandomState(0)
    lat = rng.randn(12, 128)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    proto = class_prototypes(lat, y, 3)
    rel = cosine_to_prototypes(lat, proto)
    assert proto.shape == (3, 128)
    assert rel.shape == (12, 3)
    assert np.all(np.isfinite(rel))
    assert np.max(np.abs(rel)) <= 1.000001


def test_missing_class_in_prototypes_raises():
    lat = np.ones((4, 8))
    y = np.array([0, 0, 1, 1])
    with pytest.raises(RuntimeError):
        class_prototypes(lat, y, 3)


def test_probability_normalization_helper():
    logits = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, -5.0]], dtype=np.float64)
    proba = softmax_from_logits(logits, 1.0)
    assert_probability_rows(proba, atol=1e-8)
    assert proba.shape == (2, 3)


def test_privileged_signal_gate_thresholds():
    def row(acc, f1, fam):
        return {
            "accuracy": acc,
            "macro_f1": f1,
            "family_recall": {
                "oligodendrocyte_opc": fam,
                "astrocyte": fam,
                "vascular": fam,
                "meningeal": fam,
                "other_glial_non_neuronal": fam,
                "difficult_family_mean_recall": fam,
            },
        }

    supported = privileged_signal_decision(row(0.80, 0.70, 0.50), row(0.806, 0.70, 0.50))
    assert supported["supported"] is True
    rejected = privileged_signal_decision(row(0.80, 0.70, 0.50), row(0.801, 0.701, 0.505))
    assert rejected["supported"] is False
    fam = privileged_signal_decision(row(0.80, 0.70, 0.50), row(0.80, 0.70, 0.52))
    assert fam["supported"] is True


def test_student_forward_shapes_and_input_dim():
    torch = pytest.importorskip("torch")
    from v3_e02d_privileged_gene_distillation import build_model_classes

    TeacherMLP, StudentMLP = build_model_classes(torch, torch.nn)
    student = StudentMLP(200)
    x = torch.randn(8, 200)
    logits, rel, lat = student(x)
    assert logits.shape == (8, 60)
    assert rel.shape == (8, 60)
    assert lat.shape == (8, 128)
    teacher = TeacherMLP(500)
    t_logits, t_lat = teacher(torch.randn(4, 500))
    assert t_logits.shape == (4, 60)
    assert t_lat.shape == (4, 128)
    with pytest.raises(RuntimeError):
        student(torch.randn(2, 500))


def test_class_order_matches_official_contract():
    names = allowed_labels(ROOT)
    assert len(names) == 60
    assert names == sorted(names)


@pytest.mark.skipif(not reference_h5ad_path(ROOT).is_file(), reason="approved reference h5ad missing")
def test_reference_md5_shape_genes_and_exclusions():
    data = load_dataset(ROOT)
    class_names = allowed_labels(ROOT)
    audit = audit_reference(data, class_names, root=ROOT)
    assert audit["md5"] == EXPECTED_MD5
    assert audit["raw_n_obs"] == EXPECTED_N_OBS
    assert audit["raw_n_vars"] == EXPECTED_N_VARS
    genes200 = list(data.counts_train.columns)
    genes500 = [str(v) for v in audit["_adata"].var_names.tolist()]
    assert len(genes200) == 200
    assert len(genes500) == 500
    assert set(genes200).issubset(set(genes500))
    assert genes200 == audit["competition_genes_aligned"]
    assert audit["n_train_id_overlaps_removed"] == 5000
    assert audit["n_test_id_overlaps_removed"] == 5000
    assert audit["n_exact_vector_duplicates_removed"] == HISTORICAL_VECTOR_DUPES
    assert audit["n_usable_reference_rows"] == HISTORICAL_USABLE_ROWS
    ref_ids = set(audit["_ext_ids"][audit["_is_ref"]].tolist())
    assert ref_ids.isdisjoint(set(data.counts_train.index.astype(str)))
    assert ref_ids.isdisjoint(set(data.counts_test.index.astype(str)))
    assert not (audit["_is_ref"] & audit["_is_train"]).any()
    assert not (audit["_is_ref"] & audit["_is_test"]).any()


@pytest.mark.skipif(not (OUT / "v3_e02d_complementarity.json").is_file(), reason="V3-E02D artifacts not generated")
def test_artifacts_cell_id_probabilities_and_oracle():
    data = load_dataset(ROOT)
    class_names = allowed_labels(ROOT)
    s2_val = pd.read_csv(OUT / "v3_e02d_s2_validation.csv", dtype={"Cell_ID": str})
    s2_proba = pd.read_csv(OUT / "v3_e02d_s2_validation_probabilities.csv.gz", dtype={"Cell_ID": str})
    s2_test = pd.read_csv(OUT / "v3_e02d_s2_test_probabilities.csv.gz", dtype={"Cell_ID": str})
    registry = pd.read_parquet(OUT / "v3_e00t_team_oof_registry.parquet")
    comp = json.loads((OUT / "v3_e02d_complementarity.json").read_text())
    signal = json.loads((OUT / "v3_e02d_reference_signal_metrics.json").read_text())

    assert s2_val["Cell_ID"].nunique() == 5000
    assert not s2_val["Cell_ID"].duplicated().any()
    assert s2_val["Cell_ID"].str.fullmatch(r"\d{19}").all()
    assert set(s2_val["Cell_ID"]) == set(data.meta_train.index.astype(str))
    aligned = s2_val.set_index("Cell_ID")["true_label"]
    assert list(aligned.reindex(data.meta_train.index.astype(str))) == list(data.meta_train[TARGET_COL].astype(str))
    assert list(s2_proba.columns[1:]) == class_names
    assert_probability_rows(s2_proba.loc[:, class_names].to_numpy(), atol=1e-4)
    assert s2_test.shape[0] == 5000
    assert list(s2_test.columns[1:]) == class_names
    assert_probability_rows(s2_test.loc[:, class_names].to_numpy(), atol=1e-4)
    assert s2_test["Cell_ID"].nunique() == 5000
    assert signal["reference"]["md5"] == EXPECTED_MD5
    assert signal["reference"]["n_usable"] == HISTORICAL_USABLE_ROWS
    assert len(signal["reference"]["gene200_sha256"]) == 64
    assert len(signal["reference"]["gene500_sha256"]) == 64
    assert comp["lzh_wyh_oracle_correct"] == LZH_WYH_ORACLE_CORRECT
    assert comp["shared_error_n"] == SHARED_ERROR_N
    s2_row = comp["students"]["S2"]
    assert s2_row["three_expert_oracle_correct"] == LZH_WYH_ORACLE_CORRECT + s2_row["new_unique_recoveries"]
    assert s2_row["oracle_identity_check"] is True
    teacher = json.loads((OUT / "v3_e02d_teacher_metrics.json").read_text())
    assert teacher["relation_target_shape"] == [HISTORICAL_USABLE_ROWS, 60]
    rec = pd.read_csv(OUT / "v3_e02d_unique_recoveries.csv", dtype={"Cell_ID": str})
    if len(rec):
        assert rec["Cell_ID"].str.fullmatch(r"\d{19}").all()
        assert rec["s2_recovered"].all()
    # V3-E00T numbers must remain untouched.
    e00t = json.loads((OUT / "v3_e00t_metrics.json").read_text())
    assert e00t["multi_expert_oracle"]["overall"]["oracle_correct"] == 4215
    assert set(registry["Cell_ID"].astype(str)) == set(data.meta_train.index.astype(str))


def test_official_prediction_and_frozen_models_untouched():
    dirty = subprocess.check_output(
        ["git", "diff", "--", "prediction/prediction.csv"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert dirty == ""
    v1 = subprocess.check_output(
        ["git", "diff", "--", "outputs/submissions/model_v1.csv", "docs/versions/model_v1.md"],
        cwd=str(ROOT),
    ).decode("utf-8")
    v2 = subprocess.check_output(
        ["git", "diff", "--", "outputs/submissions/model_v2_candidate.csv", "docs/versions/model_v2.md"],
        cwd=str(ROOT),
    ).decode("utf-8")
    e00t = subprocess.check_output(
        ["git", "diff", "--", "outputs/v3/v3_e00t_metrics.json", "outputs/v3/v3_e00t_expert_manifest.json"],
        cwd=str(ROOT),
    ).decode("utf-8")
    assert v1 == ""
    assert v2 == ""
    assert e00t == ""
    assert PRED.is_file()
    data_diff = subprocess.check_output(["git", "diff", "--", "data"], cwd=str(ROOT)).decode("utf-8")
    assert data_diff == ""
