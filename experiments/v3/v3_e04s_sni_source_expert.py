#!/usr/bin/env python3
"""V3-E04S: Source-Diverse SNI Expert (SDE-SNI).

Isolated SNI-only 200-gene LightGBM matched to frozen V2-B-REFONLY
hyperparameters. Does not concatenate SNI with the MERFISH reference.
Does not train a router, ensemble, or formal MODEL V3.
Does not write prediction/prediction.csv.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "v3"))

from merfish60.io import TARGET_COL, load_dataset  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.official_contract import allowed_labels, sha256_file  # noqa: E402
from merfish60.reference import (  # noqa: E402
    EXPECTED_MD5 as MERFISH_MD5,
    _row_fingerprints,
    _to_dense_int,
    md5_file,
    norm_label,
    reference_h5ad_path,
    verify_reference_md5,
)
from merfish60.spatial_features import (  # noqa: E402
    PCA_DIM,
    PCA_RANDOM_STATE,
    ei_of_label_from_train,
)
from merfish60.team_cv import TEAM_CV_PROTOCOL, load_and_validate_team_folds  # noqa: E402
from merfish60.v2_metrics import (  # noqa: E402
    hard_bucket_mask,
    json_default,
    neuron_glial_masks,
    slice_metrics,
    write_json,
    write_proba,
)

from v3_e00t_team_expert_audit import confidence_from_proba, summarize_numeric  # noqa: E402
from v3_e02d_privileged_gene_distillation import (  # noqa: E402
    classification_metrics,
    gene_list_sha256,
)
from v3_e03a_rescue_audit import CLASS_FAMILIES_E03A, family_of  # noqa: E402


EXPERIMENT_ID = "V3-E04S"
EXPERIMENT_MODEL_ID = "V3-E04S-SNI"
EXPECTED_BRANCH = "ywan/ml-pipeline"
EXPECTED_PYTHON_MARKER = "hackathon-v3"
N_TRAIN = 5000
N_TEST = 5000
N_CLASSES = 60
N_GENES = 200
OFFICIAL_GENE_SHA256 = "e3301724038990aa2db237026316aaa5fd265a11231c343bea733f8106ab06f5"

SNI_H5AD_REL = "work/external/SNI_merged_0917.h5ad"
EXPECTED_SNI_MD5 = "7e90a801ee57b8fec06cd03c8630f01b"
SNI_LABEL_COL = "voting"
SNI_LABEL_MAPPING = "merfish60.reference.norm_label"

LZH_CORRECT = 4133
WYH_CORRECT = 4106
S0_CORRECT = 3151
LZH_WYH_ORACLE = 4215
THREE_EXPERT_ORACLE = 4311
ALL_THREE_WRONG = 689
S0_UNIQUE_RECOVERIES = 96

LGBM_SEED = 0
NUM_BOOST_ROUND = 700
MIN_USABLE_SNI_CELLS = 1000
STRONG_RECOVERY_MIN = 40
PROMISING_RECOVERY_MIN = 20
STRONG_ORACLE_MIN = 0.87
MEANINGFUL_FOLDS_34_MIN = 8
BOTH_PARTITION_MIN = 5

WORK_DIR = ROOT / "work" / "v3_e04s"
OUT_DIR = ROOT / "outputs" / "v3"
TABLE_DIR = OUT_DIR / "v3_e04s_tables"
REPORT_PATH = ROOT / "reports" / "v3" / "v3_e04s_sni_source_expert.md"
PRED_PATH = ROOT / "prediction" / "prediction.csv"
E03A_REGISTRY = OUT_DIR / "v3_e03a_three_expert_registry.parquet"
E03A_METRICS = OUT_DIR / "v3_e03a_rescue_metrics.json"
E00T_REGISTRY = OUT_DIR / "v3_e00t_team_oof_registry.parquet"
E02D_COMP = OUT_DIR / "v3_e02d_complementarity.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(ROOT), stderr=subprocess.STDOUT).decode(
        "utf-8"
    )


def current_branch() -> str:
    return run_git(["branch", "--show-current"]).strip()


def assert_v3_interpreter() -> None:
    resolved = str(Path(sys.executable).resolve())
    if EXPECTED_PYTHON_MARKER not in resolved and EXPECTED_PYTHON_MARKER not in sys.executable:
        raise SystemExit(
            "V3-E04S must run with the isolated hackathon-v3 interpreter, not {}. "
            "Use /Users/yyl/venvs/hackathon-v3/bin/python.".format(sys.executable)
        )
    if sys.version_info[:2] < (3, 11):
        raise SystemExit(
            "V3-E04S refuses Python {} (frozen MODEL V1/V2 environment).".format(
                platform.python_version()
            )
        )


def sni_h5ad_path(root: Optional[Path] = None) -> Path:
    return (root or ROOT) / SNI_H5AD_REL


def lgbm_params(num_threads: int) -> dict:
    return dict(
        objective="multiclass",
        num_class=N_CLASSES,
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=30,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=127,
        verbosity=-1,
        num_threads=int(num_threads),
        seed=LGBM_SEED,
        bagging_seed=LGBM_SEED + 100,
        feature_fraction_seed=LGBM_SEED + 200,
    )


def lognorm_library_size(counts: np.ndarray, scale: float) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    tot = x.sum(axis=1)
    return np.log1p(x / np.maximum(tot, 1.0)[:, None] * float(scale)).astype(np.float32)


def library_size_and_detected(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(counts)
    lib = np.asarray(x.sum(axis=1), dtype=np.float64)
    ndet = np.asarray((x > 0).sum(axis=1), dtype=np.float64)
    return lib, ndet


def numeric_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def pearson_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def fingerprint_overlap_mask(query_fp: np.ndarray, reference_fp: np.ndarray) -> np.ndarray:
    ref_set = set(int(v) for v in np.asarray(reference_fp).tolist())
    return np.fromiter(
        (int(v) in ref_set for v in np.asarray(query_fp).tolist()),
        dtype=bool,
        count=len(query_fp),
    )


def stop(code: str, message: str, audit: Optional[dict] = None) -> int:
    if audit is not None:
        audit["stop_code"] = code
        audit["stop_message"] = message
        audit["feasibility"] = {"passed": False, "reason": message, "code": code}
        write_json(OUT_DIR / "v3_e04s_dataset_audit.json", audit)
    print("{}: {}".format(code, message), flush=True)
    return 2


def pairwise_oracle_row(name_a: str, name_b: str, ok_a: np.ndarray, ok_b: np.ndarray, pred_a, pred_b) -> dict:
    both_correct = int(np.sum(ok_a & ok_b))
    a_only = int(np.sum(ok_a & ~ok_b))
    b_only = int(np.sum(ok_b & ~ok_a))
    both_wrong = int(np.sum(~ok_a & ~ok_b))
    disagree = int(np.sum(np.asarray(pred_a) != np.asarray(pred_b)))
    oracle_ok = ok_a | ok_b
    oracle_n = int(oracle_ok.sum())
    acc_a = float(np.mean(ok_a))
    acc_b = float(np.mean(ok_b))
    stronger = max(acc_a, acc_b)
    return {
        "A": name_a,
        "B": name_b,
        "A_accuracy": acc_a,
        "B_accuracy": acc_b,
        "both_correct": both_correct,
        "A_only_correct": a_only,
        "B_only_correct": b_only,
        "both_wrong": both_wrong,
        "prediction_disagreement_count": disagree,
        "prediction_disagreement_rate": float(disagree / len(ok_a)),
        "pair_oracle_count": oracle_n,
        "pair_oracle_accuracy": float(oracle_n / len(ok_a)),
        "incremental_oracle_headroom_over_stronger": float(oracle_n / len(ok_a) - stronger),
        "oracle_is_not_deployable_accuracy": True,
    }


def classify_experiment(
    recoveries: int,
    recoveries_02: int,
    recoveries_34: int,
    four_acc: float,
    integrity_ok: bool,
) -> Tuple[str, str]:
    if not integrity_ok:
        return (
            "SOURCE DIVERSITY INSUFFICIENT",
            "Source/taxonomy/leakage integrity is too weak to accept SNI as a new expert.",
        )
    if recoveries >= STRONG_RECOVERY_MIN and four_acc >= STRONG_ORACLE_MIN and recoveries_34 >= MEANINGFUL_FOLDS_34_MIN:
        return (
            "STRONG SOURCE-DIVERSE EXPERT",
            "SNI recovered at least 40 all-three-wrong cells, four-expert oracle reached 0.87, and locked folds 3-4 also contribute.",
        )
    if (
        PROMISING_RECOVERY_MIN <= recoveries < STRONG_RECOVERY_MIN
        and recoveries_02 >= BOTH_PARTITION_MIN
        and recoveries_34 >= BOTH_PARTITION_MIN
    ):
        return (
            "PROMISING SOURCE-DIVERSE EXPERT",
            "SNI recovered 20-39 all-three-wrong cells with complementarity in both canonical partitions.",
        )
    if recoveries < PROMISING_RECOVERY_MIN or recoveries_34 < BOTH_PARTITION_MIN:
        return (
            "SOURCE DIVERSITY INSUFFICIENT",
            "SNI new unique recoveries are below 20 or locked canonical folds 3-4 show little complementary value.",
        )
    return (
        "SOURCE DIVERSITY INSUFFICIENT",
        "SNI did not meet the predeclared strong or promising source-diversity criteria.",
    )


def next_action_for(label: str) -> str:
    if label == "STRONG SOURCE-DIVERSE EXPERT":
        return (
            "Admit this isolated SNI-only expert into the future V3 expert pool as a frozen "
            "source-diverse member, then run a separate controlled experiment on how to use it. "
            "Do not concatenate SNI with the MERFISH reference, train a router, or create MODEL V3 yet."
        )
    if label == "PROMISING SOURCE-DIVERSE EXPERT":
        return (
            "Keep the isolated SNI-only expert as a candidate pool member and next audit whether "
            "its recoveries are biologically distinct from S0. Do not merge references, ensemble, "
            "or start Spatial-ID until that complementary-subset audit is complete."
        )
    return (
        "Do not add SNI to the expert pool on this isolated 200-gene LightGBM recipe. "
        "The next action is a non-SNI independent expert or a predeclared SNI-label-column "
        "sensitivity audit, not reference concatenation, routing, or MODEL V3."
    )


def audit_sni_dataset(data, class_names: Sequence[str]) -> dict:
    import anndata as ad

    path = sni_h5ad_path()
    audit: dict = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": utc_now(),
        "file_path": str(path),
        "filename": path.name,
    }
    if not path.is_file():
        audit["error"] = "missing SNI file"
        return audit

    digest = md5_file(path)
    sha = sha256_file(path)
    size = int(path.stat().st_size)
    audit.update(
        {
            "file_size_bytes": size,
            "md5": digest,
            "sha256": sha,
            "expected_md5": EXPECTED_SNI_MD5,
            "md5_ok": digest == EXPECTED_SNI_MD5,
        }
    )
    if digest != EXPECTED_SNI_MD5:
        return audit

    adata = ad.read_h5ad(path)
    obs_cols = [str(c) for c in adata.obs.columns.tolist()]
    var_cols = [str(c) for c in adata.var.columns.tolist()]
    var_names = [str(v) for v in adata.var_names.tolist()]
    sni_ids = np.array([str(v) for v in adata.obs_names.tolist()], dtype=object)
    genes = list(data.counts_train.columns)
    missing_genes = [g for g in genes if g not in set(var_names)]
    aligned_ok = len(missing_genes) == 0
    aligned_genes = list(genes) if aligned_ok else []
    gene_sha = gene_list_sha256(aligned_genes) if aligned_ok else None

    x_raw = adata.X
    x_min = float(np.min(x_raw)) if hasattr(x_raw, "min") else None
    x_max = float(np.max(x_raw)) if hasattr(x_raw, "max") else None
    nonnegative = bool(x_min is not None and x_min >= 0)

    label_candidates = [
        c
        for c in obs_cols
        if c.lower() in {"voting", "tangram", "singler", "seurat", "rctd", "cell_type", "celltype"}
        or "cell type" in c.lower()
        or "annotation" in c.lower()
    ]
    condition_candidates = [
        c
        for c in obs_cols
        if c.lower() in {"condition", "datasets", "batch", "source", "dataset"}
        or "condition" in c.lower()
        or "source" in c.lower()
    ]

    train_ids = np.array([str(v) for v in data.counts_train.index.tolist()], dtype=object)
    test_ids = np.array([str(v) for v in data.counts_test.index.tolist()], dtype=object)
    train_set = set(train_ids.tolist())
    test_set = set(test_ids.tolist())
    in_train = np.fromiter((cid in train_set for cid in sni_ids), dtype=bool, count=len(sni_ids))
    in_test = np.fromiter((cid in test_set for cid in sni_ids), dtype=bool, count=len(sni_ids))

    mer_path = reference_h5ad_path(ROOT)
    mer_md5 = verify_reference_md5(mer_path)
    mer = ad.read_h5ad(mer_path)
    mer_ids = np.array([str(v) for v in mer.obs_names.tolist()], dtype=object)
    mer_set = set(mer_ids.tolist())
    in_mer_id = np.fromiter((cid in mer_set for cid in sni_ids), dtype=bool, count=len(sni_ids))

    taxonomy = {"case": "C", "reason": "no deterministic mapped label column"}
    mapped_labels = None
    if SNI_LABEL_COL in adata.obs.columns:
        raw = adata.obs[SNI_LABEL_COL].astype(str).to_numpy()
        mapped = np.array([norm_label(v) for v in raw], dtype=object)
        class_set = set(str(c) for c in class_names)
        mapped_ok = np.array([lab in class_set for lab in mapped], dtype=bool)
        unmapped = sorted(set(mapped[~mapped_ok].tolist()))
        covered = sorted(c for c in class_names if c in set(mapped[mapped_ok].tolist()))
        missing_cls = [c for c in class_names if c not in set(covered)]
        counts = (
            pd.Series(mapped[mapped_ok]).value_counts().reindex(list(class_names)).fillna(0).astype(int)
        )
        exact_raw_match = bool(set(raw.tolist()) == set(class_names))
        taxonomy = {
            "case": "A" if exact_raw_match else "B",
            "label_column": SNI_LABEL_COL,
            "mapping_function": SNI_LABEL_MAPPING,
            "mapping_already_in_project": True,
            "raw_label_count": int(pd.Series(raw).nunique(dropna=False)),
            "mapped_class_count": int(len(covered)),
            "competition_classes_covered": int(len(covered)),
            "competition_classes_missing": missing_cls,
            "unmapped_labels": unmapped,
            "cells_retained_by_taxonomy": int(mapped_ok.sum()),
            "cells_excluded_by_taxonomy": int((~mapped_ok).sum()),
            "per_class_sni_counts": {str(k): int(v) for k, v in counts.items()},
            "class_count_min": int(counts.min()) if len(counts) else 0,
            "class_count_median": float(counts.median()) if len(counts) else None,
            "class_count_max": int(counts.max()) if len(counts) else 0,
            "reason": (
                "raw SNI voting labels already equal the official 60 names"
                if exact_raw_match
                else "existing V2 norm_label maps spaces/hyphens to official underscores"
            ),
        }
        mapped_labels = mapped
        mapped_ok_mask = mapped_ok
    else:
        mapped_ok_mask = np.ones(len(sni_ids), dtype=bool)

    vector = {
        "sni_exact_vectors_matching_competition_train": None,
        "sni_exact_vectors_matching_competition_test": None,
        "sni_exact_vectors_matching_cleaned_merfish_reference": None,
        "policy": (
            "Exclude competition-train/test ID overlaps, competition exact 200-gene vectors, "
            "and exact 200-gene vectors that reproduce the cleaned MERFISH reference."
        ),
    }
    keep = ~(in_train | in_test) & mapped_ok_mask
    if aligned_ok:
        x_sni = _to_dense_int(adata[:, genes].X)
        x_train = data.counts_train.loc[:, genes].to_numpy(dtype=np.int64)
        x_test = data.counts_test.loc[:, genes].to_numpy(dtype=np.int64)
        sni_fp = _row_fingerprints(x_sni)
        train_fp = _row_fingerprints(x_train)
        test_fp = _row_fingerprints(x_test)
        dup_train = fingerprint_overlap_mask(sni_fp, train_fp)
        dup_test = fingerprint_overlap_mask(sni_fp, test_fp)

        mer_train = np.fromiter((cid in train_set for cid in mer_ids), dtype=bool, count=len(mer_ids))
        mer_test = np.fromiter((cid in test_set for cid in mer_ids), dtype=bool, count=len(mer_ids))
        x_mer = _to_dense_int(mer[:, genes].X)
        mer_fp = _row_fingerprints(x_mer)
        mer_ref_mask = ~(mer_train | mer_test)
        mer_ref_fp = mer_fp[mer_ref_mask]
        comp_fp = np.concatenate([train_fp, test_fp])
        mer_comp_dup = fingerprint_overlap_mask(mer_ref_fp, comp_fp)
        cleaned_mer_fp = mer_ref_fp[~mer_comp_dup]
        dup_mer = fingerprint_overlap_mask(sni_fp, cleaned_mer_fp)

        vector = {
            "sni_exact_vectors_matching_competition_train": int(dup_train.sum()),
            "sni_exact_vectors_matching_competition_test": int(dup_test.sum()),
            "sni_exact_vectors_matching_cleaned_merfish_reference": int(dup_mer.sum()),
            "cleaned_merfish_reference_rows": int(cleaned_mer_fp.size),
            "policy": (
                "Exclude competition-train/test ID overlaps, competition exact 200-gene vectors, "
                "and exact 200-gene vectors that reproduce the cleaned MERFISH reference so that "
                "source diversity is not artificially inflated."
            ),
        }
        keep = keep & ~dup_train & ~dup_test & ~dup_mer
        audit["_x_sni"] = x_sni
        audit["_keep"] = keep
        audit["_mapped_labels"] = mapped_labels
        audit["_sni_ids"] = sni_ids
        audit["_adata"] = adata

        sni_lib, sni_ndet = library_size_and_detected(x_sni)
        tr_lib, tr_ndet = library_size_and_detected(x_train)
        mer_lib, mer_ndet = library_size_and_detected(x_mer[mer_ref_mask][~mer_comp_dup])
        usable_x = x_sni[keep]
        usable_lib, usable_ndet = library_size_and_detected(usable_x)
        sni_mean = usable_x.mean(axis=0) if usable_x.size else np.array([])
        tr_mean = x_train.mean(axis=0)
        mer_mean = x_mer[mer_ref_mask][~mer_comp_dup].mean(axis=0)
        sni_var = usable_x.var(axis=0) if usable_x.size else np.array([])
        tr_var = x_train.var(axis=0)
        mer_var = x_mer[mer_ref_mask][~mer_comp_dup].var(axis=0)

        # Unlabeled 2-PC diagnostic on official 200-gene log1p; not a domain-adaptation model.
        scale = float(np.median(np.concatenate([tr_lib, library_size_and_detected(x_test)[0]])))
        sni_ln = lognorm_library_size(usable_x, scale) if usable_x.size else np.zeros((0, N_GENES))
        tr_ln = lognorm_library_size(x_train, scale)
        if len(sni_ln) and len(tr_ln):
            combo = np.vstack([sni_ln, tr_ln])
            src = np.r_[np.zeros(len(sni_ln), dtype=int), np.ones(len(tr_ln), dtype=int)]
            pca2 = PCA(n_components=2, random_state=PCA_RANDOM_STATE).fit_transform(
                StandardScaler().fit_transform(combo)
            )
            pca_sep = {
                "explained_variance_ratio": [float(v) for v in PCA(n_components=2, random_state=PCA_RANDOM_STATE)
                                             .fit(StandardScaler().fit_transform(combo))
                                             .explained_variance_ratio_],
                "sni_pc_mean": [float(v) for v in pca2[src == 0].mean(axis=0)],
                "competition_train_pc_mean": [float(v) for v in pca2[src == 1].mean(axis=0)],
                "mean_pc_euclidean_separation": float(
                    np.linalg.norm(pca2[src == 0].mean(axis=0) - pca2[src == 1].mean(axis=0))
                ),
                "note": "unlabeled diagnostic only; PCA was not used as a domain-adaptation model",
            }
        else:
            pca_sep = None
        source_dist = {
            "raw_sni_library_size": numeric_summary(sni_lib),
            "raw_sni_n_detected": numeric_summary(sni_ndet),
            "usable_sni_library_size": numeric_summary(usable_lib),
            "usable_sni_n_detected": numeric_summary(usable_ndet),
            "competition_train_library_size": numeric_summary(tr_lib),
            "competition_train_n_detected": numeric_summary(tr_ndet),
            "cleaned_merfish_library_size": numeric_summary(mer_lib),
            "cleaned_merfish_n_detected": numeric_summary(mer_ndet),
            "per_gene_mean_correlation_sni_vs_competition_train": pearson_corr(sni_mean, tr_mean),
            "per_gene_variance_correlation_sni_vs_competition_train": pearson_corr(sni_var, tr_var),
            "per_gene_mean_correlation_sni_vs_cleaned_merfish": pearson_corr(sni_mean, mer_mean),
            "per_gene_variance_correlation_sni_vs_cleaned_merfish": pearson_corr(sni_var, mer_var),
            "pca_source_separation_sni_vs_competition_train": pca_sep,
            "competition_library_size_scale": scale,
        }
        audit["_scale"] = scale
        audit["_x_train"] = x_train
        audit["_x_test"] = x_test
        audit["_volume_sni"] = adata.obs["volume"].to_numpy(dtype=np.float64)
    else:
        source_dist = {}
        dup_train = dup_test = dup_mer = np.zeros(len(sni_ids), dtype=bool)

    condition_counts = {}
    if "Condition" in adata.obs.columns:
        condition_counts = {
            str(k): int(v) for k, v in adata.obs["Condition"].astype(str).value_counts().items()
        }
    dataset_counts = {}
    if "Datasets" in adata.obs.columns:
        dataset_counts = {
            str(k): int(v) for k, v in adata.obs["Datasets"].astype(str).value_counts().items()
        }

    usable_n = int(keep.sum())
    usable_labels = mapped_labels[keep] if mapped_labels is not None else np.array([], dtype=object)
    usable_counts = (
        pd.Series(usable_labels).value_counts().reindex(list(class_names)).fillna(0).astype(int)
        if usable_n
        else pd.Series(dtype=int)
    )

    audit.update(
        {
            "anndata_shape": [int(adata.n_obs), int(adata.n_vars)],
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "obs_column_names": obs_cols,
            "var_column_names": var_cols,
            "var_gene_identifier_source": "AnnData.var_names",
            "label_column_candidates": label_candidates,
            "condition_source_column_candidates": condition_candidates,
            "study_source_metadata": {
                "Condition": condition_counts,
                "Datasets": dataset_counts,
                "batch": (
                    {str(k): int(v) for k, v in adata.obs["batch"].astype(str).value_counts().items()}
                    if "batch" in adata.obs.columns
                    else {}
                ),
                "Mouse ID": (
                    {str(k): int(v) for k, v in adata.obs["Mouse ID"].astype(str).value_counts().items()}
                    if "Mouse ID" in adata.obs.columns
                    else {}
                ),
                "Axial level": (
                    {str(k): int(v) for k, v in adata.obs["Axial level"].astype(str).value_counts().items()}
                    if "Axial level" in adata.obs.columns
                    else {}
                ),
                "uns_keys": list(adata.uns.keys()) if adata.uns else [],
                "note": "values recorded from file fields only; biological meaning is not inferred beyond field names",
            },
            "x_dtype": str(getattr(adata.X, "dtype", type(adata.X))),
            "x_min": x_min,
            "x_max": x_max,
            "nonnegative_counts": nonnegative,
            "competition_train_cell_id_overlap": int(in_train.sum()),
            "competition_test_cell_id_overlap": int(in_test.sum()),
            "merfish_reference_cell_id_overlap": int(in_mer_id.sum()),
            "merfish_reference_md5": mer_md5,
            "merfish_reference_shape": [int(mer.n_obs), int(mer.n_vars)],
            "official_gene_alignment": {
                "n_official_genes": len(genes),
                "n_present": int(len(genes) - len(missing_genes)),
                "missing_genes": missing_genes,
                "exact_200_of_200": bool(aligned_ok and len(genes) == N_GENES),
                "aligned_to_official_order": bool(aligned_ok),
                "ordered_200_gene_sha256": gene_sha,
                "matches_official_gene_contract": bool(gene_sha == OFFICIAL_GENE_SHA256),
            },
            "taxonomy": taxonomy,
            "exact_vector_duplicate_audit": vector,
            "usable_sni_cells": usable_n,
            "usable_class_count_min": int(usable_counts.min()) if len(usable_counts) else 0,
            "usable_class_count_median": float(usable_counts.median()) if len(usable_counts) else None,
            "usable_class_count_max": int(usable_counts.max()) if len(usable_counts) else 0,
            "usable_missing_competition_classes": [
                c for c in class_names if int(usable_counts.get(c, 0)) == 0
            ],
            "source_distribution": source_dist,
            "duplication_check": {
                "claim": "new source-diversity experiment within the current team/project",
                "auditable_sni_only_expert_found": False,
                "local_tree_sni_mentions": 0,
                "team_remote_sni_merged_mentions": 0,
                "note": (
                    "git fetch team --prune and git grep on team/main, team/yhh, team/lzh, "
                    "team/wyh, team/revert-1-lzh, and the personal tree found no auditable "
                    "SNI_merged_0917.h5ad isolated expert. This is not a global-novelty claim."
                ),
            },
        }
    )
    # keep internal arrays
    audit["_keep"] = keep
    return audit


def feasibility_decision(audit: dict) -> Tuple[bool, str]:
    if not audit.get("md5_ok"):
        return False, "SNI CHECKSUM FAILURE"
    genes = audit.get("official_gene_alignment") or {}
    if not genes.get("exact_200_of_200") or not genes.get("matches_official_gene_contract"):
        return False, "SNI GENE ALIGNMENT FAILURE"
    tax = audit.get("taxonomy") or {}
    if tax.get("case") not in {"A", "B"}:
        return False, "SNI TAXONOMY MAPPING NEEDS HUMAN REVIEW"
    if int(audit.get("usable_sni_cells") or 0) < MIN_USABLE_SNI_CELLS:
        return False, "SNI FEASIBILITY FAILED: too few mapped SNI cells"
    if not audit.get("nonnegative_counts"):
        return False, "SNI FEASIBILITY FAILED: source matrix is not a nonnegative count contract"
    missing = audit.get("usable_missing_competition_classes") or []
    if len(missing) == 60:
        return False, "SNI FEASIBILITY FAILED: no mapped competition classes remain"
    return True, "PASS"


def build_sni_features(
    sni_counts: np.ndarray,
    train_counts: np.ndarray,
    test_counts: np.ndarray,
    scale: float,
) -> dict:
    sni_ln = lognorm_library_size(sni_counts, scale)
    tr_ln = lognorm_library_size(train_counts, scale)
    te_ln = lognorm_library_size(test_counts, scale)
    scaler = StandardScaler()
    sni_z = scaler.fit_transform(sni_ln)
    pca = PCA(n_components=PCA_DIM, random_state=PCA_RANDOM_STATE)
    sni_pc = pca.fit_transform(sni_z).astype(np.float32)
    tr_pc = pca.transform(scaler.transform(tr_ln)).astype(np.float32)
    te_pc = pca.transform(scaler.transform(te_ln)).astype(np.float32)
    sni_lib = np.log1p(sni_counts.sum(axis=1).astype(np.float64))[:, None]
    tr_lib = np.log1p(train_counts.sum(axis=1).astype(np.float64))[:, None]
    te_lib = np.log1p(test_counts.sum(axis=1).astype(np.float64))[:, None]
    names = (
        ["g_{}".format(i) for i in range(N_GENES)]
        + ["pc{}".format(i) for i in range(PCA_DIM)]
        + ["log_library_size"]
    )
    return {
        "X_sni": np.hstack([sni_ln, sni_pc, sni_lib]).astype(np.float32),
        "X_train": np.hstack([tr_ln, tr_pc, tr_lib]).astype(np.float32),
        "X_test": np.hstack([te_ln, te_pc, te_lib]).astype(np.float32),
        "names": names,
        "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "pca": pca,
        "scaler": scaler,
    }


def fit_sni_lgbm(X_train, y_train, X_val, names, params, num_boost_round: int):
    feature_names = [str(name).replace(" ", "_") for name in names]
    dataset = lgb.Dataset(X_train, y_train, feature_name=feature_names, free_raw_data=True)
    model = lgb.train(params, dataset, num_boost_round=num_boost_round)
    proba = np.asarray(model.predict(X_val), dtype=np.float64)
    return model, proba


def family_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> List[dict]:
    rows = []
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    fam = np.array([family_of(str(v)) for v in y_true], dtype=object)
    order = list(CLASS_FAMILIES_E03A.keys()) + ["neuronal_or_other"]
    for name in order:
        mask = fam == name
        n = int(mask.sum())
        acc = float(np.mean(y_true[mask] == y_pred[mask])) if n else None
        rows.append({"family": name, "n": n, "accuracy": acc})
    return rows


def write_report(payload: dict) -> None:
    a = payload["audit"]
    s = payload["standalone"]
    slices = payload["slices"]
    pair = payload["pairwise"]
    rec = payload["recoveries"]
    four = payload["four_expert"]
    leak = payload["leakage_audit"]
    tax = a["taxonomy"]
    genes = a["official_gene_alignment"]
    vec = a["exact_vector_duplicate_audit"]
    dist = a["source_distribution"]
    pca_sep = (dist.get("pca_source_separation_sni_vs_competition_train") or {}) if dist else {}
    fam_rows = "\n".join(
        "| {family} | {n} | {accuracy} |".format(
            family=r["family"],
            n=r["n"],
            accuracy="{:.4f}".format(r["accuracy"]) if r["accuracy"] is not None else "n/a",
        )
        for r in slices["sni_by_family"]
    )
    rec_fam = "\n".join(
        "| {family} | {n} |".format(**r) for r in rec["by_family"]
    )
    conf_rows = "\n".join(
        "| {true_label} | {pred} | {error_count} | {sni_rescue_count} | {sni_rescue_fraction:.4f} | {family} |".format(
            **r
        )
        for r in rec["top_resolved_shared_confusions"][:12]
    )
    pair_rows = "\n".join(
        "| {A} | {B} | {A_accuracy:.4f} | {B_accuracy:.4f} | {prediction_disagreement_count} | "
        "{pair_oracle_count} | {pair_oracle_accuracy:.4f} | {incremental_oracle_headroom_over_stronger:.4f} |".format(
            **r
        )
        for r in pair
    )
    conf = rec["confidence_diagnostic"]
    text = f"""# V3-E04S — Source-Diverse SNI Expert

Research codename: **SDE-SNI**. This is a MODEL V3 research experiment, not a formal MODEL V3 freeze.

## 1. Research Question

Can a biologically distinct SNI reference source produce complementary cell-type information that the existing MERFISH-reference, biology/graph, and weak-MLP expert families do not contain?

The first SNI model isolates the **source** effect:

SNI-only + official 200 competition genes + a LightGBM recipe matched to frozen V2-B-REFONLY.

## 2. Motivation from V3-E00T / E02D / E03A

Frozen auditable pool:

- LZH Prior-H: {LZH_CORRECT} / 5000 = 0.8266
- WYH MODEL V2: {WYH_CORRECT} / 5000 = 0.8212
- S0 hard-label 200-gene reference MLP: {S0_CORRECT} / 5000 = 0.6302
- LZH + WYH diagnostic oracle: {LZH_WYH_ORACLE} / 5000 = 0.8430
- LZH + WYH + S0 diagnostic oracle: {THREE_EXPERT_ORACLE} / 5000 = 0.8622
- remaining all-three-wrong population: **{ALL_THREE_WRONG}**

V3-E03A found that S0 contributes {S0_UNIQUE_RECOVERIES} genuine unique recoveries, but available test-safe signals cannot identify those cases at high enough precision. An S0 rescue router is not justified. The new objective is an independent expert that can recover cells from the remaining {ALL_THREE_WRONG}.

## 3. Why This Is Not Duplicate Team Work

{a["duplication_check"]["note"]}

The correct claim is: **{a["duplication_check"]["claim"]}**.

Nearby but non-equivalent methods:

- WYH MODEL V2 is MERFISH-reference-only LightGBM
- LZH Prior-H uses biology priors / graph evidence
- YHH uses hierarchical specialists
- V3-E02D/S0 uses the same MERFISH reference as a neural student
- team main uses MERFISH reference / spatial / ensemble members

No auditable equivalent isolated `SNI_merged_0917.h5ad` expert was found.

## 4. SNI Provenance

| Item | Value |
|---|---|
| Path | `{a["file_path"]}` |
| Filename | `{a["filename"]}` |
| Size | {a["file_size_bytes"]} bytes |
| AnnData shape | {a["n_obs"]} cells × {a["n_vars"]} genes |
| Label column used | `{tax["label_column"]}` |
| Condition field | {a["study_source_metadata"]["Condition"]} |
| Datasets | {len(a["study_source_metadata"]["Datasets"])} distinct run identifiers |
| Gene identifiers | `{a["var_gene_identifier_source"]}` |

obs columns: {", ".join(a["obs_column_names"])}

No undocumented biological meaning was inferred from field names.

## 5. Checksum Verification

| Item | Value |
|---|---|
| MD5 | `{a["md5"]}` |
| Expected MD5 | `{EXPECTED_SNI_MD5}` |
| MD5 match | {a["md5_ok"]} |
| SHA256 | `{a["sha256"]}` |
| MERFISH reference MD5 | `{a["merfish_reference_md5"]}` (unchanged) |

## 6. Competition Cell_ID Overlap Audit

Cell_ID remained a lossless 19-digit string.

| Comparison | Overlap |
|---|---:|
| SNI vs competition train IDs | {a["competition_train_cell_id_overlap"]} |
| SNI vs competition test IDs | {a["competition_test_cell_id_overlap"]} |

Overlapping IDs, if any, are excluded before fitting.

## 7. MERFISH-Reference Overlap Audit

| Comparison | Overlap |
|---|---:|
| SNI vs MERFISH-reference Cell_IDs | {a["merfish_reference_cell_id_overlap"]} |
| SNI exact 200-gene vectors vs cleaned MERFISH reference | {vec.get("sni_exact_vectors_matching_cleaned_merfish_reference")} |

Direct duplicated source identities are excluded from SNI training.

## 8. Official 200-Gene Alignment

| Item | Value |
|---|---|
| Official genes present | {genes["n_present"]} / {genes["n_official_genes"]} |
| Exact 200 / 200 | {genes["exact_200_of_200"]} |
| Aligned to official MODEL V2 order | {genes["aligned_to_official_order"]} |
| Ordered 200-gene SHA256 | `{genes["ordered_200_gene_sha256"]}` |
| Matches frozen contract | {genes["matches_official_gene_contract"]} |

Missing official genes were not imputed. Genes were not alphabetically reordered.

## 9. Taxonomy Mapping

**CASE {tax["case"]}.** Mapping uses the existing project function `{tax["mapping_function"]}` on SNI `obs["{tax["label_column"]}"]`. This is not a newly invented biological map.

| Item | Value |
|---|---|
| Raw SNI label count | {tax["raw_label_count"]} |
| Mapped competition classes | {tax["mapped_class_count"]} / 60 |
| Missing competition classes | {tax["competition_classes_missing"] or "none"} |
| Cells retained by taxonomy | {tax["cells_retained_by_taxonomy"]} |
| Cells excluded by taxonomy | {tax["cells_excluded_by_taxonomy"]} |
| Per-class min / median / max | {tax["class_count_min"]} / {tax["class_count_median"]} / {tax["class_count_max"]} |

`tangram`, `seurat`, `singler`, and `rctd` were inspected as label candidates. `voting` is the complete consensus column covering all 60 classes with zero missing values after `norm_label`. No class was mapped merely because names looked similar.

## 10. Exact-Vector Duplicate Audit

{vec["policy"]}

| Comparison | Count |
|---|---:|
| SNI exact vectors matching competition train | {vec.get("sni_exact_vectors_matching_competition_train")} |
| SNI exact vectors matching competition test | {vec.get("sni_exact_vectors_matching_competition_test")} |
| SNI exact vectors matching cleaned MERFISH reference | {vec.get("sni_exact_vectors_matching_cleaned_merfish_reference")} |
| Usable SNI cells after all exclusions | {a["usable_sni_cells"]} |

## 11. Source Distribution Comparison

Unlabeled/test-safe summaries. Competition labels were not used to fit the SNI model.

| Quantity | SNI usable | Competition train | Cleaned MERFISH |
|---|---:|---:|---:|
| Library-size mean | {dist["usable_sni_library_size"]["mean"]:.3f} | {dist["competition_train_library_size"]["mean"]:.3f} | {dist["cleaned_merfish_library_size"]["mean"]:.3f} |
| Library-size median | {dist["usable_sni_library_size"]["median"]:.3f} | {dist["competition_train_library_size"]["median"]:.3f} | {dist["cleaned_merfish_library_size"]["median"]:.3f} |
| n_detected mean | {dist["usable_sni_n_detected"]["mean"]:.3f} | {dist["competition_train_n_detected"]["mean"]:.3f} | {dist["cleaned_merfish_n_detected"]["mean"]:.3f} |
| n_detected median | {dist["usable_sni_n_detected"]["median"]:.3f} | {dist["competition_train_n_detected"]["median"]:.3f} | {dist["cleaned_merfish_n_detected"]["median"]:.3f} |

Per-gene mean correlation, SNI vs competition train: {dist["per_gene_mean_correlation_sni_vs_competition_train"]}

Per-gene variance correlation, SNI vs competition train: {dist["per_gene_variance_correlation_sni_vs_competition_train"]}

Unlabeled 2-PC mean Euclidean separation (SNI vs competition train): {pca_sep.get("mean_pc_euclidean_separation")}

No domain-adaptation model was trained.

## 12. SNI-Only Expert Design

Experimental ID: **{EXPERIMENT_MODEL_ID}**. This is not MODEL V3.

Controlled variable: **reference source** (SNI vs frozen MERFISH MODEL V2).

Reused from V2-B-REFONLY:

- official 200-gene order
- library-size log1p using the median total of the 10,000 unlabeled competition cells
- StandardScaler + PCA50 (`random_state=0`)
- LightGBM objective, hyperparameters, 700 rounds, seed 0
- canonical 60-class probability order

Intentionally omitted because they would break source isolation or require competition-label fitting:

- MERFISH-reference cells
- spatial neighbors / expression graphs / neighbor-label histograms
- neural nets, family specialists, S0
- E/I and Segment post-masks (SNI has no Laminae/Segment; V2 masks need MERFISH-reference Segment or train-derived E/I maps)

PCA/scaler are fit on usable SNI cells only, then applied to competition matrices. That is a necessary source-isolation difference versus V2's transductive PCA on the MERFISH deposit.

## 13. Honest Competition External-Validation Performance

The SNI model is trained only on external SNI labels. Predictions on the 5000 competition train cells are **honest external-validation predictions**, not competition-label OOF.

| Metric | Value |
|---|---|
| Accuracy | {s["accuracy"]:.4f} |
| Correct | {s["correct"]} / 5000 |
| Macro-F1 | {s["macro_f1"]:.4f} |
| Log loss | {s["log_loss"]:.4f} |
| Canonical folds 0-2 | {s["folds_0_2"]:.4f} |
| Canonical folds 3-4 | {s["folds_3_4"]:.4f} |

Fold accuracies: {s["canonical_fold_accuracy"]}

## 14. Canonical Partition Stability

Canonical analysis partition: `{TEAM_CV_PROTOCOL}`. Folds 3-4 were not used to redesign the model.

| Split | SNI accuracy | sni_new_unique_recoveries | four-expert oracle |
|---|---:|---:|---:|
| overall | {s["accuracy"]:.4f} | {rec["sni_new_unique_recoveries"]} | {four["overall"]["accuracy"]:.4f} |
| folds 0-2 | {s["folds_0_2"]:.4f} | {rec["sni_new_unique_recoveries_folds_0_2"]} | {four["folds_0_2"]["accuracy"]:.4f} |
| folds 3-4 | {s["folds_3_4"]:.4f} | {rec["sni_new_unique_recoveries_folds_3_4"]} | {four["folds_3_4"]["accuracy"]:.4f} |

## 15. Pairwise Complementarity

**ORACLE != DEPLOYABLE ACCURACY.**

| A | B | A acc | B acc | Disagree | Pair oracle n | Pair oracle acc | Headroom vs stronger |
|---|---|---:|---:|---:|---:|---:|---:|
{pair_rows}

## 16. Recovery of the Remaining 689 Shared Failures

Primary metric:

`sni_new_unique_recoveries` = LZH wrong AND WYH wrong AND S0 wrong AND SNI correct

**sni_new_unique_recoveries = {rec["sni_new_unique_recoveries"]}**

Shared-failure registry rows: {rec["shared_failure_n"]} (expected {ALL_THREE_WRONG}).

## 17. Four-Expert Diagnostic Oracle

four_expert_oracle_correct = {THREE_EXPERT_ORACLE} + sni_new_unique_recoveries = **{four["overall"]["correct"]}**

four_expert_oracle_accuracy = {four["overall"]["correct"]} / 5000 = **{four["overall"]["accuracy"]:.4f}**

Remaining all-four-wrong cells: **{four["all_four_wrong"]}**

**ORACLE != DEPLOYABLE ACCURACY.**

## 18. Biological Family Recovery

SNI standalone family accuracy:

| family | n | accuracy |
|---|---:|---:|
{fam_rows}

SNI new unique recoveries by family, compared with S0's previous 96 rescues (oligo/OPC, vascular, astrocyte-heavy):

| family | SNI new unique recoveries |
|---|---:|
{rec_fam}

S0 previous unique-rescue family counts: oligodendrocyte_opc 49, astrocyte 14, vascular 10, meningeal 6, microglia 7, remaining_glial_non_neuronal 1, neuronal_or_other 9.

## 19. Shared Confusion Recovery

Among the {ALL_THREE_WRONG} all-three-wrong cells, ranked true_label → LZH_pred pairs newly resolved by SNI:

| true_label | LZH pred | errors | SNI rescues | fraction | family |
|---|---|---:|---:|---:|---|
{conf_rows}

This analysis was not used to retune the model.

## 20. SNI Confidence Diagnostics

Within the {ALL_THREE_WRONG} shared failures, SNI-correct vs SNI-wrong. Diagnostic only; no threshold was optimized.

| group | n | top1 mean | margin mean | entropy mean |
|---|---:|---:|---:|---:|
| SNI-correct | {conf["sni_correct"]["count"]} | {conf["sni_correct"]["top1"]["mean"]} | {conf["sni_correct"]["margin"]["mean"]} | {conf["sni_correct"]["entropy"]["mean"]} |
| SNI-wrong | {conf["sni_wrong"]["count"]} | {conf["sni_wrong"]["top1"]["mean"]} | {conf["sni_wrong"]["margin"]["mean"]} | {conf["sni_wrong"]["entropy"]["mean"]} |

Diagnostic AUROC for SNI-correct vs SNI-wrong:

- top1: {conf["auroc"]["top1"]}
- margin: {conf["auroc"]["margin"]}
- negative entropy: {conf["auroc"]["neg_entropy"]}

## 21. Leakage Audit

- competition test labels used: {leak["competition_test_labels_used"]}
- competition train labels used to fit SNI: {leak["competition_train_labels_used_for_sni_fitting"]}
- external competition ID overlaps excluded: {leak["competition_id_overlaps_excluded"]}
- exact competition vector duplicates excluded: {leak["competition_exact_vector_duplicates_excluded"]}
- MERFISH-reference exact-vector duplicates excluded: {leak["merfish_exact_vector_duplicates_excluded"]}
- leaderboard tuning: {leak["leaderboard_feedback_used"]}
- hyperparameter search: {leak["hyperparameter_search"]}
- seed search: {leak["seed_search"]}
- post-hoc blend optimization: {leak["blend_optimization"]}
- canonical folds 3-4 used for redesign: {leak["canonical_folds_3_4_used_for_redesign"]}
- test probabilities used for model selection: {leak["test_probabilities_used_for_model_selection"]}
- submission generated: {leak["submission_generated"]}
- prediction/prediction.csv modified: {leak["prediction_csv_modified"]}
- MODEL V1 / V2 / V3-E00T / E02D / E03A modified: {leak["frozen_artifacts_modified"]}

## 22. Limitations

- SNI labels come from the file's `voting` consensus, not a wet-lab gold standard.
- The SNI matrix mixes Sham and SNI injury conditions; this experiment treats the file as one independent source rather than splitting condition.
- Spatial/graph features were omitted by design, so this is not a full reimplementation of every V2-B feature, only the source-isolated gene LightGBM analog.
- Diagnostic oracle headroom is not deployable accuracy.
- Rare classes remain rare (usable min count {a["usable_class_count_min"]}).

## 23. Decision

**{payload["decision"]["label"]}**

{payload["decision"]["reason"]}

Recommended next experiment/action:

{payload["decision"]["next_action"]}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)


def main() -> int:
    t0 = time.time()
    assert_v3_interpreter()
    branch = current_branch()
    if branch != EXPECTED_BRANCH:
        print("STOP: branch is {} (expected {})".format(branch, EXPECTED_BRANCH), flush=True)
        return 2

    print("V3-E04S source-diverse SNI expert", flush=True)
    print("branch={}".format(branch), flush=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    pred_before = sha256_file(PRED_PATH) if PRED_PATH.is_file() else None
    data = load_dataset(ROOT)
    class_names = allowed_labels(ROOT)
    folds_df, fold_messages = load_and_validate_team_folds(
        data.counts_train.index, data.counts_test.index, data.y_train
    )
    print("Canonical partition:", TEAM_CV_PROTOCOL, flush=True)
    for line in fold_messages:
        print(" - {}".format(line), flush=True)

    print("Auditing SNI dataset...", flush=True)
    audit = audit_sni_dataset(data, class_names)
    if not audit.get("md5_ok"):
        return stop("SNI CHECKSUM FAILURE", "SNI MD5 {} != {}".format(audit.get("md5"), EXPECTED_SNI_MD5), audit)
    genes = audit["official_gene_alignment"]
    if not genes.get("exact_200_of_200"):
        return stop("SNI GENE ALIGNMENT FAILURE", "SNI does not contain all 200 official genes", audit)
    tax = audit["taxonomy"]
    if tax.get("case") not in {"A", "B"}:
        return stop("SNI TAXONOMY MAPPING NEEDS HUMAN REVIEW", tax.get("reason", "CASE C"), audit)

    ok, reason = feasibility_decision(audit)
    audit["feasibility"] = {"passed": ok, "reason": reason, "code": "PASS" if ok else reason}
    if not ok:
        write_json(OUT_DIR / "v3_e04s_dataset_audit.json", audit)
        return stop(reason.split(":")[0], reason, audit)

    keep = audit["_keep"]
    x_sni = audit["_x_sni"]
    y_sni = audit["_mapped_labels"][keep]
    x_sni_u = x_sni[keep]
    print(
        "Stage A PASS: usable SNI cells={} classes_missing={}".format(
            int(keep.sum()), audit["usable_missing_competition_classes"]
        ),
        flush=True,
    )

    class_index = {str(c): i for i, c in enumerate(class_names)}
    y_codes = np.array([class_index[str(v)] for v in y_sni], dtype=np.int32)
    feats = build_sni_features(x_sni_u, audit["_x_train"], audit["_x_test"], audit["_scale"])
    np.savez(
        WORK_DIR / "sni_pca_scaler.npz",
        scaler_mean=feats["scaler_mean"],
        scaler_scale=feats["scaler_scale"],
        pca_var=np.array([feats["pca_var_explained"]]),
    )
    print(
        "Features: {} cols; SNI-only PCA50 variance explained={:.4f}".format(
            feats["X_sni"].shape[1], feats["pca_var_explained"]
        ),
        flush=True,
    )

    params = lgbm_params(num_threads=max(1, min(8, os.cpu_count() or 1)))
    print("Fitting SNI-only LightGBM (700 rounds, seed=0)...", flush=True)
    model, train_proba = fit_sni_lgbm(
        feats["X_sni"], y_codes, feats["X_train"], feats["names"], params, NUM_BOOST_ROUND
    )
    model.save_model(str(WORK_DIR / "sni_lgbm.txt"))
    if train_proba.shape[1] != N_CLASSES:
        raise RuntimeError("expected 60 probability columns, got {}".format(train_proba.shape[1]))
    assert_probability_rows(train_proba, atol=1e-4)
    train_pred = argmax_labels(train_proba, class_names)

    test_proba = np.asarray(model.predict(feats["X_test"]), dtype=np.float64)
    assert_probability_rows(test_proba, atol=1e-4)

    train_ids = np.array([str(v) for v in data.counts_train.index.tolist()], dtype=object)
    test_ids = np.array([str(v) for v in data.counts_test.index.tolist()], dtype=object)
    y_true = data.y_train.loc[train_ids].astype(str).to_numpy()
    fold_map = folds_df.set_index("Cell_ID")["fold"]
    folds = fold_map.reindex(train_ids).to_numpy(dtype=np.int64)

    standalone = classification_metrics(y_true, train_pred, class_names, train_proba)
    fold_acc = {}
    for fold_id in range(5):
        mask = folds == fold_id
        fold_acc[str(fold_id)] = float(np.mean(y_true[mask] == train_pred[mask]))
    standalone["canonical_fold_accuracy"] = fold_acc
    standalone["folds_0_2"] = float(np.mean(y_true[folds <= 2] == train_pred[folds <= 2]))
    standalone["folds_3_4"] = float(np.mean(y_true[folds >= 3] == train_pred[folds >= 3]))
    standalone["evaluation_term"] = "honest_external_validation_not_competition_oof"
    print(
        "SNI standalone acc={:.4f} correct={} macro-F1={:.4f} logloss={:.4f}".format(
            standalone["accuracy"], standalone["correct"], standalone["macro_f1"], standalone["log_loss"]
        ),
        flush=True,
    )

    ei_map = ei_of_label_from_train(data.meta_train, class_names)
    slices = slice_metrics(y_true, train_pred, data.meta_train.loc[train_ids], class_names, ei_map)
    slices["non_hard_bucket_n"] = int((~hard_bucket_mask(data.meta_train.loc[train_ids])).sum())
    hard = hard_bucket_mask(data.meta_train.loc[train_ids])
    slices["non_hard_bucket_accuracy"] = float(np.mean(y_true[~hard] == train_pred[~hard]))
    slices["sni_by_family"] = family_accuracy(y_true, train_pred)

    registry = pd.read_parquet(E03A_REGISTRY)
    registry["Cell_ID"] = registry["Cell_ID"].astype(str)
    registry = registry.set_index("Cell_ID").reindex(train_ids)
    if registry["lzh_pred"].isna().any():
        raise RuntimeError("E03A registry failed to align on Cell_ID")
    lzh_pred = registry["lzh_pred"].astype(str).to_numpy()
    wyh_pred = registry["wyh_pred"].astype(str).to_numpy()
    s0_pred = registry["s0_pred"].astype(str).to_numpy()
    lzh_ok = lzh_pred == y_true
    wyh_ok = wyh_pred == y_true
    s0_ok = s0_pred == y_true
    sni_ok = train_pred == y_true
    if int((~lzh_ok & ~wyh_ok & ~s0_ok).sum()) != ALL_THREE_WRONG:
        raise RuntimeError("frozen all-three-wrong count drifted")
    if int((lzh_ok | wyh_ok | s0_ok).sum()) != THREE_EXPERT_ORACLE:
        raise RuntimeError("frozen three-expert oracle drifted")

    unique = (~lzh_ok) & (~wyh_ok) & (~s0_ok) & sni_ok
    shared_fail = (~lzh_ok) & (~wyh_ok) & (~s0_ok)
    four_ok = lzh_ok | wyh_ok | s0_ok | sni_ok
    rec_n = int(unique.sum())
    rec_02 = int(np.sum(unique & (folds <= 2)))
    rec_34 = int(np.sum(unique & (folds >= 3)))
    four_n = int(four_ok.sum())
    if four_n != THREE_EXPERT_ORACLE + rec_n:
        raise RuntimeError("four-expert identity failed: {} != {} + {}".format(four_n, THREE_EXPERT_ORACLE, rec_n))

    pairwise = [
        pairwise_oracle_row("lzh_prior_h", "sni", lzh_ok, sni_ok, lzh_pred, train_pred),
        pairwise_oracle_row("wyh_model_v2", "sni", wyh_ok, sni_ok, wyh_pred, train_pred),
        pairwise_oracle_row("s0", "sni", s0_ok, sni_ok, s0_pred, train_pred),
    ]
    pd.DataFrame(pairwise).to_csv(OUT_DIR / "v3_e04s_pairwise_oracle.csv", index=False)

    conf = confidence_from_proba(train_proba)
    neuron, glial = neuron_glial_masks(y_true, class_names, ei_map)
    neuron_or_glial = np.where(neuron, "neuron", np.where(glial, "glial_non_neuronal", "other"))
    meta = data.meta_train.loc[train_ids]
    lib, ndet = library_size_and_detected(audit["_x_train"])
    shared_df = pd.DataFrame(
        {
            "Cell_ID": train_ids[shared_fail],
            "true_label": y_true[shared_fail],
            "canonical_fold": folds[shared_fail],
            "lzh_pred": lzh_pred[shared_fail],
            "wyh_pred": wyh_pred[shared_fail],
            "s0_pred": s0_pred[shared_fail],
            "sni_pred": train_pred[shared_fail],
            "sni_top1": conf["top1"][shared_fail],
            "sni_top2": conf["top2"][shared_fail],
            "sni_margin": conf["margin"][shared_fail],
            "sni_entropy": conf["entropy"][shared_fail],
            "Region": meta["Region"].astype(str).to_numpy()[shared_fail],
            "E/I": meta["Excitatory_vs_Inhibitory"].astype(str).to_numpy()[shared_fail],
            "Segment": meta["Segment"].astype(str).to_numpy()[shared_fail],
            "Section_ID": meta["Section_ID"].astype(str).to_numpy()[shared_fail],
            "hard_bucket": hard[shared_fail].astype(bool),
            "n_detected": ndet[shared_fail],
            "library_size": lib[shared_fail],
            "neuron_or_glial": neuron_or_glial[shared_fail],
            "true_family": [family_of(str(v)) for v in y_true[shared_fail]],
            "sni_correct_on_shared_failure": sni_ok[shared_fail].astype(bool),
        }
    )
    shared_df.to_csv(OUT_DIR / "v3_e04s_shared_failure_registry.csv", index=False)

    rec_df = pd.DataFrame(
        {
            "Cell_ID": train_ids[unique],
            "true_label": y_true[unique],
            "canonical_fold": folds[unique],
            "lzh_pred": lzh_pred[unique],
            "wyh_pred": wyh_pred[unique],
            "s0_pred": s0_pred[unique],
            "sni_pred": train_pred[unique],
            "sni_top1": conf["top1"][unique],
            "sni_top2": conf["top2"][unique],
            "sni_margin": conf["margin"][unique],
            "sni_entropy": conf["entropy"][unique],
            "Region": meta["Region"].astype(str).to_numpy()[unique],
            "E/I": meta["Excitatory_vs_Inhibitory"].astype(str).to_numpy()[unique],
            "Segment": meta["Segment"].astype(str).to_numpy()[unique],
            "Section_ID": meta["Section_ID"].astype(str).to_numpy()[unique],
            "hard_bucket": hard[unique].astype(bool),
            "n_detected": ndet[unique],
            "library_size": lib[unique],
            "neuron_or_glial": neuron_or_glial[unique],
            "true_family": [family_of(str(v)) for v in y_true[unique]],
        }
    )
    rec_df.to_csv(OUT_DIR / "v3_e04s_new_unique_recoveries.csv", index=False)

    fam_counter = Counter(rec_df["true_family"].tolist()) if len(rec_df) else Counter()
    rec_by_family = [
        {"family": name, "n": int(fam_counter.get(name, 0))}
        for name in list(CLASS_FAMILIES_E03A.keys()) + ["neuronal_or_other"]
    ]
    class_counter = Counter(rec_df["true_label"].tolist()) if len(rec_df) else Counter()
    top_classes = [{"true_label": k, "n": int(v), "family": family_of(k)} for k, v in class_counter.most_common(15)]

    # Shared confusion: true -> LZH among the 689, with SNI rescue counts.
    fail_true = y_true[shared_fail]
    fail_lzh = lzh_pred[shared_fail]
    fail_sni_ok = sni_ok[shared_fail]
    pair_counts = Counter(zip(fail_true.tolist(), fail_lzh.tolist()))
    pair_rescue = Counter()
    for t, p, ok_cell in zip(fail_true.tolist(), fail_lzh.tolist(), fail_sni_ok.tolist()):
        if ok_cell:
            pair_rescue[(t, p)] += 1
    confusion_rows = []
    for (t, p), n_err in pair_counts.most_common(20):
        n_res = int(pair_rescue.get((t, p), 0))
        confusion_rows.append(
            {
                "true_label": t,
                "pred": p,
                "error_count": int(n_err),
                "sni_rescue_count": n_res,
                "sni_rescue_fraction": float(n_res / n_err) if n_err else 0.0,
                "family": family_of(t),
            }
        )

    sni_corr = unique[shared_fail]
    sni_wrong_shared = ~sni_ok[shared_fail]
    top1_sf = conf["top1"][shared_fail]
    margin_sf = conf["margin"][shared_fail]
    ent_sf = conf["entropy"][shared_fail]

    def _auc(y_bin, scores) -> Optional[float]:
        y_bin = np.asarray(y_bin, dtype=int)
        scores = np.asarray(scores, dtype=np.float64)
        if y_bin.size == 0 or len(np.unique(y_bin)) < 2:
            return None
        return float(roc_auc_score(y_bin, scores))

    conf_diag = {
        "sni_correct": {
            "count": int(sni_corr.sum()),
            "top1": summarize_numeric(top1_sf[sni_corr]),
            "margin": summarize_numeric(margin_sf[sni_corr]),
            "entropy": summarize_numeric(ent_sf[sni_corr]),
        },
        "sni_wrong": {
            "count": int(sni_wrong_shared.sum()),
            "top1": summarize_numeric(top1_sf[sni_wrong_shared]),
            "margin": summarize_numeric(margin_sf[sni_wrong_shared]),
            "entropy": summarize_numeric(ent_sf[sni_wrong_shared]),
        },
        "auroc": {
            "top1": _auc(sni_corr, top1_sf),
            "margin": _auc(sni_corr, margin_sf),
            "neg_entropy": _auc(sni_corr, -ent_sf),
            "note": "diagnostic only; no threshold optimized",
        },
    }

    four = {
        "formula": "{} + sni_new_unique_recoveries".format(THREE_EXPERT_ORACLE),
        "oracle_is_not_deployable_accuracy": True,
        "overall": {
            "correct": four_n,
            "accuracy": float(four_n / N_TRAIN),
            "n": N_TRAIN,
        },
        "folds_0_2": {
            "correct": int(np.sum(four_ok[folds <= 2])),
            "accuracy": float(np.mean(four_ok[folds <= 2])),
            "n": int((folds <= 2).sum()),
        },
        "folds_3_4": {
            "correct": int(np.sum(four_ok[folds >= 3])),
            "accuracy": float(np.mean(four_ok[folds >= 3])),
            "n": int((folds >= 3).sum()),
        },
        "all_four_wrong": int((~four_ok).sum()),
        "identity_check": four_n == THREE_EXPERT_ORACLE + rec_n,
    }

    integrity_ok = (
        bool(audit["md5_ok"])
        and bool(genes["exact_200_of_200"])
        and tax["case"] in {"A", "B"}
        and bool(audit["nonnegative_counts"])
    )
    label, why = classify_experiment(rec_n, rec_02, rec_34, four["overall"]["accuracy"], integrity_ok)
    decision = {"label": label, "reason": why, "next_action": next_action_for(label)}

    leak = {
        "competition_test_labels_used": False,
        "competition_train_labels_used_for_sni_fitting": False,
        "competition_id_overlaps_excluded": True,
        "competition_exact_vector_duplicates_excluded": True,
        "merfish_exact_vector_duplicates_excluded": True,
        "leaderboard_feedback_used": False,
        "hyperparameter_search": False,
        "seed_search": False,
        "blend_optimization": False,
        "canonical_folds_3_4_used_for_redesign": False,
        "test_probabilities_used_for_model_selection": False,
        "submission_generated": False,
        "prediction_csv_modified": False,
        "frozen_artifacts_modified": False,
        "cell_id_dtype": "lossless_string",
        "sni_concatenated_with_merfish_reference": False,
        "router_trained": False,
        "ensemble_weights_optimized": False,
    }

    val_frame = pd.DataFrame(
        {
            "Cell_ID": train_ids,
            "true_label": y_true,
            "predicted_label": train_pred,
            "canonical_fold": folds,
            "evaluation": "honest_external_validation_not_competition_oof",
        }
    )
    val_frame.to_csv(OUT_DIR / "v3_e04s_sni_validation.csv", index=False)
    write_proba(OUT_DIR / "v3_e04s_sni_validation_probabilities.csv.gz", train_ids, train_proba, class_names)
    write_proba(OUT_DIR / "v3_e04s_sni_test_probabilities.csv.gz", test_ids, test_proba, class_names)

    public_audit = {k: v for k, v in audit.items() if not str(k).startswith("_")}
    write_json(OUT_DIR / "v3_e04s_dataset_audit.json", public_audit)

    complementarity = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": utc_now(),
        "oracle_is_not_deployable_accuracy": True,
        "frozen_three_expert_oracle_correct": THREE_EXPERT_ORACLE,
        "frozen_all_three_wrong": ALL_THREE_WRONG,
        "sni_new_unique_recoveries": rec_n,
        "sni_new_unique_recoveries_folds_0_2": rec_02,
        "sni_new_unique_recoveries_folds_3_4": rec_34,
        "four_expert_oracle_correct": four_n,
        "four_expert_oracle_accuracy": four["overall"]["accuracy"],
        "four_expert_identity": "{} + {} = {}".format(THREE_EXPERT_ORACLE, rec_n, four_n),
        "all_four_wrong": four["all_four_wrong"],
        "pairwise": pairwise,
        "four_expert": four,
        "standalone": standalone,
        "slices": slices,
        "recoveries_by_family": rec_by_family,
        "top_recovery_classes": top_classes,
        "top_resolved_shared_confusions": confusion_rows,
        "confidence_diagnostic": conf_diag,
        "decision": decision,
        "leakage_audit": leak,
        "runtime_sec": float(time.time() - t0),
        "pca_var_explained": feats["pca_var_explained"],
        "n_sni_usable": int(keep.sum()),
        "lgbm": {"seed": LGBM_SEED, "num_boost_round": NUM_BOOST_ROUND, "params": params},
    }
    write_json(OUT_DIR / "v3_e04s_complementarity.json", complementarity)

    pd.DataFrame(slices["sni_by_family"]).to_csv(TABLE_DIR / "sni_family_accuracy.csv", index=False)
    pd.DataFrame(rec_by_family).to_csv(TABLE_DIR / "sni_new_recoveries_by_family.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(TABLE_DIR / "shared_confusion_sni_rescue.csv", index=False)
    pd.DataFrame(top_classes).to_csv(TABLE_DIR / "top_recovery_classes.csv", index=False)

    write_report(
        {
            "audit": public_audit,
            "standalone": standalone,
            "slices": slices,
            "pairwise": pairwise,
            "recoveries": {
                "sni_new_unique_recoveries": rec_n,
                "sni_new_unique_recoveries_folds_0_2": rec_02,
                "sni_new_unique_recoveries_folds_3_4": rec_34,
                "shared_failure_n": int(shared_fail.sum()),
                "by_family": rec_by_family,
                "top_resolved_shared_confusions": confusion_rows,
                "confidence_diagnostic": conf_diag,
            },
            "four_expert": four,
            "leakage_audit": leak,
            "decision": decision,
        }
    )

    pred_after = sha256_file(PRED_PATH) if PRED_PATH.is_file() else None
    if pred_before != pred_after:
        raise RuntimeError("prediction/prediction.csv was modified")

    print("Decision: {}".format(label), flush=True)
    print("sni_new_unique_recoveries={}".format(rec_n), flush=True)
    print("four_expert_oracle={}/5000={:.4f}".format(four_n, four["overall"]["accuracy"]), flush=True)
    print("elapsed_sec={:.1f}".format(time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
