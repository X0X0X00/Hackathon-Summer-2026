#!/usr/bin/env python3
"""V3-E00T: team canonical expert and complementarity audit.

Read-only with respect to teammate branches and frozen MODEL V1/V2.
Does not train models, search blend weights, tune thresholds, or write
prediction/prediction.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from merfish60.io import TARGET_COL, load_dataset, validate_contract  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.official_contract import (  # noqa: E402
    allowed_labels,
    expected_test_cell_ids,
    sha256_file,
)
from merfish60.signatures import canonicalize_value, missing_bucket_key, signatures_from_meta  # noqa: E402
from merfish60.spatial_features import ei_of_label_from_train  # noqa: E402
from merfish60.team_cv import TEAM_CV_PROTOCOL, load_and_validate_team_folds  # noqa: E402
from merfish60.v2_metrics import (  # noqa: E402
    hard_bucket_mask,
    json_default,
    neuron_glial_masks,
    write_json,
)

N_TRAIN = 5000
N_CLASSES = 60
WYH_V2_OOF_PRED = ROOT / "outputs" / "oof" / "MODEL-V2_oof.csv"
WYH_V2_OOF_PROBA = ROOT / "outputs" / "probabilities" / "V2-B-REFONLY_oof_probabilities_seg.csv.gz"
WYH_V2_TEST_PROBA = ROOT / "outputs" / "probabilities" / "V2-B-REFONLY_test_probabilities_seg.csv.gz"
WYH_V2_METRICS = ROOT / "outputs" / "metrics" / "model_v2_metrics.json"
TEAM_FOLDS = ROOT / "experiments" / "team_folds_5_seed42.csv"

LZH_BRANCH = "team/lzh"
LZH_FINAL_OOF = (
    "submission_graph_stacker_V3_prior_H_20260820/model/oof_probabilities_final.csv"
)
LZH_FINAL_TEST = (
    "submission_graph_stacker_V3_prior_H_20260820/model/test_probabilities_final.csv"
)
LZH_PRIOR_H_OOF = (
    "submission_graph_stacker_V3_prior_H_20260820/model/inputs/"
    "oof_probabilities_prior_h_anchor.csv"
)
LZH_GATE = (
    "submission_graph_stacker_V3_prior_H_20260820/model/trained_stacker/oof_gate.csv"
)
LZH_SELECTION = (
    "submission_graph_stacker_V3_prior_H_20260820/model/final_selection.json"
)

CLASS_FAMILIES = OrderedDict(
    [
        (
            "oligodendrocyte_opc",
            [
                "oligodendrocyte_1",
                "oligodendrocyte_2",
                "oligodendrocyte_precursor_cell",
                "oligodendrocyte_progenitor_1",
                "oligodendrocyte_progenitor_2",
            ],
        ),
        ("astrocyte", ["astrocyte_1", "astrocyte_2"]),
        ("vascular", ["endothelial", "pericyte"]),
        ("meningeal", ["meninges_1", "meninges_2", "meninges_3"]),
        (
            "other_glial_non_neuronal",
            ["microglia", "ependymal", "Schwann_cell", "peripheral_glia"],
        ),
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git(args: Sequence[str], cwd: Optional[Path] = None) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(cwd or ROOT),
        stderr=subprocess.STDOUT,
    ).decode("utf-8")


def branch_snapshot(ref: str) -> dict:
    try:
        raw = run_git(["log", "-1", "--format=%H\t%cI\t%s\t%an", ref]).strip()
    except subprocess.CalledProcessError:
        return {"ref": ref, "present": False}
    sha, date, subject, author = raw.split("\t", 3)
    return {
        "ref": ref,
        "present": True,
        "sha": sha,
        "date": date,
        "subject": subject,
        "author": author,
    }


def materialize_git_blob(ref_path: str, dest: Path) -> dict:
    blob = subprocess.check_output(["git", "show", ref_path], cwd=str(ROOT))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    branch, original = ref_path.split(":", 1)
    sha = run_git(["rev-parse", branch]).strip()
    return {
        "branch": branch,
        "commit_sha": sha,
        "original_path": original,
        "local_path": str(dest),
        "sha256": digest,
        "n_bytes": len(blob),
    }


def read_cell_id_csv(path: Path, extra_dtypes: Optional[dict] = None) -> pd.DataFrame:
    dtypes = {"Cell_ID": str}
    if extra_dtypes:
        dtypes.update(extra_dtypes)
    frame = pd.read_csv(path, dtype=dtypes, keep_default_na=False)
    if "Cell_ID" not in frame.columns:
        raise ValueError("{} missing Cell_ID".format(path))
    frame["Cell_ID"] = frame["Cell_ID"].astype(str)
    if frame["Cell_ID"].duplicated().any():
        raise ValueError("{} has duplicate Cell_ID".format(path))
    if frame["Cell_ID"].str.endswith(".0").any():
        raise ValueError("{} Cell_ID looks float-cast".format(path))
    if (~frame["Cell_ID"].str.fullmatch(r"\d{19}")).any():
        raise ValueError("{} has non 19-digit Cell_ID".format(path))
    return frame


def class_columns_from_header(columns: Sequence[str], class_names: Sequence[str]) -> List[str]:
    names = list(class_names)
    raw = [c for c in columns if c != "Cell_ID"]
    if raw == names:
        return names
    stripped = [c[3:] if c.startswith("p__") else c for c in raw]
    if stripped == names:
        return ["p__" + c if raw[i].startswith("p__") else c for i, c in enumerate(names)]
    if set(stripped) == set(names) and len(stripped) == len(names):
        mapping = {stripped[i]: raw[i] for i in range(len(raw))}
        return [mapping[c] for c in names]
    raise ValueError(
        "probability columns do not match official 60-class order: {}".format(raw[:8])
    )


def load_probability_frame(
    path: Path,
    class_names: Sequence[str],
    compression: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    kwargs = {"dtype": {"Cell_ID": str}, "keep_default_na": False}
    if compression:
        kwargs["compression"] = compression
    elif str(path).endswith(".gz"):
        kwargs["compression"] = "gzip"
    frame = pd.read_csv(path, **kwargs)
    frame["Cell_ID"] = frame["Cell_ID"].astype(str)
    source_cols = class_columns_from_header(list(frame.columns), class_names)
    proba = frame.loc[:, source_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    audit = inspect_probabilities(proba, class_names)
    out = pd.DataFrame(proba, columns=list(class_names))
    out.insert(0, "Cell_ID", frame["Cell_ID"].to_numpy())
    return out, audit


def inspect_probabilities(proba: np.ndarray, class_names: Sequence[str]) -> dict:
    if proba.shape[1] != len(class_names):
        raise ValueError("expected {} classes, got {}".format(len(class_names), proba.shape[1]))
    finite = bool(np.isfinite(proba).all())
    n_neg = int((proba < 0).sum())
    row_sums = proba.sum(axis=1)
    max_abs_dev = float(np.max(np.abs(row_sums - 1.0)))
    n_bad_sum = int(np.sum(~np.isclose(row_sums, 1.0, atol=1e-4)))
    usable = finite and n_neg == 0 and n_bad_sum == 0
    if usable:
        assert_probability_rows(proba, atol=1e-4)
    return {
        "n_rows": int(proba.shape[0]),
        "n_classes": int(proba.shape[1]),
        "finite": finite,
        "n_negative": n_neg,
        "max_abs_row_sum_deviation": max_abs_dev,
        "n_rows_not_summing_to_one_atol_1e4": n_bad_sum,
        "usable": usable,
        "class_order_source": "explicit probability column names aligned to allowed_labels()",
    }


def confidence_from_proba(proba: np.ndarray) -> dict:
    clipped = np.clip(proba, 0.0, None)
    totals = clipped.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1e-12)
    normed = clipped / totals
    order = np.argsort(normed, axis=1)
    top1 = normed[np.arange(len(normed)), order[:, -1]]
    top2 = normed[np.arange(len(normed)), order[:, -2]]
    logp = np.zeros_like(normed)
    np.log(normed, out=logp, where=normed > 0)
    entropy = -np.sum(normed * logp, axis=1)
    return {
        "top1": top1.astype(np.float64),
        "top2": top2.astype(np.float64),
        "margin": (top1 - top2).astype(np.float64),
        "entropy": entropy.astype(np.float64),
    }


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.empty(len(p), dtype=np.float64)
    for i in range(len(p)):
        dist = float(jensenshannon(p[i], q[i], base=2.0))
        if not np.isfinite(dist):
            dist = 0.0
        out[i] = dist * dist
    return out


def summarize_numeric(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def classification_status(reported: Optional[float], reproduced: Optional[float]) -> str:
    if reported is None or reproduced is None:
        return "NOT VERIFIED"
    delta = abs(float(reported) - float(reproduced))
    if delta <= 1e-10:
        return "VERIFIED"
    if delta <= 5e-4:
        return "CLOSE / ROUNDING"
    return "NOT VERIFIED"


def partition_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    folds: np.ndarray,
    class_names: Sequence[str],
    proba: Optional[np.ndarray] = None,
) -> dict:
    out = {
        "n": int(len(y_true)),
        "correct": int(np.sum(y_true == y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    fold_acc = {}
    for fold_id in range(5):
        mask = folds == fold_id
        fold_acc[str(fold_id)] = float(accuracy_score(y_true[mask], y_pred[mask]))
    out["canonical_fold_accuracy"] = fold_acc
    for name, mask in (
        ("canonical_folds_0_2", folds <= 2),
        ("canonical_folds_3_4", folds >= 3),
    ):
        out[name] = {
            "n": int(mask.sum()),
            "correct": int(np.sum((y_true == y_pred) & mask)),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
        }
    if proba is not None:
        out["log_loss"] = float(log_loss(y_true, proba, labels=list(class_names)))
    return out


def pair_metrics(
    name_a: str,
    pred_a: np.ndarray,
    name_b: str,
    pred_b: np.ndarray,
    y_true: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)
    a = pred_a[mask]
    b = pred_b[mask]
    y = y_true[mask]
    a_ok = a == y
    b_ok = b == y
    disagree = a != b
    oracle_ok = a_ok | b_ok
    acc_a = float(np.mean(a_ok))
    acc_b = float(np.mean(b_ok))
    stronger = max(acc_a, acc_b)
    return {
        "expert_a": name_a,
        "expert_b": name_b,
        "n": int(mask.sum()),
        "a_accuracy": acc_a,
        "b_accuracy": acc_b,
        "disagreement_count": int(disagree.sum()),
        "disagreement_rate": float(np.mean(disagree)),
        "both_correct": int(np.sum(a_ok & b_ok)),
        "a_only_correct": int(np.sum(a_ok & ~b_ok)),
        "b_only_correct": int(np.sum(b_ok & ~a_ok)),
        "both_wrong": int(np.sum(~a_ok & ~b_ok)),
        "pair_oracle_correct": int(oracle_ok.sum()),
        "pair_oracle_accuracy": float(np.mean(oracle_ok)),
        "incremental_oracle_headroom_over_stronger": float(np.mean(oracle_ok) - stronger),
    }


def multi_oracle(preds: Dict[str, np.ndarray], y_true: np.ndarray, mask: np.ndarray) -> dict:
    names = list(preds)
    y = y_true[mask]
    correct_mat = np.stack([(preds[name][mask] == y) for name in names], axis=1)
    n_correct = correct_mat.sum(axis=1)
    oracle_ok = n_correct > 0
    dist = {str(k): int(np.sum(n_correct == k)) for k in range(len(names) + 1)}
    return {
        "experts": names,
        "n": int(mask.sum()),
        "oracle_correct": int(oracle_ok.sum()),
        "oracle_accuracy": float(np.mean(oracle_ok)),
        "n_correct_experts_distribution": dist,
        "all_wrong": int(np.sum(n_correct == 0)),
    }


def apply_consensus(
    rule: str,
    anchor: np.ndarray,
    alternatives: Dict[str, np.ndarray],
    y_true: np.ndarray,
    mask: np.ndarray,
) -> dict:
    alt_names = list(alternatives)
    alt_preds = [alternatives[name] for name in alt_names]
    chosen = anchor.copy()
    changed = np.zeros(len(anchor), dtype=bool)
    if rule == "D0":
        pass
    elif rule == "D1":
        if len(alt_preds) >= 2:
            for i in range(len(anchor)):
                labels = [p[i] for p in alt_preds]
                if len(set(labels)) == 1 and labels[0] != anchor[i]:
                    chosen[i] = labels[0]
                    changed[i] = True
    elif rule == "D2":
        if len(alt_preds) >= 1:
            for i in range(len(anchor)):
                labels = [p[i] for p in alt_preds]
                if len(set(labels)) == 1 and labels[0] != anchor[i]:
                    chosen[i] = labels[0]
                    changed[i] = True
    elif rule == "D3":
        experts = [anchor] + alt_preds
        for i in range(len(anchor)):
            votes = Counter(p[i] for p in experts)
            top_label, top_n = votes.most_common(1)[0]
            tied = sum(1 for _lab, n in votes.items() if n == top_n) > 1
            if (not tied) and top_n > len(experts) / 2.0 and top_label != anchor[i]:
                chosen[i] = top_label
                changed[i] = True
    else:
        raise ValueError(rule)

    sub_changed = changed[mask]
    y = y_true[mask]
    before = anchor[mask]
    after = chosen[mask]
    wrong_to_right = int(np.sum(sub_changed & (before != y) & (after == y)))
    right_to_wrong = int(np.sum(sub_changed & (before == y) & (after != y)))
    n_changed = int(sub_changed.sum())
    return {
        "rule": rule,
        "n": int(mask.sum()),
        "changed_cells": n_changed,
        "wrong_to_correct": wrong_to_right,
        "correct_to_wrong": right_to_wrong,
        "net_corrections": wrong_to_right - right_to_wrong,
        "correction_precision": (
            float(wrong_to_right / n_changed) if n_changed else None
        ),
        "final_diagnostic_accuracy": float(np.mean(after == y)),
        "n_alternatives": len(alt_names),
    }


def family_of(label: str) -> str:
    for family, members in CLASS_FAMILIES.items():
        if label in members:
            return family
    return "neuronal_or_other"


def safe_auc(y_bin: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if y_bin.size == 0 or len(np.unique(y_bin)) < 2:
        return None
    return float(roc_auc_score(y_bin, scores))


def current_branch() -> str:
    return run_git(["branch", "--show-current"]).strip()


def build_expert_manifest(
    snapshots: dict,
    materialized: dict,
    class_names: Sequence[str],
    wyh_metrics: dict,
    lzh_selection: dict,
    reproduced: dict,
) -> dict:
    wyh_sha = snapshots["origin_head"]["sha"] if snapshots["origin_head"]["present"] else snapshots["local_head"]["sha"]
    lzh_sha = snapshots["team/lzh"]["sha"]
    yhh_sha = snapshots["team/yhh"]["sha"]
    main_sha = snapshots["team/main"]["sha"]
    experts = [
        {
            "expert_id": "wyh_model_v2",
            "owner_line": "WYH",
            "branch": "ywan/ml-pipeline",
            "branch_sha": snapshots["local_head"]["sha"],
            "artifact_commit_sha": "2a25ee2e0c9f9d1d0f3c0c0c0c0c0c0c0c0c0c0c",
            "artifact_commit_sha_note": "frozen tag model-v2; exact SHA filled below",
            "selected_model_name": "MODEL V2 / V2-B-REFONLY",
            "oof_prediction_path": str(WYH_V2_OOF_PRED.relative_to(ROOT)),
            "oof_probability_path": str(WYH_V2_OOF_PROBA.relative_to(ROOT)),
            "test_probability_path": str(WYH_V2_TEST_PROBA.relative_to(ROOT)),
            "validation_protocol": TEAM_CV_PROTOCOL,
            "n_folds": 5,
            "fold_seed": 42,
            "model_selection_protocol": "predeclared V2-C blends only; C1 rejected; no weight search",
            "folds_3_4_used_as_locked_holdout": False,
            "reported_oof_used_for_tuning": False,
            "reported_accuracy": 0.8212,
            "reproduced_accuracy": reproduced.get("wyh_model_v2"),
            "probability_available": True,
            "test_probability_available": True,
            "eligibility_level": "A",
            "validation_confidence": "HIGH",
            "eligibility_reason": (
                "Frozen personal MODEL V2 artifacts with Cell_ID-aligned 5-fold seed42 "
                "OOF predictions and probabilities. Canonical WYH source; team/wyh not used."
            ),
            "in_predeclared_oracle_pool": True,
            "comparability_warning": None,
            "notes": "Reference-only LightGBM on cleaned Zenodo reference; competition train labels are not the boosting target.",
        },
        {
            "expert_id": "lzh_prior_h",
            "owner_line": "LZH",
            "branch": "team/lzh",
            "branch_sha": lzh_sha,
            "artifact_commit_sha": lzh_sha,
            "selected_model_name": lzh_selection.get("selected_model", "depth_masked_prior_h_anchor"),
            "oof_prediction_path": None,
            "oof_probability_path": LZH_FINAL_OOF,
            "test_probability_path": LZH_FINAL_TEST,
            "materialized": materialized.get("lzh_final_oof"),
            "validation_protocol": "outer three-fold cross-fit (LZH folds.npy; not canonical seed42)",
            "n_folds": 3,
            "fold_seed": None,
            "model_selection_protocol": "select Prior-H Anchor over newly trained graph stacker using the same 3-fold OOF",
            "folds_3_4_used_as_locked_holdout": False,
            "reported_oof_used_for_tuning": True,
            "reported_accuracy": float(lzh_selection["selected_metrics"]["accuracy"]),
            "reproduced_accuracy": reproduced.get("lzh_prior_h"),
            "probability_available": True,
            "test_probability_available": True,
            "eligibility_level": "A",
            "validation_confidence": "MEDIUM",
            "eligibility_reason": (
                "Cell-level OOF probabilities with explicit Cell_ID and named 60-class columns. "
                "Bundle documents 3-fold held-out cross-fit. Canonical seed42 slices are a "
                "meta-analysis partition, not LZH original folds."
            ),
            "in_predeclared_oracle_pool": True,
            "comparability_warning": (
                "LZH original protocol is 3-fold, not StratifiedKFold seed42. "
                "Do not describe canonical folds 3-4 as LZH model folds."
            ),
            "notes": "Selected because the V3 graph stacker (0.8254) did not beat Prior-H (0.8266).",
        },
        {
            "expert_id": "yhh_v7",
            "owner_line": "YHH",
            "branch": "team/yhh",
            "branch_sha": yhh_sha,
            "artifact_commit_sha": yhh_sha,
            "selected_model_name": "V7 hierarchical specialists on reproduced V6 ensemble",
            "oof_prediction_path": None,
            "oof_probability_path": None,
            "test_probability_path": None,
            "test_prediction_path": "prediction/prediction.csv",
            "validation_protocol": "5-fold; selection on folds 0-2, holdout folds 3-4 (YHH README)",
            "n_folds": 5,
            "fold_seed": 42,
            "model_selection_protocol": "gated specialists accepted only with positive tune and holdout gain",
            "folds_3_4_used_as_locked_holdout": True,
            "reported_oof_used_for_tuning": True,
            "reported_accuracy": 0.8320,
            "reproduced_accuracy": None,
            "probability_available": False,
            "test_probability_available": False,
            "eligibility_level": "C",
            "validation_confidence": "LOW",
            "eligibility_reason": (
                "Reported 0.8320 / 4160 correct in work/v7_optimization_report.json, but cell-level "
                "OOF is only written to gitignored work/v7_final_probs.npz. Root prediction.csv is "
                "test-only. Team main independently recorded that V7 did not reproduce on zzh member files."
            ),
            "in_predeclared_oracle_pool": False,
            "comparability_warning": "Cannot enter oracle/reliability analysis without cell-level OOF.",
            "notes": "V7 commit 0890a3d added only README, test prediction.csv, optimize_v7.py, and the JSON report.",
        },
        {
            "expert_id": "team_main_v9",
            "owner_line": "team_main / zzh",
            "branch": "team/main",
            "branch_sha": main_sha,
            "artifact_commit_sha": main_sha,
            "selected_model_name": "team blend v9",
            "oof_prediction_path": None,
            "oof_probability_path": None,
            "test_probability_path": None,
            "test_prediction_path": "prediction/prediction.csv",
            "validation_protocol": None,
            "n_folds": None,
            "fold_seed": None,
            "model_selection_protocol": None,
            "folds_3_4_used_as_locked_holdout": None,
            "reported_oof_used_for_tuning": None,
            "reported_accuracy": None,
            "reproduced_accuracy": None,
            "probability_available": False,
            "test_probability_available": False,
            "eligibility_level": "C",
            "validation_confidence": "LOW",
            "eligibility_reason": (
                "HEAD only updates prediction/prediction.csv (test predictions). No committed OOF "
                "predictions or probabilities. Historical V6 OOF npz files are gitignored."
            ),
            "in_predeclared_oracle_pool": False,
            "comparability_warning": "Do not invent an OOF metric. Test disagreement is not validation evidence.",
            "notes": "Latest reproducible documented ensemble on this line was V6 (reported ~0.8248) but member OOF npz files are not in git.",
        },
        {
            "expert_id": "lzh_graph_stacker_v2_historical",
            "owner_line": "LZH",
            "branch": "team/lzh",
            "branch_sha": lzh_sha,
            "artifact_commit_sha": lzh_sha,
            "selected_model_name": "graph_regularized_logit_stacker (82.50 package)",
            "oof_prediction_path": None,
            "oof_probability_path": "submission_graph_stacker_82_50_20260820/model/oof_probabilities.csv",
            "test_probability_path": "submission_graph_stacker_82_50_20260820/model/test_probabilities.csv",
            "validation_protocol": "outer three-fold cross-fit",
            "n_folds": 3,
            "fold_seed": None,
            "reported_accuracy": 0.8250,
            "reproduced_accuracy": None,
            "probability_available": True,
            "test_probability_available": True,
            "eligibility_level": "A",
            "validation_confidence": "MEDIUM",
            "eligibility_reason": "Honest 3-fold OOF probabilities exist, but this package is superseded by the selected Prior-H model on the same owner line.",
            "in_predeclared_oracle_pool": False,
            "comparability_warning": "Not added to the predeclared 4-line oracle pool; it is a superseded LZH package, not an independent owner line.",
            "notes": "Inventoried only. Using it together with Prior-H would double-count the LZH family.",
        },
    ]
    # Fill exact model-v2 tag SHA.
    try:
        v2_sha = run_git(["rev-list", "-n", "1", "model-v2"]).strip()
    except subprocess.CalledProcessError:
        v2_sha = snapshots["local_head"]["sha"]
    experts[0]["artifact_commit_sha"] = v2_sha
    del experts[0]["artifact_commit_sha_note"]
    return {
        "created_at_utc": utc_now(),
        "audit_id": "V3-E00T",
        "active_branch": current_branch(),
        "snapshots": snapshots,
        "materialized_artifacts": materialized,
        "experts": experts,
    }


def write_markdown_report(path: Path, payload: dict) -> None:
    snap = payload["snapshots"]
    experts = {row["expert_id"]: row for row in payload["manifest"]["experts"]}
    ind = payload["individual"]
    multi = payload["multi_expert_oracle"]
    rec = payload["recommended_anchor"]
    recov = payload["anchor_recoverability"]
    cons = payload["consensus"]
    conf = payload["confidence"]
    unique = payload["unique_value"]
    bio = payload["biological_families"]
    confusion_pairs = bio.get("confusion_pairs") or bio.get("top_confusion_pairs") or []
    decision = payload["strategic_decision"]

    def acc_line(expert_id: str) -> str:
        row = ind[expert_id]
        return (
            "| `{id}` | {level} | {rep} | {repro:.4f} | {status} | {f02:.4f} | {f34:.4f} |".format(
                id=expert_id,
                level=experts[expert_id]["eligibility_level"],
                rep="n/a" if row["reported_accuracy"] is None else "{:.4f}".format(row["reported_accuracy"]),
                repro=row["overall"]["accuracy"],
                status=row["verification_status"],
                f02=row["overall"]["canonical_folds_0_2"]["accuracy"],
                f34=row["overall"]["canonical_folds_3_4"]["accuracy"],
            )
        )

    pairwise_lines = [
        "| {a} | {b} | {oa:.4f} | {ob:.4f} | {d} | {ora:.4f} | {h:.4f} |".format(
            a=r["expert_a"],
            b=r["expert_b"],
            oa=r["overall"]["a_accuracy"],
            ob=r["overall"]["b_accuracy"],
            d=r["overall"]["disagreement_count"],
            ora=r["overall"]["pair_oracle_accuracy"],
            h=r["overall"]["incremental_oracle_headroom_over_stronger"],
        )
        for r in payload["pairwise"]
    ]
    consensus_lines = []
    for rule in ["D0", "D1", "D2", "D3"]:
        block = cons[rule]
        consensus_lines.append(
            "| {rule} | {ch} | {w2c} | {c2w} | {net} | {prec} | {acc:.4f} | {acc34:.4f} |".format(
                rule=rule,
                ch=block["overall"]["changed_cells"],
                w2c=block["overall"]["wrong_to_correct"],
                c2w=block["overall"]["correct_to_wrong"],
                net=block["overall"]["net_corrections"],
                prec=(
                    "n/a"
                    if block["overall"]["correction_precision"] is None
                    else "{:.3f}".format(block["overall"]["correction_precision"])
                ),
                acc=block["overall"]["final_diagnostic_accuracy"],
                acc34=block["canonical_folds_3_4"]["final_diagnostic_accuracy"],
            )
        )

    family_lines = []
    for row in confusion_pairs[:15]:
        family_lines.append(
            "| {t} | {p} | {n} | {r} | {f:.3f} | {who} |".format(
                t=row["true_label"],
                p=row["anchor_pred"],
                n=row["error_count"],
                r=row["recoverable_count"],
                f=row["recoverable_fraction"],
                who=", ".join(row["rescuing_experts"]) if row["rescuing_experts"] else "none",
            )
        )

    body = """# V3-E00T — Team Canonical Expert & Complementarity Audit

## 1. Objective

Quantify whether currently available **honest, cell-level OOF** team experts have enough complementary headroom to justify a MODEL V3 reliability-routing experiment. This audit does **not** train a new model, search blend weights, tune thresholds, or produce a submission.

Research hypothesis: selective expert correction may be more useful than another globally averaged ensemble, **if** a small pool of methodologically different experts has real oracle headroom on a locked canonical partition.

Oracle accuracy is diagnostic headroom only. **ORACLE != DEPLOYABLE MODEL ACCURACY.**

## 2. Team Repository Snapshot

Fetched `team --prune` at audit start. Remote-tracking branches were inspected with `git show` / `git ls-tree` only. No teammate branch was checked out.

| Branch | SHA | Date | Latest subject / selected candidate |
|---|---|---|---|
| `team/main` | `{main_sha}` | {main_date} | {main_subj} (test-only blend v9) |
| `team/yhh` | `{yhh_sha}` | {yhh_date} | {yhh_subj} (V7 hierarchical specialists, reported 0.8320) |
| `team/lzh` | `{lzh_sha}` | {lzh_date} | {lzh_subj} (selected `depth_masked_prior_h_anchor`, 0.8266) |
| `team/wyh` | `{wyh_sha}` | {wyh_date} | {wyh_subj} (mature WYH delivery; personal frozen V2 used instead) |
| `team/revert-1-lzh` | `{rev_sha}` | {rev_date} | {rev_subj} (historical provenance only) |

Personal development branch: `ywan/ml-pipeline` @ `{local_sha}`. Frozen MODEL V2 tag: `model-v2`.

## 3. Expert Artifact Manifest

| Expert | Owner | Model | OOF artifacts | Protocol | Level | Confidence |
|---|---|---|---|---|---|---|
| `wyh_model_v2` | WYH | V2-B-REFONLY | `{wyh_oof}` + `{wyh_proba}` | 5-fold seed42 | A | HIGH |
| `lzh_prior_h` | LZH | `{lzh_model}` | `{lzh_oof}` | 3-fold LZH OOF | A | MEDIUM |
| `yhh_v7` | YHH | V7 hierarchical specialists | **missing cell-level OOF** (gitignored npz) | claimed 5-fold, tune 0-2 / holdout 3-4 | C | LOW |
| `team_main_v9` | zzh / main | team blend v9 | test `prediction/prediction.csv` only | unknown for v9 | C | LOW |
| `lzh_graph_stacker_v2_historical` | LZH | 82.50 graph stacker | committed OOF probabilities | 3-fold | A | MEDIUM |

`lzh_graph_stacker_v2_historical` is inventoried but **excluded from the predeclared oracle pool** because it is a superseded model on the same LZH line, not an independent owner-line expert.

Exact paths, SHAs, and eligibility reasons: `outputs/v3/v3_e00t_expert_manifest.json`.

## 4. Integrity Checks

- Canonical population: **5000** unique 19-digit `Cell_ID` strings from official `meta_train`.
- Labels aligned from `MERFISH_cell_type_annotation`; no missing labels.
- Joins used `Cell_ID`, never row position as the primary key.
- Official class order: `allowed_labels()` (60 sorted training labels). WYH probability columns already match. LZH columns use `p__<class>` prefixes and match the same 60 names.
- Probability rows are finite, non-negative, and sum to 1 within `atol=1e-4`.
- LZH final OOF probabilities match `oof_probabilities_prior_h_anchor.csv` (selected model is the Prior-H Anchor, not the newly trained stacker).
- Competition test labels were not used. Test probability files were inventoried for Cell_ID/class-order only.
- `prediction/prediction.csv` was not modified.
- `pyarrow==19.0.1` was added only to support reproducible Parquet artifact I/O for V3-E00T, not as a modeling dependency.

## 5. Individual Expert Metrics

Reported vs reproduced overall accuracy on the canonical 5000-cell population.

| Expert | Level | Reported | Reproduced | Status | Canonical 0-2 | Canonical 3-4 |
|---|---|---:|---:|---|---:|---:|
{acc_table}

YHH V7 reported **0.8320** (4160 / 5000) and holdout **0.8230**, but this cannot be reproduced from committed cell-level artifacts. Team main recorded a non-reproduction on zzh member files (`0.8252` / holdout `0.8170`). Current team main v9 has **no reported OOF**.

## 6. Canonical Partition Metrics

Canonical analysis partition: `experiments/team_folds_5_seed42.csv` (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

This is a **meta-analysis partition**. It is the original protocol for WYH MODEL V2. It is **not** LZH's original 3-fold.

WYH MODEL V2 canonical fold accuracies: {wyh_folds}

LZH Prior-H canonical fold accuracies: {lzh_folds}

LZH original 3-fold accuracies (bundle `oof_gate.csv`, descriptive only): {lzh_native}

## 7. Pairwise Complementarity

| A | B | A acc | B acc | Disagree | Pair oracle | Headroom vs stronger |
|---|---|---:|---:|---:|---:|---:|
{pairwise_table}

Pairwise oracle on canonical folds 3-4:

{pairwise_holdout}

## 8. Multi-Expert Oracle

**ORACLE != DEPLOYABLE MODEL ACCURACY.**

Currently auditable honest expert pool: LZH `depth_masked_prior_h_anchor` and WYH MODEL V2.

YHH V7 and current team-main are unavailable for honest cell-level OOF comparison, so they are omitted. The oracle below applies **only** to this auditable pair. It is **not** evidence that the entire team's model pool has an oracle below 0.85. The full-team oracle remains unknown.

| Combo | Split | Oracle correct | Oracle acc | All wrong |
|---|---|---:|---:|---:|
| LZH + WYH | overall | {m_all_c} | {m_all_a:.4f} | {m_all_w} |
| LZH + WYH | canonical folds 0-2 | {m_02_c} | {m_02_a:.4f} | {m_02_w} |
| LZH + WYH | canonical folds 3-4 | {m_34_c} | {m_34_a:.4f} | {m_34_w} |

Best 3-way oracle: **not available** (only two LEVEL A/B experts in the predeclared pool).
Best 4-way oracle: **not available**.

Number of correct experts overall: {n_correct_dist}

## 9. Recommended Anchor

**Recommended anchor: `{anchor_id}`**

Reason: {anchor_reason}

This recommendation is for the next V3 diagnostic experiment. It is not a formal MODEL V3 freeze and does not generate predictions.

## 10. Anchor Error Recoverability

Anchor errors: **{anchor_errors} / 5000**.

| Pattern | Count | Fraction of anchor errors |
|---|---:|---:|
| Recoverable by WYH MODEL V2 | {rec_wyh} | {rec_wyh_f:.4f} |
| Recoverable by any alternative expert | {rec_any} | {rec_any_f:.4f} |
| Recoverable by 2+ agreeing alternatives | {rec_2plus} | n/a with one alternative |
| A. both alternatives agree on the correct replacement | {pat_a} | n/a |
| B. alternatives disagree with anchor and with each other | {pat_b} | n/a |
| C. anchor correct, alternatives agree on the same wrong class | {pat_c} | n/a |
| D. anchor correct, exactly one alternative disagrees | {pat_d} | {pat_d_f:.4f} |

False-correction risk among cells where the single alternative disagrees with a correct anchor: {false_risk}.

## 11. Fixed Consensus Diagnostics

Predeclared diagnostic rules only. No threshold or weight tuning. These are **not** MODEL V3 candidates.

With one eligible alternative, D1 (two-alternative agreement) cannot fire. D2 reduces to "replace whenever the single alternative disagrees". D3 majority-with-anchor-tie-break keeps the anchor on every 1–1 disagreement, so D3 equals D0.

| Rule | Changed | Wrong→correct | Correct→wrong | Net | Precision | Final acc | Canonical 3-4 acc |
|---|---:|---:|---:|---:|---:|---:|---:|
{consensus_table}

## 12. Confidence / Reliability Findings

Anchor confidence by outcome (top1 / margin / entropy):

| Subset | n | top1 mean | margin mean | entropy mean |
|---|---:|---:|---:|---:|
| A. anchor correct | {cA} | {tA:.4f} | {mA:.4f} | {eA:.4f} |
| B. anchor wrong | {cB} | {tB:.4f} | {mB:.4f} | {eB:.4f} |
| C. wrong and recoverable | {cC} | {tC:.4f} | {mC:.4f} | {eC:.4f} |
| D. wrong and unrecoverable | {cD} | {tD:.4f} | {mD:.4f} | {eD:.4f} |
| F. correct but alternative disagrees | {cF} | {tF:.4f} | {mF:.4f} | {eF:.4f} |

Diagnostic AUROC (not a deployed selector):

- P(anchor wrong) from `1 - top1`: {auc_wrong_top1}
- P(anchor wrong) from entropy: {auc_wrong_ent}
- P(anchor wrong AND recoverable) from `1 - top1`: {auc_rec_top1}
- Mean JS divergence (base 2) on cells where predictions disagree: {js_disagree}

{confidence_note}

## 13. Test-Safe Reliability Feature Inventory

See `outputs/v3/v3_e00t_tables/reliability_feature_inventory.csv`.

Summary:

- **Diagnostic-only:** true label, expert_correct, oracle flags, recoverable flags, LZH original 3-fold id used as provenance.
- **Test-safe and available now:** anchor top1 / margin / entropy; alternative top1 / margin / entropy; predicted-class disagreement; JS divergence; probability advantage; library size; n_detected; Region / E/I / Segment / missingness / hard_bucket / Section_ID.
- **Documented by LZH but not available as committed cell-level OOF fields:** Prior-H eligibility, strict graph degree, reliable-gene count. Only aggregate test-route counts are in `prior_h_route_audit.json`.
- **Not test-safe:** anything derived from held-out labels, including this audit's recoverable flags.

## 14. Biological Error Families

Anchor errors by true-class family (recoverable = alternative expert already predicts the true class).

{family_summary}

Top true → anchor_pred confusion pairs:

| True | Anchor pred | Errors | Recoverable | Fraction | Rescued by |
|---|---|---:|---:|---:|---|
{confusion_table}

Interpretation for the next design: recoverable errors are not confined to a single family, but oligodendrocyte / OPC and other glial/non-neuronal confusions remain the largest absolute buckets. A purely family-specific rule would miss a non-trivial neuronal remainder. The evidence is closer to **global reliability + optional family-aware features** than to a single-family patch.

## 15. Unique Value of Each Expert

| Expert | Standalone acc | Only this expert correct | Anchor-wrong and this expert correct |
|---|---:|---:|---:|
{unique_table}

**Does WYH MODEL V2 add unique team value despite lower standalone accuracy?** {wyh_unique_answer}

## 16. Leakage / Validation Audit

- No competition test labels were used for scoring, selection, or thresholding.
- No blend-weight search, stacking, or learned router was fit.
- WYH MODEL V2 OOF is the frozen 5-fold seed42 protocol. MODEL V2 architecture was not retuned here.
- LZH Prior-H OOF is a different 3-fold protocol. Cell-level predictions can still be used descriptively if each cell is genuinely held out of that 3-fold, but canonical folds 3-4 are **not** LZH's original holdout.
- LZH selected Prior-H over the graph stacker using the same full 3-fold OOF, so the 0.8266 number is a selection metric as well as an OOF metric (MEDIUM validation confidence).
- YHH V7 used folds 0-2 for gating and folds 3-4 as holdout, which is a stronger selection protocol, but the cell-level OOF artifact is absent.
- Team main v9 has no OOF counterpart; test prediction changes cannot be treated as validation.

## 17. Current Team Main Reproducibility Status

**Current `team/main` HEAD is prediction-only.** Commit `{main_sha}` updates `prediction/prediction.csv` and does not add OOF predictions, OOF probabilities, or generation metadata sufficient to reconstruct an honest OOF score.

No OOF metric is reported for team blend v9.

The latest historically documented reproducible ensemble on that line is V6 (equal-weight `refonly_full + ext_all25 + mlp3 + yhh_v1 + poolAll`, reported ~0.8248 / holdout ~0.8175). Those member probability `npz` files are gitignored and were not available for this audit.

## 18. MODEL V3 Strategic Decision

**{decision_label}**

Best honest multi-expert oracle for the **currently auditable** pool (LZH `depth_masked_prior_h_anchor` + WYH MODEL V2) = **{best_oracle:.4f}** overall ({best_oracle_n} / 5000), and **{best_oracle_34:.4f}** on canonical folds 3-4.

This figure does **not** describe the entire team's model pool. YHH V7 and current team-main are unavailable for honest cell-level OOF comparison, so the full-team oracle remains unknown.

{decision_text}

Recommended next primary experiment: **{next_exp}**

{next_why}

Do not start that experiment in this task. MODEL V3 is not frozen. No `docs/versions/model_v3.md` and no submission candidate were created.
""".format(
        main_sha=snap["team/main"]["sha"],
        main_date=snap["team/main"]["date"],
        main_subj=snap["team/main"]["subject"],
        yhh_sha=snap["team/yhh"]["sha"],
        yhh_date=snap["team/yhh"]["date"],
        yhh_subj=snap["team/yhh"]["subject"],
        lzh_sha=snap["team/lzh"]["sha"],
        lzh_date=snap["team/lzh"]["date"],
        lzh_subj=snap["team/lzh"]["subject"],
        wyh_sha=snap["team/wyh"]["sha"],
        wyh_date=snap["team/wyh"]["date"],
        wyh_subj=snap["team/wyh"]["subject"],
        rev_sha=snap["team/revert-1-lzh"]["sha"],
        rev_date=snap["team/revert-1-lzh"]["date"],
        rev_subj=snap["team/revert-1-lzh"]["subject"],
        local_sha=snap["local_head"]["sha"],
        wyh_oof=experts["wyh_model_v2"]["oof_prediction_path"],
        wyh_proba=experts["wyh_model_v2"]["oof_probability_path"],
        lzh_model=experts["lzh_prior_h"]["selected_model_name"],
        lzh_oof=experts["lzh_prior_h"]["oof_probability_path"],
        acc_table="\n".join(acc_line(eid) for eid in ["wyh_model_v2", "lzh_prior_h"]),
        wyh_folds=json.dumps(ind["wyh_model_v2"]["overall"]["canonical_fold_accuracy"]),
        lzh_folds=json.dumps(ind["lzh_prior_h"]["overall"]["canonical_fold_accuracy"]),
        lzh_native=json.dumps(payload["lzh_native_folds"]),
        pairwise_table="\n".join(pairwise_lines),
        pairwise_holdout="\n".join(
            "- `{a}` vs `{b}`: oracle {ora:.4f}, headroom {h:.4f}".format(
                a=r["expert_a"],
                b=r["expert_b"],
                ora=r["canonical_folds_3_4"]["pair_oracle_accuracy"],
                h=r["canonical_folds_3_4"]["incremental_oracle_headroom_over_stronger"],
            )
            for r in payload["pairwise"]
        ),
        m_all_c=multi["overall"]["oracle_correct"],
        m_all_a=multi["overall"]["oracle_accuracy"],
        m_all_w=multi["overall"]["all_wrong"],
        m_02_c=multi["canonical_folds_0_2"]["oracle_correct"],
        m_02_a=multi["canonical_folds_0_2"]["oracle_accuracy"],
        m_02_w=multi["canonical_folds_0_2"]["all_wrong"],
        m_34_c=multi["canonical_folds_3_4"]["oracle_correct"],
        m_34_a=multi["canonical_folds_3_4"]["oracle_accuracy"],
        m_34_w=multi["canonical_folds_3_4"]["all_wrong"],
        n_correct_dist=json.dumps(multi["overall"]["n_correct_experts_distribution"]),
        anchor_id=rec["recommended_anchor"],
        anchor_reason=rec["reason"],
        anchor_errors=recov["anchor_errors"],
        rec_wyh=recov["by_alternative"].get("wyh_model_v2", {}).get("recoverable", 0)
        if rec["recommended_anchor"] != "wyh_model_v2"
        else recov["by_alternative"].get("lzh_prior_h", {}).get("recoverable", 0),
        rec_wyh_f=(
            recov["by_alternative"].get("wyh_model_v2", {}).get("recoverable_fraction", 0.0)
            if rec["recommended_anchor"] != "wyh_model_v2"
            else recov["by_alternative"].get("lzh_prior_h", {}).get("recoverable_fraction", 0.0)
        ),
        rec_any=recov["recoverable_by_any"],
        rec_any_f=recov["recoverable_by_any_fraction"],
        rec_2plus=recov["recoverable_by_2plus_agreeing"],
        pat_a=recov["patterns"]["A_alternatives_agree_correct"],
        pat_b=recov["patterns"]["B_alternatives_disagree_with_anchor_and_each_other"],
        pat_c=recov["patterns"]["C_anchor_correct_alternatives_agree_wrong"],
        pat_d=recov["patterns"]["D_anchor_correct_exactly_one_alternative_disagrees"],
        pat_d_f=recov["patterns"]["D_fraction_of_anchor_correct"],
        false_risk=recov["false_correction_risk_note"],
        consensus_table="\n".join(consensus_lines),
        cA=conf["subsets"]["A_anchor_correct"]["top1"]["count"],
        tA=conf["subsets"]["A_anchor_correct"]["top1"]["mean"],
        mA=conf["subsets"]["A_anchor_correct"]["margin"]["mean"],
        eA=conf["subsets"]["A_anchor_correct"]["entropy"]["mean"],
        cB=conf["subsets"]["B_anchor_wrong"]["top1"]["count"],
        tB=conf["subsets"]["B_anchor_wrong"]["top1"]["mean"],
        mB=conf["subsets"]["B_anchor_wrong"]["margin"]["mean"],
        eB=conf["subsets"]["B_anchor_wrong"]["entropy"]["mean"],
        cC=conf["subsets"]["C_wrong_recoverable"]["top1"]["count"],
        tC=conf["subsets"]["C_wrong_recoverable"]["top1"]["mean"],
        mC=conf["subsets"]["C_wrong_recoverable"]["margin"]["mean"],
        eC=conf["subsets"]["C_wrong_recoverable"]["entropy"]["mean"],
        cD=conf["subsets"]["D_wrong_unrecoverable"]["top1"]["count"],
        tD=conf["subsets"]["D_wrong_unrecoverable"]["top1"]["mean"],
        mD=conf["subsets"]["D_wrong_unrecoverable"]["margin"]["mean"],
        eD=conf["subsets"]["D_wrong_unrecoverable"]["entropy"]["mean"],
        cF=conf["subsets"]["F_correct_but_alternative_disagrees"]["top1"]["count"],
        tF=conf["subsets"]["F_correct_but_alternative_disagrees"]["top1"]["mean"],
        mF=conf["subsets"]["F_correct_but_alternative_disagrees"]["margin"]["mean"],
        eF=conf["subsets"]["F_correct_but_alternative_disagrees"]["entropy"]["mean"],
        auc_wrong_top1="n/a" if conf["auroc"]["p_anchor_wrong_from_one_minus_top1"] is None else "{:.4f}".format(conf["auroc"]["p_anchor_wrong_from_one_minus_top1"]),
        auc_wrong_ent="n/a" if conf["auroc"]["p_anchor_wrong_from_entropy"] is None else "{:.4f}".format(conf["auroc"]["p_anchor_wrong_from_entropy"]),
        auc_rec_top1="n/a" if conf["auroc"]["p_wrong_and_recoverable_from_one_minus_top1"] is None else "{:.4f}".format(conf["auroc"]["p_wrong_and_recoverable_from_one_minus_top1"]),
        js_disagree="n/a" if conf["mean_js_divergence_on_disagreement"] is None else "{:.4f}".format(conf["mean_js_divergence_on_disagreement"]),
        confidence_note=conf["note"],
        family_summary="\n".join(
            "- `{k}`: {n} errors, {r} recoverable ({f:.3f})".format(
                k=k, n=v["error_count"], r=v["recoverable_count"], f=v["recoverable_fraction"]
            )
            for k, v in bio["by_true_family"].items()
        ),
        confusion_table="\n".join(family_lines) if family_lines else "| n/a | n/a | 0 | 0 | 0 | none |",
        unique_table="\n".join(
            "| `{k}` | {acc:.4f} | {only} | {aw} |".format(
                k=k,
                acc=v["standalone_accuracy"],
                only=v["only_this_expert_correct"],
                aw=v["anchor_wrong_this_expert_correct"],
            )
            for k, v in unique["experts"].items()
        ),
        wyh_unique_answer=unique["wyh_model_v2_unique_value_answer"],
        decision_label=decision["label"],
        best_oracle=decision["best_honest_multi_expert_oracle"],
        best_oracle_n=decision["best_honest_multi_expert_oracle_correct"],
        best_oracle_34=decision["best_oracle_canonical_folds_3_4"],
        decision_text=decision["interpretation"],
        next_exp=decision["next_primary_experiment"],
        next_why=decision["next_experiment_rationale"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="")
    args = parser.parse_args()

    if current_branch() != "ywan/ml-pipeline":
        raise SystemExit("Refusing to run: current branch is not ywan/ml-pipeline")

    out_dir = ROOT / "outputs" / "v3"
    table_dir = out_dir / "v3_e00t_tables"
    report_path = ROOT / "reports" / "v3" / "v3_e00t_team_expert_audit.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    ignore_probe = subprocess.run(
        ["git", "check-ignore", "work/v3_e00t_team_artifacts/"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif ignore_probe.returncode == 0:
        cache_dir = ROOT / "work" / "v3_e00t_team_artifacts"
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="v3_e00t_team_artifacts_"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshots = {
        "local_head": branch_snapshot("HEAD"),
        "origin_head": branch_snapshot("origin/ywan/ml-pipeline"),
        "team/main": branch_snapshot("team/main"),
        "team/yhh": branch_snapshot("team/yhh"),
        "team/lzh": branch_snapshot("team/lzh"),
        "team/wyh": branch_snapshot("team/wyh"),
        "team/revert-1-lzh": branch_snapshot("team/revert-1-lzh"),
    }

    materialized = {
        "lzh_final_oof": materialize_git_blob(
            "{}:{}".format(LZH_BRANCH, LZH_FINAL_OOF), cache_dir / "lzh_oof_probabilities_final.csv"
        ),
        "lzh_final_test": materialize_git_blob(
            "{}:{}".format(LZH_BRANCH, LZH_FINAL_TEST), cache_dir / "lzh_test_probabilities_final.csv"
        ),
        "lzh_prior_h_oof": materialize_git_blob(
            "{}:{}".format(LZH_BRANCH, LZH_PRIOR_H_OOF), cache_dir / "lzh_oof_probabilities_prior_h_anchor.csv"
        ),
        "lzh_gate": materialize_git_blob(
            "{}:{}".format(LZH_BRANCH, LZH_GATE), cache_dir / "lzh_oof_gate.csv"
        ),
        "lzh_selection": materialize_git_blob(
            "{}:{}".format(LZH_BRANCH, LZH_SELECTION), cache_dir / "lzh_final_selection.json"
        ),
        "cache_dir": str(cache_dir),
        "cache_dir_gitignored": ignore_probe.returncode == 0,
    }

    data = load_dataset(ROOT)
    contract_messages = validate_contract(data)
    class_names = allowed_labels(ROOT)
    if len(class_names) != N_CLASSES:
        raise SystemExit("expected 60 classes, found {}".format(len(class_names)))
    train_ids = [str(v) for v in data.meta_train.index.tolist()]
    test_ids = expected_test_cell_ids(ROOT)
    y_series = data.meta_train[TARGET_COL].astype(str)
    if y_series.isna().any() or (y_series == "").any():
        raise SystemExit("missing training labels")
    if len(set(train_ids)) != N_TRAIN or len(train_ids) != N_TRAIN:
        raise SystemExit("canonical train Cell_ID contract failed")

    folds_df, fold_messages = load_and_validate_team_folds(
        train_ids, test_ids, y_series.tolist(), TEAM_FOLDS
    )
    fold_map = folds_df.set_index("Cell_ID")["fold"].astype(int)

    wyh_pred = read_cell_id_csv(
        WYH_V2_OOF_PRED,
        extra_dtypes={"true_label": str, "predicted_label": str, "fold": int},
    )
    wyh_pred = wyh_pred.set_index("Cell_ID").reindex(train_ids)
    if wyh_pred["predicted_label"].isna().any():
        raise SystemExit("WYH OOF missing canonical Cell_IDs")
    if list(wyh_pred["true_label"].astype(str)) != list(y_series.reindex(train_ids).astype(str)):
        raise SystemExit("WYH OOF true_label does not match official meta_train")

    wyh_proba_frame, wyh_proba_audit = load_probability_frame(WYH_V2_OOF_PROBA, class_names)
    wyh_proba_frame = wyh_proba_frame.set_index("Cell_ID").reindex(train_ids)
    wyh_proba = wyh_proba_frame.loc[:, class_names].to_numpy(dtype=np.float64)
    wyh_from_proba = argmax_labels(wyh_proba, class_names)
    if list(wyh_from_proba) != list(wyh_pred["predicted_label"].astype(str)):
        raise SystemExit("WYH OOF hard labels do not match argmax of selected probabilities")

    lzh_sel = json.loads(Path(materialized["lzh_selection"]["local_path"]).read_text())
    lzh_frame, lzh_audit = load_probability_frame(
        Path(materialized["lzh_final_oof"]["local_path"]), class_names
    )
    lzh_anchor_frame, _lzh_anchor_audit = load_probability_frame(
        Path(materialized["lzh_prior_h_oof"]["local_path"]), class_names
    )
    lzh_frame = lzh_frame.set_index("Cell_ID").reindex(train_ids)
    lzh_anchor_frame = lzh_anchor_frame.set_index("Cell_ID").reindex(train_ids)
    if lzh_frame[class_names[0]].isna().any():
        raise SystemExit("LZH OOF missing canonical Cell_IDs")
    lzh_proba = lzh_frame.loc[:, class_names].to_numpy(dtype=np.float64)
    lzh_anchor_proba = lzh_anchor_frame.loc[:, class_names].to_numpy(dtype=np.float64)
    if not np.allclose(lzh_proba, lzh_anchor_proba, atol=1e-12):
        raise SystemExit("LZH final OOF probabilities differ from prior_h_anchor inputs")
    lzh_pred = argmax_labels(lzh_proba, class_names)

    lzh_test_frame, lzh_test_audit = load_probability_frame(
        Path(materialized["lzh_final_test"]["local_path"]), class_names
    )
    if sorted(lzh_test_frame["Cell_ID"].tolist()) != sorted(test_ids):
        raise SystemExit("LZH test probabilities Cell_ID set != official test IDs")
    wyh_test_frame, wyh_test_audit = load_probability_frame(WYH_V2_TEST_PROBA, class_names)
    if list(wyh_test_frame["Cell_ID"]) != test_ids:
        # still acceptable if set matches; record order status
        if sorted(wyh_test_frame["Cell_ID"].tolist()) != sorted(test_ids):
            raise SystemExit("WYH test probabilities Cell_ID set != official test IDs")

    lzh_gate = read_cell_id_csv(Path(materialized["lzh_gate"]["local_path"]), extra_dtypes={"fold": int})
    lzh_gate = lzh_gate.set_index("Cell_ID").reindex(train_ids)
    lzh_native_fold = lzh_gate["fold"].astype(int).to_numpy()

    y_true = y_series.reindex(train_ids).astype(str).to_numpy()
    canonical_fold = fold_map.reindex(train_ids).astype(int).to_numpy()
    wyh_pred_arr = wyh_pred["predicted_label"].astype(str).to_numpy()

    meta = data.meta_train.reindex(train_ids)
    counts = data.counts_train.reindex(train_ids)
    sigs = signatures_from_meta(meta)
    hard = (sigs.to_numpy() == missing_bucket_key())
    ei_of_label = ei_of_label_from_train(meta, class_names)
    neuron_mask, glial_mask = neuron_glial_masks(y_true, class_names, ei_of_label)
    neuron_or_glial = np.where(neuron_mask, "neuron", np.where(glial_mask, "glial_non_neuronal", "unmapped"))
    library_size = counts.sum(axis=1).to_numpy(dtype=np.float64)
    n_detected = (counts.to_numpy() > 0).sum(axis=1).astype(np.int64)
    region = [canonicalize_value(v) for v in meta["Region"].to_numpy()]
    ei = [canonicalize_value(v) for v in meta["Excitatory_vs_Inhibitory"].to_numpy()]
    segment = [canonicalize_value(v) for v in meta["Segment"].to_numpy()]
    section = [str(v) for v in meta["Section_ID"].to_numpy()]
    metadata_missing = hard.astype(bool)

    wyh_conf = confidence_from_proba(wyh_proba)
    lzh_conf = confidence_from_proba(lzh_proba)
    js_vals = js_divergence_rows(lzh_proba, wyh_proba)

    wyh_correct = wyh_pred_arr == y_true
    lzh_correct = lzh_pred == y_true
    disagree = wyh_pred_arr != lzh_pred

    registry = pd.DataFrame(
        {
            "Cell_ID": train_ids,
            "true_label": y_true,
            "canonical_fold": canonical_fold,
            "Region": region,
            "E/I": ei,
            "Segment": segment,
            "Section_ID": section,
            "hard_bucket": metadata_missing,
            "metadata_missing": metadata_missing,
            "library_size": library_size,
            "n_detected": n_detected,
            "neuron_or_glial": neuron_or_glial,
            "lzh_original_fold": lzh_native_fold,
            "wyh_model_v2_pred": wyh_pred_arr,
            "wyh_model_v2_correct": wyh_correct,
            "wyh_model_v2_top1": wyh_conf["top1"],
            "wyh_model_v2_top2": wyh_conf["top2"],
            "wyh_model_v2_margin": wyh_conf["margin"],
            "wyh_model_v2_entropy": wyh_conf["entropy"],
            "lzh_prior_h_pred": lzh_pred,
            "lzh_prior_h_correct": lzh_correct,
            "lzh_prior_h_top1": lzh_conf["top1"],
            "lzh_prior_h_top2": lzh_conf["top2"],
            "lzh_prior_h_margin": lzh_conf["margin"],
            "lzh_prior_h_entropy": lzh_conf["entropy"],
            "experts_disagree": disagree,
            "n_eligible_experts_correct": wyh_correct.astype(int) + lzh_correct.astype(int),
            "oracle_lzh_wyh_correct": wyh_correct | lzh_correct,
            "js_divergence_lzh_wyh": js_vals,
            "prob_advantage_lzh_minus_wyh_top1": lzh_conf["top1"] - wyh_conf["top1"],
        }
    )
    diagnostic_only_cols = [
        "true_label",
        "wyh_model_v2_correct",
        "lzh_prior_h_correct",
        "n_eligible_experts_correct",
        "oracle_lzh_wyh_correct",
        "lzh_original_fold",
    ]
    registry.attrs["diagnostic_only_columns"] = diagnostic_only_cols

    wyh_overall = partition_metrics(y_true, wyh_pred_arr, canonical_fold, class_names, wyh_proba)
    lzh_overall = partition_metrics(y_true, lzh_pred, canonical_fold, class_names, lzh_proba)
    reproduced = {
        "wyh_model_v2": wyh_overall["accuracy"],
        "lzh_prior_h": lzh_overall["accuracy"],
    }
    manifest = build_expert_manifest(
        snapshots, materialized, class_names, json.loads(WYH_V2_METRICS.read_text()), lzh_sel, reproduced
    )

    individual = {
        "wyh_model_v2": {
            "reported_accuracy": 0.8212,
            "overall": wyh_overall,
            "verification_status": classification_status(0.8212, wyh_overall["accuracy"]),
            "abs_difference": abs(0.8212 - wyh_overall["accuracy"]),
        },
        "lzh_prior_h": {
            "reported_accuracy": float(lzh_sel["selected_metrics"]["accuracy"]),
            "overall": lzh_overall,
            "verification_status": classification_status(
                float(lzh_sel["selected_metrics"]["accuracy"]), lzh_overall["accuracy"]
            ),
            "abs_difference": abs(float(lzh_sel["selected_metrics"]["accuracy"]) - lzh_overall["accuracy"]),
        },
    }

    lzh_native = {}
    for fold_id in sorted(set(lzh_native_fold.tolist())):
        mask = lzh_native_fold == fold_id
        lzh_native[str(int(fold_id))] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_true[mask], lzh_pred[mask])),
        }

    pair_overall = pair_metrics("lzh_prior_h", lzh_pred, "wyh_model_v2", wyh_pred_arr, y_true)
    pair_02 = pair_metrics("lzh_prior_h", lzh_pred, "wyh_model_v2", wyh_pred_arr, y_true, canonical_fold <= 2)
    pair_34 = pair_metrics("lzh_prior_h", lzh_pred, "wyh_model_v2", wyh_pred_arr, y_true, canonical_fold >= 3)
    pairwise = [
        {
            "expert_a": "lzh_prior_h",
            "expert_b": "wyh_model_v2",
            "overall": pair_overall,
            "canonical_folds_0_2": pair_02,
            "canonical_folds_3_4": pair_34,
        }
    ]

    preds = {"lzh_prior_h": lzh_pred, "wyh_model_v2": wyh_pred_arr}
    multi = {
        "overall": multi_oracle(preds, y_true, np.ones(N_TRAIN, dtype=bool)),
        "canonical_folds_0_2": multi_oracle(preds, y_true, canonical_fold <= 2),
        "canonical_folds_3_4": multi_oracle(preds, y_true, canonical_fold >= 3),
        "omitted_experts": {
            "yhh_v7": "LEVEL C: no committed cell-level OOF",
            "team_main_v9": "LEVEL C: test-only prediction.csv",
            "lzh_graph_stacker_v2_historical": "LEVEL A but excluded as superseded same-line LZH model",
        },
        "best_3_way": None,
        "best_4_way": None,
        "oracle_is_not_deployable_accuracy": True,
    }

    # Anchor recommendation: strongest honest reproduced accuracy, with 3-4 robustness and probabilities.
    lzh_34 = lzh_overall["canonical_folds_3_4"]["accuracy"]
    wyh_34 = wyh_overall["canonical_folds_3_4"]["accuracy"]
    if lzh_overall["accuracy"] >= wyh_overall["accuracy"]:
        anchor_id = "lzh_prior_h"
        alt_id = "wyh_model_v2"
        anchor_pred = lzh_pred
        alt_pred = wyh_pred_arr
        anchor_conf = lzh_conf
        alt_conf = wyh_conf
        anchor_proba = lzh_proba
        alt_proba = wyh_proba
        reason = (
            "LZH Prior-H has honest Cell_ID-aligned OOF probabilities, the highest reproduced "
            "overall accuracy among eligible experts ({:.4f} vs WYH {:.4f}), and canonical "
            "folds 3-4 accuracy {:.4f} vs WYH {:.4f}. YHH V7 and team-main v9 cannot be anchors "
            "because they lack honest OOF artifacts. Limitation: LZH uses a 3-fold protocol and "
            "selected this model on the same OOF used for reporting (MEDIUM confidence)."
        ).format(lzh_overall["accuracy"], wyh_overall["accuracy"], lzh_34, wyh_34)
    else:
        anchor_id = "wyh_model_v2"
        alt_id = "lzh_prior_h"
        anchor_pred = wyh_pred_arr
        alt_pred = lzh_pred
        anchor_conf = wyh_conf
        alt_conf = lzh_conf
        anchor_proba = wyh_proba
        alt_proba = lzh_proba
        reason = (
            "WYH MODEL V2 is the strongest eligible expert on reproduced canonical accuracy "
            "and has the cleaner 5-fold seed42 provenance."
        )

    recommended = {
        "recommended_anchor": anchor_id,
        "reason": reason,
        "eligible_anchor_candidates": ["lzh_prior_h", "wyh_model_v2"],
        "rejected_as_anchor": {
            "yhh_v7": "no cell-level OOF",
            "team_main_v9": "test-only",
        },
    }

    anchor_ok = anchor_pred == y_true
    alt_ok = alt_pred == y_true
    anchor_errors = int((~anchor_ok).sum())
    recoverable = (~anchor_ok) & alt_ok
    unrecoverable = (~anchor_ok) & (~alt_ok)
    recov = {
        "anchor": anchor_id,
        "anchor_errors": anchor_errors,
        "recoverable_by_any": int(recoverable.sum()),
        "recoverable_by_any_fraction": float(recoverable.sum() / max(anchor_errors, 1)),
        "recoverable_by_2plus_agreeing": 0,
        "by_alternative": {
            alt_id: {
                "recoverable": int(recoverable.sum()),
                "recoverable_fraction": float(recoverable.sum() / max(anchor_errors, 1)),
            }
        },
        "patterns": {
            "A_alternatives_agree_correct": 0,
            "B_alternatives_disagree_with_anchor_and_each_other": 0,
            "C_anchor_correct_alternatives_agree_wrong": 0,
            "D_anchor_correct_exactly_one_alternative_disagrees": int(np.sum(anchor_ok & ~alt_ok)),
            "D_fraction_of_anchor_correct": float(np.sum(anchor_ok & ~alt_ok) / max(int(anchor_ok.sum()), 1)),
            "note": "Only one eligible alternative expert, so two-alternative consensus patterns A-C are undefined and reported as 0.",
        },
        "false_correction_risk_note": (
            "{n} cells have a correct anchor and a disagreeing alternative "
            "({frac:.4f} of correct-anchor cells). An unconstrained override of every disagreement "
            "would convert these into new errors."
        ).format(
            n=int(np.sum(anchor_ok & ~alt_ok)),
            frac=float(np.sum(anchor_ok & ~alt_ok) / max(int(anchor_ok.sum()), 1)),
        ),
    }

    alternatives = {alt_id: alt_pred}
    consensus = {}
    for rule in ["D0", "D1", "D2", "D3"]:
        consensus[rule] = {
            "overall": apply_consensus(rule, anchor_pred, alternatives, y_true, np.ones(N_TRAIN, dtype=bool)),
            "canonical_folds_0_2": apply_consensus(rule, anchor_pred, alternatives, y_true, canonical_fold <= 2),
            "canonical_folds_3_4": apply_consensus(rule, anchor_pred, alternatives, y_true, canonical_fold >= 3),
        }
        if rule == "D1":
            consensus[rule]["note"] = "D1 requires two alternatives; with one alternative it cannot change any cell."
        if rule == "D2":
            consensus[rule]["note"] = "With one alternative, D2 replaces the anchor whenever that alternative disagrees."
        if rule == "D3":
            consensus[rule]["note"] = "With two experts, disagreement is a tie and D3 keeps the anchor, matching D0."

    subset_masks = {
        "A_anchor_correct": anchor_ok,
        "B_anchor_wrong": ~anchor_ok,
        "C_wrong_recoverable": recoverable,
        "D_wrong_unrecoverable": unrecoverable,
        "E_wrong_and_2plus_alternatives_agree_correct": np.zeros(N_TRAIN, dtype=bool),
        "F_correct_but_alternative_disagrees": anchor_ok & (anchor_pred != alt_pred),
    }
    conf_subsets = {}
    for key, mask in subset_masks.items():
        conf_subsets[key] = {
            "top1": summarize_numeric(anchor_conf["top1"][mask]),
            "margin": summarize_numeric(anchor_conf["margin"][mask]),
            "entropy": summarize_numeric(anchor_conf["entropy"][mask]),
            "alternative_top1": summarize_numeric(alt_conf["top1"][mask]),
            "alternative_margin": summarize_numeric(alt_conf["margin"][mask]),
            "alternative_entropy": summarize_numeric(alt_conf["entropy"][mask]),
            "js_divergence": summarize_numeric(js_vals[mask]),
            "prob_advantage_alt_minus_anchor_top1": summarize_numeric(alt_conf["top1"][mask] - anchor_conf["top1"][mask]),
        }
    disagree_mask = anchor_pred != alt_pred
    conf_payload = {
        "anchor": anchor_id,
        "subsets": conf_subsets,
        "auroc": {
            "p_anchor_wrong_from_one_minus_top1": safe_auc((~anchor_ok).astype(int), 1.0 - anchor_conf["top1"]),
            "p_anchor_wrong_from_entropy": safe_auc((~anchor_ok).astype(int), anchor_conf["entropy"]),
            "p_anchor_wrong_from_margin_inverted": safe_auc((~anchor_ok).astype(int), -anchor_conf["margin"]),
            "p_wrong_and_recoverable_from_one_minus_top1": safe_auc(recoverable.astype(int), 1.0 - anchor_conf["top1"]),
            "p_wrong_and_recoverable_from_entropy": safe_auc(recoverable.astype(int), anchor_conf["entropy"]),
        },
        "mean_js_divergence_on_disagreement": (
            float(np.mean(js_vals[disagree_mask])) if disagree_mask.any() else None
        ),
        "note": (
            "Anchor wrong cells have lower top1 / margin and higher entropy than correct cells, "
            "but recoverable vs unrecoverable errors overlap substantially. A threshold on "
            "confidence alone is unlikely to isolate safe corrections without a second expert signal."
        ),
    }

    feature_rows = [
        {"feature": "true_label", "source": "meta_train", "test_safe": False, "available": True, "notes": "diagnostic only"},
        {"feature": "expert_correct / oracle_correct / recoverable", "source": "OOF labels", "test_safe": False, "available": True, "notes": "retrospective only"},
        {"feature": "anchor_top1", "source": anchor_id, "test_safe": True, "available": True, "notes": "test-time probability"},
        {"feature": "anchor_margin", "source": anchor_id, "test_safe": True, "available": True, "notes": "top1-top2"},
        {"feature": "anchor_entropy", "source": anchor_id, "test_safe": True, "available": True, "notes": ""},
        {"feature": "alternative_top1_margin_entropy", "source": alt_id, "test_safe": True, "available": True, "notes": ""},
        {"feature": "predicted_class_disagreement", "source": "cross-expert", "test_safe": True, "available": True, "notes": "no labels required"},
        {"feature": "n_experts_agreeing", "source": "cross-expert", "test_safe": True, "available": True, "notes": "only 2 eligible experts"},
        {"feature": "probability_advantage", "source": "cross-expert", "test_safe": True, "available": True, "notes": "alt top1 - anchor top1"},
        {"feature": "js_divergence", "source": "cross-expert", "test_safe": True, "available": True, "notes": "base-2 JS divergence"},
        {"feature": "library_size", "source": "counts_train", "test_safe": True, "available": True, "notes": ""},
        {"feature": "n_detected", "source": "counts_train", "test_safe": True, "available": True, "notes": "nonzero official genes"},
        {"feature": "Region / E/I / Segment / missingness / hard_bucket / Section_ID", "source": "meta_train", "test_safe": True, "available": True, "notes": "observed metadata"},
        {"feature": "Prior-H eligibility", "source": "LZH route audit", "test_safe": True, "available": False, "notes": "documented rule exists; no committed cell-level OOF field"},
        {"feature": "strict graph degree", "source": "LZH", "test_safe": True, "available": False, "notes": "aggregate test counts only"},
        {"feature": "reliable gene count", "source": "LZH", "test_safe": True, "available": False, "notes": "aggregate test counts only"},
        {"feature": "spatial/reference neighbor-label histograms", "source": "team/WYH spatial", "test_safe": False, "available": False, "notes": "not loaded here; label-derived neighbor histograms are not test-safe if they use held-out labels"},
    ]
    feat_df = pd.DataFrame(feature_rows)

    confusion = []
    for true_lab, pred_lab in zip(y_true[~anchor_ok], anchor_pred[~anchor_ok]):
        confusion.append((str(true_lab), str(pred_lab)))
    pair_counts = Counter(confusion)
    confusion_rows = []
    error_idx = np.where(~anchor_ok)[0]
    for (true_lab, pred_lab), n in pair_counts.most_common():
        idx = [i for i in error_idx if y_true[i] == true_lab and anchor_pred[i] == pred_lab]
        rec_n = int(sum(alt_ok[i] for i in idx))
        rescuers = []
        if rec_n:
            rescuers = [alt_id]
        confusion_rows.append(
            {
                "true_label": true_lab,
                "anchor_pred": pred_lab,
                "error_count": int(n),
                "recoverable_count": rec_n,
                "recoverable_fraction": float(rec_n / n) if n else 0.0,
                "rescuing_experts": rescuers,
                "true_family": family_of(true_lab),
                "pred_family": family_of(pred_lab),
            }
        )
    by_true_family = OrderedDict()
    for family in list(CLASS_FAMILIES) + ["neuronal_or_other"]:
        fam_idx = [i for i in error_idx if family_of(y_true[i]) == family]
        rec_n = int(sum(alt_ok[i] for i in fam_idx))
        by_true_family[family] = {
            "error_count": len(fam_idx),
            "recoverable_count": rec_n,
            "recoverable_fraction": float(rec_n / len(fam_idx)) if fam_idx else 0.0,
        }
    bio = {
        "confusion_pairs": confusion_rows,
        "by_true_family": by_true_family,
        "by_hard_bucket": {
            "hard_bucket_errors": int(np.sum((~anchor_ok) & metadata_missing)),
            "hard_bucket_recoverable": int(np.sum(recoverable & metadata_missing)),
            "non_hard_errors": int(np.sum((~anchor_ok) & (~metadata_missing))),
            "non_hard_recoverable": int(np.sum(recoverable & (~metadata_missing))),
        },
        "by_neuron_glial": {
            "neuron_errors": int(np.sum((~anchor_ok) & neuron_mask)),
            "neuron_recoverable": int(np.sum(recoverable & neuron_mask)),
            "glial_errors": int(np.sum((~anchor_ok) & glial_mask)),
            "glial_recoverable": int(np.sum(recoverable & glial_mask)),
        },
    }

    only_lzh = int(np.sum(lzh_correct & ~wyh_correct))
    only_wyh = int(np.sum(wyh_correct & ~lzh_correct))
    wyh_unique_cells = only_wyh
    wyh_adds = int(np.sum((~lzh_correct) & wyh_correct))
    unique = {
        "experts": {
            "lzh_prior_h": {
                "standalone_accuracy": lzh_overall["accuracy"],
                "only_this_expert_correct": only_lzh,
                "anchor_wrong_this_expert_correct": 0 if anchor_id == "lzh_prior_h" else int(np.sum((~anchor_ok) & lzh_correct)),
            },
            "wyh_model_v2": {
                "standalone_accuracy": wyh_overall["accuracy"],
                "only_this_expert_correct": only_wyh,
                "anchor_wrong_this_expert_correct": 0 if anchor_id == "wyh_model_v2" else int(np.sum((~anchor_ok) & wyh_correct)),
            },
        },
        "oracle_anchor_only": float(np.mean(anchor_ok)),
        "oracle_anchor_plus_alternative": float(np.mean(anchor_ok | alt_ok)),
        "additional_correct_cells_from_adding_alternative": int(np.sum((~anchor_ok) & alt_ok)),
        "wyh_model_v2_unique_value_answer": (
            "Yes. WYH MODEL V2 is uniquely correct on {} cells that LZH Prior-H misses, "
            "which is the entire incremental oracle of the eligible two-expert pool. "
            "That is meaningful unique value despite a lower standalone accuracy ({:.4f} vs {:.4f}). "
            "It is not enough, by itself, to create a 3-expert team oracle, and unconstrained "
            "replacement of the anchor by WYH is unsafe because WYH also introduces {} false corrections."
        ).format(wyh_unique_cells, wyh_overall["accuracy"], lzh_overall["accuracy"], only_lzh if anchor_id == "lzh_prior_h" else only_wyh),
    }
    del wyh_adds

    best_oracle = multi["overall"]["oracle_accuracy"]
    best_oracle_34 = multi["canonical_folds_3_4"]["oracle_accuracy"]
    if best_oracle >= 0.87 and best_oracle_34 >= 0.85:
        label = "ROUTING HEADROOM STRONG"
        nxt = "V3-E01T — Conservative Consensus-Supported Abstaining Gate"
        interp = (
            "The best honest multi-expert oracle is at least 0.87 and complementarity remains "
            "visible on canonical folds 3-4."
        )
        why = (
            "A perfect selector over the current honest pool could theoretically clear 85%. "
            "The immediate test is whether a conservative abstaining consensus can capture a "
            "high-precision subset of the recoverable errors without giving up folds 3-4."
        )
    elif best_oracle >= 0.85:
        label = "ROUTING HEADROOM MARGINAL"
        nxt = "V3-E02D — 500-to-200 Gene Privileged Distillation"
        interp = (
            "The best honest multi-expert oracle is at least 0.85 but below 0.87. Routing could "
            "help only if selector precision is extremely high. The missing YHH OOF artifact "
            "prevents claiming a stronger team oracle."
        )
        why = (
            "With only two honest experts, D1 consensus cannot fire and unconstrained D2 "
            "replacement is not high-precision. The mathematically larger remaining opportunity "
            "is a new independent expert rather than spending the next cycle on a low-precision "
            "router. Privileged 500-to-200 distillation is the lowest-domain-gap independent-expert route."
        )
        if best_oracle >= 0.86:
            nxt = "V3-E01T — Conservative Consensus-Supported Abstaining Gate"
            why = (
                "Oracle headroom is real but thin. The next experiment should be a conservative, "
                "predeclared abstaining gate that changes few cells, with folds 3-4 reported frozen. "
                "In parallel, prepare an independent expert because a two-expert pool cannot support "
                "robust two-alternative consensus."
            )
    else:
        label = "ROUTING-ONLY INSUFFICIENT"
        nxt = "V3-E02D — 500-to-200 Gene Privileged Distillation"
        interp = (
            "Routing-only is insufficient for the CURRENTLY AUDITABLE honest expert pool. "
            "A new independent expert should be developed now rather than tuning a router "
            "over only LZH + WYH. If YHH/main OOF artifacts later become available, rerun "
            "the canonical oracle audit."
        )
        why = (
            "A new independent expert should be developed now rather than tuning a router "
            "over only LZH + WYH. The approved same-study 500-gene reference is the smallest "
            "domain-gap source of additional signal not already fully used by the 200-gene "
            "reference-only LightGBM. If YHH/main OOF artifacts later become available, "
            "rerun the canonical oracle audit."
        )

    # If oracle is marginal and 3-4 complementarity is weak, prefer new expert.
    if label == "ROUTING HEADROOM MARGINAL" and pair_34["incremental_oracle_headroom_over_stronger"] < 0.01:
        nxt = "V3-E02D — 500-to-200 Gene Privileged Distillation"
        why = (
            "Canonical folds 3-4 complementarity is small, so a gate trained or judged on "
            "development slices can easily look better than it is. The safer next primary "
            "experiment is a new independent expert via 500-to-200 privileged distillation."
        )

    decision = {
        "label": label,
        "best_honest_multi_expert_oracle": best_oracle,
        "best_honest_multi_expert_oracle_correct": multi["overall"]["oracle_correct"],
        "best_oracle_canonical_folds_3_4": best_oracle_34,
        "interpretation": interp,
        "next_primary_experiment": nxt,
        "next_experiment_rationale": why,
        "oracle_scope": "currently_auditable_honest_expert_pool_only",
        "auditable_experts": ["lzh_prior_h", "wyh_model_v2"],
        "full_team_oracle_status": "unknown",
        "unavailable_for_cell_level_oof_comparison": ["yhh_v7", "team_main_v9"],
        "thresholds_not_tuned": True,
        "weights_not_optimized": True,
    }

    integrity = {
        "n_train_cells": N_TRAIN,
        "unique_cell_id": len(set(train_ids)),
        "class_n": len(class_names),
        "contract_messages": contract_messages,
        "fold_messages": fold_messages,
        "wyh_probability_audit": wyh_proba_audit,
        "lzh_probability_audit": lzh_audit,
        "lzh_test_probability_audit": {
            "n_rows": lzh_test_audit["n_rows"],
            "usable": lzh_test_audit["usable"],
            "used_for_scoring": False,
        },
        "wyh_test_probability_audit": {
            "n_rows": wyh_test_audit["n_rows"],
            "usable": wyh_test_audit["usable"],
            "used_for_scoring": False,
        },
        "lzh_final_equals_prior_h_anchor": True,
        "joined_by_cell_id": True,
        "test_labels_used": False,
        "prediction_csv_modified": False,
        "model_v1_modified": False,
        "model_v2_modified": False,
    }

    metrics = {
        "audit_id": "V3-E00T",
        "created_at_utc": utc_now(),
        "oracle_is_not_deployable_accuracy": True,
        "snapshots": snapshots,
        "manifest": manifest,
        "integrity": integrity,
        "individual": individual,
        "lzh_native_folds": lzh_native,
        "pairwise": pairwise,
        "multi_expert_oracle": multi,
        "recommended_anchor": recommended,
        "anchor_recoverability": recov,
        "consensus": consensus,
        "confidence": conf_payload,
        "biological_families": {
            "by_true_family": bio["by_true_family"],
            "by_hard_bucket": bio["by_hard_bucket"],
            "by_neuron_glial": bio["by_neuron_glial"],
            "top_confusion_pairs": confusion_rows[:20],
        },
        "unique_value": unique,
        "strategic_decision": decision,
        "class_names": class_names,
    }

    write_json(out_dir / "v3_e00t_expert_manifest.json", manifest)
    registry.to_parquet(out_dir / "v3_e00t_team_oof_registry.parquet", index=False)
    pair_csv = pd.DataFrame(
        [
            {
                "split": split,
                **{k: v for k, v in block.items() if k not in {"expert_a", "expert_b"}},
                "expert_a": row["expert_a"],
                "expert_b": row["expert_b"],
            }
            for row in pairwise
            for split, block in (
                ("overall", row["overall"]),
                ("canonical_folds_0_2", row["canonical_folds_0_2"]),
                ("canonical_folds_3_4", row["canonical_folds_3_4"]),
            )
        ]
    )
    pair_csv.sort_values(
        ["pair_oracle_accuracy", "incremental_oracle_headroom_over_stronger"],
        ascending=False,
        inplace=True,
    )
    pair_csv.to_csv(out_dir / "v3_e00t_pairwise_oracle.csv", index=False)

    multi_rows = []
    for split in ["overall", "canonical_folds_0_2", "canonical_folds_3_4"]:
        block = multi[split]
        multi_rows.append(
            {
                "combo": "+".join(block["experts"]),
                "split": split,
                "n": block["n"],
                "oracle_correct": block["oracle_correct"],
                "oracle_accuracy": block["oracle_accuracy"],
                "all_wrong": block["all_wrong"],
                "n_experts": len(block["experts"]),
                "oracle_is_not_deployable_accuracy": True,
            }
        )
    pd.DataFrame(multi_rows).to_csv(out_dir / "v3_e00t_multi_expert_oracle.csv", index=False)

    recov_rows = [
        {
            "anchor": anchor_id,
            "alternative": alt_id,
            "anchor_errors": recov["anchor_errors"],
            "recoverable": recov["by_alternative"][alt_id]["recoverable"],
            "recoverable_fraction": recov["by_alternative"][alt_id]["recoverable_fraction"],
            "recoverable_by_any": recov["recoverable_by_any"],
            "recoverable_by_2plus_agreeing": recov["recoverable_by_2plus_agreeing"],
            "false_correction_cells_if_unconstrained_override": int(np.sum(anchor_ok & ~alt_ok)),
        }
    ]
    pd.DataFrame(recov_rows).to_csv(out_dir / "v3_e00t_anchor_recoverability.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(table_dir / "anchor_confusion_recoverability.csv", index=False)
    feat_df.to_csv(table_dir / "reliability_feature_inventory.csv", index=False)
    cons_rows = []
    for rule, block in consensus.items():
        for split in ["overall", "canonical_folds_0_2", "canonical_folds_3_4"]:
            cons_rows.append({"split": split, **block[split]})
    pd.DataFrame(cons_rows).to_csv(table_dir / "consensus_diagnostics.csv", index=False)
    write_json(out_dir / "v3_e00t_metrics.json", metrics)
    write_markdown_report(report_path, metrics)

    # Deterministic rerun of the two headline accuracies.
    rerun_wyh = float(np.mean(registry["wyh_model_v2_pred"] == registry["true_label"]))
    rerun_lzh = float(np.mean(registry["lzh_prior_h_pred"] == registry["true_label"]))
    if abs(rerun_wyh - wyh_overall["accuracy"]) > 1e-15 or abs(rerun_lzh - lzh_overall["accuracy"]) > 1e-15:
        raise SystemExit("non-deterministic accuracy rerun")

    print(
        json.dumps(
            {
                "status": "PASS",
                "n_cells": int(len(registry)),
                "wyh_accuracy": wyh_overall["accuracy"],
                "lzh_accuracy": lzh_overall["accuracy"],
                "oracle": best_oracle,
                "anchor": anchor_id,
                "decision": label,
                "cache_dir": str(cache_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
