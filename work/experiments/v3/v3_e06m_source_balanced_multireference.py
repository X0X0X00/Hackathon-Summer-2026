#!/usr/bin/env python3
"""V3-E06M: Source-Balanced Multi-Reference Transfer (SBMR).

Train two LightGBM candidates on cleaned MERFISH + cleaned SNI references
using the frozen V2-B-REFONLY recipe. M1 is naive pooling. M2 is the
predeclared source-balanced primary method.

Does not train a router, optimize source weights or ensemble weights,
add another dataset, start Spatial-ID, write prediction/prediction.csv,
or freeze MODEL V3.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WORK / "src"))
sys.path.insert(0, str(_WORK / "experiments" / "v3"))
from merfish60.io import repo_root  # noqa: E402
ROOT = repo_root()

from merfish60.ext_universe import build_or_load_ext_universe  # noqa: E402
from merfish60.io import TARGET_COL, load_dataset, validate_contract  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.neighbor_labels import (  # noqa: E402
    EX_K,
    SP_K,
    apply_ei,
    apply_segment_mask,
    build_X,
    label_hist,
    segment_allowed_from_reference,
    visible_ext_label_codes,
)
from merfish60.official_contract import allowed_labels, sha256_file  # noqa: E402
from merfish60.reference import (  # noqa: E402
    EXPECTED_MD5 as MERFISH_MD5,
    HISTORICAL_USABLE_ROWS,
    _row_fingerprints,
    audit_reference,
    md5_file,
    reference_h5ad_path,
)
from merfish60.spatial_features import (  # noqa: E402
    K_NBR,
    N_SPATIAL_MEAN,
    PCA_DIM,
    PCA_RANDOM_STATE,
    _as_knn_2d,
    ei_of_label_from_train,
)
from merfish60.team_cv import (  # noqa: E402
    TEAM_CV_PROTOCOL,
    TEAM_FOLD_VALUES,
    load_and_validate_team_folds,
)
from merfish60.v2_metrics import (  # noqa: E402
    hard_bucket_mask,
    json_default,
    neuron_glial_masks,
    slice_metrics,
    universe_fold_ids,
    write_json,
    write_proba,
)

from v3_e02d_privileged_gene_distillation import (  # noqa: E402
    classification_metrics,
    gene_list_sha256,
)
from v3_e03a_rescue_audit import CLASS_FAMILIES_E03A, family_of  # noqa: E402
from v3_e04s_sni_source_expert import (  # noqa: E402
    EXPECTED_SNI_MD5,
    audit_sni_dataset,
    fingerprint_overlap_mask,
    sni_h5ad_path,
)


EXPERIMENT_ID = "V3-E06M"
EXPERIMENT_CODENAME = "SBMR"
EXPECTED_BRANCH = "ywan/ml-pipeline"
EXPECTED_PYTHON_MARKER = "hackathon-v3"
N_TRAIN = 5000
N_TEST = 5000
N_CLASSES = 60
N_GENES = 200
OFFICIAL_GENE_SHA256 = "e3301724038990aa2db237026316aaa5fd265a11231c343bea733f8106ab06f5"
HISTORICAL_SNI_USABLE = 55193
HISTORICAL_MERFISH_USABLE = 136574
EXPECTED_COMBINED_BEFORE_EXTRA = 191767

LZH_CORRECT = 4133
WYH_CORRECT = 4106
S0_CORRECT = 3151
SNI_CORRECT = 2841
LZH_WYH_ORACLE = 4215
THREE_EXPERT_ORACLE = 4311
FOUR_EXPERT_ORACLE = 4364
ALL_FOUR_WRONG = 636
SNI_UNIQUE_RECOVERIES = 53
M0_ACCURACY = 0.8212
M0_MACRO_F1 = 0.793613323779027

LGBM_SEED = 0
NUM_BOOST_ROUND = 700
NUM_THREADS = 8
SOURCE_BALANCE = 0.5
WEIGHT_RATIO_ATOL = 1e-8
MEAN_WEIGHT_ATOL = 1e-10
PCA_REPRO_ATOL = 1e-4

STRONG_NET_VS_M0 = 25
STRONG_ACCURACY_MIN = 0.8262
STRONG_MACRO_F1_TOL = 0.002
STRONG_SNI_CAPTURE = 15
STRONG_NEW_UNIQUE = 15
PROMISING_SNI_CAPTURE = 10
PROMISING_NEW_UNIQUE = 5
MATERIAL_FOLDS34_REGRESSION_CELLS = -10
SUBSTANTIAL_MACRO_F1_LOSS = -0.01
MEANINGFUL_SNI_CAPTURE = 5

FAMILY_ORDER = list(CLASS_FAMILIES_E03A.keys()) + ["neuronal_or_other"]
FAMILY_DISPLAY = OrderedDict(
    [
        ("oligodendrocyte_opc", "oligodendrocyte / OPC"),
        ("astrocyte", "astrocyte"),
        ("vascular", "vascular / endothelial"),
        ("meningeal", "meningeal"),
        ("microglia", "microglia"),
        ("neuronal_or_other", "neuronal / other"),
        ("remaining_glial_non_neuronal", "remaining glial/non-neuronal"),
    ]
)

WORK_DIR = ROOT / "work" / "v3_e06m"
OUT_DIR = ROOT / "work" / "outputs" / "v3"
TABLE_DIR = OUT_DIR / "v3_e06m_tables"
REPORT_PATH = ROOT / "work" / "reports" / "v3" / "v3_e06m_source_balanced_multireference.md"
PRED_PATH = ROOT / "prediction" / "prediction.csv"
E05A_REGISTRY = OUT_DIR / "v3_e05a_four_expert_registry.parquet"
E04S_RECOVERIES = OUT_DIR / "v3_e04s_new_unique_recoveries.csv"
M0_OOF = ROOT / "work" / "outputs" / "oof" / "MODEL-V2_oof.csv"
M0_OOF_PROBA = ROOT / "work" / "outputs" / "probabilities" / "V2-B-REFONLY_oof_probabilities_seg.csv.gz"
M0_TEST_PROBA = ROOT / "work" / "outputs" / "probabilities" / "V2-B-REFONLY_test_probabilities_seg.csv.gz"
M0_METRICS = ROOT / "work" / "outputs" / "metrics" / "model_v2_metrics.json"

FORBIDDEN_FEATURE_TOKENS = (
    "reference_source",
    "source",
    "SNI",
    "MERFISH",
    "Condition",
    "condition",
)


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
            "V3-E06M must run with the isolated hackathon-v3 interpreter, not {}. "
            "Use /Users/yyl/venvs/hackathon-v3/bin/python.".format(sys.executable)
        )
    if sys.version_info[:2] < (3, 11):
        raise SystemExit(
            "V3-E06M refuses Python {} (frozen MODEL V1/V2 environment).".format(
                platform.python_version()
            )
        )


def assert_work_dir_ignored(path: Path) -> None:
    try:
        out = subprocess.check_output(
            ["git", "check-ignore", "-v", str(path)],
            cwd=str(ROOT),
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "work directory is not gitignored: {} ({})".format(
                path, exc.output.decode("utf-8") if exc.output else "not ignored"
            )
        )
    if "work/v3_e06m" not in out:
        raise RuntimeError("unexpected check-ignore output for {}: {}".format(path, out))


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


def source_balanced_weights(source: np.ndarray, base_weight: np.ndarray) -> np.ndarray:
    """Equal total source mass 0.5/0.5, then rescale to mean 1."""
    source = np.asarray(source)
    base = np.asarray(base_weight, dtype=np.float64)
    if source.shape[0] != base.shape[0]:
        raise ValueError("source and base_weight length mismatch")
    raw = np.zeros_like(base)
    totals = {}
    for name in ("MERFISH", "SNI"):
        mask = source == name
        total = float(base[mask].sum())
        if total <= 0:
            raise RuntimeError("source {} has zero base weight".format(name))
        totals[name] = total
        raw[mask] = base[mask] * (SOURCE_BALANCE / total)
    if abs(float(raw[source == "MERFISH"].sum()) - SOURCE_BALANCE) > 1e-12:
        raise RuntimeError("MERFISH raw mass is not 0.5")
    if abs(float(raw[source == "SNI"].sum()) - SOURCE_BALANCE) > 1e-12:
        raise RuntimeError("SNI raw mass is not 0.5")
    mean = float(raw.mean())
    if mean <= 0:
        raise RuntimeError("balanced raw weights have nonpositive mean")
    out = raw / mean
    mer_sum = float(out[source == "MERFISH"].sum())
    sni_sum = float(out[source == "SNI"].sum())
    if abs(mer_sum - sni_sum) > 1e-8:
        raise RuntimeError("M2 source masses diverged after mean-one rescale")
    if abs(float(out.mean()) - 1.0) > MEAN_WEIGHT_ATOL:
        raise RuntimeError("M2 mean sample weight is not 1")
    return out


def weight_summary(source: np.ndarray, base: np.ndarray, effective: np.ndarray) -> dict:
    source = np.asarray(source)
    base = np.asarray(base, dtype=np.float64)
    effective = np.asarray(effective, dtype=np.float64)
    payload = {
        "n": int(len(source)),
        "mean_weight": float(effective.mean()),
        "median_weight": float(np.median(effective)),
        "min_weight": float(effective.min()),
        "max_weight": float(effective.max()),
        "by_source": {},
    }
    for name in ("MERFISH", "SNI"):
        mask = source == name
        w = effective[mask]
        payload["by_source"][name] = {
            "count": int(mask.sum()),
            "total_base_weight": float(base[mask].sum()),
            "total_effective_weight": float(w.sum()),
            "min_weight": float(w.min()) if mask.any() else None,
            "max_weight": float(w.max()) if mask.any() else None,
            "mean_weight": float(w.mean()) if mask.any() else None,
            "median_weight": float(np.median(w)) if mask.any() else None,
        }
    mer = payload["by_source"]["MERFISH"]["total_effective_weight"]
    sni = payload["by_source"]["SNI"]["total_effective_weight"]
    payload["weight_sum_ratio_merfish_over_sni"] = float(mer / sni) if sni else None
    return payload


def cell_delta(pred_a, pred_b, true, mask: Optional[np.ndarray] = None) -> dict:
    pred_a = np.asarray(pred_a, dtype=object)
    pred_b = np.asarray(pred_b, dtype=object)
    true = np.asarray(true, dtype=object)
    if mask is None:
        mask = np.ones(len(true), dtype=bool)
    a = pred_a[mask]
    b = pred_b[mask]
    t = true[mask]
    a_ok = a == t
    b_ok = b == t
    changed = a != b
    return {
        "n": int(mask.sum()),
        "changed_predictions": int(changed.sum()),
        "wrong_to_correct": int((~a_ok & b_ok).sum()),
        "correct_to_wrong": int((a_ok & ~b_ok).sum()),
        "net_correction": int(b_ok.sum() - a_ok.sum()),
        "candidate_only_correct": int((~a_ok & b_ok).sum()),
        "m0_only_correct": int((a_ok & ~b_ok).sum()),
        "both_correct": int((a_ok & b_ok).sum()),
        "both_wrong": int((~a_ok & ~b_ok).sum()),
    }


def family_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> List[dict]:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    fam = np.array([family_of(str(v)) for v in y_true], dtype=object)
    rows = []
    for name in FAMILY_ORDER:
        mask = fam == name
        n = int(mask.sum())
        acc = float(np.mean(y_true[mask] == y_pred[mask])) if n else None
        rows.append(
            {
                "family": name,
                "family_display": FAMILY_DISPLAY.get(name, name),
                "n": n,
                "accuracy": acc,
            }
        )
    return rows


def split_metrics(y_true, y_pred, folds, proba, class_names) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    folds = np.asarray(folds)
    overall = classification_metrics(y_true, y_pred, class_names, proba)
    fold_acc = {}
    for fold_id in TEAM_FOLD_VALUES:
        mask = folds == fold_id
        fold_acc[str(fold_id)] = float(np.mean(y_true[mask] == y_pred[mask]))
    overall["fold_accuracy"] = fold_acc
    overall["folds_0_2_n"] = int((folds <= 2).sum())
    overall["folds_3_4_n"] = int((folds >= 3).sum())
    overall["folds_0_2_correct"] = int(np.sum((folds <= 2) & (y_true == y_pred)))
    overall["folds_3_4_correct"] = int(np.sum((folds >= 3) & (y_true == y_pred)))
    overall["folds_0_2"] = float(np.mean(y_true[folds <= 2] == y_pred[folds <= 2]))
    overall["folds_3_4"] = float(np.mean(y_true[folds >= 3] == y_pred[folds >= 3]))
    return overall


def classify_m2(
    net_vs_m0: int,
    acc: float,
    net_02: int,
    net_34: int,
    macro_f1: float,
    m0_macro_f1: float,
    m2_correct: int,
    m1_correct: int,
    sni_capture: int,
    new_unique: int,
    integrity_ok: bool,
) -> Tuple[str, str]:
    if not integrity_ok:
        return (
            "SOURCE-BALANCED INTEGRATION INSUFFICIENT",
            "Leakage, taxonomy, or source-weight contract failed.",
        )
    strong = (
        net_vs_m0 >= STRONG_NET_VS_M0
        and acc >= STRONG_ACCURACY_MIN
        and net_02 > 0
        and net_34 > 0
        and (macro_f1 >= m0_macro_f1 - STRONG_MACRO_F1_TOL)
        and m2_correct >= m1_correct
        and sni_capture >= STRONG_SNI_CAPTURE
        and new_unique >= STRONG_NEW_UNIQUE
    )
    if strong:
        return (
            "STRONG SOURCE-BALANCED TRANSFER",
            "M2 met every predeclared strong criterion versus frozen M0 and the M1 control.",
        )
    material_34 = net_34 <= MATERIAL_FOLDS34_REGRESSION_CELLS
    substantial_f1 = (macro_f1 - m0_macro_f1) <= SUBSTANTIAL_MACRO_F1_LOSS
    no_sni = sni_capture < MEANINGFUL_SNI_CAPTURE and new_unique < PROMISING_NEW_UNIQUE
    worse_than_m1 = m2_correct < m1_correct and sni_capture <= 0 and new_unique <= 0
    if net_vs_m0 <= 0 or material_34 or substantial_f1 or no_sni or worse_than_m1:
        reasons = []
        if net_vs_m0 <= 0:
            reasons.append("non-positive net correction versus M0")
        if material_34:
            reasons.append("material folds 3-4 regression")
        if substantial_f1:
            reasons.append("substantial macro-F1 loss")
        if no_sni:
            reasons.append("no meaningful SNI-source capture")
        if worse_than_m1:
            reasons.append("worse than naive M1 without a compensating benefit")
        return (
            "SOURCE-BALANCED INTEGRATION INSUFFICIENT",
            "M2 failed a major predeclared condition: {}.".format("; ".join(reasons)),
        )
    promising_signal = sni_capture >= PROMISING_SNI_CAPTURE or new_unique >= PROMISING_NEW_UNIQUE
    if net_vs_m0 > 0 and not material_34 and promising_signal:
        return (
            "PROMISING SOURCE-BALANCED TRANSFER",
            "M2 has positive overall net correction, folds 3-4 are not a material regression, "
            "and some SNI/new-unique signal transferred, but not all STRONG criteria were met.",
        )
    return (
        "SOURCE-BALANCED INTEGRATION INSUFFICIENT",
        "M2 did not meet the predeclared strong or promising source-balanced transfer criteria.",
    )


def next_action_for(label: str, m1_better: bool) -> str:
    if label == "STRONG SOURCE-BALANCED TRANSFER":
        return (
            "Freeze M2 as a candidate MODEL V3 base-model member and run a separate reviewed "
            "selection among currently auditable deployable experts. Do not train a router, "
            "tune source weights, or create model-v3 yet."
        )
    if label == "PROMISING SOURCE-BALANCED TRANSFER":
        return (
            "Keep M2 as a documented base-model candidate and next compare it against LZH Prior-H "
            "and frozen MODEL V2 under the same folds without blending or routing. Do not add "
            "another dataset or start Spatial-ID."
        )
    if m1_better:
        return (
            "Do not promote M1 automatically. Naive pooling outperformed explicit 0.5/0.5 "
            "balancing, so any pooling follow-up needs a separately reviewed decision. Do not "
            "tune source weights, add M3, or freeze MODEL V3 from this run."
        )
    return (
        "Do not use source-balanced multi-reference training as a MODEL V3 base. The next action "
        "is a reviewed selection among already frozen deployable experts (LZH Prior-H and WYH "
        "MODEL V2), not another reference-source experiment."
    )


def stop(code: str, message: str) -> int:
    print("{}: {}".format(code, message), flush=True)
    return 2


def nanmean_cols(matrix: np.ndarray, k: int) -> np.ndarray:
    return np.nanmean(np.where(np.isfinite(matrix[:, :k]), matrix[:, :k], np.nan), axis=1)


def combined_row_identity(source: np.ndarray, cell_ids: np.ndarray) -> str:
    lines = [
        "{}:{}".format(str(s), str(cid))
        for s, cid in zip(np.asarray(source), np.asarray(cell_ids))
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def reconstruct_merfish_pca(universe) -> Tuple[StandardScaler, PCA, dict]:
    lognorm = np.asarray(universe.X_static[:, :N_GENES], dtype=np.float32)
    scaler = StandardScaler()
    z = scaler.fit_transform(lognorm)
    pca = PCA(n_components=PCA_DIM, random_state=PCA_RANDOM_STATE)
    pcs = pca.fit_transform(z).astype(np.float32)
    cached = np.asarray(universe.X_static[:, N_GENES : N_GENES + PCA_DIM], dtype=np.float32)
    max_abs = float(np.max(np.abs(pcs - cached)))
    ok = bool(np.allclose(pcs, cached, atol=PCA_REPRO_ATOL, rtol=0))
    return scaler, pca, {
        "max_abs_pc_difference": max_abs,
        "atol": PCA_REPRO_ATOL,
        "reproduced": ok,
        "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
        "cached_pca_var_explained": float(universe.pca_var_explained),
    }


def expected_feature_names(genes: Sequence[str]) -> List[str]:
    names = (
        ["g_{}".format(g) for g in genes]
        + ["pc{}".format(i) for i in range(PCA_DIM)]
        + ["nbm{}".format(i) for i in range(PCA_DIM)]
        + [
            "log_total",
            "log_volume",
            "density",
            "x",
            "y",
            "rel_x",
            "rel_y",
            "sp_d1",
            "sp_d5",
            "sp_d15",
            "ex_d1",
            "ex_d10",
            "Region",
            "EI",
            "Segment",
            "AP",
            "gender",
            "mouse",
            "dataset",
        ]
    )
    names = names + ["sp_h{}".format(c) for c in range(N_CLASSES)] + ["sp_n", "sp_d"]
    names = names + ["ex_h{}".format(c) for c in range(N_CLASSES)] + ["ex_n", "ex_d"]
    return names


def feature_schema_ok(names: Sequence[str]) -> Tuple[bool, str]:
    names = [str(n) for n in names]
    lowered = [n.lower() for n in names]
    for token in FORBIDDEN_FEATURE_TOKENS:
        if token.lower() in lowered or any(token.lower() == n.lower() for n in names):
            return False, "forbidden feature token present: {}".format(token)
    if "reference_source" in names:
        return False, "reference_source used as a model input"
    return True, "ok"


def map_codes_or_nan(values: np.ndarray, categories) -> np.ndarray:
    codes = pd.Categorical(values, categories=categories).codes.astype(np.float32)
    codes[codes < 0] = np.nan
    return codes


def build_sni_v2_schema_features(
    x200: np.ndarray,
    obs: pd.DataFrame,
    keep: np.ndarray,
    y_codes: np.ndarray,
    scale: float,
    scaler: StandardScaler,
    pca: PCA,
    mer_mouse_categories,
    mer_dataset_categories,
    ap_map: dict,
) -> dict:
    """SNI rows in the frozen V2 feature schema. Graphs are SNI-internal."""
    obs_k = obs.loc[keep].copy() if isinstance(keep, pd.Index) else obs.iloc[np.where(keep)[0]]
    n = int(x200.shape[0])
    tot = x200.sum(axis=1).astype(np.float64)
    lognorm = np.log1p(x200 / np.maximum(tot, 1.0)[:, None] * float(scale)).astype(np.float32)
    pcs = pca.transform(scaler.transform(lognorm)).astype(np.float32)

    section_id = obs_k["Section ID"].astype(str).to_numpy()
    center_x = obs_k["center_x"].astype(float).to_numpy()
    center_y = obs_k["center_y"].astype(float).to_numpy()
    volume = obs_k["volume"].astype(float).to_numpy()
    mouse_id = obs_k["Mouse ID"].astype(str).to_numpy() if "Mouse ID" in obs_k.columns else np.array(["NA"] * n)
    datasets = obs_k["Datasets"].astype(str).to_numpy() if "Datasets" in obs_k.columns else np.array(["NA"] * n)
    axial = (
        obs_k["Axial level"].astype(str).replace("nan", np.nan).to_numpy()
        if "Axial level" in obs_k.columns
        else np.array([np.nan] * n, dtype=object)
    )

    region = np.full(n, np.nan, dtype=np.float32)
    ei = np.full(n, np.nan, dtype=np.float32)
    segment = np.full(n, np.nan, dtype=np.float32)
    gender = np.full(n, np.nan, dtype=np.float32)
    ap = pd.Series(axial).map(ap_map).to_numpy(dtype=np.float32)
    ap = ap - 1.0
    mouse = map_codes_or_nan(mouse_id, mer_mouse_categories)
    dsets = map_codes_or_nan(datasets, mer_dataset_categories)

    coords = np.column_stack([center_x, center_y])
    sp_idx = np.full((n, K_NBR), -1, dtype=np.int32)
    sp_dist = np.full((n, K_NBR), np.inf, dtype=np.float32)
    section_sizes: List[int] = []
    meta_tmp = pd.DataFrame({"Section_ID": section_id, "x": center_x, "y": center_y})
    for _sec, gi in meta_tmp.groupby("Section_ID").indices.items():
        gi = np.asarray(gi)
        section_sizes.append(int(len(gi)))
        k = min(K_NBR + 1, len(gi))
        tree = cKDTree(coords[gi])
        queried = tree.query(coords[gi], k=k, workers=min(8, NUM_THREADS))
        d, j = _as_knn_2d(queried[0], queried[1], n=len(gi), k=k)
        if k <= 1:
            continue
        sp_idx[gi, : k - 1] = gi[j[:, 1:]]
        sp_dist[gi, : k - 1] = d[:, 1:]

    tree = cKDTree(pcs)
    d, j = tree.query(pcs, k=K_NBR + 1, workers=min(8, NUM_THREADS))
    d, j = _as_knn_2d(d, j, n=n, k=K_NBR + 1)
    ex_idx = j[:, 1:].astype(np.int32)
    ex_dist = d[:, 1:].astype(np.float32)

    g = meta_tmp.groupby("Section_ID")
    sx = g["x"].transform("mean").to_numpy()
    sy = g["y"].transform("mean").to_numpy()
    sxs = g["x"].transform("std").to_numpy()
    sys_ = g["y"].transform("std").to_numpy()
    rel_x = ((center_x - sx) / np.maximum(sxs, 1.0)).astype(np.float32)
    rel_y = ((center_y - sy) / np.maximum(sys_, 1.0)).astype(np.float32)
    nb_mean = np.zeros((n, PCA_DIM), dtype=np.float32)
    nb_cnt = np.zeros(n, dtype=np.float32)
    for kk in range(N_SPATIAL_MEAN):
        v = sp_idx[:, kk]
        ok = v >= 0
        nb_mean[ok] += pcs[v[ok]]
        nb_cnt[ok] += 1
    nb_mean /= np.maximum(nb_cnt, 1.0)[:, None]

    cols_meta = np.column_stack(
        [
            np.log1p(tot),
            np.log1p(volume),
            tot / np.maximum(volume, 1.0),
            center_x,
            center_y,
            rel_x,
            rel_y,
            sp_dist[:, 0],
            nanmean_cols(sp_dist, 5),
            nanmean_cols(sp_dist, 15),
            ex_dist[:, 0],
            ex_dist[:, :10].mean(axis=1),
            region,
            ei,
            segment,
            ap,
            gender,
            mouse,
            dsets,
        ]
    ).astype(np.float32)
    x_static = np.hstack([lognorm, pcs, nb_mean, cols_meta]).astype(np.float32)
    known = np.asarray(y_codes, dtype=np.int64)
    if (known < 0).any():
        raise RuntimeError("SNI training rows contain unlabeled codes")
    sp_hist, sp_names = label_hist(known, sp_idx, sp_dist, SP_K, "sp")
    ex_hist, ex_names = label_hist(known, ex_idx, ex_dist, EX_K, "ex")
    del sp_names, ex_names
    features = np.hstack([x_static, sp_hist, ex_hist]).astype(np.float32)
    return {
        "X": features,
        "y_codes": known,
        "n": n,
        "section_sizes": section_sizes,
        "n_static": int(x_static.shape[1]),
        "graph_note": (
            "SNI spatial kNN is within SNI Section ID. SNI expression kNN is among SNI "
            "cells in the frozen MERFISH-fitted PCA50 space. Neighbor-label histograms "
            "use SNI reference labels only. Competition labels do not enter SNI features."
        ),
    }


def fit_predict(
    X_train,
    y_train,
    X_val,
    names,
    params,
    num_boost_round: int,
    weight: Optional[np.ndarray] = None,
):
    feature_names = [str(name).replace(" ", "_") for name in names]
    dataset = lgb.Dataset(
        X_train,
        y_train,
        weight=None if weight is None else np.asarray(weight, dtype=np.float64),
        feature_name=feature_names,
        free_raw_data=True,
    )
    model = lgb.train(params, dataset, num_boost_round=num_boost_round)
    return model, np.asarray(model.predict(X_val), dtype=np.float64)


def load_aligned_frame(path: Path, cell_ids: Sequence[str], class_names: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"Cell_ID": str})
    frame["Cell_ID"] = frame["Cell_ID"].astype(str)
    frame = frame.set_index("Cell_ID").reindex([str(v) for v in cell_ids])
    if frame.index.has_duplicates or frame.isna().all(axis=1).any():
        raise RuntimeError("failed to align {} on Cell_ID".format(path))
    missing = [c for c in class_names if c not in frame.columns]
    if missing:
        raise RuntimeError("probability file missing classes: {}".format(missing[:5]))
    return frame


def oracle_on_mask(ok: np.ndarray, folds: np.ndarray) -> dict:
    ok = np.asarray(ok, dtype=bool)
    folds = np.asarray(folds)
    return {
        "overall_correct": int(ok.sum()),
        "overall_accuracy": float(ok.mean()),
        "folds_0_2_correct": int(np.sum(ok & (folds <= 2))),
        "folds_0_2_accuracy": float(np.mean(ok[folds <= 2])),
        "folds_3_4_correct": int(np.sum(ok & (folds >= 3))),
        "folds_3_4_accuracy": float(np.mean(ok[folds >= 3])),
    }


def write_validation_csv(path: Path, cell_ids, y_true, y_pred, folds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Cell_ID": [str(v) for v in cell_ids],
            "true_label": list(y_true),
            "predicted_label": list(y_pred),
            "canonical_fold": [int(v) for v in folds],
            "evaluation": "honest_external_validation_fold_safe_neighbor_features_reference_objective",
        }
    ).to_csv(path, index=False)


def fmt_acc(value) -> str:
    if value is None:
        return "n/a"
    return "{:.4f}".format(float(value))


def write_report(payload: dict) -> None:
    m0 = payload["m0"]
    m1 = payload["m1"]
    m2 = payload["m2"]
    w = payload["source_weight_audit"]
    man = payload["dataset_manifest"]
    leak = payload["leakage_audit"]
    cap = payload["sni_capture"]
    ret = payload["anchor_retention"]
    d1 = payload["deltas"]["m1_vs_m0"]
    d2 = payload["deltas"]["m2_vs_m0"]
    ab = payload["m2_vs_m1"]
    five = payload["five_expert"]
    pair = payload["pairwise"]
    fam_m0 = "\n".join(
        "| {family_display} | {n} | {acc} |".format(
            family_display=r["family_display"], n=r["n"], acc=fmt_acc(r["accuracy"])
        )
        for r in m0["families"]
    )
    fam_m1 = "\n".join(
        "| {family_display} | {n} | {acc} |".format(
            family_display=r["family_display"], n=r["n"], acc=fmt_acc(r["accuracy"])
        )
        for r in m1["families"]
    )
    fam_m2 = "\n".join(
        "| {family_display} | {n} | {acc} |".format(
            family_display=r["family_display"], n=r["n"], acc=fmt_acc(r["accuracy"])
        )
        for r in m2["families"]
    )
    rec_fam = "\n".join(
        "| {family_display} | {n} |".format(**r) for r in five["m2_new_unique_by_family"]
    )
    pair_rows = "\n".join(
        "| {candidate} | {existing} | {both_correct} | {candidate_only_correct} | "
        "{existing_only_correct} | {both_wrong} | {oracle_count} | {oracle_accuracy:.4f} |".format(**r)
        for r in pair
    )
    extra = man["additional_sni_cross_source_duplicates_excluded"]
    text = f"""# V3-E06M — Source-Balanced Multi-Reference Transfer

Research codename: **{EXPERIMENT_CODENAME}**. This is a MODEL V3 research experiment, not a formal MODEL V3 freeze.

## 1. Research Question

Can SNI source information be internalized into a stronger single 200-gene reference model during training, by pooling cleaned MERFISH and cleaned SNI rows while explicitly preventing either source from dominating the objective?

The experiment does not route between source-specific experts at inference time. It trains two predeclared candidates only: naive pooling (M1, control) and source-balanced pooling (M2, primary).

## 2. Motivation from V3-E04S / E05A

Frozen auditable pool:

- LZH Prior-H: {LZH_CORRECT} / 5000 = 0.8266
- WYH MODEL V2 / M0: {WYH_CORRECT} / 5000 = 0.8212
- S0: {S0_CORRECT} / 5000 = 0.6302
- SNI-only expert: {SNI_CORRECT} / 5000 = 0.5682
- LZH + WYH oracle: {LZH_WYH_ORACLE} / 5000 = 0.8430
- LZH + WYH + S0 oracle: {THREE_EXPERT_ORACLE} / 5000 = 0.8622
- LZH + WYH + S0 + SNI oracle: {FOUR_EXPERT_ORACLE} / 5000 = 0.8728
- SNI unique recoveries beyond LZH+WYH+S0: **{SNI_UNIQUE_RECOVERIES}**
- remaining all-four-wrong: **{ALL_FOUR_WRONG}**

V3-E04S established that SNI is a weak global classifier but contributes {SNI_UNIQUE_RECOVERIES} genuinely new source-diverse correct cells. V3-E05A established that directional prediction rules using that complementarity have negative net correction.

## 3. Why Test-Time Routing Was Rejected

V3-E03A found that confidence-based rescue of the weak S0 expert is unsafe. V3-E05A found that predeclared directional overrides of strong-expert consensus by S0 or SNI all have negative net correction. Weak source-specific experts are therefore not used as test-time rescue members here. The alternative is to integrate SNI during training inside a single deployable 200-gene LightGBM.

## 4. Project-Level Duplication Check

{man["duplication_check"]["note"]}

The valid project-level claim is: **{man["duplication_check"]["claim"]}**.

Nearby but non-equivalent methods:

- WYH MODEL V2 is MERFISH-reference-only LightGBM with no SNI rows and no sample weights
- V3-E04S is an isolated SNI-only expert, not a combined source-balanced model
- YHH `sample_weight` is class-balancing / competition-vs-MERFISH mixing for family specialists, not MERFISH+SNI 0.5/0.5 source mass
- LZH `source_weights` downweight rare-class MERFISH-reference rows inside binary glial heads, not a 60-class source-balanced MERFISH+SNI LightGBM
- Generic ensembles and separate reference experts are not equivalent to explicit source-balanced single-model training

No auditable equivalent of MERFISH + SNI + explicit source-balanced sample weighting + single deployable 200-gene reference model was found.

## 5. Reference Provenance

| Source | Path | MD5 | Raw shape | Label column |
|---|---|---|---|---|
| MERFISH | `{man["merfish"]["path"]}` | `{man["merfish"]["md5"]}` | {man["merfish"]["raw_n_obs"]} × {man["merfish"]["raw_n_vars"]} | MERFISH cell type annotation |
| SNI | `{man["sni"]["path"]}` | `{man["sni"]["md5"]}` | {man["sni"]["raw_n_obs"]} × {man["sni"]["raw_n_vars"]} | voting + `norm_label` |

Official 200-gene SHA256: `{man["gene_order_sha256"]}`. Matches frozen V2: {man["gene_order_matches_v2"]}.

## 6. Source Cleaning / Exclusion Audit

MERFISH exclusions reproduced from the frozen V2 contract: {man["merfish"]["n_train_id_overlaps_removed"]} train IDs, {man["merfish"]["n_test_id_overlaps_removed"]} test IDs, {man["merfish"]["n_exact_vector_duplicates_removed"]} exact 200-gene competition duplicates. Usable MERFISH: **{man["merfish"]["n_usable"]}**.

SNI exclusions reproduced from V3-E04S: {man["sni"]["competition_train_id_overlap"]} train ID overlaps, {man["sni"]["competition_test_id_overlap"]} test ID overlaps, {man["sni"]["dup_train"]} train exact-vector duplicates, {man["sni"]["dup_test"]} test exact-vector duplicates, {man["sni"]["dup_merfish"]} cleaned-MERFISH exact-vector duplicates. Usable SNI after E04S contract: **{man["sni"]["n_usable_e04s"]}**.

Additional remaining cross-source exact 200-gene duplicates after that contract, excluded from SNI while retaining MERFISH: **{extra}**.

## 7. Combined Reference Population

| Quantity | Count |
|---|---:|
| Cleaned MERFISH | {man["merfish"]["n_usable"]} |
| Cleaned SNI after E04S | {man["sni"]["n_usable_e04s"]} |
| Combined before extra cross-source dups | {man["combined_before_extra"]} |
| Extra SNI copies excluded | {extra} |
| Final combined training rows | {man["n_combined"]} |
| MERFISH class coverage | {man["merfish"]["n_classes"]}/60 |
| SNI class coverage | {man["sni"]["n_classes"]}/60 |

`reference_source ∈ {{MERFISH, SNI}}` is stored for weighting/audit only and is **not** a model input.

Combined-row identity SHA256: `{man["combined_row_identity_sha256"]}`.

## 8. Frozen V2 Modeling Contract

E06M reuses V2-B-REFONLY:

- official 200-gene order and library-size log1p with the median total of the 10,000 competition cells
- PCA50 (`random_state=0`) fitted on the frozen 146,621-cell MERFISH deposit, **not** refit on SNI
- within-section spatial kNN, PCA50 expression kNN, neighbor-mean PCA, metadata, fold-safe neighbor-label histograms
- LightGBM multiclass, 700 fixed rounds, seed 0, `num_threads=8`, no early stopping
- E/I then Segment post-processing from the frozen V2 leakage-safe maps
- competition labels never enter the boosting objective
- held-out fold labels remain invisible in histograms, masks, and fitting

Competition-cell and MERFISH-reference features stay in the frozen MERFISH universe. SNI training rows receive the same feature schema, with SNI-internal graphs and SNI labels in histograms. Missing SNI metadata (Region, E/I, Segment, gender) is NaN. Unseen SNI mouse/dataset codes are NaN so they cannot become a source identifier.

Feature-schema audit: **{payload["contract_audit"]["status"]}**. {payload["contract_audit"]["note"]}

M0 was not retrained. Frozen MODEL V2 artifacts were reused.

## 9. Candidate Definitions

### M0

Frozen MERFISH-only MODEL V2 / V2-B-REFONLY. Accuracy {m0["accuracy"]:.4f} ({m0["correct"]} / 5000). Macro-F1 {m0["macro_f1"]:.4f}.

### M1 — Naive multi-reference pool (CONTROL)

Same combined rows as M2. Frozen V2 base sample-weight behavior: every external row has weight 1. Source contribution is proportional to row counts. M1 is not the primary innovation.

### M2 — Source-balanced multi-reference (PRIMARY)

Exactly the same rows, features, labels, folds, rounds, and LightGBM settings as M1. Only sample weighting differs: each source receives total mass 0.5, then one common rescale makes mean(weight)=1.

## 10. Source-Weight Audit

| Candidate | MERFISH n | SNI n | MERFISH effective sum | SNI effective sum | ratio MERFISH/SNI | mean | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | {w["M1"]["by_source"]["MERFISH"]["count"]} | {w["M1"]["by_source"]["SNI"]["count"]} | {w["M1"]["by_source"]["MERFISH"]["total_effective_weight"]:.6f} | {w["M1"]["by_source"]["SNI"]["total_effective_weight"]:.6f} | {w["M1"]["weight_sum_ratio_merfish_over_sni"]:.6f} | {w["M1"]["mean_weight"]:.6f} | {w["M1"]["min_weight"]:.6f} | {w["M1"]["max_weight"]:.6f} |
| M2 | {w["M2"]["by_source"]["MERFISH"]["count"]} | {w["M2"]["by_source"]["SNI"]["count"]} | {w["M2"]["by_source"]["MERFISH"]["total_effective_weight"]:.6f} | {w["M2"]["by_source"]["SNI"]["total_effective_weight"]:.6f} | {w["M2"]["weight_sum_ratio_merfish_over_sni"]:.6f} | {w["M2"]["mean_weight"]:.6f} | {w["M2"]["min_weight"]:.6f} | {w["M2"]["max_weight"]:.6f} |

M2 required effective source-weight ratio ≈ 1.0: **{w["M2"]["balance_ok"]}**.

No 0.6/0.4 through 0.9/0.1 grid was tested.

## 11. Overall Validation Results

Predictions on the 5000 competition-train cells are honest external-validation predictions from models whose boosting objective uses only external-reference labels. Fold-safe neighbor-label histograms may use non-held-out competition-train labels exactly as in frozen V2. This is not a model that is independent of all competition-train labels, and it is not conventional competition-label OOF for the boosting target.

| Model | Correct | Accuracy | Macro-F1 | Log loss |
|---|---:|---:|---:|---:|
| M0 | {m0["correct"]} | {m0["accuracy"]:.4f} | {m0["macro_f1"]:.4f} | {m0["log_loss"]:.4f} |
| M1 | {m1["correct"]} | {m1["accuracy"]:.4f} | {m1["macro_f1"]:.4f} | {m1["log_loss"]:.4f} |
| M2 | {m2["correct"]} | {m2["accuracy"]:.4f} | {m2["macro_f1"]:.4f} | {m2["log_loss"]:.4f} |

## 12. Fold / Stability Results

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Folds 0-2 | Folds 3-4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | {m0["fold_accuracy"]["0"]:.3f} | {m0["fold_accuracy"]["1"]:.3f} | {m0["fold_accuracy"]["2"]:.3f} | {m0["fold_accuracy"]["3"]:.3f} | {m0["fold_accuracy"]["4"]:.3f} | {m0["folds_0_2"]:.4f} | {m0["folds_3_4"]:.4f} |
| M1 | {m1["fold_accuracy"]["0"]:.3f} | {m1["fold_accuracy"]["1"]:.3f} | {m1["fold_accuracy"]["2"]:.3f} | {m1["fold_accuracy"]["3"]:.3f} | {m1["fold_accuracy"]["4"]:.3f} | {m1["folds_0_2"]:.4f} | {m1["folds_3_4"]:.4f} |
| M2 | {m2["fold_accuracy"]["0"]:.3f} | {m2["fold_accuracy"]["1"]:.3f} | {m2["fold_accuracy"]["2"]:.3f} | {m2["fold_accuracy"]["3"]:.3f} | {m2["fold_accuracy"]["4"]:.3f} | {m2["folds_0_2"]:.4f} | {m2["folds_3_4"]:.4f} |

Canonical folds 3-4 remain a retrospective stability partition, not an untouched holdout.

## 13. Slice / Biological-Family Results

| Slice | M0 | M1 | M2 |
|---|---:|---:|---:|
| Hard bucket | {fmt_acc(m0["slices"]["hard_bucket_accuracy"])} | {fmt_acc(m1["slices"]["hard_bucket_accuracy"])} | {fmt_acc(m2["slices"]["hard_bucket_accuracy"])} |
| Non-hard bucket | {fmt_acc(m0["slices"]["non_hard_bucket_accuracy"])} | {fmt_acc(m1["slices"]["non_hard_bucket_accuracy"])} | {fmt_acc(m2["slices"]["non_hard_bucket_accuracy"])} |
| Neuron | {fmt_acc(m0["slices"]["neuron_accuracy"])} | {fmt_acc(m1["slices"]["neuron_accuracy"])} | {fmt_acc(m2["slices"]["neuron_accuracy"])} |
| Glial / non-neuronal | {fmt_acc(m0["slices"]["glial_accuracy"])} | {fmt_acc(m1["slices"]["glial_accuracy"])} | {fmt_acc(m2["slices"]["glial_accuracy"])} |

M0 families:

{fam_m0}

M1 families:

{fam_m1}

M2 families:

{fam_m2}

## 14. M1 and M2 vs M0 Cell-Level Deltas

| Comparison | Split | Changed | Wrong→correct | Correct→wrong | Net |
|---|---|---:|---:|---:|---:|
| M1 vs M0 | overall | {d1["overall"]["changed_predictions"]} | {d1["overall"]["wrong_to_correct"]} | {d1["overall"]["correct_to_wrong"]} | {d1["overall"]["net_correction"]} |
| M1 vs M0 | folds 0-2 | {d1["folds_0_2"]["changed_predictions"]} | {d1["folds_0_2"]["wrong_to_correct"]} | {d1["folds_0_2"]["correct_to_wrong"]} | {d1["folds_0_2"]["net_correction"]} |
| M1 vs M0 | folds 3-4 | {d1["folds_3_4"]["changed_predictions"]} | {d1["folds_3_4"]["wrong_to_correct"]} | {d1["folds_3_4"]["correct_to_wrong"]} | {d1["folds_3_4"]["net_correction"]} |
| M2 vs M0 | overall | {d2["overall"]["changed_predictions"]} | {d2["overall"]["wrong_to_correct"]} | {d2["overall"]["correct_to_wrong"]} | {d2["overall"]["net_correction"]} |
| M2 vs M0 | folds 0-2 | {d2["folds_0_2"]["changed_predictions"]} | {d2["folds_0_2"]["wrong_to_correct"]} | {d2["folds_0_2"]["correct_to_wrong"]} | {d2["folds_0_2"]["net_correction"]} |
| M2 vs M0 | folds 3-4 | {d2["folds_3_4"]["changed_predictions"]} | {d2["folds_3_4"]["wrong_to_correct"]} | {d2["folds_3_4"]["correct_to_wrong"]} | {d2["folds_3_4"]["net_correction"]} |

## 15. M2 vs M1 Source-Balancing Ablation

| Quantity | Value |
|---|---:|
| M1 accuracy | {m1["accuracy"]:.4f} |
| M2 accuracy | {m2["accuracy"]:.4f} |
| M2 correct − M1 correct | {ab["correct_count_delta"]} |
| M2 wrong→correct vs M1 | {ab["wrong_to_correct"]} |
| M2 correct→wrong vs M1 | {ab["correct_to_wrong"]} |
| Net M2 gain over M1 | {ab["net_correction"]} |
| Macro-F1 delta | {ab["macro_f1_delta"]:.6f} |
| Folds 0-2 delta (accuracy) | {ab["folds_0_2_delta"]:.6f} |
| Folds 3-4 delta (accuracy) | {ab["folds_3_4_delta"]:.6f} |
| Does balancing help? | {ab["balancing_helps"]} |

M1 remains a control. It is not promoted as MODEL V3 from this experiment.

## 16. SNI Unique-Signal Capture

Frozen V3-E04S unique recoveries: {SNI_UNIQUE_RECOVERIES} cells where LZH, WYH, and S0 are wrong and SNI is correct.

| Candidate | Captured / 53 | Capture fraction | M0 wrong → candidate correct on the 53 | Agrees with SNI | Remains equal to M0 |
|---|---:|---:|---:|---:|---:|
| M1 | {cap["m1"]["captured"]} | {cap["m1"]["capture_fraction"]:.4f} | {cap["m1"]["m0_wrong_to_candidate_correct"]} | {cap["m1"]["agrees_with_sni"]} | {cap["m1"]["equals_m0"]} |
| M2 | {cap["m2"]["captured"]} | {cap["m2"]["capture_fraction"]:.4f} | {cap["m2"]["m0_wrong_to_candidate_correct"]} | {cap["m2"]["agrees_with_sni"]} | {cap["m2"]["equals_m0"]} |

## 17. MERFISH Anchor Retention

| Candidate | M0-correct retained | M0-correct lost | Retention rate | M0-wrong recovered | Net transfer |
|---|---:|---:|---:|---:|---:|
| M1 | {ret["m1"]["retained"]} | {ret["m1"]["lost"]} | {ret["m1"]["retention_rate"]:.4f} | {ret["m1"]["recovered"]} | {ret["m1"]["net_transfer"]} |
| M2 | {ret["m2"]["retained"]} | {ret["m2"]["lost"]} | {ret["m2"]["retention_rate"]:.4f} | {ret["m2"]["recovered"]} | {ret["m2"]["net_transfer"]} |

Source integration is unsuccessful if it captures SNI cases but destroys more frozen MERFISH-anchor correct cells than it recovers.

## 18. Pairwise Complementarity

**ORACLE != DEPLOYABLE ACCURACY.**

| Candidate | Existing | Both correct | Candidate-only | Existing-only | Both wrong | Oracle n | Oracle acc |
|---|---|---:|---:|---:|---:|---:|---:|
{pair_rows}

No blend or weight was optimized.

## 19. New Recoveries Beyond the Four-Expert Pool

New unique recoveries are cells where LZH, WYH, S0, and SNI are all wrong and the candidate is correct.

| Candidate | New unique recoveries | Five-expert oracle | Five-expert accuracy |
|---|---:|---:|---:|
| M1 | {five["m1_new_unique_recoveries"]} | {five["m1_five_expert_oracle_correct"]} | {five["m1_five_expert_oracle_accuracy"]:.4f} |
| M2 | {five["m2_new_unique_recoveries"]} | {five["m2_five_expert_oracle_correct"]} | {five["m2_five_expert_oracle_accuracy"]:.4f} |

Identity check: five-expert oracle = {FOUR_EXPERT_ORACLE} + new unique recoveries. M2 identity: **{five["m2_identity_ok"]}**.

M2 new-unique families:

{rec_fam}

## 20. Five-Expert Diagnostic Oracle

**ORACLE != DEPLOYABLE ACCURACY.**

M2 five-expert diagnostic coverage:

| Split | Correct | Accuracy | Remaining all-five-wrong |
|---|---:|---:|---:|
| Overall | {five["m2_oracle"]["overall_correct"]} | {five["m2_oracle"]["overall_accuracy"]:.4f} | {five["m2_all_five_wrong"]} |
| Folds 0-2 | {five["m2_oracle"]["folds_0_2_correct"]} | {five["m2_oracle"]["folds_0_2_accuracy"]:.4f} | {five["m2_all_five_wrong_02"]} |
| Folds 3-4 | {five["m2_oracle"]["folds_3_4_correct"]} | {five["m2_oracle"]["folds_3_4_accuracy"]:.4f} | {five["m2_all_five_wrong_34"]} |

These numbers are diagnostic coverage ceilings, not a deployable MODEL V3 score.

## 21. Leakage Audit

- competition test labels used: {leak["competition_test_labels_used"]}
- competition train labels enter boosting objective: {leak["competition_train_labels_in_boosting_objective"]}
- held-out fold labels remain invisible: {leak["held_out_fold_labels_invisible"]}
- all external competition ID overlaps excluded: {leak["competition_id_overlaps_excluded"]}
- all prohibited exact-vector duplicates excluded: {leak["prohibited_exact_vector_duplicates_excluded"]}
- hyperparameter search: {leak["hyperparameter_search"]}
- seed search: {leak["seed_search"]}
- source-weight search: {leak["source_weight_search"]}
- leaderboard tuning: {leak["leaderboard_feedback_used"]}
- post-hoc class-rule tuning: {leak["post_hoc_class_rule_tuning"]}
- test probabilities used in selection: {leak["test_probabilities_used_for_model_selection"]}
- prediction/prediction.csv modified: {leak["prediction_csv_modified"]}
- MODEL V1 / V2 / V3-E00T / E02D / E03A / E04S / E05A modified: {leak["frozen_artifacts_modified"]}
- `reference_source` used as a model input: {leak["source_used_as_feature"]}

## 22. Limitations

- SNI labels are the file's `voting` consensus after `norm_label`, not a wet-lab gold standard.
- SNI mixes Sham and SNI injury conditions; this experiment treats the file as one source.
- SNI rows lack Region / E/I / Segment / gender, so those V2 metadata channels are missing for the SNI half of the objective.
- SNI graphs are source-internal; competition cells do not gain SNI spatial neighbors. That preserves the frozen V2 competition feature contract.
- Canonical folds 3-4 have been viewed in prior V3 stages and are retrospective.
- Diagnostic oracle coverage is not deployable accuracy.
- Only the predeclared 0.5/0.5 source mass was tested.

## 23. Decision

**{payload["decision"]["label"]}**

{payload["decision"]["reason"]}

Recommended next experiment/action:

{payload["decision"]["next_action"]}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)


def main() -> int:
    t0 = time.perf_counter()
    assert_v3_interpreter()
    branch = current_branch()
    if branch != EXPECTED_BRANCH:
        return stop("STOP", "branch is {} (expected {})".format(branch, EXPECTED_BRANCH))

    print("V3-E06M source-balanced multi-reference transfer", flush=True)
    print("branch={}".format(branch), flush=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    assert_work_dir_ignored(WORK_DIR / "dummy")

    pred_before = sha256_file(PRED_PATH) if PRED_PATH.is_file() else None
    data = load_dataset(ROOT)
    for line in validate_contract(data):
        print(" - {}".format(line), flush=True)
    class_names = allowed_labels(ROOT)
    genes = list(data.counts_train.columns)
    if gene_list_sha256(genes) != OFFICIAL_GENE_SHA256:
        return stop("M0 CONTRACT REPRODUCTION FAILURE", "official 200-gene hash mismatch")

    folds_df, fold_messages = load_and_validate_team_folds(
        data.counts_train.index, data.counts_test.index, data.y_train
    )
    print("Canonical partition:", TEAM_CV_PROTOCOL, flush=True)
    for line in fold_messages:
        print(" - {}".format(line), flush=True)

    print("Auditing frozen MERFISH reference...", flush=True)
    mer_audit = audit_reference(data, class_names, root=ROOT)
    if mer_audit["md5"] != MERFISH_MD5:
        return stop("STOP", "MERFISH MD5 mismatch")
    if int(mer_audit["n_usable_reference_rows"]) != HISTORICAL_MERFISH_USABLE:
        return stop(
            "STOP",
            "cleaned MERFISH count {} != {}".format(
                mer_audit["n_usable_reference_rows"], HISTORICAL_MERFISH_USABLE
            ),
        )
    if mer_audit["reference_label_coverage"] != 60:
        return stop("STOP", "MERFISH class coverage is not 60/60")

    print("Auditing SNI reference under the V3-E04S contract...", flush=True)
    sni_audit = audit_sni_dataset(data, class_names)
    if not sni_audit.get("md5_ok"):
        return stop("STOP", "SNI MD5 {} != {}".format(sni_audit.get("md5"), EXPECTED_SNI_MD5))
    keep_e04s = np.asarray(sni_audit["_keep"])
    if int(keep_e04s.sum()) != HISTORICAL_SNI_USABLE:
        return stop(
            "STOP",
            "cleaned SNI count {} != {}".format(int(keep_e04s.sum()), HISTORICAL_SNI_USABLE),
        )
    tax = sni_audit.get("taxonomy") or {}
    if tax.get("mapped_class_count") != 60:
        return stop("STOP", "SNI class coverage is not 60/60")

    x_sni_all = np.asarray(sni_audit["_x_sni"], dtype=np.int64)
    sni_ids_all = np.asarray(sni_audit["_sni_ids"], dtype=object)
    sni_labels_all = np.asarray(sni_audit["_mapped_labels"], dtype=object)
    mer_ref_mask = np.asarray(mer_audit["_is_ref"])
    mer_x200 = np.asarray(mer_audit["_x200"], dtype=np.int64)
    mer_fp = _row_fingerprints(mer_x200[mer_ref_mask])
    sni_fp_kept = _row_fingerprints(x_sni_all[keep_e04s])
    extra_dup = fingerprint_overlap_mask(sni_fp_kept, mer_fp)
    keep_final = keep_e04s.copy()
    keep_final[np.where(keep_e04s)[0][extra_dup]] = False
    n_extra = int(extra_dup.sum())
    n_sni = int(keep_final.sum())
    n_mer = int(mer_ref_mask.sum())
    n_combined_before = n_mer + int(keep_e04s.sum())
    if n_combined_before != EXPECTED_COMBINED_BEFORE_EXTRA:
        return stop(
            "STOP",
            "combined count before extra dups {} != {}".format(
                n_combined_before, EXPECTED_COMBINED_BEFORE_EXTRA
            ),
        )
    if n_extra:
        print("Additional cross-source exact duplicates excluded from SNI: {}".format(n_extra), flush=True)

    x_sni = x_sni_all[keep_final]
    sni_ids = sni_ids_all[keep_final]
    sni_labels = sni_labels_all[keep_final]
    class_index = {str(c): i for i, c in enumerate(class_names)}
    y_sni = np.array([class_index[str(v)] for v in sni_labels], dtype=np.int64)
    sni_classes = sorted(set(str(v) for v in sni_labels.tolist()))
    mer_classes = sorted(
        set(str(v) for v in np.array([class_names[i] for i in mer_audit["_y_codes"][mer_ref_mask]]))
    )
    if len(sni_classes) != 60 or len(mer_classes) != 60:
        return stop("STOP", "usable source class coverage failed")

    train_ids = np.array([str(v) for v in data.counts_train.index.tolist()], dtype=object)
    test_ids = np.array([str(v) for v in data.counts_test.index.tolist()], dtype=object)
    train_set = set(train_ids.tolist())
    test_set = set(test_ids.tolist())
    if any(cid in train_set or cid in test_set for cid in sni_ids.tolist()):
        return stop("STOP", "SNI usable IDs overlap competition IDs")
    mer_ids = np.asarray(mer_audit["_ext_ids"], dtype=object)[mer_ref_mask]
    if len(set(mer_ids.tolist()) & set(sni_ids.tolist())):
        return stop("STOP", "usable MERFISH and SNI Cell_IDs overlap")

    print("Loading frozen V2 extended universe...", flush=True)
    universe = build_or_load_ext_universe(mer_audit, data, class_names, root=ROOT)
    scaler, pca, pca_audit = reconstruct_merfish_pca(universe)
    if not pca_audit["reproduced"]:
        return stop(
            "M0 CONTRACT REPRODUCTION FAILURE",
            "PCA reconstruction max-abs {} exceeds atol".format(pca_audit["max_abs_pc_difference"]),
        )

    dummy_known = visible_ext_label_codes(
        universe.y_codes,
        universe.is_train,
        universe.is_ref,
        np.full(len(universe.cell_ids), -1, dtype=np.int64),
        None,
    )
    _X0, names = build_X(universe, dummy_known, sp_k=SP_K, ex_k=EX_K)
    del _X0
    expected_names = expected_feature_names(genes)
    if names != expected_names:
        return stop(
            "M0 CONTRACT REPRODUCTION FAILURE",
            "feature names diverged from the frozen V2 schema",
        )
    ok_schema, schema_reason = feature_schema_ok(names)
    if not ok_schema:
        return stop("M0 CONTRACT REPRODUCTION FAILURE", schema_reason)

    mer_obs = mer_audit["_adata"].obs
    mer_mouse_cat = pd.Categorical(mer_obs["Mouse ID"].astype(str))
    mer_dset_cat = pd.Categorical(mer_obs["Datasets"].astype(str))
    tot_comp = mer_x200[mer_audit["_is_train"] | mer_audit["_is_test"]].sum(axis=1)
    scale = float(np.median(tot_comp))

    print("Building SNI features in the frozen V2 schema...", flush=True)
    sni_obs = sni_audit["_adata"].obs
    sni_pack = build_sni_v2_schema_features(
        x_sni,
        sni_obs,
        keep_final,
        y_sni,
        scale,
        scaler,
        pca,
        mer_mouse_cat.categories,
        mer_dset_cat.categories,
        mer_audit["_ap_map"],
    )
    if sni_pack["X"].shape[1] != len(names):
        return stop(
            "M0 CONTRACT REPRODUCTION FAILURE",
            "SNI feature width {} != V2 width {}".format(sni_pack["X"].shape[1], len(names)),
        )
    X_sni = np.asarray(sni_pack["X"], dtype=np.float32)
    np.savez_compressed(
        WORK_DIR / "sni_features.npz",
        X=X_sni,
        y=sni_pack["y_codes"],
        ids=sni_ids.astype(str),
    )

    source = np.array(["MERFISH"] * n_mer + ["SNI"] * n_sni, dtype=object)
    combined_ids = np.concatenate([np.asarray(mer_ids, dtype=object), sni_ids])
    identity = combined_row_identity(source, combined_ids)
    base_w = np.ones(n_mer + n_sni, dtype=np.float64)
    w_m1 = base_w.copy()
    w_m2 = source_balanced_weights(source, base_w)
    w_audit = {
        "M1": weight_summary(source, base_w, w_m1),
        "M2": weight_summary(source, base_w, w_m2),
    }
    w_audit["M2"]["balance_ok"] = bool(
        abs(w_audit["M2"]["weight_sum_ratio_merfish_over_sni"] - 1.0) <= WEIGHT_RATIO_ATOL
        and abs(w_audit["M2"]["mean_weight"] - 1.0) <= 1e-8
    )
    if not w_audit["M2"]["balance_ok"]:
        return stop("STOP", "M2 source balancing is not implemented exactly")
    write_json(OUT_DIR / "v3_e06m_source_weight_audit.json", w_audit)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": utc_now(),
        "branch": branch,
        "gene_order_sha256": gene_list_sha256(genes),
        "gene_order_matches_v2": True,
        "merfish": {
            "path": str(reference_h5ad_path(ROOT)),
            "md5": mer_audit["md5"],
            "raw_n_obs": mer_audit["raw_n_obs"],
            "raw_n_vars": mer_audit["raw_n_vars"],
            "n_train_id_overlaps_removed": mer_audit["n_train_id_overlaps_removed"],
            "n_test_id_overlaps_removed": mer_audit["n_test_id_overlaps_removed"],
            "n_exact_vector_duplicates_removed": mer_audit["n_exact_vector_duplicates_removed"],
            "n_usable": n_mer,
            "n_classes": 60,
        },
        "sni": {
            "path": str(sni_h5ad_path(ROOT)),
            "md5": sni_audit["md5"],
            "raw_n_obs": sni_audit["n_obs"],
            "raw_n_vars": sni_audit["n_vars"],
            "competition_train_id_overlap": sni_audit["competition_train_cell_id_overlap"],
            "competition_test_id_overlap": sni_audit["competition_test_cell_id_overlap"],
            "dup_train": sni_audit["exact_vector_duplicate_audit"][
                "sni_exact_vectors_matching_competition_train"
            ],
            "dup_test": sni_audit["exact_vector_duplicate_audit"][
                "sni_exact_vectors_matching_competition_test"
            ],
            "dup_merfish": sni_audit["exact_vector_duplicate_audit"][
                "sni_exact_vectors_matching_cleaned_merfish_reference"
            ],
            "n_usable_e04s": int(keep_e04s.sum()),
            "n_usable": n_sni,
            "n_classes": 60,
        },
        "combined_before_extra": n_combined_before,
        "additional_sni_cross_source_duplicates_excluded": n_extra,
        "n_combined": int(n_mer + n_sni),
        "combined_row_identity_sha256": identity,
        "reference_source_used_as_feature": False,
        "pca_audit": pca_audit,
        "sni_graph_note": sni_pack["graph_note"],
        "duplication_check": {
            "auditable_equivalent_found": False,
            "claim": (
                "Source-balanced multi-reference training is a new controlled modeling "
                "direction within the current auditable project."
            ),
            "team_shas": {
                "team/main": "c34feed7a03d1a1421c55a6fe4c40765aeff2b8b",
                "team/yhh": "9afc72774e2a6e186a04c2557110992d0befe164",
                "team/lzh": "af51ce78846063f2ec9bede33b6b23b7a75d3492",
                "team/wyh": "da127c6ca86693f8bad5941aa26cc24ba129e1af",
                "team/revert-1-lzh": "bf4b34bbb988505341ec2caec237de6a9587207d",
            },
            "note": (
                "git fetch team --prune succeeded. Inspected team/main, team/yhh, team/lzh, "
                "team/wyh, and team/revert-1-lzh. No auditable equivalent of MERFISH + SNI + "
                "explicit source-balanced sample weighting + single deployable 200-gene "
                "reference model was found. YHH sample_weight is class-balancing / "
                "competition-vs-MERFISH mixing. LZH source_weights downweight rare-class "
                "MERFISH reference rows in binary glial heads. V3-E04S is SNI-only. This is "
                "not a global first-ever novelty claim."
            ),
        },
    }
    write_json(OUT_DIR / "v3_e06m_dataset_manifest.json", manifest)

    # Frozen M0 artifacts.
    m0_oof = pd.read_csv(M0_OOF, dtype={"Cell_ID": str})
    m0_oof["Cell_ID"] = m0_oof["Cell_ID"].astype(str)
    m0_oof = m0_oof.set_index("Cell_ID").reindex(train_ids)
    y_true = data.y_train.loc[train_ids].astype(str).to_numpy()
    m0_pred = m0_oof["predicted_label"].astype(str).to_numpy()
    fold_map = folds_df.set_index("Cell_ID")["fold"]
    folds = fold_map.reindex(train_ids).to_numpy(dtype=np.int64)
    m0_proba_df = load_aligned_frame(M0_OOF_PROBA, train_ids, class_names)
    m0_proba = m0_proba_df.loc[:, class_names].to_numpy(dtype=np.float64)
    assert_probability_rows(m0_proba, atol=1e-4)
    m0_metrics = split_metrics(y_true, m0_pred, folds, m0_proba, class_names)
    if m0_metrics["correct"] != WYH_CORRECT or abs(m0_metrics["accuracy"] - M0_ACCURACY) > 1e-9:
        return stop("M0 CONTRACT REPRODUCTION FAILURE", "frozen M0 accuracy is not 0.8212")
    ei_map = ei_of_label_from_train(data.meta_train, class_names)
    meta_train = data.meta_train.loc[train_ids]
    m0_slices = slice_metrics(y_true, m0_pred, meta_train, class_names, ei_map)
    hard = hard_bucket_mask(meta_train)
    m0_slices["non_hard_bucket_n"] = int((~hard).sum())
    m0_slices["non_hard_bucket_accuracy"] = float(np.mean(y_true[~hard] == m0_pred[~hard]))
    m0_metrics["slices"] = m0_slices
    m0_metrics["families"] = family_accuracy(y_true, m0_pred)

    contract_audit = {
        "status": "PASS",
        "note": (
            "Frozen V2 feature names, PCA50 reconstruction, LightGBM hyperparameters, "
            "700 rounds, seed 0, E/I+Segment post-processing, and M0 accuracy 0.8212 were "
            "reproduced. The only intended differences are the combined reference population "
            "and M2 sample weights."
        ),
        "n_features": len(names),
        "lgbm_params": lgbm_params(NUM_THREADS),
        "num_boost_round": NUM_BOOST_ROUND,
        "pca_audit": pca_audit,
        "source_used_as_feature": False,
    }
    mer_audit.pop("_adata", None)
    sni_audit.pop("_adata", None)

    id_pos = {str(cid): i for i, cid in enumerate(universe.cell_ids.tolist())}
    train_pos = np.array([id_pos[cid] for cid in train_ids.tolist()], dtype=np.int64)
    test_pos = np.array([id_pos[cid] for cid in test_ids.tolist()], dtype=np.int64)
    fold_universe = universe_fold_ids(folds_df, universe.cell_ids, universe.is_train)
    fold_train = fold_universe[train_pos]
    if not np.array_equal(fold_train, folds):
        return stop("STOP", "universe fold mapping diverged from team folds")
    ref_idx = np.where(universe.is_ref)[0]
    if (universe.y_codes[ref_idx] < 0).any():
        return stop("STOP", "unlabeled usable MERFISH reference rows")
    y_mer = universe.y_codes[ref_idx]
    y_fit = np.concatenate([y_mer, sni_pack["y_codes"]])
    if y_fit.shape[0] != w_m1.shape[0]:
        return stop("STOP", "weight length does not match combined training rows")

    params = lgbm_params(NUM_THREADS)
    oof_m1 = np.zeros((N_TRAIN, N_CLASSES), dtype=np.float64)
    oof_m2 = np.zeros((N_TRAIN, N_CLASSES), dtype=np.float64)
    allowed_seg = segment_allowed_from_reference(universe.segment, universe.is_ref, universe.y_codes)

    for fold_id in TEAM_FOLD_VALUES:
        known = visible_ext_label_codes(
            universe.y_codes, universe.is_train, universe.is_ref, fold_universe, fold_id
        )
        if (known[train_pos][fold_train == fold_id] != -1).any():
            return stop("STOP", "held-out fold labels leaked into known")
        if (known[test_pos] != -1).any():
            return stop("STOP", "test labels visible")
        features, feat_names = build_X(universe, known, sp_k=SP_K, ex_k=EX_K)
        if feat_names != names:
            return stop("M0 CONTRACT REPRODUCTION FAILURE", "fold feature names changed")
        val_idx = train_pos[fold_train == fold_id]
        X_fit = np.vstack([features[ref_idx], X_sni])
        X_val = features[val_idx]
        print(
            "  fold {} n_fit={} n_val={} n_features={}".format(
                fold_id, X_fit.shape[0], X_val.shape[0], X_fit.shape[1]
            ),
            flush=True,
        )
        fold_t0 = time.perf_counter()
        _m, p1 = fit_predict(X_fit, y_fit, X_val, names, params, NUM_BOOST_ROUND, w_m1)
        oof_m1[fold_train == fold_id] = p1
        del _m
        _m, p2 = fit_predict(X_fit, y_fit, X_val, names, params, NUM_BOOST_ROUND, w_m2)
        oof_m2[fold_train == fold_id] = p2
        del _m, features, X_fit, X_val
        pred1 = argmax_labels(p1, class_names)
        pred2 = argmax_labels(p2, class_names)
        true_fold = y_true[fold_train == fold_id]
        print(
            "  fold {} M1={:.4f} M2={:.4f} ({:.0f}s)".format(
                fold_id,
                float(np.mean(pred1 == true_fold)),
                float(np.mean(pred2 == true_fold)),
                time.perf_counter() - fold_t0,
            ),
            flush=True,
        )

    def postprocess(probs: np.ndarray, pos: np.ndarray) -> np.ndarray:
        assert_probability_rows(probs, atol=1e-4)
        ei = apply_ei(probs, universe.ei_known[pos], ei_map)
        assert_probability_rows(ei, atol=1e-4)
        seg = apply_segment_mask(ei, universe.segment[pos], allowed_seg)
        assert_probability_rows(seg, atol=1e-4)
        return seg

    val_m1 = postprocess(oof_m1, train_pos)
    val_m2 = postprocess(oof_m2, train_pos)
    pred_m1 = argmax_labels(val_m1, class_names)
    pred_m2 = argmax_labels(val_m2, class_names)
    m1_metrics = split_metrics(y_true, pred_m1, folds, val_m1, class_names)
    m2_metrics = split_metrics(y_true, pred_m2, folds, val_m2, class_names)
    for block, pred in ((m1_metrics, pred_m1), (m2_metrics, pred_m2)):
        sl = slice_metrics(y_true, pred, meta_train, class_names, ei_map)
        sl["non_hard_bucket_n"] = int((~hard).sum())
        sl["non_hard_bucket_accuracy"] = float(np.mean(y_true[~hard] == pred[~hard]))
        block["slices"] = sl
        block["families"] = family_accuracy(y_true, pred)

    print("Fitting full-reference M1/M2 test models...", flush=True)
    known_test = visible_ext_label_codes(
        universe.y_codes, universe.is_train, universe.is_ref, fold_universe, None
    )
    if (known_test[test_pos] != -1).any():
        return stop("STOP", "test labels visible at test-time")
    features_full, names_full = build_X(universe, known_test, sp_k=SP_K, ex_k=EX_K)
    if names_full != names:
        return stop("M0 CONTRACT REPRODUCTION FAILURE", "test-time feature names changed")
    X_fit_full = np.vstack([features_full[ref_idx], X_sni])
    X_test = features_full[test_pos]
    _m, test_m1 = fit_predict(X_fit_full, y_fit, X_test, names, params, NUM_BOOST_ROUND, w_m1)
    _m.save_model(str(WORK_DIR / "m1_full.txt"))
    del _m
    _m, test_m2 = fit_predict(X_fit_full, y_fit, X_test, names, params, NUM_BOOST_ROUND, w_m2)
    _m.save_model(str(WORK_DIR / "m2_full.txt"))
    del _m, features_full, X_fit_full, X_test
    test_m1 = postprocess(test_m1, test_pos)
    test_m2 = postprocess(test_m2, test_pos)
    if test_m1.shape != (N_TEST, N_CLASSES) or test_m2.shape != (N_TEST, N_CLASSES):
        return stop("STOP", "test probability shape is not 5000 x 60")

    write_validation_csv(OUT_DIR / "v3_e06m_m1_validation.csv", train_ids, y_true, pred_m1, folds)
    write_validation_csv(OUT_DIR / "v3_e06m_m2_validation.csv", train_ids, y_true, pred_m2, folds)
    write_proba(OUT_DIR / "v3_e06m_m1_validation_probabilities.csv.gz", train_ids, val_m1, class_names)
    write_proba(OUT_DIR / "v3_e06m_m2_validation_probabilities.csv.gz", train_ids, val_m2, class_names)
    write_proba(OUT_DIR / "v3_e06m_m1_test_probabilities.csv.gz", test_ids, test_m1, class_names)
    write_proba(OUT_DIR / "v3_e06m_m2_test_probabilities.csv.gz", test_ids, test_m2, class_names)

    def delta_block(pred_a, pred_b) -> dict:
        return {
            "overall": cell_delta(pred_a, pred_b, y_true),
            "folds_0_2": cell_delta(pred_a, pred_b, y_true, folds <= 2),
            "folds_3_4": cell_delta(pred_a, pred_b, y_true, folds >= 3),
        }

    deltas = {"m1_vs_m0": delta_block(m0_pred, pred_m1), "m2_vs_m0": delta_block(m0_pred, pred_m2)}
    m2_vs_m1_overall = cell_delta(pred_m1, pred_m2, y_true)
    m2_vs_m1 = {
        "correct_count_delta": int(m2_metrics["correct"] - m1_metrics["correct"]),
        "macro_f1_delta": float(m2_metrics["macro_f1"] - m1_metrics["macro_f1"]),
        "folds_0_2_delta": float(m2_metrics["folds_0_2"] - m1_metrics["folds_0_2"]),
        "folds_3_4_delta": float(m2_metrics["folds_3_4"] - m1_metrics["folds_3_4"]),
        "wrong_to_correct": m2_vs_m1_overall["wrong_to_correct"],
        "correct_to_wrong": m2_vs_m1_overall["correct_to_wrong"],
        "net_correction": m2_vs_m1_overall["net_correction"],
        "balancing_helps": bool(m2_metrics["correct"] > m1_metrics["correct"]),
    }

    registry = pd.read_parquet(E05A_REGISTRY)
    registry["Cell_ID"] = registry["Cell_ID"].astype(str)
    registry = registry.set_index("Cell_ID").reindex(train_ids)
    lzh_pred = registry["lzh_pred"].astype(str).to_numpy()
    wyh_pred = registry["wyh_pred"].astype(str).to_numpy()
    s0_pred = registry["s0_pred"].astype(str).to_numpy()
    sni_pred = registry["sni_pred"].astype(str).to_numpy()
    lzh_ok = lzh_pred == y_true
    wyh_ok = wyh_pred == y_true
    s0_ok = s0_pred == y_true
    sni_ok = sni_pred == y_true
    m0_ok = m0_pred == y_true
    m1_ok = pred_m1 == y_true
    m2_ok = pred_m2 == y_true
    if int(lzh_ok.sum()) != LZH_CORRECT or int(wyh_ok.sum()) != WYH_CORRECT:
        return stop("STOP", "frozen expert correct counts drifted")
    if int(s0_ok.sum()) != S0_CORRECT or int(sni_ok.sum()) != SNI_CORRECT:
        return stop("STOP", "frozen S0/SNI correct counts drifted")
    four_ok = lzh_ok | wyh_ok | s0_ok | sni_ok
    if int(four_ok.sum()) != FOUR_EXPERT_ORACLE:
        return stop("STOP", "frozen four-expert oracle drifted")
    if int((~four_ok).sum()) != ALL_FOUR_WRONG:
        return stop("STOP", "frozen all-four-wrong drifted")

    rec53 = pd.read_csv(E04S_RECOVERIES, dtype={"Cell_ID": str})
    rec53["Cell_ID"] = rec53["Cell_ID"].astype(str)
    if len(rec53) != SNI_UNIQUE_RECOVERIES:
        return stop("STOP", "frozen 53 SNI unique recoveries drifted")
    rec_ids = set(rec53["Cell_ID"].tolist())
    rec_mask = np.array([cid in rec_ids for cid in train_ids.tolist()], dtype=bool)
    if int(rec_mask.sum()) != SNI_UNIQUE_RECOVERIES:
        return stop("STOP", "could not align the frozen 53 SNI recoveries")

    def capture_stats(pred, ok) -> dict:
        captured = int(np.sum(rec_mask & ok))
        return {
            "captured": captured,
            "capture_fraction": float(captured / SNI_UNIQUE_RECOVERIES),
            "m0_wrong_to_candidate_correct": int(np.sum(rec_mask & (~m0_ok) & ok)),
            "agrees_with_sni": int(np.sum(rec_mask & (pred == sni_pred))),
            "equals_m0": int(np.sum(rec_mask & (pred == m0_pred))),
        }

    sni_capture = {"m1": capture_stats(pred_m1, m1_ok), "m2": capture_stats(pred_m2, m2_ok)}
    capture_rows = []
    for cid, tlab, fold, sni_p, m0p, m1p, m2p in zip(
        train_ids[rec_mask],
        y_true[rec_mask],
        folds[rec_mask],
        sni_pred[rec_mask],
        m0_pred[rec_mask],
        pred_m1[rec_mask],
        pred_m2[rec_mask],
    ):
        capture_rows.append(
            {
                "Cell_ID": str(cid),
                "true_label": tlab,
                "canonical_fold": int(fold),
                "true_family": family_of(str(tlab)),
                "sni_pred": sni_p,
                "m0_pred": m0p,
                "m1_pred": m1p,
                "m2_pred": m2p,
                "m0_correct": bool(m0p == tlab),
                "m1_correct": bool(m1p == tlab),
                "m2_correct": bool(m2p == tlab),
                "m1_agrees_sni": bool(m1p == sni_p),
                "m2_agrees_sni": bool(m2p == sni_p),
                "m1_equals_m0": bool(m1p == m0p),
                "m2_equals_m0": bool(m2p == m0p),
            }
        )
    pd.DataFrame(capture_rows).to_csv(OUT_DIR / "v3_e06m_sni_transfer_capture.csv", index=False)

    def retention(ok) -> dict:
        retained = int(np.sum(m0_ok & ok))
        lost = int(np.sum(m0_ok & ~ok))
        recovered = int(np.sum((~m0_ok) & ok))
        return {
            "retained": retained,
            "lost": lost,
            "retention_rate": float(retained / int(m0_ok.sum())),
            "recovered": recovered,
            "net_transfer": recovered - lost,
        }

    anchor_retention = {"m1": retention(m1_ok), "m2": retention(m2_ok)}

    existing = [
        ("lzh_prior_h", lzh_ok, lzh_pred),
        ("wyh_model_v2", wyh_ok, wyh_pred),
        ("s0", s0_ok, s0_pred),
        ("sni", sni_ok, sni_pred),
        ("m0", m0_ok, m0_pred),
    ]
    pairwise_rows = []
    for cand_name, cand_ok, cand_pred in (("M1", m1_ok, pred_m1), ("M2", m2_ok, pred_m2)):
        for ex_name, ex_ok, ex_pred in existing:
            both_c = int(np.sum(cand_ok & ex_ok))
            cand_only = int(np.sum(cand_ok & ~ex_ok))
            ex_only = int(np.sum(ex_ok & ~cand_ok))
            both_w = int(np.sum(~cand_ok & ~ex_ok))
            oracle_n = int(np.sum(cand_ok | ex_ok))
            pairwise_rows.append(
                {
                    "candidate": cand_name,
                    "existing": ex_name,
                    "both_correct": both_c,
                    "candidate_only_correct": cand_only,
                    "existing_only_correct": ex_only,
                    "both_wrong": both_w,
                    "oracle_count": oracle_n,
                    "oracle_accuracy": float(oracle_n / N_TRAIN),
                    "oracle_is_not_deployable_accuracy": True,
                }
            )
    pd.DataFrame(pairwise_rows).to_csv(OUT_DIR / "v3_e06m_pairwise_oracle.csv", index=False)

    m1_new = (~four_ok) & m1_ok
    m2_new = (~four_ok) & m2_ok
    m1_new_n = int(m1_new.sum())
    m2_new_n = int(m2_new.sum())
    five_m1 = FOUR_EXPERT_ORACLE + m1_new_n
    five_m2 = FOUR_EXPERT_ORACLE + m2_new_n
    if five_m2 != int((four_ok | m2_ok).sum()):
        return stop("STOP", "five-expert identity failed for M2")
    new_rows = []
    for cid, tlab, fold, pred, hb in zip(
        train_ids[m2_new],
        y_true[m2_new],
        folds[m2_new],
        pred_m2[m2_new],
        hard[m2_new],
    ):
        new_rows.append(
            {
                "Cell_ID": str(cid),
                "true_label": tlab,
                "canonical_fold": int(fold),
                "m2_pred": pred,
                "true_family": family_of(str(tlab)),
                "hard_bucket": bool(hb),
            }
        )
    pd.DataFrame(new_rows).to_csv(OUT_DIR / "v3_e06m_new_unique_recoveries.csv", index=False)
    fam_counts = Counter(family_of(str(v)) for v in y_true[m2_new])
    m2_new_fam = [
        {"family": k, "family_display": FAMILY_DISPLAY.get(k, k), "n": int(fam_counts.get(k, 0))}
        for k in FAMILY_ORDER
    ]
    five_ok_m2 = four_ok | m2_ok
    five = {
        "m1_new_unique_recoveries": m1_new_n,
        "m2_new_unique_recoveries": m2_new_n,
        "m1_five_expert_oracle_correct": five_m1,
        "m2_five_expert_oracle_correct": five_m2,
        "m1_five_expert_oracle_accuracy": float(five_m1 / N_TRAIN),
        "m2_five_expert_oracle_accuracy": float(five_m2 / N_TRAIN),
        "m2_identity_ok": True,
        "formula": "{} + new_unique_recoveries".format(FOUR_EXPERT_ORACLE),
        "oracle_is_not_deployable_accuracy": True,
        "m2_oracle": oracle_on_mask(five_ok_m2, folds),
        "m2_all_five_wrong": int((~five_ok_m2).sum()),
        "m2_all_five_wrong_02": int(np.sum((~five_ok_m2) & (folds <= 2))),
        "m2_all_five_wrong_34": int(np.sum((~five_ok_m2) & (folds >= 3))),
        "m2_new_unique_by_family": m2_new_fam,
    }

    delta_rows = []
    for name, block in (("M1_vs_M0", deltas["m1_vs_m0"]), ("M2_vs_M0", deltas["m2_vs_m0"])):
        for split, row in block.items():
            rec = {"comparison": name, "split": split}
            rec.update(row)
            delta_rows.append(rec)
    pd.DataFrame(delta_rows).to_csv(OUT_DIR / "v3_e06m_cell_deltas.csv", index=False)

    cmp_rows = []
    for name, block in (("M0", m0_metrics), ("M1", m1_metrics), ("M2", m2_metrics)):
        cmp_rows.append(
            {
                "model": name,
                "correct": block["correct"],
                "accuracy": block["accuracy"],
                "macro_f1": block["macro_f1"],
                "log_loss": block["log_loss"],
                "fold_0": block["fold_accuracy"]["0"],
                "fold_1": block["fold_accuracy"]["1"],
                "fold_2": block["fold_accuracy"]["2"],
                "fold_3": block["fold_accuracy"]["3"],
                "fold_4": block["fold_accuracy"]["4"],
                "folds_0_2": block["folds_0_2"],
                "folds_3_4": block["folds_3_4"],
                "hard_bucket": block["slices"]["hard_bucket_accuracy"],
                "non_hard_bucket": block["slices"]["non_hard_bucket_accuracy"],
                "neuron": block["slices"]["neuron_accuracy"],
                "glial": block["slices"]["glial_accuracy"],
            }
        )
    pd.DataFrame(cmp_rows).to_csv(OUT_DIR / "v3_e06m_candidate_comparison.csv", index=False)

    pred_after = sha256_file(PRED_PATH) if PRED_PATH.is_file() else None
    leak = {
        "competition_test_labels_used": False,
        "competition_train_labels_in_boosting_objective": False,
        "held_out_fold_labels_invisible": True,
        "competition_id_overlaps_excluded": True,
        "prohibited_exact_vector_duplicates_excluded": True,
        "hyperparameter_search": False,
        "seed_search": False,
        "source_weight_search": False,
        "leaderboard_feedback_used": False,
        "post_hoc_class_rule_tuning": False,
        "test_probabilities_used_for_model_selection": False,
        "prediction_csv_modified": bool(pred_before != pred_after),
        "frozen_artifacts_modified": False,
        "source_used_as_feature": False,
        "evaluation_term": (
            "honest external validation; boosting objective uses external reference labels "
            "only; fold-safe neighbor-label histograms match frozen V2"
        ),
    }
    if leak["prediction_csv_modified"]:
        return stop("STOP", "prediction/prediction.csv changed")

    integrity_ok = (
        w_audit["M2"]["balance_ok"]
        and contract_audit["status"] == "PASS"
        and not leak["prediction_csv_modified"]
        and n_mer == HISTORICAL_MERFISH_USABLE
    )
    label, reason = classify_m2(
        net_vs_m0=deltas["m2_vs_m0"]["overall"]["net_correction"],
        acc=m2_metrics["accuracy"],
        net_02=deltas["m2_vs_m0"]["folds_0_2"]["net_correction"],
        net_34=deltas["m2_vs_m0"]["folds_3_4"]["net_correction"],
        macro_f1=m2_metrics["macro_f1"],
        m0_macro_f1=m0_metrics["macro_f1"],
        m2_correct=m2_metrics["correct"],
        m1_correct=m1_metrics["correct"],
        sni_capture=sni_capture["m2"]["captured"],
        new_unique=m2_new_n,
        integrity_ok=integrity_ok,
    )
    decision = {
        "label": label,
        "reason": reason,
        "next_action": next_action_for(label, m1_metrics["correct"] > m2_metrics["correct"]),
        "criteria_not_changed_after_results": True,
    }

    complementarity = {
        "experiment_id": EXPERIMENT_ID,
        "oracle_is_not_deployable_accuracy": True,
        "four_expert_oracle": FOUR_EXPERT_ORACLE,
        "all_four_wrong": ALL_FOUR_WRONG,
        "sni_unique_recoveries": SNI_UNIQUE_RECOVERIES,
        "m1_new_unique_recoveries": m1_new_n,
        "m2_new_unique_recoveries": m2_new_n,
        "m1_five_expert_oracle_correct": five_m1,
        "m2_five_expert_oracle_correct": five_m2,
        "m2_all_five_wrong": five["m2_all_five_wrong"],
        "sni_capture": sni_capture,
        "anchor_retention": anchor_retention,
        "deltas": deltas,
        "m2_vs_m1": m2_vs_m1,
        "decision": decision,
        "runtime_seconds": time.perf_counter() - t0,
    }
    write_json(OUT_DIR / "v3_e06m_complementarity.json", complementarity)

    payload = {
        "dataset_manifest": manifest,
        "source_weight_audit": w_audit,
        "contract_audit": contract_audit,
        "m0": m0_metrics,
        "m1": m1_metrics,
        "m2": m2_metrics,
        "deltas": deltas,
        "m2_vs_m1": m2_vs_m1,
        "sni_capture": sni_capture,
        "anchor_retention": anchor_retention,
        "pairwise": pairwise_rows,
        "five_expert": five,
        "leakage_audit": leak,
        "decision": decision,
    }
    write_json(OUT_DIR / "v3_e06m_metrics.json", {
        "m0": {k: v for k, v in m0_metrics.items() if k != "per_class_recall" and k != "per_class_precision"},
        "m1": {k: v for k, v in m1_metrics.items() if k != "per_class_recall" and k != "per_class_precision"},
        "m2": {k: v for k, v in m2_metrics.items() if k != "per_class_recall" and k != "per_class_precision"},
        "decision": decision,
        "contract_audit": contract_audit,
    })
    write_report(payload)
    pd.DataFrame(m2_metrics["families"]).to_csv(TABLE_DIR / "m2_family_accuracy.csv", index=False)
    pd.DataFrame(m2_new_fam).to_csv(TABLE_DIR / "m2_new_unique_by_family.csv", index=False)

    print(
        "M0={:.4f} M1={:.4f} ({}) M2={:.4f} ({}) decision={}".format(
            m0_metrics["accuracy"],
            m1_metrics["accuracy"],
            m1_metrics["correct"],
            m2_metrics["accuracy"],
            m2_metrics["correct"],
            label,
        ),
        flush=True,
    )
    print("runtime_seconds={:.1f}".format(time.perf_counter() - t0), flush=True)
    print("wrote {}".format(REPORT_PATH), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
