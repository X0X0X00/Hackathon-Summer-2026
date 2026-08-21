#!/usr/bin/env python3
"""V3-E07D: Final Deployable Candidate Decision Audit (FDC-AUDIT).

Analysis only. Does not train a model, router, or stacker. Does not blend
experts, optimize ensemble weights, search thresholds, tune source weights,
add a dataset, start Spatial-ID, create M3, modify test predictions, freeze
MODEL V3, create a model-v3 tag, commit, or push.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "v3"))

from merfish60.io import TARGET_COL, load_dataset  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.official_contract import allowed_labels, sha256_file  # noqa: E402
from merfish60.team_cv import TEAM_FOLD_VALUES, load_and_validate_team_folds  # noqa: E402
from merfish60.v2_metrics import json_default, write_json  # noqa: E402

from v3_e00t_team_expert_audit import (  # noqa: E402
    LZH_BRANCH,
    LZH_FINAL_OOF,
    confidence_from_proba,
    load_probability_frame,
    materialize_git_blob,
)
from v3_e02d_privileged_gene_distillation import classification_metrics  # noqa: E402
from v3_e03a_rescue_audit import CLASS_FAMILIES_E03A, family_of  # noqa: E402
from v3_e06m_source_balanced_multireference import (  # noqa: E402
    STRONG_ACCURACY_MIN,
    STRONG_MACRO_F1_TOL,
    STRONG_NET_VS_M0,
    STRONG_NEW_UNIQUE,
    STRONG_SNI_CAPTURE,
    cell_delta,
    classify_m2,
)

EXPERIMENT_ID = "V3-E07D"
RESEARCH_CODENAME = "FDC-AUDIT"
N_TRAIN = 5000
N_TEST = 5000
N_CLASSES = 60
EXPECTED_BRANCH = "ywan/ml-pipeline"
BOOTSTRAP_SEED = 20260819
N_BOOTSTRAP = 10000
N_ECE_BINS = 10

M0_CORRECT = 4106
M0_ACCURACY = 0.8212
M0_MACRO_F1 = 0.793613323779027
M0_LOG_LOSS = 0.6416423806253077
M2_CORRECT = 4109
M2_ACCURACY = 0.8218
M2_MACRO_F1 = 0.7955218319991503
M2_LOG_LOSS = 0.6221664761889664
LZH_CORRECT = 4133
LZH_ACCURACY = 0.8266
M2_VS_M0_CHANGED = 252
M2_VS_M0_WRONG_TO_CORRECT = 105
M2_VS_M0_CORRECT_TO_WRONG = 102
M2_VS_M0_NET = 3
M2_VS_M0_NET_02 = 12
M2_VS_M0_NET_34 = -9
FOUR_EXPERT_ORACLE = 4364
FIVE_EXPERT_ORACLE = 4393
M2_NEW_UNIQUE = 29
S0_ACCURACY = 0.6302
SNI_ACCURACY = 0.5682
S0_CORRECT = 3151
SNI_CORRECT = 2841

E06M_STRONG_NET_VS_M0 = 25
E06M_STRONG_ACCURACY_MIN = 0.8262
E06M_FROZEN_LABEL = "PROMISING SOURCE-BALANCED TRANSFER"

ALLOWED_TEAM_DECISIONS = (
    "LZH REMAINS STRONGEST AUDITABLE STANDALONE EXPERT",
    "M2 IS STRONGEST AUDITABLE STANDALONE EXPERT",
    "M0 IS STRONGEST AUDITABLE STANDALONE EXPERT",
    "NO CLEAR STANDALONE WINNER",
)
ALLOWED_PERSONAL_DECISIONS = (
    "MODEL V3 PROMOTION JUSTIFIED",
    "MODEL V3 PROMOTION NOT JUSTIFIED",
)

FAMILY_ORDER = list(CLASS_FAMILIES_E03A.keys()) + ["neuronal_or_other"]
FAMILY_DISPLAY = OrderedDict(
    [
        ("oligodendrocyte_opc", "oligodendrocyte / OPC"),
        ("astrocyte", "astrocyte"),
        ("vascular", "vascular / endothelial"),
        ("meningeal", "meningeal"),
        ("microglia", "microglia"),
        ("remaining_glial_non_neuronal", "remaining glial"),
        ("neuronal_or_other", "neuronal / other"),
    ]
)
SLICE_ORDER = [
    ("hard_bucket", "hard bucket"),
    ("non_hard_bucket", "non-hard bucket"),
    ("neuron", "neuron"),
    ("glial", "glial / non-neuronal"),
] + [(name, FAMILY_DISPLAY[name]) for name in FAMILY_ORDER]

E00T_REGISTRY = ROOT / "outputs" / "v3" / "v3_e00t_team_oof_registry.parquet"
E05A_REGISTRY = ROOT / "outputs" / "v3" / "v3_e05a_four_expert_registry.parquet"
M0_OOF = ROOT / "outputs" / "oof" / "MODEL-V2_oof.csv"
M0_VAL_PROBA = ROOT / "outputs" / "probabilities" / "V2-B-REFONLY_oof_probabilities_seg.csv.gz"
M0_TEST_PROBA = ROOT / "outputs" / "probabilities" / "V2-B-REFONLY_test_probabilities_seg.csv.gz"
M0_METRICS = ROOT / "outputs" / "metrics" / "model_v2_metrics.json"
M2_VAL = ROOT / "outputs" / "v3" / "v3_e06m_m2_validation.csv"
M2_VAL_PROBA = ROOT / "outputs" / "v3" / "v3_e06m_m2_validation_probabilities.csv.gz"
M2_TEST_PROBA = ROOT / "outputs" / "v3" / "v3_e06m_m2_test_probabilities.csv.gz"
E06M_COMP = ROOT / "outputs" / "v3" / "v3_e06m_complementarity.json"
E06M_CMP = ROOT / "outputs" / "v3" / "v3_e06m_candidate_comparison.csv"
E06M_DELTAS = ROOT / "outputs" / "v3" / "v3_e06m_cell_deltas.csv"
E06M_NEW = ROOT / "outputs" / "v3" / "v3_e06m_new_unique_recoveries.csv"
PRED_PATH = ROOT / "prediction" / "prediction.csv"
OUT_DIR = ROOT / "outputs" / "v3"
TABLE_DIR = OUT_DIR / "v3_e07d_tables"
REPORT_PATH = ROOT / "reports" / "v3" / "v3_e07d_final_deployable_decision_audit.md"
MODEL_V3_DOC = ROOT / "docs" / "versions" / "model_v3.md"

FROZEN_PATHS = [
    E00T_REGISTRY,
    E05A_REGISTRY,
    M0_OOF,
    M0_VAL_PROBA,
    M0_TEST_PROBA,
    M0_METRICS,
    M2_VAL,
    M2_VAL_PROBA,
    M2_TEST_PROBA,
    E06M_COMP,
    E06M_CMP,
    E06M_DELTAS,
    E06M_NEW,
    ROOT / "docs" / "versions" / "model_v1.md",
    ROOT / "docs" / "versions" / "model_v2.md",
    ROOT / "outputs" / "submissions" / "model_v1.csv",
    ROOT / "outputs" / "submissions" / "model_v2_candidate.csv",
    ROOT / "reports" / "v3" / "v3_e00t_team_expert_audit.md",
    ROOT / "reports" / "v3" / "v3_e02d_privileged_gene_distillation.md",
    ROOT / "reports" / "v3" / "v3_e03a_rescue_audit.md",
    ROOT / "reports" / "v3" / "v3_e04s_sni_source_expert.md",
    ROOT / "reports" / "v3" / "v3_e05a_directional_complementarity_audit.md",
    ROOT / "reports" / "v3" / "v3_e06m_source_balanced_multireference.md",
    PRED_PATH,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(ROOT), stderr=subprocess.STDOUT).decode(
        "utf-8"
    )


def current_branch() -> str:
    return run_git(["branch", "--show-current"]).strip()


def integrity_failure(message: str) -> None:
    raise SystemExit("E07D FROZEN METRIC INTEGRITY FAILURE: {}".format(message))


def read_cell_id_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"Cell_ID": str}, keep_default_na=False)
    frame["Cell_ID"] = frame["Cell_ID"].astype(str)
    if frame["Cell_ID"].duplicated().any():
        raise ValueError("{} has duplicate Cell_ID".format(path))
    if frame["Cell_ID"].str.endswith(".0").any():
        raise ValueError("{} Cell_ID looks float-cast".format(path))
    if (~frame["Cell_ID"].str.fullmatch(r"\d{19}")).any():
        raise ValueError("{} has non 19-digit Cell_ID".format(path))
    return frame


def align_frame(frame: pd.DataFrame, cell_ids: Sequence[str], path: Path) -> pd.DataFrame:
    out = frame.set_index("Cell_ID").reindex([str(v) for v in cell_ids])
    if out.index.has_duplicates:
        integrity_failure("{} has duplicate Cell_ID after align".format(path))
    if out.index.isna().any() or out.isna().all(axis=1).any():
        integrity_failure("failed to align {} on lossless Cell_ID".format(path))
    return out


def align_proba(path: Path, cell_ids: Sequence[str], class_names: Sequence[str]) -> np.ndarray:
    frame, audit = load_probability_frame(path, class_names)
    if not audit["usable"]:
        integrity_failure("{} probabilities failed row audit".format(path))
    aligned = align_frame(frame, cell_ids, path)
    proba = aligned.loc[:, list(class_names)].to_numpy(dtype=np.float64)
    assert_probability_rows(proba, atol=1e-4)
    if proba.shape != (len(cell_ids), len(class_names)):
        integrity_failure("{} probability shape {}".format(path, proba.shape))
    return proba


def load_lzh_probabilities(cell_ids: Sequence[str], class_names: Sequence[str]) -> np.ndarray:
    refs = [
        "{}:{}".format(LZH_BRANCH, LZH_FINAL_OOF),
        "af51ce78846063f2ec9bede33b6b23b7a75d3492:{}".format(LZH_FINAL_OOF),
    ]
    last_error = None
    with tempfile.TemporaryDirectory(prefix="v3_e07d_lzh_") as tmp:
        dest = Path(tmp) / "lzh_oof.csv"
        for ref in refs:
            try:
                materialize_git_blob(ref, dest)
                frame, audit = load_probability_frame(dest, class_names)
                if not audit["usable"]:
                    raise RuntimeError("LZH probabilities failed row audit")
                aligned = align_frame(frame, cell_ids, dest)
                proba = aligned.loc[:, list(class_names)].to_numpy(dtype=np.float64)
                assert_probability_rows(proba, atol=1e-4)
                return proba
            except (subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
                last_error = exc
    integrity_failure("unable to load frozen LZH Prior-H OOF probabilities: {}".format(last_error))
    raise AssertionError("unreachable")


def yhh_status() -> dict:
    return {
        "included": False,
        "reason": (
            "No previously unavailable auditable 5000-cell YHH probability or prediction "
            "artifact is present and provenance-verified. YHH is excluded; it is not reconstructed."
        ),
    }


def team_main_v9_status() -> dict:
    return {
        "included": False,
        "reason": (
            "team main v9 is test/submission-oriented without the same auditable cell-level "
            "OOF protocol and is excluded from standalone ranking."
        ),
    }


def mcnemar_exact(wrong_to_correct: int, correct_to_wrong: int) -> dict:
    """Exact two-sided McNemar test via binomial on discordant cells.

    Null: among discordant cells, each model is equally likely to be correct.
    """
    b = int(wrong_to_correct)
    c = int(correct_to_wrong)
    n = b + c
    if n == 0:
        return {
            "wrong_to_correct": b,
            "correct_to_wrong": c,
            "n_discordant": 0,
            "net": b - c,
            "pvalue": 1.0,
            "method": "scipy.stats.binomtest two-sided p=0.5",
            "note": "no discordant cells",
        }
    result = binomtest(k=b, n=n, p=0.5, alternative="two-sided")
    return {
        "wrong_to_correct": b,
        "correct_to_wrong": c,
        "n_discordant": n,
        "net": b - c,
        "pvalue": float(result.pvalue),
        "method": "scipy.stats.binomtest two-sided p=0.5",
        "note": (
            "A non-significant result does not prove equivalence. It means the observed "
            "paired advantage does not provide strong evidence that the first listed "
            "model is superior among discordant cells."
        ),
    }


def paired_accuracy_deltas(
    ok_a: np.ndarray,
    ok_b: np.ndarray,
    seed: int = BOOTSTRAP_SEED,
    n_resamples: int = N_BOOTSTRAP,
) -> np.ndarray:
    ok_a = np.asarray(ok_a, dtype=np.float64)
    ok_b = np.asarray(ok_b, dtype=np.float64)
    if ok_a.shape != ok_b.shape:
        raise ValueError("paired bootstrap inputs must have the same shape")
    n = int(ok_a.shape[0])
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, n, size=(int(n_resamples), n), endpoint=False)
    return ok_b[index].mean(axis=1) - ok_a[index].mean(axis=1)


def summarize_deltas(deltas: np.ndarray) -> dict:
    deltas = np.asarray(deltas, dtype=np.float64)
    return {
        "n_resamples": int(deltas.size),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "p2_5": float(np.percentile(deltas, 2.5)),
        "p97_5": float(np.percentile(deltas, 97.5)),
        "fraction_gt_0": float(np.mean(deltas > 0)),
        "fraction_eq_0": float(np.mean(deltas == 0)),
        "fraction_lt_0": float(np.mean(deltas < 0)),
    }


def section_cluster_deltas(
    section_ids: np.ndarray,
    ok_a: np.ndarray,
    ok_b: np.ndarray,
    seed: int = BOOTSTRAP_SEED,
    n_resamples: int = N_BOOTSTRAP,
) -> np.ndarray:
    section_ids = np.asarray(section_ids)
    ok_a = np.asarray(ok_a, dtype=np.float64)
    ok_b = np.asarray(ok_b, dtype=np.float64)
    unique = np.array(sorted(set(section_ids.tolist())), dtype=object)
    n_sec = int(unique.size)
    n_cells = np.zeros(n_sec, dtype=np.int64)
    a_corr = np.zeros(n_sec, dtype=np.float64)
    b_corr = np.zeros(n_sec, dtype=np.float64)
    lookup = {sec: i for i, sec in enumerate(unique.tolist())}
    for i, sec in enumerate(section_ids.tolist()):
        j = lookup[sec]
        n_cells[j] += 1
        a_corr[j] += ok_a[i]
        b_corr[j] += ok_b[i]
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, n_sec, size=(int(n_resamples), n_sec), endpoint=False)
    n_tot = n_cells[sampled].sum(axis=1).astype(np.float64)
    a_sum = a_corr[sampled].sum(axis=1)
    b_sum = b_corr[sampled].sum(axis=1)
    return (b_sum / n_tot) - (a_sum / n_tot)


def js_divergence_vectors(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """Jensen-Shannon divergence (not distance). Deterministic scipy implementation."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    dist = float(jensenshannon(p, q, base=base))
    if not np.isfinite(dist):
        return 0.0
    return float(dist * dist)


def predicted_class_frequency(pred: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    pred = np.asarray(pred, dtype=object)
    names = list(class_names)
    counts = np.array([(pred == name).sum() for name in names], dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return np.full(len(names), 1.0 / len(names), dtype=np.float64)
    return counts / total


def equal_width_ece(
    correct: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = N_ECE_BINS,
) -> dict:
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    n = int(len(correct))
    bin_id = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = bin_id == b
        n_b = int(mask.sum())
        left = b / n_bins
        right = (b + 1) / n_bins
        if n_b == 0:
            bins.append(
                {
                    "bin": b,
                    "left": left,
                    "right": right,
                    "n": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "abs_gap": None,
                }
            )
            continue
        mean_conf = float(confidence[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(acc - mean_conf)
        ece += (n_b / n) * gap
        bins.append(
            {
                "bin": b,
                "left": left,
                "right": right,
                "n": n_b,
                "mean_confidence": mean_conf,
                "accuracy": acc,
                "abs_gap": gap,
            }
        )
    return {"n_bins": n_bins, "ece": float(ece), "bins": bins}


def confidence_deciles(correct: np.ndarray, confidence: np.ndarray) -> List[dict]:
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    n = int(len(correct))
    ranks = pd.Series(confidence).rank(method="first").to_numpy(dtype=np.float64)
    decile = np.ceil(ranks / n * 10.0).astype(np.int64)
    decile = np.clip(decile, 1, 10)
    rows = []
    for d in range(1, 11):
        mask = decile == d
        n_d = int(mask.sum())
        rows.append(
            {
                "decile": d,
                "n": n_d,
                "mean_confidence": float(confidence[mask].mean()) if n_d else None,
                "accuracy": float(correct[mask].mean()) if n_d else None,
            }
        )
    return rows


def fold_stability(correct: np.ndarray, folds: np.ndarray) -> dict:
    correct = np.asarray(correct, dtype=bool)
    folds = np.asarray(folds)
    fold_acc = {}
    fold_n = {}
    fold_correct = {}
    values = []
    for fold_id in TEAM_FOLD_VALUES:
        mask = folds == fold_id
        n = int(mask.sum())
        c = int(np.sum(correct[mask]))
        acc = float(c / n) if n else None
        fold_acc[str(fold_id)] = acc
        fold_n[str(fold_id)] = n
        fold_correct[str(fold_id)] = c
        if acc is not None:
            values.append(acc)
    arr = np.asarray(values, dtype=np.float64)
    mask02 = folds <= 2
    mask34 = folds >= 3
    return {
        "fold_n": fold_n,
        "fold_correct": fold_correct,
        "fold_accuracy": fold_acc,
        "folds_0_2_n": int(mask02.sum()),
        "folds_3_4_n": int(mask34.sum()),
        "folds_0_2_correct": int(np.sum(correct[mask02])),
        "folds_3_4_correct": int(np.sum(correct[mask34])),
        "folds_0_2": float(np.mean(correct[mask02])),
        "folds_3_4": float(np.mean(correct[mask34])),
        "mean_fold_accuracy": float(np.mean(arr)),
        "std_fold_accuracy": float(np.std(arr, ddof=1)),
        "min_fold_accuracy": float(np.min(arr)),
        "max_fold_accuracy": float(np.max(arr)),
    }


def pair_accounting(ok_a: np.ndarray, ok_b: np.ndarray, pred_a=None, pred_b=None) -> dict:
    ok_a = np.asarray(ok_a, dtype=bool)
    ok_b = np.asarray(ok_b, dtype=bool)
    both_correct = int((ok_a & ok_b).sum())
    a_only = int((ok_a & ~ok_b).sum())
    b_only = int((~ok_a & ok_b).sum())
    both_wrong = int((~ok_a & ~ok_b).sum())
    payload = {
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "net_b_minus_a": b_only - a_only,
        "pair_oracle_correct": both_correct + a_only + b_only,
        "pair_oracle_accuracy": float((both_correct + a_only + b_only) / len(ok_a)),
        "discordant": a_only + b_only,
    }
    if pred_a is not None and pred_b is not None:
        payload["changed_predictions"] = int(np.sum(np.asarray(pred_a) != np.asarray(pred_b)))
    return payload


def candidate_block(
    name: str,
    pred: np.ndarray,
    correct: np.ndarray,
    folds: np.ndarray,
    proba: Optional[np.ndarray],
    class_names: Sequence[str],
    y_true: np.ndarray,
    conf: Optional[dict],
) -> dict:
    metrics = classification_metrics(y_true, pred, class_names, proba)
    stab = fold_stability(correct, folds)
    block = {
        "name": name,
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "macro_f1": metrics["macro_f1"],
        "log_loss": metrics["log_loss"],
        **stab,
    }
    if conf is not None:
        block["mean_top1"] = float(np.mean(conf["top1"]))
        block["mean_margin"] = float(np.mean(conf["margin"]))
        block["mean_entropy"] = float(np.mean(conf["entropy"]))
    return block


def slice_masks(registry: pd.DataFrame) -> OrderedDict:
    masks = OrderedDict()
    hard = registry["hard_bucket"].to_numpy(dtype=bool)
    masks["hard_bucket"] = hard
    masks["non_hard_bucket"] = ~hard
    neuron = registry["neuron_or_glial"].astype(str).to_numpy() == "neuron"
    glial = registry["neuron_or_glial"].astype(str).to_numpy() == "glial_non_neuronal"
    masks["neuron"] = neuron
    masks["glial"] = glial
    fam = registry["biological_family"].astype(str).to_numpy()
    for name in FAMILY_ORDER:
        masks[name] = fam == name
    return masks


def strong_criteria_met(
    net: int,
    acc: float,
    net_02: int,
    net_34: int,
    macro_f1: float,
    m0_macro_f1: float,
    sni_capture: int,
    new_unique: int,
    m2_correct: int,
    m1_correct: int,
) -> Tuple[bool, List[str]]:
    checks = OrderedDict(
        [
            ("net_ge_25", net >= E06M_STRONG_NET_VS_M0),
            ("acc_ge_0.8262", acc >= E06M_STRONG_ACCURACY_MIN),
            ("folds_0_2_net_positive", net_02 > 0),
            ("folds_3_4_net_positive", net_34 > 0),
            ("macro_f1_not_materially_worse", macro_f1 >= m0_macro_f1 - STRONG_MACRO_F1_TOL),
            ("m2_ge_m1", m2_correct >= m1_correct),
            ("sni_capture_ge_15", sni_capture >= STRONG_SNI_CAPTURE),
            ("new_unique_ge_15", new_unique >= STRONG_NEW_UNIQUE),
        ]
    )
    failed = [name for name, ok in checks.items() if not ok]
    return bool(all(checks.values())), failed


def personal_promotion_decision(
    e06m_label: str,
    strong_ok: bool,
    m2_correct: int,
    m0_correct: int,
    net: int,
    net_34: int,
) -> str:
    """Promotion is justified only if frozen E06M STRONG criteria were met.

    A +3-cell M2 vs M0 difference does not lower the bar.
    """
    del m2_correct, m0_correct, net, net_34
    if strong_ok and e06m_label == "STRONG SOURCE-BALANCED TRANSFER":
        return "MODEL V3 PROMOTION JUSTIFIED"
    return "MODEL V3 PROMOTION NOT JUSTIFIED"


def team_standalone_decision(
    lzh_correct: int,
    m2_correct: int,
    m0_correct: int,
    lzh_net_vs_m0: int,
    lzh_net_vs_m2: int,
    m2_net_vs_m0: int,
    lzh_folds_34: float,
    m2_folds_34: float,
    m0_folds_34: float,
) -> str:
    ranking = sorted(
        [("LZH", lzh_correct), ("M2", m2_correct), ("M0", m0_correct)],
        key=lambda item: (-item[1], item[0]),
    )
    if ranking[0][1] == ranking[1][1]:
        return "NO CLEAR STANDALONE WINNER"
    winner = ranking[0][0]
    if winner == "LZH" and lzh_net_vs_m0 > 0 and lzh_net_vs_m2 > 0:
        if lzh_folds_34 + 1e-12 >= min(m0_folds_34, m2_folds_34):
            return "LZH REMAINS STRONGEST AUDITABLE STANDALONE EXPERT"
        return "LZH REMAINS STRONGEST AUDITABLE STANDALONE EXPERT"
    if winner == "M2" and m2_net_vs_m0 > 0 and lzh_net_vs_m2 < 0:
        return "M2 IS STRONGEST AUDITABLE STANDALONE EXPERT"
    if winner == "M0":
        return "M0 IS STRONGEST AUDITABLE STANDALONE EXPERT"
    return "NO CLEAR STANDALONE WINNER"


def fmt_acc(value) -> str:
    if value is None:
        return "n/a"
    return "{:.4f}".format(float(value))


def fmt_p(value) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if value < 0.001:
        return "{:.3e}".format(value)
    return "{:.4f}".format(value)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def next_action_for(personal: str, team: str) -> str:
    del team
    if personal == "MODEL V3 PROMOTION JUSTIFIED":
        return (
            "Begin a separately reviewed MODEL V3 freeze checklist around M2 only after a "
            "human confirms the frozen E06M STRONG criteria and this audit. Do not blend, "
            "route, or retune first."
        )
    return (
        "Close the personal V3 research program without creating MODEL V3. Keep frozen "
        "MODEL V2 as the personal deployable candidate. Treat LZH Prior-H as the strongest "
        "currently auditable standalone team expert for any later separately reviewed "
        "team-selection discussion. Do not blend M2, train a router, tune source weights, "
        "add Spatial-ID, or create a model-v3 tag."
    )


def render_report(m: dict) -> str:
    cand = m["candidates"]
    m0 = cand["M0"]
    m2 = cand["M2"]
    lzh = cand["LZH"]
    paired = m["m2_vs_m0"]
    boot = m["bootstrap"]["cell_level"]
    sec_boot = m["bootstrap"]["section_cluster"]
    sec = m["section_stability"]
    rare = m["rare_class"]
    cal = m["calibration"]
    shift = m["test_shift"]
    lzh_m0 = m["lzh_vs_m0"]
    lzh_m2 = m["lzh_vs_m2"]
    cls = m["classwise"]
    slices = m["slices"]
    integ = m["integrity"]
    e06m = m["e06m_strong_audit"]

    slice_rows = []
    for key, display in SLICE_ORDER:
        row = slices[key]
        slice_rows.append(
            [
                display,
                row["n"],
                fmt_acc(row["M0"]),
                fmt_acc(row["M2"]),
                fmt_acc(row["LZH"]),
                "{:+.4f}".format(row["M2_minus_M0"]),
                "{:+.4f}".format(row["LZH_minus_M0"]),
                "{:+.4f}".format(row["LZH_minus_M2"]),
            ]
        )

    def fold_row(block):
        fa = block["fold_accuracy"]
        return [
            block["name"],
            "{:.3f}".format(fa["0"]),
            "{:.3f}".format(fa["1"]),
            "{:.3f}".format(fa["2"]),
            "{:.3f}".format(fa["3"]),
            "{:.3f}".format(fa["4"]),
            fmt_acc(block["folds_0_2"]),
            fmt_acc(block["folds_3_4"]),
            fmt_acc(block["mean_fold_accuracy"]),
            "{:.4f}".format(block["std_fold_accuracy"]),
            fmt_acc(block["min_fold_accuracy"]),
            fmt_acc(block["max_fold_accuracy"]),
        ]

    gain_rows = [["class", "support", "M0 recall", "M2 recall", "delta"]]
    for row in cls["top10_m2_gains"]:
        gain_rows.append(
            [
                row["class"],
                row["support"],
                fmt_acc(row["m0_recall"]),
                fmt_acc(row["m2_recall"]),
                "{:+.4f}".format(row["m2_minus_m0"]),
            ]
        )
    loss_rows = [["class", "support", "M0 recall", "M2 recall", "delta"]]
    for row in cls["top10_m2_losses"]:
        loss_rows.append(
            [
                row["class"],
                row["support"],
                fmt_acc(row["m0_recall"]),
                fmt_acc(row["m2_recall"]),
                "{:+.4f}".format(row["m2_minus_m0"]),
            ]
        )

    failed = ", ".join(e06m["failed_strong_checks"]) if e06m["failed_strong_checks"] else "none"

    return """# V3-E07D — Final Deployable Candidate Decision Audit

Research codename: **FDC-AUDIT**. This experiment is ANALYSIS ONLY. It does not train a model, router, or stacker; does not blend experts; does not optimize ensemble weights, thresholds, or source weights; does not add a dataset; does not start Spatial-ID; does not create M3; does not modify test predictions; and does not freeze MODEL V3.

## 1. Decision Question

Which currently auditable standalone expert is strongest for deployment? Does M2 provide enough stable evidence to replace frozen personal MODEL V2? Is promotion to MODEL V3 scientifically justified? What should be frozen as the final conclusion of the current V3 research program?

Oracle values below are diagnostic coverage ceilings. **ORACLE != DEPLOYABLE ACCURACY.**

## 2. Frozen V3 Research Context

The current program is a sequence of controlled experiments, not a version freeze:

| Experiment | Result |
|---|---|
| V3-E00T Team Canonical Expert & Complementarity Audit | ROUTING-ONLY INSUFFICIENT for the then-auditable LZH+WYH pool; diagnostic two-expert oracle 0.8430 |
| V3-E02D Privileged-Gene Dual-Level Distillation | PROMISING BUT INSUFFICIENT; 500-gene teacher signal exists, but the 200-gene student was not a mature third expert |
| V3-E03A Weak-but-Diverse Expert Rescue Audit | RESCUE SIGNAL WEAK; S0 unique recoveries exist but test-safe confidence cannot identify them |
| V3-E04S Source-Diverse SNI Expert | STRONG SOURCE-DIVERSE EXPERT as a diversity/oracle member (standalone 0.5682); 53 unique recoveries |
| V3-E05A Asymmetric Directional Complementarity Audit | DIRECTIONAL SIGNAL WEAK; predeclared S0/SNI directional overrides had negative net correction |
| V3-E06M Source-Balanced Multi-Reference Transfer | PROMISING SOURCE-BALANCED TRANSFER, not STRONG; M2 = 4109 vs M0 = 4106, net +3, folds 3-4 net -9 |

Frozen personal versions remain MODEL V1 (historical) and MODEL V2 (current deployable). MODEL V3 is not defined.

## 3. Candidate Eligibility

Primary personal comparison: frozen WYH MODEL V2 (**M0**) versus V3-E06M source-balanced MERFISH+SNI (**M2**).

Team-level auditable standalone comparison additionally includes **LZH Prior-H**.

Excluded from standalone ranking:

| Candidate | Why excluded |
|---|---|
| S0 (accuracy {s0_acc:.4f}) | Diversity / oracle-analysis expert only; not a deployable standalone candidate |
| SNI (accuracy {sni_acc:.4f}) | Diversity / oracle-analysis expert only; not a deployable standalone candidate |
| team main v9 | Test/submission-oriented without the same auditable cell-level OOF protocol |
| YHH | {yhh} |

No YHH reconstruction was attempted.

## 4. Integrity Reproduction

All frozen identity checks reproduced exactly before any decision audit:

{integ_table}

Re-evaluating the frozen E06M STRONG classifier on these reproduced numbers returns **{e06m_label}**, not STRONG SOURCE-BALANCED TRANSFER. Failed STRONG checks: {failed}.

E07D does not lower the promotion bar after observing E06M. The frozen STRONG thresholds remain net correction ≥ {strong_net}, accuracy ≥ {strong_acc:.4f}, positive net on both folds 0-2 and folds 3-4, and the remaining predeclared SNI-capture / new-unique / macro-F1 conditions.

## 5. Overall Standalone Performance

{overall_table}

LZH Prior-H has the highest auditable standalone accuracy. M2 is +3 cells versus M0 and remains below LZH.

## 6. Fold Stability

Canonical analysis partition: `experiments/team_folds_5_seed42.csv`. Folds 3-4 are a retrospective stability partition, not an untouched holdout.

{fold_table}

M2 improves folds 0-2 versus M0 (net {net02:+d} cells) and regresses folds 3-4 (net {net34:+d} cells). LZH remains ahead of both personal candidates on folds 0-2 and does not share M2's folds 3-4 regression versus M0.

## 7. Standard Deployment Slices

Slice comparison is deployment characterization only. No model is chosen from a post-hoc favorable slice.

{slice_table}

## 8. M2 vs M0 Paired Cell-Level Audit

This is the primary personal-version comparison.

- Changed predictions: **{changed}**
- M2 wrong → correct: **{w2c}**
- M2 correct → wrong: **{c2w}**
- Net: **{net:+d}**
- Discordant correctness count: **{disc}**
- Exact two-sided McNemar p-value (`scipy.stats.binomtest` on discordant cells): **{p_m2m0}**

A non-significant result does **not** prove equivalence. It means the observed +3-cell advantage does not provide strong paired evidence that M2 is superior. Offset errors remain almost as large as the recoveries: 105 vs 102.

## 9. Paired Bootstrap Uncertainty

Deterministic seed `{seed}`, {nboot} paired cell-level bootstrap resamples of accuracy(M2) − accuracy(M0). Diagnostic uncertainty only. Not used to invent a new promotion threshold.

| Quantity | Value |
|---|---|
| Mean delta | {b_mean:+.6f} |
| Median delta | {b_med:+.6f} |
| 2.5th percentile | {b_lo:+.6f} |
| 97.5th percentile | {b_hi:+.6f} |
| Fraction delta > 0 | {b_gt:.4f} |
| Fraction delta = 0 | {b_eq:.4f} |
| Fraction delta < 0 | {b_lt:.4f} |

The 95% percentile interval includes 0. Bootstrap agreement with a tiny positive mean does not convert PROMISING E06M evidence into STRONG promotion evidence.

## 10. Section-Level Robustness

Cells are grouped by `Section_ID`. Per-section M2−M0 accuracy deltas:

| Quantity | Value |
|---|---|
| Sections | {n_sec} |
| M2 > M0 | {sec_win} |
| M2 = M0 | {sec_tie} |
| M2 < M0 | {sec_lose} |
| Median per-section accuracy delta | {sec_med:+.6f} |
| IQR | {sec_iqr:.6f} |
| Minimum | {sec_min:+.6f} |
| Maximum | {sec_max:+.6f} |

Section-cluster bootstrap (seed `{seed}`, {nboot} resamples of Sections with replacement):

| Quantity | Value |
|---|---|
| Mean delta | {sb_mean:+.6f} |
| Median delta | {sb_med:+.6f} |
| 2.5th percentile | {sb_lo:+.6f} |
| 97.5th percentile | {sb_hi:+.6f} |
| Fraction > 0 | {sb_gt:.4f} |
| Fraction < 0 | {sb_lt:.4f} |

M2's overall +3-cell gain is **not a broad, section-stable improvement**. Wins and losses both exist; the cluster-bootstrap interval includes 0. The gain is small and compatible with concentrated / unstable section-level fluctuation.

## 11. Classwise / Rare-Class Stability

No class-specific routing or patch rule is created.

Classes where M2 improves over M0: **{n_gain}**. Tied: **{n_tie}**. Worsens: **{n_loss}**.

Top 10 M2 recall gains vs M0:

{gain_table}

Top 10 M2 recall losses vs M0:

{loss_table}

Macro-F1: M0 {m0_f1:.4f}; M2 {m2_f1:.4f}; LZH {lzh_f1:.4f}. M2 does not show a material macro-F1 regression versus M0.

Rare-class buckets (true-class support):

{rare_table}

The +3-cell overall result did not come from a large, obvious minority-class collapse, but neither did it produce a robust rare-class gain that could justify replacing MODEL V2.

## 12. Calibration Characterization

No temperature scaling, confidence threshold, or calibration tuning was performed.

| Model | Log loss | Mean top1 | Mean margin | Mean entropy | ECE (10 equal-width bins) |
|---|---:|---:|---:|---:|---:|
| M0 | {m0_ll:.4f} | {m0_t1:.4f} | {m0_mg:.4f} | {m0_ent:.4f} | {m0_ece:.4f} |
| M2 | {m2_ll:.4f} | {m2_t1:.4f} | {m2_mg:.4f} | {m2_ent:.4f} | {m2_ece:.4f} |

M2 has lower log loss than M0. That is a characterization result only. It is not a promotion criterion and is not used to set thresholds.

Accuracy by confidence decile is in `outputs/v3/v3_e07d_calibration_audit.csv`.

## 13. Test-Side Descriptive Shift

**NO TEST LABELS WERE USED.** Hidden test labels were not read, inferred, or scored.

Frozen M0 and M2 test probabilities are compared descriptively with validation probabilities.

| Quantity | M0 | M2 |
|---|---:|---:|
| Validation mean top1 | {m0_t1:.4f} | {m2_t1:.4f} |
| Test mean top1 | {m0_te_t1:.4f} | {m2_te_t1:.4f} |
| Validation mean entropy | {m0_ent:.4f} | {m2_ent:.4f} |
| Test mean entropy | {m0_te_ent:.4f} | {m2_te_ent:.4f} |
| Validation vs test predicted-class JS divergence | {m0_js:.6f} | {m2_js:.6f} |

M0 vs M2 test prediction agreement: **{test_agree:.4f}** ({test_agree_n} / {n_test}). Changed test predictions: **{test_changed}**.

These figures are not test accuracy. They are not used to choose thresholds, weights, classes, or post-processing.

## 14. LZH vs M0 / M2

No LZH blend, router, or stacker is formed.

### LZH vs M0

| Quantity | Value |
|---|---:|
| Both correct | {lm0_both} |
| LZH-only correct | {lm0_lzh} |
| M0-only correct | {lm0_m0} |
| Both wrong | {lm0_wrong} |
| Net (LZH − M0) | {lm0_net:+d} |
| Pair oracle | {lm0_oracle} / {n} = {lm0_oracle_acc:.4f} |
| Exact McNemar p-value | {lm0_p} |
| Folds 0-2 net | {lm0_02:+d} |
| Folds 3-4 net | {lm0_34:+d} |

Pair oracle is diagnostic only.

### LZH vs M2

| Quantity | Value |
|---|---:|
| Both correct | {lm2_both} |
| LZH-only correct | {lm2_lzh} |
| M2-only correct | {lm2_m2} |
| Both wrong | {lm2_wrong} |
| Net (LZH − M2) | {lm2_net:+d} |
| Pair oracle | {lm2_oracle} / {n} = {lm2_oracle_acc:.4f} |
| Exact McNemar p-value | {lm2_p} |
| Folds 0-2 net | {lm2_02:+d} |
| Folds 3-4 net | {lm2_34:+d} |

LZH remains ahead of both personal candidates overall. M2 does not overtake LZH, and pairing M2 with LZH is not a deployment action in this audit.

## 15. Strongest Auditable Standalone Expert

Standalone ranking uses overall accuracy, paired evidence, fold stability, and slice characterization. Diagnostic oracle coverage is **not** used.

**{team_decision}**

LZH Prior-H is the strongest currently auditable standalone expert: {lzh_correct} / {n} = {lzh_acc:.4f}, with a positive paired net versus both M0 and M2. M2 is not a standalone replacement for LZH. M0 remains the frozen personal deployable model, not the strongest team-auditable expert.

## 16. Personal MODEL V3 Promotion Decision

Personal version decision is separate from the team standalone ranking.

Frozen E06M STRONG promotion evidence required, among other conditions: M2 net correction versus M0 ≥ +25 cells, M2 accuracy ≥ 0.8262, positive net on both folds 0-2 and folds 3-4, no material macro-F1 regression, and sufficient SNI-signal / new-unique capture.

Frozen and reproduced E06M result: M2 = {m2_correct}, M0 = {m0_correct}, net {net:+d}, folds 3-4 net {net34:+d}, label **PROMISING SOURCE-BALANCED TRANSFER**.

**{personal_decision}**

This is preregistration discipline, not a subjective downgrade of a 4109 vs 4106 difference. E07D does not retroactively lower the criterion because a three-cell gain was observed.

## 17. Why Diagnostic Oracle Is Not Deployable Accuracy

Frozen four-expert diagnostic oracle: **{four} / {n} = {four_acc:.4f}**.

Frozen five-expert diagnostic oracle including M2: **{five} / {n} = {five_acc:.4f}**.

M2 new unique recoveries beyond the frozen four-expert pool: **{new_unique}**.

Identity: 4364 + 29 = 4393. These numbers measure coverage if a perfect selector existed. They are not achievable OOF scores. V3-E03A and V3-E05A showed that converting complementary coverage into net-correct cells is the actual bottleneck: weak-expert confidence rescue and directional overrides both produced offsetting errors. V3-E06M internalized some source diversity into one model and still converted that coverage into only +3 net cells, with folds 3-4 negative.

Do not call 0.8786 an achievable OOF score.

## 18. Final V3 Research Synthesis

The current bottleneck is not lack of expert coverage. The auditable expert pool reaches a **0.8786 diagnostic oracle**, but multiple controlled experiments show that complementary information is difficult to convert into stable deployable gains without causing offsetting errors.

Evidence chain, without exaggeration:

- E00T: LZH and MODEL V2 are complementary (pair oracle 0.8430), but routing-only over that two-expert pool was insufficient.
- E02D: privileged 500-gene information exists, but dual-level distillation did not yield a mature deployable third expert.
- E03A: S0 contributes 96 unique recoveries, but available test-safe signals cannot isolate them at usable precision.
- E04S: SNI is a weak global classifier and a real source-diversity expert (53 unique recoveries; four-expert oracle 0.8728).
- E05A: the obvious directional patches over that complementarity have negative net correction.
- E06M: source-balanced training is better than naive pooling and slightly better than M0 overall, but not STRONG, and folds 3-4 regress.
- E07D: paired, bootstrap, section-cluster, and slice audits do not convert that +3-cell result into a justified personal version freeze.

Deployable accuracy and diagnostic oracle coverage remain distinct quantities. MODEL V2 stays the frozen personal deployable model because the V3 program did not produce a robust replacement.

## 19. Limitations

- Canonical folds 3-4 have been viewed in prior V3 stages and are retrospective.
- LZH Prior-H uses a different original 3-fold protocol; canonical 5-fold numbers are a common evaluation partition, not LZH's native validation.
- LZH 0.8266 is also a selection metric on that OOF bundle (MEDIUM validation confidence from V3-E00T).
- YHH and team main v9 remain unavailable for honest cell-level standalone comparison.
- SNI labels are consensus `voting` labels, not a wet-lab gold standard.
- Bootstrap and McNemar assume the resampled units; the section-cluster bootstrap is the dependence-aware check, not a new selection rule.
- No temperature scaling or threshold was fit; calibration numbers are descriptive.
- Test-side JS divergence is a predicted-class distribution comparison only. It is not test accuracy.

## 20. Final Decision

TEAM STANDALONE DECISION:
{team_decision}

PERSONAL VERSION DECISION:
{personal_decision}

Recommended next action (do not start it automatically):

{next_action}

Project state if V3 is not promoted, which is the decision above:

- MODEL V1: frozen historical version
- MODEL V2: current frozen personal deployable model
- V3 Research Program: completed controlled research program
- V3-E00T: expert complementarity audit
- V3-E02D: privileged-gene transfer negative result
- V3-E03A: weak-expert confidence rescue negative result
- V3-E04S: strong positive source-diversity discovery
- V3-E05A: directional routing negative result
- V3-E06M: promising but insufficient source-balanced transfer

Do not create a weaker or statistically indistinguishable MODEL V3 merely for version numbering.
""".format(
        s0_acc=S0_ACCURACY,
        sni_acc=SNI_ACCURACY,
        yhh=m["eligibility"]["yhh"]["reason"],
        integ_table=md_table(
            ["Check", "Expected", "Reproduced"],
            [
                ["M0 / MODEL V2 correct", M0_CORRECT, integ["m0_correct"]],
                ["M0 accuracy", "{:.4f}".format(M0_ACCURACY), "{:.4f}".format(integ["m0_accuracy"])],
                ["M2 correct", M2_CORRECT, integ["m2_correct"]],
                ["M2 accuracy", "{:.4f}".format(M2_ACCURACY), "{:.4f}".format(integ["m2_accuracy"])],
                ["M2 vs M0 wrong→correct", M2_VS_M0_WRONG_TO_CORRECT, integ["m2_wrong_to_correct"]],
                ["M2 vs M0 correct→wrong", M2_VS_M0_CORRECT_TO_WRONG, integ["m2_correct_to_wrong"]],
                ["M2 vs M0 net", M2_VS_M0_NET, integ["m2_net"]],
                ["Folds 0-2 net", M2_VS_M0_NET_02, integ["m2_net_02"]],
                ["Folds 3-4 net", M2_VS_M0_NET_34, integ["m2_net_34"]],
                ["LZH correct", LZH_CORRECT, integ["lzh_correct"]],
                ["Four-expert oracle", FOUR_EXPERT_ORACLE, integ["four_expert_oracle"]],
                ["Five-expert oracle", FIVE_EXPERT_ORACLE, integ["five_expert_oracle"]],
                ["M2 new unique recoveries", M2_NEW_UNIQUE, integ["m2_new_unique"]],
            ],
        ),
        e06m_label=e06m["reproduced_label"],
        failed=failed,
        strong_net=E06M_STRONG_NET_VS_M0,
        strong_acc=E06M_STRONG_ACCURACY_MIN,
        overall_table=md_table(
            ["Model", "Correct", "Accuracy", "Macro-F1", "Log loss"],
            [
                ["M0 / MODEL V2", m0["correct"], fmt_acc(m0["accuracy"]), fmt_acc(m0["macro_f1"]), "{:.4f}".format(m0["log_loss"])],
                ["M2 / V3-E06M", m2["correct"], fmt_acc(m2["accuracy"]), fmt_acc(m2["macro_f1"]), "{:.4f}".format(m2["log_loss"])],
                ["LZH Prior-H", lzh["correct"], fmt_acc(lzh["accuracy"]), fmt_acc(lzh["macro_f1"]), "{:.4f}".format(lzh["log_loss"])],
            ],
        ),
        fold_table=md_table(
            [
                "Model",
                "Fold 0",
                "Fold 1",
                "Fold 2",
                "Fold 3",
                "Fold 4",
                "Folds 0-2",
                "Folds 3-4",
                "Mean",
                "Std",
                "Min",
                "Max",
            ],
            [fold_row(m0), fold_row(m2), fold_row(lzh)],
        ),
        net02=paired["folds_0_2"]["net"],
        net34=paired["folds_3_4"]["net"],
        slice_table=md_table(
            ["Slice", "n", "M0", "M2", "LZH", "M2−M0", "LZH−M0", "LZH−M2"],
            slice_rows,
        ),
        changed=paired["overall"]["changed_predictions"],
        w2c=paired["overall"]["wrong_to_correct"],
        c2w=paired["overall"]["correct_to_wrong"],
        net=paired["overall"]["net"],
        disc=paired["mcnemar"]["n_discordant"],
        p_m2m0=fmt_p(paired["mcnemar"]["pvalue"]),
        seed=BOOTSTRAP_SEED,
        nboot=N_BOOTSTRAP,
        b_mean=boot["mean_delta"],
        b_med=boot["median_delta"],
        b_lo=boot["p2_5"],
        b_hi=boot["p97_5"],
        b_gt=boot["fraction_gt_0"],
        b_eq=boot["fraction_eq_0"],
        b_lt=boot["fraction_lt_0"],
        n_sec=sec["n_sections"],
        sec_win=sec["m2_gt_m0"],
        sec_tie=sec["m2_eq_m0"],
        sec_lose=sec["m2_lt_m0"],
        sec_med=sec["median_accuracy_delta"],
        sec_iqr=sec["iqr_accuracy_delta"],
        sec_min=sec["min_accuracy_delta"],
        sec_max=sec["max_accuracy_delta"],
        sb_mean=sec_boot["mean_delta"],
        sb_med=sec_boot["median_delta"],
        sb_lo=sec_boot["p2_5"],
        sb_hi=sec_boot["p97_5"],
        sb_gt=sec_boot["fraction_gt_0"],
        sb_lt=sec_boot["fraction_lt_0"],
        n_gain=cls["n_m2_improves"],
        n_tie=cls["n_tied"],
        n_loss=cls["n_m2_worsens"],
        gain_table=md_table(gain_rows[0], gain_rows[1:]),
        loss_table=md_table(loss_rows[0], loss_rows[1:]),
        m0_f1=m0["macro_f1"],
        m2_f1=m2["macro_f1"],
        lzh_f1=lzh["macro_f1"],
        rare_table=md_table(
            ["Support bucket", "n classes", "n cells", "M0 mean recall", "M2 mean recall", "M0 mean F1", "M2 mean F1"],
            [
                [
                    row["bucket"],
                    row["n_classes"],
                    row["n_cells"],
                    fmt_acc(row["m0_mean_recall"]),
                    fmt_acc(row["m2_mean_recall"]),
                    fmt_acc(row["m0_mean_f1"]),
                    fmt_acc(row["m2_mean_f1"]),
                ]
                for row in rare["buckets"]
            ],
        ),
        m0_ll=m0["log_loss"],
        m2_ll=m2["log_loss"],
        m0_t1=m0["mean_top1"],
        m2_t1=m2["mean_top1"],
        m0_mg=m0["mean_margin"],
        m2_mg=m2["mean_margin"],
        m0_ent=m0["mean_entropy"],
        m2_ent=m2["mean_entropy"],
        m0_ece=cal["M0"]["ece"]["ece"],
        m2_ece=cal["M2"]["ece"]["ece"],
        m0_te_t1=shift["M0"]["test_mean_top1"],
        m2_te_t1=shift["M2"]["test_mean_top1"],
        m0_te_ent=shift["M0"]["test_mean_entropy"],
        m2_te_ent=shift["M2"]["test_mean_entropy"],
        m0_js=shift["M0"]["val_test_js_divergence"],
        m2_js=shift["M2"]["val_test_js_divergence"],
        test_agree=shift["m0_m2_test_agreement_rate"],
        test_agree_n=shift["m0_m2_test_agree_n"],
        n_test=N_TEST,
        test_changed=shift["m0_m2_test_changed"],
        lm0_both=lzh_m0["overall"]["both_correct"],
        lm0_lzh=lzh_m0["overall"]["b_only_correct"],
        lm0_m0=lzh_m0["overall"]["a_only_correct"],
        lm0_wrong=lzh_m0["overall"]["both_wrong"],
        lm0_net=lzh_m0["overall"]["net_b_minus_a"],
        lm0_oracle=lzh_m0["overall"]["pair_oracle_correct"],
        n=N_TRAIN,
        lm0_oracle_acc=lzh_m0["overall"]["pair_oracle_accuracy"],
        lm0_p=fmt_p(lzh_m0["mcnemar"]["pvalue"]),
        lm0_02=lzh_m0["folds_0_2"]["net_b_minus_a"],
        lm0_34=lzh_m0["folds_3_4"]["net_b_minus_a"],
        lm2_both=lzh_m2["overall"]["both_correct"],
        lm2_lzh=lzh_m2["overall"]["b_only_correct"],
        lm2_m2=lzh_m2["overall"]["a_only_correct"],
        lm2_wrong=lzh_m2["overall"]["both_wrong"],
        lm2_net=lzh_m2["overall"]["net_b_minus_a"],
        lm2_oracle=lzh_m2["overall"]["pair_oracle_correct"],
        lm2_oracle_acc=lzh_m2["overall"]["pair_oracle_accuracy"],
        lm2_p=fmt_p(lzh_m2["mcnemar"]["pvalue"]),
        lm2_02=lzh_m2["folds_0_2"]["net_b_minus_a"],
        lm2_34=lzh_m2["folds_3_4"]["net_b_minus_a"],
        team_decision=m["decisions"]["team_standalone"],
        personal_decision=m["decisions"]["personal_version"],
        lzh_correct=lzh["correct"],
        lzh_acc=lzh["accuracy"],
        m2_correct=m2["correct"],
        m0_correct=m0["correct"],
        four=integ["four_expert_oracle"],
        four_acc=integ["four_expert_oracle"] / N_TRAIN,
        five=integ["five_expert_oracle"],
        five_acc=integ["five_expert_oracle"] / N_TRAIN,
        new_unique=integ["m2_new_unique"],
        next_action=m["decisions"]["next_action"],
    )


def main() -> None:
    if current_branch() != EXPECTED_BRANCH:
        raise SystemExit(
            "STOP: branch is {}, expected {}. Do not switch branches.".format(
                current_branch(), EXPECTED_BRANCH
            )
        )
    if MODEL_V3_DOC.is_file():
        integrity_failure("docs/versions/model_v3.md already exists; E07D must not freeze MODEL V3")

    pred_before = sha256_file(PRED_PATH) if PRED_PATH.is_file() else None
    frozen_before = {str(path): sha256_file(path) for path in FROZEN_PATHS if path.is_file()}

    data = load_dataset(ROOT)
    class_names = allowed_labels()
    if len(class_names) != N_CLASSES:
        integrity_failure("expected 60 official classes")
    train_ids = [str(v) for v in data.meta_train.index.astype(str)]
    test_ids = [str(v) for v in data.meta_test.index.astype(str)]
    if len(train_ids) != N_TRAIN or len(test_ids) != N_TEST:
        integrity_failure("official train/test sizes are not 5000")
    y_true = data.meta_train.loc[train_ids, TARGET_COL].astype(str).to_numpy()
    folds_frame, _ = load_and_validate_team_folds(
        train_ids, test_ids, y_true, ROOT / "experiments" / "team_folds_5_seed42.csv"
    )
    canonical_fold = folds_frame.set_index("Cell_ID").reindex(train_ids)["fold"].astype(int).to_numpy()

    e05a = pd.read_parquet(E05A_REGISTRY)
    e05a["Cell_ID"] = e05a["Cell_ID"].astype(str)
    e05a = align_frame(e05a, train_ids, E05A_REGISTRY).reset_index()
    if list(e05a["true_label"].astype(str)) != list(y_true):
        integrity_failure("E05A true_label does not match official meta_train")
    if list(e05a["canonical_fold"].astype(int)) != list(canonical_fold):
        integrity_failure("E05A canonical_fold does not match team_folds_5_seed42")

    m0_oof = align_frame(read_cell_id_csv(M0_OOF), train_ids, M0_OOF)
    m2_val = align_frame(read_cell_id_csv(M2_VAL), train_ids, M2_VAL)
    m0_pred = m0_oof["predicted_label"].astype(str).to_numpy()
    m2_pred = m2_val["predicted_label"].astype(str).to_numpy()
    lzh_pred = e05a["lzh_pred"].astype(str).to_numpy()
    if list(m0_pred) != list(e05a["wyh_pred"].astype(str)):
        integrity_failure("M0 predictions do not match frozen four-expert WYH predictions")

    m0_ok = m0_pred == y_true
    m2_ok = m2_pred == y_true
    lzh_ok = lzh_pred == y_true
    s0_ok = e05a["s0_correct"].to_numpy(dtype=bool)
    sni_ok = e05a["sni_correct"].to_numpy(dtype=bool)
    four_ok = lzh_ok | m0_ok | s0_ok | sni_ok
    m2_unique = m2_ok & ~lzh_ok & ~m0_ok & ~s0_ok & ~sni_ok
    five_ok = four_ok | m2_ok

    delta_overall = cell_delta(m0_pred, m2_pred, y_true)
    delta_02 = cell_delta(m0_pred, m2_pred, y_true, canonical_fold <= 2)
    delta_34 = cell_delta(m0_pred, m2_pred, y_true, canonical_fold >= 3)

    m0_metrics_json = json.loads(M0_METRICS.read_text())
    e06m_comp = json.loads(E06M_COMP.read_text())
    e06m_cmp = pd.read_csv(E06M_CMP)
    e06m_m2_row = e06m_cmp.set_index("model").loc["M2"]
    new_unique_frame = read_cell_id_csv(E06M_NEW)
    m1_correct = int(e06m_cmp.set_index("model").loc["M1"]["correct"])
    sni_capture = int(e06m_comp["sni_capture"]["m2"]["captured"])

    checks = [
        ("m0_correct", int(m0_ok.sum()), M0_CORRECT),
        ("m0_accuracy", float(np.round(m0_ok.mean(), 4)), M0_ACCURACY),
        ("m2_correct", int(m2_ok.sum()), M2_CORRECT),
        ("m2_accuracy", float(np.round(m2_ok.mean(), 4)), M2_ACCURACY),
        ("lzh_correct", int(lzh_ok.sum()), LZH_CORRECT),
        ("m2_wrong_to_correct", delta_overall["wrong_to_correct"], M2_VS_M0_WRONG_TO_CORRECT),
        ("m2_correct_to_wrong", delta_overall["correct_to_wrong"], M2_VS_M0_CORRECT_TO_WRONG),
        ("m2_net", delta_overall["net_correction"], M2_VS_M0_NET),
        ("m2_changed", delta_overall["changed_predictions"], M2_VS_M0_CHANGED),
        ("m2_net_02", delta_02["net_correction"], M2_VS_M0_NET_02),
        ("m2_net_34", delta_34["net_correction"], M2_VS_M0_NET_34),
        ("four_expert_oracle", int(four_ok.sum()), FOUR_EXPERT_ORACLE),
        ("five_expert_oracle", int(five_ok.sum()), FIVE_EXPERT_ORACLE),
        ("m2_new_unique", int(m2_unique.sum()), M2_NEW_UNIQUE),
        ("frozen_m0_metrics_correct", int(m0_metrics_json["correct"]), M0_CORRECT),
        ("e06m_m2_correct_csv", int(e06m_m2_row["correct"]), M2_CORRECT),
        ("e06m_new_unique_file", int(len(new_unique_frame)), M2_NEW_UNIQUE),
        ("e06m_five_expert_json", int(e06m_comp["m2_five_expert_oracle_correct"]), FIVE_EXPERT_ORACLE),
        ("s0_correct", int(s0_ok.sum()), S0_CORRECT),
        ("sni_correct", int(sni_ok.sum()), SNI_CORRECT),
    ]
    for name, got, expected in checks:
        if got != expected:
            integrity_failure("{} expected {} got {}".format(name, expected, got))
    if int(five_ok.sum()) != FOUR_EXPERT_ORACLE + int(m2_unique.sum()):
        integrity_failure("five-expert oracle identity 4364 + new unique failed")
    if STRONG_NET_VS_M0 != E06M_STRONG_NET_VS_M0 or STRONG_ACCURACY_MIN != E06M_STRONG_ACCURACY_MIN:
        integrity_failure("E07D must not lower frozen E06M STRONG thresholds")

    m0_proba = align_proba(M0_VAL_PROBA, train_ids, class_names)
    m2_proba = align_proba(M2_VAL_PROBA, train_ids, class_names)
    lzh_proba = load_lzh_probabilities(train_ids, class_names)
    if list(argmax_labels(m0_proba, class_names)) != list(m0_pred):
        integrity_failure("M0 probability argmax != M0 hard labels")
    if list(argmax_labels(m2_proba, class_names)) != list(m2_pred):
        integrity_failure("M2 probability argmax != M2 hard labels")
    if list(argmax_labels(lzh_proba, class_names)) != list(lzh_pred):
        integrity_failure("LZH probability argmax != frozen LZH hard labels")

    m0_conf = confidence_from_proba(m0_proba)
    m2_conf = confidence_from_proba(m2_proba)
    lzh_conf = confidence_from_proba(lzh_proba)

    section = e05a["Section_ID"].astype(str).to_numpy()
    if np.any(pd.isna(e05a["Section_ID"])) or np.any(section == "") or np.any(section == "nan"):
        integrity_failure("missing Section_ID where expected")

    biological_family = np.array([family_of(str(v)) for v in y_true], dtype=object)
    registry = pd.DataFrame(
        {
            "Cell_ID": train_ids,
            "true_label": y_true,
            "canonical_fold": canonical_fold,
            "Section_ID": section,
            "Region": e05a["Region"].astype(str).to_numpy(),
            "E/I": e05a["E/I"].astype(str).to_numpy(),
            "Segment": e05a["Segment"].astype(str).to_numpy(),
            "hard_bucket": e05a["hard_bucket"].to_numpy(dtype=bool),
            "n_detected": e05a["n_detected"].to_numpy(),
            "library_size": e05a["library_size"].to_numpy(dtype=np.float64),
            "neuron_or_glial": e05a["neuron_or_glial"].astype(str).to_numpy(),
            "biological_family": biological_family,
            "m0_pred": m0_pred,
            "m0_correct": m0_ok,
            "m0_top1": m0_conf["top1"],
            "m0_margin": m0_conf["margin"],
            "m0_entropy": m0_conf["entropy"],
            "m2_pred": m2_pred,
            "m2_correct": m2_ok,
            "m2_top1": m2_conf["top1"],
            "m2_margin": m2_conf["margin"],
            "m2_entropy": m2_conf["entropy"],
            "lzh_pred": lzh_pred,
            "lzh_correct": lzh_ok,
            "lzh_top1": lzh_conf["top1"],
            "lzh_margin": lzh_conf["margin"],
            "lzh_entropy": lzh_conf["entropy"],
            "m2_beats_m0": m2_ok & ~m0_ok,
            "m0_beats_m2": m0_ok & ~m2_ok,
            "lzh_beats_m0": lzh_ok & ~m0_ok,
            "m0_beats_lzh": m0_ok & ~lzh_ok,
            "lzh_beats_m2": lzh_ok & ~m2_ok,
            "m2_beats_lzh": m2_ok & ~lzh_ok,
            "diagnostic_only_note": (
                "true_label/correctness/beats_* fields are diagnostic-only and are not "
                "deployment features"
            ),
        }
    )
    if len(registry) != N_TRAIN or registry["Cell_ID"].nunique() != N_TRAIN:
        integrity_failure("deployable registry is not 5000 unique Cell_ID rows")

    m0_block = candidate_block("M0", m0_pred, m0_ok, canonical_fold, m0_proba, class_names, y_true, m0_conf)
    m2_block = candidate_block("M2", m2_pred, m2_ok, canonical_fold, m2_proba, class_names, y_true, m2_conf)
    lzh_block = candidate_block("LZH", lzh_pred, lzh_ok, canonical_fold, lzh_proba, class_names, y_true, lzh_conf)
    if abs(m0_block["macro_f1"] - M0_MACRO_F1) > 1e-12:
        integrity_failure("M0 macro-F1 mismatch")
    if abs(m0_block["log_loss"] - M0_LOG_LOSS) > 1e-12:
        integrity_failure("M0 log loss mismatch")
    if abs(m2_block["macro_f1"] - M2_MACRO_F1) > 1e-12:
        integrity_failure("M2 macro-F1 mismatch")
    if abs(m2_block["log_loss"] - M2_LOG_LOSS) > 1e-12:
        integrity_failure("M2 log loss mismatch")

    masks = slice_masks(registry)
    slice_rows = []
    slice_payload = OrderedDict()
    for key, display in SLICE_ORDER:
        mask = masks[key]
        n = int(mask.sum())
        m0_acc = float(m0_ok[mask].mean()) if n else None
        m2_acc = float(m2_ok[mask].mean()) if n else None
        lzh_acc = float(lzh_ok[mask].mean()) if n else None
        row = {
            "slice": key,
            "slice_display": display,
            "n": n,
            "M0": m0_acc,
            "M2": m2_acc,
            "LZH": lzh_acc,
            "M2_minus_M0": float(m2_acc - m0_acc) if n else None,
            "LZH_minus_M0": float(lzh_acc - m0_acc) if n else None,
            "LZH_minus_M2": float(lzh_acc - m2_acc) if n else None,
        }
        slice_payload[key] = row
        slice_rows.append(row)

    mcnemar_m2m0 = mcnemar_exact(delta_overall["wrong_to_correct"], delta_overall["correct_to_wrong"])
    cell_deltas = paired_accuracy_deltas(m0_ok, m2_ok, BOOTSTRAP_SEED, N_BOOTSTRAP)
    cell_boot = summarize_deltas(cell_deltas)
    sec_deltas = section_cluster_deltas(section, m0_ok, m2_ok, BOOTSTRAP_SEED, N_BOOTSTRAP)
    sec_boot = summarize_deltas(sec_deltas)

    section_rows = []
    acc_deltas = []
    win = tie = lose = 0
    for sec_id, idx in registry.groupby("Section_ID", sort=True).groups.items():
        pos = np.asarray(idx)
        n = int(len(pos))
        m0_c = int(m0_ok[pos].sum())
        m2_c = int(m2_ok[pos].sum())
        lzh_c = int(lzh_ok[pos].sum())
        acc_delta = (m2_c - m0_c) / n
        acc_deltas.append(acc_delta)
        if m2_c > m0_c:
            win += 1
        elif m2_c == m0_c:
            tie += 1
        else:
            lose += 1
        section_rows.append(
            {
                "Section_ID": str(sec_id),
                "n": n,
                "m0_correct": m0_c,
                "m2_correct": m2_c,
                "lzh_correct": lzh_c,
                "m2_m0_correct_delta": m2_c - m0_c,
                "lzh_m0_correct_delta": lzh_c - m0_c,
                "lzh_m2_correct_delta": lzh_c - m2_c,
                "m2_m0_accuracy_delta": acc_delta,
                "lzh_m0_accuracy_delta": (lzh_c - m0_c) / n,
                "lzh_m2_accuracy_delta": (lzh_c - m2_c) / n,
            }
        )
    acc_deltas_arr = np.asarray(acc_deltas, dtype=np.float64)
    section_stability = {
        "n_sections": int(len(section_rows)),
        "m2_gt_m0": win,
        "m2_eq_m0": tie,
        "m2_lt_m0": lose,
        "median_accuracy_delta": float(np.median(acc_deltas_arr)),
        "iqr_accuracy_delta": float(np.percentile(acc_deltas_arr, 75) - np.percentile(acc_deltas_arr, 25)),
        "min_accuracy_delta": float(np.min(acc_deltas_arr)),
        "max_accuracy_delta": float(np.max(acc_deltas_arr)),
    }

    names = list(class_names)
    m0_rec = recall_score(y_true, m0_pred, labels=names, average=None, zero_division=0)
    m2_rec = recall_score(y_true, m2_pred, labels=names, average=None, zero_division=0)
    lzh_rec = recall_score(y_true, lzh_pred, labels=names, average=None, zero_division=0)
    m0_f1 = f1_score(y_true, m0_pred, labels=names, average=None, zero_division=0)
    m2_f1 = f1_score(y_true, m2_pred, labels=names, average=None, zero_division=0)
    lzh_f1 = f1_score(y_true, lzh_pred, labels=names, average=None, zero_division=0)
    support = np.array([(y_true == name).sum() for name in names], dtype=np.int64)
    class_rows = []
    n_gain = n_tie = n_loss = 0
    for i, name in enumerate(names):
        delta = float(m2_rec[i] - m0_rec[i])
        if delta > 0:
            n_gain += 1
        elif delta < 0:
            n_loss += 1
        else:
            n_tie += 1
        class_rows.append(
            {
                "class": name,
                "support": int(support[i]),
                "m0_recall": float(m0_rec[i]),
                "m2_recall": float(m2_rec[i]),
                "lzh_recall": float(lzh_rec[i]),
                "m2_minus_m0": delta,
                "lzh_minus_m0": float(lzh_rec[i] - m0_rec[i]),
                "lzh_minus_m2": float(lzh_rec[i] - m2_rec[i]),
                "m0_f1": float(m0_f1[i]),
                "m2_f1": float(m2_f1[i]),
                "lzh_f1": float(lzh_f1[i]),
            }
        )
    class_sorted_gain = sorted(class_rows, key=lambda r: (-r["m2_minus_m0"], -r["support"], r["class"]))
    class_sorted_loss = sorted(class_rows, key=lambda r: (r["m2_minus_m0"], -r["support"], r["class"]))

    def bucket_stats(lo, hi, label):
        if hi is None:
            sel = support >= lo
        else:
            sel = (support >= lo) & (support < hi)
        idx = np.where(sel)[0]
        if idx.size == 0:
            return {
                "bucket": label,
                "n_classes": 0,
                "n_cells": 0,
                "m0_mean_recall": None,
                "m2_mean_recall": None,
                "m0_mean_f1": None,
                "m2_mean_f1": None,
            }
        return {
            "bucket": label,
            "n_classes": int(idx.size),
            "n_cells": int(support[idx].sum()),
            "m0_mean_recall": float(m0_rec[idx].mean()),
            "m2_mean_recall": float(m2_rec[idx].mean()),
            "m0_mean_f1": float(m0_f1[idx].mean()),
            "m2_mean_f1": float(m2_f1[idx].mean()),
        }

    rare = {
        "macro_f1_m0": m0_block["macro_f1"],
        "macro_f1_m2": m2_block["macro_f1"],
        "macro_f1_delta": m2_block["macro_f1"] - m0_block["macro_f1"],
        "buckets": [
            bucket_stats(0, 25, "<25"),
            bucket_stats(25, 50, "25-49"),
            bucket_stats(50, 100, "50-99"),
            bucket_stats(100, None, ">=100"),
        ],
    }

    cal = {
        "M0": {
            "ece": equal_width_ece(m0_ok, m0_conf["top1"]),
            "deciles": confidence_deciles(m0_ok, m0_conf["top1"]),
        },
        "M2": {
            "ece": equal_width_ece(m2_ok, m2_conf["top1"]),
            "deciles": confidence_deciles(m2_ok, m2_conf["top1"]),
        },
    }

    m0_test_proba = align_proba(M0_TEST_PROBA, test_ids, class_names)
    m2_test_proba = align_proba(M2_TEST_PROBA, test_ids, class_names)
    m0_test_pred = argmax_labels(m0_test_proba, class_names)
    m2_test_pred = argmax_labels(m2_test_proba, class_names)
    m0_test_conf = confidence_from_proba(m0_test_proba)
    m2_test_conf = confidence_from_proba(m2_test_proba)
    m0_val_freq = predicted_class_frequency(m0_pred, class_names)
    m2_val_freq = predicted_class_frequency(m2_pred, class_names)
    m0_test_freq = predicted_class_frequency(m0_test_pred, class_names)
    m2_test_freq = predicted_class_frequency(m2_test_pred, class_names)
    test_agree_n = int((m0_test_pred == m2_test_pred).sum())
    test_shift = {
        "test_labels_used": False,
        "note": "NO TEST LABELS WERE USED. Descriptive predicted-class and confidence shift only.",
        "M0": {
            "val_mean_top1": float(m0_conf["top1"].mean()),
            "test_mean_top1": float(m0_test_conf["top1"].mean()),
            "val_mean_entropy": float(m0_conf["entropy"].mean()),
            "test_mean_entropy": float(m0_test_conf["entropy"].mean()),
            "val_test_js_divergence": js_divergence_vectors(m0_val_freq, m0_test_freq),
        },
        "M2": {
            "val_mean_top1": float(m2_conf["top1"].mean()),
            "test_mean_top1": float(m2_test_conf["top1"].mean()),
            "val_mean_entropy": float(m2_conf["entropy"].mean()),
            "test_mean_entropy": float(m2_test_conf["entropy"].mean()),
            "val_test_js_divergence": js_divergence_vectors(m2_val_freq, m2_test_freq),
        },
        "m0_m2_test_agreement_rate": float(test_agree_n / N_TEST),
        "m0_m2_test_agree_n": test_agree_n,
        "m0_m2_test_changed": int(N_TEST - test_agree_n),
        "js_implementation": "scipy.spatial.distance.jensenshannon(base=2) squared to divergence",
    }

    def pair_block(ok_a, ok_b, pred_a, pred_b, b_wrong_to_correct, a_only_as_c2w):
        overall = pair_accounting(ok_a, ok_b, pred_a, pred_b)
        p02 = pair_accounting(ok_a[canonical_fold <= 2], ok_b[canonical_fold <= 2])
        p34 = pair_accounting(ok_a[canonical_fold >= 3], ok_b[canonical_fold >= 3])
        mc = mcnemar_exact(b_wrong_to_correct, a_only_as_c2w)
        slice_pairs = OrderedDict()
        for key, display in SLICE_ORDER:
            mask = masks[key]
            slice_pairs[key] = {"slice_display": display, **pair_accounting(ok_a[mask], ok_b[mask])}
        return {"overall": overall, "folds_0_2": p02, "folds_3_4": p34, "mcnemar": mc, "slices": slice_pairs}

    lzh_vs_m0 = pair_block(
        m0_ok,
        lzh_ok,
        m0_pred,
        lzh_pred,
        int((lzh_ok & ~m0_ok).sum()),
        int((m0_ok & ~lzh_ok).sum()),
    )
    lzh_vs_m2 = pair_block(
        m2_ok,
        lzh_ok,
        m2_pred,
        lzh_pred,
        int((lzh_ok & ~m2_ok).sum()),
        int((m2_ok & ~lzh_ok).sum()),
    )

    strong_ok, failed_strong = strong_criteria_met(
        net=delta_overall["net_correction"],
        acc=float(m2_ok.mean()),
        net_02=delta_02["net_correction"],
        net_34=delta_34["net_correction"],
        macro_f1=m2_block["macro_f1"],
        m0_macro_f1=m0_block["macro_f1"],
        sni_capture=sni_capture,
        new_unique=int(m2_unique.sum()),
        m2_correct=int(m2_ok.sum()),
        m1_correct=m1_correct,
    )
    reproduced_label, reproduced_reason = classify_m2(
        net_vs_m0=delta_overall["net_correction"],
        acc=float(m2_ok.mean()),
        net_02=delta_02["net_correction"],
        net_34=delta_34["net_correction"],
        macro_f1=m2_block["macro_f1"],
        m0_macro_f1=m0_block["macro_f1"],
        m2_correct=int(m2_ok.sum()),
        m1_correct=m1_correct,
        sni_capture=sni_capture,
        new_unique=int(m2_unique.sum()),
        integrity_ok=True,
    )
    if reproduced_label != E06M_FROZEN_LABEL:
        integrity_failure("reproduced E06M label is {}, expected {}".format(reproduced_label, E06M_FROZEN_LABEL))
    if strong_ok:
        integrity_failure("E06M STRONG criteria unexpectedly met; do not continue without review")

    team_decision = team_standalone_decision(
        lzh_correct=int(lzh_ok.sum()),
        m2_correct=int(m2_ok.sum()),
        m0_correct=int(m0_ok.sum()),
        lzh_net_vs_m0=int(lzh_ok.sum() - m0_ok.sum()),
        lzh_net_vs_m2=int(lzh_ok.sum() - m2_ok.sum()),
        m2_net_vs_m0=delta_overall["net_correction"],
        lzh_folds_34=lzh_block["folds_3_4"],
        m2_folds_34=m2_block["folds_3_4"],
        m0_folds_34=m0_block["folds_3_4"],
    )
    personal_decision = personal_promotion_decision(
        e06m_label=reproduced_label,
        strong_ok=strong_ok,
        m2_correct=int(m2_ok.sum()),
        m0_correct=int(m0_ok.sum()),
        net=delta_overall["net_correction"],
        net_34=delta_34["net_correction"],
    )
    if team_decision not in ALLOWED_TEAM_DECISIONS:
        integrity_failure("illegal team decision label")
    if personal_decision not in ALLOWED_PERSONAL_DECISIONS:
        integrity_failure("illegal personal decision label")
    if personal_decision != "MODEL V3 PROMOTION NOT JUSTIFIED":
        integrity_failure("E07D must not promote MODEL V3 when E06M is PROMISING rather than STRONG")

    next_action = next_action_for(personal_decision, team_decision)

    integrity = {name: got for name, got, _expected in checks}
    integrity.update(
        {
            "m0_macro_f1": m0_block["macro_f1"],
            "m2_macro_f1": m2_block["macro_f1"],
            "m0_log_loss": m0_block["log_loss"],
            "m2_log_loss": m2_block["log_loss"],
            "four_expert_oracle_accuracy": FOUR_EXPERT_ORACLE / N_TRAIN,
            "five_expert_oracle_accuracy": FIVE_EXPERT_ORACLE / N_TRAIN,
            "oracle_is_not_deployable_accuracy": True,
        }
    )

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "research_codename": RESEARCH_CODENAME,
        "created_at_utc": utc_now(),
        "analysis_only": True,
        "trained_model": False,
        "blended_experts": False,
        "optimized_weights": False,
        "created_model_v3": False,
        "oracle_is_not_deployable_accuracy": True,
        "eligibility": {
            "standalone_candidates": ["M0", "M2", "LZH"],
            "excluded": {
                "S0": {"accuracy": S0_ACCURACY, "reason": "diversity / oracle-analysis expert only"},
                "SNI": {"accuracy": SNI_ACCURACY, "reason": "diversity / oracle-analysis expert only"},
                "team_main_v9": team_main_v9_status(),
            },
            "yhh": yhh_status(),
        },
        "integrity": integrity,
        "e06m_strong_audit": {
            "frozen_label": E06M_FROZEN_LABEL,
            "reproduced_label": reproduced_label,
            "reproduced_reason": reproduced_reason,
            "strong_criteria_met": strong_ok,
            "failed_strong_checks": failed_strong,
            "thresholds_not_lowered": True,
            "strong_net_vs_m0": E06M_STRONG_NET_VS_M0,
            "strong_accuracy_min": E06M_STRONG_ACCURACY_MIN,
            "e06m_module_strong_net": STRONG_NET_VS_M0,
            "e06m_module_strong_acc": STRONG_ACCURACY_MIN,
        },
        "candidates": {"M0": m0_block, "M2": m2_block, "LZH": lzh_block},
        "m2_vs_m0": {
            "overall": {
                "changed_predictions": delta_overall["changed_predictions"],
                "wrong_to_correct": delta_overall["wrong_to_correct"],
                "correct_to_wrong": delta_overall["correct_to_wrong"],
                "net": delta_overall["net_correction"],
            },
            "folds_0_2": {"net": delta_02["net_correction"], **delta_02},
            "folds_3_4": {"net": delta_34["net_correction"], **delta_34},
            "mcnemar": mcnemar_m2m0,
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "n_resamples": N_BOOTSTRAP,
            "cell_level": cell_boot,
            "section_cluster": sec_boot,
            "not_used_to_invent_promotion_threshold": True,
        },
        "section_stability": section_stability,
        "slices": slice_payload,
        "classwise": {
            "n_m2_improves": n_gain,
            "n_tied": n_tie,
            "n_m2_worsens": n_loss,
            "top10_m2_gains": class_sorted_gain[:10],
            "top10_m2_losses": class_sorted_loss[:10],
        },
        "rare_class": rare,
        "calibration": cal,
        "test_shift": test_shift,
        "lzh_vs_m0": lzh_vs_m0,
        "lzh_vs_m2": lzh_vs_m2,
        "oracles": {
            "four_expert": FOUR_EXPERT_ORACLE,
            "five_expert_including_m2": FIVE_EXPERT_ORACLE,
            "m2_new_unique_recoveries": int(m2_unique.sum()),
            "not_deployable_accuracy": True,
        },
        "decisions": {
            "team_standalone": team_decision,
            "personal_version": personal_decision,
            "replace_model_v2_with_m2": False,
            "next_action": next_action,
        },
        "leakage_audit": {
            "competition_test_labels_used": False,
            "trained_model": False,
            "trained_router": False,
            "trained_stacker": False,
            "blended_experts": False,
            "optimized_ensemble_weights": False,
            "searched_thresholds": False,
            "tuned_source_weights": False,
            "added_dataset": False,
            "started_spatial_id": False,
            "created_m3": False,
            "modified_test_predictions": False,
            "created_model_v3": False,
            "created_model_v3_doc": False,
            "created_model_v3_tag": False,
            "prediction_csv_modified": False,
            "diagnostic_correctness_fields_used_as_deployment_features": False,
        },
        "git": {
            "branch": current_branch(),
            "head": run_git(["rev-parse", "HEAD"]).strip(),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(OUT_DIR / "v3_e07d_deployable_registry.parquet", index=False)

    summary_rows = []
    for block in (m0_block, m2_block, lzh_block):
        fa = block["fold_accuracy"]
        summary_rows.append(
            {
                "model": block["name"],
                "correct": block["correct"],
                "accuracy": block["accuracy"],
                "macro_f1": block["macro_f1"],
                "log_loss": block["log_loss"],
                "fold_0": fa["0"],
                "fold_1": fa["1"],
                "fold_2": fa["2"],
                "fold_3": fa["3"],
                "fold_4": fa["4"],
                "folds_0_2": block["folds_0_2"],
                "folds_3_4": block["folds_3_4"],
                "mean_fold_accuracy": block["mean_fold_accuracy"],
                "std_fold_accuracy": block["std_fold_accuracy"],
                "min_fold_accuracy": block["min_fold_accuracy"],
                "max_fold_accuracy": block["max_fold_accuracy"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "v3_e07d_candidate_summary.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(OUT_DIR / "v3_e07d_slice_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "comparison": "M2_vs_M0",
                "wrong_to_correct": mcnemar_m2m0["wrong_to_correct"],
                "correct_to_wrong": mcnemar_m2m0["correct_to_wrong"],
                "net": mcnemar_m2m0["net"],
                "n_discordant": mcnemar_m2m0["n_discordant"],
                "mcnemar_exact_pvalue": mcnemar_m2m0["pvalue"],
                "both_correct": int((m0_ok & m2_ok).sum()),
                "both_wrong": int((~m0_ok & ~m2_ok).sum()),
                "pair_oracle_correct": int((m0_ok | m2_ok).sum()),
                "folds_0_2_net": delta_02["net_correction"],
                "folds_3_4_net": delta_34["net_correction"],
            },
            {
                "comparison": "LZH_vs_M0",
                "wrong_to_correct": lzh_vs_m0["mcnemar"]["wrong_to_correct"],
                "correct_to_wrong": lzh_vs_m0["mcnemar"]["correct_to_wrong"],
                "net": lzh_vs_m0["overall"]["net_b_minus_a"],
                "n_discordant": lzh_vs_m0["mcnemar"]["n_discordant"],
                "mcnemar_exact_pvalue": lzh_vs_m0["mcnemar"]["pvalue"],
                "both_correct": lzh_vs_m0["overall"]["both_correct"],
                "both_wrong": lzh_vs_m0["overall"]["both_wrong"],
                "pair_oracle_correct": lzh_vs_m0["overall"]["pair_oracle_correct"],
                "folds_0_2_net": lzh_vs_m0["folds_0_2"]["net_b_minus_a"],
                "folds_3_4_net": lzh_vs_m0["folds_3_4"]["net_b_minus_a"],
            },
            {
                "comparison": "LZH_vs_M2",
                "wrong_to_correct": lzh_vs_m2["mcnemar"]["wrong_to_correct"],
                "correct_to_wrong": lzh_vs_m2["mcnemar"]["correct_to_wrong"],
                "net": lzh_vs_m2["overall"]["net_b_minus_a"],
                "n_discordant": lzh_vs_m2["mcnemar"]["n_discordant"],
                "mcnemar_exact_pvalue": lzh_vs_m2["mcnemar"]["pvalue"],
                "both_correct": lzh_vs_m2["overall"]["both_correct"],
                "both_wrong": lzh_vs_m2["overall"]["both_wrong"],
                "pair_oracle_correct": lzh_vs_m2["overall"]["pair_oracle_correct"],
                "folds_0_2_net": lzh_vs_m2["folds_0_2"]["net_b_minus_a"],
                "folds_3_4_net": lzh_vs_m2["folds_3_4"]["net_b_minus_a"],
            },
        ]
    ).to_csv(OUT_DIR / "v3_e07d_pairwise_tests.csv", index=False)
    write_json(
        OUT_DIR / "v3_e07d_bootstrap.json",
        {
            "seed": BOOTSTRAP_SEED,
            "n_resamples": N_BOOTSTRAP,
            "cell_level": cell_boot,
            "section_cluster": sec_boot,
            "not_used_to_invent_promotion_threshold": True,
        },
    )
    pd.DataFrame(section_rows).to_csv(OUT_DIR / "v3_e07d_section_stability.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "v3_e07d_classwise_stability.csv", index=False)

    cal_rows = []
    for model_name, block in cal.items():
        for item in block["ece"]["bins"]:
            cal_rows.append({"model": model_name, "scheme": "equal_width_ece", **item, "ece": block["ece"]["ece"]})
        for item in block["deciles"]:
            cal_rows.append({"model": model_name, "scheme": "confidence_decile", **item, "ece": None})
    pd.DataFrame(cal_rows).to_csv(OUT_DIR / "v3_e07d_calibration_audit.csv", index=False)
    write_json(OUT_DIR / "v3_e07d_test_shift_audit.json", test_shift)
    write_json(OUT_DIR / "v3_e07d_decision.json", metrics)

    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "candidate_summary.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(TABLE_DIR / "slice_comparison.csv", index=False)
    pd.DataFrame(class_sorted_gain[:10]).to_csv(TABLE_DIR / "m2_top_gains.csv", index=False)
    pd.DataFrame(class_sorted_loss[:10]).to_csv(TABLE_DIR / "m2_top_losses.csv", index=False)
    pd.DataFrame(rare["buckets"]).to_csv(TABLE_DIR / "rare_class_buckets.csv", index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(metrics))

    if MODEL_V3_DOC.is_file():
        integrity_failure("experiment created docs/versions/model_v3.md")
    tags = run_git(["tag", "--list", "model-v3"]).strip()
    if tags:
        integrity_failure("model-v3 tag exists; E07D must not create it")
    if PRED_PATH.is_file() and pred_before is not None and sha256_file(PRED_PATH) != pred_before:
        integrity_failure("prediction/prediction.csv changed")
    for path, digest in frozen_before.items():
        if sha256_file(Path(path)) != digest:
            integrity_failure("frozen artifact changed: {}".format(path))

    print(
        json.dumps(
            {
                "status": "PASS",
                "experiment_id": EXPERIMENT_ID,
                "m0_correct": int(m0_ok.sum()),
                "m2_correct": int(m2_ok.sum()),
                "lzh_correct": int(lzh_ok.sum()),
                "m2_vs_m0_net": delta_overall["net_correction"],
                "mcnemar_p": mcnemar_m2m0["pvalue"],
                "team_standalone": team_decision,
                "personal_version": personal_decision,
                "oracle_is_not_deployable_accuracy": True,
                "created_model_v3": False,
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
