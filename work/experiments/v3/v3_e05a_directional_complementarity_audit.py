#!/usr/bin/env python3
"""V3-E05A: Asymmetric Directional Expert Complementarity Audit (ADE-AUDIT).

Analysis only. Does not train a model, router, or ensemble. Does not search
thresholds or weights. Does not add data. Does not write prediction/prediction.csv.
Does not freeze MODEL V3.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

_WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WORK / "src"))
sys.path.insert(0, str(_WORK / "experiments" / "v3"))
from merfish60.io import repo_root  # noqa: E402
ROOT = repo_root()

from merfish60.io import TARGET_COL, load_dataset  # noqa: E402
from merfish60.metrics import overall_accuracy  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.official_contract import allowed_labels, sha256_file  # noqa: E402
from merfish60.v2_metrics import json_default, write_json  # noqa: E402

from v3_e00t_team_expert_audit import confidence_from_proba  # noqa: E402
from v3_e03a_rescue_audit import CLASS_FAMILIES_E03A, family_of  # noqa: E402

EXPERIMENT_ID = "V3-E05A"
RESEARCH_CODENAME = "ADE-AUDIT"
N_TRAIN = 5000
N_CLASSES = 60
EXPECTED_BRANCH = "ywan/ml-pipeline"
EXPECTED_HEAD_PREFIX = "61f6194"

LZH_CORRECT = 4133
WYH_CORRECT = 4106
S0_CORRECT = 3151
SNI_CORRECT = 2841
TWO_EXPERT_ORACLE = 4215
THREE_EXPERT_ORACLE = 4311
FOUR_EXPERT_ORACLE = 4364
ALL_FOUR_WRONG = 636
S0_UNIQUE_RECOVERIES = 96
SNI_UNIQUE_RECOVERIES = 53
FOUR_EXPERT_FOLDS_02_CORRECT = 2627
FOUR_EXPERT_FOLDS_34_CORRECT = 1737

OLIGO_1 = "oligodendrocyte_1"
OLIGO_2 = "oligodendrocyte_2"
OPC_1 = "oligodendrocyte_progenitor_1"
OPC_2 = "oligodendrocyte_progenitor_2"
PRECURSOR = "oligodendrocyte_precursor_cell"

OLIGO_OPC_CLASSES = [
    OLIGO_1,
    OLIGO_2,
    PRECURSOR,
    OPC_1,
    OPC_2,
]

CONTROL_FAMILIES = [
    "astrocyte",
    "vascular",
    "meningeal",
    "microglia",
    "remaining_glial_non_neuronal",
    "neuronal_or_other",
]

MIN_SUPPORT = 15
MIN_WRONG_TO_CORRECT = 10
MIN_PRECISION_OVERALL = 0.65
MIN_PRECISION_FOLDS_02 = 0.60
MIN_PRECISION_FOLDS_34 = 0.55
SPARSE_DOMINANCE = 0.80

E00T_REGISTRY = ROOT / "work" / "outputs" / "v3" / "v3_e00t_team_oof_registry.parquet"
E02D_S0 = ROOT / "work" / "outputs" / "v3" / "v3_e02d_s0_validation.csv"
E03A_REGISTRY = ROOT / "work" / "outputs" / "v3" / "v3_e03a_three_expert_registry.parquet"
E04S_VAL = ROOT / "work" / "outputs" / "v3" / "v3_e04s_sni_validation.csv"
E04S_PROBA = ROOT / "work" / "outputs" / "v3" / "v3_e04s_sni_validation_probabilities.csv.gz"
PRED_PATH = ROOT / "prediction" / "prediction.csv"
OUT_DIR = ROOT / "work" / "outputs" / "v3"
TABLE_DIR = OUT_DIR / "v3_e05a_tables"
REPORT_PATH = ROOT / "work" / "reports" / "v3" / "v3_e05a_directional_complementarity_audit.md"

FROZEN_PATHS = [
    E00T_REGISTRY,
    E02D_S0,
    E03A_REGISTRY,
    E04S_VAL,
    E04S_PROBA,
    ROOT / "work" / "outputs" / "v3" / "v3_e04s_complementarity.json",
    ROOT / "work" / "outputs" / "v3" / "v3_e04s_shared_failure_registry.csv",
    ROOT / "work" / "outputs" / "v3" / "v3_e04s_new_unique_recoveries.csv",
    ROOT / "work" / "docs" / "versions" / "model_v1.md",
    ROOT / "work" / "docs" / "versions" / "model_v2.md",
    ROOT / "work" / "outputs" / "submissions" / "model_v1.csv",
    ROOT / "work" / "outputs" / "submissions" / "model_v2_candidate.csv",
    PRED_PATH,
]

HYPOTHESES = OrderedDict(
    [
        (
            "H1",
            {
                "id": "H1",
                "name": "SNI progenitor-2 rescue from oligo-1 consensus",
                "strong_class": OLIGO_1,
                "candidate_expert": "sni",
                "candidate_class": OPC_2,
                "trigger_uses_true_label": False,
                "trigger_uses_correctness_flags": False,
                "motivation": "Frozen E04S: SNI rescued 20/36 shared oligodendrocyte_progenitor_2 cells that LZH called oligodendrocyte_1.",
            },
        ),
        (
            "H2",
            {
                "id": "H2",
                "name": "S0 oligo-1 rescue from progenitor-2 consensus",
                "strong_class": OPC_2,
                "candidate_expert": "s0",
                "candidate_class": OLIGO_1,
                "trigger_uses_true_label": False,
                "trigger_uses_correctness_flags": False,
                "motivation": "Frozen E03A: S0 uniquely recovered 26–28 oligodendrocyte_1 cells that strong experts called oligodendrocyte_progenitor_2.",
            },
        ),
        (
            "H3",
            {
                "id": "H3",
                "name": "SNI progenitor-2 rescue from oligo-2 consensus",
                "strong_class": OLIGO_2,
                "candidate_expert": "sni",
                "candidate_class": OPC_2,
                "trigger_uses_true_label": False,
                "trigger_uses_correctness_flags": False,
                "motivation": "Frozen E04S: SNI rescued 8/35 shared oligodendrocyte_progenitor_2 cells that LZH called oligodendrocyte_2.",
            },
        ),
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(ROOT), stderr=subprocess.STDOUT).decode(
        "utf-8"
    )


def current_branch() -> str:
    return run_git(["branch", "--show-current"]).strip()


def integrity_failure(message: str) -> None:
    raise SystemExit("E05A REGISTRY INTEGRITY FAILURE: {}".format(message))


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


def trigger_h1(lzh_pred: np.ndarray, wyh_pred: np.ndarray, sni_pred: np.ndarray) -> np.ndarray:
    """Observable H1 trigger. Predictions only. Never uses true labels."""
    lzh_pred = np.asarray(lzh_pred, dtype=object)
    wyh_pred = np.asarray(wyh_pred, dtype=object)
    sni_pred = np.asarray(sni_pred, dtype=object)
    return (lzh_pred == OLIGO_1) & (wyh_pred == OLIGO_1) & (sni_pred == OPC_2)


def trigger_h2(lzh_pred: np.ndarray, wyh_pred: np.ndarray, s0_pred: np.ndarray) -> np.ndarray:
    """Observable H2 trigger. Predictions only. Never uses true labels."""
    lzh_pred = np.asarray(lzh_pred, dtype=object)
    wyh_pred = np.asarray(wyh_pred, dtype=object)
    s0_pred = np.asarray(s0_pred, dtype=object)
    return (lzh_pred == OPC_2) & (wyh_pred == OPC_2) & (s0_pred == OLIGO_1)


def trigger_h3(lzh_pred: np.ndarray, wyh_pred: np.ndarray, sni_pred: np.ndarray) -> np.ndarray:
    """Observable H3 trigger. Predictions only. Never uses true labels."""
    lzh_pred = np.asarray(lzh_pred, dtype=object)
    wyh_pred = np.asarray(wyh_pred, dtype=object)
    sni_pred = np.asarray(sni_pred, dtype=object)
    return (lzh_pred == OLIGO_2) & (wyh_pred == OLIGO_2) & (sni_pred == OPC_2)


TRIGGER_FUNCTIONS = {"H1": trigger_h1, "H2": trigger_h2, "H3": trigger_h3}


def trigger_functions_use_predictions_only() -> bool:
    for fn in (trigger_h1, trigger_h2, trigger_h3):
        names = set(inspect.signature(fn).parameters)
        banned = {"true_label", "y_true", "correct", "oracle", "rescue"}
        if names & banned:
            return False
        if "pred" not in " ".join(names):
            return False
    return True


def empty_accounting() -> dict:
    return {
        "support": 0,
        "strong_correct": 0,
        "candidate_correct": 0,
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "net": 0,
        "patch_precision": None,
        "candidate_conditional_accuracy": None,
        "strong_conditional_accuracy": None,
        "oracle_conditional_accuracy": None,
    }


def directional_accounting(
    trigger: np.ndarray,
    strong_pred: np.ndarray,
    candidate_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Retrospective accounting of a predefined observable trigger.

    True labels are used only to score a trigger that was defined from predictions.
    """
    t = np.asarray(trigger, dtype=bool)
    n = int(t.sum())
    if n == 0:
        return empty_accounting()
    y = np.asarray(y_true, dtype=object)[t]
    s = np.asarray(strong_pred, dtype=object)[t]
    c = np.asarray(candidate_pred, dtype=object)[t]
    s_ok = s == y
    c_ok = c == y
    w2c = (~s_ok) & c_ok
    c2w = s_ok & (~c_ok)
    both_ok = s_ok & c_ok
    both_bad = (~s_ok) & (~c_ok)
    denom = int(w2c.sum()) + int(c2w.sum())
    return {
        "support": n,
        "strong_correct": int(s_ok.sum()),
        "candidate_correct": int(c_ok.sum()),
        "wrong_to_correct": int(w2c.sum()),
        "correct_to_wrong": int(c2w.sum()),
        "both_correct": int(both_ok.sum()),
        "both_wrong": int(both_bad.sum()),
        "net": int(w2c.sum()) - int(c2w.sum()),
        "patch_precision": (float(w2c.sum() / denom) if denom else None),
        "candidate_conditional_accuracy": float(c_ok.mean()),
        "strong_conditional_accuracy": float(s_ok.mean()),
        "oracle_conditional_accuracy": float((s_ok | c_ok).mean()),
    }


def partition_accounting(
    trigger: np.ndarray,
    strong_pred: np.ndarray,
    candidate_pred: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
) -> dict:
    folds = np.asarray(folds, dtype=int)
    overall = directional_accounting(trigger, strong_pred, candidate_pred, y_true)
    f02 = directional_accounting(trigger & (folds <= 2), strong_pred, candidate_pred, y_true)
    f34 = directional_accounting(trigger & (folds >= 3), strong_pred, candidate_pred, y_true)
    return {"overall": overall, "folds_0_2": f02, "folds_3_4": f34}


def actionable_checks(overall: dict, f02: dict, f34: dict) -> dict:
    prec = overall.get("patch_precision")
    p02 = f02.get("patch_precision")
    p34 = f34.get("patch_precision")
    checks = OrderedDict(
        [
            ("support_ge_15", overall["support"] >= MIN_SUPPORT),
            ("wrong_to_correct_ge_10", overall["wrong_to_correct"] >= MIN_WRONG_TO_CORRECT),
            ("precision_ge_065", prec is not None and float(prec) >= MIN_PRECISION_OVERALL),
            ("net_gt_0", overall["net"] > 0),
            ("folds_0_2_net_gt_0", f02["net"] > 0),
            ("folds_3_4_net_gt_0", f34["net"] > 0),
            ("folds_0_2_precision_ge_060", p02 is not None and float(p02) >= MIN_PRECISION_FOLDS_02),
            ("folds_3_4_precision_ge_055", p34 is not None and float(p34) >= MIN_PRECISION_FOLDS_34),
        ]
    )
    return {"actionable": bool(all(checks.values())), "checks": dict(checks)}


def sparsity_audit(
    trigger: np.ndarray,
    folds: np.ndarray,
    sections: np.ndarray,
    true_labels: np.ndarray,
    hard_bucket: np.ndarray,
) -> dict:
    t = np.asarray(trigger, dtype=bool)
    n = int(t.sum())
    fold_counts = {str(f): int((np.asarray(folds)[t] == f).sum()) for f in range(5)}
    if n == 0:
        return {
            "support": 0,
            "support_by_fold": fold_counts,
            "n_sections": 0,
            "n_true_classes_retrospective": 0,
            "hard_bucket_fraction": None,
            "max_section_id": None,
            "max_section_n": 0,
            "max_section_fraction": None,
            "max_fold": None,
            "max_fold_n": 0,
            "max_fold_fraction": None,
            "n_folds_with_support": 0,
            "single_section_artifact": False,
            "sparse_unstable": True,
            "sparse_reasons": ["zero_support"],
        }
    sec = pd.Series(np.asarray(sections, dtype=object)[t]).astype(str)
    sec_counts = sec.value_counts()
    fold_pos = {k: v for k, v in fold_counts.items() if v > 0}
    max_fold_key = max(fold_counts, key=lambda k: fold_counts[k])
    max_fold_n = fold_counts[max_fold_key]
    max_sec_id = str(sec_counts.index[0])
    max_sec_n = int(sec_counts.iloc[0])
    max_sec_frac = float(max_sec_n / n)
    max_fold_frac = float(max_fold_n / n)
    reasons = []
    if n < MIN_SUPPORT:
        reasons.append("support_below_15")
    if int(sec.nunique()) <= 1:
        reasons.append("single_section")
    if max_sec_frac >= SPARSE_DOMINANCE:
        reasons.append("one_section_ge_80pct")
    if len(fold_pos) <= 1:
        reasons.append("single_fold")
    if max_fold_frac >= SPARSE_DOMINANCE:
        reasons.append("one_fold_ge_80pct")
    single_section_artifact = ("single_section" in reasons) or ("one_section_ge_80pct" in reasons)
    return {
        "support": n,
        "support_by_fold": fold_counts,
        "n_sections": int(sec.nunique()),
        "n_true_classes_retrospective": int(pd.Series(np.asarray(true_labels)[t]).nunique()),
        "hard_bucket_fraction": float(np.asarray(hard_bucket, dtype=bool)[t].mean()),
        "max_section_id": max_sec_id,
        "max_section_n": max_sec_n,
        "max_section_fraction": max_sec_frac,
        "max_fold": int(max_fold_key),
        "max_fold_n": max_fold_n,
        "max_fold_fraction": max_fold_frac,
        "n_folds_with_support": int(len(fold_pos)),
        "single_section_artifact": bool(single_section_artifact),
        "sparse_unstable": bool(len(reasons) > 0),
        "sparse_reasons": reasons,
        "true_class_counts_retrospective": dict(
            Counter(np.asarray(true_labels, dtype=object)[t].tolist())
        ),
        "section_counts": {str(k): int(v) for k, v in sec_counts.to_dict().items()},
    }


def apply_fixed_patches(
    lzh_pred: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    h3: np.ndarray,
    s0_pred: np.ndarray,
    sni_pred: np.ndarray,
    system: str,
) -> np.ndarray:
    """Deterministic D0-D3 patches. H1/H2/H3 are mutually exclusive by construction."""
    if system not in {"D0", "D1", "D2", "D3"}:
        raise ValueError("unknown diagnostic system {}".format(system))
    out = np.array(lzh_pred, dtype=object, copy=True)
    if system in {"D1", "D2", "D3"}:
        out[np.asarray(h1, dtype=bool)] = np.asarray(sni_pred, dtype=object)[np.asarray(h1, dtype=bool)]
    if system in {"D2", "D3"}:
        out[np.asarray(h2, dtype=bool)] = np.asarray(s0_pred, dtype=object)[np.asarray(h2, dtype=bool)]
    if system == "D3":
        out[np.asarray(h3, dtype=bool)] = np.asarray(sni_pred, dtype=object)[np.asarray(h3, dtype=bool)]
    return out


def system_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    folds: np.ndarray,
    class_names: Sequence[str],
    system: str,
) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    pred = np.asarray(pred, dtype=object)
    baseline = np.asarray(baseline, dtype=object)
    folds = np.asarray(folds, dtype=int)
    changed = pred != baseline
    base_ok = baseline == y_true
    new_ok = pred == y_true
    w2c = int((changed & (~base_ok) & new_ok).sum())
    c2w = int((changed & base_ok & (~new_ok)).sum())
    f02 = folds <= 2
    f34 = folds >= 3
    return {
        "system": system,
        "label": "RETROSPECTIVE FIXED-PATCH DIAGNOSTIC",
        "not_unbiased_final_oof": True,
        "not_model_v3": True,
        "changed": int(changed.sum()),
        "wrong_to_correct": w2c,
        "correct_to_wrong": c2w,
        "net": w2c - c2w,
        "correct": int(new_ok.sum()),
        "accuracy": float(overall_accuracy(y_true, pred)),
        "macro_f1": float(
            f1_score(y_true, pred, labels=list(class_names), average="macro", zero_division=0)
        ),
        "folds_0_2_n": int(f02.sum()),
        "folds_0_2_correct": int(new_ok[f02].sum()),
        "folds_0_2_accuracy": float(new_ok[f02].mean()) if f02.any() else None,
        "folds_3_4_n": int(f34.sum()),
        "folds_3_4_correct": int(new_ok[f34].sum()),
        "folds_3_4_accuracy": float(new_ok[f34].mean()) if f34.any() else None,
    }


def classify_experiment(hypotheses: Dict[str, dict]) -> dict:
    strong = []
    limited = []
    for hid, row in hypotheses.items():
        sparse = bool(row["sparsity"]["sparse_unstable"])
        artifact = bool(row["sparsity"]["single_section_artifact"])
        if row["actionable"] and (not sparse) and (not artifact):
            strong.append(hid)
        elif row["overall"]["net"] > 0 and row["overall"]["wrong_to_correct"] > 0:
            limited.append(hid)
    if strong:
        return {
            "label": "DIRECTIONAL SIGNAL STRONG",
            "qualifying_hypotheses": strong,
            "reason": (
                "{} meet all predeclared stable/actionable criteria with positive net "
                "correction on both canonical folds 0-2 and the retrospective stability "
                "partition (folds 3-4), and are not single-section artifacts.".format(
                    ", ".join(strong)
                )
            ),
            "next_action": "V3-E05B — Cross-Fitted Directional Abstaining Patch",
        }
    if limited:
        best = max(
            limited,
            key=lambda k: (
                hypotheses[k]["overall"]["net"],
                hypotheses[k]["overall"]["patch_precision"] or 0.0,
                hypotheses[k]["overall"]["wrong_to_correct"],
            ),
        )
        h = hypotheses[best]
        both_pos = h["folds_0_2"]["net"] > 0 and h["folds_3_4"]["net"] > 0
        prec_ok = (h["overall"]["patch_precision"] or 0.0) >= 0.55
        support_ok = h["overall"]["support"] >= 10
        if both_pos and prec_ok and support_ok and not h["sparsity"]["single_section_artifact"]:
            next_action = (
                "V3-E05B-NARROW — predeclared {}-only cross-fitted directional patch "
                "with explicit small-N and section-stability gates. Do not search new "
                "patterns and do not train a router.".format(best)
            )
            reason = (
                "{} show positive net correction, but support, precision, or fold "
                "stability is insufficient for a global directional patch. {} is the "
                "strongest limited direction.".format(", ".join(limited), best)
            )
        else:
            next_action = (
                "Do not implement a global directional patch. Move to a conservative "
                "four-expert integration / model-selection strategy. Do not train a "
                "learned router."
            )
            reason = (
                "{} show some positive correction value, but precision, fold "
                "stability, or sparsity is insufficient even for a narrow global patch.".format(
                    ", ".join(limited)
                )
            )
        return {
            "label": "DIRECTIONAL SIGNAL LIMITED",
            "qualifying_hypotheses": limited,
            "strongest_limited_hypothesis": best,
            "reason": reason,
            "next_action": next_action,
        }
    return {
        "label": "DIRECTIONAL SIGNAL WEAK",
        "qualifying_hypotheses": [],
        "reason": (
            "None of H1/H2/H3 provides stable positive net correction under the "
            "predeclared actionable criteria."
        ),
        "next_action": (
            "Move to a final conservative model-selection / integration strategy. "
            "Do not train a learned router."
        ),
    }


def numeric_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return "{:.4f}".format(v)
        if isinstance(v, (np.floating,)):
            return "{:.4f}".format(float(v))
        if isinstance(v, (bool, np.bool_)):
            return "yes" if bool(v) else "no"
        return str(v)

    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def acc_text(correct: int, n: int = N_TRAIN) -> str:
    return "{} / {} = {:.4f}".format(correct, n, correct / n)


def fmt_prec(v: Optional[float]) -> str:
    return "n/a" if v is None else "{:.4f}".format(v)


def hypothesis_block(hid: str, row: dict) -> str:
    o = row["overall"]
    a = row["folds_0_2"]
    b = row["folds_3_4"]
    s = row["sparsity"]
    lines = [
        "### {}".format(hid),
        "",
        "Observable trigger: `{}`.".format(row["trigger_text"]),
        "Hypothetical action: replace strong consensus with `{}` prediction `{}`.".format(
            row["candidate_expert"].upper(), row["candidate_class"]
        ),
        "The trigger uses **predictions only**. It does not condition on true class.",
        "",
        _md_table(
            [
                "split",
                "support",
                "wrong→correct",
                "correct→wrong",
                "precision",
                "net",
                "strong acc",
                "candidate acc",
                "oracle acc",
            ],
            [
                [
                    "overall",
                    o["support"],
                    o["wrong_to_correct"],
                    o["correct_to_wrong"],
                    o["patch_precision"],
                    o["net"],
                    o["strong_conditional_accuracy"],
                    o["candidate_conditional_accuracy"],
                    o["oracle_conditional_accuracy"],
                ],
                [
                    "folds 0-2 (development)",
                    a["support"],
                    a["wrong_to_correct"],
                    a["correct_to_wrong"],
                    a["patch_precision"],
                    a["net"],
                    a["strong_conditional_accuracy"],
                    a["candidate_conditional_accuracy"],
                    a["oracle_conditional_accuracy"],
                ],
                [
                    "folds 3-4 (retrospective stability)",
                    b["support"],
                    b["wrong_to_correct"],
                    b["correct_to_wrong"],
                    b["patch_precision"],
                    b["net"],
                    b["strong_conditional_accuracy"],
                    b["candidate_conditional_accuracy"],
                    b["oracle_conditional_accuracy"],
                ],
            ],
        ),
        "",
        "Actionable under predeclared criteria: **{}**.".format(
            "yes" if row["actionable"] else "no"
        ),
        "Failed checks: {}.".format(
            ", ".join(k for k, v in row["checks"].items() if not v) or "none"
        ),
        "Sparsity flag: **{}** ({}).".format(
            "SPARSE / UNSTABLE" if s["sparse_unstable"] else "not sparse",
            ", ".join(s["sparse_reasons"]) or "none",
        ),
        "Sections represented: {}. Max section fraction: {}.".format(
            s["n_sections"], fmt_prec(s.get("max_section_fraction"))
        ),
        "Support by fold: {}.".format(
            "/".join(str(s["support_by_fold"][str(i)]) for i in range(5))
        ),
        "Retrospective true-class mix (diagnostic only; not a trigger input): {}.".format(
            ", ".join(
                "{}={}".format(k, v)
                for k, v in sorted(
                    s.get("true_class_counts_retrospective", {}).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            )
            or "none"
        ),
    ]
    return "\n".join(lines)


def render_report(metrics: dict) -> str:
    integ = metrics["integrity"]
    hyps = metrics["hypotheses"]
    decision = metrics["final_classification"]
    dsys = metrics["fixed_patch_diagnostics"]
    sc = metrics["strong_consensus"]
    ctrl = metrics["non_oligo_controls"]
    hard = metrics["hard_bucket"]
    asym = metrics["s0_vs_sni"]
    conf = metrics["confidence_secondary"]
    leak = metrics["leakage_audit"]
    sparse_bits = []
    for hid in ("H1", "H2", "H3"):
        s = hyps[hid]["sparsity"]
        sparse_bits.append(
            [
                hid,
                s["support"],
                "/".join(str(s["support_by_fold"][str(i)]) for i in range(5)),
                s["n_sections"],
                s["n_true_classes_retrospective"],
                s["hard_bucket_fraction"],
                s["max_section_fraction"],
                "SPARSE / UNSTABLE" if s["sparse_unstable"] else "ok",
                ", ".join(s["sparse_reasons"]) or "none",
            ]
        )
    oligo_nonzero = [
        r
        for r in metrics["oligo_direction_matrix"]
        if r["support"] > 0
    ]
    oligo_rows = [
        [
            r["expert"],
            r["strong_consensus_class"],
            r["proposed_class"],
            r["support"],
            r["wrong_to_correct"],
            r["correct_to_wrong"],
            r["net"],
            r["patch_precision"],
            r["strong_conditional_accuracy"],
            r["candidate_conditional_accuracy"],
            "H1" if r["is_h1"] else "H2" if r["is_h2"] else "H3" if r["is_h3"] else "",
        ]
        for r in oligo_nonzero
    ]
    d_rows = []
    for key in ("D0", "D1", "D2", "D3"):
        r = dsys[key]
        d_rows.append(
            [
                key,
                r["changed"],
                r["wrong_to_correct"],
                r["correct_to_wrong"],
                r["net"],
                r["correct"],
                r["accuracy"],
                r["macro_f1"],
                r["folds_0_2_accuracy"],
                r["folds_3_4_accuracy"],
            ]
        )
    sc_class_rows = [
        [
            r["strong_consensus_class"],
            r["n"],
            r["errors"],
            r["accuracy"],
            r["s0_rescues"],
            r["sni_rescues"],
            r["either_rescues"],
            r["both_rescues"],
        ]
        for r in sc["by_consensus_class"]
        if r["strong_consensus_class"] in OLIGO_OPC_CLASSES or r["errors"] > 0
    ]
    ctrl_rows = [
        [
            r["family"],
            r["n_true"],
            r["strong_consensus_n"],
            r["strong_consensus_errors"],
            r["s0_rescues"],
            r["sni_rescues"],
            r["either_rescues"],
        ]
        for r in ctrl["by_true_family"]
    ]
    hard_rows = []
    for hid in ("H1", "H2", "H3"):
        for bucket, lab in (("True", "hard"), ("False", "not-hard")):
            r = hard[hid][bucket]
            hard_rows.append(
                [
                    hid,
                    lab,
                    r["support"],
                    r["wrong_to_correct"],
                    r["correct_to_wrong"],
                    r["net"],
                    r["patch_precision"],
                ]
            )
    conf_rows = []
    for hid in ("H1", "H2", "H3"):
        block = conf[hid]
        for grp in ("wrong_to_correct", "correct_to_wrong"):
            g = block[grp]
            conf_rows.append(
                [
                    hid,
                    grp,
                    g["n"],
                    g["candidate_top1"]["mean"],
                    g["candidate_margin"]["mean"],
                    g["candidate_entropy"]["mean"],
                    g["lzh_margin"]["mean"],
                    g["wyh_margin"]["mean"],
                    g["lzh_entropy"]["mean"],
                    g["wyh_entropy"]["mean"],
                ]
            )
    asym_rows = []
    for hid in ("H1", "H2", "H3"):
        r = asym[hid]
        asym_rows.append(
            [
                hid,
                r["support"],
                r["s0_correct"],
                r["sni_correct"],
                r["both_correct"],
                r["both_wrong"],
                r["s0_only_correct"],
                r["sni_only_correct"],
                r["s0_sni_agree"],
            ]
        )
    return """# V3-E05A — Asymmetric Directional Expert Complementarity Audit

Research codename: **{codename}**. This experiment is ANALYSIS ONLY. It does not train a model, router, or ensemble, does not search thresholds or weights, does not add a dataset, and does not freeze MODEL V3.

## 1. Research Question

Can observable predicted-class patterns identify asymmetric confusion directions where S0 or SNI provides stable, low-risk correction value?

A future deployable concept, if supported, would **not** be "choose the most confident expert" or "average all experts". It would be:

strong anchor prediction pattern → specific predefined confusion direction? → abstain and keep the strong prediction, or consult a designated directional expert with a fixed patch.

V3-E05A only evaluates whether such directional structure exists. It does **not** implement a final patch.

Oracle values below are diagnostic coverage ceilings. **ORACLE != DEPLOYABLE ACCURACY.**

## 2. Frozen Four-Expert Evidence

Reproduced from the four-expert canonical registry before any directional analysis:

| Quantity | Frozen | Reproduced |
|---|---:|---:|
| LZH Prior-H | 4133 / 5000 = 0.8266 | {lzh} |
| WYH MODEL V2 | 4106 / 5000 = 0.8212 | {wyh} |
| S0 hard-label 200-gene MLP | 3151 / 5000 = 0.6302 | {s0} |
| SNI-only source-diverse expert | 2841 / 5000 = 0.5682 | {sni} |
| LZH + WYH oracle | 4215 / 5000 = 0.8430 | {two} |
| LZH + WYH + S0 oracle | 4311 / 5000 = 0.8622 | {three} |
| LZH + WYH + S0 + SNI oracle | 4364 / 5000 = 0.8728 | {four} |
| S0 unique recoveries beyond LZH+WYH | 96 | {s0u} |
| SNI unique recoveries beyond LZH+WYH+S0 | 53 | {sniu} |
| Remaining all-four-wrong | 636 | {afw} |
| Four-expert oracle folds 0-2 | 2627 / 3000 = 0.8757 | {four02} |
| Four-expert oracle folds 3-4 | 1737 / 2000 = 0.8685 | {four34} |

These oracles are **diagnostic coverage ceilings**, not deployable accuracy.

The question is no longer whether another expert can raise the oracle. The question is whether a simple, interpretable, predefined directional mechanism can convert some of this coverage into real net corrections.

## 3. Why Confidence Routing Was Rejected

V3-E03A found that S0 confidence does not identify the 96 unique S0 rescues: high S0 top1/margin deciles contained almost none of them, and P vs N1 AUROC for S0 top1 was 0.4142.

V3-E04S found that inside the 689 all-three-wrong cells, SNI-correct cases had **lower** mean top1/margin than SNI-wrong cases (top1 AUROC 0.4195).

Neither weak expert has globally useful confidence-based rescue behavior. Confidence is therefore **not** the primary E05A hypothesis. It appears only as a secondary descriptive diagnostic inside the predeclared triggers.

## 4. Directional-Expertise Hypothesis

S0 and SNI are weak global classifiers (0.6302 and 0.5682) but contribute 96 and 53 unique recoveries. Frozen confusion-pair evidence was asymmetric:

- SNI rescued true `oligodendrocyte_progenitor_2` when strong experts predicted `oligodendrocyte_1` (20 / 36 shared errors).
- SNI rescued true `oligodendrocyte_progenitor_2` when LZH predicted `oligodendrocyte_2` (8 / 35).
- SNI rescued **0** of the reverse failure true `oligodendrocyte_1` / strong `oligodendrocyte_progenitor_2`.
- S0 previously contributed in that reverse oligo-1 direction.

Hypothesis: weak/source-diverse experts may possess **direction-specific expertise** rather than globally useful reliability.

A future rule may use only inference-time observables (predicted classes, probabilities, agreement, test-safe metadata). It must never condition on true class, correctness flags, oracle membership, or rescue flags.

## 5. Integrity Checks

Four-expert registry rows: **5000**. Unique 19-digit `Cell_ID` strings: **5000**. Labels aligned to official `meta_train`. Joins used `Cell_ID`, never row position.

All predeclared counts reproduced exactly:

{integ_table}

If any of these had failed, the experiment would have stopped with `E05A REGISTRY INTEGRITY FAILURE`.

H1/H2/H3 trigger functions use predictions only: **{pred_only}**.

## 6. Predeclared H1 / H2 / H3

Trigger definitions were frozen from prior E03A/E04S observations **before** E05A accounting. They were not modified after seeing E05A results. No confidence thresholds were added. Newly discovered full-data patterns are **not** promoted to candidate rules here.

Canonical folds 3-4 have been viewed in prior research stages. They are a **RETROSPECTIVE STABILITY PARTITION**, not an untouched holdout. E05A does not claim an unbiased final MODEL V3 result.

{h1_block}

{h2_block}

{h3_block}

## 7. Oligo / OPC Direction Matrix

Matrix uses only cells with observable strong consensus (`LZH_pred == WYH_pred`) and ordered pairs among the frozen oligo/OPC family:

`oligodendrocyte_1`, `oligodendrocyte_2`, `oligodendrocyte_precursor_cell`, `oligodendrocyte_progenitor_1`, `oligodendrocyte_progenitor_2`.

This table is **descriptive**. Non-predeclared rows are **not** candidate rules. Highlighted H1/H2/H3 rows are the only directions eligible for E05A actionability.

Rows with support 0 are omitted below; the full 5×5 × two-expert matrix is in `work/outputs/v3/v3_e05a_oligo_direction_matrix.csv`.

{oligo_table}

Any non-H1/H2/H3 pair that looks numerically attractive is recorded only as a **HYPOTHESIS FOR FUTURE WORK**, not as validated E05A evidence.

{future_hyp}

## 8. S0 vs SNI Asymmetric Expertise

On the same observable H1/H2/H3 cells, are S0 and SNI complementary in opposite directions? Standalone accuracy is not the question.

{asym_table}

{asym_text}

## 9. Strong-Consensus Failure Analysis

E03A established that weak-expert rescues often occur even when `LZH == WYH`. Strong agreement does not imply correctness on specific biological boundaries.

| Quantity | Value |
|---|---:|
| LZH == WYH cells | {sc_n} |
| Strong-consensus accuracy | {sc_acc} |
| Strong-consensus errors | {sc_err} |
| Rescued by S0 | {sc_s0} |
| Rescued by SNI | {sc_sni} |
| Rescued by either | {sc_either} |
| Rescued by both | {sc_both} |

Breakdown by strong consensus class (oligo/OPC classes plus any class with at least one consensus error):

{sc_table}

## 10. Non-Oligo Controls

These summaries test whether directional rescues of strong-consensus failures are concentrated in oligo/OPC rather than generic weak-expert complementarity. **No candidate patch rules are invented for these families.**

{ctrl_table}

Oligo/OPC share of S0 strong-consensus rescues: {ctrl_s0_oligo} / {sc_s0}.
Oligo/OPC share of SNI strong-consensus rescues: {ctrl_sni_oligo} / {sc_sni}.

{ctrl_text}

## 11. Confidence Secondary Diagnostics

Confidence is **not** used to modify H1/H2/H3. No threshold is chosen. Means are descriptive only, comparing wrong→correct versus correct→wrong cells inside each predeclared trigger.

{conf_table}

{conf_text}

## 12. Hard-Bucket Analysis

Diagnostic only. No hard-bucket threshold or rule is created.

{hard_table}

## 13. Retrospective Fixed-Patch Diagnostics D0-D3

**D0-D3 ARE NOT UNBIASED FINAL OOF RESULTS.**

They are **RETROSPECTIVE FIXED-PATCH DIAGNOSTICS**. H1/H2/H3 were motivated by prior full-data error analysis, so these numbers estimate whether the directional mechanism is promising enough to justify a separately implemented cross-fitted/predeclared E05B evaluation. They are **not** MODEL V3 and must not be frozen as a version score.

| System | Rule |
|---|---|
| D0 | LZH prediction only |
| D1 | LZH baseline + apply H1 only |
| D2 | LZH baseline + apply H1 and H2 |
| D3 | LZH baseline + apply H1, H2, and H3 |

No combination was optimized. No additional pattern was added to improve D3.

{d_table}

## 14. Sparsity / Stability Audit

Directional rules can overfit if trigger support is tiny or confined to one fold or one Section.

{sparse_table}

{sparse_text}

## 15. Leakage / Selection-Bias Audit

H1/H2/H3 were motivated by **prior frozen full-data analyses** (V3-E03A confusion-pair rescues and V3-E04S shared-failure recoveries). Those prior looks used the same 5000 training cells, including canonical folds 3-4. Therefore:

- E05A cannot claim an unbiased final OOF estimate for a directional patch.
- Canonical folds 3-4 are a retrospective stability partition, not a pristine holdout.
- D0-D3 must not be presented as MODEL V3 performance.

True labels were used only retrospectively to score predefined observable prediction patterns.

- competition test labels used: {leak_test}
- true labels used to define H1/H2/H3 triggers: {leak_true_trigger}
- true labels used only for retrospective scoring: {leak_diag}
- learned router/classifier trained: {leak_router}
- threshold optimized: {leak_thr}
- ensemble weights optimized: {leak_w}
- post-hoc pattern mining promoted to a candidate rule: {leak_posthoc}
- new external dataset added: {leak_data}
- expert probabilities altered: {leak_proba}
- S0 / SNI / LZH / WYH retrained: {leak_retrain}
- Spatial-ID started: {leak_sid}
- MODEL V3 created: {leak_v3}
- submission generated: {leak_sub}
- prediction/prediction.csv modified: {leak_pred}
- V3-E00T / E02D / E03A / E04S artifacts modified: {leak_frozen}

## 16. What a Future E05B May Use

Only observable, predeclared prediction-direction patterns, for example:

- LZH predicted class
- WYH predicted class
- S0 predicted class
- SNI predicted class
- whether LZH and WYH agree
- the H1/H2/H3 triggers exactly as frozen here
- test-safe metadata as covariates or stratum descriptors, not as newly searched rules

E05A does **not** by itself justify starting E05B. If a later experiment still implements a directional patch, it should be cross-fitted or otherwise predeclared so that trigger evaluation is not scored on the same full-data look that motivated H1/H2/H3.

## 17. What a Future E05B Must NOT Use

- true labels at inference
- expert correctness flags
- oracle membership
- rescue flags
- a confusion direction defined using the true class ("if the true cell is progenitor_2")
- post-hoc mining of prediction tuples on full OOF
- confidence thresholds selected on full data
- arbitrary ensemble-weight search
- a learned router fit on the same cells used to report the score

## 18. Decision

**{label}**

{reason}

Recommended next action: **{next_action}**

Do not start that experiment in this task. MODEL V3 is not frozen. No `docs/versions/model_v3.md` and no submission candidate were created.
""".format(
        codename=RESEARCH_CODENAME,
        lzh=acc_text(integ["lzh_correct"]),
        wyh=acc_text(integ["wyh_correct"]),
        s0=acc_text(integ["s0_correct"]),
        sni=acc_text(integ["sni_correct"]),
        two=acc_text(integ["two_expert_oracle"]),
        three=acc_text(integ["three_expert_oracle"]),
        four=acc_text(integ["four_expert_oracle"]),
        s0u=integ["s0_unique_recoveries"],
        sniu=integ["sni_unique_recoveries"],
        afw=integ["all_four_wrong"],
        four02=acc_text(integ["four_expert_folds_0_2_correct"], 3000),
        four34=acc_text(integ["four_expert_folds_3_4_correct"], 2000),
        integ_table=_md_table(
            ["Check", "Expected", "Reproduced"],
            [
                ["LZH correct", LZH_CORRECT, integ["lzh_correct"]],
                ["WYH correct", WYH_CORRECT, integ["wyh_correct"]],
                ["S0 correct", S0_CORRECT, integ["s0_correct"]],
                ["SNI correct", SNI_CORRECT, integ["sni_correct"]],
                ["two-expert oracle", TWO_EXPERT_ORACLE, integ["two_expert_oracle"]],
                ["three-expert oracle", THREE_EXPERT_ORACLE, integ["three_expert_oracle"]],
                ["four-expert oracle", FOUR_EXPERT_ORACLE, integ["four_expert_oracle"]],
                ["all-four-wrong", ALL_FOUR_WRONG, integ["all_four_wrong"]],
                ["S0 incremental unique recoveries", S0_UNIQUE_RECOVERIES, integ["s0_unique_recoveries"]],
                ["SNI incremental unique recoveries", SNI_UNIQUE_RECOVERIES, integ["sni_unique_recoveries"]],
            ],
        ),
        pred_only="yes" if metrics["triggers_use_predictions_only"] else "NO",
        h1_block=hypothesis_block("H1", hyps["H1"]),
        h2_block=hypothesis_block("H2", hyps["H2"]),
        h3_block=hypothesis_block("H3", hyps["H3"]),
        oligo_table=_md_table(
            [
                "expert",
                "strong consensus",
                "proposed",
                "support",
                "w→c",
                "c→w",
                "net",
                "precision",
                "strong acc",
                "cand acc",
                "predeclared",
            ],
            oligo_rows,
        )
        if oligo_rows
        else "_No positive-support oligo/OPC consensus→candidate pairs._",
        future_hyp=metrics["future_work_note"],
        asym_table=_md_table(
            [
                "H",
                "support",
                "S0 correct",
                "SNI correct",
                "both correct",
                "both wrong",
                "S0-only",
                "SNI-only",
                "S0==SNI",
            ],
            asym_rows,
        ),
        asym_text=metrics["asymmetry_narrative"],
        sc_n=sc["n_agree"],
        sc_acc="{:.4f}".format(sc["accuracy"]) if sc["accuracy"] is not None else "n/a",
        sc_err=sc["n_errors"],
        sc_s0=sc["s0_rescues"],
        sc_sni=sc["sni_rescues"],
        sc_either=sc["either_rescues"],
        sc_both=sc["both_rescues"],
        sc_table=_md_table(
            ["consensus class", "n", "errors", "acc", "S0 rescues", "SNI rescues", "either", "both"],
            sc_class_rows[:25],
        ),
        ctrl_table=_md_table(
            [
                "family",
                "n true",
                "strong-consensus n",
                "consensus errors",
                "S0 rescues",
                "SNI rescues",
                "either",
            ],
            ctrl_rows,
        ),
        ctrl_s0_oligo=ctrl["oligo_s0_rescues"],
        ctrl_sni_oligo=ctrl["oligo_sni_rescues"],
        ctrl_text=metrics["control_narrative"],
        conf_table=_md_table(
            [
                "H",
                "group",
                "n",
                "cand top1",
                "cand margin",
                "cand entropy",
                "LZH margin",
                "WYH margin",
                "LZH entropy",
                "WYH entropy",
            ],
            conf_rows,
        ),
        conf_text=metrics["confidence_narrative"],
        hard_table=_md_table(
            ["H", "bucket", "support", "w→c", "c→w", "net", "precision"],
            hard_rows,
        ),
        d_table=_md_table(
            [
                "system",
                "changed",
                "w→c",
                "c→w",
                "net",
                "correct",
                "accuracy",
                "macro-F1",
                "folds 0-2",
                "folds 3-4",
            ],
            d_rows,
        ),
        sparse_table=_md_table(
            [
                "H",
                "support",
                "by fold",
                "n sections",
                "n true classes",
                "hard frac",
                "max section frac",
                "flag",
                "reasons",
            ],
            sparse_bits,
        ),
        sparse_text=metrics["sparsity_narrative"],
        leak_test=leak["competition_test_labels_used"],
        leak_true_trigger=leak["true_labels_used_to_define_triggers"],
        leak_diag=leak["true_labels_used_only_for_retrospective_scoring"],
        leak_router=leak["learned_router_trained"],
        leak_thr=leak["threshold_optimized"],
        leak_w=leak["ensemble_weights_optimized"],
        leak_posthoc=leak["posthoc_pattern_promoted_to_candidate_rule"],
        leak_data=leak["new_external_dataset_added"],
        leak_proba=leak["expert_probabilities_altered"],
        leak_retrain=leak["experts_retrained"],
        leak_sid=leak["spatial_id_started"],
        leak_v3=leak["model_v3_created"],
        leak_sub=leak["submission_generated"],
        leak_pred=leak["prediction_csv_modified"],
        leak_frozen=leak["frozen_v3_artifacts_modified"],
        label=decision["label"],
        reason=decision["reason"],
        next_action=decision["next_action"],
    )


def direction_matrix_rows(
    agree: np.ndarray,
    strong_class: np.ndarray,
    candidate_pred: np.ndarray,
    y_true: np.ndarray,
    expert: str,
) -> List[dict]:
    rows = []
    for src in OLIGO_OPC_CLASSES:
        for dst in OLIGO_OPC_CLASSES:
            mask = agree & (strong_class == src) & (candidate_pred == dst)
            acc = directional_accounting(mask, strong_class, candidate_pred, y_true)
            rows.append(
                {
                    "expert": expert,
                    "strong_consensus_class": src,
                    "proposed_class": dst,
                    "support": acc["support"],
                    "wrong_to_correct": acc["wrong_to_correct"],
                    "correct_to_wrong": acc["correct_to_wrong"],
                    "both_correct": acc["both_correct"],
                    "both_wrong": acc["both_wrong"],
                    "net": acc["net"],
                    "patch_precision": acc["patch_precision"],
                    "strong_conditional_accuracy": acc["strong_conditional_accuracy"],
                    "candidate_conditional_accuracy": acc["candidate_conditional_accuracy"],
                    "oracle_conditional_accuracy": acc["oracle_conditional_accuracy"],
                    "is_h1": expert == "sni" and src == OLIGO_1 and dst == OPC_2,
                    "is_h2": expert == "s0" and src == OPC_2 and dst == OLIGO_1,
                    "is_h3": expert == "sni" and src == OLIGO_2 and dst == OPC_2,
                    "predeclared": (
                        (expert == "sni" and src == OLIGO_1 and dst == OPC_2)
                        or (expert == "s0" and src == OPC_2 and dst == OLIGO_1)
                        or (expert == "sni" and src == OLIGO_2 and dst == OPC_2)
                    ),
                }
            )
    return rows


def expert_asymmetry(
    trigger: np.ndarray,
    s0_pred: np.ndarray,
    sni_pred: np.ndarray,
    s0_ok: np.ndarray,
    sni_ok: np.ndarray,
) -> dict:
    t = np.asarray(trigger, dtype=bool)
    n = int(t.sum())
    if n == 0:
        return {
            "support": 0,
            "s0_correct": 0,
            "sni_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "s0_only_correct": 0,
            "sni_only_correct": 0,
            "s0_sni_agree": 0,
            "s0_pred_counts": {},
            "sni_pred_counts": {},
        }
    s0c = np.asarray(s0_ok, dtype=bool)[t]
    snic = np.asarray(sni_ok, dtype=bool)[t]
    return {
        "support": n,
        "s0_correct": int(s0c.sum()),
        "sni_correct": int(snic.sum()),
        "both_correct": int((s0c & snic).sum()),
        "both_wrong": int((~s0c & ~snic).sum()),
        "s0_only_correct": int((s0c & ~snic).sum()),
        "sni_only_correct": int((~s0c & snic).sum()),
        "s0_sni_agree": int(
            (np.asarray(s0_pred, dtype=object)[t] == np.asarray(sni_pred, dtype=object)[t]).sum()
        ),
        "s0_pred_counts": dict(Counter(np.asarray(s0_pred, dtype=object)[t].tolist())),
        "sni_pred_counts": dict(Counter(np.asarray(sni_pred, dtype=object)[t].tolist())),
    }


def conf_group(mask: np.ndarray, registry: pd.DataFrame, candidate_prefix: str) -> dict:
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        empty = numeric_summary(np.array([]))
        return {
            "n": 0,
            "candidate_top1": empty,
            "candidate_margin": empty,
            "candidate_entropy": empty,
            "lzh_margin": empty,
            "wyh_margin": empty,
            "lzh_entropy": empty,
            "wyh_entropy": empty,
            "lzh_top1": empty,
            "wyh_top1": empty,
        }
    sub = registry.loc[m]
    return {
        "n": n,
        "candidate_top1": numeric_summary(sub["{}_top1".format(candidate_prefix)].to_numpy()),
        "candidate_margin": numeric_summary(sub["{}_margin".format(candidate_prefix)].to_numpy()),
        "candidate_entropy": numeric_summary(sub["{}_entropy".format(candidate_prefix)].to_numpy()),
        "lzh_margin": numeric_summary(sub["lzh_margin"].to_numpy()),
        "wyh_margin": numeric_summary(sub["wyh_margin"].to_numpy()),
        "lzh_entropy": numeric_summary(sub["lzh_entropy"].to_numpy()),
        "wyh_entropy": numeric_summary(sub["wyh_entropy"].to_numpy()),
        "lzh_top1": numeric_summary(sub["lzh_top1"].to_numpy()),
        "wyh_top1": numeric_summary(sub["wyh_top1"].to_numpy()),
    }


def future_work_from_matrix(rows: List[dict]) -> str:
    extras = []
    for r in rows:
        if r["predeclared"] or r["support"] < MIN_SUPPORT:
            continue
        if r["net"] >= 5 and (r["patch_precision"] or 0.0) >= 0.60:
            extras.append(
                "{} `{}` → `{}` (support {}, net {}, precision {})".format(
                    r["expert"].upper(),
                    r["strong_consensus_class"],
                    r["proposed_class"],
                    r["support"],
                    r["net"],
                    fmt_prec(r["patch_precision"]),
                )
            )
    if not extras:
        return (
            "No non-predeclared oligo/OPC consensus→candidate pair met the descriptive "
            "filter (support ≥ 15, net ≥ 5, precision ≥ 0.60). Nothing is promoted."
        )
    return (
        "HYPOTHESIS FOR FUTURE WORK (not E05A evidence; not a candidate rule): "
        + "; ".join(extras)
        + ". These pairs were not added to H1/H2/H3."
    )


def main() -> int:
    if current_branch() != EXPECTED_BRANCH:
        raise SystemExit("Refusing to run: current branch is not {}".format(EXPECTED_BRANCH))
    head = run_git(["rev-parse", "--short=7", "HEAD"]).strip()
    if not trigger_functions_use_predictions_only():
        integrity_failure("H1/H2/H3 trigger signatures are not prediction-only")

    pred_before = sha256_file(PRED_PATH) if PRED_PATH.is_file() else None
    frozen_before = {str(p): sha256_file(p) for p in FROZEN_PATHS if p.is_file()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset(ROOT)
    class_names = allowed_labels(ROOT)
    if len(class_names) != N_CLASSES:
        integrity_failure("expected 60 classes, found {}".format(len(class_names)))
    train_ids = [str(v) for v in data.meta_train.index.tolist()]
    y_true_official = data.meta_train[TARGET_COL].astype(str)
    if y_true_official.isna().any() or (y_true_official == "").any():
        integrity_failure("missing training labels")

    e03a = pd.read_parquet(E03A_REGISTRY)
    e03a["Cell_ID"] = e03a["Cell_ID"].astype(str)
    if len(e03a) != N_TRAIN or e03a["Cell_ID"].nunique() != N_TRAIN:
        integrity_failure("E03A registry is not 5000 unique Cell_IDs")
    e03a = e03a.set_index("Cell_ID").reindex(train_ids)
    if e03a["true_label"].isna().any():
        integrity_failure("E03A registry missing canonical Cell_IDs")

    s0_val = read_cell_id_csv(E02D_S0).set_index("Cell_ID").reindex(train_ids)
    sni_val = read_cell_id_csv(E04S_VAL).set_index("Cell_ID").reindex(train_ids)
    if s0_val["predicted_label"].isna().any():
        integrity_failure("S0 validation missing canonical Cell_IDs")
    if sni_val["predicted_label"].isna().any():
        integrity_failure("SNI validation missing canonical Cell_IDs")

    official_labels = list(y_true_official.reindex(train_ids).astype(str))
    if list(e03a["true_label"].astype(str)) != official_labels:
        integrity_failure("E03A labels do not match official meta_train")
    if list(s0_val["true_label"].astype(str)) != official_labels:
        integrity_failure("S0 validation labels do not match official meta_train")
    if list(sni_val["true_label"].astype(str)) != official_labels:
        integrity_failure("SNI validation labels do not match official meta_train")
    if list(e03a["s0_pred"].astype(str)) != list(s0_val["predicted_label"].astype(str)):
        integrity_failure("E03A s0_pred does not match frozen E02D S0 validation")

    sni_proba = pd.read_csv(E04S_PROBA, dtype={"Cell_ID": str})
    sni_proba["Cell_ID"] = sni_proba["Cell_ID"].astype(str)
    sni_proba = sni_proba.set_index("Cell_ID").reindex(train_ids)
    if list(sni_proba.columns) != list(class_names):
        integrity_failure("SNI probability columns are not official class order")
    sni_arr = sni_proba.loc[:, class_names].to_numpy(dtype=np.float64)
    assert_probability_rows(sni_arr, atol=1e-4)
    sni_argmax = argmax_labels(sni_arr, class_names)
    if list(sni_argmax) != list(sni_val["predicted_label"].astype(str)):
        integrity_failure("SNI probability argmax does not match frozen SNI validation labels")
    sni_conf = confidence_from_proba(sni_arr)

    y_true = np.asarray(official_labels, dtype=object)
    lzh_pred = e03a["lzh_pred"].astype(str).to_numpy()
    wyh_pred = e03a["wyh_pred"].astype(str).to_numpy()
    s0_pred = e03a["s0_pred"].astype(str).to_numpy()
    sni_pred = sni_val["predicted_label"].astype(str).to_numpy()
    canonical_fold = e03a["canonical_fold"].astype(int).to_numpy()
    hard_bucket = e03a["hard_bucket"].astype(bool).to_numpy()
    sections = e03a["Section_ID"].astype(str).to_numpy()

    lzh_ok = lzh_pred == y_true
    wyh_ok = wyh_pred == y_true
    s0_ok = s0_pred == y_true
    sni_ok = sni_pred == y_true
    two_ok = lzh_ok | wyh_ok
    three_ok = two_ok | s0_ok
    four_ok = three_ok | sni_ok
    s0_unique = (~lzh_ok) & (~wyh_ok) & s0_ok
    sni_unique = (~lzh_ok) & (~wyh_ok) & (~s0_ok) & sni_ok
    all_four_wrong = ~four_ok
    fold_02 = canonical_fold <= 2
    fold_34 = canonical_fold >= 3

    checks = [
        ("lzh_correct", int(lzh_ok.sum()), LZH_CORRECT),
        ("wyh_correct", int(wyh_ok.sum()), WYH_CORRECT),
        ("s0_correct", int(s0_ok.sum()), S0_CORRECT),
        ("sni_correct", int(sni_ok.sum()), SNI_CORRECT),
        ("two_expert_oracle", int(two_ok.sum()), TWO_EXPERT_ORACLE),
        ("three_expert_oracle", int(three_ok.sum()), THREE_EXPERT_ORACLE),
        ("four_expert_oracle", int(four_ok.sum()), FOUR_EXPERT_ORACLE),
        ("all_four_wrong", int(all_four_wrong.sum()), ALL_FOUR_WRONG),
        ("s0_unique_recoveries", int(s0_unique.sum()), S0_UNIQUE_RECOVERIES),
        ("sni_unique_recoveries", int(sni_unique.sum()), SNI_UNIQUE_RECOVERIES),
        ("four_expert_folds_0_2_correct", int(four_ok[fold_02].sum()), FOUR_EXPERT_FOLDS_02_CORRECT),
        ("four_expert_folds_3_4_correct", int(four_ok[fold_34].sum()), FOUR_EXPERT_FOLDS_34_CORRECT),
    ]
    integrity = {}
    for name, got, expected in checks:
        integrity[name] = got
        if got != expected:
            integrity_failure("{} {} != {}".format(name, got, expected))
    if int(four_ok.sum()) != THREE_EXPERT_ORACLE + SNI_UNIQUE_RECOVERIES:
        integrity_failure("four-expert oracle identity 4311 + 53 failed")

    lzh_wyh_agree = lzh_pred == wyh_pred
    strong_consensus_class = np.where(lzh_wyh_agree, lzh_pred, "")
    s0_disagrees_with_strong = lzh_wyh_agree & (s0_pred != lzh_pred)
    sni_disagrees_with_strong = lzh_wyh_agree & (sni_pred != lzh_pred)
    s0_sni_agree = s0_pred == sni_pred
    all_four_agree = lzh_wyh_agree & (lzh_pred == s0_pred) & (lzh_pred == sni_pred)
    all_four_different = (
        (lzh_pred != wyh_pred)
        & (lzh_pred != s0_pred)
        & (lzh_pred != sni_pred)
        & (wyh_pred != s0_pred)
        & (wyh_pred != sni_pred)
        & (s0_pred != sni_pred)
    )

    h1 = trigger_h1(lzh_pred, wyh_pred, sni_pred)
    h2 = trigger_h2(lzh_pred, wyh_pred, s0_pred)
    h3 = trigger_h3(lzh_pred, wyh_pred, sni_pred)
    if np.any(h1 & h2) or np.any(h1 & h3) or np.any(h2 & h3):
        integrity_failure("H1/H2/H3 triggers overlap; they must be mutually exclusive")
    if np.any(h1 & ~lzh_wyh_agree) or np.any(h2 & ~lzh_wyh_agree) or np.any(h3 & ~lzh_wyh_agree):
        integrity_failure("a directional trigger fired without LZH/WYH consensus")

    registry = pd.DataFrame(
        {
            "Cell_ID": train_ids,
            "true_label": y_true,
            "canonical_fold": canonical_fold,
            "lzh_pred": lzh_pred,
            "lzh_correct": lzh_ok,
            "lzh_top1": e03a["lzh_top1"].to_numpy(dtype=np.float64),
            "lzh_margin": e03a["lzh_margin"].to_numpy(dtype=np.float64),
            "lzh_entropy": e03a["lzh_entropy"].to_numpy(dtype=np.float64),
            "wyh_pred": wyh_pred,
            "wyh_correct": wyh_ok,
            "wyh_top1": e03a["wyh_top1"].to_numpy(dtype=np.float64),
            "wyh_margin": e03a["wyh_margin"].to_numpy(dtype=np.float64),
            "wyh_entropy": e03a["wyh_entropy"].to_numpy(dtype=np.float64),
            "s0_pred": s0_pred,
            "s0_correct": s0_ok,
            "s0_top1": e03a["s0_top1"].to_numpy(dtype=np.float64),
            "s0_margin": e03a["s0_margin"].to_numpy(dtype=np.float64),
            "s0_entropy": e03a["s0_entropy"].to_numpy(dtype=np.float64),
            "sni_pred": sni_pred,
            "sni_correct": sni_ok,
            "sni_top1": sni_conf["top1"],
            "sni_margin": sni_conf["margin"],
            "sni_entropy": sni_conf["entropy"],
            "Region": e03a["Region"].astype(str).to_numpy(),
            "E/I": e03a["E/I"].astype(str).to_numpy(),
            "Segment": e03a["Segment"].astype(str).to_numpy(),
            "Section_ID": sections,
            "hard_bucket": hard_bucket,
            "n_detected": e03a["n_detected"].to_numpy(),
            "library_size": e03a["library_size"].to_numpy(dtype=np.float64),
            "neuron_or_glial": e03a["neuron_or_glial"].astype(str).to_numpy(),
            "lzh_wyh_agree": lzh_wyh_agree,
            "strong_consensus_class": strong_consensus_class,
            "s0_disagrees_with_strong": s0_disagrees_with_strong,
            "sni_disagrees_with_strong": sni_disagrees_with_strong,
            "s0_sni_agree": s0_sni_agree,
            "all_four_agree": all_four_agree,
            "all_four_different": all_four_different,
            "h1_trigger": h1,
            "h2_trigger": h2,
            "h3_trigger": h3,
            "four_expert_oracle_correct": four_ok,
            "all_four_wrong": all_four_wrong,
            "s0_unique_recovery": s0_unique,
            "sni_unique_recovery": sni_unique,
            "true_family": [family_of(v) for v in y_true],
            "diagnostic_only_note": (
                "true_label/correctness/oracle/rescue/unique-recovery flags are diagnostic-only; "
                "deployment features are predicted classes, agreement, probabilities, and test-safe metadata"
            ),
        }
    )
    registry.to_parquet(OUT_DIR / "v3_e05a_four_expert_registry.parquet", index=False)

    trigger_specs = {
        "H1": {
            "mask": h1,
            "candidate_pred": sni_pred,
            "candidate_expert": "sni",
            "candidate_class": OPC_2,
            "strong_class": OLIGO_1,
            "trigger_text": "LZH_pred == oligodendrocyte_1 AND WYH_pred == oligodendrocyte_1 AND SNI_pred == oligodendrocyte_progenitor_2",
        },
        "H2": {
            "mask": h2,
            "candidate_pred": s0_pred,
            "candidate_expert": "s0",
            "candidate_class": OLIGO_1,
            "strong_class": OPC_2,
            "trigger_text": "LZH_pred == oligodendrocyte_progenitor_2 AND WYH_pred == oligodendrocyte_progenitor_2 AND S0_pred == oligodendrocyte_1",
        },
        "H3": {
            "mask": h3,
            "candidate_pred": sni_pred,
            "candidate_expert": "sni",
            "candidate_class": OPC_2,
            "strong_class": OLIGO_2,
            "trigger_text": "LZH_pred == oligodendrocyte_2 AND WYH_pred == oligodendrocyte_2 AND SNI_pred == oligodendrocyte_progenitor_2",
        },
    }

    hyp_rows = []
    hyp_metrics = OrderedDict()
    hard_metrics = {}
    conf_metrics = {}
    asym_metrics = {}
    for hid, spec in trigger_specs.items():
        parts = partition_accounting(
            spec["mask"], lzh_pred, spec["candidate_pred"], y_true, canonical_fold
        )
        act = actionable_checks(parts["overall"], parts["folds_0_2"], parts["folds_3_4"])
        sparse = sparsity_audit(spec["mask"], canonical_fold, sections, y_true, hard_bucket)
        row = {
            "id": hid,
            "name": HYPOTHESES[hid]["name"],
            "trigger_text": spec["trigger_text"],
            "strong_class": spec["strong_class"],
            "candidate_expert": spec["candidate_expert"],
            "candidate_class": spec["candidate_class"],
            "trigger_uses_true_label": False,
            "trigger_uses_correctness_flags": False,
            "motivation": HYPOTHESES[hid]["motivation"],
            "overall": parts["overall"],
            "folds_0_2": parts["folds_0_2"],
            "folds_3_4": parts["folds_3_4"],
            "actionable": act["actionable"],
            "checks": act["checks"],
            "sparsity": sparse,
        }
        hyp_metrics[hid] = row
        for split_name, acc in (
            ("overall", parts["overall"]),
            ("folds_0_2", parts["folds_0_2"]),
            ("folds_3_4", parts["folds_3_4"]),
        ):
            hyp_rows.append(
                {
                    "hypothesis": hid,
                    "split": split_name,
                    "trigger_uses_true_label": False,
                    "strong_class": spec["strong_class"],
                    "candidate_expert": spec["candidate_expert"],
                    "candidate_class": spec["candidate_class"],
                    **acc,
                    "actionable_overall": act["actionable"] if split_name == "overall" else "",
                    "sparse_unstable": sparse["sparse_unstable"] if split_name == "overall" else "",
                }
            )
        hard_metrics[hid] = {
            "True": directional_accounting(
                spec["mask"] & hard_bucket, lzh_pred, spec["candidate_pred"], y_true
            ),
            "False": directional_accounting(
                spec["mask"] & (~hard_bucket), lzh_pred, spec["candidate_pred"], y_true
            ),
        }
        t = spec["mask"]
        s_ok = lzh_pred[t] == y_true[t]
        c_ok = spec["candidate_pred"][t] == y_true[t]
        w2c_local = np.zeros(N_TRAIN, dtype=bool)
        c2w_local = np.zeros(N_TRAIN, dtype=bool)
        idx = np.where(t)[0]
        w2c_local[idx] = (~s_ok) & c_ok
        c2w_local[idx] = s_ok & (~c_ok)
        conf_metrics[hid] = {
            "wrong_to_correct": conf_group(w2c_local, registry, spec["candidate_expert"]),
            "correct_to_wrong": conf_group(c2w_local, registry, spec["candidate_expert"]),
            "trigger": conf_group(t, registry, spec["candidate_expert"]),
        }
        asym_metrics[hid] = expert_asymmetry(t, s0_pred, sni_pred, s0_ok, sni_ok)

    pd.DataFrame(hyp_rows).to_csv(OUT_DIR / "v3_e05a_directional_hypotheses.csv", index=False)

    oligo_rows = direction_matrix_rows(
        lzh_wyh_agree, strong_consensus_class, s0_pred, y_true, "s0"
    ) + direction_matrix_rows(lzh_wyh_agree, strong_consensus_class, sni_pred, y_true, "sni")
    pd.DataFrame(oligo_rows).to_csv(OUT_DIR / "v3_e05a_oligo_direction_matrix.csv", index=False)

    d0 = apply_fixed_patches(lzh_pred, h1, h2, h3, s0_pred, sni_pred, "D0")
    d1 = apply_fixed_patches(lzh_pred, h1, h2, h3, s0_pred, sni_pred, "D1")
    d2p = apply_fixed_patches(lzh_pred, h1, h2, h3, s0_pred, sni_pred, "D2")
    d3 = apply_fixed_patches(lzh_pred, h1, h2, h3, s0_pred, sni_pred, "D3")
    dsys = {
        "D0": system_metrics(y_true, d0, lzh_pred, canonical_fold, class_names, "D0"),
        "D1": system_metrics(y_true, d1, lzh_pred, canonical_fold, class_names, "D1"),
        "D2": system_metrics(y_true, d2p, lzh_pred, canonical_fold, class_names, "D2"),
        "D3": system_metrics(y_true, d3, lzh_pred, canonical_fold, class_names, "D3"),
    }
    if dsys["D0"]["correct"] != LZH_CORRECT:
        integrity_failure("D0 correct {} != LZH 4133".format(dsys["D0"]["correct"]))
    if dsys["D1"]["net"] != hyp_metrics["H1"]["overall"]["net"]:
        integrity_failure("D1 net does not equal H1 net")
    if dsys["D2"]["net"] != hyp_metrics["H1"]["overall"]["net"] + hyp_metrics["H2"]["overall"]["net"]:
        integrity_failure("D2 net does not equal H1+H2 net")
    if dsys["D3"]["net"] != (
        hyp_metrics["H1"]["overall"]["net"]
        + hyp_metrics["H2"]["overall"]["net"]
        + hyp_metrics["H3"]["overall"]["net"]
    ):
        integrity_failure("D3 net does not equal H1+H2+H3 net")
    pd.DataFrame([dsys[k] for k in ("D0", "D1", "D2", "D3")]).to_csv(
        OUT_DIR / "v3_e05a_fixed_patch_diagnostics.csv", index=False
    )

    agree_err = lzh_wyh_agree & (~lzh_ok)
    sc_by_class = []
    for cls in list(OLIGO_OPC_CLASSES) + sorted(
        set(strong_consensus_class[lzh_wyh_agree].tolist()) - set(OLIGO_OPC_CLASSES)
    ):
        m = lzh_wyh_agree & (strong_consensus_class == cls)
        n = int(m.sum())
        if n == 0:
            continue
        errors = m & (~lzh_ok)
        sc_by_class.append(
            {
                "strong_consensus_class": cls,
                "n": n,
                "errors": int(errors.sum()),
                "accuracy": float(lzh_ok[m].mean()),
                "s0_rescues": int((errors & s0_ok).sum()),
                "sni_rescues": int((errors & sni_ok).sum()),
                "either_rescues": int((errors & (s0_ok | sni_ok)).sum()),
                "both_rescues": int((errors & s0_ok & sni_ok).sum()),
                "family": family_of(cls),
            }
        )
    sc_by_class.sort(key=lambda r: (-r["errors"], -r["n"], r["strong_consensus_class"]))
    pd.DataFrame(sc_by_class).to_csv(OUT_DIR / "v3_e05a_strong_consensus_analysis.csv", index=False)
    strong_consensus = {
        "n_agree": int(lzh_wyh_agree.sum()),
        "n_disagree": int((~lzh_wyh_agree).sum()),
        "accuracy": float(lzh_ok[lzh_wyh_agree].mean()) if lzh_wyh_agree.any() else None,
        "n_errors": int(agree_err.sum()),
        "s0_rescues": int((agree_err & s0_ok).sum()),
        "sni_rescues": int((agree_err & sni_ok).sum()),
        "either_rescues": int((agree_err & (s0_ok | sni_ok)).sum()),
        "both_rescues": int((agree_err & s0_ok & sni_ok).sum()),
        "by_consensus_class": sc_by_class,
    }

    true_family = np.asarray([family_of(v) for v in y_true], dtype=object)
    ctrl_rows = []
    families = ["oligodendrocyte_opc"] + CONTROL_FAMILIES
    for fam in families:
        fam_mask = true_family == fam
        cons = fam_mask & lzh_wyh_agree
        errors = cons & (~lzh_ok)
        ctrl_rows.append(
            {
                "family": fam,
                "n_true": int(fam_mask.sum()),
                "strong_consensus_n": int(cons.sum()),
                "strong_consensus_errors": int(errors.sum()),
                "s0_rescues": int((errors & s0_ok).sum()),
                "sni_rescues": int((errors & sni_ok).sum()),
                "either_rescues": int((errors & (s0_ok | sni_ok)).sum()),
                "both_rescues": int((errors & s0_ok & sni_ok).sum()),
            }
        )
    oligo_s0 = next(r["s0_rescues"] for r in ctrl_rows if r["family"] == "oligodendrocyte_opc")
    oligo_sni = next(r["sni_rescues"] for r in ctrl_rows if r["family"] == "oligodendrocyte_opc")
    non_oligo_s0 = strong_consensus["s0_rescues"] - oligo_s0
    non_oligo_sni = strong_consensus["sni_rescues"] - oligo_sni
    pd.DataFrame(ctrl_rows).to_csv(TABLE_DIR / "non_oligo_controls.csv", index=False)

    pd.DataFrame(
        [
            {"hypothesis": hid, "hard_bucket": bucket, **acc}
            for hid, buckets in hard_metrics.items()
            for bucket, acc in buckets.items()
        ]
    ).to_csv(TABLE_DIR / "hard_bucket_hypotheses.csv", index=False)
    pd.DataFrame([{"hypothesis": hid, **row} for hid, row in asym_metrics.items()]).to_csv(
        TABLE_DIR / "s0_vs_sni_on_triggers.csv", index=False
    )

    triggered = registry.loc[h1 | h2 | h3, [
        "Cell_ID",
        "true_label",
        "canonical_fold",
        "Section_ID",
        "hard_bucket",
        "lzh_pred",
        "wyh_pred",
        "s0_pred",
        "sni_pred",
        "h1_trigger",
        "h2_trigger",
        "h3_trigger",
        "lzh_correct",
        "s0_correct",
        "sni_correct",
    ]]
    triggered.to_csv(TABLE_DIR / "triggered_cells.csv", index=False)

    h1a = hyp_metrics["H1"]["overall"]
    h2a = hyp_metrics["H2"]["overall"]
    h3a = hyp_metrics["H3"]["overall"]
    if h1a["net"] > 0 and asym_metrics["H1"]["sni_only_correct"] >= asym_metrics["H1"]["s0_only_correct"]:
        h1_asymm = (
            "On H1 cells (strong oligo-1 consensus, SNI proposes progenitor-2), SNI is the "
            "designated expert. SNI-only correct={} vs S0-only correct={}.".format(
                asym_metrics["H1"]["sni_only_correct"], asym_metrics["H1"]["s0_only_correct"]
            )
        )
    else:
        h1_asymm = (
            "On H1 cells, the designated SNI expert is not clearly superior to S0 "
            "(SNI-only {}, S0-only {}, net {}).".format(
                asym_metrics["H1"]["sni_only_correct"],
                asym_metrics["H1"]["s0_only_correct"],
                h1a["net"],
            )
        )
    if h2a["net"] > 0 and asym_metrics["H2"]["s0_only_correct"] >= asym_metrics["H2"]["sni_only_correct"]:
        h2_asymm = (
            "On H2 cells (strong progenitor-2 consensus, S0 proposes oligo-1), S0 is the "
            "designated expert. S0-only correct={} vs SNI-only correct={}.".format(
                asym_metrics["H2"]["s0_only_correct"], asym_metrics["H2"]["sni_only_correct"]
            )
        )
    else:
        h2_asymm = (
            "On H2 cells, the designated S0 expert is not clearly superior to SNI "
            "(S0-only {}, SNI-only {}, net {}).".format(
                asym_metrics["H2"]["s0_only_correct"],
                asym_metrics["H2"]["sni_only_correct"],
                h2a["net"],
            )
        )
    opposite = (
        h1a["net"] > 0
        and h2a["net"] > 0
        and asym_metrics["H1"]["sni_only_correct"] > 0
        and asym_metrics["H2"]["s0_only_correct"] > 0
    )
    asymmetry_narrative = (
        "{} {} H3 net={}. Opposite-direction complementarity between S0 and SNI is {}.".format(
            h1_asymm,
            h2_asymm,
            h3a["net"],
            "supported on the predeclared H1 vs H2 pair" if opposite else "not cleanly supported",
        )
    )

    oligo_err = next(r["strong_consensus_errors"] for r in ctrl_rows if r["family"] == "oligodendrocyte_opc")
    total_err = strong_consensus["n_errors"]
    s0_frac = oligo_s0 / strong_consensus["s0_rescues"] if strong_consensus["s0_rescues"] else None
    sni_frac = oligo_sni / strong_consensus["sni_rescues"] if strong_consensus["sni_rescues"] else None
    err_frac = oligo_err / total_err if total_err else None
    control_narrative = (
        "Oligo/OPC accounts for {} / {} strong-consensus errors ({:.1%}). "
        "S0 strong-consensus rescues in that family: {} / {} ({:.1%}). "
        "SNI strong-consensus rescues in that family: {} / {} ({:.1%}). "
        "Absolute unique-rescue mass is largest in oligo/OPC, but S0 is close to the error "
        "base rate and non-oligo families still contribute. This is family-skewed unique "
        "coverage, not a license to patch non-oligo directions. No non-oligo patch rule is proposed.".format(
            oligo_err,
            total_err,
            err_frac or 0.0,
            oligo_s0,
            strong_consensus["s0_rescues"],
            s0_frac or 0.0,
            oligo_sni,
            strong_consensus["sni_rescues"],
            sni_frac or 0.0,
        )
    )

    conf_notes = []
    for hid in ("H1", "H2", "H3"):
        w = conf_metrics[hid]["wrong_to_correct"]
        c = conf_metrics[hid]["correct_to_wrong"]
        if w["n"] == 0 or c["n"] == 0:
            conf_notes.append(
                "{} lacks both a wrong→correct and a correct→wrong group, so confidence "
                "cannot separate safe vs unsafe patches.".format(hid)
            )
            continue
        cand_sep = (w["candidate_margin"]["mean"] or 0) - (c["candidate_margin"]["mean"] or 0)
        conf_notes.append(
            "{} candidate margin mean is {:.3f} on wrong→correct vs {:.3f} on correct→wrong "
            "(difference {:+.3f}); this is descriptive only and does not justify a threshold.".format(
                hid,
                w["candidate_margin"]["mean"] or float("nan"),
                c["candidate_margin"]["mean"] or float("nan"),
                cand_sep,
            )
        )
    confidence_narrative = " ".join(conf_notes)

    sparse_flags = [hid for hid, row in hyp_metrics.items() if row["sparsity"]["sparse_unstable"]]
    if sparse_flags:
        sparsity_narrative = (
            "{} flagged SPARSE / UNSTABLE because support is small or concentrated in one "
            "fold or Section. A high patch precision on a tiny trigger is not sufficient.".format(
                ", ".join(sparse_flags)
            )
        )
    else:
        sparsity_narrative = (
            "None of H1/H2/H3 is flagged as a single-fold or single-section artifact under "
            "the predeclared sparsity rules, though absolute support remains limited."
        )

    decision = classify_experiment(hyp_metrics)
    if decision["label"] == "DIRECTIONAL SIGNAL WEAK":
        decision["reason"] = (
            "None of H1/H2/H3 provides stable positive net correction under the "
            "predeclared actionable criteria. Each observable trigger mixes both sides of "
            "an oligo/OPC confusion: H1 net -30 (precision 0.3125), H2 net -8 "
            "(precision 0.4310), H3 net -14 (precision 0.2812). Unconditional replacement "
            "of strong consensus therefore harms more cells than it rescues on both "
            "canonical folds 0-2 and the retrospective stability partition."
        )

    pred_after = sha256_file(PRED_PATH) if PRED_PATH.is_file() else None
    frozen_after = {str(p): sha256_file(p) for p in FROZEN_PATHS if p.is_file()}
    frozen_changed = sorted(k for k in frozen_before if frozen_before[k] != frozen_after.get(k))
    if pred_before != pred_after:
        integrity_failure("prediction/prediction.csv changed during E05A")
    if frozen_changed:
        integrity_failure("frozen artifacts changed: {}".format(frozen_changed))

    leakage = {
        "competition_test_labels_used": False,
        "true_labels_used_to_define_triggers": False,
        "true_labels_used_only_for_retrospective_scoring": True,
        "learned_router_trained": False,
        "classifier_fit": False,
        "threshold_optimized": False,
        "ensemble_weights_optimized": False,
        "posthoc_pattern_promoted_to_candidate_rule": False,
        "new_external_dataset_added": False,
        "sni_concatenated_with_merfish_reference": False,
        "expert_probabilities_altered": False,
        "experts_retrained": False,
        "spatial_id_started": False,
        "model_v3_created": False,
        "model_v3_tag_created": False,
        "submission_generated": False,
        "prediction_csv_modified": False,
        "frozen_v3_artifacts_modified": False,
        "h1_h2_h3_motivated_by_prior_full_data_error_analysis": True,
        "d0_d3_are_retrospective_diagnostics_not_unbiased_oof": True,
        "canonical_folds_3_4_are_retrospective_stability_partition": True,
        "oracle_is_not_deployable_accuracy": True,
    }

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "research_codename": RESEARCH_CODENAME,
        "created_at_utc": utc_now(),
        "branch": current_branch(),
        "head": head,
        "oracle_is_not_deployable_accuracy": True,
        "d0_d3_are_not_unbiased_final_oof": True,
        "not_model_v3": True,
        "triggers_use_predictions_only": True,
        "integrity": integrity,
        "hypotheses": hyp_metrics,
        "actionable_criteria": {
            "min_support": MIN_SUPPORT,
            "min_wrong_to_correct": MIN_WRONG_TO_CORRECT,
            "min_precision_overall": MIN_PRECISION_OVERALL,
            "min_precision_folds_0_2": MIN_PRECISION_FOLDS_02,
            "min_precision_folds_3_4": MIN_PRECISION_FOLDS_34,
            "net_must_be_positive_overall_and_both_partitions": True,
            "thresholds_not_changed_after_seeing_results": True,
        },
        "oligo_direction_matrix": oligo_rows,
        "s0_vs_sni": asym_metrics,
        "strong_consensus": strong_consensus,
        "non_oligo_controls": {
            "by_true_family": ctrl_rows,
            "oligo_s0_rescues": oligo_s0,
            "oligo_sni_rescues": oligo_sni,
            "non_oligo_s0_rescues": non_oligo_s0,
            "non_oligo_sni_rescues": non_oligo_sni,
        },
        "confidence_secondary": conf_metrics,
        "hard_bucket": hard_metrics,
        "fixed_patch_diagnostics": dsys,
        "final_classification": decision,
        "leakage_audit": leakage,
        "future_work_note": future_work_from_matrix(oligo_rows),
        "asymmetry_narrative": asymmetry_narrative,
        "control_narrative": control_narrative,
        "confidence_narrative": confidence_narrative,
        "sparsity_narrative": sparsity_narrative,
        "any_predeclared_rule_meets_all_criteria": bool(decision["label"] == "DIRECTIONAL SIGNAL STRONG"),
        "four_expert_coverage_context": {
            "four_expert_oracle_correct": FOUR_EXPERT_ORACLE,
            "four_expert_oracle_accuracy": FOUR_EXPERT_ORACLE / N_TRAIN,
            "all_four_wrong": ALL_FOUR_WRONG,
            "note": "No post-hoc pattern variant was added to raise this oracle.",
        },
    }
    write_json(OUT_DIR / "v3_e05a_directional_metrics.json", metrics)
    REPORT_PATH.write_text(render_report(metrics))

    print("V3-E05A ADE-AUDIT complete", flush=True)
    print("classification={}".format(decision["label"]), flush=True)
    print("H1 support={} net={} precision={}".format(h1a["support"], h1a["net"], h1a["patch_precision"]), flush=True)
    print("H2 support={} net={} precision={}".format(h2a["support"], h2a["net"], h2a["patch_precision"]), flush=True)
    print("H3 support={} net={} precision={}".format(h3a["support"], h3a["net"], h3a["patch_precision"]), flush=True)
    print("report={}".format(REPORT_PATH), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
