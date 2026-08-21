#!/usr/bin/env python3
"""V3-E03A: Weak-but-Diverse Expert Rescue Audit (CER-AUDIT).

Diagnostic-only. Does not train a gate, retrain S0, continue distillation,
or write prediction/prediction.csv.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "v3"))

from merfish60.io import TARGET_COL, load_dataset  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.official_contract import allowed_labels, sha256_file  # noqa: E402
from merfish60.v2_metrics import json_default, write_json, write_proba  # noqa: E402

from v3_e00t_team_expert_audit import (  # noqa: E402
    LZH_BRANCH,
    LZH_FINAL_OOF,
    confidence_from_proba,
    js_divergence_rows,
    load_probability_frame,
    materialize_git_blob,
    summarize_numeric,
)
from v3_e02d_privileged_gene_distillation import (  # noqa: E402
    build_model_classes,
    evaluate_model,
    family_of as family_of_e02d,
)

EXPERIMENT_ID = "V3-E03A"
N_TRAIN = 5000
N_CLASSES = 60
EXPECTED_BRANCH = "ywan/ml-pipeline"
EXPECTED_HEAD_PREFIX = "718395d"

LZH_CORRECT = 4133
WYH_CORRECT = 4106
S0_CORRECT = 3151
LZH_WYH_ORACLE = 4215
BOTH_WRONG = 785
POSITIVE_RESCUES = 96
THREE_EXPERT_ORACLE = 4311
ALL_THREE_WRONG = 689
FOLD_RESCUES = {0: 25, 1: 28, 2: 16, 3: 17, 4: 10}

E00T_REGISTRY = ROOT / "outputs" / "v3" / "v3_e00t_team_oof_registry.parquet"
S0_VALIDATION = ROOT / "outputs" / "v3" / "v3_e02d_s0_validation.csv"
S0_CHECKPOINT = ROOT / "work" / "v3_e02d" / "s0.pt"
WYH_V2_OOF_PROBA = ROOT / "outputs" / "probabilities" / "V2-B-REFONLY_oof_probabilities_seg.csv.gz"
PRED_PATH = ROOT / "prediction" / "prediction.csv"
OUT_DIR = ROOT / "outputs" / "v3"
TABLE_DIR = OUT_DIR / "v3_e03a_tables"
REPORT_PATH = ROOT / "reports" / "v3" / "v3_e03a_rescue_audit.md"

CLASS_FAMILIES_E03A = OrderedDict(
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
        ("microglia", ["microglia"]),
        (
            "remaining_glial_non_neuronal",
            ["ependymal", "Schwann_cell", "peripheral_glia"],
        ),
    ]
)

SCALAR_FEATURES = [
    "s0_top1",
    "s0_margin",
    "s0_entropy",
    "lzh_top1",
    "lzh_margin",
    "lzh_entropy",
    "wyh_top1",
    "wyh_margin",
    "wyh_entropy",
    "lzh_wyh_agree",
    "lzh_s0_agree",
    "wyh_s0_agree",
    "all_three_agree",
    "all_three_different",
    "n_experts_supporting_s0",
    "n_strong_supporting_s0",
    "s0_prob_advantage_lzh",
    "s0_prob_advantage_wyh",
    "js_s0_lzh",
    "js_s0_wyh",
    "js_lzh_wyh",
    "n_detected",
    "library_size",
    "hard_bucket",
    "rescue_evidence_score",
]

QUANTILE_FEATURES = ["s0_top1", "s0_margin", "s0_entropy", "lzh_margin", "wyh_margin"]

FEATURE_META = {
    "s0_top1": ("S0 probability", True),
    "s0_margin": ("S0 probability", True),
    "s0_entropy": ("S0 probability", True),
    "lzh_top1": ("LZH Prior-H probability", True),
    "lzh_margin": ("LZH Prior-H probability", True),
    "lzh_entropy": ("LZH Prior-H probability", True),
    "wyh_top1": ("WYH MODEL V2 probability", True),
    "wyh_margin": ("WYH MODEL V2 probability", True),
    "wyh_entropy": ("WYH MODEL V2 probability", True),
    "lzh_wyh_agree": ("cross-expert predicted class", True),
    "lzh_s0_agree": ("cross-expert predicted class", True),
    "wyh_s0_agree": ("cross-expert predicted class", True),
    "all_three_agree": ("cross-expert predicted class", True),
    "all_three_different": ("cross-expert predicted class", True),
    "n_experts_supporting_s0": ("cross-expert predicted class", True),
    "n_strong_supporting_s0": ("cross-expert predicted class", True),
    "s0_prob_advantage_lzh": ("cross-expert top1 difference", True),
    "s0_prob_advantage_wyh": ("cross-expert top1 difference", True),
    "js_s0_lzh": ("cross-expert JS divergence, base 2", True),
    "js_s0_wyh": ("cross-expert JS divergence, base 2", True),
    "js_lzh_wyh": ("cross-expert JS divergence, base 2", True),
    "n_detected": ("counts_train nonzero official genes", True),
    "library_size": ("counts_train library size", True),
    "hard_bucket": ("metadata missingness", True),
    "rescue_evidence_score": ("predeclared composite of z-scored margins", True),
    "lzh_prior_h_eligibility": ("LZH route audit", True),
    "lzh_graph_degree": ("LZH graph", True),
    "lzh_reliable_gene_count": ("LZH Prior-H", True),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(ROOT), stderr=subprocess.STDOUT).decode(
        "utf-8"
    )


def current_branch() -> str:
    return run_git(["branch", "--show-current"]).strip()


def family_of(label: str) -> str:
    for family, members in CLASS_FAMILIES_E03A.items():
        if label in members:
            return family
    return "neuronal_or_other"


def zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def rescue_evidence_score(s0_margin: np.ndarray, lzh_margin: np.ndarray, wyh_margin: np.ndarray) -> np.ndarray:
    return zscore(s0_margin) - 0.5 * zscore(lzh_margin) - 0.5 * zscore(wyh_margin)


def safe_auc(y_bin: np.ndarray, scores: np.ndarray) -> Optional[float]:
    mask = np.isfinite(scores)
    y = np.asarray(y_bin, dtype=int)[mask]
    s = np.asarray(scores, dtype=np.float64)[mask]
    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def oriented_auc(auc: Optional[float]) -> Optional[float]:
    if auc is None:
        return None
    return float(max(auc, 1.0 - auc))


def direction_from_means(mean_p: Optional[float], mean_n: Optional[float]) -> Optional[str]:
    if mean_p is None or mean_n is None:
        return None
    if mean_p > mean_n:
        return "higher_in_P"
    if mean_p < mean_n:
        return "lower_in_P"
    return "tied"


def integrity_failure(message: str) -> None:
    raise SystemExit("REGISTRY INTEGRITY FAILURE: {}".format(message))


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


def infer_s0_probabilities(
    cell_ids: Sequence[str],
    class_names: Sequence[str],
    x_log1p: np.ndarray,
    frozen_pred: np.ndarray,
) -> dict:
    if not S0_CHECKPOINT.is_file():
        return {
            "available": False,
            "reason": "frozen S0 checkpoint missing: {}".format(S0_CHECKPOINT),
            "proba": None,
        }
    import torch

    sha = sha256_file(S0_CHECKPOINT)
    TeacherMLP, StudentMLP = build_model_classes(torch, torch.nn)
    ckpt = torch.load(S0_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(ckpt.get("in_dim", -1)) != 200:
        return {
            "available": False,
            "reason": "S0 checkpoint in_dim {} != 200".format(ckpt.get("in_dim")),
            "proba": None,
            "checkpoint_sha256": sha,
        }
    device = torch.device("cpu")
    model = StudentMLP(200)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    _, proba, _ = evaluate_model(torch, model, x_log1p, None, device)
    assert_probability_rows(proba, atol=1e-4)
    pred = argmax_labels(proba, class_names)
    n_match = int(np.sum(pred == frozen_pred))
    if n_match != len(frozen_pred):
        return {
            "available": False,
            "reason": "S0 inference labels match frozen validation on {} / {} cells".format(
                n_match, len(frozen_pred)
            ),
            "proba": None,
            "checkpoint_sha256": sha,
            "n_label_match": n_match,
        }
    return {
        "available": True,
        "reason": "deterministic CPU inference from frozen work/v3_e02d/s0.pt; no retraining",
        "proba": proba.astype(np.float64),
        "pred": pred,
        "checkpoint_path": str(S0_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha,
        "checkpoint_bytes": int(S0_CHECKPOINT.stat().st_size),
        "device": "cpu",
        "n_label_match": n_match,
        "retrained": False,
    }


def group_feature_stats(values: np.ndarray, mask: np.ndarray) -> dict:
    return summarize_numeric(np.asarray(values, dtype=np.float64)[mask])


def decile_labels(values: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=np.float64))
    try:
        cats = pd.qcut(s, 10, labels=False, duplicates="drop")
    except ValueError:
        cats = pd.Series(np.full(len(s), np.nan))
    return cats.to_numpy()


def mask_metrics(df: pd.DataFrame, mask: np.ndarray) -> dict:
    sub = df.loc[mask]
    n = int(mask.sum())
    if n == 0:
        return {
            "n": 0,
            "lzh_accuracy": None,
            "wyh_accuracy": None,
            "s0_accuracy": None,
            "positive_rescue_count": 0,
            "positive_rescue_rate": None,
            "n1": 0,
            "n2": 0,
            "n3": 0,
            "three_expert_oracle_correct": 0,
            "three_expert_oracle_accuracy": None,
        }
    p = int(sub["positive_s0_rescue"].sum())
    n3 = int(sub["strong_any_correct_s0_wrong"].sum())
    return {
        "n": n,
        "lzh_accuracy": float(sub["lzh_correct"].mean()),
        "wyh_accuracy": float(sub["wyh_correct"].mean()),
        "s0_accuracy": float(sub["s0_correct"].mean()),
        "positive_rescue_count": p,
        "positive_rescue_rate": float(p / n),
        "n1": int(sub["group_n1"].sum()),
        "n2": int(sub["group_n2"].sum()),
        "n3": n3,
        "danger_rate_n3": float(n3 / n),
        "three_expert_oracle_correct": int(sub["three_expert_oracle_correct"].sum()),
        "three_expert_oracle_accuracy": float(sub["three_expert_oracle_correct"].mean()),
    }


def confusion_counts(true_vals: Sequence[str], pred_vals: Sequence[str]) -> List[dict]:
    counts = Counter(zip([str(v) for v in true_vals], [str(v) for v in pred_vals]))
    return [
        {"true_label": t, "pred": p, "n": int(n)}
        for (t, p), n in counts.most_common()
    ]


DEFINITIONAL_ON_P = {
    "lzh_s0_agree",
    "wyh_s0_agree",
    "all_three_agree",
    "n_experts_supporting_s0",
    "n_strong_supporting_s0",
}


def classify_feature(row: dict) -> str:
    if not row["available"]:
        return "UNAVAILABLE"
    feat = row.get("feature")
    if feat in DEFINITIONAL_ON_P:
        row["stability"] = "DEFINITIONALLY_FALSE_ON_P"
        row["notes"] = (
            (row.get("notes") or "")
            + " GROUP P requires S0 to disagree with both currently wrong strong experts; "
            "this feature is not an independent reliability cue."
        ).strip()
        return "WEAK"
    o02_n1 = oriented_auc(row.get("auc_p_vs_n1_folds_0_2"))
    o34_n1 = oriented_auc(row.get("auc_p_vs_n1_folds_3_4"))
    o02_n3 = oriented_auc(row.get("auc_p_vs_n3_folds_0_2"))
    o34_n3 = oriented_auc(row.get("auc_p_vs_n3_folds_3_4"))
    dir02 = row.get("folds_0_2_direction") or row.get("direction_folds_0_2")
    dir34 = row.get("folds_3_4_direction") or row.get("direction_folds_3_4")
    stable = bool(dir02 and dir34 and dir02 == dir34 and dir02 != "tied")
    row["stability"] = "STABLE" if stable else "UNSTABLE_OR_TIED"
    if (
        o02_n1 is not None
        and o02_n1 >= 0.70
        and o02_n3 is not None
        and o02_n3 >= 0.70
        and stable
        and o34_n1 is not None
        and o34_n1 >= 0.60
        and o34_n3 is not None
        and o34_n3 >= 0.60
    ):
        return "STRONG"
    if stable and (
        (o02_n3 is not None and o02_n3 >= 0.70)
        or (o02_n1 is not None and o02_n1 >= 0.65)
    ):
        return "MODERATE"
    return "WEAK"


def decide_experiment(metrics: dict) -> dict:
    inventory = metrics["feature_inventory"]
    strong = [r for r in inventory if r["classification"] == "STRONG"]
    moderate = [r for r in inventory if r["classification"] == "MODERATE"]
    families = metrics["family_rescue_map"]
    overall_rate = metrics["integrity"]["positive_rescues"] / metrics["integrity"]["both_wrong"]
    enriched = []
    for row in families:
        rate = row["rescue_rate_among_shared_errors"]
        if rate is None:
            continue
        if row["positive_rescues"] >= 15 and rate >= 1.5 * overall_rate:
            enriched.append(row)
    fam_02 = metrics["family_rescue_map_folds_0_2"]
    fam_34 = metrics["family_rescue_map_folds_3_4"]
    top_02 = max(fam_02, key=lambda r: r["positive_rescues"])["family"] if fam_02 else None
    top_34 = max(fam_34, key=lambda r: r["positive_rescues"])["family"] if fam_34 else None
    family_stable = bool(top_02 and top_34 and top_02 == top_34)
    composite = next((r for r in inventory if r["feature"] == "rescue_evidence_score"), None)
    top_decile = metrics["quantile_diagnostics"].get("rescue_evidence_score", {}).get("deciles", [])
    top = top_decile[-1] if top_decile else None
    high_precision = False
    if top and top.get("n"):
        rescue_rate = top.get("rescue_rate") or 0.0
        danger_rate = top.get("danger_rate_n3") or 0.0
        base_rescue = metrics["integrity"]["positive_rescues"] / N_TRAIN
        high_precision = rescue_rate >= 3.0 * base_rescue and rescue_rate > danger_rate
    high_precision_stable = bool(
        metrics["quantile_stability"]["rescue_evidence_score_top_decile_same_direction"]
    )
    if (
        len(strong) >= 2
        and high_precision
        and high_precision_stable
        and metrics["integrity"]["positive_rescues"] >= 35
    ):
        label = "RESCUE SIGNAL STRONG"
        next_exp = "V3-E03B — Cross-Fitted Abstaining Rescue Gate"
        reason = (
            "{} test-safe features are STRONG with fold-stable direction, and the "
            "predeclared composite top decile shows a high-precision regime that "
            "remains directionally stable on folds 3-4.".format(len(strong))
        )
    elif enriched and family_stable and len(strong) < 2:
        label = "RESCUE SIGNAL FAMILY-SPECIFIC"
        next_exp = "V3-E03F — Family-Aware Abstaining Rescue"
        reason = (
            "Global reliability separation is not STRONG, but positive rescues are "
            "enriched in {} and that family remains the leading rescue family on "
            "folds 3-4.".format(", ".join(r["family"] for r in enriched))
        )
    else:
        label = "RESCUE SIGNAL WEAK"
        next_exp = (
            "Create another independent expert rather than a complex S0 rescue router. "
            "Do not start V3-E03B or V3-E03F on the current S0 reliability evidence."
        )
        reason = (
            "S0 adds unique oracle headroom (96 cells; diagnostic three-expert oracle "
            "0.8622), but available test-safe signals do not isolate those rescues "
            "from dangerous overrides with fold-stable high precision."
        )
    return {
        "label": label,
        "reason": reason,
        "next_experiment": next_exp,
        "n_strong_features": len(strong),
        "n_moderate_features": len(moderate),
        "enriched_families": [r["family"] for r in enriched],
        "top_family_folds_0_2": top_02,
        "top_family_folds_3_4": top_34,
        "high_precision_regime_predeclared_composite_top_decile": high_precision,
        "high_precision_stable_folds_3_4": high_precision_stable,
    }


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return "{:.4f}".format(v)
        return str(v)

    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def render_report(metrics: dict) -> str:
    integ = metrics["integrity"]
    s0p = metrics["s0_probability"]
    p = metrics["positive_rescues"]
    dang = metrics["dangerous_overrides"]
    agr = metrics["agreement_states"]
    hb = metrics["hard_bucket"]
    fam = metrics["family_rescue_map"]
    conf = metrics["confusion_pairs"]
    inv = metrics["feature_inventory"]
    feat = metrics["feature_diagnostics"]
    quant = metrics["quantile_diagnostics"]
    folds = metrics["fold_stability"]
    ceil = metrics["ceilings"]
    leak = metrics["leakage_audit"]
    decision = metrics["final_classification"]
    sep = metrics["separability_summary"]

    def acc(correct: int) -> str:
        return "{} / 5000 = {:.4f}".format(correct, correct / 5000)

    s0_prob_text = (
        "S0 60-class probabilities were recovered by deterministic CPU inference "
        "from frozen `{}` (SHA256 `{}`). Frozen S0 hard labels matched on 5000 / 5000 cells. "
        "No retraining occurred.".format(s0p["checkpoint_path"], s0p["checkpoint_sha256"])
        if s0p["available"]
        else "S0 probability features are UNAVAILABLE: {}.".format(s0p.get("reason"))
    )

    family_rows = [
        [
            r["family"],
            r["population"],
            r["shared_errors"],
            r["positive_rescues"],
            r["rescue_rate_among_shared_errors"],
            r["n3"],
        ]
        for r in fam
    ]
    top_lzh = conf["lzh_pairs"][:8]
    top_wyh = conf["wyh_pairs"][:8]
    strong_feats = [r for r in inv if r["classification"] in {"STRONG", "MODERATE"}]
    auc_rows = []
    for r in feat:
        auc_rows.append(
            [
                r["feature"],
                r.get("p_mean"),
                r.get("n1_mean"),
                r.get("n3_mean"),
                r.get("auc_p_vs_n1"),
                r.get("auc_p_vs_n3"),
            ]
        )

    q_blocks = []
    for name in QUANTILE_FEATURES + ["rescue_evidence_score"]:
        block = quant.get(name)
        if not block:
            continue
        rows = [
            [
                d["decile"],
                d["n"],
                d["positive_rescue_count"],
                d["rescue_rate"],
                d["n3"],
                d["danger_rate_n3"],
            ]
            for d in block["deciles"]
        ]
        q_blocks.append(
            "### {}\n\n".format(name)
            + _md_table(
                ["decile", "n", "rescues", "rescue rate", "N3", "N3 rate"],
                rows,
            )
        )

    inv_rows = [
        [
            r["feature"],
            r["source"],
            r["test_safe"],
            r["available"],
            r["diagnostic_AUROC_P_vs_N1"],
            r["diagnostic_AUROC_P_vs_N3"],
            r["folds_0_2_direction"],
            r["folds_3_4_direction"],
            r["stability"],
            r["classification"],
        ]
        for r in inv
    ]

    return """# V3-E03A — Weak-but-Diverse Expert Rescue Audit

Research codename: **CER-AUDIT**. This is a MODEL V3 research experiment, not a formal MODEL V3 freeze.

## 1. Research Question

Can test-safe reliability, confidence, disagreement, biological-family, and metadata signals identify the small subset of cells where the weak-but-diverse S0 expert should be trusted, while abstaining everywhere else?

S0 is not treated as a globally competitive classifier. It is a **WEAK-BUT-DIVERSE RESCUE EXPERT**. This audit does not implement a deployment gate.

Diagnostic oracle accuracy is **not** deployable model accuracy.

## 2. Frozen Starting Evidence

| Quantity | Value |
|---|---|
| LZH Prior-H | {lzh} |
| WYH MODEL V2 | {wyh} |
| LZH + WYH diagnostic oracle | {pair} |
| Both LZH and WYH wrong | {both} |
| S0 hard-label 200-gene reference MLP | {s0} |
| S0 unique recoveries | {rescues} |
| LZH + WYH + S0 diagnostic oracle | {three} |
| All three wrong | {all_wrong} |

S0 unique recoveries by canonical fold: 25 / 28 / 16 / 17 / 10. Folds 0-2: 69. Locked folds 3-4: 27.

`0.8622` is a diagnostic oracle, **NOT** deployable accuracy.

V3-E00T and V3-E02D artifacts were read only. Distillation was not continued. S0 was not retrained.

## 3. Why S0 Is Not a Global Classifier Candidate

S0 standalone accuracy is **0.6302**, far below LZH (0.8266) and WYH MODEL V2 (0.8212). Unconstrained replacement of either strong expert by S0 would destroy accuracy.

S0's project value is complementarity: it is uniquely correct on **96** cells that both strong experts miss. The research question is whether those 96 cells can be recognized with test-safe signals.

## 4. Integrity Reproduction

All predeclared counts reproduced exactly from the joined three-expert registry:

| Check | Expected | Reproduced |
|---|---:|---:|
| LZH correct | 4133 | {lzh_n} |
| WYH correct | 4106 | {wyh_n} |
| S0 correct | 3151 | {s0_n} |
| LZH + WYH oracle | 4215 | {pair_n} |
| both strong wrong | 785 | {both_n} |
| positive S0 rescues | 96 | {rescues_n} |
| three-expert oracle | 4311 | {three_n} |
| all three wrong | 689 | {all_wrong_n} |
| fold rescues | 25/28/16/17/10 | {fold_rescues} |

Registry rows: **5000**. Unique 19-digit `Cell_ID` strings: **5000**. Labels aligned to official `meta_train`.

{s0_prob_text}

## 5. The 96 Unique S0 Rescues

GROUP P is defined as LZH wrong AND WYH wrong AND S0 correct. Count: **96**.

{p_class_table}

Hard-bucket fraction of the 96: **{p_hard_frac:.4f}** ({p_hard} / 96).

Canonical-fold counts: {fold_rescues}.

## 6. Dangerous Override Populations

| Group | Definition | Count |
|---|---|---:|
| N1 | LZH wrong AND WYH wrong AND S0 wrong | {n1} |
| N2 | LZH correct AND WYH correct AND S0 wrong | {n2} |
| N3 | at least one of LZH/WYH correct AND S0 wrong | {n3} |

N1 is the failed-rescue-opportunity set. N2 is the strongest dangerous-override set. N3 is the full dangerous-override set for any future S0 replacement of a currently correct strong expert.

S0 wrong total = N1 + N3 = {s0_wrong}.

## 7. LZH / WYH Agreement-State Analysis

{agree_table}

When LZH == WYH AND both are wrong: S0 is correct on **{agree_both_wrong_s0}** / **{agree_both_wrong}** cells ({agree_both_wrong_rate}).

When LZH != WYH: S0 is correct on **{disagree_s0_correct}** / **{disagree_n}** cells ({disagree_s0_rate}). Positive S0 rescues in this state: **{disagree_rescues}**.

When LZH != WYH: at least one strong expert is already correct on **{disagree_strong_any}** / **{disagree_n}** cells ({disagree_strong_rate}).

Interpretation: {agree_interp}

## 8. Hard-Bucket Analysis

{hard_table}

Fraction of the 96 positive rescues in the hard bucket: **{p_hard_frac:.4f}**.

## 9. Biological Family Rescue Map

E03A family mapping keeps the frozen E02D oligo/OPC, astrocyte, vascular, and meningeal definitions, then splits `microglia` out of the remaining glial/non-neuronal group.

E02D-mapping reproduction of S0 unique recoveries: oligodendrocyte_opc **{e02d_oligo}**, vascular **{e02d_vasc}**.

{family_table}

## 10. Confusion-Pair Rescue Map

Among the 96 rescues, LZH_pred == WYH_pred (both wrong, S0 correct): **{agree_wrong_rescue}**.

LZH_pred != WYH_pred and S0 correct: **{disagree_rescue}**.

Top true → LZH_pred pairs among GROUP P:

{lzh_pair_table}

Top true → WYH_pred pairs among GROUP P:

{wyh_pair_table}

Interpretation: {confusion_interp}

## 11. Test-Safe Reliability Features

Candidate test-safe features were restricted to quantities observable without held-out labels: expert confidence, predicted-class agreement, probability advantage, JS divergence, library size, detected-gene count, hard-bucket / metadata missingness, and the single predeclared composite margin score.

True labels, correctness flags, and rescue flags are **DIAGNOSTIC-ONLY**. They are not future deployment features.

LZH Prior-H eligibility, graph degree, and reliable-gene count remain **UNAVAILABLE** at cell level, as in V3-E00T.

{s0_prob_text}

## 12. Diagnostic Separability

AUROC values below are diagnostic only. No classifier was fit and no threshold was chosen.

{auc_table}

Summary: {sep_text}

## 13. Quantile Diagnostics

All 5000 cells were split into fixed feature-value deciles. Deciles were not optimized and are not candidate rules.

{quantile_blocks}

## 14. Canonical Fold Stability

Development analysis uses canonical folds 0-2 (69 rescues). Locked confirmation uses folds 3-4 (27 rescues). Nothing was tuned on folds 3-4.

{fold_table}

Direction stability of the strongest available scalar signals: {fold_dir_text}

## 15. Rescue Opportunity Ceilings

These are oracle-style diagnostic ceilings, **not** deployable scores.

{ceiling_table}

Restriction J is not a count. LZH/WYH confidence among GROUP P versus the 5000-cell population:

{j_text}

## 16. What a Future Gate Should and Should Not Use

**Should consider only if a later experiment is justified:** {should_use}

**Should not use:** true labels; correctness / rescue flags; any threshold chosen on full OOF; blend-weight search; S0 as a global replacement; LZH Prior-H eligibility / graph degree / reliable-gene count until cell-level OOF fields exist; folds 3-4 for redesign.

## 17. Leakage Audit

- competition test labels used: {leak_test}
- true labels used only for retrospective diagnostics: {leak_diag}
- learned gate trained: {leak_gate}
- threshold optimized: {leak_thr}
- weights optimized: {leak_w}
- leaderboard feedback used: {leak_lb}
- S0 retrained: {leak_retrain}
- folds 3-4 used for redesign: {leak_34}
- submission generated: {leak_sub}
- V3-E00T / V3-E02D numerical artifacts modified: {leak_frozen}
- prediction/prediction.csv modified: {leak_pred}

## 18. Limitations

- S0 is an external-reference MLP, not competition OOF; its errors are not fold-exchangeable with LZH/WYH in the same protocol
- LZH uses a native 3-fold protocol; canonical seed42 folds are a meta-analysis partition
- YHH and current team-main remain unavailable for honest cell-level OOF
- Diagnostic AUROC on 96 vs 689 or 96 vs N3 is a small-N characterization
- A diagnostic oracle of 0.8622 is not a model score
- No spatial, graph-degree, or Prior-H eligibility cell-level fields were available

## 19. Decision

**{label}**

{reason}

Recommended next experiment: **{next_exp}**

Do not start that experiment in this task. MODEL V3 is not frozen. No `docs/versions/model_v3.md` and no submission candidate were created.
""".format(
        lzh=acc(LZH_CORRECT),
        wyh=acc(WYH_CORRECT),
        pair=acc(LZH_WYH_ORACLE),
        both=BOTH_WRONG,
        s0=acc(S0_CORRECT),
        rescues=POSITIVE_RESCUES,
        three=acc(THREE_EXPERT_ORACLE),
        all_wrong=ALL_THREE_WRONG,
        lzh_n=integ["lzh_correct"],
        wyh_n=integ["wyh_correct"],
        s0_n=integ["s0_correct"],
        pair_n=integ["lzh_wyh_oracle"],
        both_n=integ["both_wrong"],
        rescues_n=integ["positive_rescues"],
        three_n=integ["three_expert_oracle"],
        all_wrong_n=integ["all_three_wrong"],
        fold_rescues="/".join(str(FOLD_RESCUES[i]) for i in range(5)),
        s0_prob_text=s0_prob_text,
        p_class_table=_md_table(
            ["true class", "n", "family"],
            [[r["true_label"], r["n"], r.get("true_family", r.get("family"))] for r in p["by_true_class"][:15]],
        ),
        p_hard=p["hard_bucket_count"],
        p_hard_frac=p["hard_bucket_fraction"],
        n1=dang["n1"],
        n2=dang["n2"],
        n3=dang["n3"],
        s0_wrong=dang["s0_wrong"],
        agree_table=_md_table(
            [
                "state",
                "n",
                "LZH acc",
                "WYH acc",
                "S0 acc",
                "P count",
                "P rate",
                "N3",
                "oracle",
            ],
            [
                [
                    "A: LZH == WYH",
                    agr["state_A"]["n"],
                    agr["state_A"]["lzh_accuracy"],
                    agr["state_A"]["wyh_accuracy"],
                    agr["state_A"]["s0_accuracy"],
                    agr["state_A"]["positive_rescue_count"],
                    agr["state_A"]["positive_rescue_rate"],
                    agr["state_A"]["n3"],
                    agr["state_A"]["three_expert_oracle_accuracy"],
                ],
                [
                    "B: LZH != WYH",
                    agr["state_B"]["n"],
                    agr["state_B"]["lzh_accuracy"],
                    agr["state_B"]["wyh_accuracy"],
                    agr["state_B"]["s0_accuracy"],
                    agr["state_B"]["positive_rescue_count"],
                    agr["state_B"]["positive_rescue_rate"],
                    agr["state_B"]["n3"],
                    agr["state_B"]["three_expert_oracle_accuracy"],
                ],
            ],
        ),
        agree_both_wrong=agr["when_agree_and_both_wrong"]["n"],
        agree_both_wrong_s0=agr["when_agree_and_both_wrong"]["s0_correct"],
        agree_both_wrong_rate="{:.4f}".format(agr["when_agree_and_both_wrong"]["s0_accuracy"] or 0.0),
        disagree_n=agr["state_B"]["n"],
        disagree_s0_correct=agr["when_disagree"]["s0_correct"],
        disagree_s0_rate="{:.4f}".format(agr["when_disagree"]["s0_accuracy"] or 0.0),
        disagree_rescues=agr["state_B"]["positive_rescue_count"],
        disagree_strong_any=agr["when_disagree"]["strong_any_correct"],
        disagree_strong_rate="{:.4f}".format(agr["when_disagree"]["strong_any_correct_rate"] or 0.0),
        agree_interp=sep["agreement_interpretation"],
        hard_table=_md_table(
            ["hard_bucket", "n", "LZH acc", "WYH acc", "S0 acc", "P count", "P rate", "N3"],
            [
                [
                    str(k),
                    hb[k]["n"],
                    hb[k]["lzh_accuracy"],
                    hb[k]["wyh_accuracy"],
                    hb[k]["s0_accuracy"],
                    hb[k]["positive_rescue_count"],
                    hb[k]["positive_rescue_rate"],
                    hb[k]["n3"],
                ]
                for k in ["True", "False"]
            ],
        ),
        e02d_oligo=p["e02d_family_counts"].get("oligodendrocyte_opc", 0),
        e02d_vasc=p["e02d_family_counts"].get("vascular", 0),
        family_table=_md_table(
            ["family", "n", "shared errors", "S0 rescues", "rescue rate", "N3"],
            family_rows,
        ),
        agree_wrong_rescue=conf["lzh_wyh_agree_both_wrong_s0_correct"],
        disagree_rescue=conf["lzh_wyh_disagree_s0_correct"],
        lzh_pair_table=_md_table(
            ["true", "LZH pred", "n"],
            [[r["true_label"], r["pred"], r["n"]] for r in top_lzh],
        ),
        wyh_pair_table=_md_table(
            ["true", "WYH pred", "n"],
            [[r["true_label"], r["pred"], r["n"]] for r in top_wyh],
        ),
        confusion_interp=sep["confusion_interpretation"],
        auc_table=_md_table(
            ["feature", "P mean", "N1 mean", "N3 mean", "AUROC P vs N1", "AUROC P vs N3"],
            auc_rows,
        ),
        sep_text=sep["global_text"],
        quantile_blocks="\n\n".join(q_blocks),
        fold_table=_md_table(
            ["split", "n", "P", "N1", "N3", "S0 acc", "oracle"],
            [
                [
                    "folds 0-2",
                    folds["folds_0_2"]["n"],
                    folds["folds_0_2"]["positive_rescue_count"],
                    folds["folds_0_2"]["n1"],
                    folds["folds_0_2"]["n3"],
                    folds["folds_0_2"]["s0_accuracy"],
                    folds["folds_0_2"]["three_expert_oracle_accuracy"],
                ],
                [
                    "folds 3-4",
                    folds["folds_3_4"]["n"],
                    folds["folds_3_4"]["positive_rescue_count"],
                    folds["folds_3_4"]["n1"],
                    folds["folds_3_4"]["n3"],
                    folds["folds_3_4"]["s0_accuracy"],
                    folds["folds_3_4"]["three_expert_oracle_accuracy"],
                ],
            ],
        ),
        fold_dir_text=sep["fold_stability_text"],
        ceiling_table=_md_table(
            ["restriction", "unique recoveries", "note"],
            [[r["id"], r["count"], r["note"]] for r in ceil["count_restrictions"]],
        ),
        j_text=ceil["j_text"],
        should_use=sep["future_gate_should_use"],
        leak_test=leak["competition_test_labels_used"],
        leak_diag=leak["true_labels_used_only_for_retrospective_diagnostics"],
        leak_gate=leak["learned_gate_trained"],
        leak_thr=leak["threshold_optimized"],
        leak_w=leak["weights_optimized"],
        leak_lb=leak["leaderboard_feedback_used"],
        leak_retrain=leak["s0_retrained"],
        leak_34=leak["canonical_folds_3_4_used_for_redesign"],
        leak_sub=leak["submission_generated"],
        leak_frozen=leak["v3_e00t_or_e02d_numerical_artifacts_modified"],
        leak_pred=leak["prediction_csv_modified"],
        label=decision["label"],
        reason=decision["reason"],
        next_exp=decision["next_experiment"],
    )


def family_map_table(df: pd.DataFrame, mask: Optional[np.ndarray] = None) -> List[dict]:
    if mask is None:
        sub = df
    else:
        sub = df.loc[mask]
    rows = []
    families = list(CLASS_FAMILIES_E03A.keys()) + ["neuronal_or_other"]
    for family in families:
        fam_mask = sub["true_family"] == family
        shared = fam_mask & sub["lzh_wyh_both_wrong"]
        rescues = fam_mask & sub["positive_s0_rescue"]
        n_shared = int(shared.sum())
        n_rescue = int(rescues.sum())
        rows.append(
            {
                "family": family,
                "population": int(fam_mask.sum()),
                "shared_errors": n_shared,
                "positive_rescues": n_rescue,
                "rescue_rate_among_shared_errors": (float(n_rescue / n_shared) if n_shared else None),
                "n1": int((fam_mask & sub["group_n1"]).sum()),
                "n2": int((fam_mask & sub["group_n2"]).sum()),
                "n3": int((fam_mask & sub["group_n3"]).sum()),
            }
        )
    return rows


def main() -> int:
    if current_branch() != EXPECTED_BRANCH:
        raise SystemExit("Refusing to run: current branch is not {}".format(EXPECTED_BRANCH))
    head = run_git(["rev-parse", "--short=7", "HEAD"]).strip()
    if not head.startswith(EXPECTED_HEAD_PREFIX[:7]) and head != EXPECTED_HEAD_PREFIX:
        # Allow running after later commits, but warn in metrics. The freeze is the input artifacts.
        head_note = "HEAD is {}, frozen E02D commit was {}".format(head, EXPECTED_HEAD_PREFIX)
    else:
        head_note = "HEAD matches frozen E02D checkpoint {}".format(EXPECTED_HEAD_PREFIX)

    pred_before = sha256_file(PRED_PATH) if PRED_PATH.is_file() else None
    e00t_before = sha256_file(E00T_REGISTRY)
    s0_val_before = sha256_file(S0_VALIDATION)

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

    e00t = pd.read_parquet(E00T_REGISTRY)
    e00t["Cell_ID"] = e00t["Cell_ID"].astype(str)
    if len(e00t) != N_TRAIN or e00t["Cell_ID"].nunique() != N_TRAIN:
        integrity_failure("E00T registry is not 5000 unique Cell_IDs")
    e00t = e00t.set_index("Cell_ID").reindex(train_ids)
    if e00t.isna().any().any() and e00t["true_label"].isna().any():
        integrity_failure("E00T registry missing canonical Cell_IDs")

    s0_val = read_cell_id_csv(S0_VALIDATION).set_index("Cell_ID").reindex(train_ids)
    if s0_val["predicted_label"].isna().any():
        integrity_failure("S0 validation missing canonical Cell_IDs")
    if list(s0_val["true_label"].astype(str)) != list(y_true_official.reindex(train_ids).astype(str)):
        integrity_failure("S0 validation labels do not match official meta_train")
    if list(e00t["true_label"].astype(str)) != list(y_true_official.reindex(train_ids).astype(str)):
        integrity_failure("E00T labels do not match official meta_train")

    y_true = y_true_official.reindex(train_ids).astype(str).to_numpy()
    lzh_pred = e00t["lzh_prior_h_pred"].astype(str).to_numpy()
    wyh_pred = e00t["wyh_model_v2_pred"].astype(str).to_numpy()
    s0_pred = s0_val["predicted_label"].astype(str).to_numpy()
    canonical_fold = e00t["canonical_fold"].astype(int).to_numpy()

    lzh_ok = lzh_pred == y_true
    wyh_ok = wyh_pred == y_true
    s0_ok = s0_pred == y_true
    both_wrong = (~lzh_ok) & (~wyh_ok)
    positive = both_wrong & s0_ok
    n1 = both_wrong & (~s0_ok)
    n2 = lzh_ok & wyh_ok & (~s0_ok)
    n3 = (lzh_ok | wyh_ok) & (~s0_ok)
    three_ok = lzh_ok | wyh_ok | s0_ok

    if int(lzh_ok.sum()) != LZH_CORRECT:
        integrity_failure("LZH correct {} != 4133".format(int(lzh_ok.sum())))
    if int(wyh_ok.sum()) != WYH_CORRECT:
        integrity_failure("WYH correct {} != 4106".format(int(wyh_ok.sum())))
    if int(s0_ok.sum()) != S0_CORRECT:
        integrity_failure("S0 correct {} != 3151".format(int(s0_ok.sum())))
    if int((lzh_ok | wyh_ok).sum()) != LZH_WYH_ORACLE:
        integrity_failure("LZH+WYH oracle {} != 4215".format(int((lzh_ok | wyh_ok).sum())))
    if int(both_wrong.sum()) != BOTH_WRONG:
        integrity_failure("both-wrong {} != 785".format(int(both_wrong.sum())))
    if int(positive.sum()) != POSITIVE_RESCUES:
        integrity_failure("positive S0 rescues {} != 96".format(int(positive.sum())))
    if int(three_ok.sum()) != THREE_EXPERT_ORACLE:
        integrity_failure("three-expert oracle {} != 4311".format(int(three_ok.sum())))
    if int((~three_ok).sum()) != ALL_THREE_WRONG:
        integrity_failure("all-three-wrong {} != 689".format(int((~three_ok).sum())))
    fold_counts = {int(k): int(v) for k, v in pd.Series(canonical_fold[positive]).value_counts().sort_index().items()}
    if fold_counts != FOLD_RESCUES:
        integrity_failure("fold rescues {} != {}".format(fold_counts, FOLD_RESCUES))

    x_log1p = np.log1p(data.counts_train.loc[train_ids].to_numpy(dtype=np.float32))
    if x_log1p.shape != (N_TRAIN, 200):
        integrity_failure("competition train feature shape {}".format(x_log1p.shape))
    s0_inf = infer_s0_probabilities(train_ids, class_names, x_log1p, s0_pred)
    s0_proba_available = bool(s0_inf.get("available"))
    if s0_proba_available:
        s0_conf = confidence_from_proba(s0_inf["proba"])
        write_proba(TABLE_DIR / "s0_inference_probabilities.csv.gz", train_ids, s0_inf["proba"], class_names)
    else:
        s0_conf = {
            "top1": np.full(N_TRAIN, np.nan),
            "top2": np.full(N_TRAIN, np.nan),
            "margin": np.full(N_TRAIN, np.nan),
            "entropy": np.full(N_TRAIN, np.nan),
        }

    wyh_frame, _wyh_audit = load_probability_frame(WYH_V2_OOF_PROBA, class_names)
    wyh_frame = wyh_frame.set_index("Cell_ID").reindex(train_ids)
    wyh_proba = wyh_frame.loc[:, class_names].to_numpy(dtype=np.float64)
    if list(argmax_labels(wyh_proba, class_names)) != list(wyh_pred):
        integrity_failure("WYH probability argmax does not match E00T WYH predictions")

    with tempfile.TemporaryDirectory(prefix="v3_e03a_lzh_") as tmp:
        lzh_path = Path(tmp) / "lzh_oof_probabilities_final.csv"
        materialize_git_blob("{}:{}".format(LZH_BRANCH, LZH_FINAL_OOF), lzh_path)
        lzh_frame, _lzh_audit = load_probability_frame(lzh_path, class_names)
    lzh_frame = lzh_frame.set_index("Cell_ID").reindex(train_ids)
    lzh_proba = lzh_frame.loc[:, class_names].to_numpy(dtype=np.float64)
    if list(argmax_labels(lzh_proba, class_names)) != list(lzh_pred):
        integrity_failure("LZH probability argmax does not match E00T LZH predictions")

    if s0_proba_available:
        js_s0_lzh = js_divergence_rows(s0_inf["proba"], lzh_proba)
        js_s0_wyh = js_divergence_rows(s0_inf["proba"], wyh_proba)
    else:
        js_s0_lzh = np.full(N_TRAIN, np.nan)
        js_s0_wyh = np.full(N_TRAIN, np.nan)
    js_lzh_wyh = e00t["js_divergence_lzh_wyh"].to_numpy(dtype=np.float64)

    lzh_s0_agree = lzh_pred == s0_pred
    wyh_s0_agree = wyh_pred == s0_pred
    lzh_wyh_agree = lzh_pred == wyh_pred
    all_three_agree = lzh_s0_agree & wyh_s0_agree
    all_three_different = (lzh_pred != wyh_pred) & (lzh_pred != s0_pred) & (wyh_pred != s0_pred)
    n_support_s0 = lzh_s0_agree.astype(int) + wyh_s0_agree.astype(int) + np.ones(N_TRAIN, dtype=int)
    n_strong_support_s0 = lzh_s0_agree.astype(int) + wyh_s0_agree.astype(int)

    if s0_proba_available:
        score = rescue_evidence_score(s0_conf["margin"], e00t["lzh_prior_h_margin"].to_numpy(), e00t["wyh_model_v2_margin"].to_numpy())
        s0_adv_lzh = s0_conf["top1"] - e00t["lzh_prior_h_top1"].to_numpy()
        s0_adv_wyh = s0_conf["top1"] - e00t["wyh_model_v2_top1"].to_numpy()
    else:
        score = np.full(N_TRAIN, np.nan)
        s0_adv_lzh = np.full(N_TRAIN, np.nan)
        s0_adv_wyh = np.full(N_TRAIN, np.nan)

    registry = pd.DataFrame(
        {
            "Cell_ID": train_ids,
            "true_label": y_true,
            "canonical_fold": canonical_fold,
            "lzh_pred": lzh_pred,
            "lzh_correct": lzh_ok,
            "lzh_top1": e00t["lzh_prior_h_top1"].to_numpy(dtype=np.float64),
            "lzh_top2": e00t["lzh_prior_h_top2"].to_numpy(dtype=np.float64),
            "lzh_margin": e00t["lzh_prior_h_margin"].to_numpy(dtype=np.float64),
            "lzh_entropy": e00t["lzh_prior_h_entropy"].to_numpy(dtype=np.float64),
            "wyh_pred": wyh_pred,
            "wyh_correct": wyh_ok,
            "wyh_top1": e00t["wyh_model_v2_top1"].to_numpy(dtype=np.float64),
            "wyh_top2": e00t["wyh_model_v2_top2"].to_numpy(dtype=np.float64),
            "wyh_margin": e00t["wyh_model_v2_margin"].to_numpy(dtype=np.float64),
            "wyh_entropy": e00t["wyh_model_v2_entropy"].to_numpy(dtype=np.float64),
            "s0_pred": s0_pred,
            "s0_correct": s0_ok,
            "s0_top1": s0_conf["top1"],
            "s0_top2": s0_conf["top2"],
            "s0_margin": s0_conf["margin"],
            "s0_entropy": s0_conf["entropy"],
            "Region": e00t["Region"].astype(str).to_numpy(),
            "E/I": e00t["E/I"].astype(str).to_numpy(),
            "Segment": e00t["Segment"].astype(str).to_numpy(),
            "Section_ID": e00t["Section_ID"].astype(str).to_numpy(),
            "hard_bucket": e00t["hard_bucket"].astype(bool).to_numpy(),
            "n_detected": e00t["n_detected"].to_numpy(),
            "library_size": e00t["library_size"].to_numpy(dtype=np.float64),
            "neuron_or_glial": e00t["neuron_or_glial"].astype(str).to_numpy(),
            "lzh_wyh_agree": lzh_wyh_agree,
            "lzh_s0_agree": lzh_s0_agree,
            "wyh_s0_agree": wyh_s0_agree,
            "all_three_agree": all_three_agree,
            "all_three_different": all_three_different,
            "n_experts_supporting_s0": n_support_s0,
            "n_strong_supporting_s0": n_strong_support_s0,
            "s0_prob_advantage_lzh": s0_adv_lzh,
            "s0_prob_advantage_wyh": s0_adv_wyh,
            "js_s0_lzh": js_s0_lzh,
            "js_s0_wyh": js_s0_wyh,
            "js_lzh_wyh": js_lzh_wyh,
            "rescue_evidence_score": score,
            "lzh_wyh_both_wrong": both_wrong,
            "positive_s0_rescue": positive,
            "group_n1": n1,
            "group_n2": n2,
            "group_n3": n3,
            "strong_any_correct_s0_wrong": n3,
            "strong_both_correct_s0_wrong": n2,
            "three_expert_oracle_correct": three_ok,
            "true_family": [family_of(v) for v in y_true],
            "true_family_e02d": [family_of_e02d(v) for v in y_true],
            "diagnostic_only_note": "true_label/correctness/rescue flags are diagnostic-only",
        }
    )
    registry.to_parquet(OUT_DIR / "v3_e03a_three_expert_registry.parquet", index=False)

    rescue_cols = [
        "Cell_ID",
        "true_label",
        "canonical_fold",
        "lzh_pred",
        "wyh_pred",
        "s0_pred",
        "lzh_top1",
        "lzh_margin",
        "lzh_entropy",
        "wyh_top1",
        "wyh_margin",
        "wyh_entropy",
        "s0_top1",
        "s0_margin",
        "s0_entropy",
        "Region",
        "E/I",
        "Segment",
        "Section_ID",
        "hard_bucket",
        "n_detected",
        "library_size",
        "neuron_or_glial",
        "true_family",
        "lzh_wyh_agree",
        "lzh_s0_agree",
        "wyh_s0_agree",
        "all_three_agree",
        "all_three_different",
        "n_experts_supporting_s0",
        "rescue_evidence_score",
        "js_s0_lzh",
        "js_s0_wyh",
        "js_lzh_wyh",
    ]
    registry.loc[positive, rescue_cols].to_csv(OUT_DIR / "v3_e03a_positive_rescues.csv", index=False)
    danger_mask = n1 | n2 | n3
    danger_cols = rescue_cols + ["group_n1", "group_n2", "group_n3", "lzh_correct", "wyh_correct", "s0_correct"]
    registry.loc[danger_mask, danger_cols].to_csv(OUT_DIR / "v3_e03a_dangerous_overrides.csv", index=False)

    p_df = registry.loc[positive]
    by_class = (
        p_df.groupby(["true_label", "true_family"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    by_section = p_df["Section_ID"].value_counts().head(15)
    e02d_family_counts = Counter(p_df["true_family_e02d"].tolist())

    agree_both_wrong = lzh_wyh_agree & both_wrong
    state_a = mask_metrics(registry, lzh_wyh_agree)
    state_b = mask_metrics(registry, ~lzh_wyh_agree)
    agreement = {
        "state_A": state_a,
        "state_B": state_b,
        "when_agree_and_both_wrong": {
            "n": int(agree_both_wrong.sum()),
            "s0_correct": int((agree_both_wrong & s0_ok).sum()),
            "s0_accuracy": float(s0_ok[agree_both_wrong].mean()) if agree_both_wrong.any() else None,
        },
        "when_disagree": {
            "n": int((~lzh_wyh_agree).sum()),
            "s0_correct": int(((~lzh_wyh_agree) & s0_ok).sum()),
            "s0_accuracy": float(s0_ok[~lzh_wyh_agree].mean()) if (~lzh_wyh_agree).any() else None,
            "strong_any_correct": int(((~lzh_wyh_agree) & (lzh_ok | wyh_ok)).sum()),
            "strong_any_correct_rate": float((lzh_ok | wyh_ok)[~lzh_wyh_agree].mean()) if (~lzh_wyh_agree).any() else None,
        },
    }
    if agree_both_wrong.any() and (agreement["when_agree_and_both_wrong"]["s0_accuracy"] or 0) >= 0.20:
        agree_interp = (
            "Shared-confidence failures still contain a non-trivial S0 rescue rate; "
            "future rescue should not ignore the LZH==WYH and both-wrong regime."
        )
    else:
        agree_interp = (
            "Most S0 unique rescues are not concentrated in strong-expert disagreement. "
            "Future rescue, if any, must handle rare shared-confidence failures rather than "
            "only LZH/WYH disagreement."
        )
    if state_b["positive_rescue_count"] > state_a["positive_rescue_count"]:
        agree_interp = (
            "More positive S0 rescues occur when LZH and WYH disagree than when they agree. "
            "Disagreement is a candidate test-safe cue, but most disagreements already have "
            "a correct strong expert, so unconstrained disagreement override is unsafe."
        )

    hard_true = mask_metrics(registry, registry["hard_bucket"].to_numpy())
    hard_false = mask_metrics(registry, ~registry["hard_bucket"].to_numpy())

    family_all = family_map_table(registry)
    family_02 = family_map_table(registry, canonical_fold <= 2)
    family_34 = family_map_table(registry, canonical_fold >= 3)
    pd.DataFrame(family_all).to_csv(OUT_DIR / "v3_e03a_family_rescue_map.csv", index=False)

    lzh_pairs = confusion_counts(p_df["true_label"], p_df["lzh_pred"])
    wyh_pairs = confusion_counts(p_df["true_label"], p_df["wyh_pred"])
    confusion = {
        "lzh_pairs": lzh_pairs,
        "wyh_pairs": wyh_pairs,
        "lzh_wyh_agree_both_wrong_s0_correct": int((positive & lzh_wyh_agree).sum()),
        "lzh_wyh_disagree_s0_correct": int((positive & ~lzh_wyh_agree).sum()),
    }
    if confusion["lzh_wyh_agree_both_wrong_s0_correct"] >= 0.6 * POSITIVE_RESCUES:
        confusion_interp = (
            "Most unique S0 rescues occur when LZH and WYH make the same wrong prediction. "
            "That favors a global or family-aware rescue of shared-confidence failures over "
            "a pure disagreement router."
        )
    else:
        confusion_interp = (
            "Unique S0 rescues are mixed between shared wrong predictions and strong-expert "
            "disagreement. A single confusion-pair patch would miss a large remainder."
        )

    fold_02 = canonical_fold <= 2
    fold_34 = canonical_fold >= 3
    feature_rows = []
    inventory_rows = []
    for feat in SCALAR_FEATURES:
        values = registry[feat].to_numpy(dtype=np.float64)
        available = bool(np.isfinite(values).any())
        p_stats = group_feature_stats(values, positive) if available else summarize_numeric(np.array([]))
        n1_stats = group_feature_stats(values, n1) if available else summarize_numeric(np.array([]))
        n3_stats = group_feature_stats(values, n3) if available else summarize_numeric(np.array([]))
        auc_n1 = safe_auc(positive[positive | n1], values[positive | n1]) if available else None
        auc_n3 = safe_auc(positive[positive | n3], values[positive | n3]) if available else None
        auc_n1_02 = safe_auc(positive[(positive | n1) & fold_02], values[(positive | n1) & fold_02]) if available else None
        auc_n3_02 = safe_auc(positive[(positive | n3) & fold_02], values[(positive | n3) & fold_02]) if available else None
        auc_n1_34 = safe_auc(positive[(positive | n1) & fold_34], values[(positive | n1) & fold_34]) if available else None
        auc_n3_34 = safe_auc(positive[(positive | n3) & fold_34], values[(positive | n3) & fold_34]) if available else None
        dir_all = direction_from_means(p_stats["mean"], n3_stats["mean"])
        dir_02 = direction_from_means(
            float(np.nanmean(values[positive & fold_02])) if (positive & fold_02).any() else None,
            float(np.nanmean(values[n3 & fold_02])) if (n3 & fold_02).any() else None,
        )
        dir_34 = direction_from_means(
            float(np.nanmean(values[positive & fold_34])) if (positive & fold_34).any() else None,
            float(np.nanmean(values[n3 & fold_34])) if (n3 & fold_34).any() else None,
        )
        row = {
            "feature": feat,
            "available": available,
            "p": p_stats,
            "n1": n1_stats,
            "n3": n3_stats,
            "auc_p_vs_n1": auc_n1,
            "auc_p_vs_n3": auc_n3,
            "auc_p_vs_n1_folds_0_2": auc_n1_02,
            "auc_p_vs_n3_folds_0_2": auc_n3_02,
            "auc_p_vs_n1_folds_3_4": auc_n1_34,
            "auc_p_vs_n3_folds_3_4": auc_n3_34,
            "direction_overall": dir_all,
            "direction_folds_0_2": dir_02,
            "direction_folds_3_4": dir_34,
        }
        feature_rows.append(row)
        source, test_safe = FEATURE_META[feat]
        inv = {
            "feature": feat,
            "source": source,
            "test_safe": test_safe,
            "available": available and s0_proba_available if feat.startswith("s0_") or feat in {
                "s0_prob_advantage_lzh",
                "s0_prob_advantage_wyh",
                "js_s0_lzh",
                "js_s0_wyh",
                "rescue_evidence_score",
            } else available,
            "diagnostic_AUROC_P_vs_N1": auc_n1,
            "diagnostic_AUROC_P_vs_N3": auc_n3,
            "folds_0_2_direction": dir_02,
            "folds_3_4_direction": dir_34,
            "auc_p_vs_n1_folds_0_2": auc_n1_02,
            "auc_p_vs_n3_folds_0_2": auc_n3_02,
            "auc_p_vs_n1_folds_3_4": auc_n1_34,
            "auc_p_vs_n3_folds_3_4": auc_n3_34,
            "notes": "diagnostic AUROC only; no threshold selected",
        }
        inv["classification"] = classify_feature(inv)
        inventory_rows.append(inv)

    for missing_feat, source in [
        ("lzh_prior_h_eligibility", "LZH route audit"),
        ("lzh_graph_degree", "LZH graph"),
        ("lzh_reliable_gene_count", "LZH Prior-H"),
    ]:
        inventory_rows.append(
            {
                "feature": missing_feat,
                "source": source,
                "test_safe": True,
                "available": False,
                "diagnostic_AUROC_P_vs_N1": None,
                "diagnostic_AUROC_P_vs_N3": None,
                "folds_0_2_direction": None,
                "folds_3_4_direction": None,
                "stability": "UNAVAILABLE",
                "classification": "UNAVAILABLE",
                "notes": "documented by LZH but no committed cell-level OOF field",
            }
        )

    feat_csv_rows = []
    for row in feature_rows:
        feat_csv_rows.append(
            {
                "feature": row["feature"],
                "p_count": row["p"]["count"],
                "p_mean": row["p"]["mean"],
                "p_median": row["p"]["median"],
                "p_p25": row["p"]["p25"],
                "p_p75": row["p"]["p75"],
                "n1_count": row["n1"]["count"],
                "n1_mean": row["n1"]["mean"],
                "n1_median": row["n1"]["median"],
                "n1_p25": row["n1"]["p25"],
                "n1_p75": row["n1"]["p75"],
                "n3_count": row["n3"]["count"],
                "n3_mean": row["n3"]["mean"],
                "n3_median": row["n3"]["median"],
                "n3_p25": row["n3"]["p25"],
                "n3_p75": row["n3"]["p75"],
                "auc_p_vs_n1": row["auc_p_vs_n1"],
                "auc_p_vs_n3": row["auc_p_vs_n3"],
                "auc_p_vs_n1_folds_0_2": row["auc_p_vs_n1_folds_0_2"],
                "auc_p_vs_n3_folds_0_2": row["auc_p_vs_n3_folds_0_2"],
                "auc_p_vs_n1_folds_3_4": row["auc_p_vs_n1_folds_3_4"],
                "auc_p_vs_n3_folds_3_4": row["auc_p_vs_n3_folds_3_4"],
                "direction_overall": row["direction_overall"],
                "direction_folds_0_2": row["direction_folds_0_2"],
                "direction_folds_3_4": row["direction_folds_3_4"],
            }
        )
    pd.DataFrame(feat_csv_rows).to_csv(OUT_DIR / "v3_e03a_feature_diagnostics.csv", index=False)
    pd.DataFrame(inventory_rows).to_csv(TABLE_DIR / "feature_inventory.csv", index=False)

    quantile = {}
    q_names = list(QUANTILE_FEATURES)
    if s0_proba_available:
        q_names.append("rescue_evidence_score")
    for name in q_names:
        values = registry[name].to_numpy(dtype=np.float64)
        labs = decile_labels(values)
        deciles = []
        for d in range(10):
            mask = labs == d
            if not np.any(mask):
                continue
            n_mask = int(mask.sum())
            p_count = int((mask & positive).sum())
            n3_count = int((mask & n3).sum())
            deciles.append(
                {
                    "decile": d + 1,
                    "n": n_mask,
                    "positive_rescue_count": p_count,
                    "rescue_rate": float(p_count / n_mask),
                    "n3": n3_count,
                    "danger_rate_n3": float(n3_count / n_mask),
                }
            )
        quantile[name] = {"n_bins": len(deciles), "deciles": deciles}

    def top_decile_enrichment(name: str, split_mask: np.ndarray) -> Optional[str]:
        if name not in quantile:
            return None
        values = registry[name].to_numpy(dtype=np.float64)
        labs = decile_labels(values)
        if not np.isfinite(labs[split_mask]).any():
            return None
        # compare lowest vs highest populated bin on this split
        present = sorted(set(int(v) for v in labs[split_mask] if np.isfinite(v)))
        if len(present) < 2:
            return None
        low = present[0]
        high = present[-1]
        def rate(bin_id):
            m = (labs == bin_id) & split_mask
            if not m.any():
                return None
            return float((m & positive).sum() / m.sum())
        r_low, r_high = rate(low), rate(high)
        if r_low is None or r_high is None:
            return None
        return "higher_decile" if r_high >= r_low else "lower_decile"

    q_stable = {
        "rescue_evidence_score_top_decile_same_direction": (
            top_decile_enrichment("rescue_evidence_score", fold_02)
            == top_decile_enrichment("rescue_evidence_score", fold_34)
            and top_decile_enrichment("rescue_evidence_score", fold_02) is not None
        )
    }

    hard_p = int((positive & registry["hard_bucket"].to_numpy()).sum())
    oligo = registry["true_family"].to_numpy() == "oligodendrocyte_opc"
    astro = registry["true_family"].to_numpy() == "astrocyte"
    vasc = registry["true_family"].to_numpy() == "vascular"
    if s0_proba_available:
        h_top1 = (registry["s0_top1"] > registry["lzh_top1"]) & (registry["s0_top1"] > registry["wyh_top1"])
        i_margin = (registry["s0_margin"] > registry["lzh_margin"]) & (registry["s0_margin"] > registry["wyh_margin"])
        h_count = int((positive & h_top1.to_numpy()).sum())
        i_count = int((positive & i_margin.to_numpy()).sum())
        h_note = "S0 top1 > LZH top1 and S0 top1 > WYH top1"
        i_note = "S0 margin > LZH margin and S0 margin > WYH margin"
    else:
        h_count = None
        i_count = None
        h_note = "UNAVAILABLE: S0 probabilities missing"
        i_note = "UNAVAILABLE: S0 probabilities missing"

    ceilings = {
        "count_restrictions": [
            {"id": "A", "count": int(positive.sum()), "note": "all positive unique S0 rescues"},
            {"id": "B", "count": int((positive & ~lzh_wyh_agree).sum()), "note": "only LZH != WYH"},
            {"id": "C", "count": int((positive & lzh_wyh_agree).sum()), "note": "only LZH == WYH"},
            {"id": "D", "count": hard_p, "note": "only hard-bucket cells"},
            {"id": "E", "count": int((positive & oligo).sum()), "note": "only oligodendrocyte / OPC"},
            {"id": "F", "count": int((positive & astro).sum()), "note": "only astrocyte"},
            {"id": "G", "count": int((positive & vasc).sum()), "note": "only vascular / endothelial"},
            {"id": "H", "count": h_count, "note": h_note},
            {"id": "I", "count": i_count, "note": i_note},
        ],
        "j_distributions": {
            "lzh_margin_P": summarize_numeric(registry.loc[positive, "lzh_margin"].to_numpy()),
            "lzh_margin_all": summarize_numeric(registry["lzh_margin"].to_numpy()),
            "wyh_margin_P": summarize_numeric(registry.loc[positive, "wyh_margin"].to_numpy()),
            "wyh_margin_all": summarize_numeric(registry["wyh_margin"].to_numpy()),
            "lzh_top1_P": summarize_numeric(registry.loc[positive, "lzh_top1"].to_numpy()),
            "wyh_top1_P": summarize_numeric(registry.loc[positive, "wyh_top1"].to_numpy()),
        },
    }
    jm = ceilings["j_distributions"]
    ceilings["j_text"] = (
        "GROUP P LZH margin mean {p_lzh_m:.4f} (population {all_lzh_m:.4f}); "
        "WYH margin mean {p_wyh_m:.4f} (population {all_wyh_m:.4f}). "
        "No threshold was chosen.".format(
            p_lzh_m=jm["lzh_margin_P"]["mean"],
            all_lzh_m=jm["lzh_margin_all"]["mean"],
            p_wyh_m=jm["wyh_margin_P"]["mean"],
            all_wyh_m=jm["wyh_margin_all"]["mean"],
        )
    )

    fold_block = {
        "fold_rescue_counts": fold_counts,
        "folds_0_2": mask_metrics(registry, fold_02),
        "folds_3_4": mask_metrics(registry, fold_34),
    }

    n_strong = sum(1 for r in inventory_rows if r["classification"] == "STRONG")
    n_moderate = sum(1 for r in inventory_rows if r["classification"] == "MODERATE")
    global_text = (
        "P vs N3 (dangerous override of a currently correct strong expert) is where LZH/WYH "
        "low-confidence and hard-bucket missingness show the strongest diagnostic AUROCs. "
        "P vs N1 (both strong experts already wrong, S0 also wrong) is the operational bottleneck: "
        "S0 top1/margin do not identify the 96 unique rescues among the 785 shared errors, and "
        "high S0 confidence deciles contain almost none of the 96 rescues. "
        "A cue that finds shared-confidence failures still leaves about a 12% S0 hit rate on those "
        "cells, which is unique-oracle headroom rather than a high-precision rescue regime."
    )

    future_use = (
        "LZH/WYH low confidence and metadata-missingness as negative controls against overriding "
        "a currently correct strong expert. Do not use high S0 confidence as a rescue trigger: "
        "the 96 unique rescues sit in low-to-mid S0 confidence. Do not treat LZH/WYH disagreement "
        "as the primary rescue state: 89 / 96 unique rescues occur when LZH == WYH."
    )

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": utc_now(),
        "head_note": head_note,
        "oracle_is_not_deployable_accuracy": True,
        "s0_probability": {
            "available": s0_proba_available,
            "reason": s0_inf.get("reason"),
            "checkpoint_path": s0_inf.get("checkpoint_path"),
            "checkpoint_sha256": s0_inf.get("checkpoint_sha256"),
            "checkpoint_bytes": s0_inf.get("checkpoint_bytes"),
            "device": s0_inf.get("device"),
            "n_label_match": s0_inf.get("n_label_match"),
            "retrained": False,
        },
        "integrity": {
            "n_rows": int(len(registry)),
            "unique_cell_id": int(registry["Cell_ID"].nunique()),
            "lzh_correct": int(lzh_ok.sum()),
            "wyh_correct": int(wyh_ok.sum()),
            "s0_correct": int(s0_ok.sum()),
            "lzh_accuracy": float(lzh_ok.mean()),
            "wyh_accuracy": float(wyh_ok.mean()),
            "s0_accuracy": float(s0_ok.mean()),
            "lzh_wyh_oracle": int((lzh_ok | wyh_ok).sum()),
            "lzh_wyh_oracle_accuracy": float((lzh_ok | wyh_ok).mean()),
            "both_wrong": int(both_wrong.sum()),
            "positive_rescues": int(positive.sum()),
            "three_expert_oracle": int(three_ok.sum()),
            "three_expert_oracle_accuracy": float(three_ok.mean()),
            "all_three_wrong": int((~three_ok).sum()),
            "fold_rescues": fold_counts,
        },
        "positive_rescues": {
            "n": int(positive.sum()),
            "by_true_class": by_class.to_dict(orient="records"),
            "by_section": [{"Section_ID": k, "n": int(v)} for k, v in by_section.items()],
            "hard_bucket_count": hard_p,
            "hard_bucket_fraction": float(hard_p / POSITIVE_RESCUES),
            "e02d_family_counts": dict(e02d_family_counts),
            "neuron_or_glial": p_df["neuron_or_glial"].value_counts().to_dict(),
        },
        "dangerous_overrides": {
            "n1": int(n1.sum()),
            "n2": int(n2.sum()),
            "n3": int(n3.sum()),
            "s0_wrong": int((~s0_ok).sum()),
        },
        "agreement_states": agreement,
        "hard_bucket": {"True": hard_true, "False": hard_false},
        "family_rescue_map": family_all,
        "family_rescue_map_folds_0_2": family_02,
        "family_rescue_map_folds_3_4": family_34,
        "confusion_pairs": confusion,
        "feature_diagnostics": feat_csv_rows,
        "feature_inventory": inventory_rows,
        "quantile_diagnostics": quantile,
        "quantile_stability": q_stable,
        "fold_stability": fold_block,
        "ceilings": ceilings,
        "separability_summary": {
            "global_text": global_text,
            "agreement_interpretation": agree_interp,
            "confusion_interpretation": confusion_interp,
            "fold_stability_text": (
                "Folds 0-2 vs 3-4 keep the unique-rescue identity 69 vs 27 and oligodendrocyte/OPC "
                "remains the largest rescue family by count. The oligo/OPC rescue rate among shared "
                "errors is not stable (0.176 on folds 0-2 vs 0.069 on folds 3-4). LZH/WYH "
                "low-confidence vs N3 keeps the same direction on both splits; S0 high-confidence "
                "does not isolate GROUP P on either split."
            ),
            "future_gate_should_use": future_use,
            "n_strong_features": n_strong,
            "n_moderate_features": n_moderate,
        },
        "leakage_audit": {
            "competition_test_labels_used": False,
            "true_labels_used_only_for_retrospective_diagnostics": True,
            "learned_gate_trained": False,
            "threshold_optimized": False,
            "weights_optimized": False,
            "leaderboard_feedback_used": False,
            "s0_retrained": False,
            "canonical_folds_3_4_used_for_redesign": False,
            "submission_generated": False,
            "v3_e00t_or_e02d_numerical_artifacts_modified": False,
            "prediction_csv_modified": False,
            "model_v1_modified": False,
            "model_v2_modified": False,
            "cell_id_dtype": "lossless_string",
        },
    }
    metrics["final_classification"] = decide_experiment(metrics)

    pd.DataFrame(
        [
            {"state": "A_LZH_eq_WYH", **state_a},
            {"state": "B_LZH_ne_WYH", **state_b},
        ]
    ).to_csv(TABLE_DIR / "agreement_states.csv", index=False)
    pd.DataFrame(lzh_pairs).to_csv(TABLE_DIR / "confusion_pairs_lzh.csv", index=False)
    pd.DataFrame(wyh_pairs).to_csv(TABLE_DIR / "confusion_pairs_wyh.csv", index=False)
    q_rows = []
    for name, block in quantile.items():
        for d in block["deciles"]:
            q_rows.append({"feature": name, **d})
    pd.DataFrame(q_rows).to_csv(TABLE_DIR / "quantile_diagnostics.csv", index=False)
    pd.DataFrame(ceilings["count_restrictions"]).to_csv(TABLE_DIR / "rescue_ceilings.csv", index=False)

    write_json(OUT_DIR / "v3_e03a_rescue_metrics.json", metrics)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(metrics))

    if sha256_file(E00T_REGISTRY) != e00t_before:
        integrity_failure("V3-E00T registry changed during the audit")
    if sha256_file(S0_VALIDATION) != s0_val_before:
        integrity_failure("V3-E02D S0 validation changed during the audit")
    if PRED_PATH.is_file() and sha256_file(PRED_PATH) != pred_before:
        integrity_failure("prediction/prediction.csv changed during the audit")

    print(
        json.dumps(
            {
                "status": "PASS",
                "experiment_id": EXPERIMENT_ID,
                "positive_rescues": int(positive.sum()),
                "n1": int(n1.sum()),
                "n2": int(n2.sum()),
                "n3": int(n3.sum()),
                "three_expert_oracle_accuracy": float(three_ok.mean()),
                "s0_probabilities": "available" if s0_proba_available else "UNAVAILABLE",
                "decision": metrics["final_classification"]["label"],
                "oracle_is_not_deployable_accuracy": True,
            },
            indent=2,
            default=json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
