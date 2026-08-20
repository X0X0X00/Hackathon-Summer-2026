"""V2-C fixed-weight blend of saved expert probabilities.

No model fitting. No weight search. The five predeclared candidates are the
complete search space.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from merfish60.io import N_CLASSES, N_TEST_CELLS, N_TRAIN_CELLS
from merfish60.metrics import summarize_oof
from merfish60.models import argmax_labels, assert_probability_rows
from merfish60.v2_metrics import confusion_pairs, slice_metrics


class V2CAlignmentError(Exception):
    pass


EXPERT_A_OOF = "outputs/probabilities/BRIDGE-YW004-5F_oof_probabilities.csv.gz"
EXPERT_B_OOF = "outputs/probabilities/V2-A-SPATIAL-LGBM_oof_probabilities_ei.csv.gz"
EXPERT_C_OOF = "outputs/probabilities/V2-B-REFONLY_oof_probabilities_seg.csv.gz"
# BRIDGE never wrote a dedicated test file; MODEL V1 is the frozen YW-004
# full-train hierarchical test artifact (same family as BRIDGE).
EXPERT_A_TEST = "outputs/probabilities/model_v1_test_probabilities.csv.gz"
EXPERT_B_TEST = "outputs/probabilities/V2-A-SPATIAL-LGBM_test_probabilities.csv.gz"
EXPERT_C_TEST = "outputs/probabilities/V2-B-REFONLY_test_probabilities_seg.csv.gz"

FIXED_BLENDS: Dict[str, Tuple[float, float, float]] = {
    "C0": (0.00, 0.00, 1.00),
    "C1": (0.00, 0.25, 0.75),
    "C2": (0.15, 0.15, 0.70),
    "C3": (0.20, 0.20, 0.60),
    "C4": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
}
BLEND_IDS = ("C0", "C1", "C2", "C3", "C4")
V2B_OOF_TARGET = 0.8212
V2B_CORRECT_TARGET = 4106
MIN_NET_CELLS_TO_REPLACE_C0 = 10
MAX_SLICE_REGRESSION = 0.005
MAX_MACRO_F1_REGRESSION = 0.005


def load_proba_frame(path, class_names: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"Cell_ID": str}, compression="gzip")
    expected = ["Cell_ID"] + list(class_names)
    if list(frame.columns) != expected:
        raise V2CAlignmentError(
            "{} columns {} != official 60-class order".format(path, list(frame.columns)[:5])
        )
    frame["Cell_ID"] = frame["Cell_ID"].astype(str)
    return frame


def load_oof_labels(path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"Cell_ID": str, "true_label": str, "predicted_label": str, "fold": int},
    )


def assert_oof_alignment(
    frames: Mapping[str, pd.DataFrame],
    train_ids: Sequence[str],
    folds: pd.DataFrame,
    class_names: Sequence[str],
) -> List[str]:
    messages = []
    train_ids = [str(v) for v in train_ids]
    ref_ids = None
    for name, frame in frames.items():
        ids = [str(v) for v in frame["Cell_ID"].tolist()]
        if len(ids) != N_TRAIN_CELLS:
            raise V2CAlignmentError("{} has {} OOF rows".format(name, len(ids)))
        if len(set(ids)) != N_TRAIN_CELLS:
            raise V2CAlignmentError("{} has duplicate OOF Cell_IDs".format(name))
        if ids != train_ids:
            raise V2CAlignmentError(
                "{} OOF Cell_ID order != counts_train; refusing silent reorder".format(name)
            )
        if ref_ids is None:
            ref_ids = ids
        elif ids != ref_ids:
            raise V2CAlignmentError("{} OOF Cell_ID order != {}".format(name, next(iter(frames))))
        matrix = frame[list(class_names)].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise V2CAlignmentError("{} has NaN/inf probabilities".format(name))
        assert_probability_rows(matrix, atol=1e-5)
        messages.append("{} OOF aligned n=5000 class_order=official row_sums~1".format(name))
    fold_map = folds.set_index("Cell_ID")["fold"]
    oof_folds = [int(fold_map.loc[cid]) for cid in train_ids]
    if sorted(set(oof_folds)) != [0, 1, 2, 3, 4]:
        raise V2CAlignmentError("team folds are not 0..4")
    messages.append("team-compatible folds aligned on OOF Cell_IDs")
    return messages


def assert_test_alignment(
    frames: Mapping[str, pd.DataFrame],
    test_ids: Sequence[str],
    class_names: Sequence[str],
) -> List[str]:
    messages = []
    test_ids = [str(v) for v in test_ids]
    if len(test_ids) != N_TEST_CELLS:
        raise V2CAlignmentError("expected 5000 test Cell_IDs")
    for name, frame in frames.items():
        ids = [str(v) for v in frame["Cell_ID"].tolist()]
        if ids != test_ids:
            raise V2CAlignmentError(
                "{} test Cell_ID order != meta_test; refusing silent reorder".format(name)
            )
        matrix = frame[list(class_names)].to_numpy(dtype=np.float64)
        if matrix.shape != (N_TEST_CELLS, N_CLASSES):
            raise V2CAlignmentError("{} test shape {}".format(name, matrix.shape))
        if not np.isfinite(matrix).all():
            raise V2CAlignmentError("{} test has NaN/inf".format(name))
        assert_probability_rows(matrix, atol=1e-5)
        messages.append("{} test aligned n=5000 meta_test order row_sums~1".format(name))
    return messages


def mix_fixed(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    wa, wb, wc = (float(weights[0]), float(weights[1]), float(weights[2]))
    if abs(wa + wb + wc - 1.0) > 1e-12:
        raise V2CAlignmentError("weights {} do not sum to 1".format(weights))
    mixed = wa * pa + wb * pb + wc * pc
    totals = mixed.sum(axis=1, keepdims=True)
    return mixed / np.maximum(totals, 1e-15)


def pair_table(ok_left: np.ndarray, ok_right: np.ndarray, pred_left, pred_right) -> dict:
    both_correct = int((ok_left & ok_right).sum())
    both_wrong = int((~ok_left & ~ok_right).sum())
    left_only = int((ok_left & ~ok_right).sum())
    right_only = int((~ok_left & ok_right).sum())
    disagree = int((np.asarray(pred_left) != np.asarray(pred_right)).sum())
    oracle = int((ok_left | ok_right).sum())
    n = int(len(ok_left))
    return {
        "n": n,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "first_correct_second_wrong": left_only,
        "first_wrong_second_correct": right_only,
        "argmax_disagreement": disagree,
        "disagreement_rate": disagree / n,
        "pairwise_oracle_correct": oracle,
        "pairwise_oracle_accuracy": oracle / n,
    }


def complementarity(ok_a, ok_b, ok_c, pred_a, pred_b, pred_c) -> dict:
    n = int(len(ok_a))
    three_oracle = int((ok_a | ok_b | ok_c).sum())
    return {
        "n": n,
        "A_vs_B": pair_table(ok_a, ok_b, pred_a, pred_b),
        "A_vs_C": pair_table(ok_a, ok_c, pred_a, pred_c),
        "B_vs_C": pair_table(ok_b, ok_c, pred_b, pred_c),
        "only_A_correct": int((ok_a & ~ok_b & ~ok_c).sum()),
        "only_B_correct": int((ok_b & ~ok_a & ~ok_c).sum()),
        "only_C_correct": int((ok_c & ~ok_a & ~ok_b).sum()),
        "A_and_B_correct_C_wrong": int((ok_a & ok_b & ~ok_c).sum()),
        "A_and_C_correct_B_wrong": int((ok_a & ok_c & ~ok_b).sum()),
        "B_and_C_correct_A_wrong": int((ok_b & ok_c & ~ok_a).sum()),
        "all_three_wrong": int((~ok_a & ~ok_b & ~ok_c).sum()),
        "three_expert_oracle_correct": three_oracle,
        "three_expert_oracle_accuracy": three_oracle / n,
        "headroom_only": True,
    }


def cell_delta(pred_ref, pred_new, true) -> dict:
    pred_ref = np.asarray(pred_ref, dtype=object)
    pred_new = np.asarray(pred_new, dtype=object)
    true = np.asarray(true, dtype=object)
    ref_ok = pred_ref == true
    new_ok = pred_new == true
    return {
        "wrong_to_correct": int((~ref_ok & new_ok).sum()),
        "correct_to_wrong": int((ref_ok & ~new_ok).sum()),
        "net_gain": int(new_ok.sum() - ref_ok.sum()),
        "changed_predictions": int((pred_ref != pred_new).sum()),
    }


def fold_deltas(fold_acc: Mapping[int, float], ref_fold_acc: Mapping[int, float]) -> dict:
    deltas = {}
    for fold_id in (0, 1, 2, 3, 4):
        deltas[str(fold_id)] = float(fold_acc[fold_id] - ref_fold_acc[fold_id])
    values = [deltas[str(i)] for i in range(5)]
    n_pos = sum(v > 1e-12 for v in values)
    n_neg = sum(v < -1e-12 for v in values)
    total = float(np.mean(values))
    fold34 = min(values[3], values[4])
    if n_neg == 0 and fold34 >= -0.003:
        flag = "STABLE"
    elif n_neg >= 3 or (n_pos == 1 and total > 0):
        flag = "UNSTABLE"
    else:
        flag = "MIXED"
    return {"per_fold_delta": deltas, "n_positive": n_pos, "n_negative": n_neg, "flag": flag}


def score_blend(
    proba: np.ndarray,
    y_true: Sequence[str],
    folds: Sequence[int],
    meta_train,
    class_names: Sequence[str],
    ei_of_label: np.ndarray,
) -> dict:
    pred = argmax_labels(proba, class_names)
    y_true = np.asarray(y_true, dtype=object)
    metrics = summarize_oof(y_true, pred, folds, labels=class_names)
    slices = slice_metrics(y_true, pred, meta_train, class_names, ei_of_label)
    return {
        "fold_accuracy": {str(k): v for k, v in metrics["fold_accuracy"].items()},
        "oof_accuracy": metrics["oof_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "correct": int((pred == y_true).sum()),
        "wrong": int((pred != y_true).sum()),
        "pred": pred,
        "slices": slices,
        "confusion_pairs_top10": slices["confusion_pairs_top10"],
    }


def confusion_shift(y_true, pred_ref, pred_new, k: int = 8) -> dict:
    ref = Counter()
    new = Counter()
    for t, p in zip(y_true, pred_ref):
        if t != p:
            ref[(str(t), str(p))] += 1
    for t, p in zip(y_true, pred_new):
        if t != p:
            new[(str(t), str(p))] += 1
    keys = set(ref) | set(new)
    reductions = sorted(
        (
            {"true_label": a, "predicted_label": b, "c0": ref[(a, b)], "cand": new[(a, b)], "delta": ref[(a, b)] - new[(a, b)]}
            for (a, b) in keys
        ),
        key=lambda row: row["delta"],
        reverse=True,
    )
    introductions = sorted(
        (
            {"true_label": a, "predicted_label": b, "c0": ref[(a, b)], "cand": new[(a, b)], "delta": new[(a, b)] - ref[(a, b)]}
            for (a, b) in keys
        ),
        key=lambda row: row["delta"],
        reverse=True,
    )
    families = {
        "oligo1_vs_opc2": {
            "c0": ref[("oligodendrocyte_1", "oligodendrocyte_progenitor_2")]
            + ref[("oligodendrocyte_progenitor_2", "oligodendrocyte_1")],
            "cand": new[("oligodendrocyte_1", "oligodendrocyte_progenitor_2")]
            + new[("oligodendrocyte_progenitor_2", "oligodendrocyte_1")],
        },
        "oligo2_vs_opc2": {
            "c0": ref[("oligodendrocyte_2", "oligodendrocyte_progenitor_2")]
            + ref[("oligodendrocyte_progenitor_2", "oligodendrocyte_2")],
            "cand": new[("oligodendrocyte_2", "oligodendrocyte_progenitor_2")]
            + new[("oligodendrocyte_progenitor_2", "oligodendrocyte_2")],
        },
        "endothelial_vs_astrocyte_1": {
            "c0": ref[("endothelial", "astrocyte_1")] + ref[("astrocyte_1", "endothelial")],
            "cand": new[("endothelial", "astrocyte_1")] + new[("astrocyte_1", "endothelial")],
        },
        "meninges_1_vs_2": {
            "c0": ref[("meninges_1", "meninges_2")] + ref[("meninges_2", "meninges_1")],
            "cand": new[("meninges_1", "meninges_2")] + new[("meninges_2", "meninges_1")],
        },
    }
    return {
        "largest_reductions": reductions[:k],
        "new_or_increased": introductions[:k],
        "residual_families": families,
        "c0_top": confusion_pairs(y_true, pred_ref, k=10),
        "cand_top": confusion_pairs(y_true, pred_new, k=10),
    }


def write_candidate_csv(path, cell_ids: Sequence[str], pred: Sequence[str]) -> None:
    from merfish60.official_contract import (
        SUBMISSION_CELL_ID_COL,
        SUBMISSION_COLUMNS,
        SUBMISSION_PRED_COL,
    )

    frame = pd.DataFrame(
        {
            SUBMISSION_CELL_ID_COL: [str(v) for v in cell_ids],
            SUBMISSION_PRED_COL: [str(v) for v in pred],
        }
    )
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise RuntimeError("submission columns {}".format(list(frame.columns)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def select_model_v2(scoreboard: Mapping[str, dict]) -> dict:
    """Choose C0 unless a predeclared blend clearly and stably improves it."""
    c0 = scoreboard["C0"]
    if abs(c0["oof_accuracy"] - V2B_OOF_TARGET) > 1e-6:
        raise V2CAlignmentError(
            "C0 OOF {} != validated V2-B {}".format(c0["oof_accuracy"], V2B_OOF_TARGET)
        )
    winner = "C0"
    reason = (
        "Default to V2-B/C0: simpler reference-only architecture unless a "
        "predeclared blend clearly exceeds 0.8212 with stable folds and no "
        "material slice regression."
    )
    for cand_id in ("C1", "C2", "C3", "C4"):
        row = scoreboard[cand_id]
        net = int(row["vs_c0"]["net_gain"])
        oof = float(row["oof_accuracy"])
        stable = row["stability"]["flag"]
        hard_delta = row["slices"]["hard_bucket_accuracy"] - c0["slices"]["hard_bucket_accuracy"]
        neuron_delta = row["slices"]["neuron_accuracy"] - c0["slices"]["neuron_accuracy"]
        glial_delta = row["slices"]["glial_accuracy"] - c0["slices"]["glial_accuracy"]
        f1_delta = row["macro_f1"] - c0["macro_f1"]
        fold34 = min(
            row["stability"]["per_fold_delta"]["3"],
            row["stability"]["per_fold_delta"]["4"],
        )
        ok = (
            oof > V2B_OOF_TARGET
            and net >= MIN_NET_CELLS_TO_REPLACE_C0
            and stable != "UNSTABLE"
            and hard_delta >= -MAX_SLICE_REGRESSION
            and neuron_delta >= -MAX_SLICE_REGRESSION
            and glial_delta >= -MAX_SLICE_REGRESSION
            and f1_delta >= -MAX_MACRO_F1_REGRESSION
            and fold34 >= -0.005
        )
        row["selection_eligible"] = bool(ok)
        if ok and oof > scoreboard[winner]["oof_accuracy"]:
            winner = cand_id
            reason = (
                "{} exceeds V2-B ({:.4f} vs 0.8212), net {:+d} cells, "
                "stability={}, no material slice/macro-F1 regression.".format(
                    cand_id, oof, net, stable
                )
            )
    if winner == "C0":
        eligible = [cid for cid in ("C1", "C2", "C3", "C4") if scoreboard[cid].get("selection_eligible")]
        if not eligible:
            reason = (
                "No predeclared blend clearly improves OOF with fold-stable, "
                "slice-safe gain. Keep C0 = V2-B reference-only "
                "(OOF={:.4f}, {}/5000).".format(c0["oof_accuracy"], c0["correct"])
            )
    return {"selected_id": winner, "reason": reason}
