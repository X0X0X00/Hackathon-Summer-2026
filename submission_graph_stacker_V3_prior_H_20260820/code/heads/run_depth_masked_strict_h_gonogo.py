from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_depth_routed_tree_strict_h_gonogo as base
from run_reliability_piecewise_soft_slot_gonogo import fit_zero_reliability

OUTPUT_DIR = ROOT / "outputs" / "depth_masked_strict_h_gonogo"
GLOBAL_FUSION_OOF = (
    base.H_DIR / "oof_probabilities_anchor_piecewise_mnn_crossfit.csv"
)


def route_probabilities(
    anchor: np.ndarray,
    h_probabilities: np.ndarray,
    depth: np.ndarray,
    reliable_count: np.ndarray,
    degree: np.ndarray,
    threshold: int,
    h_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = (
        (depth > threshold)
        & (reliable_count >= base.RELIABLE_MINIMUM)
        & (degree >= base.DEGREE_MINIMUM)
    )
    routed = np.array(anchor, copy=True)
    if np.any(eligible):
        routed[eligible] = base.mix(
            anchor[eligible], h_probabilities[eligible], h_weight
        )
    routed = np.clip(routed, 1e-10, None)
    routed /= routed.sum(axis=1, keepdims=True)
    return routed, eligible


def tune_weight(
    truth: np.ndarray,
    anchor: np.ndarray,
    h_probabilities: np.ndarray,
    depth: np.ndarray,
    reliable_count: np.ndarray,
    degree: np.ndarray,
    threshold: int,
    tuning_mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best_weight = 0.0
    best_accuracy = -1.0
    best_logloss = float("inf")
    for weight in base.H_WEIGHT_GRID:
        routed, _ = route_probabilities(
            anchor,
            h_probabilities,
            depth,
            reliable_count,
            degree,
            threshold,
            float(weight),
        )
        metrics = base.class_f1_metrics(
            truth[tuning_mask], routed[tuning_mask]
        )
        if (
            metrics["accuracy"] > best_accuracy + 1e-12
            or (
                abs(metrics["accuracy"] - best_accuracy) <= 1e-12
                and metrics["log_loss"] < best_logloss
            )
        ):
            best_weight = float(weight)
            best_accuracy = metrics["accuracy"]
            best_logloss = metrics["log_loss"]
    return best_weight, {
        "accuracy": best_accuracy,
        "log_loss": best_logloss,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    h_ids, h_probabilities, classes = base.load_probabilities(base.H_OOF)
    anchor_ids, anchor_raw, anchor_classes = base.load_probabilities(
        base.ANCHOR_OOF
    )
    global_ids, global_raw, global_classes = base.load_probabilities(
        GLOBAL_FUSION_OOF
    )
    if classes != anchor_classes or classes != global_classes:
        raise ValueError("Probability class columns do not match")
    anchor = base.align_probabilities(h_ids, anchor_ids, anchor_raw)
    global_fusion = base.align_probabilities(h_ids, global_ids, global_raw)

    ids = np.load(base.DATA_DIR / "ids.npy", allow_pickle=True).astype(str)
    labels = np.load(base.DATA_DIR / "labels.npy")
    folds = np.load(base.DATA_DIR / "folds.npy")
    raw_counts = np.load(base.DATA_DIR / "raw_counts.npy", mmap_mode="r")
    reference_positions = np.load(
        base.DATA_DIR / "reference_positions.npy"
    )
    class_names = np.load(
        base.DATA_DIR / "class_names.npy", allow_pickle=True
    ).astype(str)
    if classes != class_names.tolist():
        raise ValueError("Dataset class order does not match")

    id_lookup = {cell_id: index for index, cell_id in enumerate(ids)}
    positions = np.asarray([id_lookup[cell_id] for cell_id in h_ids])
    truth = labels[positions].astype(np.int64)
    oof_folds = folds[positions].astype(np.int64)
    if np.any(truth < 0) or np.any(oof_folds < 0):
        raise ValueError("OOF cells do not have labels and folds")

    detected_all = np.asarray(raw_counts) > 0
    depth_all = detected_all.sum(axis=1).astype(np.int16)
    segment_codes = np.load(base.SHARED_DIR / "segment_codes.npy")
    zero_reliability, _, reliability_audit = fit_zero_reliability(
        detected_all,
        depth_all.astype(np.float32),
        segment_codes,
        reference_positions,
        floor=0.15,
        smoothing=20.0,
    )
    reliable_count_all = np.logical_or(
        detected_all, np.asarray(zero_reliability) >= 0.5
    ).sum(axis=1).astype(np.int16)
    depth = depth_all[positions]
    reliable_count = reliable_count_all[positions]

    degree = np.zeros(h_ids.shape[0], dtype=np.int16)
    for fold in np.unique(oof_folds):
        fold_mask = oof_folds == fold
        graph = np.load(
            base.GRAPH_ROOT / f"fold{int(fold)}" / "neighbors.npy",
            mmap_mode="r",
        )
        degree[fold_mask] = np.sum(
            graph[positions[fold_mask]] >= 0, axis=1
        ).astype(np.int16)

    baselines = {
        "current_anchor": base.class_f1_metrics(truth, anchor),
        "strict_h": base.class_f1_metrics(truth, h_probabilities),
        "global_anchor_strict_h_crossfit": base.class_f1_metrics(
            truth, global_fusion
        ),
    }

    depth_bins = (
        ("detected_le_11", depth <= 11),
        ("detected_12_14", (depth >= 12) & (depth <= 14)),
        ("detected_15_21", (depth >= 15) & (depth <= 21)),
        ("detected_gt_21", depth > 21),
    )
    conditional_rows: list[dict[str, object]] = []
    for name, mask in depth_bins:
        conditional_rows.append(
            {
                "depth_bin": name,
                "n": int(np.sum(mask)),
                "anchor_accuracy": base.subset_accuracy(
                    truth, anchor, mask
                ),
                "strict_h_accuracy": base.subset_accuracy(
                    truth, h_probabilities, mask
                ),
                "global_fusion_accuracy": base.subset_accuracy(
                    truth, global_fusion, mask
                ),
            }
        )

    unique_folds = np.unique(oof_folds)
    route_rows: list[dict[str, object]] = []
    route_summaries: dict[str, object] = {}
    route_outputs: dict[int, np.ndarray] = {}
    for threshold in base.DEPTH_THRESHOLDS:
        crossfit = np.zeros_like(anchor)
        weights: list[float] = []
        fold_details: list[dict[str, object]] = []
        for held_fold in unique_folds:
            tuning_mask = oof_folds != held_fold
            held_mask = oof_folds == held_fold
            weight, tuning_metrics = tune_weight(
                truth,
                anchor,
                h_probabilities,
                depth,
                reliable_count,
                degree,
                threshold,
                tuning_mask,
            )
            routed, eligible = route_probabilities(
                anchor,
                h_probabilities,
                depth,
                reliable_count,
                degree,
                threshold,
                weight,
            )
            crossfit[held_mask] = routed[held_mask]
            held_pair = base.paired(
                truth[held_mask],
                anchor[held_mask],
                routed[held_mask],
            )
            fold_details.append(
                {
                    "fold": int(held_fold),
                    "h_weight": weight,
                    "tuning_metrics": tuning_metrics,
                    "held_metrics": base.class_f1_metrics(
                        truth[held_mask], routed[held_mask]
                    ),
                    "held_h_eligible_count": int(
                        np.sum(held_mask & eligible)
                    ),
                    "paired_vs_anchor": held_pair,
                }
            )
            weights.append(weight)

        route_outputs[threshold] = crossfit
        metrics = base.class_f1_metrics(truth, crossfit)
        pair = base.paired(truth, anchor, crossfit)
        positive_folds = int(
            np.sum(
                [
                    detail["paired_vs_anchor"]["net"] > 0
                    for detail in fold_details
                ]
            )
        )
        eligible_all = (
            (depth > threshold)
            & (reliable_count >= base.RELIABLE_MINIMUM)
            & (degree >= base.DEGREE_MINIMUM)
        )
        go = (
            metrics["accuracy"] > baselines["current_anchor"]["accuracy"]
            and metrics["macro_f1"]
            >= baselines["current_anchor"]["macro_f1"]
            and pair["net"] > 0
            and positive_folds >= 2
        )
        summary = {
            "threshold": threshold,
            "metrics": metrics,
            "paired_vs_anchor": pair,
            "mean_h_weight": float(np.mean(weights)),
            "fold_weights": weights,
            "h_eligible_fraction": float(np.mean(eligible_all)),
            "h_disabled_fraction": float(np.mean(~eligible_all)),
            "positive_folds": positive_folds,
            "fold_details": fold_details,
            "decision": "GO" if go else "NO-GO",
        }
        route_summaries[str(threshold)] = summary
        route_rows.append(
            {
                "threshold": threshold,
                **metrics,
                **{
                    f"paired_{key}": value
                    for key, value in pair.items()
                },
                "mean_h_weight": summary["mean_h_weight"],
                "h_eligible_fraction": summary["h_eligible_fraction"],
                "positive_folds": positive_folds,
                "decision": summary["decision"],
            }
        )
        print(
            f"threshold={threshold} accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"logloss={metrics['log_loss']:.4f} "
            f"gained={pair['gained']} lost={pair['lost']} "
            f"weights={weights} eligible={np.mean(eligible_all):.4f} "
            f"decision={summary['decision']}",
            flush=True,
        )

    passing = [
        int(threshold)
        for threshold, summary in route_summaries.items()
        if summary["decision"] == "GO"
    ]
    recommended = None
    if passing:
        recommended = max(
            passing,
            key=lambda threshold: (
                route_summaries[str(threshold)]["metrics"]["accuracy"],
                route_summaries[str(threshold)]["metrics"]["macro_f1"],
            ),
        )

    result = {
        "configuration": {
            "depth_thresholds": base.DEPTH_THRESHOLDS,
            "low_depth_behavior": "CurrentAnchor with H weight exactly zero",
            "high_depth_base": "CurrentAnchor",
            "high_depth_optional_expert": "strict Invariant H",
            "reliable_gene_minimum": base.RELIABLE_MINIMUM,
            "strict_graph_degree_minimum": base.DEGREE_MINIMUM,
            "h_weight_grid": base.H_WEIGHT_GRID.tolist(),
            "crossfit_protocol": "H weight tuned on the other two folds",
        },
        "baselines": baselines,
        "conditional_depth_accuracy": conditional_rows,
        "routes": route_summaries,
        "overall_decision": "GO" if recommended is not None else "NO-GO",
        "recommended_threshold": recommended,
        "reliability_audit": reliability_audit,
    }
    pd.DataFrame(conditional_rows).to_csv(
        OUTPUT_DIR / "conditional_depth_accuracy.csv", index=False
    )
    pd.DataFrame(route_rows).to_csv(
        OUTPUT_DIR / "route_metrics.csv", index=False
    )
    for threshold, probabilities in route_outputs.items():
        frame = pd.DataFrame(
            probabilities, columns=[f"p__{name}" for name in classes]
        )
        frame.insert(0, "Cell_ID", h_ids)
        frame.to_csv(
            OUTPUT_DIR
            / f"oof_probabilities_depth_{threshold}_masked_strict_h.csv",
            index=False,
        )
    with (OUTPUT_DIR / "gonogo_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, default=str)
    print(
        f"overall={result['overall_decision']} "
        f"recommended_threshold={recommended}",
        flush=True,
    )


if __name__ == "__main__":
    main()
