from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the biology-prior + depth-masked strict-H stacker inputs."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-threshold", type=int, default=14)
    parser.add_argument("--h-weight", type=float, default=0.21)
    return parser.parse_args()


def read_probabilities(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    frame = pd.read_csv(path)
    id_column = "Cell_ID" if "Cell_ID" in frame.columns else frame.columns[0]
    probability_columns = [column for column in frame.columns if column != id_column]
    classes = [column[3:] if column.startswith("p__") else column for column in probability_columns]
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-8, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return frame[id_column].astype(str).to_numpy(), classes, probabilities


def align_probabilities(
    ids: np.ndarray,
    classes: list[str],
    probabilities: np.ndarray,
    target_ids: np.ndarray,
    target_classes: list[str],
) -> np.ndarray:
    row_index = pd.Index(ids).get_indexer(target_ids)
    if np.any(row_index < 0):
        missing = target_ids[row_index < 0][:5].tolist()
        raise ValueError(f"Missing Cell_ID values while aligning probabilities: {missing}")
    class_index = {name: index for index, name in enumerate(classes)}
    missing_classes = [name for name in target_classes if name not in class_index]
    if missing_classes:
        raise ValueError(f"Missing probability classes: {missing_classes}")
    column_index = [class_index[name] for name in target_classes]
    return probabilities[row_index][:, column_index]


def write_probabilities(
    path: Path,
    ids: np.ndarray,
    classes: list[str],
    probabilities: np.ndarray,
) -> None:
    frame = pd.DataFrame(probabilities, columns=[f"p__{name}" for name in classes])
    frame.insert(0, "Cell_ID", ids)
    frame.to_csv(path, index=False)


def copy_head(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = project_root / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from run_depth_masked_strict_h_gonogo import route_probabilities
    from run_depth_routed_tree_strict_h_gonogo import DEGREE_MINIMUM, RELIABLE_MINIMUM
    from run_reliability_piecewise_soft_slot_gonogo import fit_zero_reliability

    outputs = project_root / "outputs"
    strict_h_dir = outputs / "invariant_piecewise_mnn_segment_centered"
    graph_root = strict_h_dir / "graphs"
    shared_dir = strict_h_dir / "graph_shared"

    oof_prior_h_source = (
        outputs
        / "depth_masked_strict_h_gonogo"
        / f"oof_probabilities_depth_{args.depth_threshold}_masked_strict_h.csv"
    )
    anchor_test_source = outputs / "ordered_biology_submission" / "test_probabilities_conservative_union.csv"
    h_test_source = strict_h_dir / "test_probabilities_piecewise_mnn_segment_centered.csv"

    test_ids, classes, anchor_test = read_probabilities(anchor_test_source)
    h_ids, h_classes, h_test_raw = read_probabilities(h_test_source)
    h_test = align_probabilities(h_ids, h_classes, h_test_raw, test_ids, classes)

    raw_counts = np.load(cache_dir / "raw_counts.npy", mmap_mode="r")
    reference_positions = np.load(cache_dir / "reference_positions.npy")
    test_positions = np.load(cache_dir / "test_positions.npy")
    all_ids = np.load(cache_dir / "ids.npy")
    segment_codes = np.load(shared_dir / "segment_codes.npy")

    cache_test_ids = np.asarray(all_ids[test_positions]).astype(str)
    if not np.array_equal(cache_test_ids, test_ids):
        cache_row_index = pd.Index(cache_test_ids).get_indexer(test_ids)
        if np.any(cache_row_index < 0):
            raise ValueError("Test probability IDs cannot be aligned to cache test_positions.")
        test_positions = test_positions[cache_row_index]

    detected_all = np.asarray(raw_counts) > 0
    depth_all = detected_all.sum(axis=1).astype(np.int16)
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

    fold_degrees = []
    for fold in range(3):
        neighbors = np.load(graph_root / f"fold{fold}" / "neighbors.npy", mmap_mode="r")
        fold_degrees.append(np.sum(neighbors[test_positions] >= 0, axis=1).astype(np.int16))
    fold_degrees_array = np.stack(fold_degrees, axis=1)
    consensus_degree = np.median(fold_degrees_array, axis=1)

    test_depth = depth_all[test_positions]
    test_reliable_count = reliable_count_all[test_positions]
    routed_test, eligible = route_probabilities(
        anchor_test,
        h_test,
        test_depth,
        test_reliable_count,
        consensus_degree,
        args.depth_threshold,
        args.h_weight,
    )

    copy_head(oof_prior_h_source, output_dir / "oof_probabilities_prior_h_anchor.csv")
    write_probabilities(
        output_dir / "test_probabilities_prior_h_anchor.csv",
        test_ids,
        classes,
        routed_test,
    )

    head_sources = {
        "graph_stacker_v2": (
            outputs / "graph_regularized_logit_stacker" / "oof_probabilities_confidence_gated_stacker.csv",
            outputs / "graph_regularized_logit_stacker" / "test_probabilities_confidence_gated_stacker.csv",
        ),
        "current_anchor_prior": (
            outputs / "ordered_biology_submission" / "oof_probabilities_conservative_union.csv",
            outputs / "ordered_biology_submission" / "test_probabilities_conservative_union.csv",
        ),
        "gene_token": (
            outputs / "external_reference_fusion" / "oof_probabilities_external_gene_token.csv",
            outputs / "external_reference_fusion" / "test_probabilities_external_gene_token.csv",
        ),
        "strict_h": (
            strict_h_dir / "oof_probabilities_piecewise_mnn_segment_centered.csv",
            strict_h_dir / "test_probabilities_piecewise_mnn_segment_centered.csv",
        ),
    }
    for name, (oof_source, test_source) in head_sources.items():
        copy_head(oof_source, output_dir / f"oof_probabilities_{name}.csv")
        copy_head(test_source, output_dir / f"test_probabilities_{name}.csv")

    audit = {
        "route": {
            "base": "ordered_biology_submission/conservative_union",
            "optional_expert": "Invariant + Piecewise Segment-centered strict H",
            "depth_rule": f"n_detected > {args.depth_threshold}",
            "reliable_gene_minimum": int(RELIABLE_MINIMUM),
            "strict_graph_degree_minimum": int(DEGREE_MINIMUM),
            "test_degree_consensus": "median degree across three fold-specific strict graphs",
            "h_weight": float(args.h_weight),
        },
        "test": {
            "n_cells": int(test_ids.shape[0]),
            "eligible_cells": int(eligible.sum()),
            "eligible_fraction": float(eligible.mean()),
            "depth_pass": int((test_depth > args.depth_threshold).sum()),
            "reliable_pass": int((test_reliable_count >= RELIABLE_MINIMUM).sum()),
            "degree_pass": int((consensus_degree >= DEGREE_MINIMUM).sum()),
            "median_depth": float(np.median(test_depth)),
            "median_reliable_gene_count": float(np.median(test_reliable_count)),
            "median_consensus_degree": float(np.median(consensus_degree)),
        },
        "reliability_fit": reliability_audit,
        "stacker_heads": ["graph_stacker_v2", "current_anchor_prior", "gene_token", "strict_h"],
    }
    (output_dir / "prior_h_route_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit["test"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
