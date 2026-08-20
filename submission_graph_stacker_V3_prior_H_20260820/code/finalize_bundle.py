from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "MERFISH_cell_type_annotation.y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the reproducible submission bundle.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    return parser.parse_args()


def probability_classes(frame: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    id_column = "Cell_ID" if "Cell_ID" in frame.columns else frame.columns[0]
    probability_columns = [column for column in frame.columns if column != id_column]
    classes = [column[3:] if column.startswith("p__") else column for column in probability_columns]
    return id_column, probability_columns, classes


def metric_tuple(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["accuracy"]),
        float(metrics["macro_f1"]),
        -float(metrics["log_loss"]),
    )


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    bundle_root = args.bundle_root.resolve()
    trained_dir = bundle_root / "model" / "trained_stacker"
    inputs_dir = bundle_root / "model" / "inputs"
    metrics_path = trained_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    anchor_metrics = metrics["anchor_metrics"]
    stacker_metrics = metrics["stacker_metrics"]

    use_stacker = metric_tuple(stacker_metrics) > metric_tuple(anchor_metrics)
    if use_stacker:
        selected_name = "prior_h_graph_regularized_stacker"
        selected_oof = trained_dir / "oof_probabilities_confidence_gated_stacker.csv"
        selected_test = trained_dir / "test_probabilities_confidence_gated_stacker.csv"
        selected_metrics = stacker_metrics
        selection_reason = "The newly trained stacker wins the OOF lexicographic audit: accuracy, macro-F1, then log loss."
    else:
        selected_name = "depth_masked_prior_h_anchor"
        selected_oof = inputs_dir / "oof_probabilities_prior_h_anchor.csv"
        selected_test = inputs_dir / "test_probabilities_prior_h_anchor.csv"
        selected_metrics = anchor_metrics
        selection_reason = "The newly trained stacker did not beat the Prior-H Anchor, so submission falls back to the safer OOF winner."

    final_oof = bundle_root / "model" / "oof_probabilities_final.csv"
    final_test = bundle_root / "model" / "test_probabilities_final.csv"
    shutil.copy2(selected_oof, final_oof)
    shutil.copy2(selected_test, final_test)

    test_frame = pd.read_csv(final_test)
    id_column, probability_columns, classes = probability_classes(test_frame)
    prediction_index = np.argmax(
        test_frame[probability_columns].to_numpy(dtype=np.float64), axis=1
    )
    prediction = pd.DataFrame(
        {
            "Cell_ID": test_frame[id_column].astype(str),
            TARGET_COLUMN: np.asarray(classes, dtype=object)[prediction_index],
        }
    )
    prediction_dir = bundle_root / "prediction"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(prediction_dir / "prediction.csv", index=False)

    source_files = [
        "build_reference_class_similarity_graph.py",
        "train_confidence_gated_logit_stacker.py",
        "train_graph_regularized_logit_stacker.py",
        "train_external_gene_token_encoder.py",
        "evaluate_current_anchor_metadata_prior_head.py",
        "train_biology_aware_glia_head.py",
        "evaluate_anchor_marker_completion_residual.py",
        "evaluate_anatomical_topology_prior.py",
        "evaluate_ordered_oligodendrocyte_prior.py",
        "audit_ordered_biology_head_overlap.py",
        "export_biology_head_test_and_combine.py",
        "run_invariant_mnn_gonogo.py",
        "run_reliability_piecewise_soft_slot_gonogo.py",
        "build_invariant_piecewise_segment_mnn_graphs.py",
        "train_invariant_piecewise_mnn_segment_centered_encoder.py",
        "run_depth_masked_strict_h_gonogo.py",
        "build_external_segment_mnn_graph.py",
        "train_external_mnn_residual_encoder.py",
        "build_soft_slot_neighbor_graph.py",
        "evaluate_external_mnn_residual_fusion.py",
    ]
    heads_dir = bundle_root / "code" / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    copied_sources = []
    missing_sources = []
    for name in source_files:
        source = project_root / "src" / name
        if copy_if_exists(source, heads_dir / name):
            copied_sources.append(name)
        else:
            missing_sources.append(name)

    audit_sources = {
        "depth_masked_strict_h_gonogo": ["gonogo_metrics.json", "route_metrics.csv", "conditional_depth_accuracy.csv"],
        "invariant_piecewise_mnn_segment_centered": ["metrics.json", "model_comparison.csv", "graph_build_metrics.json"],
        "current_anchor_metadata_prior_head": ["metrics.json", "metadata_head_only_metrics.csv"],
        "biology_aware_glia_head": ["metrics.json", "configuration_metrics.csv"],
        "anchor_marker_completion_residual": ["metrics.json", "configuration_metrics.csv"],
        "anatomical_topology_prior": ["metrics.json", "configuration_metrics.csv"],
        "ordered_oligodendrocyte_prior": ["metrics.json", "configuration_metrics.csv"],
        "ordered_biology_submission": ["metrics.json"],
        "graph_regularized_logit_stacker": ["metrics.json"],
    }
    audits_dir = bundle_root / "model" / "audits"
    for directory, filenames in audit_sources.items():
        for filename in filenames:
            copy_if_exists(
                project_root / "outputs" / directory / filename,
                audits_dir / f"{directory}__{filename}",
            )

    route_audit = json.loads(
        (inputs_dir / "prior_h_route_audit.json").read_text(encoding="utf-8")
    )
    final_selection = {
        "selected_model": selected_name,
        "selection_reason": selection_reason,
        "selected_metrics": selected_metrics,
        "prior_h_anchor_metrics": anchor_metrics,
        "trained_stacker_metrics": stacker_metrics,
        "device": metrics.get("device", "unknown"),
        "test_route": route_audit["test"],
        "copied_head_sources": copied_sources,
        "missing_head_sources": missing_sources,
    }
    (bundle_root / "model" / "final_selection.json").write_text(
        json.dumps(final_selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    replacements = {
        "{{SELECTED_MODEL}}": selected_name,
        "{{SELECTION_REASON}}": selection_reason,
        "{{FINAL_ACC}}": f"{100.0 * float(selected_metrics['accuracy']):.2f}%",
        "{{FINAL_MACRO}}": f"{100.0 * float(selected_metrics['macro_f1']):.4f}%",
        "{{FINAL_WEIGHTED}}": f"{100.0 * float(selected_metrics['weighted_f1']):.4f}%",
        "{{FINAL_LOGLOSS}}": f"{float(selected_metrics['log_loss']):.6f}",
        "{{STACKER_ACC}}": f"{100.0 * float(stacker_metrics['accuracy']):.2f}%",
        "{{STACKER_MACRO}}": f"{100.0 * float(stacker_metrics['macro_f1']):.4f}%",
        "{{STACKER_WEIGHTED}}": f"{100.0 * float(stacker_metrics['weighted_f1']):.4f}%",
        "{{STACKER_LOGLOSS}}": f"{float(stacker_metrics['log_loss']):.6f}",
        "{{ANCHOR_ACC}}": f"{100.0 * float(anchor_metrics['accuracy']):.2f}%",
        "{{ANCHOR_MACRO}}": f"{100.0 * float(anchor_metrics['macro_f1']):.4f}%",
        "{{ANCHOR_WEIGHTED}}": f"{100.0 * float(anchor_metrics['weighted_f1']):.4f}%",
        "{{ANCHOR_LOGLOSS}}": f"{float(anchor_metrics['log_loss']):.6f}",
        "{{DEVICE}}": str(metrics.get("device", "unknown")),
        "{{ELIGIBLE_TEST}}": str(route_audit["test"]["eligible_cells"]),
        "{{ELIGIBLE_TEST_FRAC}}": f"{100.0 * float(route_audit['test']['eligible_fraction']):.2f}%",
    }
    template = (bundle_root / "README.template.md").read_text(encoding="utf-8")
    for token, value in replacements.items():
        template = template.replace(token, value)
    (bundle_root / "README.md").write_text(template, encoding="utf-8")
    print(json.dumps(final_selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
