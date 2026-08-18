from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from hierarchical_glia import build_internal_features, redistribute_family
from neuronal_experts import fit_family_expert
from train_model import LABEL, N_FOLDS, OFFICIAL, OUT, SEED, load_data


BASE_PROBABILITIES = OUT / "neuronal_experts_probabilities.npz"
EXTERNAL_REFERENCE = Path(__file__).resolve().parents[1] / "cache" / "external_glia_reference.npz"

PAIR_SPECS = [
    ("opc2_vs_oligo1", ["oligodendrocyte_progenitor_2", "oligodendrocyte_1"]),
    ("opc2_vs_oligo2", ["oligodendrocyte_progenitor_2", "oligodendrocyte_2"]),
    ("oligo1_vs_oligo2", ["oligodendrocyte_1", "oligodendrocyte_2"]),
    ("opc2_vs_astro1", ["oligodendrocyte_progenitor_2", "astrocyte_1"]),
    ("endothelial_vs_astro1", ["endothelial", "astrocyte_1"]),
    ("endothelial_vs_oligo1", ["endothelial", "oligodendrocyte_1"]),
    ("astro1_vs_astro2", ["astrocyte_1", "astrocyte_2"]),
    ("opc1_vs_precursor", ["oligodendrocyte_progenitor_1", "oligodendrocyte_precursor_cell"]),
    ("meninges1_vs_meninges2", ["meninges_1", "meninges_2"]),
    ("endothelial_vs_pericyte", ["endothelial", "pericyte"]),
    ("peripheral_vs_schwann", ["peripheral_glia", "Schwann_cell"]),
]


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "correct": int((prediction == y).sum()),
    }


def strict_gate(
    base: np.ndarray,
    specialist: np.ndarray,
    family_indices: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[dict, np.ndarray]:
    base_prediction = base.argmax(axis=1)
    base_correct = base_prediction == y
    candidates = []
    for gate in (0.30, 0.45, 0.60, 0.75):
        for confidence_cap in (0.55, 0.70, 0.85, 1.01):
            for weight in (0.25, 0.50, 0.75, 1.00):
                probability, selected = redistribute_family(
                    base, specialist, family_indices, gate, confidence_cap, weight
                )
                prediction = probability.argmax(axis=1)
                fold_delta = [
                    int(
                        (prediction[fold_ids == fold] == y[fold_ids == fold]).sum()
                        - base_correct[fold_ids == fold].sum()
                    )
                    for fold in range(N_FOLDS)
                ]
                candidates.append({
                    "gate": gate,
                    "confidence_cap": confidence_cap,
                    "weight": weight,
                    "delta_correct": int((prediction == y).sum() - base_correct.sum()),
                    "fold_delta_correct": fold_delta,
                    "selected_cells": int(selected.sum()),
                    "strictly_stable": min(fold_delta) >= 0 and sum(v > 0 for v in fold_delta) >= 2,
                })
    stable = [row for row in candidates if row["strictly_stable"] and row["delta_correct"] >= 2]
    if not stable:
        skipped = {
            "gate": 1.01,
            "confidence_cap": 0.0,
            "weight": 0.0,
            "delta_correct": 0,
            "fold_delta_correct": [0] * N_FOLDS,
            "selected_cells": 0,
            "strictly_stable": True,
            "skipped": True,
        }
        return skipped, base.copy()
    best = max(
        stable,
        key=lambda row: (
            row["delta_correct"],
            sum(v > 0 for v in row["fold_delta_correct"]),
            -row["selected_cells"],
            -row["weight"],
        ),
    )
    probability, _ = redistribute_family(
        base,
        specialist,
        family_indices,
        best["gate"],
        best["confidence_cap"],
        best["weight"],
    )
    return best, probability


def main() -> None:
    counts_train, counts_test, meta_train, meta_test = load_data()
    encoder = LabelEncoder().fit(meta_train[LABEL].astype(str))
    y = encoder.transform(meta_train[LABEL].astype(str))
    expr_train, expr_test, full_train, full_test = build_internal_features(
        counts_train, counts_test, meta_train, meta_test
    )
    splits = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(full_train, y)
    )
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, validation) in enumerate(splits):
        fold_ids[validation] = fold

    external = np.load(EXTERNAL_REFERENCE, allow_pickle=False)
    external_features = external["features"].astype(np.float32)
    external_labels = external["labels"].astype(str)
    stored = np.load(BASE_PROBABILITIES)
    current_oof = stored["oof"].astype(np.float32).copy()
    current_test = stored["test"].astype(np.float32).copy()
    baseline_oof = current_oof.copy()
    baseline_test_prediction = current_test.argmax(axis=1)
    report: dict = {"baseline": metrics(y, current_oof), "pairs": {}}

    for pair_name, labels in PAIR_SPECS:
        specialist_oof, specialist_test, family_indices, training = fit_family_expert(
            pair_name,
            labels,
            encoder,
            y,
            expr_train,
            expr_test,
            full_train,
            full_test,
            splits,
            external_features,
            external_labels,
        )
        gate, updated_oof = strict_gate(
            current_oof, specialist_oof, family_indices, y, fold_ids
        )
        updated_test, selected_test = redistribute_family(
            current_test,
            specialist_test,
            family_indices,
            gate["gate"],
            gate["confidence_cap"],
            gate["weight"],
        )
        current_oof = updated_oof
        current_test = updated_test
        report["pairs"][pair_name] = {
            "labels": labels,
            "training": training,
            "gate": gate,
            "test_cells_selected": int(selected_test.sum()),
            "cumulative_metrics": metrics(y, current_oof),
        }
        print(json.dumps({"stage": "pair_selected", "pair": pair_name, **report["pairs"][pair_name]}, indent=2), flush=True)

    baseline_prediction = baseline_oof.argmax(axis=1)
    final_prediction = current_oof.argmax(axis=1)
    report["final"] = metrics(y, current_oof)
    report["delta_correct"] = int(
        (final_prediction == y).sum() - (baseline_prediction == y).sum()
    )
    report["fold_delta_correct"] = [
        int(
            (final_prediction[fold_ids == fold] == y[fold_ids == fold]).sum()
            - (baseline_prediction[fold_ids == fold] == y[fold_ids == fold]).sum()
        )
        for fold in range(N_FOLDS)
    ]
    report["test_label_changes"] = int(
        (current_test.argmax(axis=1) != baseline_test_prediction).sum()
    )
    (OUT / "pair_experts_80_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "pair_experts_80_probabilities.npz", oof=current_oof, test=current_test, y=y
    )
    sample = pd.read_csv(OFFICIAL / "prediction" / "prediction.csv")
    sample.iloc[:, 1] = encoder.inverse_transform(current_test.argmax(axis=1))
    sample.to_csv(OUT / "prediction_pair_experts_80.csv", index=False)
    print(json.dumps({"stage": "final", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
