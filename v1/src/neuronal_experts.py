from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from hierarchical_glia import (
    aligned_probability,
    build_internal_features,
    choose_gate,
    redistribute_family,
    sqrt_balancing_weights,
)
from train_model import LABEL, N_FOLDS, OFFICIAL, OUT, SEED, load_data


BASE_PROBABILITIES = OUT / "targeted_80_probabilities.npz"
EXTERNAL_REFERENCE = Path(__file__).resolve().parents[1] / "cache" / "external_neuronal_reference.npz"


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "correct": int((prediction == y).sum()),
    }


def fold_delta_correct(
    y: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    fold_ids: np.ndarray,
) -> list[int]:
    before_prediction = before.argmax(axis=1)
    after_prediction = after.argmax(axis=1)
    return [
        int(
            (after_prediction[fold_ids == fold] == y[fold_ids == fold]).sum()
            - (before_prediction[fold_ids == fold] == y[fold_ids == fold]).sum()
        )
        for fold in range(N_FOLDS)
    ]


def fit_family_expert(
    family_name: str,
    family_names: list[str],
    encoder: LabelEncoder,
    y: np.ndarray,
    expr_train: np.ndarray,
    expr_test: np.ndarray,
    full_train: np.ndarray,
    full_test: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    external_features: np.ndarray,
    external_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    family_indices = np.asarray(
        [int(np.where(encoder.classes_ == name)[0][0]) for name in family_names]
    )
    global_to_local = {int(index): local for local, index in enumerate(family_indices)}
    family_mask = np.isin(y, family_indices)
    external_mask = np.isin(external_labels, family_names)
    external_x = external_features[external_mask]
    label_lookup = {name: local for local, name in enumerate(family_names)}
    external_y = np.asarray([label_lookup[name] for name in external_labels[external_mask]])
    n_classes = len(family_names)

    oof_internal = np.zeros((len(y), n_classes), dtype=np.float32)
    test_internal = np.zeros((len(full_test), n_classes), dtype=np.float32)
    oof_reference = np.zeros((len(y), n_classes), dtype=np.float32)
    test_reference = np.zeros((len(full_test), n_classes), dtype=np.float32)
    fold_rows = []

    for fold, (fit_idx, val_idx) in enumerate(splits, 1):
        family_fit = fit_idx[family_mask[fit_idx]]
        family_val = val_idx[family_mask[val_idx]]
        local_y = np.asarray([global_to_local[int(value)] for value in y[family_fit]])
        local_val_y = np.asarray([global_to_local[int(value)] for value in y[family_val]])

        internal_tree = LGBMClassifier(
            objective="binary" if n_classes == 2 else "multiclass",
            n_estimators=480,
            learning_rate=0.025,
            num_leaves=14,
            min_child_samples=12,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            reg_alpha=0.20,
            random_state=SEED + 701 * fold + n_classes,
            n_jobs=-1,
            verbosity=-1,
        )
        internal_tree.fit(
            full_train[family_fit],
            local_y,
            sample_weight=sqrt_balancing_weights(local_y),
            eval_set=[(full_train[family_val], local_val_y)],
            callbacks=[lgb.early_stopping(65, verbose=False)],
        )
        scaler = StandardScaler()
        scaled_fit = scaler.fit_transform(full_train[family_fit])
        internal_linear = LogisticRegression(C=0.28, max_iter=2200, solver="lbfgs")
        internal_linear.fit(
            scaled_fit, local_y, sample_weight=sqrt_balancing_weights(local_y)
        )
        oof_internal[val_idx] = (
            0.65 * aligned_probability(internal_tree, full_train[val_idx], n_classes)
            + 0.35
            * aligned_probability(
                internal_linear, scaler.transform(full_train[val_idx]), n_classes
            )
        )
        test_internal += (
            0.65 * aligned_probability(internal_tree, full_test, n_classes)
            + 0.35
            * aligned_probability(
                internal_linear, scaler.transform(full_test), n_classes
            )
        ) / N_FOLDS

        combined_x = np.vstack([expr_train[family_fit], external_x])
        combined_y = np.concatenate([local_y, external_y])
        combined_weight = np.concatenate([
            sqrt_balancing_weights(local_y),
            0.30 * sqrt_balancing_weights(external_y),
        ])
        reference_tree = LGBMClassifier(
            objective="binary" if n_classes == 2 else "multiclass",
            n_estimators=520 if n_classes > 10 else 440,
            learning_rate=0.025,
            num_leaves=18,
            min_child_samples=18,
            colsample_bytree=0.88,
            reg_lambda=6.0,
            reg_alpha=0.25,
            random_state=SEED + 1709 * fold + n_classes,
            n_jobs=-1,
            verbosity=-1,
        )
        reference_tree.fit(combined_x, combined_y, sample_weight=combined_weight)
        oof_reference[val_idx] = aligned_probability(
            reference_tree, expr_train[val_idx], n_classes
        )
        test_reference += aligned_probability(reference_tree, expr_test, n_classes) / N_FOLDS

        fold_rows.append({
            "fold": fold,
            "fit_family_cells": int(len(family_fit)),
            "validation_family_cells": int(len(family_val)),
            "external_family_cells": int(len(external_y)),
            "internal_accuracy": float(
                accuracy_score(local_val_y, oof_internal[family_val].argmax(axis=1))
            ),
            "reference_accuracy": float(
                accuracy_score(local_val_y, oof_reference[family_val].argmax(axis=1))
            ),
            "internal_tree_best_iteration": int(internal_tree.best_iteration_),
        })
        print(json.dumps({"stage": "neuronal_training", "family": family_name, **fold_rows[-1]}), flush=True)

    all_local_y = np.asarray([global_to_local[int(value)] for value in y[family_mask]])
    blend_scores = []
    for external_weight in np.linspace(0.0, 1.0, 11):
        probability = (
            (1.0 - external_weight) * oof_internal
            + external_weight * oof_reference
        )
        blend_scores.append({
            "external_weight": float(external_weight),
            "family_accuracy": float(
                accuracy_score(all_local_y, probability[family_mask].argmax(axis=1))
            ),
        })
    selected_weight = max(
        blend_scores,
        key=lambda row: (row["family_accuracy"], -abs(row["external_weight"] - 0.6)),
    )["external_weight"]
    oof = (1.0 - selected_weight) * oof_internal + selected_weight * oof_reference
    test = (1.0 - selected_weight) * test_internal + selected_weight * test_reference
    report = {
        "folds": fold_rows,
        "blend_scores": blend_scores,
        "selected_external_weight": float(selected_weight),
        "selected_family_accuracy": float(
            accuracy_score(all_local_y, oof[family_mask].argmax(axis=1))
        ),
    }
    return oof, test, family_indices, report


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
    for fold, (_, val_idx) in enumerate(splits):
        fold_ids[val_idx] = fold

    external = np.load(EXTERNAL_REFERENCE, allow_pickle=False)
    external_features = external["features"].astype(np.float32)
    external_labels = external["labels"].astype(str)
    families = {
        "dorsal_horn_excitatory": [
            name for name in encoder.classes_ if name.startswith("DH_ex_")
        ],
        "dorsal_horn_inhibitory": [
            name for name in encoder.classes_ if name.startswith("DH_in_")
        ],
        "motoneuron": [name for name in encoder.classes_ if "motoneuron" in name],
    }

    stored = np.load(BASE_PROBABILITIES)
    current_oof = stored["oof"].astype(np.float32).copy()
    current_test = stored["test"].astype(np.float32).copy()
    baseline_oof = current_oof.copy()
    report = {"baseline": metrics(y, current_oof), "families": {}}

    for family_name, family_names in families.items():
        specialist_oof, specialist_test, family_indices, training = fit_family_expert(
            family_name,
            family_names,
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
        gate, updated_oof = choose_gate(
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
        current_oof, current_test = updated_oof, updated_test
        report["families"][family_name] = {
            "labels": family_names,
            "training": training,
            "gate": gate,
            "test_cells_modified": int(selected_test.sum()),
            "cumulative_metrics": metrics(y, current_oof),
        }
        print(json.dumps({"stage": "neuronal_selected", "family": family_name, **report["families"][family_name]}, indent=2), flush=True)

    report["final"] = metrics(y, current_oof)
    report["delta_correct"] = report["final"]["correct"] - report["baseline"]["correct"]
    report["fold_delta_correct"] = fold_delta_correct(y, baseline_oof, current_oof, fold_ids)
    print(json.dumps({"stage": "final", **report}, indent=2), flush=True)

    (OUT / "neuronal_experts_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "neuronal_experts_probabilities.npz",
        oof=current_oof,
        test=current_test,
        y=y,
    )
    sample = pd.read_csv(OFFICIAL / "prediction" / "prediction.csv")
    assert sample.iloc[:, 0].astype(str).tolist() == meta_test.index.astype(str).tolist()
    sample.iloc[:, 1] = encoder.inverse_transform(current_test.argmax(axis=1))
    sample.to_csv(OUT / "prediction_neuronal_experts.csv", index=False)


if __name__ == "__main__":
    main()
