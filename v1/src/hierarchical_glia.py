from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train_model import (
    LABEL,
    N_FOLDS,
    OFFICIAL,
    OUT,
    SEED,
    expression_features,
    load_data,
)


EXTERNAL_REFERENCE = Path(__file__).resolve().parents[1] / "cache" / "external_glia_reference.npz"
USE_EXTERNAL_REFERENCE = os.environ.get("HACKATHON_USE_EXTERNAL_REFERENCE", "1") != "0"
FAMILIES = {
    "oligodendrocyte_lineage": [
        "oligodendrocyte_1",
        "oligodendrocyte_2",
        "oligodendrocyte_precursor_cell",
        "oligodendrocyte_progenitor_1",
        "oligodendrocyte_progenitor_2",
    ],
    "astrocyte": ["astrocyte_1", "astrocyte_2"],
    "vascular_meningeal": [
        "endothelial",
        "pericyte",
        "meninges_1",
        "meninges_2",
        "meninges_3",
    ],
    "peripheral_glia": ["peripheral_glia", "Schwann_cell"],
}
COARSE_GROUPS = {
    "oligodendrocyte_lineage": FAMILIES["oligodendrocyte_lineage"],
    "astrocyte": FAMILIES["astrocyte"],
    "vascular_meningeal": FAMILIES["vascular_meningeal"],
    "immune": ["microglia"],
    "peripheral_glia": FAMILIES["peripheral_glia"],
    "ependymal": ["ependymal"],
}


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
    }


def section_spatial_features(meta_train: pd.DataFrame, meta_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Unsupervised, section-relative morphology and coordinate features."""
    both = pd.concat([meta_train, meta_test], axis=0).copy()
    result = pd.DataFrame(index=both.index)
    group = both.groupby("Section_ID", dropna=False)
    for column in ["center_x", "center_y", "volume"]:
        values = pd.to_numeric(both[column], errors="coerce").fillna(0.0)
        means = group[column].transform("mean")
        stds = group[column].transform("std").replace(0, np.nan).fillna(1.0)
        result[f"{column}_section_z"] = (values - means) / stds
        result[f"{column}_section_rank"] = group[column].rank(pct=True).fillna(0.5)
    result["abs_x_section_z"] = result["center_x_section_z"].abs()
    result["abs_y_section_z"] = result["center_y_section_z"].abs()
    result["radius_section"] = np.sqrt(
        result["center_x_section_z"] ** 2 + result["center_y_section_z"] ** 2
    )
    result["log_volume"] = np.log1p(pd.to_numeric(both["volume"], errors="coerce").clip(lower=0).fillna(0))
    values = result.to_numpy(dtype=np.float32)
    return values[: len(meta_train)], values[len(meta_train) :]


def build_internal_features(
    counts_train: pd.DataFrame,
    counts_test: pd.DataFrame,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expr_train, expr_test = expression_features(counts_train, counts_test)
    spatial_train, spatial_test = section_spatial_features(meta_train, meta_test)
    categorical = ["Region", "Segment", "Gender", "AP_position", "Datasets"]
    both_meta = pd.concat([meta_train[categorical], meta_test[categorical]], axis=0)
    dummy = pd.get_dummies(both_meta.fillna("__MISSING__").astype(str), dtype=np.float32).to_numpy()
    dummy_train, dummy_test = dummy[: len(meta_train)], dummy[len(meta_train) :]
    full_train = np.hstack([expr_train, spatial_train, dummy_train]).astype(np.float32)
    full_test = np.hstack([expr_test, spatial_test, dummy_test]).astype(np.float32)
    return expr_train, expr_test, full_train, full_test


def sqrt_balancing_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    target = counts.max()
    return np.sqrt(target / counts[y]).astype(np.float32)


def aligned_probability(model, values: np.ndarray, n_local_classes: int) -> np.ndarray:
    raw = model.predict_proba(values)
    probability = np.zeros((len(values), n_local_classes), dtype=np.float32)
    probability[:, np.asarray(model.classes_, dtype=int)] = raw
    return probability


def load_external_reference(
    family_names: list[str],
    family_lookup: dict[str, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not USE_EXTERNAL_REFERENCE or not EXTERNAL_REFERENCE.exists():
        return None
    stored = np.load(EXTERNAL_REFERENCE, allow_pickle=False)
    labels = stored["labels"].astype(str)
    mask = np.isin(labels, family_names)
    if not mask.any():
        return None
    external_y = np.asarray([family_lookup[label] for label in labels[mask]], dtype=int)
    return stored["features"][mask].astype(np.float32), external_y


def fit_family_models(
    family_name: str,
    family_names: list[str],
    family_indices: np.ndarray,
    encoder: LabelEncoder,
    y: np.ndarray,
    expr_train: np.ndarray,
    expr_test: np.ndarray,
    full_train: np.ndarray,
    full_test: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    family_lookup = {name: i for i, name in enumerate(family_names)}
    global_to_local = {global_index: family_lookup[encoder.classes_[global_index]] for global_index in family_indices}
    family_mask = np.isin(y, family_indices)
    external = load_external_reference(family_names, family_lookup)
    n_local = len(family_names)
    oof_internal = np.zeros((len(y), n_local), dtype=np.float32)
    test_internal = np.zeros((len(full_test), n_local), dtype=np.float32)
    oof_reference = np.zeros((len(y), n_local), dtype=np.float32)
    test_reference = np.zeros((len(full_test), n_local), dtype=np.float32)
    fold_rows = []

    for fold, (fit_idx, val_idx) in enumerate(splits, 1):
        family_fit = fit_idx[family_mask[fit_idx]]
        local_y = np.asarray([global_to_local[value] for value in y[family_fit]], dtype=int)
        family_val = val_idx[family_mask[val_idx]]
        local_val_y = np.asarray([global_to_local[value] for value in y[family_val]], dtype=int)

        tree = LGBMClassifier(
            objective="multiclass" if n_local > 2 else "binary",
            n_estimators=550,
            learning_rate=0.025,
            num_leaves=14 if n_local > 2 else 10,
            min_child_samples=14,
            colsample_bytree=0.85,
            reg_lambda=4.0,
            reg_alpha=0.15,
            random_state=SEED + 301 * fold + len(family_names),
            n_jobs=-1,
            verbosity=-1,
        )
        tree.fit(
            full_train[family_fit],
            local_y,
            sample_weight=sqrt_balancing_weights(local_y),
            eval_set=[(full_train[family_val], local_val_y)],
            callbacks=[lgb.early_stopping(65, verbose=False)],
        )

        scaler = StandardScaler()
        scaled_fit = scaler.fit_transform(full_train[family_fit])
        linear = LogisticRegression(C=0.35, max_iter=2200, solver="lbfgs")
        linear.fit(scaled_fit, local_y, sample_weight=sqrt_balancing_weights(local_y))

        val_tree = aligned_probability(tree, full_train[val_idx], n_local)
        val_linear = aligned_probability(linear, scaler.transform(full_train[val_idx]), n_local)
        test_tree = aligned_probability(tree, full_test, n_local)
        test_linear = aligned_probability(linear, scaler.transform(full_test), n_local)
        val_internal = 0.65 * val_tree + 0.35 * val_linear
        test_internal_fold = 0.65 * test_tree + 0.35 * test_linear
        oof_internal[val_idx] = val_internal
        test_internal += test_internal_fold / N_FOLDS

        external_used = 0
        if external is not None:
            external_x, external_y = external
            combined_x = np.vstack([expr_train[family_fit], external_x])
            combined_y = np.concatenate([local_y, external_y])
            weights = np.concatenate([
                sqrt_balancing_weights(local_y),
                0.35 * sqrt_balancing_weights(external_y),
            ])
            reference = LGBMClassifier(
                objective="multiclass" if n_local > 2 else "binary",
                n_estimators=650,
                learning_rate=0.025,
                num_leaves=18,
                min_child_samples=20,
                colsample_bytree=0.9,
                reg_lambda=5.0,
                reg_alpha=0.2,
                random_state=SEED + 701 * fold + len(family_names),
                n_jobs=-1,
                verbosity=-1,
            )
            reference.fit(combined_x, combined_y, sample_weight=weights)
            val_reference = aligned_probability(reference, expr_train[val_idx], n_local)
            test_reference_fold = aligned_probability(reference, expr_test, n_local)
            oof_reference[val_idx] = val_reference
            test_reference += test_reference_fold / N_FOLDS
            external_used = len(external_y)

        family_prediction = val_internal[family_mask[val_idx]].argmax(axis=1)
        fold_rows.append({
            "fold": fold,
            "internal_family_accuracy": float(accuracy_score(local_val_y, family_prediction)),
            "validation_family_cells": int(len(family_val)),
            "external_reference_cells": int(external_used),
            "tree_best_iteration": int(tree.best_iteration_),
        })
        print(json.dumps({"family": family_name, **fold_rows[-1]}), flush=True)

    external_weight = 0.0
    weight_scores = []
    if external is not None:
        all_local_y = np.asarray([global_to_local[value] for value in y[family_mask]], dtype=int)
        for weight in np.linspace(0.0, 1.0, 11):
            blended = (1.0 - weight) * oof_internal + weight * oof_reference
            score = float(accuracy_score(all_local_y, blended[family_mask].argmax(axis=1)))
            weight_scores.append({"external_weight": float(weight), "family_accuracy": score})
        external_weight = max(weight_scores, key=lambda row: (row["family_accuracy"], -abs(row["external_weight"] - 0.6)))["external_weight"]
    oof = (1.0 - external_weight) * oof_internal + external_weight * oof_reference
    test = (1.0 - external_weight) * test_internal + external_weight * test_reference
    for row, (_, val_idx) in zip(fold_rows, splits):
        family_val = val_idx[family_mask[val_idx]]
        local_val_y = np.asarray([global_to_local[value] for value in y[family_val]], dtype=int)
        row["selected_blend_family_accuracy"] = float(
            accuracy_score(local_val_y, oof[family_val].argmax(axis=1))
        )
    report = {
        "folds": fold_rows,
        "external_reference": external is not None,
        "selected_external_probability_weight": float(external_weight),
        "external_weight_scores": weight_scores,
    }
    return oof, test, report


def fit_coarse_router(
    encoder: LabelEncoder,
    y: np.ndarray,
    expr_train: np.ndarray,
    expr_test: np.ndarray,
    full_train: np.ndarray,
    full_test: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict]:
    group_indices = [np.asarray([np.where(encoder.classes_ == name)[0][0] for name in names]) for names in COARSE_GROUPS.values()]
    global_to_group = {
        int(global_index): group_index
        for group_index, indices in enumerate(group_indices)
        for global_index in indices
    }
    label_to_group = {
        label: group_index
        for group_index, labels in enumerate(COARSE_GROUPS.values())
        for label in labels
    }
    glia_mask = np.isin(y, np.concatenate(group_indices))
    n_groups = len(group_indices)
    oof_internal = np.zeros((len(y), n_groups), dtype=np.float32)
    test_internal = np.zeros((len(full_test), n_groups), dtype=np.float32)
    oof_reference = np.zeros((len(y), n_groups), dtype=np.float32)
    test_reference = np.zeros((len(full_test), n_groups), dtype=np.float32)
    external = None
    if USE_EXTERNAL_REFERENCE and EXTERNAL_REFERENCE.exists():
        stored = np.load(EXTERNAL_REFERENCE, allow_pickle=False)
        external_labels = stored["labels"].astype(str)
        external_mask = np.isin(external_labels, list(label_to_group))
        external = (
            stored["features"][external_mask].astype(np.float32),
            np.asarray([label_to_group[label] for label in external_labels[external_mask]], dtype=int),
        )
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(splits, 1):
        router_fit = fit_idx[glia_mask[fit_idx]]
        router_y = np.asarray([global_to_group[value] for value in y[router_fit]], dtype=int)
        router_val = val_idx[glia_mask[val_idx]]
        router_val_y = np.asarray([global_to_group[value] for value in y[router_val]], dtype=int)
        tree = LGBMClassifier(
            objective="multiclass",
            n_estimators=520,
            learning_rate=0.025,
            num_leaves=14,
            min_child_samples=18,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            reg_alpha=0.2,
            random_state=SEED + 900 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        tree.fit(
            full_train[router_fit],
            router_y,
            sample_weight=sqrt_balancing_weights(router_y),
            eval_set=[(full_train[router_val], router_val_y)],
            callbacks=[lgb.early_stopping(65, verbose=False)],
        )
        scaler = StandardScaler()
        fit_scaled = scaler.fit_transform(full_train[router_fit])
        linear = LogisticRegression(C=0.35, max_iter=2200, solver="lbfgs")
        linear.fit(fit_scaled, router_y, sample_weight=sqrt_balancing_weights(router_y))
        oof_internal[val_idx] = (
            0.7 * aligned_probability(tree, full_train[val_idx], n_groups)
            + 0.3 * aligned_probability(linear, scaler.transform(full_train[val_idx]), n_groups)
        )
        test_internal += (
            0.7 * aligned_probability(tree, full_test, n_groups)
            + 0.3 * aligned_probability(linear, scaler.transform(full_test), n_groups)
        ) / N_FOLDS
        external_used = 0
        if external is not None:
            external_x, external_y = external
            combined_x = np.vstack([expr_train[router_fit], external_x])
            combined_y = np.concatenate([router_y, external_y])
            weights = np.concatenate([
                sqrt_balancing_weights(router_y),
                0.35 * sqrt_balancing_weights(external_y),
            ])
            reference = LGBMClassifier(
                objective="multiclass",
                n_estimators=650,
                learning_rate=0.025,
                num_leaves=18,
                min_child_samples=20,
                colsample_bytree=0.9,
                reg_lambda=5.0,
                reg_alpha=0.2,
                random_state=SEED + 1900 + fold,
                n_jobs=-1,
                verbosity=-1,
            )
            reference.fit(combined_x, combined_y, sample_weight=weights)
            oof_reference[val_idx] = aligned_probability(reference, expr_train[val_idx], n_groups)
            test_reference += aligned_probability(reference, expr_test, n_groups) / N_FOLDS
            external_used = len(external_y)
        rows.append({
            "fold": fold,
            "internal_coarse_glia_accuracy": float(accuracy_score(router_val_y, oof_internal[router_val].argmax(axis=1))),
            "validation_glia_cells": int(len(router_val)),
            "external_reference_cells": int(external_used),
            "tree_best_iteration": int(tree.best_iteration_),
        })
        print(json.dumps({"family": "coarse_router", **rows[-1]}), flush=True)

    external_weight = 0.0
    weight_scores = []
    if external is not None:
        all_router_y = np.asarray([global_to_group[value] for value in y[glia_mask]], dtype=int)
        for weight in np.linspace(0.0, 1.0, 11):
            blended = (1.0 - weight) * oof_internal + weight * oof_reference
            score = float(accuracy_score(all_router_y, blended[glia_mask].argmax(axis=1)))
            weight_scores.append({"external_weight": float(weight), "coarse_glia_accuracy": score})
        external_weight = max(weight_scores, key=lambda row: (row["coarse_glia_accuracy"], -abs(row["external_weight"] - 0.6)))["external_weight"]
    oof = (1.0 - external_weight) * oof_internal + external_weight * oof_reference
    test = (1.0 - external_weight) * test_internal + external_weight * test_reference
    for row, (_, val_idx) in zip(rows, splits):
        router_val = val_idx[glia_mask[val_idx]]
        router_val_y = np.asarray([global_to_group[value] for value in y[router_val]], dtype=int)
        row["selected_blend_coarse_glia_accuracy"] = float(
            accuracy_score(router_val_y, oof[router_val].argmax(axis=1))
        )
    report = {
        "folds": rows,
        "groups": list(COARSE_GROUPS),
        "external_reference": external is not None,
        "selected_external_probability_weight": float(external_weight),
        "external_weight_scores": weight_scores,
    }
    return oof, test, group_indices, report


def redistribute_coarse_groups(
    base: np.ndarray,
    router: np.ndarray,
    group_indices: list[np.ndarray],
    family_gate: float,
    confidence_cap: float,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = base.copy()
    all_indices = np.concatenate(group_indices)
    glia_mass = base[:, all_indices].sum(axis=1)
    confidence = base.max(axis=1)
    top_in_glia = np.isin(base.argmax(axis=1), all_indices)
    selected = (glia_mass >= family_gate) & (top_in_glia | (confidence <= confidence_cap))
    if not selected.any():
        return result, selected
    current_group_probability = np.column_stack([base[:, indices].sum(axis=1) for indices in group_indices])
    current_group_probability = current_group_probability[selected] / np.maximum(glia_mass[selected, None], 1e-12)
    corrected_group_probability = (1.0 - weight) * current_group_probability + weight * router[selected]
    for group_number, indices in enumerate(group_indices):
        within = base[np.ix_(selected, indices)]
        within /= np.maximum(within.sum(axis=1, keepdims=True), 1e-12)
        new_mass = glia_mass[selected] * corrected_group_probability[:, group_number]
        result[np.ix_(selected, indices)] = within * new_mass[:, None]
    return result, selected


def choose_router_gate(
    base: np.ndarray,
    router: np.ndarray,
    group_indices: list[np.ndarray],
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[dict, np.ndarray]:
    base_correct = base.argmax(axis=1) == y
    candidates = []
    for gate in [0.30, 0.45, 0.60, 0.75]:
        for cap in [0.55, 0.70, 0.85, 1.01]:
            for weight in [0.15, 0.30, 0.50, 0.70]:
                probability, selected = redistribute_coarse_groups(base, router, group_indices, gate, cap, weight)
                pred = probability.argmax(axis=1)
                fold_delta = [
                    int((pred[fold_ids == fold] == y[fold_ids == fold]).sum() - base_correct[fold_ids == fold].sum())
                    for fold in range(N_FOLDS)
                ]
                candidates.append({
                    "gate": gate,
                    "confidence_cap": cap,
                    "weight": weight,
                    "delta_correct": int((pred == y).sum() - base_correct.sum()),
                    "fold_delta_correct": fold_delta,
                    "selected_cells": int(selected.sum()),
                    "stable": sum(delta >= 0 for delta in fold_delta) >= 3 and min(fold_delta) >= -2,
                })
    stable = [candidate for candidate in candidates if candidate["stable"]]
    best = max(stable if stable else candidates, key=lambda row: (row["delta_correct"], -row["selected_cells"], -row["weight"]))
    if best["delta_correct"] < 2:
        best = {
            "gate": 1.01, "confidence_cap": 0.0, "weight": 0.0,
            "delta_correct": 0, "fold_delta_correct": [0] * N_FOLDS,
            "selected_cells": 0, "stable": True, "skipped": True,
        }
    probability, _ = redistribute_coarse_groups(
        base, router, group_indices, best["gate"], best["confidence_cap"], best["weight"]
    )
    return best, probability


def redistribute_family(
    base: np.ndarray,
    specialist: np.ndarray,
    family_indices: np.ndarray,
    family_gate: float,
    confidence_cap: float,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = base.copy()
    family_mass = base[:, family_indices].sum(axis=1)
    confidence = base.max(axis=1)
    top_in_family = np.isin(base.argmax(axis=1), family_indices)
    selected = (family_mass >= family_gate) & (top_in_family | (confidence <= confidence_cap))
    if not selected.any():
        return result, selected
    original = base[np.ix_(selected, family_indices)] / np.maximum(family_mass[selected, None], 1e-12)
    corrected = (1.0 - weight) * original + weight * specialist[selected]
    result[np.ix_(selected, family_indices)] = corrected * family_mass[selected, None]
    return result, selected


def choose_gate(
    base: np.ndarray,
    specialist: np.ndarray,
    family_indices: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[dict, np.ndarray]:
    base_pred = base.argmax(axis=1)
    base_correct = base_pred == y
    candidates = []
    for gate in [0.30, 0.45, 0.60, 0.75]:
        for cap in [0.55, 0.70, 0.85, 1.01]:
            for weight in [0.25, 0.50, 0.75, 1.00]:
                probability, selected = redistribute_family(base, specialist, family_indices, gate, cap, weight)
                pred = probability.argmax(axis=1)
                total_delta = int((pred == y).sum() - base_correct.sum())
                fold_delta = [
                    int(((pred[idx] == y[idx]).sum() - base_correct[idx].sum()))
                    for idx in [fold_ids == fold for fold in range(N_FOLDS)]
                ]
                stable = sum(delta >= 0 for delta in fold_delta) >= 3 and min(fold_delta) >= -2
                candidates.append({
                    "gate": gate,
                    "confidence_cap": cap,
                    "weight": weight,
                    "delta_correct": total_delta,
                    "fold_delta_correct": fold_delta,
                    "selected_cells": int(selected.sum()),
                    "stable": stable,
                })
    stable = [candidate for candidate in candidates if candidate["stable"]]
    pool = stable if stable else candidates
    best = max(
        pool,
        key=lambda row: (
            row["delta_correct"],
            -row["selected_cells"],
            -row["weight"],
        ),
    )
    if best["delta_correct"] < 2:
        best = {
            "gate": 1.01,
            "confidence_cap": 0.0,
            "weight": 0.0,
            "delta_correct": 0,
            "fold_delta_correct": [0] * N_FOLDS,
            "selected_cells": 0,
            "stable": True,
            "skipped": True,
        }
    probability, _ = redistribute_family(
        base,
        specialist,
        family_indices,
        best["gate"],
        best["confidence_cap"],
        best["weight"],
    )
    return best, probability


def confusion_rows(y: np.ndarray, probability: np.ndarray, encoder: LabelEncoder, limit: int = 15) -> list[dict]:
    pred = probability.argmax(axis=1)
    pairs = Counter(zip(y[pred != y], pred[pred != y]))
    return [
        {"count": int(count), "truth": encoder.classes_[truth], "prediction": encoder.classes_[guess]}
        for (truth, guess), count in pairs.most_common(limit)
    ]


def main() -> None:
    counts_train, counts_test, meta_train, meta_test = load_data()
    encoder = LabelEncoder().fit(meta_train[LABEL].astype(str))
    y = encoder.transform(meta_train[LABEL].astype(str))
    class_lookup = {name: index for index, name in enumerate(encoder.classes_)}
    expr_train, expr_test, full_train, full_test = build_internal_features(
        counts_train, counts_test, meta_train, meta_test
    )
    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(full_train, y))
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        fold_ids[val_idx] = fold

    base_stored = np.load(OUT / "prior_postprocess_probabilities.npz")
    current_oof = base_stored["oof"].copy()
    current_test = base_stored["test"].copy()
    baseline = metrics(y, current_oof)
    baseline_confusions = confusion_rows(y, current_oof, encoder)
    family_reports = {}

    router_oof, router_test, router_groups, router_training = fit_coarse_router(
        encoder, y, expr_train, expr_test, full_train, full_test, splits
    )
    router_gate, current_oof = choose_router_gate(
        current_oof, router_oof, router_groups, y, fold_ids
    )
    current_test, router_selected_test = redistribute_coarse_groups(
        current_test,
        router_test,
        router_groups,
        router_gate["gate"],
        router_gate["confidence_cap"],
        router_gate["weight"],
    )
    family_reports["coarse_router"] = {
        "training": router_training,
        "gate": router_gate,
        "test_cells_modified": int(router_selected_test.sum()),
        "cumulative_metrics": metrics(y, current_oof),
    }

    for family_name, family_names in FAMILIES.items():
        family_indices = np.asarray([class_lookup[name] for name in family_names], dtype=int)
        specialist_oof, specialist_test, training_report = fit_family_models(
            family_name,
            family_names,
            family_indices,
            encoder,
            y,
            expr_train,
            expr_test,
            full_train,
            full_test,
            splits,
        )
        gate, updated_oof = choose_gate(current_oof, specialist_oof, family_indices, y, fold_ids)
        updated_test, selected_test = redistribute_family(
            current_test,
            specialist_test,
            family_indices,
            gate["gate"],
            gate["confidence_cap"],
            gate["weight"],
        )
        current_oof, current_test = updated_oof, updated_test
        family_reports[family_name] = {
            "labels": family_names,
            "training": training_report,
            "gate": gate,
            "test_cells_modified": int(selected_test.sum()),
            "cumulative_metrics": metrics(y, current_oof),
        }
        print(json.dumps({"family": family_name, **family_reports[family_name]}, indent=2), flush=True)

    summary = {
        "baseline": baseline,
        "hierarchical_glia": metrics(y, current_oof),
        "delta_correct": int((current_oof.argmax(axis=1) == y).sum() - (base_stored["oof"].argmax(axis=1) == y).sum()),
        "external_reference_file": str(EXTERNAL_REFERENCE),
        "external_reference_used": USE_EXTERNAL_REFERENCE and EXTERNAL_REFERENCE.exists(),
        "family_reports": family_reports,
        "baseline_top_confusions": baseline_confusions,
        "final_top_confusions": confusion_rows(y, current_oof, encoder),
    }
    print(json.dumps(summary, indent=2), flush=True)
    variant = "external" if summary["external_reference_used"] else "internal"
    (OUT / f"hierarchical_glia_{variant}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / f"hierarchical_glia_{variant}_probabilities.npz", oof=current_oof, test=current_test, y=y)
    if variant == "external":
        (OUT / "hierarchical_glia_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        np.savez_compressed(OUT / "hierarchical_glia_probabilities.npz", oof=current_oof, test=current_test, y=y)

    sample = pd.read_csv(OFFICIAL / "prediction" / "prediction.csv")
    assert sample.iloc[:, 0].astype(str).tolist() == meta_test.index.astype(str).tolist()
    sample.iloc[:, 1] = encoder.inverse_transform(current_test.argmax(axis=1))
    sample.to_csv(OUT / f"prediction_hierarchical_glia_{variant}.csv", index=False)
    if variant == "external":
        sample.to_csv(OUT / "prediction_hierarchical_glia.csv", index=False)


if __name__ == "__main__":
    main()
