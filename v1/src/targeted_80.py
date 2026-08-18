from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from hierarchical_glia import build_internal_features, sqrt_balancing_weights
from train_model import LABEL, N_FOLDS, OFFICIAL, OUT, SEED, load_data


BASE_PROBABILITIES = OUT / "hierarchical_glia_external_probabilities.npz"
EXTERNAL_REFERENCE = Path(__file__).resolve().parents[1] / "cache" / "external_glia_reference.npz"
PAIR_SPECS = [
    ("oligo_p2_vs_oligo1", "oligodendrocyte_progenitor_2", "oligodendrocyte_1"),
    ("oligo_p2_vs_oligo2", "oligodendrocyte_progenitor_2", "oligodendrocyte_2"),
    ("oligo_p1_vs_precursor", "oligodendrocyte_progenitor_1", "oligodendrocyte_precursor_cell"),
]


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


def aligned_binary_probability(model, values: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(values)
    probability = np.zeros((len(values), 2), dtype=np.float32)
    probability[:, np.asarray(model.classes_, dtype=int)] = raw
    return probability


@dataclass
class PairProbability:
    name: str
    labels: tuple[str, str]
    indices: np.ndarray
    oof: np.ndarray
    test: np.ndarray
    training: dict


def fit_pair_expert(
    name: str,
    labels: tuple[str, str],
    encoder: LabelEncoder,
    y: np.ndarray,
    expr_train: np.ndarray,
    expr_test: np.ndarray,
    full_train: np.ndarray,
    full_test: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    external_features: np.ndarray,
    external_labels: np.ndarray,
) -> PairProbability:
    indices = np.asarray([int(np.where(encoder.classes_ == label)[0][0]) for label in labels])
    global_to_local = {int(index): local for local, index in enumerate(indices)}
    pair_mask = np.isin(y, indices)
    ext_mask = np.isin(external_labels, labels)
    ext_x = external_features[ext_mask]
    ext_y = np.asarray([labels.index(label) for label in external_labels[ext_mask]], dtype=int)

    oof_internal = np.zeros((len(y), 2), dtype=np.float32)
    test_internal = np.zeros((len(full_test), 2), dtype=np.float32)
    oof_external = np.zeros((len(y), 2), dtype=np.float32)
    test_external = np.zeros((len(full_test), 2), dtype=np.float32)
    fold_rows = []

    for fold, (fit_idx, val_idx) in enumerate(splits, 1):
        pair_fit = fit_idx[pair_mask[fit_idx]]
        pair_val = val_idx[pair_mask[val_idx]]
        local_y = np.asarray([global_to_local[int(value)] for value in y[pair_fit]], dtype=int)
        local_val_y = np.asarray([global_to_local[int(value)] for value in y[pair_val]], dtype=int)
        weights = sqrt_balancing_weights(local_y)

        tree = LGBMClassifier(
            objective="binary",
            n_estimators=520,
            learning_rate=0.022,
            num_leaves=10,
            min_child_samples=16,
            colsample_bytree=0.82,
            reg_lambda=6.0,
            reg_alpha=0.25,
            random_state=SEED + 1103 * fold + int(indices.sum()),
            n_jobs=-1,
            verbosity=-1,
        )
        tree.fit(
            full_train[pair_fit],
            local_y,
            sample_weight=weights,
            eval_set=[(full_train[pair_val], local_val_y)],
            callbacks=[lgb.early_stopping(70, verbose=False)],
        )

        scaler = StandardScaler()
        scaled_fit = scaler.fit_transform(full_train[pair_fit])
        linear = LogisticRegression(C=0.22, max_iter=2400, solver="lbfgs")
        linear.fit(scaled_fit, local_y, sample_weight=weights)

        oof_internal[val_idx] = (
            0.65 * aligned_binary_probability(tree, full_train[val_idx])
            + 0.35 * aligned_binary_probability(linear, scaler.transform(full_train[val_idx]))
        )
        test_internal += (
            0.65 * aligned_binary_probability(tree, full_test)
            + 0.35 * aligned_binary_probability(linear, scaler.transform(full_test))
        ) / N_FOLDS

        combined_x = np.vstack([expr_train[pair_fit], ext_x])
        combined_y = np.concatenate([local_y, ext_y])
        combined_weight = np.concatenate([
            sqrt_balancing_weights(local_y),
            0.30 * sqrt_balancing_weights(ext_y),
        ])
        reference = LGBMClassifier(
            objective="binary",
            n_estimators=620,
            learning_rate=0.022,
            num_leaves=14,
            min_child_samples=20,
            colsample_bytree=0.88,
            reg_lambda=6.0,
            reg_alpha=0.25,
            random_state=SEED + 1901 * fold + int(indices.sum()),
            n_jobs=-1,
            verbosity=-1,
        )
        reference.fit(combined_x, combined_y, sample_weight=combined_weight)
        oof_external[val_idx] = aligned_binary_probability(reference, expr_train[val_idx])
        test_external += aligned_binary_probability(reference, expr_test) / N_FOLDS

        fold_rows.append({
            "fold": fold,
            "fit_pair_cells": int(len(pair_fit)),
            "validation_pair_cells": int(len(pair_val)),
            "external_pair_cells": int(len(ext_y)),
            "tree_best_iteration": int(tree.best_iteration_),
            "internal_pair_accuracy": float(
                accuracy_score(local_val_y, oof_internal[pair_val].argmax(axis=1))
            ),
            "external_pair_accuracy": float(
                accuracy_score(local_val_y, oof_external[pair_val].argmax(axis=1))
            ),
        })
        print(json.dumps({"stage": "pair_training", "pair": name, **fold_rows[-1]}), flush=True)

    pair_truth = np.asarray([global_to_local[int(value)] for value in y[pair_mask]], dtype=int)
    blend_scores = []
    for external_weight in np.linspace(0.0, 0.8, 9):
        blended = (1.0 - external_weight) * oof_internal + external_weight * oof_external
        blend_scores.append({
            "external_weight": float(external_weight),
            "pair_accuracy": float(accuracy_score(pair_truth, blended[pair_mask].argmax(axis=1))),
        })
    best_weight = max(
        blend_scores,
        key=lambda row: (row["pair_accuracy"], -abs(row["external_weight"] - 0.4)),
    )["external_weight"]
    oof = (1.0 - best_weight) * oof_internal + best_weight * oof_external
    test = (1.0 - best_weight) * test_internal + best_weight * test_external
    training = {
        "folds": fold_rows,
        "blend_scores": blend_scores,
        "selected_external_weight": float(best_weight),
        "selected_pair_accuracy": float(accuracy_score(pair_truth, oof[pair_mask].argmax(axis=1))),
    }
    return PairProbability(name, labels, indices, oof, test, training)


def apply_pair_correction(
    base: np.ndarray,
    specialist: np.ndarray,
    pair_indices: np.ndarray,
    mass_gate: float,
    confidence_cap: float,
    specialist_confidence: float,
    base_margin_cap: float,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = base.copy()
    pair_mass = base[:, pair_indices].sum(axis=1)
    pair_base = base[:, pair_indices] / np.maximum(pair_mass[:, None], 1e-12)
    top_in_pair = np.isin(base.argmax(axis=1), pair_indices)
    selected = (
        (pair_mass >= mass_gate)
        & (top_in_pair | (base.max(axis=1) <= confidence_cap))
        & (specialist.max(axis=1) >= specialist_confidence)
        & (np.abs(pair_base[:, 0] - pair_base[:, 1]) <= base_margin_cap)
    )
    if selected.any():
        corrected = (1.0 - weight) * pair_base[selected] + weight * specialist[selected]
        result[np.ix_(selected, pair_indices)] = corrected * pair_mass[selected, None]
    return result, selected


def choose_pair_gate(
    base: np.ndarray,
    pair: PairProbability,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[dict, np.ndarray]:
    candidates = []
    for mass_gate in [0.25, 0.40, 0.55, 0.70]:
        for confidence_cap in [0.55, 0.70, 0.85, 1.01]:
            for specialist_confidence in [0.55, 0.65, 0.75, 0.85]:
                for base_margin_cap in [0.15, 0.30, 0.50, 1.01]:
                    for weight in [0.50, 0.75, 1.00]:
                        probability, selected = apply_pair_correction(
                            base,
                            pair.oof,
                            pair.indices,
                            mass_gate,
                            confidence_cap,
                            specialist_confidence,
                            base_margin_cap,
                            weight,
                        )
                        fold_delta = fold_delta_correct(y, base, probability, fold_ids)
                        delta = metrics(y, probability)["correct"] - metrics(y, base)["correct"]
                        candidates.append({
                            "mass_gate": mass_gate,
                            "confidence_cap": confidence_cap,
                            "specialist_confidence": specialist_confidence,
                            "base_margin_cap": base_margin_cap,
                            "weight": weight,
                            "delta_correct": int(delta),
                            "fold_delta_correct": fold_delta,
                            "selected_cells": int(selected.sum()),
                            "stable": sum(value >= 0 for value in fold_delta) >= 3 and min(fold_delta) >= -2,
                        })
    stable = [row for row in candidates if row["stable"]]
    best = max(
        stable if stable else candidates,
        key=lambda row: (row["delta_correct"], min(row["fold_delta_correct"]), -row["selected_cells"]),
    )
    if best["delta_correct"] < 2:
        best = {
            "mass_gate": 1.01,
            "confidence_cap": 0.0,
            "specialist_confidence": 1.01,
            "base_margin_cap": 0.0,
            "weight": 0.0,
            "delta_correct": 0,
            "fold_delta_correct": [0] * N_FOLDS,
            "selected_cells": 0,
            "stable": True,
            "skipped": True,
        }
    probability, _ = apply_pair_correction(
        base,
        pair.oof,
        pair.indices,
        best["mass_gate"],
        best["confidence_cap"],
        best["specialist_confidence"],
        best["base_margin_cap"],
        best["weight"],
    )
    return best, probability


def standardized_section_coordinates(
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    both = pd.concat([meta_train, meta_test], axis=0).copy()
    result = np.zeros((len(both), 3), dtype=np.float32)
    for column_number, column in enumerate(["center_x", "center_y"]):
        values = pd.to_numeric(both[column], errors="coerce").fillna(0.0)
        grouped = values.groupby(both["Section_ID"], dropna=False)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan).fillna(1.0)
        result[:, column_number] = ((values - mean) / std).to_numpy(dtype=np.float32)
    volume = np.log1p(pd.to_numeric(both["volume"], errors="coerce").fillna(0).clip(lower=0))
    volume_group = volume.groupby(both["Section_ID"], dropna=False)
    volume_z = (volume - volume_group.transform("mean")) / volume_group.transform("std").replace(0, np.nan).fillna(1.0)
    result[:, 2] = 0.20 * volume_z.to_numpy(dtype=np.float32)
    return result[: len(meta_train)], result[len(meta_train) :]


def section_knn_prior(
    train_coordinates: np.ndarray,
    test_coordinates: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    k: int,
    embedding_name: str,
    distance_metric: str,
    smoothing: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    global_prior = np.bincount(y, minlength=n_classes).astype(np.float64)
    global_prior /= global_prior.sum()
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    test = np.zeros((len(meta_test), n_classes), dtype=np.float32)
    section_train = meta_train["Section_ID"].astype(str).to_numpy()
    section_test = meta_test["Section_ID"].astype(str).to_numpy()
    fold_neighbor_counts = []

    def build_for_targets(target_indices, target_coordinates, target_sections, source_indices):
        output = np.zeros((len(target_indices), n_classes), dtype=np.float32)
        neighbor_counts = []
        for section in np.unique(target_sections[target_indices]):
            target_local = np.flatnonzero(target_sections[target_indices] == section)
            source_local = source_indices[section_train[source_indices] == section]
            if len(source_local) == 0:
                output[target_local] = global_prior
                neighbor_counts.extend([0] * len(target_local))
                continue
            distances = cdist(
                target_coordinates[target_indices[target_local]],
                train_coordinates[source_local],
                metric=distance_metric,
            )
            distances = np.nan_to_num(distances, nan=2.0, posinf=2.0, neginf=0.0)
            effective_k = min(k, len(source_local))
            nearest_order = np.argpartition(distances, effective_k - 1, axis=1)[:, :effective_k]
            nearest_distance = np.take_along_axis(distances, nearest_order, axis=1)
            local_scale = np.maximum(np.median(nearest_distance, axis=1, keepdims=True), 0.10)
            weights = np.exp(-nearest_distance / local_scale)
            for row_number, positions in enumerate(nearest_order):
                labels = y[source_local[positions]]
                class_weight = np.bincount(labels, weights=weights[row_number], minlength=n_classes)
                class_weight += smoothing * global_prior
                output[target_local[row_number]] = class_weight / class_weight.sum()
            neighbor_counts.extend([effective_k] * len(target_local))
        return output, neighbor_counts

    for fold, (fit_idx, val_idx) in enumerate(splits):
        values, counts = build_for_targets(
            np.asarray(val_idx), train_coordinates, section_train, np.asarray(fit_idx)
        )
        oof[val_idx] = values
        fold_neighbor_counts.append({
            "fold": fold + 1,
            "minimum_neighbors": int(min(counts)),
            "median_neighbors": float(np.median(counts)),
            "maximum_neighbors": int(max(counts)),
        })

    all_train = np.arange(len(meta_train))
    test_indices = np.arange(len(meta_test))
    for section in np.unique(section_test):
        target_local = np.flatnonzero(section_test == section)
        source_local = all_train[section_train == section]
        if len(source_local) == 0:
            test[target_local] = global_prior
            continue
        distances = cdist(
            test_coordinates[target_local], train_coordinates[source_local], metric=distance_metric
        )
        distances = np.nan_to_num(distances, nan=2.0, posinf=2.0, neginf=0.0)
        effective_k = min(k, len(source_local))
        nearest_order = np.argpartition(distances, effective_k - 1, axis=1)[:, :effective_k]
        nearest_distance = np.take_along_axis(distances, nearest_order, axis=1)
        local_scale = np.maximum(np.median(nearest_distance, axis=1, keepdims=True), 0.10)
        weights = np.exp(-nearest_distance / local_scale)
        for row_number, positions in enumerate(nearest_order):
            labels = y[source_local[positions]]
            class_weight = np.bincount(labels, weights=weights[row_number], minlength=n_classes)
            class_weight += smoothing * global_prior
            test[target_local[row_number]] = class_weight / class_weight.sum()
    return oof, test, {
        "embedding": embedding_name,
        "distance_metric": distance_metric,
        "k": k,
        "smoothing": smoothing,
        "fold_neighbors": fold_neighbor_counts,
    }


def section_neighbor_embeddings(
    expr_train: np.ndarray,
    expr_test: np.ndarray,
    spatial_train: np.ndarray,
    spatial_test: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """Unsupervised expression/PCA embeddings with optional spatial coordinates."""
    log_normalized_train = expr_train[:, 200:400]
    log_normalized_test = expr_test[:, 200:400]
    combined = np.vstack([log_normalized_train, log_normalized_test])
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    pca_white = PCA(n_components=32, whiten=True, random_state=SEED)
    combined_pca_white = pca_white.fit_transform(combined_scaled).astype(np.float32)
    pca_train = combined_pca_white[: len(expr_train)]
    pca_test = combined_pca_white[len(expr_train) :]
    pca_weighted = PCA(n_components=32, whiten=False, random_state=SEED)
    combined_pca_weighted = pca_weighted.fit_transform(combined_scaled).astype(np.float32)
    weighted_train = combined_pca_weighted[: len(expr_train)]
    weighted_test = combined_pca_weighted[len(expr_train) :]
    log_raw_train = expr_train[:, :200].astype(np.float32)
    log_raw_test = expr_test[:, :200].astype(np.float32)
    embeddings = [
        ("spatial_only", spatial_train, spatial_test, "euclidean"),
        ("log_normalized_cosine", log_normalized_train, log_normalized_test, "cosine"),
        ("log_raw_cosine", log_raw_train, log_raw_test, "cosine"),
        ("expression_pca_nonwhite", weighted_train, weighted_test, "euclidean"),
    ]
    for spatial_weight in [0.0, 0.35, 0.70, 1.40]:
        if spatial_weight == 0.0:
            train = pca_train
            test = pca_test
            name = "expression_pca"
        else:
            train = np.hstack([pca_train, spatial_weight * spatial_train]).astype(np.float32)
            test = np.hstack([pca_test, spatial_weight * spatial_test]).astype(np.float32)
            name = f"expression_pca_plus_spatial_{spatial_weight:.2f}"
        embeddings.append((name, train, test, "euclidean"))
    return embeddings


def apply_spatial_prior(
    base: np.ndarray,
    spatial: np.ndarray,
    method: str,
    weight: float,
    confidence_cap: float,
    spatial_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = (base.max(axis=1) <= confidence_cap) & (spatial.max(axis=1) >= spatial_confidence)
    result = base.copy()
    if not selected.any():
        return result, selected
    if method == "additive":
        result[selected] = (1.0 - weight) * base[selected] + weight * spatial[selected]
    elif method == "multiplicative":
        corrected = base[selected] * np.power(np.maximum(spatial[selected], 1e-5), weight)
        result[selected] = corrected / corrected.sum(axis=1, keepdims=True)
    else:
        raise ValueError(method)
    return result, selected


def choose_spatial_prior(
    base: np.ndarray,
    candidates: list[tuple[np.ndarray, np.ndarray, dict]],
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    rows = []
    for oof, test, metadata in candidates:
        for method, weights in [("additive", [0.05, 0.10, 0.15, 0.20, 0.30]), ("multiplicative", [0.05, 0.10, 0.20, 0.30, 0.50])]:
            for weight in weights:
                for confidence_cap in [0.45, 0.55, 0.70, 0.85, 1.01]:
                    for spatial_confidence in [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]:
                        probability, selected = apply_spatial_prior(
                            base, oof, method, weight, confidence_cap, spatial_confidence
                        )
                        delta = metrics(y, probability)["correct"] - metrics(y, base)["correct"]
                        fold_delta = fold_delta_correct(y, base, probability, fold_ids)
                        rows.append({
                            **metadata,
                            "method": method,
                            "weight": weight,
                            "confidence_cap": confidence_cap,
                            "spatial_confidence": spatial_confidence,
                            "delta_correct": int(delta),
                            "fold_delta_correct": fold_delta,
                            "selected_cells": int(selected.sum()),
                            "stable": sum(value >= 0 for value in fold_delta) >= 3 and min(fold_delta) >= -2,
                            "oof": oof,
                            "test": test,
                        })
    stable = [row for row in rows if row["stable"]]
    best = max(
        stable if stable else rows,
        key=lambda row: (row["delta_correct"], min(row["fold_delta_correct"]), -row["selected_cells"]),
    )
    if best["delta_correct"] < 2:
        report = {
            "skipped": True,
            "delta_correct": 0,
            "fold_delta_correct": [0] * N_FOLDS,
            "stable": True,
        }
        return report, base.copy(), np.zeros_like(candidates[0][1])
    probability, _ = apply_spatial_prior(
        base,
        best["oof"],
        best["method"],
        best["weight"],
        best["confidence_cap"],
        best["spatial_confidence"],
    )
    report = {key: value for key, value in best.items() if key not in {"oof", "test"}}
    return report, probability, best["test"]


def main() -> None:
    counts_train, counts_test, meta_train, meta_test = load_data()
    encoder = LabelEncoder().fit(meta_train[LABEL].astype(str))
    y = encoder.transform(meta_train[LABEL].astype(str))
    expr_train, expr_test, full_train, full_test = build_internal_features(
        counts_train, counts_test, meta_train, meta_test
    )
    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(full_train, y))
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        fold_ids[val_idx] = fold

    stored = np.load(BASE_PROBABILITIES)
    current_oof = stored["oof"].astype(np.float32).copy()
    current_test = stored["test"].astype(np.float32).copy()
    baseline_oof = current_oof.copy()
    external = np.load(EXTERNAL_REFERENCE, allow_pickle=False)
    external_features = external["features"].astype(np.float32)
    external_labels = external["labels"].astype(str)
    report = {"baseline": metrics(y, current_oof), "pair_experts": {}}

    for name, first, second in PAIR_SPECS:
        pair = fit_pair_expert(
            name,
            (first, second),
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
        gate, updated_oof = choose_pair_gate(current_oof, pair, y, fold_ids)
        updated_test, selected_test = apply_pair_correction(
            current_test,
            pair.test,
            pair.indices,
            gate["mass_gate"],
            gate["confidence_cap"],
            gate["specialist_confidence"],
            gate["base_margin_cap"],
            gate["weight"],
        )
        current_oof, current_test = updated_oof, updated_test
        report["pair_experts"][name] = {
            "labels": [first, second],
            "training": pair.training,
            "gate": gate,
            "test_cells_modified": int(selected_test.sum()),
            "cumulative_metrics": metrics(y, current_oof),
        }
        print(json.dumps({"stage": "pair_selected", "pair": name, **report["pair_experts"][name]}, indent=2), flush=True)

    report["after_pair_experts"] = metrics(y, current_oof)
    train_coordinates, test_coordinates = standardized_section_coordinates(meta_train, meta_test)
    neighbor_embeddings = section_neighbor_embeddings(
        expr_train, expr_test, train_coordinates, test_coordinates
    )
    spatial_candidates = []
    for embedding_name, neighbor_train, neighbor_test, distance_metric in neighbor_embeddings:
        for k in [3, 5, 10, 15, 20, 30, 40]:
            candidate = section_knn_prior(
                neighbor_train,
                neighbor_test,
                meta_train,
                meta_test,
                y,
                splits,
                len(encoder.classes_),
                k,
                embedding_name,
                distance_metric,
            )
            spatial_candidates.append(candidate)
            print(json.dumps({"stage": "spatial_candidate", **candidate[2]}), flush=True)

    spatial_report, spatial_oof, _ = choose_spatial_prior(
        current_oof, spatial_candidates, y, fold_ids
    )
    if spatial_report.get("skipped"):
        spatial_test = current_test.copy()
    else:
        selected_candidate = next(
            item
            for item in spatial_candidates
            if item[2]["k"] == spatial_report["k"]
            and item[2]["embedding"] == spatial_report["embedding"]
            and item[2]["distance_metric"] == spatial_report["distance_metric"]
        )
        spatial_test, selected_test = apply_spatial_prior(
            current_test,
            selected_candidate[1],
            spatial_report["method"],
            spatial_report["weight"],
            spatial_report["confidence_cap"],
            spatial_report["spatial_confidence"],
        )
        spatial_report["test_cells_modified"] = int(selected_test.sum())
    current_oof, current_test = spatial_oof, spatial_test
    report["spatial_knn"] = {
        **spatial_report,
        "cumulative_metrics": metrics(y, current_oof),
    }
    report["final"] = metrics(y, current_oof)
    report["delta_correct"] = report["final"]["correct"] - report["baseline"]["correct"]
    report["fold_delta_correct"] = fold_delta_correct(y, baseline_oof, current_oof, fold_ids)
    print(json.dumps({"stage": "final", **report}, indent=2), flush=True)

    (OUT / "targeted_80_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "targeted_80_probabilities.npz",
        oof=current_oof,
        test=current_test,
        y=y,
    )
    sample = pd.read_csv(OFFICIAL / "prediction" / "prediction.csv")
    assert sample.iloc[:, 0].astype(str).tolist() == meta_test.index.astype(str).tolist()
    sample.iloc[:, 1] = encoder.inverse_transform(current_test.argmax(axis=1))
    sample.to_csv(OUT / "prediction_targeted_80.csv", index=False)


if __name__ == "__main__":
    main()
