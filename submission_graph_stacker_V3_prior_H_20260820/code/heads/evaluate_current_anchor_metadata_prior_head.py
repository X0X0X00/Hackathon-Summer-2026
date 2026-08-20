from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "current_anchor_metadata_prior_head"
ANCHOR_OOF = (
    PROJECT_ROOT
    / "outputs"
    / "external_reference_fusion"
    / "oof_probabilities_external_primary_crossfit.csv"
)
ANCHOR_TEST = (
    PROJECT_ROOT
    / "outputs"
    / "external_reference_fusion"
    / "test_probabilities_external_primary_crossfit.csv"
)
FOLD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metadata_ablation_minimal_combinations"
    / "fold_assignments.csv"
)

ALPHAS = np.asarray([0.0, 0.025, 0.05, 0.1, 0.2], dtype=np.float64)
TAU = 20.0
LEVELS = {
    "segment": [("Segment",)],
    "segment_ei": [("Segment",), ("Segment", "EI")],
    "segment_ei_ap": [
        ("Segment",),
        ("Segment", "EI"),
        ("Segment", "EI", "AP"),
    ],
}


@dataclass
class HierarchicalPrior:
    global_prior: np.ndarray
    levels: list[tuple[str, ...]]
    tables: list[dict[tuple[str, ...], np.ndarray]]


def read_probabilities(
    path: Path, cell_ids: pd.Index, expected_classes: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, dtype={"Cell_ID": str})
    probability_columns = [column for column in frame.columns if column.startswith("p__")]
    classes = [column[3:] for column in probability_columns]
    if expected_classes is not None:
        if set(classes) != set(expected_classes):
            raise ValueError(f"Class mismatch in {path}")
        probability_columns = [f"p__{label}" for label in expected_classes]
        classes = expected_classes
    aligned = frame.set_index("Cell_ID").reindex(cell_ids)
    if aligned[probability_columns].isna().any().any():
        raise ValueError(f"Cell_ID or probability mismatch in {path}")
    probabilities = aligned[probability_columns].to_numpy(dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, classes


def metadata_features(meta: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=meta.index)
    for output_name, source_name in [
        ("Segment", "Segment"),
        ("EI", "Excitatory_vs_Inhibitory"),
        ("AP", "AP_position"),
    ]:
        result[output_name] = (
            meta[source_name].astype("string").fillna("__MISSING__").astype(str)
        )
    return result


def key_array(features: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple[str, ...]]:
    return list(features.loc[:, list(columns)].itertuples(index=False, name=None))


def fit_prior(
    features: pd.DataFrame,
    labels: np.ndarray,
    levels: list[tuple[str, ...]],
    n_classes: int,
    tau: float = TAU,
) -> HierarchicalPrior:
    global_counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + n_classes)
    tables: list[dict[tuple[str, ...], np.ndarray]] = []
    for level_index, columns in enumerate(levels):
        keys = key_array(features, columns)
        grouped_indices: dict[tuple[str, ...], list[int]] = {}
        for row_index, key in enumerate(keys):
            grouped_indices.setdefault(key, []).append(row_index)
        table: dict[tuple[str, ...], np.ndarray] = {}
        for key, indices in grouped_indices.items():
            counts = np.bincount(labels[indices], minlength=n_classes).astype(np.float64)
            if level_index == 0:
                parent = global_prior
            else:
                parent_key = key[: len(levels[level_index - 1])]
                parent = tables[level_index - 1].get(parent_key, global_prior)
            table[key] = (counts + tau * parent) / (len(indices) + tau)
        tables.append(table)
    return HierarchicalPrior(global_prior=global_prior, levels=levels, tables=tables)


def predict_prior(
    model: HierarchicalPrior, features: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(features)
    probabilities = np.repeat(model.global_prior[None, :], n, axis=0)
    deepest_level = np.zeros(n, dtype=np.int64)
    deepest_support_proxy = np.zeros(n, dtype=np.int64)
    for level_index, (columns, table) in enumerate(zip(model.levels, model.tables), start=1):
        for row_index, key in enumerate(key_array(features, columns)):
            value = table.get(key)
            if value is not None:
                probabilities[row_index] = value
                deepest_level[row_index] = level_index
                deepest_support_proxy[row_index] = 1
    return probabilities, deepest_level, deepest_support_proxy


def build_oof_prior(
    features: pd.DataFrame,
    labels: np.ndarray,
    folds: np.ndarray,
    levels: list[tuple[str, ...]],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = np.zeros((len(labels), n_classes), dtype=np.float64)
    global_prior = np.zeros_like(prior)
    depth = np.zeros(len(labels), dtype=np.int64)
    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        model = fit_prior(features.loc[train_mask], labels[train_mask], levels, n_classes)
        fold_prior, fold_depth, _ = predict_prior(model, features.loc[valid_mask])
        prior[valid_mask] = fold_prior
        global_prior[valid_mask] = model.global_prior
        depth[valid_mask] = fold_depth
    return prior, global_prior, depth


def combine_probabilities(
    anchor: np.ndarray,
    prior: np.ndarray,
    global_prior: np.ndarray,
    alpha: float,
) -> np.ndarray:
    logits = np.log(np.clip(anchor, 1e-12, None))
    residual = np.log(np.clip(prior, 1e-12, None)) - np.log(
        np.clip(global_prior, 1e-12, None)
    )
    logits = logits + alpha * residual
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def evaluate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    anchor_prediction: np.ndarray,
    n_classes: int,
) -> dict[str, object]:
    prediction = probabilities.argmax(axis=1)
    anchor_correct = anchor_prediction == labels
    prediction_correct = prediction == labels
    corrected = int(((~anchor_correct) & prediction_correct).sum())
    harmed = int((anchor_correct & (~prediction_correct)).sum())
    discordant = corrected + harmed
    return {
        "accuracy": float(prediction_correct.mean()),
        "macro_f1": float(
            f1_score(
                labels,
                prediction,
                labels=np.arange(n_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(n_classes))),
        "anchor_wrong_head_right": corrected,
        "anchor_right_head_wrong": harmed,
        "net_corrections": corrected - harmed,
        "changed_predictions": int((prediction != anchor_prediction).sum()),
        "mcnemar_exact_p": float(
            binomtest(min(corrected, harmed), discordant, 0.5).pvalue
        )
        if discordant
        else 1.0,
    }


def choose_alpha(
    labels: np.ndarray,
    anchor: np.ndarray,
    prior: np.ndarray,
    global_prior: np.ndarray,
    n_classes: int,
    criterion: str,
) -> float:
    rows: list[tuple[float, float, float, float]] = []
    for alpha in ALPHAS:
        probabilities = combine_probabilities(anchor, prior, global_prior, float(alpha))
        prediction = probabilities.argmax(axis=1)
        accuracy = float((prediction == labels).mean())
        macro_f1 = float(
            f1_score(
                labels,
                prediction,
                labels=np.arange(n_classes),
                average="macro",
                zero_division=0,
            )
        )
        loss = float(log_loss(labels, probabilities, labels=np.arange(n_classes)))
        rows.append((float(alpha), accuracy, macro_f1, loss))
    if criterion == "accuracy":
        return max(rows, key=lambda row: (row[1], row[2], -row[0]))[0]
    if criterion == "log_loss":
        return min(rows, key=lambda row: (row[3], row[0]))[0]
    raise ValueError(criterion)


def nested_select(
    features: pd.DataFrame,
    labels: np.ndarray,
    folds: np.ndarray,
    anchor: np.ndarray,
    levels: list[tuple[str, ...]],
    n_classes: int,
    criterion: str,
) -> tuple[np.ndarray, dict[str, float]]:
    output = np.zeros_like(anchor)
    selected: dict[str, float] = {}
    unique_folds = sorted(np.unique(folds))
    for outer_fold in unique_folds:
        outer_train = folds != outer_fold
        outer_valid = folds == outer_fold
        inner_prior = np.zeros((outer_train.sum(), n_classes), dtype=np.float64)
        inner_global = np.zeros_like(inner_prior)
        outer_train_positions = np.flatnonzero(outer_train)
        for inner_fold in [fold for fold in unique_folds if fold != outer_fold]:
            inner_valid_global = folds == inner_fold
            inner_fit_global = outer_train & (folds != inner_fold)
            model = fit_prior(
                features.loc[inner_fit_global],
                labels[inner_fit_global],
                levels,
                n_classes,
            )
            predicted, _, _ = predict_prior(model, features.loc[inner_valid_global])
            local_mask = np.isin(outer_train_positions, np.flatnonzero(inner_valid_global))
            inner_prior[local_mask] = predicted
            inner_global[local_mask] = model.global_prior
        alpha = choose_alpha(
            labels[outer_train],
            anchor[outer_train],
            inner_prior,
            inner_global,
            n_classes,
            criterion,
        )
        selected[str(int(outer_fold))] = alpha
        outer_model = fit_prior(
            features.loc[outer_train], labels[outer_train], levels, n_classes
        )
        outer_prior, _, _ = predict_prior(outer_model, features.loc[outer_valid])
        outer_global = np.repeat(
            outer_model.global_prior[None, :], outer_valid.sum(), axis=0
        )
        output[outer_valid] = combine_probabilities(
            anchor[outer_valid], outer_prior, outer_global, alpha
        )
    return output, selected


def shuffled_metadata_control(meta: pd.DataFrame, seed: int = 20260819) -> pd.DataFrame:
    features = metadata_features(meta)
    shuffled = features.copy()
    rng = np.random.default_rng(seed)
    datasets = meta["Datasets"].astype("string").fillna("__MISSING__").astype(str)
    for dataset in datasets.unique():
        positions = np.flatnonzero(datasets.to_numpy() == dataset)
        permutation = rng.permutation(positions)
        shuffled.iloc[positions] = features.iloc[permutation].to_numpy()
    return shuffled


def probability_frame(
    cell_ids: pd.Index, probabilities: np.ndarray, classes: list[str]
) -> pd.DataFrame:
    frame = pd.DataFrame(probabilities, columns=[f"p__{label}" for label in classes])
    frame.insert(0, "Cell_ID", cell_ids.to_numpy())
    return frame


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_train = pd.read_csv(
        PROJECT_ROOT / "data" / "meta_train.csv", index_col=0, dtype={0: str}
    )
    meta_test = pd.read_csv(
        PROJECT_ROOT / "data" / "meta_test.csv", index_col=0, dtype={0: str}
    )
    meta_train.index = meta_train.index.astype(str)
    meta_test.index = meta_test.index.astype(str)
    anchor_oof, classes = read_probabilities(ANCHOR_OOF, meta_train.index)
    anchor_test, _ = read_probabilities(ANCHOR_TEST, meta_test.index, classes)
    label_to_index = {label: index for index, label in enumerate(classes)}
    labels = (
        meta_train["MERFISH_cell_type_annotation"]
        .astype(str)
        .map(label_to_index)
        .to_numpy(dtype=np.int64)
    )
    n_classes = len(classes)
    anchor_prediction = anchor_oof.argmax(axis=1)

    fold_frame = pd.read_csv(FOLD_PATH, dtype={"Cell_ID": str}).set_index("Cell_ID")
    fold_frame = fold_frame.reindex(meta_train.index)
    if fold_frame.isna().any().any():
        raise ValueError("Fold assignments do not align with meta_train")
    folds = {
        "cell": fold_frame["cell_fold"].to_numpy(dtype=np.int64),
        "dataset": fold_frame["dataset_fold"].to_numpy(dtype=np.int64),
        "mouse": fold_frame["mouse_fold"].to_numpy(dtype=np.int64),
        "section": fold_frame["section_fold"].to_numpy(dtype=np.int64),
    }

    real_features = metadata_features(meta_train)
    test_features = metadata_features(meta_test)
    control_features = shuffled_metadata_control(meta_train)
    variants = {**LEVELS, "shuffled_segment_ei_ap": LEVELS["segment_ei_ap"]}
    feature_sets = {
        "segment": real_features,
        "segment_ei": real_features,
        "segment_ei_ap": real_features,
        "shuffled_segment_ei_ap": control_features,
    }

    alpha_rows: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []
    nested_rows: list[dict[str, object]] = []
    fixed_best_rows: list[dict[str, object]] = []
    fixed_best_probabilities: dict[str, np.ndarray] = {}
    deployment_rows: list[dict[str, object]] = []

    baseline_metrics = evaluate(
        labels, anchor_oof, anchor_prediction, n_classes
    )
    for variant, levels in variants.items():
        features = feature_sets[variant]
        prior, global_prior, depth = build_oof_prior(
            features, labels, folds["cell"], levels, n_classes
        )
        head_metrics = evaluate(labels, prior, anchor_prediction, n_classes)
        head_rows.append(
            {
                "variant": variant,
                "tau": TAU,
                "mean_deepest_level": float(depth.mean()),
                **head_metrics,
            }
        )
        variant_rows: list[dict[str, object]] = []
        for alpha in ALPHAS:
            probabilities = combine_probabilities(
                anchor_oof, prior, global_prior, float(alpha)
            )
            row = {
                "variant": variant,
                "alpha": float(alpha),
                **evaluate(labels, probabilities, anchor_prediction, n_classes),
            }
            alpha_rows.append(row)
            variant_rows.append(row)
        best = max(
            variant_rows,
            key=lambda row: (
                float(row["accuracy"]),
                float(row["macro_f1"]),
                -float(row["alpha"]),
            ),
        )
        best_alpha = float(best["alpha"])
        fixed_best = combine_probabilities(anchor_oof, prior, global_prior, best_alpha)
        fixed_best_probabilities[variant] = fixed_best
        fixed_best_rows.append({"variant": variant, **best})

        for criterion in ["accuracy", "log_loss"]:
            nested_probabilities, selected = nested_select(
                features,
                labels,
                folds["cell"],
                anchor_oof,
                levels,
                n_classes,
                criterion,
            )
            nested_rows.append(
                {
                    "variant": variant,
                    "selection_criterion": criterion,
                    "selected_alpha_by_outer_fold": json.dumps(selected, sort_keys=True),
                    **evaluate(
                        labels, nested_probabilities, anchor_prediction, n_classes
                    ),
                }
            )

        if variant != "shuffled_segment_ei_ap":
            full_model = fit_prior(real_features, labels, levels, n_classes)
            test_prior, _, _ = predict_prior(full_model, test_features)
            test_global = np.repeat(
                full_model.global_prior[None, :], len(meta_test), axis=0
            )
            test_probabilities = combine_probabilities(
                anchor_test, test_prior, test_global, best_alpha
            )
            probability_frame(meta_test.index, test_probabilities, classes).to_csv(
                OUTPUT_DIR / f"test_probabilities_current_anchor_{variant}_prior.csv",
                index=False,
            )
            probability_frame(meta_train.index, fixed_best, classes).to_csv(
                OUTPUT_DIR / f"oof_probabilities_current_anchor_{variant}_prior.csv",
                index=False,
            )
            deployment_rows.append(
                {
                    "variant": variant,
                    "deployment_alpha_exploratory_full_oof_choice": best_alpha,
                    "oof_accuracy": best["accuracy"],
                    "oof_macro_f1": best["macro_f1"],
                    "oof_log_loss": best["log_loss"],
                    "warning": "Use only if nested selection and controls support the fixed-alpha result.",
                }
            )

    transfer_rows: list[dict[str, object]] = []
    for split_name in ["dataset", "mouse", "section"]:
        prior, global_prior, _ = build_oof_prior(
            real_features,
            labels,
            folds[split_name],
            LEVELS["segment_ei_ap"],
            n_classes,
        )
        for alpha in ALPHAS:
            probabilities = combine_probabilities(
                anchor_oof, prior, global_prior, float(alpha)
            )
            transfer_rows.append(
                {
                    "prior_transfer_split": split_name,
                    "alpha": float(alpha),
                    **evaluate(labels, probabilities, anchor_prediction, n_classes),
                }
            )

    alpha_curve = pd.DataFrame(alpha_rows)
    head_metrics = pd.DataFrame(head_rows)
    nested_metrics = pd.DataFrame(nested_rows)
    fixed_best = pd.DataFrame(fixed_best_rows)
    deployment = pd.DataFrame(deployment_rows)
    transfer = pd.DataFrame(transfer_rows)

    per_class_rows: list[dict[str, object]] = []
    for variant in ["segment", "segment_ei", "segment_ei_ap"]:
        prediction = fixed_best_probabilities[variant].argmax(axis=1)
        for class_index, class_name in enumerate(classes):
            mask = labels == class_index
            anchor_correct = anchor_prediction[mask] == labels[mask]
            head_correct = prediction[mask] == labels[mask]
            per_class_rows.append(
                {
                    "variant": variant,
                    "class_name": class_name,
                    "support": int(mask.sum()),
                    "anchor_accuracy": float(anchor_correct.mean()),
                    "head_accuracy": float(head_correct.mean()),
                    "accuracy_delta": float(head_correct.mean() - anchor_correct.mean()),
                    "anchor_wrong_head_right": int(((~anchor_correct) & head_correct).sum()),
                    "anchor_right_head_wrong": int((anchor_correct & (~head_correct)).sum()),
                    "net_corrections": int(head_correct.sum() - anchor_correct.sum()),
                }
            )
    per_class = pd.DataFrame(per_class_rows)

    alpha_curve.to_csv(OUTPUT_DIR / "alpha_curve.csv", index=False)
    head_metrics.to_csv(OUTPUT_DIR / "metadata_head_only_metrics.csv", index=False)
    nested_metrics.to_csv(OUTPUT_DIR / "nested_selection_metrics.csv", index=False)
    fixed_best.to_csv(OUTPUT_DIR / "fixed_alpha_best_exploratory.csv", index=False)
    transfer.to_csv(OUTPUT_DIR / "prior_transfer_stress.csv", index=False)
    per_class.to_csv(OUTPUT_DIR / "per_class_impact.csv", index=False)
    deployment.to_csv(OUTPUT_DIR / "deployment_candidates.csv", index=False)

    real_nested = nested_metrics[
        nested_metrics["variant"] != "shuffled_segment_ei_ap"
    ].copy()
    control_nested = nested_metrics[
        nested_metrics["variant"] == "shuffled_segment_ei_ap"
    ].copy()
    metrics = {
        "protocol": {
            "anchor": "external_primary_crossfit",
            "head": "cross-fitted hierarchical empirical-Bayes Segment/EI/AP class prior",
            "combination": "log(anchor) + alpha * (log(metadata_prior) - log(global_prior))",
            "alpha_grid": ALPHAS.tolist(),
            "hierarchical_shrinkage_tau": TAU,
            "nested_selection": "outer cell fold; alpha selected by inner folds; outer labels excluded from head fitting and alpha choice",
            "shuffle_control": "joint Segment/EI/AP tuples permuted within Dataset",
            "transfer_stress_caveat": "Only the prior head is group-held-out; Current Anchor probabilities remain its original cell-level OOF.",
        },
        "baseline": baseline_metrics,
        "metadata_head_only": json_records(head_metrics),
        "fixed_alpha_best_exploratory": json_records(fixed_best),
        "nested_real_metadata": json_records(real_nested),
        "nested_shuffle_control": json_records(control_nested),
        "deployment_candidates": json_records(deployment),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
