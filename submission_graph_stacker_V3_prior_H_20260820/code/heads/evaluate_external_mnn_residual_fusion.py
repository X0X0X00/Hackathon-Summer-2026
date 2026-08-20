from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability.astype(np.float64), 1e-12, None)
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = probability.argmax(axis=1)
    f1 = []
    support = []
    for class_id in range(probability.shape[1]):
        truth = y == class_id
        pred = prediction == class_id
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        denominator = 2 * tp + fp + fn
        f1.append(0.0 if denominator == 0 else 2 * tp / denominator)
        support.append(int(truth.sum()))
    f1_array = np.asarray(f1)
    support_array = np.asarray(support)
    return {
        "accuracy": float(np.mean(prediction == y)),
        "macro_f1": float(f1_array.mean()),
        "weighted_f1": float(np.sum(f1_array * support_array) / support_array.sum()),
        "log_loss": float(-np.log(probability[np.arange(len(y)), y]).mean()),
    }


def load_probability(path: Path, ids: np.ndarray, classes: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    columns = [name if name in frame.columns else f"p__{name}" for name in classes.astype(str)]
    return frame.loc[ids.astype(str), columns].to_numpy(dtype=np.float64)


def constrain(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    allowed = reference > 0.0
    constrained = candidate * allowed
    sums = constrained.sum(axis=1, keepdims=True)
    empty = sums[:, 0] <= 0.0
    constrained[empty] = reference[empty]
    constrained /= constrained.sum(axis=1, keepdims=True)
    return constrained


def crossfit_pair(
    y: np.ndarray,
    folds: np.ndarray,
    base_oof: np.ndarray,
    member_oof: np.ndarray,
    base_test: np.ndarray,
    member_test: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], float]:
    output = np.zeros_like(base_oof)
    weights = []
    details = []
    grid = np.arange(0.0, 1.0 + step / 2.0, step)
    for fold in np.sort(np.unique(folds)):
        tuning = folds != fold
        held = folds == fold
        best_key = None
        best_weight = 0.0
        best_tuning = None
        for weight in grid:
            probability = (1.0 - weight) * base_oof[tuning] + weight * member_oof[tuning]
            current = metric_values(y[tuning], probability)
            key = (current["accuracy"], current["macro_f1"], -current["log_loss"])
            if best_key is None or key > best_key:
                best_key = key
                best_weight = float(weight)
                best_tuning = current
        output[held] = (1.0 - best_weight) * base_oof[held] + best_weight * member_oof[held]
        weights.append(best_weight)
        details.append(
            {"fold": int(fold), "member_weight": best_weight, "n_tuning": int(tuning.sum()),
             "n_held_out": int(held.sum()), "tuning_metrics": best_tuning,
             "held_out_metrics": metric_values(y[held], output[held])}
        )
    test_weight = float(np.mean(weights))
    test = (1.0 - test_weight) * base_test + test_weight * member_test
    return output, test, details, test_weight


def paired_accuracy(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = 42) -> dict[str, float | int]:
    correct_a = a.argmax(axis=1) == y
    correct_b = b.argmax(axis=1) == y
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    differences = correct_a.astype(np.float64) - correct_b.astype(np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([differences[rng.integers(0, len(y), len(y))].mean() for _ in range(5000)])
    discordant = a_only + b_only
    return {
        "accuracy_a": float(correct_a.mean()), "accuracy_b": float(correct_b.mean()),
        "delta_a_minus_b": float(differences.mean()),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "a_only_correct": a_only, "b_only_correct": b_only,
        "mcnemar_exact_p_value": float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MNN residual encoder and strict cross-fitted fusion.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mnn-dir", type=Path, default=PROJECT_ROOT / "outputs" / "external_mnn_residual_encoder")
    parser.add_argument("--existing-dir", type=Path, default=PROJECT_ROOT / "outputs" / "external_reference_fusion")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "external_mnn_residual_fusion")
    parser.add_argument("--weight-step", type=float, default=0.025)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = np.load(data_dir / "ids.npy", allow_pickle=False).astype(str)
    train_positions = np.load(data_dir / "train_positions.npy", allow_pickle=False)
    test_positions = np.load(data_dir / "test_positions.npy", allow_pickle=False)
    folds_all = np.load(data_dir / "folds.npy", allow_pickle=False)
    labels = np.load(data_dir / "labels.npy", allow_pickle=False).astype(np.int64)[train_positions]
    classes = np.load(data_dir / "class_names.npy", allow_pickle=False).astype(str)
    folds = folds_all[train_positions]
    train_ids = ids[train_positions]
    test_ids = ids[test_positions]

    mnn_oof_raw = load_probability(args.mnn_dir / "oof_probabilities_mnn_residual_encoder.csv", train_ids, classes)
    mnn_test_raw = load_probability(args.mnn_dir / "test_probabilities_mnn_residual_encoder.csv", test_ids, classes)
    external_oof = load_probability(args.existing_dir / "oof_probabilities_external_refonly.csv", train_ids, classes)
    external_test = load_probability(args.existing_dir / "test_probabilities_external_refonly.csv", test_ids, classes)
    primary_oof = load_probability(args.existing_dir / "oof_probabilities_external_primary_crossfit.csv", train_ids, classes)
    primary_test = load_probability(args.existing_dir / "test_probabilities_external_primary_crossfit.csv", test_ids, classes)
    mnn_oof = constrain(mnn_oof_raw, external_oof)
    mnn_test = constrain(mnn_test_raw, external_test)

    external_mnn_oof, external_mnn_test, external_details, external_weight = crossfit_pair(
        labels, folds, external_oof, mnn_oof, external_test, mnn_test, args.weight_step
    )
    primary_mnn_oof, primary_mnn_test, primary_details, primary_weight = crossfit_pair(
        labels, folds, primary_oof, mnn_oof, primary_test, mnn_test, args.weight_step
    )
    candidates = {
        "mnn_residual_raw": (mnn_oof_raw, mnn_test_raw, None),
        "mnn_residual_constrained": (mnn_oof, mnn_test, None),
        "external_refonly": (external_oof, external_test, None),
        "external_mnn_crossfit": (external_mnn_oof, external_mnn_test, {"member_test_weight": external_weight, "folds": external_details}),
        "external_primary_crossfit": (primary_oof, primary_test, None),
        "external_primary_mnn_crossfit": (primary_mnn_oof, primary_mnn_test, {"member_test_weight": primary_weight, "folds": primary_details}),
    }
    candidate_metrics = {name: metric_values(labels, values[0]) for name, values in candidates.items()}
    best_name = max(candidate_metrics, key=lambda name: (candidate_metrics[name]["accuracy"], candidate_metrics[name]["macro_f1"]))
    comparison = pd.DataFrame([{"candidate": name, **value} for name, value in candidate_metrics.items()]).sort_values(
        ["accuracy", "macro_f1"], ascending=False
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    for name, (oof_probability, test_probability, _) in candidates.items():
        pd.DataFrame(oof_probability, index=train_ids, columns=classes).rename_axis("Cell_ID").to_csv(
            output_dir / f"oof_probabilities_{name}.csv"
        )
        pd.DataFrame(test_probability, index=test_ids, columns=classes).rename_axis("Cell_ID").to_csv(
            output_dir / f"test_probabilities_{name}.csv"
        )

    template = pd.read_csv(args.existing_dir / "submission_external_primary_crossfit.csv")
    target_column = template.columns[1]
    best_test = candidates[best_name][1]
    pd.DataFrame({"Cell_ID": test_ids, target_column: classes[best_test.argmax(axis=1)]}).to_csv(
        output_dir / f"submission_{best_name}.csv", index=False
    )
    report = {
        "configuration": {
            "weight_step": args.weight_step,
            "weight_selection": "each held-out fold uses the weight selected only on the other folds",
            "constraint": "same reference-derived E/I and Segment support mask as external_refonly",
            "mnn_context": "within-Segment reciprocal neighbors; H uses Segment-centered expression residual",
        },
        "best_candidate": best_name,
        "candidates": {
            name: {"metrics": candidate_metrics[name], "fusion": values[2]}
            for name, values in candidates.items()
        },
        "paired_external_primary_mnn_vs_primary": paired_accuracy(labels, primary_mnn_oof, primary_oof),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(json.dumps(report["paired_external_primary_mnn_vs_primary"], ensure_ascii=False, indent=2))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
