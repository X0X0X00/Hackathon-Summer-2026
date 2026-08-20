from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_biology_aware_glia_head as biology


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = ROOT / "outputs" / "external_reference_fusion"
BIOLOGY_OOF = (
    ROOT
    / "outputs"
    / "biology_aware_glia_head_expanded_v2"
    / "oof_probabilities_best_exploratory.csv"
)
ORDERED_DIR = ROOT / "outputs" / "ordered_oligodendrocyte_prior"
FIXED = {
    "mode": "rare_reference",
    "high_confidence_threshold": 0.85,
    "high_main_margin": 1.0,
    "medium_confidence_threshold": 0.65,
    "medium_main_margin": 0.2,
    "alpha": 2.0,
    "moderate_weight": 0.05,
}


def load_probabilities(
    path: Path, ids: np.ndarray, classes: list[str]
) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"Cell_ID": str}).set_index("Cell_ID")
    columns = [
        f"p__{name}" if f"p__{name}" in frame.columns else name
        for name in classes
    ]
    p = frame.loc[ids, columns].to_numpy(dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def save_probabilities(
    path: Path,
    ids: np.ndarray,
    p: np.ndarray,
    classes: list[str],
) -> None:
    frame = pd.DataFrame(
        p, columns=[f"p__{name}" for name in classes]
    )
    frame.insert(0, "Cell_ID", ids)
    frame.to_csv(path, index=False)


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    p = np.clip(probabilities.astype(np.float64), 1e-12, None)
    p /= p.sum(axis=1, keepdims=True)
    prediction = p.argmax(axis=1)
    f1_values = []
    support = []
    for class_id in range(p.shape[1]):
        truth = labels == class_id
        predicted = prediction == class_id
        tp = int((truth & predicted).sum())
        fp = int((~truth & predicted).sum())
        fn = int((truth & ~predicted).sum())
        denominator = 2 * tp + fp + fn
        f1_values.append(
            0.0 if denominator == 0 else 2 * tp / denominator
        )
        support.append(int(truth.sum()))
    f1 = np.asarray(f1_values)
    support_array = np.asarray(support)
    return {
        "accuracy": float((prediction == labels).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(
            (f1 * support_array).sum() / support_array.sum()
        ),
        "log_loss": float(
            -np.log(p[np.arange(len(labels)), labels]).mean()
        ),
    }


def geometric_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    logits = 0.5 * (
        np.log(np.clip(left, 1e-12, None))
        + np.log(np.clip(right, 1e-12, None))
    )
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def conservative_union(
    anchor: np.ndarray,
    biology_probability: np.ndarray,
    ordered_probability: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    output = anchor.copy()
    anchor_prediction = anchor.argmax(axis=1)
    biology_prediction = biology_probability.argmax(axis=1)
    ordered_prediction = ordered_probability.argmax(axis=1)
    biology_changed = biology_prediction != anchor_prediction
    ordered_changed = ordered_prediction != anchor_prediction
    biology_only = biology_changed & ~ordered_changed
    ordered_only = ordered_changed & ~biology_changed
    agreement = (
        biology_changed
        & ordered_changed
        & (biology_prediction == ordered_prediction)
    )
    conflict = (
        biology_changed
        & ordered_changed
        & (biology_prediction != ordered_prediction)
    )
    output[biology_only] = biology_probability[biology_only]
    output[ordered_only] = ordered_probability[ordered_only]
    if agreement.any():
        output[agreement] = geometric_rows(
            biology_probability[agreement],
            ordered_probability[agreement],
        )
    audit = {
        "biology_changed": int(biology_changed.sum()),
        "ordered_changed": int(ordered_changed.sum()),
        "biology_only": int(biology_only.sum()),
        "ordered_only": int(ordered_only.sum()),
        "both_agree": int(agreement.sum()),
        "both_conflict_returned_to_anchor": int(conflict.sum()),
        "combined_changed": int(
            (output.argmax(axis=1) != anchor_prediction).sum()
        ),
    }
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "ordered_biology_submission",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(biology.CACHE_PATH, allow_pickle=True)
    values = cache["X"].astype(np.float32)
    names = cache["names"].astype(str)
    ids = cache["ids"].astype(str)
    labels = cache["y"].astype(np.int64)
    classes = cache["labels"].astype(str).tolist()
    is_train = cache["is_train"].astype(bool)
    is_test = cache["is_test"].astype(bool)
    is_reference = cache["is_ref"].astype(bool)
    folds_all = cache["folds"].astype(np.int64)
    train_rows = np.flatnonzero(is_train)
    test_rows = np.flatnonzero(is_test)
    reference_rows = np.flatnonzero(is_reference)
    train_folds = folds_all[train_rows]
    train_ids = ids[train_rows]
    test_ids = ids[test_rows]
    class_index = {name: index for index, name in enumerate(classes)}

    anchor_oof = load_probabilities(
        ANCHOR_DIR / "oof_probabilities_external_primary_crossfit.csv",
        train_ids,
        classes,
    )
    anchor_test = load_probabilities(
        ANCHOR_DIR / "test_probabilities_external_primary_crossfit.csv",
        test_ids,
        classes,
    )
    ordered_oof = load_probabilities(
        ORDERED_DIR / "oof_probabilities_ordered_0.25.csv",
        train_ids,
        classes,
    )
    ordered_test = load_probabilities(
        ORDERED_DIR / "test_probabilities_ordered_0.25.csv",
        test_ids,
        classes,
    )
    biology_oof = load_probabilities(
        BIOLOGY_OOF, train_ids, classes
    )

    anchor_test_prediction = anchor_test.argmax(axis=1)
    anchor_test_top5 = np.argsort(-anchor_test, axis=1)[:, :5]
    pair_classes: list[tuple[int, int | None]] = []
    pair_families: list[np.ndarray] = []
    for spec in biology.PAIR_SPECS:
        positive = class_index[str(spec["positive"])]
        negative = (
            None
            if spec["negative"] is None
            else class_index[str(spec["negative"])]
        )
        pair_classes.append((positive, negative))
        family_names = spec.get("family", biology.CENTRAL_GLIA)
        pair_families.append(
            np.asarray(
                [class_index[name] for name in family_names],
                dtype=np.int64,
            )
        )

    fold_test_probabilities = []
    training_rows = []
    application_rows = []
    mode_index = biology.MODES.index(FIXED["mode"])
    for fold in range(5):
        competition_fit = train_rows[train_folds != fold]
        modules, auxiliary, _ = biology.build_fold_features(
            values, names, competition_fit
        )
        pair_probability = np.zeros(
            (len(test_rows), len(biology.PAIR_SPECS)),
            dtype=np.float32,
        )
        pair_main = np.zeros_like(pair_probability)
        for pair_index, spec in enumerate(biology.PAIR_SPECS):
            inputs, main_scores, input_names = biology.pair_inputs(
                spec, modules, auxiliary
            )
            pair_main[:, pair_index] = main_scores[test_rows]
            rng = np.random.default_rng(
                biology.SEED
                + fold * 1000
                + pair_index * 20
                + mode_index
            )
            (
                fit_rows,
                targets,
                source_weights,
                counts,
            ) = biology.training_rows_for_spec(
                spec,
                labels,
                competition_fit,
                reference_rows,
                class_index,
                FIXED["mode"],
                rng,
            )
            predicted, final_loss, main_weight = (
                biology.fit_predict_head(
                    inputs,
                    fit_rows,
                    targets,
                    source_weights,
                    test_rows,
                    device,
                    biology.SEED
                    + fold * 1000
                    + pair_index * 20
                    + mode_index,
                )
            )
            pair_probability[:, pair_index] = predicted
            training_rows.append(
                {
                    "fold": fold,
                    "pair": spec["name"],
                    "tier": spec["tier"],
                    **counts,
                    "final_loss": final_loss,
                    "learned_positive_main_weight": main_weight,
                    "input_features": "|".join(input_names),
                }
            )

        fold_probability, masks, total_applications = biology.apply_heads(
            anchor_test,
            anchor_test_prediction,
            anchor_test_top5,
            pair_probability,
            pair_main,
            pair_classes,
            pair_families,
            FIXED["high_confidence_threshold"],
            FIXED["high_main_margin"],
            FIXED["medium_confidence_threshold"],
            FIXED["medium_main_margin"],
            FIXED["alpha"],
            FIXED["moderate_weight"],
        )
        fold_test_probabilities.append(fold_probability)
        invoked = np.zeros(len(test_rows), dtype=bool)
        for pair_name, mask in masks.items():
            invoked |= mask
            application_rows.append(
                {
                    "fold": fold,
                    "pair": pair_name,
                    "applications": int(mask.sum()),
                }
            )
        print(
            f"fold={fold} invoked_cells={int(invoked.sum())} "
            f"head_applications={total_applications}",
            flush=True,
        )

    biology_test = np.mean(
        np.stack(fold_test_probabilities, axis=0), axis=0
    ).astype(np.float64)
    biology_test /= biology_test.sum(axis=1, keepdims=True)

    combined_oof, oof_audit = conservative_union(
        anchor_oof, biology_oof, ordered_oof
    )
    combined_test, test_audit = conservative_union(
        anchor_test, biology_test, ordered_test
    )
    oof_metrics = classification_metrics(
        labels[train_rows], combined_oof
    )

    save_probabilities(
        output_dir / "oof_probabilities_conservative_union.csv",
        train_ids,
        combined_oof,
        classes,
    )
    save_probabilities(
        output_dir / "test_probabilities_biology_head_fixed.csv",
        test_ids,
        biology_test,
        classes,
    )
    save_probabilities(
        output_dir / "test_probabilities_conservative_union.csv",
        test_ids,
        combined_test,
        classes,
    )
    pd.DataFrame(
        {
            "Cell_ID": test_ids,
            "CellType": np.asarray(classes)[biology_test.argmax(axis=1)],
        }
    ).to_csv(
        output_dir / "submission_biology_head_fixed.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "Cell_ID": test_ids,
            "CellType": np.asarray(classes)[combined_test.argmax(axis=1)],
        }
    ).to_csv(
        output_dir / "submission_conservative_union.csv",
        index=False,
    )
    pd.DataFrame(training_rows).to_csv(
        output_dir / "biology_head_training.csv", index=False
    )
    pd.DataFrame(application_rows).to_csv(
        output_dir / "biology_head_test_applications.csv",
        index=False,
    )

    report = {
        "protocol": {
            "biology_fixed_configuration": FIXED,
            "biology_test_ensemble": (
                "five fold-matched heads; each excludes its competition "
                "validation fold and predicts all test cells"
            ),
            "combination": (
                "accept single-prior changes, accept matching dual "
                "changes, return to Anchor on conflict"
            ),
            "ordered_candidate": "ordered_0.25",
            "device": torch.cuda.get_device_name(0),
            "selection": "no additional test-time or OOF tuning",
            "caveat": (
                "The fixed Biology and ordered configurations were "
                "chosen exploratorily on OOF."
            ),
        },
        "oof_metrics": oof_metrics,
        "oof_route_audit": oof_audit,
        "test_route_audit": test_audit,
        "test_prediction_distribution": {
            "anchor_changed_by_biology": int(
                (
                    biology_test.argmax(axis=1)
                    != anchor_test.argmax(axis=1)
                ).sum()
            ),
            "anchor_changed_by_ordered": int(
                (
                    ordered_test.argmax(axis=1)
                    != anchor_test.argmax(axis=1)
                ).sum()
            ),
            "anchor_changed_by_combined": int(
                (
                    combined_test.argmax(axis=1)
                    != anchor_test.argmax(axis=1)
                ).sum()
            ),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

