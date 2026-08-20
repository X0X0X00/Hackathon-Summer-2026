from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = ROOT / "outputs" / "external_reference_fusion"
MARKER_DIR = ROOT / "outputs" / "mnn_marker_completion_ablation"
DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)
WEIGHTS = (0.05, 0.075, 0.10)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p.astype(np.float64), 1e-12, None)
    p /= p.sum(axis=1, keepdims=True)
    prediction = p.argmax(axis=1)
    f1_values = []
    supports = []
    for class_id in range(p.shape[1]):
        truth = y == class_id
        predicted = prediction == class_id
        tp = int((truth & predicted).sum())
        fp = int((~truth & predicted).sum())
        fn = int((truth & ~predicted).sum())
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
        supports.append(int(truth.sum()))
    f1 = np.asarray(f1_values)
    support = np.asarray(supports)
    return {
        "accuracy": float((prediction == y).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "log_loss": float(-np.log(p[np.arange(len(y)), y]).mean()),
    }


def mcnemar(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    tail = min(gained, lost)
    logs = np.asarray(
        [
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            - n * math.log(2.0)
            for k in range(tail + 1)
        ]
    )
    maximum = float(logs.max())
    return min(
        1.0,
        2.0 * math.exp(maximum) * float(np.exp(logs - maximum).sum()),
    )


def load_probabilities(
    path: Path, ids: np.ndarray, class_names: np.ndarray
) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"Cell_ID": str}).set_index("Cell_ID")
    columns = [
        f"p__{name}" if f"p__{name}" in frame.columns else name
        for name in class_names
    ]
    missing = [name for name in ids if name not in frame.index]
    if missing:
        raise ValueError(f"{path.name} is missing {len(missing)} requested IDs")
    p = frame.loc[ids, columns].to_numpy(dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def marker_residual(
    full: np.ndarray, no_marker: np.ndarray, clip: float
) -> np.ndarray:
    residual = np.log(np.clip(full, 1e-12, None)) - np.log(
        np.clip(no_marker, 1e-12, None)
    )
    residual -= residual.mean(axis=1, keepdims=True)
    return np.clip(residual, -clip, clip)


def inject(anchor: np.ndarray, residual: np.ndarray, weight: float) -> np.ndarray:
    logits = np.log(np.clip(anchor, 1e-12, None)) + weight * residual
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def paired(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    anchor_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> dict[str, float | int]:
    anchor_prediction = anchor.argmax(axis=1)
    candidate_prediction = candidate.argmax(axis=1)
    anchor_correct = anchor_prediction == y
    candidate_correct = candidate_prediction == y
    gained = int((~anchor_correct & candidate_correct).sum())
    lost = int((anchor_correct & ~candidate_correct).sum())
    return {
        "accuracy_delta": candidate_metrics["accuracy"] - anchor_metrics["accuracy"],
        "macro_f1_delta": candidate_metrics["macro_f1"] - anchor_metrics["macro_f1"],
        "log_loss_delta": candidate_metrics["log_loss"] - anchor_metrics["log_loss"],
        "gained": gained,
        "lost": lost,
        "net": gained - lost,
        "changed_predictions": int((anchor_prediction != candidate_prediction).sum()),
        "mcnemar_exact_p": mcnemar(gained, lost),
    }


def save_probabilities(
    path: Path,
    ids: np.ndarray,
    probabilities: np.ndarray,
    class_names: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        probabilities,
        columns=[f"p__{name}" for name in class_names],
    )
    frame.insert(0, "Cell_ID", ids)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "anchor_marker_completion_residual",
    )
    parser.add_argument("--residual-clip", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ids_all = np.load(DATA_DIR / "ids.npy", allow_pickle=False).astype(str)
    labels_all = np.load(DATA_DIR / "labels.npy", allow_pickle=False).astype(np.int64)
    folds_all = np.load(DATA_DIR / "folds.npy", allow_pickle=False).astype(np.int64)
    train_positions = np.load(
        DATA_DIR / "train_positions.npy", allow_pickle=False
    )
    test_positions = np.load(
        DATA_DIR / "test_positions.npy", allow_pickle=False
    )
    class_names = np.load(
        DATA_DIR / "class_names.npy", allow_pickle=False
    ).astype(str)
    raw_counts = np.load(
        DATA_DIR / "raw_counts.npy", mmap_mode="r", allow_pickle=False
    )
    train_ids = ids_all[train_positions]
    test_ids = ids_all[test_positions]
    y = labels_all[train_positions]
    folds = folds_all[train_positions]

    anchor_oof = load_probabilities(
        ANCHOR_DIR / "oof_probabilities_external_primary_crossfit.csv",
        train_ids,
        class_names,
    )
    anchor_test = load_probabilities(
        ANCHOR_DIR / "test_probabilities_external_primary_crossfit.csv",
        test_ids,
        class_names,
    )
    no_marker_oof = load_probabilities(
        MARKER_DIR / "oof_probabilities_self_mnn_denoised.csv",
        train_ids,
        class_names,
    )
    full_oof = load_probabilities(
        MARKER_DIR / "oof_probabilities_self_mnn_marker_completion.csv",
        train_ids,
        class_names,
    )
    no_marker_test = load_probabilities(
        MARKER_DIR / "test_probabilities_self_mnn_denoised.csv",
        test_ids,
        class_names,
    )
    full_test = load_probabilities(
        MARKER_DIR / "test_probabilities_self_mnn_marker_completion.csv",
        test_ids,
        class_names,
    )

    residual_oof = marker_residual(full_oof, no_marker_oof, args.residual_clip)
    residual_test = marker_residual(full_test, no_marker_test, args.residual_clip)
    shuffled_residual = np.empty_like(residual_oof)
    rng = np.random.default_rng(args.seed)
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        shuffled_residual[rows] = residual_oof[rng.permutation(rows)]

    anchor_metrics = metrics(y, anchor_oof)
    metric_rows = [
        {
            "configuration": "anchor",
            "weight": 0.0,
            "control": False,
            **anchor_metrics,
            "accuracy_delta": 0.0,
            "macro_f1_delta": 0.0,
            "log_loss_delta": 0.0,
            "gained": 0,
            "lost": 0,
            "net": 0,
            "changed_predictions": 0,
            "mcnemar_exact_p": 1.0,
        }
    ]
    probabilities = {}
    for weight in WEIGHTS:
        for control, residual in [
            (False, residual_oof),
            (True, shuffled_residual),
        ]:
            name = (
                f"marker_residual_{weight:g}"
                if not control
                else f"shuffled_residual_{weight:g}"
            )
            candidate = inject(anchor_oof, residual, weight)
            candidate_metrics = metrics(y, candidate)
            metric_rows.append(
                {
                    "configuration": name,
                    "weight": weight,
                    "control": control,
                    **candidate_metrics,
                    **paired(
                        y,
                        anchor_oof,
                        candidate,
                        anchor_metrics,
                        candidate_metrics,
                    ),
                }
            )
            if not control:
                probabilities[name] = candidate

    detected = (np.asarray(raw_counts[train_positions]) > 0).sum(axis=1)
    sparsity_rows = []
    for bin_name, mask in [
        ("0-5", detected <= 5),
        ("6-10", (detected >= 6) & (detected <= 10)),
        ("11-15", (detected >= 11) & (detected <= 15)),
        ("16-25", (detected >= 16) & (detected <= 25)),
        ("26+", detected >= 26),
    ]:
        if not mask.any():
            continue
        for name, candidate in [("anchor", anchor_oof), *probabilities.items()]:
            sparsity_rows.append(
                {
                    "detected_gene_bin": bin_name,
                    "configuration": name,
                    "n_cells": int(mask.sum()),
                    **metrics(y[mask], candidate[mask]),
                }
            )

    class_rows = []
    for class_id, class_name in enumerate(class_names):
        mask = y == class_id
        if not mask.any():
            continue
        anchor_correct = anchor_oof.argmax(axis=1)[mask] == class_id
        for name, candidate in probabilities.items():
            new_correct = candidate.argmax(axis=1)[mask] == class_id
            class_rows.append(
                {
                    "class_name": class_name,
                    "support": int(mask.sum()),
                    "configuration": name,
                    "anchor_accuracy": float(anchor_correct.mean()),
                    "new_accuracy": float(new_correct.mean()),
                    "delta": float(new_correct.mean() - anchor_correct.mean()),
                }
            )

    for weight in WEIGHTS:
        name = f"marker_residual_{weight:g}"
        test_probability = inject(anchor_test, residual_test, weight)
        save_probabilities(
            output_dir / f"oof_probabilities_{name}.csv",
            train_ids,
            probabilities[name],
            class_names,
        )
        save_probabilities(
            output_dir / f"test_probabilities_{name}.csv",
            test_ids,
            test_probability,
            class_names,
        )
        pd.DataFrame(
            {
                "Cell_ID": test_ids,
                "CellType": class_names[test_probability.argmax(axis=1)],
            }
        ).to_csv(output_dir / f"submission_{name}.csv", index=False)

    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(output_dir / "configuration_metrics.csv", index=False)
    pd.DataFrame(sparsity_rows).to_csv(
        output_dir / "sparsity_metrics.csv", index=False
    )
    pd.DataFrame(class_rows).to_csv(
        output_dir / "class_metrics.csv", index=False
    )
    diagnostics = {
        "mean_residual_l2": float(
            np.linalg.norm(residual_oof, axis=1).mean()
        ),
        "median_residual_l2": float(
            np.median(np.linalg.norm(residual_oof, axis=1))
        ),
        "mean_max_abs_residual": float(
            np.abs(residual_oof).max(axis=1).mean()
        ),
        "residual_clip": args.residual_clip,
    }
    report = {
        "protocol": {
            "anchor": "external_primary_crossfit",
            "marker_residual": (
                "centered clipped log P(self+MNN+marker) "
                "- log P(self+MNN)"
            ),
            "fixed_weights": WEIGHTS,
            "selection": "none; all pre-specified weights are reported",
            "negative_control": (
                "marker residual permuted within each OOF fold"
            ),
            "label_leakage": "none",
        },
        "anchor": anchor_metrics,
        "configurations": metric_rows,
        "residual_diagnostics": diagnostics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

