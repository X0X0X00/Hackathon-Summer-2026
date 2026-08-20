from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)
ANCHOR_PATH = (
    ROOT
    / "outputs"
    / "external_reference_fusion"
    / "oof_probabilities_external_primary_crossfit.csv"
)
BIOLOGY_PATH = (
    ROOT
    / "outputs"
    / "biology_aware_glia_head_expanded_v2"
    / "oof_probabilities_best_exploratory.csv"
)
ORDERED_PATH = (
    ROOT
    / "outputs"
    / "ordered_oligodendrocyte_prior"
    / "oof_probabilities_ordered_0.25.csv"
)


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
        f1_values.append(
            0.0 if denominator == 0 else 2 * tp / denominator
        )
        supports.append(int(truth.sum()))
    f1 = np.asarray(f1_values)
    support = np.asarray(supports)
    return {
        "accuracy": float((prediction == y).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "log_loss": float(
            -np.log(p[np.arange(len(y)), y]).mean()
        ),
    }


def mcnemar(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    logs = np.asarray(
        [
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            - n * math.log(2.0)
            for k in range(min(gained, lost) + 1)
        ]
    )
    maximum = float(logs.max())
    return min(
        1.0,
        2.0
        * math.exp(maximum)
        * float(np.exp(logs - maximum).sum()),
    )


def paired(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, float | int]:
    a = baseline.argmax(axis=1)
    b = candidate.argmax(axis=1)
    a_correct = a == y
    b_correct = b == y
    gained = int((~a_correct & b_correct).sum())
    lost = int((a_correct & ~b_correct).sum())
    baseline_metrics = metrics(y, baseline)
    candidate_metrics = metrics(y, candidate)
    return {
        "accuracy_delta": (
            candidate_metrics["accuracy"]
            - baseline_metrics["accuracy"]
        ),
        "macro_f1_delta": (
            candidate_metrics["macro_f1"]
            - baseline_metrics["macro_f1"]
        ),
        "log_loss_delta": (
            candidate_metrics["log_loss"]
            - baseline_metrics["log_loss"]
        ),
        "gained": gained,
        "lost": lost,
        "net": gained - lost,
        "changed_predictions": int((a != b).sum()),
        "mcnemar_exact_p": mcnemar(gained, lost),
    }


def load_probabilities(
    path: Path, ids: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"Cell_ID": str}).set_index("Cell_ID")
    columns = [
        f"p__{name}" if f"p__{name}" in frame.columns else name
        for name in classes
    ]
    p = frame.loc[ids, columns].to_numpy(dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def centered_residual(
    candidate: np.ndarray, anchor: np.ndarray
) -> np.ndarray:
    value = np.log(np.clip(candidate, 1e-12, None)) - np.log(
        np.clip(anchor, 1e-12, None)
    )
    return value - value.mean(axis=1, keepdims=True)


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def additive(
    anchor: np.ndarray, biology: np.ndarray, ordered: np.ndarray
) -> np.ndarray:
    biology_residual = centered_residual(biology, anchor)
    ordered_residual = centered_residual(ordered, anchor)
    logits = (
        np.log(np.clip(anchor, 1e-12, None))
        + biology_residual
        + ordered_residual
    )
    return softmax_logits(logits)


def geometric_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    logits = 0.5 * (
        np.log(np.clip(left, 1e-12, None))
        + np.log(np.clip(right, 1e-12, None))
    )
    return softmax_logits(logits)


def consensus_only(
    anchor: np.ndarray, biology: np.ndarray, ordered: np.ndarray
) -> np.ndarray:
    output = anchor.copy()
    anchor_prediction = anchor.argmax(axis=1)
    biology_prediction = biology.argmax(axis=1)
    ordered_prediction = ordered.argmax(axis=1)
    active = (
        (biology_prediction == ordered_prediction)
        & (biology_prediction != anchor_prediction)
    )
    if active.any():
        output[active] = geometric_rows(
            biology[active], ordered[active]
        )
    return output


def conservative_union(
    anchor: np.ndarray, biology: np.ndarray, ordered: np.ndarray
) -> np.ndarray:
    output = anchor.copy()
    anchor_prediction = anchor.argmax(axis=1)
    biology_prediction = biology.argmax(axis=1)
    ordered_prediction = ordered.argmax(axis=1)
    biology_changed = biology_prediction != anchor_prediction
    ordered_changed = ordered_prediction != anchor_prediction
    biology_only = biology_changed & ~ordered_changed
    ordered_only = ordered_changed & ~biology_changed
    agreement = (
        biology_changed
        & ordered_changed
        & (biology_prediction == ordered_prediction)
    )
    output[biology_only] = biology[biology_only]
    output[ordered_only] = ordered[ordered_only]
    if agreement.any():
        output[agreement] = geometric_rows(
            biology[agreement], ordered[agreement]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "ordered_biology_overlap",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ids_all = np.load(DATA / "ids.npy", allow_pickle=False).astype(str)
    labels_all = np.load(
        DATA / "labels.npy", allow_pickle=False
    ).astype(np.int64)
    train_positions = np.load(
        DATA / "train_positions.npy", allow_pickle=False
    )
    classes = np.load(
        DATA / "class_names.npy", allow_pickle=False
    ).astype(str)
    ids = ids_all[train_positions]
    y = labels_all[train_positions]

    anchor = load_probabilities(ANCHOR_PATH, ids, classes)
    biology = load_probabilities(BIOLOGY_PATH, ids, classes)
    ordered = load_probabilities(ORDERED_PATH, ids, classes)
    combined = {
        "logit_additive": additive(anchor, biology, ordered),
        "consensus_only": consensus_only(anchor, biology, ordered),
        "conservative_union": conservative_union(
            anchor, biology, ordered
        ),
    }
    configurations = {
        "anchor": anchor,
        "biology_head": biology,
        "ordered_prior": ordered,
        **combined,
    }

    anchor_prediction = anchor.argmax(axis=1)
    biology_prediction = biology.argmax(axis=1)
    ordered_prediction = ordered.argmax(axis=1)
    anchor_correct = anchor_prediction == y
    biology_correct = biology_prediction == y
    ordered_correct = ordered_prediction == y
    biology_gain = ~anchor_correct & biology_correct
    ordered_gain = ~anchor_correct & ordered_correct
    biology_loss = anchor_correct & ~biology_correct
    ordered_loss = anchor_correct & ~ordered_correct
    biology_changed = biology_prediction != anchor_prediction
    ordered_changed = ordered_prediction != anchor_prediction

    overlap = {
        "biology_gained": int(biology_gain.sum()),
        "ordered_gained": int(ordered_gain.sum()),
        "gain_intersection": int(
            (biology_gain & ordered_gain).sum()
        ),
        "gain_biology_only": int(
            (biology_gain & ~ordered_gain).sum()
        ),
        "gain_ordered_only": int(
            (ordered_gain & ~biology_gain).sum()
        ),
        "gain_union": int((biology_gain | ordered_gain).sum()),
        "biology_lost": int(biology_loss.sum()),
        "ordered_lost": int(ordered_loss.sum()),
        "loss_intersection": int(
            (biology_loss & ordered_loss).sum()
        ),
        "loss_biology_only": int(
            (biology_loss & ~ordered_loss).sum()
        ),
        "loss_ordered_only": int(
            (ordered_loss & ~biology_loss).sum()
        ),
        "loss_union": int((biology_loss | ordered_loss).sum()),
        "both_changed": int(
            (biology_changed & ordered_changed).sum()
        ),
        "both_changed_same_prediction": int(
            (
                biology_changed
                & ordered_changed
                & (biology_prediction == ordered_prediction)
            ).sum()
        ),
        "both_changed_conflict": int(
            (
                biology_changed
                & ordered_changed
                & (biology_prediction != ordered_prediction)
            ).sum()
        ),
        "biology_changed_only": int(
            (biology_changed & ~ordered_changed).sum()
        ),
        "ordered_changed_only": int(
            (ordered_changed & ~biology_changed).sum()
        ),
        "oracle_union_accuracy": float(
            (anchor_correct | biology_correct | ordered_correct).mean()
        ),
    }

    metric_rows = []
    for name, probability in configurations.items():
        metric_rows.append(
            {
                "configuration": name,
                **metrics(y, probability),
                **(
                    {
                        "accuracy_delta": 0.0,
                        "macro_f1_delta": 0.0,
                        "log_loss_delta": 0.0,
                        "gained": 0,
                        "lost": 0,
                        "net": 0,
                        "changed_predictions": 0,
                        "mcnemar_exact_p": 1.0,
                    }
                    if name == "anchor"
                    else paired(y, anchor, probability)
                ),
            }
        )

    pair_rows = []
    for baseline_name in ("biology_head", "ordered_prior"):
        for candidate_name in combined:
            pair_rows.append(
                {
                    "baseline": baseline_name,
                    "candidate": candidate_name,
                    **paired(
                        y,
                        configurations[baseline_name],
                        configurations[candidate_name],
                    ),
                }
            )

    biology_residual = centered_residual(biology, anchor)
    ordered_residual = centered_residual(ordered, anchor)
    flattened_correlation = float(
        np.corrcoef(
            biology_residual.ravel(), ordered_residual.ravel()
        )[0, 1]
    )
    biology_norm = np.linalg.norm(biology_residual, axis=1)
    ordered_norm = np.linalg.norm(ordered_residual, axis=1)
    cosine = (
        (biology_residual * ordered_residual).sum(axis=1)
        / np.maximum(biology_norm * ordered_norm, 1e-12)
    )

    cell_frame = pd.DataFrame(
        {
            "Cell_ID": ids,
            "true_label": classes[y],
            "anchor_prediction": classes[anchor_prediction],
            "biology_prediction": classes[biology_prediction],
            "ordered_prediction": classes[ordered_prediction],
            "anchor_correct": anchor_correct,
            "biology_correct": biology_correct,
            "ordered_correct": ordered_correct,
            "biology_gain": biology_gain,
            "ordered_gain": ordered_gain,
            "biology_loss": biology_loss,
            "ordered_loss": ordered_loss,
            "biology_changed": biology_changed,
            "ordered_changed": ordered_changed,
            "residual_cosine": cosine,
        }
    )
    for name, probability in combined.items():
        prediction = probability.argmax(axis=1)
        cell_frame[f"{name}_prediction"] = classes[prediction]
        cell_frame[f"{name}_correct"] = prediction == y

    pd.DataFrame(metric_rows).to_csv(
        output_dir / "configuration_metrics.csv", index=False
    )
    pd.DataFrame(pair_rows).to_csv(
        output_dir / "combined_vs_components.csv", index=False
    )
    cell_frame.to_csv(
        output_dir / "cell_overlap_audit.csv", index=False
    )
    report = {
        "protocol": {
            "biology_head": str(BIOLOGY_PATH),
            "ordered_prior": str(ORDERED_PATH),
            "combination_rules": {
                "logit_additive": (
                    "Anchor log probabilities plus both centered "
                    "candidate-vs-Anchor log residuals"
                ),
                "consensus_only": (
                    "change only when both priors choose the same "
                    "non-Anchor class"
                ),
                "conservative_union": (
                    "accept a single prior change; accept matching "
                    "dual changes; return to Anchor on conflict"
                ),
            },
            "selection": "none",
            "caveat": (
                "Both component candidates were selected "
                "exploratorily on these OOF labels."
            ),
        },
        "correction_overlap": overlap,
        "residual_overlap": {
            "flattened_pearson_correlation": flattened_correlation,
            "mean_cell_cosine": float(cosine.mean()),
            "median_cell_cosine": float(np.median(cosine)),
        },
        "configurations": metric_rows,
        "combined_vs_components": pair_rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

