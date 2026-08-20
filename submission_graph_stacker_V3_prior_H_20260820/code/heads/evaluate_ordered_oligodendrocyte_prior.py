from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from train_biology_aware_glia_head import build_fold_features


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\static.npz"
)
ANCHOR_DIR = ROOT / "outputs" / "external_reference_fusion"
FAMILY = [
    "oligodendrocyte_precursor_cell",
    "oligodendrocyte_progenitor_1",
    "oligodendrocyte_progenitor_2",
    "oligodendrocyte_1",
    "oligodendrocyte_2",
]
REAL_STAGE = {
    "oligodendrocyte_precursor_cell": 0,
    "oligodendrocyte_progenitor_1": 0,
    "oligodendrocyte_progenitor_2": 1,
    "oligodendrocyte_1": 1,
    "oligodendrocyte_2": 2,
}
HEADS = ("ordered", "unordered", "shuffled_order")
WEIGHTS = (0.10, 0.25, 0.50)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        2.0 * math.exp(maximum) * float(np.exp(logs - maximum).sum()),
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
    a_metrics = metrics(y, baseline)
    b_metrics = metrics(y, candidate)
    return {
        "accuracy_delta": b_metrics["accuracy"] - a_metrics["accuracy"],
        "macro_f1_delta": b_metrics["macro_f1"] - a_metrics["macro_f1"],
        "log_loss_delta": b_metrics["log_loss"] - a_metrics["log_loss"],
        "gained": gained,
        "lost": lost,
        "net": gained - lost,
        "changed_predictions": int((a != b).sum()),
        "mcnemar_exact_p": mcnemar(gained, lost),
    }


def load_probabilities(
    path: Path, ids: np.ndarray, class_names: np.ndarray
) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"Cell_ID": str}).set_index("Cell_ID")
    columns = [
        f"p__{name}" if f"p__{name}" in frame.columns else name
        for name in class_names
    ]
    p = frame.loc[ids, columns].to_numpy(dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def save_probabilities(
    path: Path,
    ids: np.ndarray,
    p: np.ndarray,
    class_names: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        p, columns=[f"p__{name}" for name in class_names]
    )
    frame.insert(0, "Cell_ID", ids)
    frame.to_csv(path, index=False)


class OrdinalHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.first_threshold = nn.Parameter(torch.tensor(-0.5))
        self.threshold_gap = nn.Parameter(torch.tensor(0.0))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        score = self.score(x)
        first = self.first_threshold
        second = first + F.softplus(self.threshold_gap) + 0.10
        return torch.cat([score - first, score - second], dim=1)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        cumulative = torch.sigmoid(self.logits(x))
        p0 = 1.0 - cumulative[:, :1]
        p1 = cumulative[:, :1] - cumulative[:, 1:2]
        p2 = cumulative[:, 1:2]
        return torch.cat([p0, p1, p2], dim=1)


class UnorderedHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(x), dim=1)


def fit_ordinal(
    x: np.ndarray,
    stage: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
) -> OrdinalHead:
    seed_all(seed)
    model = OrdinalHead(x.shape[1]).to(device)
    features = torch.as_tensor(x, dtype=torch.float32, device=device)
    target = torch.as_tensor(
        np.column_stack([stage > 0, stage > 1]),
        dtype=torch.float32,
        device=device,
    )
    positive = target.sum(dim=0)
    negative = len(target) - positive
    positive_weight = negative / torch.clamp(positive, min=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.025, weight_decay=0.02
    )
    for _ in range(epochs):
        model.train()
        logits = model.logits(features)
        loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=positive_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def fit_unordered(
    x: np.ndarray,
    stage: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
) -> UnorderedHead:
    seed_all(seed)
    model = UnorderedHead(x.shape[1]).to(device)
    features = torch.as_tensor(x, dtype=torch.float32, device=device)
    target = torch.as_tensor(stage, dtype=torch.long, device=device)
    counts = torch.bincount(target, minlength=3).float()
    weights = torch.sqrt(counts.sum() / torch.clamp(3 * counts, min=1.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.02, weight_decay=0.02
    )
    for _ in range(epochs):
        model.train()
        loss = F.cross_entropy(model.network(features), target, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def head_predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            features = torch.as_tensor(
                x[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            output.append(model.probabilities(features).cpu().numpy())
    return np.concatenate(output, axis=0)


def reweight_family(
    anchor: np.ndarray,
    stage_probability: np.ndarray,
    family_indices: np.ndarray,
    class_stage: np.ndarray,
    weight: float,
) -> np.ndarray:
    output = anchor.copy()
    family = np.clip(anchor[:, family_indices], 1e-12, None)
    family_mass = family.sum(axis=1, keepdims=True)
    conditional = family / family_mass
    anchor_stage = np.column_stack(
        [
            conditional[:, class_stage == stage].sum(axis=1)
            for stage in range(3)
        ]
    )
    residual_stage = np.log(np.clip(stage_probability, 1e-8, None)) - np.log(
        np.clip(anchor_stage, 1e-8, None)
    )
    residual_stage = np.clip(residual_stage, -2.0, 2.0)
    residual_class = residual_stage[:, class_stage]
    logits = np.log(conditional) + weight * residual_class
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    anchor_prediction = anchor.argmax(axis=1)
    active = np.isin(anchor_prediction, family_indices)
    output[np.ix_(active, family_indices)] = (
        adjusted[active] * family_mass[active]
    )
    output /= output.sum(axis=1, keepdims=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "ordered_oligodendrocyte_prior",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(CACHE, allow_pickle=True)
    values = cache["X"].astype(np.float32)
    feature_names = cache["names"].astype(str)
    ids = cache["ids"].astype(str)
    labels = cache["y"].astype(np.int64)
    class_names = cache["labels"].astype(str)
    is_train = cache["is_train"].astype(bool)
    is_test = cache["is_test"].astype(bool)
    folds_all = cache["folds"].astype(np.int64)
    train_rows = np.flatnonzero(is_train)
    test_rows = np.flatnonzero(is_test)
    train_ids = ids[train_rows]
    test_ids = ids[test_rows]
    train_labels = labels[train_rows]
    folds = folds_all[train_rows]
    class_index = {name: i for i, name in enumerate(class_names)}
    family_indices = np.asarray(
        [class_index[name] for name in FAMILY], dtype=np.int64
    )
    class_stage = np.asarray([REAL_STAGE[name] for name in FAMILY])
    true_stage_by_class = np.full(len(class_names), -1, dtype=np.int64)
    for name, stage in REAL_STAGE.items():
        true_stage_by_class[class_index[name]] = stage

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
    stage_oof = {
        head: np.zeros((len(train_rows), 3), dtype=np.float32)
        for head in HEADS
    }
    stage_test_folds = {head: [] for head in HEADS}
    training_rows = []
    permutation = np.asarray([2, 0, 1], dtype=np.int64)

    for fold in np.sort(np.unique(folds)):
        fold = int(fold)
        fit_global = train_rows[folds != fold]
        valid_global = train_rows[folds == fold]
        valid_local = np.flatnonzero(folds == fold)
        modules, _, _ = build_fold_features(
            values, feature_names, fit_global
        )
        biology = np.column_stack(
            [
                modules["early_ol"],
                modules["transition_ol"],
                modules["mature_ol"],
                modules["transition_ol"] - modules["early_ol"],
                modules["mature_ol"] - modules["transition_ol"],
            ]
        ).astype(np.float32)
        fit_family = np.isin(labels[fit_global], family_indices)
        head_rows = fit_global[fit_family]
        head_stage = true_stage_by_class[labels[head_rows]]
        mean = biology[head_rows].mean(axis=0)
        scale = biology[head_rows].std(axis=0)
        scale[scale < 1e-5] = 1.0
        standardized = np.clip((biology - mean) / scale, -8, 8)
        training_rows.append(
            {
                "fold": fold,
                "n_family_train": len(head_rows),
                "stage0": int((head_stage == 0).sum()),
                "stage1": int((head_stage == 1).sum()),
                "stage2": int((head_stage == 2).sum()),
            }
        )

        ordered = fit_ordinal(
            standardized[head_rows],
            head_stage,
            device,
            args.seed + fold * 100,
            args.epochs,
        )
        unordered = fit_unordered(
            standardized[head_rows],
            head_stage,
            device,
            args.seed + fold * 100 + 1,
            args.epochs,
        )
        shuffled = fit_ordinal(
            standardized[head_rows],
            permutation[head_stage],
            device,
            args.seed + fold * 100 + 2,
            args.epochs,
        )
        ordered_valid = head_predict(
            ordered, standardized[valid_global], device
        )
        unordered_valid = head_predict(
            unordered, standardized[valid_global], device
        )
        shuffled_pseudo_valid = head_predict(
            shuffled, standardized[valid_global], device
        )
        shuffled_valid = np.empty_like(shuffled_pseudo_valid)
        for real_stage in range(3):
            shuffled_valid[:, real_stage] = shuffled_pseudo_valid[
                :, permutation[real_stage]
            ]
        stage_oof["ordered"][valid_local] = ordered_valid
        stage_oof["unordered"][valid_local] = unordered_valid
        stage_oof["shuffled_order"][valid_local] = shuffled_valid

        ordered_test = head_predict(
            ordered, standardized[test_rows], device
        )
        unordered_test = head_predict(
            unordered, standardized[test_rows], device
        )
        shuffled_pseudo_test = head_predict(
            shuffled, standardized[test_rows], device
        )
        shuffled_test = np.empty_like(shuffled_pseudo_test)
        for real_stage in range(3):
            shuffled_test[:, real_stage] = shuffled_pseudo_test[
                :, permutation[real_stage]
            ]
        stage_test_folds["ordered"].append(ordered_test)
        stage_test_folds["unordered"].append(unordered_test)
        stage_test_folds["shuffled_order"].append(shuffled_test)

    candidates = {}
    test_candidates = {}
    for head in HEADS:
        test_stage = np.stack(stage_test_folds[head]).mean(axis=0)
        for weight in WEIGHTS:
            name = f"{head}_{weight:g}"
            candidates[name] = reweight_family(
                anchor_oof,
                stage_oof[head],
                family_indices,
                class_stage,
                weight,
            )
            test_candidates[name] = reweight_family(
                anchor_test,
                test_stage,
                family_indices,
                class_stage,
                weight,
            )

    anchor_metrics = metrics(train_labels, anchor_oof)
    metric_rows = [
        {
            "configuration": "anchor",
            "head": "anchor",
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
    for head in HEADS:
        for weight in WEIGHTS:
            name = f"{head}_{weight:g}"
            metric_rows.append(
                {
                    "configuration": name,
                    "head": head,
                    "weight": weight,
                    "control": head == "shuffled_order",
                    **metrics(train_labels, candidates[name]),
                    **paired(train_labels, anchor_oof, candidates[name]),
                }
            )

    family_truth = np.isin(train_labels, family_indices)
    true_family_stage = true_stage_by_class[train_labels[family_truth]]
    stage_metrics = {}
    for head in HEADS:
        probability = stage_oof[head][family_truth]
        stage_metrics[head] = {
            "n_family": int(family_truth.sum()),
            "stage_accuracy": float(
                (probability.argmax(axis=1) == true_family_stage).mean()
            ),
            "stage_log_loss": float(
                -np.log(
                    np.clip(
                        probability[
                            np.arange(len(true_family_stage)),
                            true_family_stage,
                        ],
                        1e-12,
                        None,
                    )
                ).mean()
            ),
        }

    order_comparisons = []
    for weight in WEIGHTS:
        ordered_name = f"ordered_{weight:g}"
        for control in ("unordered", "shuffled_order"):
            control_name = f"{control}_{weight:g}"
            order_comparisons.append(
                {
                    "weight": weight,
                    "baseline": control_name,
                    "candidate": ordered_name,
                    **paired(
                        train_labels,
                        candidates[control_name],
                        candidates[ordered_name],
                    ),
                }
            )

    family_rows = []
    anchor_prediction = anchor_oof.argmax(axis=1)
    for name in ["anchor", *candidates]:
        probability = anchor_oof if name == "anchor" else candidates[name]
        prediction = probability.argmax(axis=1)
        for class_id in family_indices:
            mask = train_labels == class_id
            family_rows.append(
                {
                    "configuration": name,
                    "class_name": class_names[class_id],
                    "support": int(mask.sum()),
                    "accuracy": float((prediction[mask] == class_id).mean()),
                    "anchor_accuracy": float(
                        (anchor_prediction[mask] == class_id).mean()
                    ),
                }
            )

    for weight in WEIGHTS:
        name = f"ordered_{weight:g}"
        save_probabilities(
            output_dir / f"oof_probabilities_{name}.csv",
            train_ids,
            candidates[name],
            class_names,
        )
        save_probabilities(
            output_dir / f"test_probabilities_{name}.csv",
            test_ids,
            test_candidates[name],
            class_names,
        )
        pd.DataFrame(
            {
                "Cell_ID": test_ids,
                "CellType": class_names[
                    test_candidates[name].argmax(axis=1)
                ],
            }
        ).to_csv(
            output_dir / f"submission_{name}.csv", index=False
        )

    pd.DataFrame(metric_rows).to_csv(
        output_dir / "configuration_metrics.csv", index=False
    )
    pd.DataFrame(training_rows).to_csv(
        output_dir / "fold_training_support.csv", index=False
    )
    pd.DataFrame(order_comparisons).to_csv(
        output_dir / "ordered_vs_controls.csv", index=False
    )
    pd.DataFrame(family_rows).to_csv(
        output_dir / "family_class_metrics.csv", index=False
    )
    report = {
        "protocol": {
            "family": FAMILY,
            "ordered_stages": {
                "0_early": FAMILY[:2],
                "1_transition": FAMILY[2:4],
                "2_mature": FAMILY[4:],
            },
            "features": [
                "early_ol",
                "transition_ol",
                "mature_ol",
                "transition_minus_early",
                "mature_minus_transition",
            ],
            "head": "two-threshold proportional-odds ordinal model",
            "application": (
                "preserve Anchor oligodendrocyte-family mass; "
                "reweight only within family when Anchor top-1 is in family"
            ),
            "fixed_weights": WEIGHTS,
            "controls": [
                "capacity-matched unordered three-stage head",
                "ordinal head trained with permuted stage order",
            ],
            "device": torch.cuda.get_device_name(0),
            "selection": "none",
        },
        "anchor": anchor_metrics,
        "stage_head_metrics": stage_metrics,
        "configurations": metric_rows,
        "ordered_vs_controls": order_comparisons,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

