from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)
MODEL_SOURCES = {
    "anchor_external_primary": PROJECT_ROOT / "outputs" / "external_reference_fusion",
    "external_refonly": PROJECT_ROOT / "outputs" / "external_reference_fusion",
    "gene_token": PROJECT_ROOT / "outputs" / "external_reference_fusion",
    "segment_mnn": PROJECT_ROOT / "outputs" / "external_mnn_residual_fusion",
    "soft_slot_segment_center": PROJECT_ROOT / "outputs" / "external_soft_slot_segment_center_fusion",
}
PROBABILITY_STEMS = {
    "anchor_external_primary": "external_primary_crossfit",
    "external_refonly": "external_refonly",
    "gene_token": "external_gene_token",
    "segment_mnn": "mnn_residual_constrained",
    "soft_slot_segment_center": "mnn_residual_constrained",
}
EXPERT_NAMES = ["external_refonly", "gene_token", "segment_mnn", "soft_slot_segment_center"]
GATE_FEATURE_NAMES = [
    "anchor_uncertainty",
    "anchor_margin",
    "mean_expert_entropy",
    "expert_disagreement_rate",
    "anchor_class_probability_std",
    "maximum_expert_confidence",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_probability_csv(path: Path, ids: np.ndarray, classes: np.ndarray) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        column_lookup = {name: index for index, name in enumerate(header)}
        columns = []
        for class_name in classes:
            if class_name in column_lookup:
                columns.append(column_lookup[class_name])
            elif f"p__{class_name}" in column_lookup:
                columns.append(column_lookup[f"p__{class_name}"])
            else:
                raise ValueError(f"Missing probability column {class_name} in {path}")
        rows: dict[str, np.ndarray] = {}
        for row in reader:
            rows[str(row[0])] = np.asarray([float(row[index]) for index in columns], dtype=np.float64)
    missing = [cell_id for cell_id in ids if str(cell_id) not in rows]
    if missing:
        raise ValueError(f"Missing Cell_IDs in {path}: {missing[:3]}")
    values = np.stack([rows[str(cell_id)] for cell_id in ids])
    values = np.clip(values, 1e-9, None)
    values /= values.sum(axis=1, keepdims=True)
    return values.astype(np.float32)


def save_probability_csv(path: Path, ids: np.ndarray, classes: np.ndarray, probability: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Cell_ID", *classes.tolist()])
        for cell_id, row in zip(ids, probability):
            writer.writerow([cell_id, *[f"{float(value):.9g}" for value in row]])


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


def gate_features(anchor: np.ndarray, experts: np.ndarray) -> np.ndarray:
    all_models = np.concatenate([anchor[:, None, :], experts], axis=1)
    anchor_entropy = -(anchor * np.log(np.clip(anchor, 1e-9, None))).sum(axis=1) / math.log(anchor.shape[1])
    sorted_anchor = np.sort(anchor, axis=1)
    anchor_margin = sorted_anchor[:, -1] - sorted_anchor[:, -2]
    expert_entropy = -(experts * np.log(np.clip(experts, 1e-9, None))).sum(axis=2) / math.log(anchor.shape[1])
    anchor_prediction = anchor.argmax(axis=1)
    expert_prediction = experts.argmax(axis=2)
    disagreement = (expert_prediction != anchor_prediction[:, None]).mean(axis=1)
    row = np.arange(len(anchor))
    anchor_class_probabilities = all_models[row[:, None], np.arange(all_models.shape[1])[None, :], anchor_prediction[:, None]]
    return np.stack(
        [
            anchor_entropy,
            anchor_margin,
            expert_entropy.mean(axis=1),
            disagreement,
            anchor_class_probabilities.std(axis=1),
            experts.max(axis=2).max(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


class ConfidenceGatedClasswiseStacker(nn.Module):
    def __init__(self, n_experts: int, n_classes: int, n_gate_features: int, gate_mode: str = "learned") -> None:
        super().__init__()
        self.gate_mode = gate_mode
        self.global_raw = nn.Parameter(torch.zeros(n_experts))
        self.class_delta_raw = nn.Parameter(torch.zeros(n_experts, n_classes))
        self.class_bias_raw = nn.Parameter(torch.zeros(n_classes))
        self.gate = nn.Linear(n_gate_features, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.5)

    def forward(
        self,
        anchor_log: torch.Tensor,
        expert_log: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_weights = 0.50 * torch.tanh(self.global_raw[:, None] + self.class_delta_raw)
        correction = ((expert_log - anchor_log[:, None, :]) * class_weights[None, :, :]).sum(dim=1)
        if self.gate_mode == "fixed":
            gate = torch.ones((len(anchor_log), 1), device=anchor_log.device, dtype=anchor_log.dtype)
        else:
            gate = torch.sigmoid(self.gate(features))
        class_bias = 0.25 * torch.tanh(self.class_bias_raw)
        logits = anchor_log + gate * correction + class_bias
        return logits, gate.squeeze(1), class_weights


def fit_stacker(
    anchor: np.ndarray,
    experts: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    gate_mode: str,
) -> tuple[ConfidenceGatedClasswiseStacker, np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    set_seed(seed)
    feature_mean = features[indices].mean(axis=0).astype(np.float32)
    feature_std = np.maximum(features[indices].std(axis=0), 1e-5).astype(np.float32)
    normalized = (features[indices] - feature_mean) / feature_std
    anchor_log = torch.from_numpy(np.log(np.clip(anchor[indices], 1e-9, None))).to(device)
    expert_log = torch.from_numpy(np.log(np.clip(experts[indices], 1e-9, None))).to(device)
    feature_tensor = torch.from_numpy(normalized.astype(np.float32)).to(device)
    label_tensor = torch.from_numpy(labels[indices]).to(device)
    model = ConfidenceGatedClasswiseStacker(
        experts.shape[1], anchor.shape[1], features.shape[1], gate_mode=gate_mode
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history = []
    for epoch in range(epochs):
        model.train()
        logits, gate, _ = model(anchor_log, expert_log, feature_tensor)
        cross_entropy = nn.functional.cross_entropy(logits, label_tensor)
        gate_regularization = (
            0.01 * model.gate.weight.square().mean() + 0.005 * gate.mean()
            if gate_mode == "learned"
            else torch.zeros((), device=device)
        )
        regularization = (
            0.01 * model.global_raw.square().mean()
            + 0.12 * model.class_delta_raw.square().mean()
            + 0.05 * model.class_bias_raw.square().mean()
            + gate_regularization
        )
        loss = cross_entropy + regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.item()),
                    "cross_entropy": float(cross_entropy.item()),
                    "regularization": float(regularization.item()),
                    "mean_gate": float(gate.mean().item()),
                }
            )
    return model, feature_mean, feature_std, history


@torch.inference_mode()
def predict_stacker(
    model: ConfidenceGatedClasswiseStacker,
    anchor: np.ndarray,
    experts: np.ndarray,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    anchor_log = torch.from_numpy(np.log(np.clip(anchor, 1e-9, None))).to(device)
    expert_log = torch.from_numpy(np.log(np.clip(experts, 1e-9, None))).to(device)
    normalized = ((features - feature_mean) / feature_std).astype(np.float32)
    feature_tensor = torch.from_numpy(normalized).to(device)
    logits, gate, weights = model(anchor_log, expert_log, feature_tensor)
    return (
        torch.softmax(logits, dim=1).cpu().numpy(),
        gate.cpu().numpy(),
        weights.cpu().numpy(),
    )


def exact_mcnemar(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = min(a_only, b_only)
    probability = sum(math.comb(n, value) for value in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * probability))


def paired_comparison(y: np.ndarray, candidate: np.ndarray, anchor: np.ndarray, seed: int) -> dict[str, float | int]:
    candidate_correct = candidate.argmax(axis=1) == y
    anchor_correct = anchor.argmax(axis=1) == y
    differences = candidate_correct.astype(np.float64) - anchor_correct.astype(np.float64)
    candidate_only = int(np.sum(candidate_correct & ~anchor_correct))
    anchor_only = int(np.sum(~candidate_correct & anchor_correct))
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([differences[rng.integers(0, len(y), len(y))].mean() for _ in range(5000)])
    return {
        "candidate_accuracy": float(candidate_correct.mean()),
        "anchor_accuracy": float(anchor_correct.mean()),
        "delta_candidate_minus_anchor": float(differences.mean()),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "candidate_only_correct": candidate_only,
        "anchor_only_correct": anchor_only,
        "mcnemar_exact_p_value": exact_mcnemar(candidate_only, anchor_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a nested cross-fitted confidence-gated classwise logit stacker.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "confidence_gated_logit_stacker")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--gate-mode", choices=["learned", "fixed"], default="learned")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    started = time.time()
    device = torch.device("cuda")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = np.load(data_dir / "ids.npy", allow_pickle=False).astype(str)
    train_positions = np.load(data_dir / "train_positions.npy", allow_pickle=False)
    test_positions = np.load(data_dir / "test_positions.npy", allow_pickle=False)
    folds_all = np.load(data_dir / "folds.npy", allow_pickle=False)
    folds = folds_all[train_positions]
    labels = np.load(data_dir / "labels.npy", allow_pickle=False).astype(np.int64)[train_positions]
    classes = np.load(data_dir / "class_names.npy", allow_pickle=False).astype(str)
    train_ids = ids[train_positions]
    test_ids = ids[test_positions]

    train_probabilities = {}
    test_probabilities = {}
    for name, source_dir in MODEL_SOURCES.items():
        stem = PROBABILITY_STEMS[name]
        train_probabilities[name] = read_probability_csv(
            source_dir / f"oof_probabilities_{stem}.csv", train_ids, classes
        )
        test_probabilities[name] = read_probability_csv(
            source_dir / f"test_probabilities_{stem}.csv", test_ids, classes
        )
        print(f"loaded={name}", flush=True)

    anchor_train = train_probabilities["anchor_external_primary"]
    anchor_test = test_probabilities["anchor_external_primary"]
    expert_train = np.stack([train_probabilities[name] for name in EXPERT_NAMES], axis=1)
    expert_test = np.stack([test_probabilities[name] for name in EXPERT_NAMES], axis=1)
    features_train = gate_features(anchor_train, expert_train)
    features_test = gate_features(anchor_test, expert_test)

    oof = np.zeros_like(anchor_train)
    oof_gate = np.zeros(len(anchor_train), dtype=np.float32)
    fold_details = []
    history_rows = []
    for fold in np.sort(np.unique(folds)):
        fit_indices = np.flatnonzero(folds != fold)
        held_indices = np.flatnonzero(folds == fold)
        model, feature_mean, feature_std, history = fit_stacker(
            anchor_train,
            expert_train,
            features_train,
            labels,
            fit_indices,
            args.epochs,
            args.learning_rate,
            args.seed + int(fold),
            device,
            args.gate_mode,
        )
        probability, gate, weights = predict_stacker(
            model,
            anchor_train[held_indices],
            expert_train[held_indices],
            features_train[held_indices],
            feature_mean,
            feature_std,
            device,
        )
        oof[held_indices] = probability
        oof_gate[held_indices] = gate
        fold_details.append(
            {
                "fold": int(fold),
                "n_fit": int(len(fit_indices)),
                "n_held_out": int(len(held_indices)),
                "mean_gate": float(gate.mean()),
                "held_out_metrics": metric_values(labels[held_indices], probability),
                "anchor_held_out_metrics": metric_values(labels[held_indices], anchor_train[held_indices]),
                "mean_classwise_weights": {
                    name: float(weights[index].mean()) for index, name in enumerate(EXPERT_NAMES)
                },
            }
        )
        history_rows.extend([{"fold": int(fold), **row} for row in history])
        print(
            f"fold={fold} gate={gate.mean():.4f} accuracy={fold_details[-1]['held_out_metrics']['accuracy']:.4f} "
            f"anchor={fold_details[-1]['anchor_held_out_metrics']['accuracy']:.4f}",
            flush=True,
        )

    all_indices = np.arange(len(labels))
    final_model, final_mean, final_std, final_history = fit_stacker(
        anchor_train,
        expert_train,
        features_train,
        labels,
        all_indices,
        args.epochs,
        args.learning_rate,
        args.seed + 1000,
        device,
        args.gate_mode,
    )
    test_probability, test_gate, final_weights = predict_stacker(
        final_model,
        anchor_test,
        expert_test,
        features_test,
        final_mean,
        final_std,
        device,
    )
    save_probability_csv(output_dir / "oof_probabilities_confidence_gated_stacker.csv", train_ids, classes, oof)
    save_probability_csv(output_dir / "test_probabilities_confidence_gated_stacker.csv", test_ids, classes, test_probability)
    with (output_dir / "submission_confidence_gated_stacker.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Cell_ID", "CellType"])
        for cell_id, prediction in zip(test_ids, classes[test_probability.argmax(axis=1)]):
            writer.writerow([cell_id, prediction])

    with (output_dir / "classwise_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", *EXPERT_NAMES])
        for class_id, class_name in enumerate(classes):
            writer.writerow([class_name, *[float(final_weights[index, class_id]) for index in range(len(EXPERT_NAMES))]])
    gate_coefficients = final_model.gate.weight.detach().cpu().numpy()[0]
    with (output_dir / "gate_coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "coefficient"])
        writer.writerow(["intercept", float(final_model.gate.bias.detach().cpu().item())])
        for name, coefficient in zip(GATE_FEATURE_NAMES, gate_coefficients):
            writer.writerow([name, float(coefficient)])
    with (output_dir / "oof_gate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Cell_ID", "fold", "gate"])
        for cell_id, fold, gate in zip(train_ids, folds, oof_gate):
            writer.writerow([cell_id, int(fold), float(gate)])
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = history_rows + [{"fold": "final", **row} for row in final_history]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "configuration": {
            "anchor": "external_primary_crossfit",
            "experts": EXPERT_NAMES,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gate_mode": args.gate_mode,
            "validation": "outer three-fold cross-fit; each held fold is absent from stacker and gate fitting",
            "formula": "anchor_logit + confidence_gate * classwise_expert_correction + bounded_class_bias",
            "parameterization": "global expert weight plus strongly regularized class delta; bounded by 0.5*tanh",
        },
        "anchor_metrics": metric_values(labels, anchor_train),
        "stacker_metrics": metric_values(labels, oof),
        "paired_vs_anchor": paired_comparison(labels, oof, anchor_train, args.seed),
        "mean_oof_gate": float(oof_gate.mean()),
        "mean_test_gate": float(test_gate.mean()),
        "fold_details": fold_details,
        "final_mean_classwise_weights": {
            name: float(final_weights[index].mean()) for index, name in enumerate(EXPERT_NAMES)
        },
        "runtime_seconds": float(time.time() - started),
        "device": torch.cuda.get_device_name(0),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
