from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from test_competitive_slot_neighbor_purity import CompetitiveSlotAutoencoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Go/No-Go audit for reliability-aware Piecewise Soft-slot MNN."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--activation-parameters",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gene_threshold_activation_ablation"
        / "learned_activation_parameters.csv",
    )
    parser.add_argument(
        "--raw-slot-model",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "competitive_slot_neighbor_purity"
        / "competitive_slot_autoencoder.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "reliability_piecewise_soft_slot_gonogo",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--encoding-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--slot-weight", type=float, default=4.0)
    parser.add_argument("--minimum-reliable-overlap", type=int, default=20)
    parser.add_argument("--thinning-rate", type=float, default=0.30)
    parser.add_argument("--negative-detection-floor", type=float, default=0.70)
    parser.add_argument("--detectability-smoothing", type=float, default=20.0)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fold_parameters(
    parameters: pd.DataFrame,
    gene_names: np.ndarray,
    fold: int,
) -> dict[str, np.ndarray]:
    subset = parameters[
        parameters["variant"].eq("gene_piecewise_gate_detection")
        & parameters["fold"].eq(fold)
    ].copy()
    if len(subset) == 0:
        raise ValueError(f"No gene_piecewise_gate_detection parameters for fold {fold}")
    subset = subset.set_index("gene")
    defaults = {
        "standardized_threshold": 0.0,
        "low_slope": 1.0,
        "high_slope": 1.0,
        "gate_amplitude": 0.0,
        "gate_temperature": 1.0,
        "detection_weight": 0.0,
    }
    result: dict[str, np.ndarray] = {}
    for column, default in defaults.items():
        values = subset[column].reindex(gene_names).fillna(default).to_numpy(np.float32)
        result[column] = values
    return result


def piecewise_transform(
    expression: np.ndarray,
    detected: np.ndarray,
    parameters: dict[str, np.ndarray],
    reference_positions: np.ndarray,
) -> np.ndarray:
    threshold = parameters["standardized_threshold"][None, :]
    low = parameters["low_slope"][None, :]
    high = parameters["high_slope"][None, :]
    amplitude = parameters["gate_amplitude"][None, :]
    temperature = np.maximum(parameters["gate_temperature"][None, :], 0.05)
    detection_weight = parameters["detection_weight"][None, :]
    delta = expression - threshold
    sigmoid = 1.0 / (1.0 + np.exp(np.clip(-delta / temperature, -30.0, 30.0)))
    transformed = (
        low * expression
        + (high - low) * np.maximum(delta, 0.0)
        + amplitude * sigmoid
        + detection_weight * detected.astype(np.float32)
    ).astype(np.float32)
    mean = transformed[reference_positions].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = transformed[reference_positions].std(axis=0, dtype=np.float64).astype(np.float32)
    transformed -= mean[None, :]
    transformed /= np.maximum(std[None, :], 1e-4)
    return transformed


def fit_zero_reliability(
    detected: np.ndarray,
    depth: np.ndarray,
    segment_codes: np.ndarray,
    reference_positions: np.ndarray,
    floor: float,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    reference_depth = depth[reference_positions]
    quantiles = np.unique(np.quantile(reference_depth, [0.2, 0.4, 0.6, 0.8]))
    depth_bin = np.searchsorted(quantiles, depth, side="right").astype(np.int16)
    unique_segments, segment_index = np.unique(segment_codes, return_inverse=True)
    n_bins = len(quantiles) + 1
    global_rate = detected[reference_positions].mean(axis=0, dtype=np.float64)
    rates = np.zeros((len(unique_segments), n_bins, detected.shape[1]), dtype=np.float32)
    counts = np.zeros((len(unique_segments), n_bins), dtype=np.int64)
    ref_segment = segment_index[reference_positions]
    ref_bin = depth_bin[reference_positions]
    for segment in range(len(unique_segments)):
        for bin_index in range(n_bins):
            positions = reference_positions[(ref_segment == segment) & (ref_bin == bin_index)]
            counts[segment, bin_index] = len(positions)
            if len(positions):
                positive = detected[positions].sum(axis=0, dtype=np.float64)
                rates[segment, bin_index] = (
                    (positive + smoothing * global_rate) / (len(positions) + smoothing)
                ).astype(np.float32)
            else:
                rates[segment, bin_index] = global_rate.astype(np.float32)
    expected_detection = rates[segment_index, depth_bin]
    zero_reliability = np.clip(
        (expected_detection - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0
    ).astype(np.float32)
    zero_reliability[detected] = 0.0
    reliable = detected | (zero_reliability > 0.0)
    diagnostics = {
        "depth_quantiles": quantiles.astype(float).tolist(),
        "segment_depth_group_minimum_reference_cells": int(counts.min()),
        "segment_depth_group_median_reference_cells": float(np.median(counts)),
        "mean_reliable_genes_per_cell": float(reliable.sum(axis=1).mean()),
        "median_reliable_genes_per_cell": float(np.median(reliable.sum(axis=1))),
        "mean_reliable_negative_genes_per_cell": float((zero_reliability > 0).sum(axis=1).mean()),
        "fraction_cells_with_at_least_20_reliable_genes": float((reliable.sum(axis=1) >= 20).mean()),
    }
    return zero_reliability, reliable, diagnostics


def tri_state_transform(
    naive_piecewise: np.ndarray,
    expression: np.ndarray,
    detected: np.ndarray,
    zero_reliability: np.ndarray,
    parameters: dict[str, np.ndarray],
    reference_positions: np.ndarray,
) -> np.ndarray:
    threshold = parameters["standardized_threshold"][None, :]
    temperature = np.maximum(parameters["gate_temperature"][None, :], 0.05)
    gate = 1.0 / (
        1.0 + np.exp(np.clip(-(expression - threshold) / temperature, -30.0, 30.0))
    )
    positive = gate + 0.25 * np.maximum(naive_piecewise, 0.0)
    tri_state = np.zeros_like(expression, dtype=np.float32)
    tri_state[detected] = positive[detected]
    absent = ~detected
    tri_state[absent] = -0.5 * zero_reliability[absent]
    # RMS scaling preserves Unknown == 0, unlike mean centering.
    rms = np.sqrt(
        np.mean(np.square(tri_state[reference_positions]), axis=0, dtype=np.float64)
    ).astype(np.float32)
    tri_state /= np.maximum(rms[None, :], 1e-4)
    return tri_state


def make_thinned_detection(
    detected: np.ndarray,
    thinning_rate: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    thinned = detected.copy()
    for start in range(0, len(detected), 4096):
        end = min(start + 4096, len(detected))
        drop = rng.random((end - start, detected.shape[1]), dtype=np.float32) < thinning_rate
        thinned[start:end] &= ~drop
    return thinned


def zero_expression_values(
    expression: np.ndarray,
    detected: np.ndarray,
    reference_positions: np.ndarray,
) -> np.ndarray:
    values = np.zeros(expression.shape[1], dtype=np.float32)
    reference_expression = expression[reference_positions]
    reference_detected = detected[reference_positions]
    for gene in range(expression.shape[1]):
        zero = ~reference_detected[:, gene]
        values[gene] = (
            float(np.median(reference_expression[zero, gene])) if zero.any() else 0.0
        )
    return values


def train_slot_model(
    feature: np.ndarray,
    reconstruction_mask: np.ndarray | None,
    positive_mask: np.ndarray,
    args: argparse.Namespace,
    robust: bool,
    seed: int,
) -> tuple[CompetitiveSlotAutoencoder, list[dict[str, float | int]]]:
    set_seed(seed)
    device = torch.device("cuda")
    feature_tensor = torch.from_numpy(feature).to(device)
    positive_tensor = torch.from_numpy(positive_mask).to(device)
    reliable_tensor = (
        torch.from_numpy(reconstruction_mask).to(device)
        if reconstruction_mask is not None
        else None
    )
    model = CompetitiveSlotAutoencoder(
        input_dim=feature.shape[1],
        hidden_dim=256,
        slot_dim=128,
        n_slots=64,
        temperature=0.15,
        dropout=0.10,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    indices = torch.arange(len(feature_tensor), device=device)
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        model.train()
        permutation = indices[torch.randperm(len(indices), device=device)]
        totals = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "consistency": 0.0,
            "orthogonal": 0.0,
            "balance": 0.0,
            "entropy": 0.0,
        }
        seen = 0
        for start in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[start : start + args.batch_size]
            clean = feature_tensor[batch_indices]
            positive = positive_tensor[batch_indices]
            reliable = reliable_tensor[batch_indices] if reliable_tensor is not None else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if robust:
                    view1 = clean.masked_fill(
                        positive & (torch.rand_like(clean) < 0.20), 0.0
                    )
                    view2 = clean.masked_fill(
                        positive & (torch.rand_like(clean) < 0.40), 0.0
                    )
                    reconstruction1, latent1, assignment1, _ = model(view1)
                    reconstruction2, latent2, assignment2, _ = model(view2)
                    reconstruction_element = 0.5 * (
                        F.smooth_l1_loss(reconstruction1, clean, reduction="none")
                        + F.smooth_l1_loss(reconstruction2, clean, reduction="none")
                    )
                    if reliable is None:
                        reconstruction_loss = reconstruction_element.mean()
                    else:
                        reconstruction_loss = (
                            reconstruction_element * reliable
                        ).sum() / reliable.sum().clamp_min(1)
                    latent_consistency = 1.0 - F.cosine_similarity(latent1, latent2, dim=1).mean()
                    mean_assignment = 0.5 * (assignment1 + assignment2)
                    assignment_consistency = 0.5 * (
                        F.kl_div(
                            torch.log(assignment1.clamp_min(1e-8)),
                            mean_assignment,
                            reduction="batchmean",
                        )
                        + F.kl_div(
                            torch.log(assignment2.clamp_min(1e-8)),
                            mean_assignment,
                            reduction="batchmean",
                        )
                    )
                    consistency = latent_consistency + assignment_consistency
                    assignment = mean_assignment
                else:
                    reconstruction, _, assignment, _ = model(clean)
                    reconstruction_element = F.smooth_l1_loss(
                        reconstruction, clean, reduction="none"
                    )
                    if reliable is None:
                        reconstruction_loss = reconstruction_element.mean()
                    else:
                        reconstruction_loss = (
                            reconstruction_element * reliable
                        ).sum() / reliable.sum().clamp_min(1)
                    consistency = torch.zeros((), device=device)
                orthogonal = model.orthogonality_loss()
                slot_share = assignment.mean(dim=0)
                balance = ((slot_share - 1.0 / 64) ** 2).mean() / ((1.0 / 64) ** 2)
                entropy = -(
                    assignment * torch.log(assignment.clamp_min(1e-8))
                ).sum(dim=1).mean() / math.log(64)
                loss = (
                    reconstruction_loss
                    + 0.20 * orthogonal
                    + 0.05 * balance
                    + 0.02 * entropy
                    + args.consistency_weight * consistency
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch_indices)
            for key, value in [
                ("loss", loss),
                ("reconstruction", reconstruction_loss),
                ("consistency", consistency),
                ("orthogonal", orthogonal),
                ("balance", balance),
                ("entropy", entropy),
            ]:
                totals[key] += float(value.detach().item()) * count
            seen += count
        row = {"epoch": epoch + 1, **{key: value / seen for key, value in totals.items()}}
        history.append(row)
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={row['loss']:.5f} "
            f"recon={row['reconstruction']:.5f} consistency={row['consistency']:.5f}",
            flush=True,
        )
    del feature_tensor, positive_tensor, reliable_tensor
    torch.cuda.empty_cache()
    return model, history


@torch.inference_mode()
def encode_model(
    model: CompetitiveSlotAutoencoder,
    feature: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    latents: list[np.ndarray] = []
    assignments: list[np.ndarray] = []
    for start in range(0, len(feature), batch_size):
        batch = torch.from_numpy(feature[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            latent, assignment, _ = model.encode(batch)
        latents.append(latent.float().cpu().numpy())
        assignments.append(assignment.float().cpu().numpy())
    return np.concatenate(latents), np.concatenate(assignments)


@torch.inference_mode()
def segment_neighbor_search(
    latent: np.ndarray,
    assignment: np.ndarray,
    segment_codes: np.ndarray,
    reference_positions: np.ndarray,
    train_positions: np.ndarray,
    n_neighbors: int,
    slot_weight: float,
    reliable: np.ndarray | None,
    minimum_reliable_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda")
    knn = np.full((len(train_positions), n_neighbors), -1, dtype=np.int32)
    mnn = np.full_like(knn, -1)
    train_segments = segment_codes[train_positions]
    reference_segments = segment_codes[reference_positions]
    for segment in np.unique(train_segments):
        query_local = np.flatnonzero(train_segments == segment)
        reference_local = np.flatnonzero(reference_segments == segment)
        if len(query_local) == 0 or len(reference_local) == 0:
            continue
        query_global = train_positions[query_local]
        reference_global = reference_positions[reference_local]
        query_latent = F.normalize(torch.from_numpy(latent[query_global]).to(device), dim=1)
        reference_latent = F.normalize(torch.from_numpy(latent[reference_global]).to(device), dim=1)
        query_slot = F.normalize(torch.from_numpy(assignment[query_global]).to(device), dim=1)
        reference_slot = F.normalize(torch.from_numpy(assignment[reference_global]).to(device), dim=1)
        score = query_latent @ reference_latent.T + slot_weight * (query_slot @ reference_slot.T)
        if reliable is not None:
            query_reliable = torch.from_numpy(reliable[query_global].astype(np.float32)).to(device)
            reference_reliable = torch.from_numpy(reliable[reference_global].astype(np.float32)).to(device)
            overlap = query_reliable @ reference_reliable.T
            score = score.masked_fill(overlap < minimum_reliable_overlap, -torch.inf)
        k_reference = min(n_neighbors, len(reference_global))
        forward_score, forward = torch.topk(score, k=k_reference, dim=1, largest=True, sorted=True)
        forward_np = forward.cpu().numpy()
        forward_valid = torch.isfinite(forward_score).cpu().numpy()
        for row_index, local_references in enumerate(forward_np):
            valid = forward_valid[row_index]
            selected = reference_global[local_references[valid]]
            knn[query_local[row_index], : len(selected)] = selected
        k_train = min(n_neighbors, len(query_global))
        reverse_score, reverse = torch.topk(score.T, k=k_train, dim=1, largest=True, sorted=False)
        reverse_np = reverse.cpu().numpy()
        reverse_valid = torch.isfinite(reverse_score).cpu().numpy()
        for local_query, local_references in enumerate(forward_np):
            write = 0
            for valid_forward, local_reference in zip(forward_valid[local_query], local_references):
                if not valid_forward:
                    continue
                reverse_targets = reverse_np[local_reference, reverse_valid[local_reference]]
                if np.any(reverse_targets == local_query):
                    mnn[query_local[local_query], write] = reference_global[local_reference]
                    write += 1
                    if write == n_neighbors:
                        break
        del score, query_latent, reference_latent, query_slot, reference_slot
    return knn, mnn


def neighbor_metrics(
    neighbors: np.ndarray,
    labels: np.ndarray,
    train_positions: np.ndarray,
    depth: np.ndarray,
    class_names: np.ndarray,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    target_labels = labels[train_positions]
    valid = neighbors >= 0
    neighbor_labels = np.full_like(neighbors, -1, dtype=np.int64)
    neighbor_labels[valid] = labels[neighbors[valid]]
    same = valid & (neighbor_labels == target_labels[:, None])
    covered = valid.any(axis=1)
    vote = np.full(len(train_positions), -1, dtype=np.int64)
    per_cell_purity = np.full(len(train_positions), np.nan, dtype=np.float64)
    for row in np.flatnonzero(covered):
        row_labels = neighbor_labels[row, valid[row]]
        vote[row] = np.bincount(row_labels).argmax()
        per_cell_purity[row] = np.mean(row_labels == target_labels[row])
    class_rows: list[dict[str, object]] = []
    class_purity: list[float] = []
    class_vote_recall: list[float] = []
    for label in np.unique(target_labels):
        mask = target_labels == label
        class_valid = valid[mask]
        class_same = same[mask]
        edge_purity = float(class_same.sum() / max(int(class_valid.sum()), 1))
        vote_recall = float(np.mean(vote[mask & covered] == label)) if (mask & covered).any() else 0.0
        class_purity.append(edge_purity)
        class_vote_recall.append(vote_recall)
        class_rows.append(
            {
                "class": class_names[label],
                "n": int(mask.sum()),
                "edge_purity": edge_purity,
                "majority_vote_recall": vote_recall,
                "coverage": float(covered[mask].mean()),
            }
        )
    train_depth = depth[train_positions]
    low_threshold, high_threshold = np.quantile(train_depth, [0.25, 0.75])
    low = train_depth <= low_threshold
    high = train_depth >= high_threshold

    def edge_rate(mask: np.ndarray) -> float:
        return float(same[mask].sum() / max(int(valid[mask].sum()), 1))

    name_to_label = {name: index for index, name in enumerate(class_names)}
    oligodendrocyte_2 = name_to_label.get("oligodendrocyte_2", -999)
    progenitors = {
        name_to_label[name]
        for name in [
            "oligodendrocyte_precursor_cell",
            "oligodendrocyte_progenitor_1",
            "oligodendrocyte_progenitor_2",
        ]
        if name in name_to_label
    }
    oligo_mask = target_labels == oligodendrocyte_2
    oligo_valid_labels = neighbor_labels[oligo_mask & covered]
    oligo_valid = oligo_valid_labels >= 0
    oligo_to_progenitor = (
        float(np.isin(oligo_valid_labels[oligo_valid], list(progenitors)).mean())
        if oligo_valid.any()
        else 0.0
    )
    schwann = name_to_label.get("Schwann_cell", -999)
    peripheral = name_to_label.get("peripheral_glia", -999)
    peripheral_mask = np.isin(target_labels, [schwann, peripheral])
    peripheral_neighbor = neighbor_labels[peripheral_mask & covered]
    peripheral_valid = peripheral_neighbor >= 0
    peripheral_cross = 0
    peripheral_total = 0
    for local_row, target_label in enumerate(target_labels[peripheral_mask & covered]):
        row = peripheral_neighbor[local_row]
        row = row[row >= 0]
        peripheral_cross += int(np.sum((target_label == schwann) & (row == peripheral)))
        peripheral_cross += int(np.sum((target_label == peripheral) & (row == schwann)))
        peripheral_total += len(row)
    metrics = {
        "edge_count": int(valid.sum()),
        "same_label_edge_rate": float(same.sum() / max(int(valid.sum()), 1)),
        "class_balanced_edge_purity": float(np.mean(class_purity)),
        "cell_coverage": float(covered.mean()),
        "majority_vote_accuracy": float(np.mean(vote[covered] == target_labels[covered])) if covered.any() else 0.0,
        "class_balanced_majority_vote_recall": float(np.mean(class_vote_recall)),
        "low_depth_edge_purity": edge_rate(low),
        "high_depth_edge_purity": edge_rate(high),
        "low_minus_high_depth_purity": edge_rate(low) - edge_rate(high),
        "oligodendrocyte_2_to_progenitor_edge_rate": oligo_to_progenitor,
        "schwann_peripheral_cross_edge_rate": float(peripheral_cross / max(peripheral_total, 1)),
        "mean_cell_purity": float(np.nanmean(per_cell_purity)),
    }
    return metrics, class_rows


def neighbor_jaccard(original: np.ndarray, thinned: np.ndarray) -> float:
    values: list[float] = []
    for left, right in zip(original, thinned):
        a = set(int(value) for value in left if value >= 0)
        b = set(int(value) for value in right if value >= 0)
        if a or b:
            values.append(len(a & b) / len(a | b))
    return float(np.mean(values)) if values else 0.0


def assignment_diagnostics(
    assignment: np.ndarray,
    thinned_assignment: np.ndarray,
    train_positions: np.ndarray,
) -> dict[str, float]:
    clean = assignment[train_positions]
    thin = thinned_assignment[train_positions]
    return {
        "hard_slot_stability": float(np.mean(clean.argmax(axis=1) == thin.argmax(axis=1))),
        "assignment_cosine_stability": float(
            np.mean(
                np.sum(clean * thin, axis=1)
                / (
                    np.linalg.norm(clean, axis=1) * np.linalg.norm(thin, axis=1)
                    + 1e-12
                )
            )
        ),
        "mean_max_assignment_probability": float(clean.max(axis=1).mean()),
        "mean_normalized_assignment_entropy": float(
            (-(clean * np.log(np.clip(clean, 1e-12, None))).sum(axis=1) / math.log(clean.shape[1])).mean()
        ),
    }


def load_raw_model(path: Path, device: torch.device) -> CompetitiveSlotAutoencoder:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    configuration = checkpoint["configuration"]
    model = CompetitiveSlotAutoencoder(
        input_dim=200,
        hidden_dim=int(configuration["hidden_dim"]),
        slot_dim=int(configuration["slot_dim"]),
        n_slots=int(configuration["n_slots"]),
        temperature=float(configuration["temperature"]),
        dropout=float(configuration["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    set_seed(args.seed)
    started = time.time()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)

    expression = np.array(
        np.load(data_dir / "mnn_residual" / "expression.npy", mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
        copy=True,
    )
    raw_counts = np.load(data_dir / "raw_counts.npy", mmap_mode="r", allow_pickle=False)
    detected = np.asarray(raw_counts > 0)
    depth = detected.sum(axis=1).astype(np.int16)
    labels = np.load(data_dir / "labels.npy", allow_pickle=False).astype(np.int64)
    folds = np.load(data_dir / "folds.npy", allow_pickle=False).astype(np.int16)
    gene_names = np.load(data_dir / "gene_names.npy", allow_pickle=False).astype(str)
    class_names = np.load(data_dir / "class_names.npy", allow_pickle=False).astype(str)
    reference_positions = np.load(data_dir / "reference_positions.npy", allow_pickle=False)
    train_positions = np.load(data_dir / "train_positions.npy", allow_pickle=False)
    segment_codes = np.load(data_dir / "mnn_residual" / "segment_codes.npy", allow_pickle=False)
    parameters = pd.read_csv(args.activation_parameters.resolve())
    zero_values = zero_expression_values(expression, detected, reference_positions)
    thinned_detected = make_thinned_detection(
        detected, args.thinning_rate, args.seed + 9000
    )
    dropped = detected & ~thinned_detected
    zero_reliability, reliable, reliability_diagnostics = fit_zero_reliability(
        detected,
        depth,
        segment_codes,
        reference_positions,
        args.negative_detection_floor,
        args.detectability_smoothing,
    )
    thinned_reliable = reliable.copy()
    thinned_reliable[dropped] = False
    print(
        f"cells={len(expression)} genes={expression.shape[1]} train={len(train_positions)} "
        f"reference={len(reference_positions)} reliable_genes={reliability_diagnostics['mean_reliable_genes_per_cell']:.1f}",
        flush=True,
    )

    variants = [
        "raw_soft_slot",
        "naive_piecewise_soft_slot",
        "reliability_piecewise_soft_slot",
        "reliability_piecewise_consistent_soft_slot",
    ]
    neighbor_store = {
        variant: {
            "knn": np.full((len(train_positions), args.n_neighbors), -1, dtype=np.int32),
            "mnn": np.full((len(train_positions), args.n_neighbors), -1, dtype=np.int32),
            "thin_knn": np.full((len(train_positions), args.n_neighbors), -1, dtype=np.int32),
            "thin_mnn": np.full((len(train_positions), args.n_neighbors), -1, dtype=np.int32),
        }
        for variant in variants
    }
    assignment_parts: dict[str, list[dict[str, float]]] = {variant: [] for variant in variants}
    history_rows: list[dict[str, object]] = []

    # A: historical Soft-slot checkpoint, re-audited with the same Segment constraint.
    raw_model = load_raw_model(args.raw_slot_model.resolve(), device)
    raw_latent, raw_assignment = encode_model(raw_model, expression, args.encoding_batch_size)
    thinned_expression = expression.copy()
    for gene in range(expression.shape[1]):
        thinned_expression[dropped[:, gene], gene] = zero_values[gene]
    raw_thin_latent, raw_thin_assignment = encode_model(
        raw_model, thinned_expression, args.encoding_batch_size
    )
    raw_knn, raw_mnn = segment_neighbor_search(
        raw_latent,
        raw_assignment,
        segment_codes,
        reference_positions,
        train_positions,
        args.n_neighbors,
        args.slot_weight,
        None,
        args.minimum_reliable_overlap,
    )
    raw_thin_knn, raw_thin_mnn = segment_neighbor_search(
        raw_thin_latent,
        raw_thin_assignment,
        segment_codes,
        reference_positions,
        train_positions,
        args.n_neighbors,
        args.slot_weight,
        None,
        args.minimum_reliable_overlap,
    )
    neighbor_store["raw_soft_slot"] = {
        "knn": raw_knn,
        "mnn": raw_mnn,
        "thin_knn": raw_thin_knn,
        "thin_mnn": raw_thin_mnn,
    }
    assignment_parts["raw_soft_slot"].append(
        assignment_diagnostics(raw_assignment, raw_thin_assignment, train_positions)
    )
    del raw_model, raw_latent, raw_assignment, raw_thin_latent, raw_thin_assignment
    torch.cuda.empty_cache()

    for fold in [0, 1, 2]:
        fold_params = fold_parameters(parameters, gene_names, fold)
        naive = piecewise_transform(expression, detected, fold_params, reference_positions)
        thin_expression_fold = expression.copy()
        for gene in range(expression.shape[1]):
            thin_expression_fold[dropped[:, gene], gene] = zero_values[gene]
        naive_thin = piecewise_transform(
            thin_expression_fold, thinned_detected, fold_params, reference_positions
        )
        tri_state = tri_state_transform(
            naive,
            expression,
            detected,
            zero_reliability,
            fold_params,
            reference_positions,
        )
        tri_state_thin = tri_state.copy()
        tri_state_thin[dropped] = 0.0
        fold_train_mask = folds[train_positions] == fold
        for variant, feature, thin_feature, mask, robust in [
            (
                "naive_piecewise_soft_slot",
                naive,
                naive_thin,
                None,
                False,
            ),
            (
                "reliability_piecewise_soft_slot",
                tri_state,
                tri_state_thin,
                reliable,
                False,
            ),
            (
                "reliability_piecewise_consistent_soft_slot",
                tri_state,
                tri_state_thin,
                reliable,
                True,
            ),
        ]:
            print(f"variant={variant} fold={fold}", flush=True)
            model, history = train_slot_model(
                feature,
                mask,
                detected,
                args,
                robust,
                args.seed + 100 * fold + variants.index(variant),
            )
            for row in history:
                history_rows.append({"variant": variant, "fold": fold, **row})
            latent, assignment = encode_model(model, feature, args.encoding_batch_size)
            thin_latent, thin_assignment = encode_model(
                model, thin_feature, args.encoding_batch_size
            )
            search_reliable = reliable if mask is not None else None
            thin_search_reliable = thinned_reliable if mask is not None else None
            knn, mnn = segment_neighbor_search(
                latent,
                assignment,
                segment_codes,
                reference_positions,
                train_positions,
                args.n_neighbors,
                args.slot_weight,
                search_reliable,
                args.minimum_reliable_overlap,
            )
            thin_knn, thin_mnn = segment_neighbor_search(
                thin_latent,
                thin_assignment,
                segment_codes,
                reference_positions,
                train_positions,
                args.n_neighbors,
                args.slot_weight,
                thin_search_reliable,
                args.minimum_reliable_overlap,
            )
            for key, values in [
                ("knn", knn),
                ("mnn", mnn),
                ("thin_knn", thin_knn),
                ("thin_mnn", thin_mnn),
            ]:
                neighbor_store[variant][key][fold_train_mask] = values[fold_train_mask]
            assignment_parts[variant].append(
                assignment_diagnostics(
                    assignment,
                    thin_assignment,
                    train_positions[fold_train_mask],
                )
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": variant,
                    "fold": fold,
                    "labels_used_for_training": False,
                },
                output_dir / f"{variant}_fold{fold}.pt",
            )
            del model, latent, assignment, thin_latent, thin_assignment
            torch.cuda.empty_cache()
        del naive, naive_thin, tri_state, tri_state_thin, thin_expression_fold

    comparison_rows: list[dict[str, object]] = []
    class_rows_all: list[dict[str, object]] = []
    report_variants: dict[str, object] = {}
    for variant in variants:
        store = neighbor_store[variant]
        knn_metrics, knn_class = neighbor_metrics(
            store["knn"], labels, train_positions, depth, class_names
        )
        mnn_metrics, mnn_class = neighbor_metrics(
            store["mnn"], labels, train_positions, depth, class_names
        )
        thin_mnn_metrics, _ = neighbor_metrics(
            store["thin_mnn"], labels, train_positions, depth, class_names
        )
        diagnostics_keys = assignment_parts[variant][0].keys()
        assignment_summary = {
            key: float(np.mean([row[key] for row in assignment_parts[variant]]))
            for key in diagnostics_keys
        }
        stability = {
            "knn_neighbor_jaccard_after_thinning": neighbor_jaccard(
                store["knn"], store["thin_knn"]
            ),
            "mnn_neighbor_jaccard_after_thinning": neighbor_jaccard(
                store["mnn"], store["thin_mnn"]
            ),
            "mnn_purity_change_after_thinning": thin_mnn_metrics["same_label_edge_rate"]
            - mnn_metrics["same_label_edge_rate"],
        }
        report_variants[variant] = {
            "knn": knn_metrics,
            "mnn": mnn_metrics,
            "thinned_mnn": thin_mnn_metrics,
            "assignment": assignment_summary,
            "stability": stability,
        }
        comparison_rows.append(
            {
                "variant": variant,
                **{f"mnn_{key}": value for key, value in mnn_metrics.items()},
                **stability,
                **assignment_summary,
            }
        )
        for method, rows in [("knn", knn_class), ("mnn", mnn_class)]:
            class_rows_all.extend(
                {"variant": variant, "method": method, **row} for row in rows
            )
        for key, array in store.items():
            np.save(output_dir / f"{variant}_{key}_neighbors.npy", array, allow_pickle=False)

    comparison = {row["variant"]: row for row in comparison_rows}
    baseline = comparison["raw_soft_slot"]
    naive_row = comparison["naive_piecewise_soft_slot"]
    candidate = comparison["reliability_piecewise_consistent_soft_slot"]
    criteria = [
        {
            "criterion": "class_balanced_purity_vs_raw",
            "value": candidate["mnn_class_balanced_edge_purity"]
            - baseline["mnn_class_balanced_edge_purity"],
            "threshold": 0.02,
            "passed": candidate["mnn_class_balanced_edge_purity"]
            >= baseline["mnn_class_balanced_edge_purity"] + 0.02,
        },
        {
            "criterion": "thinning_jaccard_vs_naive",
            "value": candidate["mnn_neighbor_jaccard_after_thinning"]
            - naive_row["mnn_neighbor_jaccard_after_thinning"],
            "threshold": 0.10,
            "passed": candidate["mnn_neighbor_jaccard_after_thinning"]
            >= naive_row["mnn_neighbor_jaccard_after_thinning"] + 0.10,
        },
        {
            "criterion": "depth_gap_not_worse_than_raw_by_2pp",
            "value": abs(candidate["mnn_low_minus_high_depth_purity"]),
            "threshold": abs(baseline["mnn_low_minus_high_depth_purity"]) + 0.02,
            "passed": abs(candidate["mnn_low_minus_high_depth_purity"])
            <= abs(baseline["mnn_low_minus_high_depth_purity"]) + 0.02,
        },
        {
            "criterion": "oligo2_progenitor_cross_not_worse_than_raw",
            "value": candidate["mnn_oligodendrocyte_2_to_progenitor_edge_rate"],
            "threshold": baseline["mnn_oligodendrocyte_2_to_progenitor_edge_rate"],
            "passed": candidate["mnn_oligodendrocyte_2_to_progenitor_edge_rate"]
            <= baseline["mnn_oligodendrocyte_2_to_progenitor_edge_rate"],
        },
        {
            "criterion": "coverage",
            "value": candidate["mnn_cell_coverage"],
            "threshold": 0.90,
            "passed": candidate["mnn_cell_coverage"] >= 0.90,
        },
    ]
    mandatory = criteria[0]["passed"] and criteria[1]["passed"] and criteria[3]["passed"]
    passed_count = sum(bool(row["passed"]) for row in criteria)
    decision = "GO" if mandatory and passed_count >= 4 else "NO-GO"
    report = {
        "configuration": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "labels_used_for_slot_training_or_neighbor_search": False,
            "labels_used_for_final_audit_only": True,
            "piecewise_parameters_are_supervised_but_fold_matched": True,
            "neighbor_constraint": "same Segment exact reciprocal top-k",
            "qk_value_design": "Piecewise/tri-state used only for neighbor Q/K audit; future H Value remains continuous Segment-centered expression",
        },
        "historical_reference": {
            "global_soft_slot_overlap_lambda4_mnn_same_label_edge_rate": 0.35198500333829796,
            "global_soft_slot_overlap_lambda4_mnn_majority_vote_accuracy": 0.45229045809161833,
        },
        "reliability_diagnostics": reliability_diagnostics,
        "variants": report_variants,
        "go_no_go": {
            "decision": decision,
            "passed_count": passed_count,
            "required": "at least 4/5 plus balanced-purity, thinning-stability, and oligodendrocyte maturity criteria",
            "criteria": criteria,
        },
        "runtime_seconds": float(time.time() - started),
    }
    save_rows(output_dir / "variant_comparison.csv", comparison_rows)
    save_rows(output_dir / "class_metrics.csv", class_rows_all)
    save_rows(output_dir / "training_history.csv", history_rows)
    save_rows(output_dir / "go_no_go_criteria.csv", criteria)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nVARIANT COMPARISON", flush=True)
    print(pd.DataFrame(comparison_rows).to_string(index=False), flush=True)
    print("\nGO/NO-GO", flush=True)
    print(json.dumps(report["go_no_go"], ensure_ascii=False, indent=2), flush=True)
    print(f"runtime_seconds={report['runtime_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
