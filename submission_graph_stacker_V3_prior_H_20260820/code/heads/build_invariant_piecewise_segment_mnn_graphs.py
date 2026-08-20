from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_invariant_mnn_gonogo import (
    ThreeViewInvariantEncoder,
    encode_inputs,
    make_inputs_numpy,
    normalized_expression,
)
from run_reliability_piecewise_soft_slot_gonogo import (
    fit_zero_reliability,
    fold_parameters,
    piecewise_transform,
    tri_state_transform,
)

DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)
OLD_OUTPUT = ROOT / "outputs" / "piecewise_mnn_segment_centered"
OUTPUT_DIR = ROOT / "outputs" / "invariant_piecewise_mnn_segment_centered"
GRAPH_ROOT = OUTPUT_DIR / "graphs"
SHARED_DIR = OUTPUT_DIR / "graph_shared"
CHECKPOINT = ROOT / "outputs" / "invariant_mnn_gonogo" / "three_view_invariant_encoder.pt"
PARAMETERS = (
    ROOT
    / "outputs"
    / "gene_threshold_activation_ablation"
    / "learned_activation_parameters.csv"
)
N_NEIGHBORS = 8
QUERY_BATCH_SIZE = 384
ALPHA = 1.0


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def fold_train_positions(
    folds: np.ndarray, reference_positions: np.ndarray, fold: int, n_cells: int
) -> np.ndarray:
    if folds.shape[0] == n_cells:
        return reference_positions[folds[reference_positions] != fold]
    if folds.shape[0] == reference_positions.shape[0]:
        return reference_positions[folds != fold]
    raise ValueError(
        f"Unexpected folds length {folds.shape[0]} for "
        f"{n_cells} cells and {reference_positions.shape[0]} references"
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(denom, 1e-8)).astype(np.float32)


def reliable_evidence(
    detected: np.ndarray, zero_reliability: np.ndarray
) -> np.ndarray:
    zero_reliability = np.asarray(zero_reliability)
    if zero_reliability.shape == detected.shape:
        return np.logical_or(detected, zero_reliability >= 0.5)
    return detected.astype(bool, copy=False)


def directed_segment_knn(
    invariant: np.ndarray,
    piecewise: np.ndarray,
    evidence: np.ndarray,
    positions: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    n_segment = positions.shape[0]
    if n_segment <= 1:
        return np.empty((n_segment, 0), dtype=np.int32)

    k = min(N_NEIGHBORS, n_segment - 1)
    inv_t = torch.from_numpy(invariant[positions]).to(device)
    piece_t = torch.from_numpy(piecewise[positions]).to(device)
    evidence_t = torch.from_numpy(evidence[positions].astype(np.float32)).to(device)
    evidence_count = evidence_t.sum(dim=1).clamp_min(1.0)
    result = np.empty((n_segment, k), dtype=np.int32)

    with torch.inference_mode():
        for start in range(0, n_segment, QUERY_BATCH_SIZE):
            stop = min(start + QUERY_BATCH_SIZE, n_segment)
            inv_score = inv_t[start:stop] @ inv_t.T
            piece_score = piece_t[start:stop] @ piece_t.T
            overlap = evidence_t[start:stop] @ evidence_t.T
            quality = overlap / torch.sqrt(
                evidence_count[start:stop, None] * evidence_count[None, :]
            )
            quality = quality * torch.clamp(overlap / 20.0, min=0.0, max=1.0)
            score = inv_score + ALPHA * quality * piece_score

            row = torch.arange(stop - start, device=device)
            col = torch.arange(start, stop, device=device)
            score[row, col] = -torch.inf
            result[start:stop] = (
                torch.topk(score, k=k, dim=1, largest=True, sorted=True)
                .indices.cpu()
                .numpy()
                .astype(np.int32)
            )
            del inv_score, piece_score, overlap, quality, score

    return result


def mutualize(
    directed: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_segment = directed.shape[0]
    neighbors = np.full((n_segment, N_NEIGHBORS), -1, dtype=np.int64)
    ranks = np.full((n_segment, N_NEIGHBORS), -1, dtype=np.int16)
    candidate_sets = [set(row.tolist()) for row in directed]

    for source in range(n_segment):
        slot = 0
        for source_rank, target in enumerate(directed[source]):
            if source in candidate_sets[int(target)]:
                neighbors[source, slot] = positions[int(target)]
                ranks[source, slot] = source_rank
                slot += 1
                if slot == N_NEIGHBORS:
                    break
    return neighbors, ranks


def make_edge_features(
    neighbors: np.ndarray,
    ranks: np.ndarray,
    invariant: np.ndarray,
    piecewise: np.ndarray,
    evidence: np.ndarray,
) -> np.ndarray:
    n_cells = neighbors.shape[0]
    features = np.zeros((n_cells, N_NEIGHBORS, 5), dtype=np.float32)
    source, slot = np.where(neighbors >= 0)
    if source.size == 0:
        return features

    target = neighbors[source, slot]
    inv_cos = np.sum(invariant[source] * invariant[target], axis=1)
    piece_cos = np.sum(piecewise[source] * piecewise[target], axis=1)

    overlap = np.sum(evidence[source] & evidence[target], axis=1).astype(np.float32)
    source_count = np.maximum(evidence[source].sum(axis=1), 1)
    target_count = np.maximum(evidence[target].sum(axis=1), 1)
    quality = overlap / np.sqrt(source_count * target_count)
    quality *= np.clip(overlap / 20.0, 0.0, 1.0)
    combined = inv_cos + ALPHA * quality * piece_cos
    reciprocal_rank = 1.0 / (ranks[source, slot].astype(np.float32) + 1.0)

    features[source, slot, 0] = inv_cos
    features[source, slot, 1] = piece_cos
    features[source, slot, 2] = quality
    features[source, slot, 3] = combined
    features[source, slot, 4] = reciprocal_rank
    return features


def graph_metrics(
    neighbors: np.ndarray,
    labels: np.ndarray,
    detected: np.ndarray,
    class_names: np.ndarray,
) -> dict[str, object]:
    valid = neighbors >= 0
    degree = valid.sum(axis=1)
    coverage = float(np.mean(degree > 0))
    mean_degree = float(np.mean(degree))

    source, slot = np.where(valid)
    target = neighbors[source, slot]
    labelled_edges = (labels[source] >= 0) & (labels[target] >= 0)
    if np.any(labelled_edges):
        edge_same = float(
            np.mean(labels[source[labelled_edges]] == labels[target[labelled_edges]])
        )
    else:
        edge_same = float("nan")

    labelled_positions = np.flatnonzero(labels >= 0)
    votes_correct: list[bool] = []
    vote_by_class: dict[int, list[bool]] = {}
    for position in labelled_positions:
        local = neighbors[position]
        local = local[(local >= 0) & (labels[local] >= 0)]
        if local.size == 0:
            continue
        counts = np.bincount(labels[local].astype(np.int64))
        prediction = int(np.argmax(counts))
        correct = prediction == int(labels[position])
        votes_correct.append(correct)
        vote_by_class.setdefault(int(labels[position]), []).append(correct)

    vote_accuracy = float(np.mean(votes_correct)) if votes_correct else float("nan")
    vote_balanced = (
        float(np.mean([np.mean(v) for v in vote_by_class.values()]))
        if vote_by_class
        else float("nan")
    )

    depth = detected.sum(axis=1)
    labelled_depth = depth[labelled_positions]
    median_depth = float(np.median(labelled_depth))
    low = labelled_positions[labelled_depth <= median_depth]
    high = labelled_positions[labelled_depth > median_depth]

    def edge_purity_for(positions: np.ndarray) -> float:
        same: list[bool] = []
        for position in positions:
            local = neighbors[position]
            local = local[(local >= 0) & (labels[local] >= 0)]
            if local.size:
                same.extend((labels[local] == labels[position]).tolist())
        return float(np.mean(same)) if same else float("nan")

    low_purity = edge_purity_for(low)
    high_purity = edge_purity_for(high)

    names = np.asarray(class_names).astype(str)
    lower_names = np.char.lower(names)
    oligo = np.flatnonzero(np.char.find(lower_names, "oligo") >= 0)
    progenitor = np.flatnonzero(
        (np.char.find(lower_names, "progenitor") >= 0)
        | (np.char.find(lower_names, "opc") >= 0)
    )
    schwann = np.flatnonzero(np.char.find(lower_names, "schwann") >= 0)
    peripheral = np.flatnonzero(
        (np.char.find(lower_names, "peripheral") >= 0)
        | (np.char.find(lower_names, "satellite") >= 0)
    )

    def directional_cross(source_classes: np.ndarray, target_classes: np.ndarray) -> float:
        if source_classes.size == 0 or target_classes.size == 0:
            return float("nan")
        source_mask = np.isin(labels[source], source_classes) & labelled_edges
        chosen = source[source_mask]
        chosen_slot = slot[source_mask]
        if chosen.size == 0:
            return float("nan")
        chosen_target = neighbors[chosen, chosen_slot]
        return float(np.mean(np.isin(labels[chosen_target], target_classes)))

    return {
        "coverage": coverage,
        "mean_degree": mean_degree,
        "edge_same_label_rate": edge_same,
        "neighbor_vote_accuracy": vote_accuracy,
        "neighbor_vote_balanced_accuracy": vote_balanced,
        "low_depth_edge_purity": low_purity,
        "high_depth_edge_purity": high_purity,
        "depth_purity_gap_low_minus_high": low_purity - high_purity,
        "oligo_to_progenitor_cross_rate": directional_cross(oligo, progenitor),
        "schwann_to_peripheral_cross_rate": directional_cross(schwann, peripheral),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This experiment requires a CUDA GPU")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(OLD_OUTPUT / "graph_shared", SHARED_DIR, dirs_exist_ok=True)

    counts = np.load(DATA_DIR / "raw_counts.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "labels.npy")
    folds = np.load(DATA_DIR / "folds.npy")
    gene_names = np.load(DATA_DIR / "gene_names.npy", allow_pickle=True)
    reference_positions = np.load(DATA_DIR / "reference_positions.npy")
    class_names = np.load(DATA_DIR / "class_names.npy", allow_pickle=True)
    segment_codes = np.load(SHARED_DIR / "segment_codes.npy")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    expression, _, _ = normalized_expression(
        counts,
        reference_positions,
        mean=np.asarray(checkpoint["expression_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["expression_std"], dtype=np.float32),
    )
    detected = np.asarray(counts) > 0
    depth = detected.sum(axis=1).astype(np.float32)
    inputs = make_inputs_numpy(
        expression,
        detected,
        float(checkpoint["depth_mean"]),
        float(checkpoint["depth_std"]),
    )

    model = ThreeViewInvariantEncoder(expression.shape[1], 128).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    invariant = normalize_rows(encode_inputs(model, inputs, batch_size=2048))
    del inputs, model
    torch.cuda.empty_cache()

    parameter_frame = pd.read_csv(PARAMETERS)
    all_metrics: dict[str, object] = {
        "device": torch.cuda.get_device_name(0),
        "n_cells": int(expression.shape[0]),
        "n_genes": int(expression.shape[1]),
        "n_neighbors": N_NEIGHBORS,
        "alpha": ALPHA,
        "score": "invariant_cosine + alpha * reliable_overlap * piecewise_cosine",
        "aggregation": "continuous expression minus all-cell Segment mean",
        "folds": {},
    }

    for fold in range(3):
        train_positions = fold_train_positions(
            folds, reference_positions, fold, expression.shape[0]
        )
        parameters = fold_parameters(
            parameter_frame, np.asarray(gene_names).astype(str), fold
        )
        naive_piecewise = piecewise_transform(
            expression, detected, parameters, train_positions
        )
        zero_reliability, _, reliability_audit = fit_zero_reliability(
            detected,
            depth,
            segment_codes,
            train_positions,
            floor=0.15,
            smoothing=20.0,
        )
        piecewise = tri_state_transform(
            naive_piecewise,
            expression,
            detected,
            zero_reliability,
            parameters,
            train_positions,
        )
        piecewise = normalize_rows(piecewise.astype(np.float32, copy=False))
        evidence = reliable_evidence(detected, zero_reliability)

        neighbors = np.full(
            (expression.shape[0], N_NEIGHBORS), -1, dtype=np.int64
        )
        ranks = np.full(
            (expression.shape[0], N_NEIGHBORS), -1, dtype=np.int16
        )
        for segment in np.unique(segment_codes):
            positions = np.flatnonzero(segment_codes == segment)
            directed = directed_segment_knn(
                invariant, piecewise, evidence, positions, device
            )
            local_neighbors, local_ranks = mutualize(directed, positions)
            neighbors[positions] = local_neighbors
            ranks[positions] = local_ranks

        edge_features = make_edge_features(
            neighbors, ranks, invariant, piecewise, evidence
        )
        fold_dir = GRAPH_ROOT / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.save(fold_dir / "neighbors.npy", neighbors)
        np.save(fold_dir / "edge_features.npy", edge_features)

        metrics = graph_metrics(neighbors, labels, detected, class_names)
        metrics["train_reference_count"] = int(train_positions.shape[0])
        metrics["reliability_audit"] = reliability_audit
        all_metrics["folds"][str(fold)] = metrics
        print(
            f"fold={fold} metrics={json.dumps(metrics, default=json_default)}",
            flush=True,
        )

        del naive_piecewise, zero_reliability, piecewise, evidence
        del neighbors, ranks, edge_features
        torch.cuda.empty_cache()

    with (OUTPUT_DIR / "graph_build_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, default=json_default)
    print(f"saved={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
