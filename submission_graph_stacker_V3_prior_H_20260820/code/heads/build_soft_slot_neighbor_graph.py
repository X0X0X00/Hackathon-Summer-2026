from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from test_competitive_slot_neighbor_purity import (
    CompetitiveSlotAutoencoder,
    encode_all,
    majority_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full soft-slot reference-neighbor graph.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--slot-model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "competitive_slot_neighbor_purity" / "competitive_slot_autoencoder.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--slot-weight", type=float, default=4.0)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--encoding-batch-size", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    started = time.time()
    device = torch.device("cuda")
    data_dir = args.data_dir.resolve()
    base_graph_dir = data_dir / "mnn_residual"
    output_dir = (args.output_dir or (data_dir / "soft_slot_neighbors")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.slot_model.resolve(), map_location="cpu", weights_only=False)
    configuration = checkpoint["configuration"]
    expression_np = np.load(base_graph_dir / "expression.npy", mmap_mode="r", allow_pickle=False)
    expression = torch.from_numpy(np.array(expression_np, dtype=np.float32, copy=True)).to(device)
    model = CompetitiveSlotAutoencoder(
        input_dim=expression.shape[1],
        hidden_dim=int(configuration["hidden_dim"]),
        slot_dim=int(configuration["slot_dim"]),
        n_slots=int(configuration["n_slots"]),
        temperature=float(configuration["temperature"]),
        dropout=float(configuration["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    latent_np, assignment_np = encode_all(model, expression, args.encoding_batch_size)

    reference_positions = np.load(data_dir / "reference_positions.npy", allow_pickle=False)
    train_positions = np.load(data_dir / "train_positions.npy", allow_pickle=False)
    labels = np.load(data_dir / "labels.npy", allow_pickle=False).astype(np.int64)
    raw = np.load(data_dir / "raw_counts.npy", mmap_mode="r", allow_pickle=False)
    n_cells = len(expression_np)
    inverse_reference = np.full(n_cells, -1, dtype=np.int64)
    inverse_reference[reference_positions] = np.arange(len(reference_positions), dtype=np.int64)

    latent = F.normalize(torch.from_numpy(latent_np).to(device), dim=1)
    assignment_score = F.normalize(torch.from_numpy(assignment_np).to(device), dim=1)
    reference_latent = latent[torch.from_numpy(reference_positions).to(device)]
    reference_assignment = assignment_score[torch.from_numpy(reference_positions).to(device)]
    neighbors = np.full((n_cells, args.n_neighbors), -1, dtype=np.int64)
    edge_features = np.zeros((n_cells, args.n_neighbors, 5), dtype=np.float32)

    counts = np.asarray(raw, dtype=np.float32)
    log_total = np.log1p(counts.sum(axis=1)).astype(np.float32)
    detected = (counts > 0).mean(axis=1).astype(np.float32)
    log_total = (log_total - log_total.mean()) / max(float(log_total.std()), 1e-6)
    detected = (detected - detected.mean()) / max(float(detected.std()), 1e-6)

    print(
        f"device={torch.cuda.get_device_name(0)} cells={n_cells} reference={len(reference_positions)} "
        f"k={args.n_neighbors} slot_weight={args.slot_weight}",
        flush=True,
    )
    with torch.inference_mode():
        for start in range(0, n_cells, args.query_batch_size):
            end = min(start + args.query_batch_size, n_cells)
            latent_similarity = latent[start:end] @ reference_latent.T
            slot_similarity = assignment_score[start:end] @ reference_assignment.T
            score = latent_similarity + args.slot_weight * slot_similarity
            own_reference = inverse_reference[start:end]
            has_self = own_reference >= 0
            if np.any(has_self):
                rows = torch.from_numpy(np.flatnonzero(has_self)).to(device)
                columns = torch.from_numpy(own_reference[has_self]).to(device)
                score[rows, columns] = -torch.inf
            _, local_neighbors = torch.topk(
                score, k=args.n_neighbors, dim=1, largest=True, sorted=True
            )
            selected_latent = torch.gather(latent_similarity, 1, local_neighbors).cpu().numpy()
            selected_slots = torch.gather(slot_similarity, 1, local_neighbors).cpu().numpy()
            local_neighbors_np = local_neighbors.cpu().numpy()
            global_neighbors = reference_positions[local_neighbors_np]
            neighbors[start:end] = global_neighbors
            source_positions = np.arange(start, end)[:, None]
            edge_features[start:end, :, 0] = selected_latent
            edge_features[start:end, :, 1] = selected_slots
            edge_features[start:end, :, 2] = selected_latent + args.slot_weight * selected_slots
            edge_features[start:end, :, 3] = np.abs(log_total[source_positions] - log_total[global_neighbors])
            edge_features[start:end, :, 4] = np.abs(detected[source_positions] - detected[global_neighbors])
            if start == 0 or end == n_cells or (start // args.query_batch_size) % 200 == 0:
                print(f"searched={end}/{n_cells}", flush=True)
            del latent_similarity, slot_similarity, score, local_neighbors

    assignment = torch.from_numpy(assignment_np).to(device)
    slot_expression_means = assignment.T @ expression
    slot_expression_means /= assignment.sum(dim=0).clamp_min(1e-8).unsqueeze(1)
    context_center = np.lib.format.open_memmap(
        output_dir / "context_center.npy",
        mode="w+",
        dtype=np.float32,
        shape=expression_np.shape,
    )
    with torch.inference_mode():
        for start in range(0, n_cells, args.encoding_batch_size):
            end = min(start + args.encoding_batch_size, n_cells)
            context_center[start:end] = (assignment[start:end] @ slot_expression_means).cpu().numpy()
    context_center.flush()
    del context_center

    np.save(output_dir / "neighbors.npy", neighbors, allow_pickle=False)
    np.save(output_dir / "edge_features.npy", edge_features, allow_pickle=False)
    np.save(output_dir / "slot_assignment.npy", assignment_np.astype(np.float16), allow_pickle=False)

    train_neighbor_labels = labels[neighbors[train_positions]]
    purity = majority_metrics(train_neighbor_labels, labels[train_positions])
    report = {
        "n_cells": int(n_cells),
        "n_reference_candidates": int(len(reference_positions)),
        "n_neighbors": int(args.n_neighbors),
        "slot_weight": float(args.slot_weight),
        "neighbor_rule": "global directed top-k reference neighbors by latent cosine plus weighted soft-slot overlap",
        "self_edge_rule": "reference-cell self candidate is excluded",
        "context_rule": "neighbor expression residual subtracts the query's soft-assignment-weighted slot mean expression",
        "labels_used_for_graph": False,
        "hyperparameter_note": "slot_weight=4 was selected exploratorily from competition-train purity",
        "competition_train_neighbor_purity": purity,
        "runtime_seconds": float(time.time() - started),
        "device": torch.cuda.get_device_name(0),
    }
    (output_dir / "graph_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
