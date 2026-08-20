from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)


def _zscore(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    scaled = np.clip((values - mean) / std, -8.0, 8.0).astype(np.float32)
    return scaled, mean, std


def _build_segment_knn(
    features: np.ndarray,
    positions: np.ndarray,
    n_neighbors: int,
    query_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    local = torch.from_numpy(features[positions]).to(device=device, dtype=torch.float32)
    local = F.normalize(local, p=2, dim=1, eps=1e-8)
    n_local = len(positions)
    k = min(n_neighbors, max(n_local - 1, 0))
    if k <= 0:
        return (
            np.full((n_local, n_neighbors), -1, dtype=np.int32),
            np.full((n_local, n_neighbors), -np.inf, dtype=np.float32),
        )

    indices = np.full((n_local, n_neighbors), -1, dtype=np.int32)
    similarities = np.full((n_local, n_neighbors), -np.inf, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n_local, query_batch_size):
            end = min(start + query_batch_size, n_local)
            similarity = local[start:end] @ local.T
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            similarity[rows, cols] = -torch.inf
            values, neighbors = torch.topk(similarity, k=k, dim=1, largest=True, sorted=True)
            indices[start:end, :k] = neighbors.cpu().numpy().astype(np.int32)
            similarities[start:end, :k] = values.cpu().numpy().astype(np.float32)
            del similarity, values, neighbors
    del local
    return indices, similarities


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact within-Segment mutual-nearest-neighbor graph on GPU.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 146k-cell exact MNN graph")
    device = torch.device("cuda")
    started = time.time()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or (data_dir / "mnn_residual")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(data_dir / "raw_counts.npy", mmap_mode="r", allow_pickle=False)
    auxiliary_raw = np.load(data_dir / "auxiliary.npy", mmap_mode="r", allow_pickle=False)
    segments = np.load(data_dir / "segments.npy", allow_pickle=False).astype(str)
    reference_positions = np.load(data_dir / "reference_positions.npy", allow_pickle=False)
    if len(raw) != len(segments):
        raise ValueError("raw_counts and segments have different row counts")

    counts = np.asarray(raw, dtype=np.float32)
    totals = counts.sum(axis=1).astype(np.float32)
    positive_totals = totals[totals > 0]
    target_total = float(np.median(positive_totals)) if len(positive_totals) else 1.0
    normalized = counts * (target_total / np.maximum(totals, 1e-6))[:, None]
    log_expression = np.log1p(normalized).astype(np.float32)
    expression, expression_mean, expression_std = _zscore(log_expression)

    auxiliary_values = np.nan_to_num(
        np.asarray(auxiliary_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    auxiliary_mean = auxiliary_values[reference_positions].mean(axis=0, dtype=np.float64).astype(np.float32)
    auxiliary_std = auxiliary_values[reference_positions].std(axis=0, dtype=np.float64).astype(np.float32)
    auxiliary_std = np.maximum(auxiliary_std, 1e-5)
    auxiliary = np.clip((auxiliary_values - auxiliary_mean) / auxiliary_std, -8.0, 8.0).astype(np.float32)

    segment_names, segment_codes = np.unique(segments, return_inverse=True)
    segment_codes = segment_codes.astype(np.int16)
    segment_means = np.stack(
        [expression[segment_codes == code].mean(axis=0) for code in range(len(segment_names))]
    ).astype(np.float32)

    quality_total = np.log1p(totals).astype(np.float32)
    quality_detected = (counts > 0).mean(axis=1).astype(np.float32)
    quality_total = (quality_total - quality_total.mean()) / max(float(quality_total.std()), 1e-6)
    quality_detected = (quality_detected - quality_detected.mean()) / max(float(quality_detected.std()), 1e-6)

    n_cells = len(expression)
    neighbors = np.full((n_cells, args.n_neighbors), -1, dtype=np.int64)
    edge_features = np.zeros((n_cells, args.n_neighbors, 5), dtype=np.float32)
    segment_reports: list[dict[str, object]] = []

    print(
        f"device={torch.cuda.get_device_name(0)} cells={n_cells} segments={len(segment_names)} "
        f"genes={expression.shape[1]} k={args.n_neighbors}",
        flush=True,
    )
    for code, name in enumerate(segment_names):
        positions = np.flatnonzero(segment_codes == code)
        local_knn, local_sim = _build_segment_knn(
            expression,
            positions,
            n_neighbors=args.n_neighbors,
            query_batch_size=args.query_batch_size,
            device=device,
        )
        mutual_count = 0
        for source in range(len(positions)):
            write_position = 0
            for rank, target in enumerate(local_knn[source]):
                if target < 0 or not np.any(local_knn[target] == source):
                    continue
                global_source = int(positions[source])
                global_target = int(positions[target])
                neighbors[global_source, write_position] = global_target
                edge_features[global_source, write_position] = np.asarray(
                    [
                        local_sim[source, rank],
                        abs(quality_total[global_source] - quality_total[global_target]),
                        abs(quality_detected[global_source] - quality_detected[global_target]),
                        quality_total[global_target],
                        quality_detected[global_target],
                    ],
                    dtype=np.float32,
                )
                write_position += 1
                mutual_count += 1
                if write_position == args.n_neighbors:
                    break
        segment_reports.append(
            {"segment": str(name), "n_cells": int(len(positions)), "directed_mnn_edges": int(mutual_count)}
        )
        print(
            f"segment={name} cells={len(positions)} directed_mnn_edges={mutual_count}",
            flush=True,
        )

    valid = neighbors >= 0
    np.save(output_dir / "expression.npy", expression, allow_pickle=False)
    np.save(output_dir / "auxiliary.npy", auxiliary, allow_pickle=False)
    np.save(output_dir / "segment_codes.npy", segment_codes, allow_pickle=False)
    np.save(output_dir / "segment_names.npy", np.asarray(segment_names, dtype=str), allow_pickle=False)
    np.save(output_dir / "segment_means.npy", segment_means, allow_pickle=False)
    np.save(output_dir / "neighbors.npy", neighbors, allow_pickle=False)
    np.save(output_dir / "edge_features.npy", edge_features, allow_pickle=False)
    np.savez(
        output_dir / "normalization.npz",
        target_total=np.asarray(target_total, dtype=np.float32),
        expression_mean=expression_mean,
        expression_std=expression_std,
        auxiliary_mean=auxiliary_mean,
        auxiliary_std=auxiliary_std,
    )

    report = {
        "n_cells": int(n_cells),
        "n_genes": int(expression.shape[1]),
        "auxiliary_dim": int(auxiliary.shape[1]),
        "n_segments": int(len(segment_names)),
        "n_neighbors": int(args.n_neighbors),
        "directed_mnn_edges": int(valid.sum()),
        "mean_mnn_neighbors": float(valid.sum(axis=1).mean()),
        "cells_without_mnn": int((~valid.any(axis=1)).sum()),
        "neighbor_rule": "exact cosine top-k within Segment, retained only when reciprocal",
        "context_rule": "neighbor expression is centered by the mean expression of all cells in its Segment",
        "device": torch.cuda.get_device_name(0),
        "runtime_seconds": float(time.time() - started),
        "segments": segment_reports,
    }
    (output_dir / "graph_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "segments"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
