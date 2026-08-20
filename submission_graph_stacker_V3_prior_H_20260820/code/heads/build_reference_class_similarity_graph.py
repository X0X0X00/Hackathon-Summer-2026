from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "reference_class_similarity_graph"
K_NEIGHBORS = 3


def cosine_matrix(values: np.ndarray) -> np.ndarray:
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return normalized @ normalized.T


def main() -> None:
    raw_counts = np.load(CACHE_DIR / "raw_counts.npy", mmap_mode="r")
    labels = np.load(CACHE_DIR / "labels.npy", mmap_mode="r")
    reference_positions = np.load(CACHE_DIR / "reference_positions.npy")
    class_names = np.load(CACHE_DIR / "class_names.npy").astype(str)

    x_raw = np.asarray(raw_counts[reference_positions], dtype=np.float64)
    y = np.asarray(labels[reference_positions], dtype=np.int64)
    library_size = np.sum(x_raw, axis=1, keepdims=True)
    x = np.log1p(10_000.0 * x_raw / np.maximum(library_size, 1.0))
    centroids = np.stack(
        [np.mean(x[y == class_index], axis=0) for class_index in range(len(class_names))]
    )
    similarities = cosine_matrix(centroids)
    np.fill_diagonal(similarities, -np.inf)

    neighbor_indices = np.argsort(similarities, axis=1)[:, -K_NEIGHBORS:]
    distances = np.maximum(1.0 - similarities, 0.0)
    local_scale = np.take_along_axis(distances, neighbor_indices, axis=1).max(axis=1)
    local_scale = np.maximum(local_scale, 1e-6)
    affinity = np.zeros_like(similarities)
    neighbor_sets = [set(row.tolist()) for row in neighbor_indices]
    edge_rows = []
    for left in range(len(class_names)):
        for right in range(left + 1, len(class_names)):
            if right not in neighbor_sets[left] or left not in neighbor_sets[right]:
                continue
            distance = distances[left, right]
            weight = float(
                np.exp(-(distance * distance) / (local_scale[left] * local_scale[right]))
            )
            affinity[left, right] = affinity[right, left] = weight
            edge_rows.append(
                {
                    "left_class": class_names[left],
                    "right_class": class_names[right],
                    "centroid_cosine": float(similarities[left, right]),
                    "adaptive_affinity": weight,
                }
            )

    positive = affinity[affinity > 0]
    if len(positive):
        affinity /= np.mean(positive)
        for row in edge_rows:
            row["normalized_affinity"] = row.pop("adaptive_affinity") / float(
                np.mean(positive)
            )
    else:
        raise RuntimeError("Mutual-neighbor graph has no edges")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "reference_class_graph.npz",
        class_names=class_names,
        affinity=affinity.astype(np.float32),
        centroids=centroids.astype(np.float32),
    )
    with (OUTPUT_DIR / "reference_class_graph_edges.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(edge_rows, key=lambda row: row["normalized_affinity"], reverse=True))

    class_index = {name: index for index, name in enumerate(class_names)}
    audit_pairs = [
        ("oligodendrocyte_progenitor_2", "oligodendrocyte_1"),
        ("oligodendrocyte_precursor_cell", "oligodendrocyte_progenitor_1"),
        ("astrocyte_1", "astrocyte_2"),
    ]
    summary = {
        "source": "external reference cells only",
        "n_reference": int(len(reference_positions)),
        "n_classes": int(len(class_names)),
        "k_neighbors": K_NEIGHBORS,
        "n_undirected_edges": int(len(edge_rows)),
        "isolated_classes": [
            class_names[index]
            for index in range(len(class_names))
            if not np.any(affinity[index] > 0)
        ],
        "audit_pairs": {
            f"{left} <-> {right}": {
                "cosine": float(similarities[class_index[left], class_index[right]]),
                "connected": bool(affinity[class_index[left], class_index[right]] > 0),
                "weight": float(affinity[class_index[left], class_index[right]]),
            }
            for left, right in audit_pairs
        },
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
