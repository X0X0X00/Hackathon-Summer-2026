from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from hierarchical_glia import COARSE_GROUPS, EXTERNAL_REFERENCE, FAMILIES
from train_model import OFFICIAL


V1_ROOT = Path(__file__).resolve().parents[1]
SOURCE = V1_ROOT / "cache" / "external_MERFISH_spinal_cord.h5ad"
PROVENANCE = V1_ROOT / "reports" / "external_glia_reference_provenance.json"
EXPECTED_MD5 = "ce06f62c0ec4973581dae17bb76f0cd9"
MAX_PER_CLASS = 2500
SEED = 2026


def decode(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def normalize_label(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_")


def expression_features(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    totals = values.sum(axis=1, keepdims=True)
    detected = (values > 0).sum(axis=1, keepdims=True).astype(np.float32)
    log_raw = np.log1p(values)
    log_normalized = np.log1p(values / np.maximum(totals, 1) * 100.0)
    return np.hstack([log_raw, log_normalized, np.log1p(totals), detected]).astype(np.float32)


def row_fingerprints(values: np.ndarray) -> np.ndarray:
    return pd.util.hash_pandas_object(pd.DataFrame(values), index=False).to_numpy(dtype=np.uint64)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    source_md5 = md5(SOURCE)
    if source_md5 != EXPECTED_MD5:
        raise RuntimeError(f"external file checksum mismatch: {source_md5} != {EXPECTED_MD5}")

    counts_train = pd.read_csv(OFFICIAL / "data" / "counts_train.csv", index_col=0)
    counts_test = pd.read_csv(OFFICIAL / "data" / "counts_test.csv", index_col=0)
    contest_counts = pd.concat([counts_train, counts_test], axis=0)
    contest_ids = set(contest_counts.index.astype(str))
    contest_fingerprints = set(row_fingerprints(contest_counts.to_numpy()))
    genes = counts_train.columns.astype(str).tolist()
    target_labels = sorted({
        label
        for groups in (FAMILIES, COARSE_GROUPS)
        for labels in groups.values()
        for label in labels
    })
    rng = np.random.default_rng(SEED)

    with h5py.File(SOURCE, "r") as handle:
        external_ids = decode(handle["obs"]["_index"])
        label_group = handle["obs"]["MERFISH cell type annotation"]
        categories = decode(label_group["categories"])
        codes = np.asarray(label_group["codes"], dtype=int)
        labels = np.asarray([normalize_label(categories[code]) for code in codes])
        external_genes = decode(handle["var"]["_index"])
        gene_lookup = {gene: index for index, gene in enumerate(external_genes)}
        missing_genes = [gene for gene in genes if gene not in gene_lookup]
        if missing_genes:
            raise RuntimeError(f"external reference is missing competition genes: {missing_genes}")
        gene_indices = np.asarray([gene_lookup[gene] for gene in genes], dtype=int)

        id_overlap = np.isin(external_ids, list(contest_ids))
        eligible = (~id_overlap) & np.isin(labels, target_labels)
        sampled_indices = []
        before_counts = {}
        for label in target_labels:
            candidates = np.flatnonzero(eligible & (labels == label))
            before_counts[label] = int(len(candidates))
            if len(candidates) > MAX_PER_CLASS:
                candidates = rng.choice(candidates, size=MAX_PER_CLASS, replace=False)
            sampled_indices.append(np.sort(candidates))
        sampled_indices = np.sort(np.concatenate(sampled_indices))

        x_group = handle["X"]
        matrix = csr_matrix(
            (
                np.asarray(x_group["data"]),
                np.asarray(x_group["indices"]),
                np.asarray(x_group["indptr"]),
            ),
            shape=tuple(x_group.attrs["shape"]),
        )
        sampled_counts = matrix[sampled_indices][:, gene_indices].toarray()
        sampled_labels = labels[sampled_indices]
        sampled_ids = external_ids[sampled_indices]

    sampled_fingerprints = row_fingerprints(sampled_counts)
    fingerprint_overlap = np.isin(sampled_fingerprints, list(contest_fingerprints))
    keep = ~fingerprint_overlap
    sampled_counts = sampled_counts[keep]
    sampled_labels = sampled_labels[keep]
    sampled_ids = sampled_ids[keep]
    if np.isin(sampled_ids, list(contest_ids)).any():
        raise RuntimeError("contest Cell_ID remained after external-reference filtering")

    features = expression_features(sampled_counts)
    np.savez_compressed(
        EXTERNAL_REFERENCE,
        features=features,
        labels=sampled_labels.astype("U64"),
        cell_ids=sampled_ids.astype("U32"),
        genes=np.asarray(genes, dtype="U32"),
    )
    after_counts = pd.Series(sampled_labels).value_counts().sort_index().astype(int).to_dict()
    report = {
        "source": "https://doi.org/10.5281/zenodo.18039571",
        "source_file": SOURCE.name,
        "source_md5": source_md5,
        "source_cells": int(len(external_ids)),
        "competition_ids_excluded": int(id_overlap.sum()),
        "competition_expression_fingerprints_excluded_after_id_filter": int(fingerprint_overlap.sum()),
        "competition_genes_aligned": len(genes),
        "max_reference_cells_per_class": MAX_PER_CLASS,
        "eligible_counts_before_cap": before_counts,
        "reference_counts_after_cap_and_filters": after_counts,
        "reference_cells": int(len(sampled_labels)),
        "direct_test_label_lookup_used": False,
    }
    PROVENANCE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
