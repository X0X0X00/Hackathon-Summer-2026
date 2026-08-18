from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from prepare_external_reference import EXPECTED_MD5, md5, normalize_label, row_fingerprints
from train_model import LABEL, OFFICIAL


V1_ROOT = Path(__file__).resolve().parents[1]
SOURCE = V1_ROOT / "cache" / "external_MERFISH_spinal_cord.h5ad"
OUTPUT = V1_ROOT / "cache" / "external_neuronal_reference.npz"
PROVENANCE = V1_ROOT / "reports" / "external_neuronal_reference_provenance.json"
MAX_PER_CLASS = 2500
SEED = 2026


def decode(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def expression_features(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    totals = values.sum(axis=1, keepdims=True)
    detected = (values > 0).sum(axis=1, keepdims=True).astype(np.float32)
    log_raw = np.log1p(values)
    log_normalized = np.log1p(values / np.maximum(totals, 1) * 100.0)
    return np.hstack([log_raw, log_normalized, np.log1p(totals), detected]).astype(np.float32)


def main() -> None:
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    source_md5 = md5(SOURCE)
    if source_md5 != EXPECTED_MD5:
        raise RuntimeError(f"external file checksum mismatch: {source_md5} != {EXPECTED_MD5}")

    data = OFFICIAL / "data"
    counts_train = pd.read_csv(data / "counts_train.csv", index_col=0)
    counts_test = pd.read_csv(data / "counts_test.csv", index_col=0)
    meta_train = pd.read_csv(data / "meta_train.csv", index_col=0)
    contest_counts = pd.concat([counts_train, counts_test], axis=0)
    contest_ids = set(contest_counts.index.astype(str))
    contest_fingerprints = set(row_fingerprints(contest_counts.to_numpy()))
    genes = counts_train.columns.astype(str).tolist()
    competition_labels = sorted(meta_train[LABEL].astype(str).unique())
    target_labels = [
        label
        for label in competition_labels
        if label.startswith("DH_ex_")
        or label.startswith("DH_in_")
        or label.startswith("MV_ex_")
        or label.startswith("MV_in_")
        or label.startswith("M_ex_")
        or label.startswith("M_in_")
        or label.startswith("VH_in_")
        or label == "cholinergic_interneuron"
        or "motoneuron" in label
    ]
    rng = np.random.default_rng(SEED)

    with h5py.File(SOURCE, "r") as handle:
        external_ids = decode(handle["obs"]["_index"])
        label_group = handle["obs"]["MERFISH cell type annotation"]
        categories = decode(label_group["categories"])
        codes = np.asarray(label_group["codes"], dtype=int)
        labels = np.asarray([normalize_label(categories[code]) for code in codes])
        external_genes = decode(handle["var"]["_index"])
        gene_lookup = {gene: index for index, gene in enumerate(external_genes)}
        missing = [gene for gene in genes if gene not in gene_lookup]
        if missing:
            raise RuntimeError(f"external reference missing competition genes: {missing}")
        gene_indices = np.asarray([gene_lookup[gene] for gene in genes], dtype=int)

        eligible = (~np.isin(external_ids, list(contest_ids))) & np.isin(labels, target_labels)
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

    fingerprint_overlap = np.isin(
        row_fingerprints(sampled_counts), list(contest_fingerprints)
    )
    keep = ~fingerprint_overlap
    sampled_counts = sampled_counts[keep]
    sampled_labels = sampled_labels[keep]
    sampled_ids = sampled_ids[keep]
    if np.isin(sampled_ids, list(contest_ids)).any():
        raise RuntimeError("competition Cell_ID remained in neuronal reference")

    features = expression_features(sampled_counts)
    np.savez_compressed(
        OUTPUT,
        features=features,
        labels=sampled_labels.astype("U"),
        cell_ids=sampled_ids.astype("U"),
        genes=np.asarray(genes, dtype="U"),
    )
    after_counts = {
        label: int((sampled_labels == label).sum()) for label in target_labels
    }
    provenance = {
        "source": "https://doi.org/10.5281/zenodo.18039571",
        "source_file": SOURCE.name,
        "source_md5": source_md5,
        "competition_ids_removed": 10000,
        "fingerprint_matches_removed": int(fingerprint_overlap.sum()),
        "target_labels": target_labels,
        "before_cap_counts": before_counts,
        "after_filter_counts": after_counts,
        "reference_cells": int(len(sampled_labels)),
        "genes": int(len(genes)),
        "max_per_class": MAX_PER_CLASS,
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
