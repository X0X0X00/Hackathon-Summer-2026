"""Approved Zenodo MERFISH reference: provenance, exclusion, and alignment.

Source: Zenodo 18039571, MERFISH_spinal_cord_resolved_0718.h5ad
Expected MD5: ce06f62c0ec4973581dae17bb76f0cd9
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from merfish60.io import repo_root
from merfish60.official_contract import sha256_file


EXPECTED_MD5 = "ce06f62c0ec4973581dae17bb76f0cd9"
EXPECTED_N_OBS = 146621
EXPECTED_N_VARS = 500
REFERENCE_H5AD_REL = "work/external/MERFISH_spinal_cord_resolved_0718.h5ad"
LABEL_OBS_COL = "MERFISH cell type annotation"
ZENOD_RECORD = "18039571"
HISTORICAL_USABLE_ROWS = 136574
HISTORICAL_VECTOR_DUPES = 47
MATERIAL_USABLE_DELTA = 100
MATERIAL_DUPE_DELTA = 20


class ReferenceContractError(Exception):
    pass


def reference_h5ad_path(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / REFERENCE_H5AD_REL


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_md5(path: Optional[Path] = None) -> str:
    path = path or reference_h5ad_path()
    if not path.is_file():
        raise ReferenceContractError("missing reference file: {}".format(path))
    digest = md5_file(path)
    if digest != EXPECTED_MD5:
        raise ReferenceContractError(
            "reference MD5 {} != expected {}".format(digest, EXPECTED_MD5)
        )
    return digest


def norm_label(value) -> str:
    """Same-team normalization: spaces and hyphens to underscores."""
    return str(value).replace(" ", "_").replace("-", "_")


def _row_fingerprints(matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(matrix, dtype=np.int64)
    width = matrix.dtype.itemsize * matrix.shape[1]
    view = matrix.view(np.dtype((np.void, width))).ravel()
    if hasattr(pd.util, "hash_array"):
        return pd.util.hash_array(view)
    return pd.util.hash_pandas_object(pd.Series(view), index=False).to_numpy()


def _to_dense_int(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.int64)


def audit_reference(data, class_names: Sequence[str], root: Optional[Path] = None) -> dict:
    """MD5, exclusion, gene order, label coverage, train-only Laminae/Segment map."""
    import anndata as ad

    root = root or repo_root()
    path = reference_h5ad_path(root)
    digest = verify_reference_md5(path)
    adata = ad.read_h5ad(path)
    if adata.shape != (EXPECTED_N_OBS, EXPECTED_N_VARS):
        raise ReferenceContractError(
            "reference shape {} != ({}, {})".format(
                adata.shape, EXPECTED_N_OBS, EXPECTED_N_VARS
            )
        )

    genes = list(data.counts_train.columns)
    var_names = [str(v) for v in adata.var_names.tolist()]
    missing_genes = [g for g in genes if g not in set(var_names)]
    if missing_genes:
        raise ReferenceContractError(
            "competition genes missing from reference: {}".format(missing_genes[:10])
        )
    if LABEL_OBS_COL not in adata.obs.columns:
        raise ReferenceContractError("missing label column {}".format(LABEL_OBS_COL))

    ext_ids = np.array(adata.obs_names.astype(str).tolist(), dtype=object)
    train_ids = np.array([str(v) for v in data.counts_train.index.tolist()], dtype=object)
    test_ids = np.array([str(v) for v in data.counts_test.index.tolist()], dtype=object)
    train_set = set(train_ids.tolist())
    test_set = set(test_ids.tolist())
    ext_set = set(ext_ids.tolist())
    in_train = np.fromiter((cid in train_set for cid in ext_ids), dtype=bool, count=len(ext_ids))
    in_test = np.fromiter((cid in test_set for cid in ext_ids), dtype=bool, count=len(ext_ids))
    n_train_overlap = int(in_train.sum())
    n_test_overlap = int(in_test.sum())
    if train_set - ext_set or test_set - ext_set:
        raise ReferenceContractError("some competition Cell_IDs are absent from the deposit")
    aligned = adata[:, genes]
    aligned_genes = [str(v) for v in aligned.var_names.tolist()]
    if aligned_genes != genes:
        raise ReferenceContractError("reference gene order does not match competition columns")
    x200 = _to_dense_int(aligned.X)
    if x200.shape != (EXPECTED_N_OBS, len(genes)):
        raise ReferenceContractError("aligned count matrix shape {}".format(x200.shape))

    # Alignment check on TRAIN cells only (counts + labels + coords).
    train_pos = {cid: i for i, cid in enumerate(ext_ids.tolist())}
    tr_idx = np.array([train_pos[cid] for cid in train_ids.tolist()], dtype=np.int64)
    train_counts = data.counts_train.loc[train_ids, genes].to_numpy(dtype=np.int64)
    if not np.array_equal(x200[tr_idx], train_counts):
        raise ReferenceContractError("train 200-gene counts do not match the deposit")
    raw_train_lab = adata.obs[LABEL_OBS_COL].astype(str).to_numpy()[tr_idx]
    norm_train_lab = np.array([norm_label(v) for v in raw_train_lab], dtype=object)
    official_train_lab = data.y_train.loc[train_ids].astype(str).to_numpy()
    if not np.array_equal(norm_train_lab, official_train_lab):
        raise ReferenceContractError("normalized train labels do not match meta_train")
    coords = adata.obs[["center_x", "center_y"]].to_numpy(dtype=np.float64)[tr_idx]
    official_xy = data.meta_train.loc[train_ids, ["center_x", "center_y"]].to_numpy(dtype=np.float64)
    if not np.allclose(coords, official_xy, atol=1e-3):
        raise ReferenceContractError("train coordinates do not match the deposit")

    ref_mask = ~(in_train | in_test)
    comp_counts = np.vstack(
        [
            data.counts_train.loc[train_ids, genes].to_numpy(dtype=np.int64),
            data.counts_test.loc[test_ids, genes].to_numpy(dtype=np.int64),
        ]
    )
    ref_fp = _row_fingerprints(x200[ref_mask])
    comp_fp = set(_row_fingerprints(comp_counts).tolist())
    dup = np.array([int(v) in comp_fp for v in ref_fp], dtype=bool)
    n_dup = int(dup.sum())
    keep_ids = ext_ids[ref_mask][~dup]
    is_ref = np.zeros(len(ext_ids), dtype=bool)
    keep_set = set(keep_ids.tolist())
    is_ref[np.fromiter((cid in keep_set for cid in ext_ids), dtype=bool, count=len(ext_ids))] = True

    raw_labels = adata.obs[LABEL_OBS_COL].astype(str)
    n_raw_unique = int(raw_labels.nunique(dropna=False))
    norm_all = np.array([norm_label(v) for v in raw_labels.to_numpy()], dtype=object)
    class_set = set(str(c) for c in class_names)
    mapped = np.array([lab in class_set for lab in norm_all], dtype=bool)
    unmapped = sorted(set(norm_all[~mapped].tolist()))
    ref_norm = norm_all[is_ref]
    ref_mapped = np.array([lab in class_set for lab in ref_norm], dtype=bool)
    missing_from_ref = sorted(c for c in class_names if c not in set(ref_norm[ref_mapped].tolist()))
    n_ref_unmapped = int((~ref_mapped).sum())
    if n_ref_unmapped:
        raise ReferenceContractError(
            "usable reference contains unmapped labels: n={}".format(n_ref_unmapped)
        )

    # Laminae -> Segment on TRAIN only.
    lam = adata.obs["Laminae"].astype(str).replace("nan", np.nan).to_numpy()
    train_seg = data.meta_train.loc[train_ids, "Segment"]
    train_lam = pd.Series(lam[tr_idx], index=train_ids)
    pair = pd.DataFrame({"num": train_seg.to_numpy(), "txt": train_lam.to_numpy()}).dropna()
    pair["num"] = pd.to_numeric(pair["num"], errors="coerce")
    pair = pair.dropna().drop_duplicates()
    nunique_txt = int(pair.groupby("txt")["num"].nunique().max()) if len(pair) else 0
    nunique_num = int(pair.groupby("num")["txt"].nunique().max()) if len(pair) else 0
    lam2seg = {str(t): float(n) for t, n in zip(pair["txt"], pair["num"])}
    mapped_train = train_lam.map(lam2seg)
    comparable = train_seg.notna() & mapped_train.notna()
    n_match = int((pd.to_numeric(train_seg[comparable], errors="coerce") == mapped_train[comparable]).sum())
    n_comp = int(comparable.sum())
    segment_map_valid = bool(
        nunique_txt == 1 and nunique_num == 1 and n_comp > 0 and n_match == n_comp
    )

    def _map_train(txt_values, official_num) -> dict:
        pair_df = pd.DataFrame({"num": official_num, "txt": txt_values}).dropna()
        pair_df["num"] = pd.to_numeric(pair_df["num"], errors="coerce")
        pair_df = pair_df.dropna().drop_duplicates()
        txt_n = int(pair_df.groupby("txt")["num"].nunique().max()) if len(pair_df) else 0
        num_n = int(pair_df.groupby("num")["txt"].nunique().max()) if len(pair_df) else 0
        mapping = {str(t): float(n) for t, n in zip(pair_df["txt"], pair_df["num"])}
        return {"map": mapping, "nunique_txt_to_num": txt_n, "nunique_num_to_txt": num_n, "n_pairs": len(mapping)}

    region_txt = adata.obs["Region"].astype(str).replace("nan", np.nan).to_numpy()
    region_audit = _map_train(region_txt[tr_idx], data.meta_train.loc[train_ids, "Region"].to_numpy())
    axial_txt = adata.obs["Axial level"].astype(str).replace("nan", np.nan).to_numpy()
    ap_audit = _map_train(axial_txt[tr_idx], data.meta_train.loc[train_ids, "AP_position"].to_numpy())
    te_idx = np.array([train_pos[cid] for cid in test_ids.tolist()], dtype=np.int64)

    y_codes = np.full(len(ext_ids), -1, dtype=np.int64)
    index = {str(name): i for i, name in enumerate(class_names)}
    mapped_codes = np.array([index.get(str(lab), -1) for lab in norm_all], dtype=np.int64)
    y_codes[is_ref] = mapped_codes[is_ref]
    y_codes[tr_idx] = np.array([index[str(v)] for v in official_train_lab], dtype=np.int64)

    n_usable = int(is_ref.sum())
    payload = {
        "source": "Zenodo {}".format(ZENOD_RECORD),
        "filename": REFERENCE_H5AD_REL,
        "md5": digest,
        "sha256": sha256_file(path),
        "byte_size": int(path.stat().st_size),
        "raw_n_obs": int(adata.n_obs),
        "raw_n_vars": int(adata.n_vars),
        "obs_columns": list(adata.obs.columns.astype(str)),
        "var_names_head": var_names[:10],
        "label_column": LABEL_OBS_COL,
        "competition_genes_aligned": genes,
        "n_competition_genes": len(genes),
        "gene_order_sha256": hashlib.sha256(",".join(genes).encode("utf-8")).hexdigest(),
        "gene_order_ok": True,
        "n_train_id_overlaps_removed": n_train_overlap,
        "n_test_id_overlaps_removed": n_test_overlap,
        "n_exact_vector_duplicates_removed": n_dup,
        "n_usable_reference_rows": n_usable,
        "historical_usable_rows": HISTORICAL_USABLE_ROWS,
        "historical_vector_dupes": HISTORICAL_VECTOR_DUPES,
        "usable_delta_vs_historical": n_usable - HISTORICAL_USABLE_ROWS,
        "dup_delta_vs_historical": n_dup - HISTORICAL_VECTOR_DUPES,
        "exclusion_materially_different": bool(
            abs(n_usable - HISTORICAL_USABLE_ROWS) > MATERIAL_USABLE_DELTA
            or abs(n_dup - HISTORICAL_VECTOR_DUPES) > MATERIAL_DUPE_DELTA
        ),
        "n_raw_unique_labels": n_raw_unique,
        "raw_unique_labels": sorted(set(str(v) for v in raw_labels.tolist())),
        "n_normalized_labels_matching_taxonomy": int(mapped.sum()),
        "unmapped_labels": unmapped,
        "competition_classes_missing_from_reference": missing_from_ref,
        "reference_label_coverage": int(len(class_names) - len(missing_from_ref)),
        "n_usable_reference_unmapped_rows": n_ref_unmapped,
        "laminae_segment_nunique_txt_to_num": nunique_txt,
        "laminae_segment_nunique_num_to_txt": nunique_num,
        "laminae_segment_train_comparable": n_comp,
        "laminae_segment_train_matches": n_match,
        "laminae_segment_map_n": len(lam2seg),
        "segment_map_valid": segment_map_valid,
        "region_map_n": region_audit["n_pairs"],
        "region_map_nunique_txt_to_num": region_audit["nunique_txt_to_num"],
        "region_map_nunique_num_to_txt": region_audit["nunique_num_to_txt"],
        "region_map_valid": bool(
            region_audit["nunique_txt_to_num"] == 1 and region_audit["nunique_num_to_txt"] == 1
        ),
        "ap_map_n": ap_audit["n_pairs"],
        "ap_map_nunique_txt_to_num": ap_audit["nunique_txt_to_num"],
        "ap_map_nunique_num_to_txt": ap_audit["nunique_num_to_txt"],
        "ap_map_valid": bool(ap_audit["nunique_txt_to_num"] == 1),
        "test_target_used": False,
        "train_count_alignment_ok": True,
        "train_label_alignment_ok": True,
        "train_coordinate_alignment_ok": True,
        "_ext_ids": ext_ids,
        "_is_ref": is_ref,
        "_is_train": in_train,
        "_is_test": in_test,
        "_x200": x200,
        "_y_codes": y_codes,
        "_adata": adata,
        "_lam2seg": lam2seg,
        "_region_map": region_audit["map"],
        "_ap_map": ap_audit["map"],
        "_genes": genes,
        "_tr_idx": tr_idx,
        "_te_idx": te_idx,
    }
    return payload
