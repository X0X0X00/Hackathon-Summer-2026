"""Competition-only transductive spatial / expression features.

Graphs and PCA are fit on train+test *features* only. No labels are used.
Record this as transductive unsupervised preprocessing.

Feature layout follows the same-team work/prep.py implementation, but Cell_IDs
are loaded as lossless 19-digit strings via load_dataset().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from merfish60.io import N_GENES, N_TEST_CELLS, N_TRAIN_CELLS, TARGET_COL


K_NBR = 50
N_SPATIAL_MEAN = 10
PCA_DIM = 50
PCA_RANDOM_STATE = 0


@dataclass
class SpatialUniverse:
    """Train-then-test concatenated universe (10_000 cells)."""

    cell_ids: np.ndarray
    is_train: np.ndarray
    X_static: np.ndarray
    names: List[str]
    sp_idx: np.ndarray
    sp_dist: np.ndarray
    ex_idx: np.ndarray
    ex_dist: np.ndarray
    ei_known: np.ndarray
    pca_var_explained: float
    section_sizes: List[int]
    transductive_note: str


def _as_knn_2d(d, j, n: int, k: int):
    d = np.asarray(d)
    j = np.asarray(j)
    return d.reshape(n, k), j.reshape(n, k)


def _compact_exclude_self(idx: np.ndarray, dist: np.ndarray, k_keep: int):
    """Drop self-hits and empty slots, then keep the first k_keep neighbors."""
    n = idx.shape[0]
    out_idx = np.full((n, k_keep), -1, dtype=np.int32)
    out_dist = np.full((n, k_keep), np.inf, dtype=np.float32)
    for i in range(n):
        keep = (idx[i] >= 0) & (idx[i] != i) & np.isfinite(dist[i])
        kept_j = idx[i, keep][:k_keep]
        kept_d = dist[i, keep][:k_keep]
        out_idx[i, : len(kept_j)] = kept_j
        out_dist[i, : len(kept_d)] = kept_d
    return out_idx, out_dist


def _codes(series: pd.Series):
    cat = pd.Categorical(series)
    out = cat.codes.astype(np.float32)
    out[out < 0] = np.nan
    return out


def ei_of_label_from_train(meta_train: pd.DataFrame, class_names: Sequence[str]) -> np.ndarray:
    """Map each class to observed E/I using training labels only.

    excitatory=1, inhibitory=0, missing/non-neuronal=-1.
    """
    y = meta_train[TARGET_COL].astype(str)
    ei = meta_train["Excitatory_vs_Inhibitory"]
    mapping = {}
    for lab in class_names:
        vals = ei[y == str(lab)]
        first = None
        for v in vals.tolist():
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except (ValueError, TypeError):
                pass
            text = str(v).strip().lower()
            if text in {"", "na", "nan", "none"}:
                continue
            first = text
            break
        mapping[str(lab)] = first
    out = np.full(len(class_names), -1, dtype=np.int8)
    for i, lab in enumerate(class_names):
        token = mapping.get(str(lab))
        if token == "excitatory":
            out[i] = 1
        elif token == "inhibitory":
            out[i] = 0
    return out


def build_spatial_universe(data, class_names: Optional[Sequence[str]] = None) -> SpatialUniverse:
    """Build static features and kNN graphs on the official train-then-test universe."""
    del class_names  # labels are not used in this function
    if data.counts_train.shape != (N_TRAIN_CELLS, N_GENES):
        raise ValueError("unexpected counts_train shape {}".format(data.counts_train.shape))
    if data.counts_test.shape != (N_TEST_CELLS, N_GENES):
        raise ValueError("unexpected counts_test shape {}".format(data.counts_test.shape))

    counts = pd.concat([data.counts_train, data.counts_test], axis=0)
    meta = pd.concat([data.meta_train, data.meta_test], axis=0)
    cell_ids = np.asarray([str(v) for v in counts.index.tolist()], dtype=object)
    n = len(cell_ids)
    is_train = np.r_[
        np.ones(N_TRAIN_CELLS, dtype=bool),
        np.zeros(N_TEST_CELLS, dtype=bool),
    ]

    tot = counts.to_numpy(dtype=np.float64).sum(axis=1)
    scale = float(np.median(tot))
    lognorm = np.log1p(counts.to_numpy(dtype=np.float64) / np.maximum(tot, 1.0)[:, None] * scale)
    lognorm = lognorm.astype(np.float32)
    z = StandardScaler().fit_transform(lognorm)
    pca = PCA(n_components=PCA_DIM, random_state=PCA_RANDOM_STATE)
    pcs = pca.fit_transform(z).astype(np.float32)
    pca_var = float(pca.explained_variance_ratio_.sum())

    coords = meta[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    sp_idx = np.full((n, K_NBR), -1, dtype=np.int32)
    sp_dist = np.full((n, K_NBR), np.inf, dtype=np.float32)
    section_sizes: List[int] = []
    section_ids = meta["Section_ID"]
    for _sec, group in meta.groupby("Section_ID", dropna=True):
        gi = meta.index.get_indexer(group.index)
        section_sizes.append(int(len(gi)))
        k = min(K_NBR + 2, len(gi))
        tree = cKDTree(coords[gi])
        d, j = tree.query(coords[gi], k=k)
        d, j = _as_knn_2d(d, j, n=len(gi), k=k)
        local_idx, local_dist = _compact_exclude_self(j, d, K_NBR)
        mapped = np.where(local_idx >= 0, gi[np.maximum(local_idx, 0)], -1)
        sp_idx[gi] = mapped
        sp_dist[gi] = local_dist
    del section_ids

    tree = cKDTree(pcs)
    d, j = tree.query(pcs, k=K_NBR + 2)
    d, j = _as_knn_2d(d, j, n=n, k=K_NBR + 2)
    ex_idx, ex_dist = _compact_exclude_self(j, d, K_NBR)

    sec_stats = meta.groupby("Section_ID", dropna=True)[["center_x", "center_y"]].agg(
        ["mean", "std"]
    )
    sx = meta["Section_ID"].map(sec_stats[("center_x", "mean")]).to_numpy(dtype=np.float64)
    sy = meta["Section_ID"].map(sec_stats[("center_y", "mean")]).to_numpy(dtype=np.float64)
    sxs = meta["Section_ID"].map(sec_stats[("center_x", "std")]).to_numpy(dtype=np.float64)
    sys_ = meta["Section_ID"].map(sec_stats[("center_y", "std")]).to_numpy(dtype=np.float64)
    rel_x = ((meta["center_x"].to_numpy(dtype=np.float64) - sx) / np.maximum(sxs, 1.0)).astype(
        np.float32
    )
    rel_y = ((meta["center_y"].to_numpy(dtype=np.float64) - sy) / np.maximum(sys_, 1.0)).astype(
        np.float32
    )

    nb_mean = np.zeros((n, PCA_DIM), dtype=np.float32)
    nb_cnt = np.zeros(n, dtype=np.float32)
    for kk in range(N_SPATIAL_MEAN):
        v = sp_idx[:, kk]
        ok = v >= 0
        nb_mean[ok] += pcs[v[ok]]
        nb_cnt[ok] += 1
    nb_mean /= np.maximum(nb_cnt, 1.0)[:, None]

    region = _codes(meta["Region"])
    segment = _codes(meta["Segment"])
    ap = _codes(meta["AP_position"])
    gender = (meta["Gender"] == "male").to_numpy(dtype=np.float32)
    mouse = _codes(meta["Mouse_ID"])
    dsets = _codes(meta["Datasets"])
    ei_obs = (
        meta["Excitatory_vs_Inhibitory"]
        .map({"excitatory": 1.0, "inhibitory": 0.0})
        .to_numpy(dtype=np.float32)
    )

    volume = meta["volume"].to_numpy(dtype=np.float64)
    cols_meta = np.column_stack(
        [
            np.log1p(tot),
            np.log1p(volume),
            tot / np.maximum(volume, 1.0),
            meta["center_x"].to_numpy(dtype=np.float64),
            meta["center_y"].to_numpy(dtype=np.float64),
            rel_x,
            rel_y,
            sp_dist[:, 0],
            np.nanmean(np.where(np.isfinite(sp_dist[:, :5]), sp_dist[:, :5], np.nan), axis=1),
            np.nanmean(np.where(np.isfinite(sp_dist[:, :15]), sp_dist[:, :15], np.nan), axis=1),
            ex_dist[:, 0],
            ex_dist[:, :10].mean(axis=1),
            region,
            ei_obs,
            segment,
            ap,
            gender,
            mouse,
            dsets,
        ]
    ).astype(np.float32)
    gene_names = ["g_{}".format(c) for c in counts.columns]
    meta_names = [
        "log_total",
        "log_volume",
        "density",
        "x",
        "y",
        "rel_x",
        "rel_y",
        "sp_d1",
        "sp_d5",
        "sp_d15",
        "ex_d1",
        "ex_d10",
        "Region",
        "EI",
        "Segment",
        "AP",
        "gender",
        "mouse",
        "dataset",
    ]
    names = (
        gene_names
        + ["pc{}".format(i) for i in range(PCA_DIM)]
        + ["nbm{}".format(i) for i in range(PCA_DIM)]
        + meta_names
    )
    x_static = np.hstack([lognorm, pcs, nb_mean, cols_meta]).astype(np.float32)
    if x_static.shape[1] != len(names):
        raise RuntimeError("static feature name mismatch")

    return SpatialUniverse(
        cell_ids=cell_ids,
        is_train=is_train,
        X_static=x_static,
        names=names,
        sp_idx=sp_idx,
        sp_dist=sp_dist,
        ex_idx=ex_idx,
        ex_dist=ex_dist,
        ei_known=ei_obs,
        pca_var_explained=pca_var,
        section_sizes=section_sizes,
        transductive_note=(
            "PCA50, within-section spatial kNN, and PCA50 expression kNN are fit on "
            "the concatenated 5000-train + 5000-test feature universe. No labels, "
            "including hidden test labels, are used."
        ),
    )
