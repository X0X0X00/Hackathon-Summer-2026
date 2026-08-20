"""Extended competition + Zenodo reference universe (same-team prep_ext layout).

PCA50, within-section spatial kNN, and expression kNN are fit on all 146,621
deposit cells. No hidden test labels are used. Caches are keyed by source MD5
and preprocessing config under work/external/cache/ (gitignored).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from merfish60.io import repo_root
from merfish60.reference import EXPECTED_MD5, EXPECTED_N_OBS
from merfish60.spatial_features import K_NBR, N_SPATIAL_MEAN, PCA_DIM, PCA_RANDOM_STATE, _as_knn_2d


CONFIG_ID = "v2b_ext_static_v1"
CACHE_DIR_REL = "work/external/cache"


@dataclass
class ExtUniverse:
    cell_ids: np.ndarray
    is_train: np.ndarray
    is_test: np.ndarray
    is_ref: np.ndarray
    X_static: np.ndarray
    names: List[str]
    sp_idx: np.ndarray
    sp_dist: np.ndarray
    ex_idx: np.ndarray
    ex_dist: np.ndarray
    ei_known: np.ndarray
    segment: np.ndarray
    y_codes: np.ndarray
    pca_var_explained: float
    section_sizes: List[int]
    source_md5: str
    config_id: str
    transductive_note: str


def cache_dir(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / CACHE_DIR_REL


def _cache_stem(source_md5: str, gene_sha: str) -> str:
    return "v2b_{}_{}_{}".format(source_md5[:12], gene_sha[:8], CONFIG_ID)


def cache_paths(source_md5: str, gene_sha: str, root: Optional[Path] = None):
    stem = _cache_stem(source_md5, gene_sha)
    directory = cache_dir(root)
    return (
        directory / (stem + "_static.npz"),
        directory / (stem + "_nbrs.npz"),
        directory / (stem + "_meta.json"),
    )


def _overwrite_competition(values: np.ndarray, idx: np.ndarray, official) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    official = pd.to_numeric(pd.Series(official), errors="coerce").to_numpy(dtype=np.float32)
    out[idx] = official
    return out


def build_ext_universe(audit: dict, data, class_names: Sequence[str], root: Optional[Path] = None) -> ExtUniverse:
    """Build static features and kNN graphs on the full 146,621-cell deposit."""
    del class_names
    root = root or repo_root()
    ids = np.asarray(audit["_ext_ids"], dtype=object)
    n = len(ids)
    if n != EXPECTED_N_OBS:
        raise RuntimeError("unexpected universe size {}".format(n))
    is_train = np.asarray(audit["_is_train"])
    is_test = np.asarray(audit["_is_test"])
    is_ref = np.asarray(audit["_is_ref"])
    y_codes = np.asarray(audit["_y_codes"], dtype=np.int64)
    x200 = np.asarray(audit["_x200"], dtype=np.float32)
    obs = audit["_adata"].obs
    tr_idx = np.asarray(audit["_tr_idx"], dtype=np.int64)
    te_idx = np.asarray(audit["_te_idx"], dtype=np.int64)
    comp_idx = np.concatenate([tr_idx, te_idx])

    tot = x200.sum(axis=1).astype(np.float64)
    is_comp = is_train | is_test
    scale = float(np.median(tot[is_comp]))
    lognorm = np.log1p(x200 / np.maximum(tot, 1.0)[:, None] * scale).astype(np.float32)
    z = StandardScaler().fit_transform(lognorm)
    pca = PCA(n_components=PCA_DIM, random_state=PCA_RANDOM_STATE)
    pcs = pca.fit_transform(z).astype(np.float32)
    pca_var = float(pca.explained_variance_ratio_.sum())
    del z

    section_id = obs["Section ID"].astype(str).to_numpy()
    center_x = obs["center_x"].astype(float).to_numpy()
    center_y = obs["center_y"].astype(float).to_numpy()
    volume = obs["volume"].astype(float).to_numpy()
    datasets = obs["Datasets"].astype(str).to_numpy()
    gender_txt = obs["Gender"].astype(str).to_numpy()
    mouse_id = obs["Mouse ID"].astype(str).to_numpy()
    axial = obs["Axial level"].astype(str).replace("nan", np.nan).to_numpy()
    ei_txt = obs["Excitatory_vs_Inhibitory"].astype(str).replace("nan", np.nan).to_numpy()
    region_txt = obs["Region"].astype(str).replace("nan", np.nan).to_numpy()
    laminae = obs["Laminae"].astype(str).replace("nan", np.nan).to_numpy()

    region = pd.Series(region_txt).map(audit["_region_map"]).to_numpy(dtype=np.float32)
    region = _overwrite_competition(
        region,
        tr_idx,
        data.meta_train.loc[data.counts_train.index, "Region"],
    )
    region = _overwrite_competition(
        region,
        te_idx,
        data.meta_test.loc[data.counts_test.index, "Region"],
    )
    ap = pd.Series(axial).map(audit["_ap_map"]).to_numpy(dtype=np.float32)
    ap = _overwrite_competition(ap, tr_idx, data.meta_train.loc[data.counts_train.index, "AP_position"])
    ap = _overwrite_competition(ap, te_idx, data.meta_test.loc[data.counts_test.index, "AP_position"])
    ap = ap - 1.0
    ei = (
        pd.Series(ei_txt)
        .map({"excitatory": 1.0, "inhibitory": 0.0})
        .to_numpy(dtype=np.float32)
    )
    ei = _overwrite_competition(
        ei,
        tr_idx,
        data.meta_train.loc[data.counts_train.index, "Excitatory_vs_Inhibitory"].map(
            {"excitatory": 1.0, "inhibitory": 0.0}
        ),
    )
    ei = _overwrite_competition(
        ei,
        te_idx,
        data.meta_test.loc[data.counts_test.index, "Excitatory_vs_Inhibitory"].map(
            {"excitatory": 1.0, "inhibitory": 0.0}
        ),
    )
    if audit["segment_map_valid"]:
        segment = pd.Series(laminae).map(audit["_lam2seg"]).to_numpy(dtype=np.float32)
    else:
        segment = np.full(n, np.nan, dtype=np.float32)
    segment = _overwrite_competition(
        segment, tr_idx, data.meta_train.loc[data.counts_train.index, "Segment"]
    )
    segment = _overwrite_competition(
        segment, te_idx, data.meta_test.loc[data.counts_test.index, "Segment"]
    )
    gender = (np.asarray(gender_txt) == "male").astype(np.float32)
    mouse = pd.Categorical(mouse_id).codes.astype(np.float32)
    dsets = pd.Categorical(datasets).codes.astype(np.float32)

    coords = np.column_stack([center_x, center_y])
    sp_idx = np.full((n, K_NBR), -1, dtype=np.int32)
    sp_dist = np.full((n, K_NBR), np.inf, dtype=np.float32)
    section_sizes: List[int] = []
    meta_tmp = pd.DataFrame({"Section_ID": section_id, "x": center_x, "y": center_y})
    for _sec, gi in meta_tmp.groupby("Section_ID").indices.items():
        gi = np.asarray(gi)
        section_sizes.append(int(len(gi)))
        k = min(K_NBR + 1, len(gi))
        tree = cKDTree(coords[gi])
        queried = tree.query(coords[gi], k=k, workers=8)
        d, j = _as_knn_2d(queried[0], queried[1], n=len(gi), k=k)
        if k <= 1:
            continue
        sp_idx[gi, : k - 1] = gi[j[:, 1:]]
        sp_dist[gi, : k - 1] = d[:, 1:]

    tree = cKDTree(pcs)
    d, j = tree.query(pcs, k=K_NBR + 1, workers=8)
    d, j = _as_knn_2d(d, j, n=n, k=K_NBR + 1)
    ex_idx = j[:, 1:].astype(np.int32)
    ex_dist = d[:, 1:].astype(np.float32)

    g = meta_tmp.groupby("Section_ID")
    sx = g["x"].transform("mean").to_numpy()
    sy = g["y"].transform("mean").to_numpy()
    sxs = g["x"].transform("std").to_numpy()
    sys_ = g["y"].transform("std").to_numpy()
    rel_x = ((center_x - sx) / np.maximum(sxs, 1.0)).astype(np.float32)
    rel_y = ((center_y - sy) / np.maximum(sys_, 1.0)).astype(np.float32)
    nb_mean = np.zeros((n, PCA_DIM), dtype=np.float32)
    nb_cnt = np.zeros(n, dtype=np.float32)
    for kk in range(N_SPATIAL_MEAN):
        v = sp_idx[:, kk]
        ok = v >= 0
        nb_mean[ok] += pcs[v[ok]]
        nb_cnt[ok] += 1
    nb_mean /= np.maximum(nb_cnt, 1.0)[:, None]

    def nanmean_cols(matrix, k):
        return np.nanmean(np.where(np.isfinite(matrix[:, :k]), matrix[:, :k], np.nan), axis=1)

    cols_meta = np.column_stack(
        [
            np.log1p(tot),
            np.log1p(volume),
            tot / np.maximum(volume, 1.0),
            center_x,
            center_y,
            rel_x,
            rel_y,
            sp_dist[:, 0],
            nanmean_cols(sp_dist, 5),
            nanmean_cols(sp_dist, 15),
            ex_dist[:, 0],
            ex_dist[:, :10].mean(axis=1),
            region,
            ei,
            segment,
            ap,
            gender,
            mouse,
            dsets,
        ]
    ).astype(np.float32)
    genes = list(audit["_genes"])
    names = (
        ["g_{}".format(g) for g in genes]
        + ["pc{}".format(i) for i in range(PCA_DIM)]
        + ["nbm{}".format(i) for i in range(PCA_DIM)]
        + [
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
    )
    x_static = np.hstack([lognorm, pcs, nb_mean, cols_meta]).astype(np.float32)
    if x_static.shape[1] != len(names):
        raise RuntimeError("static feature name mismatch")
    del comp_idx

    return ExtUniverse(
        cell_ids=ids,
        is_train=is_train,
        is_test=is_test,
        is_ref=is_ref,
        X_static=x_static,
        names=names,
        sp_idx=sp_idx,
        sp_dist=sp_dist,
        ex_idx=ex_idx,
        ex_dist=ex_dist,
        ei_known=ei,
        segment=segment,
        y_codes=y_codes,
        pca_var_explained=pca_var,
        section_sizes=section_sizes,
        source_md5=str(audit["md5"]),
        config_id=CONFIG_ID,
        transductive_note=(
            "PCA50, within-section spatial kNN, and PCA50 expression kNN are fit on "
            "all {} deposit cells. Scale = median total of the 10,000 competition "
            "cells. No hidden test labels are used.".format(EXPECTED_N_OBS)
        ),
    )


def save_ext_universe(universe: ExtUniverse, gene_sha: str, root: Optional[Path] = None) -> dict:
    static_path, nbrs_path, meta_path = cache_paths(universe.source_md5, gene_sha, root)
    static_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        static_path,
        X=universe.X_static,
        names=np.array(universe.names),
        ids=universe.cell_ids.astype(str),
        is_train=universe.is_train,
        is_test=universe.is_test,
        is_ref=universe.is_ref,
        y=universe.y_codes,
        ei_known=universe.ei_known,
        segment=universe.segment,
        pca_var=np.array([universe.pca_var_explained]),
        section_sizes=np.array(universe.section_sizes, dtype=np.int32),
    )
    np.savez_compressed(
        nbrs_path,
        sp_idx=universe.sp_idx,
        sp_dist=universe.sp_dist,
        ex_idx=universe.ex_idx,
        ex_dist=universe.ex_dist,
    )
    meta = {
        "config_id": CONFIG_ID,
        "source_md5": universe.source_md5,
        "expected_md5": EXPECTED_MD5,
        "gene_order_sha256": gene_sha,
        "n_obs": int(len(universe.cell_ids)),
        "n_static": int(universe.X_static.shape[1]),
        "pca_dim": PCA_DIM,
        "k_nbr": K_NBR,
        "n_spatial_mean": N_SPATIAL_MEAN,
        "pca_random_state": PCA_RANDOM_STATE,
        "pca_var_explained": universe.pca_var_explained,
        "n_ref": int(universe.is_ref.sum()),
        "n_train": int(universe.is_train.sum()),
        "n_test": int(universe.is_test.sum()),
        "static_path": str(static_path),
        "nbrs_path": str(nbrs_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def load_ext_universe(source_md5: str, gene_sha: str, root: Optional[Path] = None) -> Optional[ExtUniverse]:
    static_path, nbrs_path, meta_path = cache_paths(source_md5, gene_sha, root)
    if not (static_path.is_file() and nbrs_path.is_file() and meta_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("config_id") != CONFIG_ID:
        return None
    if meta.get("source_md5") != source_md5:
        return None
    if meta.get("gene_order_sha256") != gene_sha:
        return None
    if int(meta.get("n_obs", -1)) != EXPECTED_N_OBS:
        return None
    static = np.load(static_path, allow_pickle=True)
    nbrs = np.load(nbrs_path)
    return ExtUniverse(
        cell_ids=np.asarray(static["ids"], dtype=object),
        is_train=np.asarray(static["is_train"]),
        is_test=np.asarray(static["is_test"]),
        is_ref=np.asarray(static["is_ref"]),
        X_static=np.asarray(static["X"], dtype=np.float32),
        names=[str(v) for v in static["names"].tolist()],
        sp_idx=np.asarray(nbrs["sp_idx"], dtype=np.int32),
        sp_dist=np.asarray(nbrs["sp_dist"], dtype=np.float32),
        ex_idx=np.asarray(nbrs["ex_idx"], dtype=np.int32),
        ex_dist=np.asarray(nbrs["ex_dist"], dtype=np.float32),
        ei_known=np.asarray(static["ei_known"], dtype=np.float32),
        segment=np.asarray(static["segment"], dtype=np.float32),
        y_codes=np.asarray(static["y"], dtype=np.int64),
        pca_var_explained=float(np.asarray(static["pca_var"]).reshape(-1)[0]),
        section_sizes=np.asarray(static["section_sizes"]).astype(int).tolist(),
        source_md5=source_md5,
        config_id=CONFIG_ID,
        transductive_note=(
            "loaded cache {} keyed by MD5 {} and gene-order {}".format(
                CONFIG_ID, source_md5[:12], gene_sha[:8]
            )
        ),
    )


def build_or_load_ext_universe(
    audit: dict, data, class_names: Sequence[str], root: Optional[Path] = None
) -> ExtUniverse:
    gene_sha = str(audit["gene_order_sha256"])
    source_md5 = str(audit["md5"])
    cached = load_ext_universe(source_md5, gene_sha, root)
    if cached is not None:
        cached.y_codes = np.asarray(audit["_y_codes"], dtype=np.int64)
        cached.is_ref = np.asarray(audit["_is_ref"])
        cached.is_train = np.asarray(audit["_is_train"])
        cached.is_test = np.asarray(audit["_is_test"])
        return cached
    universe = build_ext_universe(audit, data, class_names, root=root)
    save_ext_universe(universe, gene_sha, root=root)
    return universe
