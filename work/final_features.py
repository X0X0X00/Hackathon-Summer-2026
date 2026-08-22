"""Feature construction shared by train_final.py / predict_final.py.

Universe = reference cells (public deposit, Zenodo 18039571) + competition TRAIN cells
         + whatever cells are in data/meta_test.csv + counts_test.csv (ANY cells: if a test id
           exists in the deposit its deposit row is used but it is never labelled; otherwise the
           row is appended from the CSVs).
Reference (labelled) = deposit cells MINUS every competition id (train + current test)
         MINUS exact 200-gene fingerprint duplicates of competition cells MINUS `exclude_ids`.
Labels of competition TEST cells are never read.

`prep` (fit at train time, saved to artifacts) holds every data-dependent constant:
  scale, scaler mean/std, PCA components, categorical vocabularies, Region/AP/Segment maps.
"""
import json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from scipy.spatial import cKDTree
from common import label_hist, N_CLASSES

WORK = Path(__file__).resolve().parent
BASE = WORK.parent
ART = WORK / "final_artifacts"
DEPOSIT = WORK / "external/MERFISH_spinal_cord_resolved_0718.h5ad"
K_NBR = 50
META_NAMES = ["log_total", "log_volume", "density", "x", "y", "rel_x", "rel_y",
              "sp_d1", "sp_d5", "sp_d15", "ex_d1", "ex_d10",
              "Region", "EI", "Segment", "AP", "gender", "mouse", "dataset"]
META_LABEL_COLS = ["Region", "EI", "Segment"]          # the label-correlated metadata


def norm_label(s):
    return str(s).replace(" ", "_").replace("-", "_")


def row_hash(M):
    M = np.ascontiguousarray(M.astype(np.int64))
    return pd.util.hash_array(M.view(np.dtype((np.void, M.dtype.itemsize * M.shape[1]))).ravel())


def read_competition(test_meta=None, test_counts=None):
    mtr = pd.read_csv(BASE / "data/meta_train.csv", index_col=0)
    ctr = pd.read_csv(BASE / "data/counts_train.csv", index_col=0)
    mte = pd.read_csv(test_meta or BASE / "data/meta_test.csv", index_col=0)
    cte = pd.read_csv(test_counts or BASE / "data/counts_test.csv", index_col=0)
    assert (mtr.index == ctr.index).all() and (mte.index == cte.index).all()
    for df in (mtr, ctr, mte, cte):
        df.index = df.index.astype(str)
    return mtr, ctr, mte, cte


def read_deposit(genes):
    a = ad.read_h5ad(DEPOSIT)
    obs = a.obs.copy(); obs.index = obs.index.astype(str)
    X = a[:, genes].X
    X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    return obs, X


def build_universe(genes, test_meta=None, test_counts=None, exclude_ids=()):
    """Returns dict U with per-row arrays (deposit rows first, then appended test rows)."""
    mtr, ctr, mte, cte = read_competition(test_meta, test_counts)
    obs, Xd = read_deposit(genes)
    dep_ids = np.array(obs.index, dtype=str)
    dep_pos = pd.Series(np.arange(len(dep_ids)), index=dep_ids)
    train_ids = np.array(mtr.index, dtype=str); test_ids = np.array(mte.index, dtype=str)
    assert set(train_ids) <= set(dep_ids), "competition train cells must be deposit cells"
    in_dep = np.isin(test_ids, dep_ids)
    new_test = test_ids[~in_dep]
    n = len(dep_ids) + len(new_test)
    ids = np.concatenate([dep_ids, new_test])
    pos = pd.Series(np.arange(n), index=ids)

    # ---- expression (200 competition genes) ----
    X200 = np.zeros((n, len(genes)), np.float32)
    X200[:len(dep_ids)] = Xd
    X200[pos.loc[train_ids].values] = ctr[genes].values.astype(np.float32)   # identical to deposit (verified)
    X200[pos.loc[test_ids].values] = cte[genes].values.astype(np.float32)
    # ---- metadata frame (text form) ----
    meta = pd.DataFrame(index=ids)
    meta["Section_ID"] = np.concatenate([obs["Section ID"].astype(str).values, np.array([""] * len(new_test))])
    meta["center_x"] = np.r_[obs["center_x"].astype(float).values, np.full(len(new_test), np.nan)]
    meta["center_y"] = np.r_[obs["center_y"].astype(float).values, np.full(len(new_test), np.nan)]
    meta["volume"] = np.r_[obs["volume"].astype(float).values, np.full(len(new_test), np.nan)]
    meta["Datasets"] = np.concatenate([obs["Datasets"].astype(str).values, np.array([""] * len(new_test))])
    meta["Gender"] = np.concatenate([obs["Gender"].astype(str).values, np.array([""] * len(new_test))])
    meta["Mouse_ID"] = np.concatenate([obs["Mouse ID"].astype(str).values, np.array([""] * len(new_test))])
    meta["AP_txt"] = np.concatenate([obs["Axial level"].astype(str).values, np.array([""] * len(new_test))])
    meta["EI_txt"] = np.concatenate([obs["Excitatory_vs_Inhibitory"].astype(str).values, np.array(["nan"] * len(new_test))])
    meta["Region_txt"] = np.concatenate([obs["Region"].astype(str).values, np.array(["nan"] * len(new_test))])
    meta["Laminae_txt"] = np.concatenate([obs["Laminae"].astype(str).values, np.array(["nan"] * len(new_test))])
    # competition rows: official CSV values win (numeric Region/AP/Segment, text EI)
    comp = pd.concat([mtr, mte])
    cp = pos.loc[comp.index.astype(str)].values
    for col in ["Section_ID", "center_x", "center_y", "volume", "Datasets", "Gender", "Mouse_ID"]:
        meta.loc[comp.index.astype(str), col] = comp[col].astype(str if col in ("Section_ID", "Datasets", "Gender", "Mouse_ID") else float).values
    meta["Region_num"] = np.nan; meta["AP_num"] = np.nan; meta["Segment_num"] = np.nan; meta["EI_num"] = np.nan
    meta.loc[comp.index.astype(str), "Region_num"] = comp["Region"].values.astype(float)
    meta.loc[comp.index.astype(str), "AP_num"] = comp["AP_position"].values.astype(float)
    meta.loc[comp.index.astype(str), "Segment_num"] = comp["Segment"].values.astype(float)
    meta.loc[comp.index.astype(str), "EI_num"] = comp["Excitatory_vs_Inhibitory"].map({"excitatory": 1.0, "inhibitory": 0.0}).values
    # ---- labels & roles ----
    y = np.array([norm_label(l) for l in obs["MERFISH cell type annotation"].astype(str).values] + [""] * len(new_test))
    labels = sorted(mtr.MERFISH_cell_type_annotation.unique())
    lab2code = {l: i for i, l in enumerate(labels)}
    ycode = np.array([lab2code.get(l, -1) for l in y], np.int16)
    is_train = np.zeros(n, bool); is_train[pos.loc[train_ids].values] = True
    is_test = np.zeros(n, bool); is_test[pos.loc[test_ids].values] = True
    ycode[is_test] = -1                                           # never use deposit labels of test cells
    ycode[is_train] = mtr.MERFISH_cell_type_annotation.map(lab2code).values
    is_ref = (ycode >= 0) & ~is_train & ~is_test
    is_ref[pos.loc[[i for i in exclude_ids if i in pos.index]].values] = False
    # fingerprint duplicates of any competition cell -> not reference
    comp_h = set(row_hash(np.vstack([ctr[genes].values, cte[genes].values])))
    dup = np.isin(row_hash(X200), list(comp_h)) & ~is_train & ~is_test
    is_ref &= ~dup
    U = dict(ids=ids, X200=X200, meta=meta, y=ycode, labels=np.array(labels), is_train=is_train,
             is_test=is_test, is_ref=is_ref, n_dep=len(dep_ids), test_in_deposit=int(in_dep.sum()),
             test_new=int((~in_dep).sum()), mtr=mtr, mte=mte)
    return U


def fit_prep(U):
    """Fit every data-dependent constant on reference + train rows; return prep dict."""
    rows = U["is_ref"] | U["is_train"]
    X200 = U["X200"]; meta = U["meta"]
    tot = X200.sum(1).astype(np.float64)
    scale = float(np.median(tot[U["is_train"]]))
    lognorm = np.log1p(X200 / np.maximum(tot, 1)[:, None] * scale)
    mu = lognorm[rows].mean(0); sd = lognorm[rows].std(0) + 1e-6
    Z = (lognorm[rows] - mu) / sd
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=0).fit(Z)
    tr = U["is_train"]
    def learn_map(num, txt):
        df = pd.DataFrame({"num": num, "txt": txt}).dropna()
        df = df[df.txt != "nan"].drop_duplicates()
        assert df.groupby("txt").num.nunique().max() == 1, df
        return {k: float(v) for k, v in zip(df.txt, df.num)}
    prep = dict(
        scale=scale, mu=mu.tolist(), sd=sd.tolist(), pca_components=pca.components_.tolist(),
        pca_mean=pca.mean_.tolist(),
        region_map=learn_map(meta.Region_num[tr].values, meta.Region_txt[tr].values),
        ap_map=learn_map(meta.AP_num[tr].values, meta.AP_txt[tr].values),
        segment_map=learn_map(meta.Segment_num[tr].values, meta.Laminae_txt[tr].values),
        mouse_vocab=sorted(set(meta.Mouse_ID[rows])), dataset_vocab=sorted(set(meta.Datasets[rows])),
        labels=list(U["labels"]),
    )
    # label -> E/I (from competition train) and Segment -> allowed labels (from reference only)
    lab_ei = U["mtr"].groupby("MERFISH_cell_type_annotation").Excitatory_vs_Inhibitory.first()
    prep["ei_of_label"] = [{"excitatory": 1, "inhibitory": 0}.get(lab_ei.get(l), -1) for l in U["labels"]]
    seg_ref = meta.Laminae_txt[U["is_ref"]].map(prep["segment_map"]).values
    yref = U["y"][U["is_ref"]]
    tab = {}
    for s in np.unique(seg_ref[~np.isnan(seg_ref)]):
        tab[str(int(s))] = sorted(set(yref[seg_ref == s].tolist()))
    prep["segment_table"] = tab
    return prep


def featurize(U, prep):
    """Static features (n, 319) in the exact prep.py layout + kNN caches."""
    X200 = U["X200"]; meta = U["meta"]; n = len(U["ids"])
    tot = X200.sum(1).astype(np.float64)
    lognorm = np.log1p(X200 / np.maximum(tot, 1)[:, None] * prep["scale"]).astype(np.float32)
    Z = (lognorm - np.array(prep["mu"], np.float32)) / np.array(prep["sd"], np.float32)
    P = ((Z - np.array(prep["pca_mean"], np.float32)) @ np.array(prep["pca_components"], np.float32).T).astype(np.float32)
    # metadata codes
    region = meta.Region_txt.map(prep["region_map"]).astype(float).values
    ap = meta.AP_txt.map(prep["ap_map"]).astype(float).values
    segment = meta.Laminae_txt.map(prep["segment_map"]).astype(float).values
    ei = meta.EI_txt.map({"excitatory": 1.0, "inhibitory": 0.0}).astype(float).values
    comp = U["is_train"] | U["is_test"]
    region[comp] = meta.Region_num.values[comp]; ap[comp] = meta.AP_num.values[comp]
    segment[comp] = meta.Segment_num.values[comp]; ei[comp] = meta.EI_num.values[comp]
    ap = ap - 1
    gender = (meta.Gender.values == "male").astype(np.float32)
    mouse = pd.Categorical(meta.Mouse_ID, categories=prep["mouse_vocab"]).codes.astype(np.float32); mouse[mouse < 0] = np.nan
    dsets = pd.Categorical(meta.Datasets, categories=prep["dataset_vocab"]).codes.astype(np.float32); dsets[dsets < 0] = np.nan
    # spatial kNN within section (all universe cells)
    coords = meta[["center_x", "center_y"]].values.astype(float)
    sp_idx = np.full((n, K_NBR), -1, np.int32); sp_dist = np.full((n, K_NBR), np.inf, np.float32)
    for sec, gi in meta.groupby("Section_ID").indices.items():
        gi = np.asarray(gi)
        if len(gi) < 2: continue
        k = min(K_NBR + 1, len(gi))
        d, j = cKDTree(coords[gi]).query(coords[gi], k=k)
        sp_idx[gi, :k - 1] = gi[j[:, 1:]]; sp_dist[gi, :k - 1] = d[:, 1:]
    d, j = cKDTree(P).query(P, k=K_NBR + 1, workers=4)
    ex_idx = j[:, 1:].astype(np.int32); ex_dist = d[:, 1:].astype(np.float32)
    g = meta.groupby("Section_ID")
    sx = g.center_x.transform("mean").values; sy = g.center_y.transform("mean").values
    sxs = g.center_x.transform("std").fillna(1).values; sys_ = g.center_y.transform("std").fillna(1).values
    rel_x = ((meta.center_x.values - sx) / np.maximum(sxs, 1)).astype(np.float32)
    rel_y = ((meta.center_y.values - sy) / np.maximum(sys_, 1)).astype(np.float32)
    nb_mean = np.zeros((n, 50), np.float32); nb_cnt = np.zeros(n, np.float32)
    for kk in range(10):
        v = sp_idx[:, kk]; ok = v >= 0
        nb_mean[ok] += P[v[ok]]; nb_cnt[ok] += 1
    nb_mean /= np.maximum(nb_cnt, 1)[:, None]
    def nanmean_cols(Dm, k):
        return np.nanmean(np.where(np.isfinite(Dm[:, :k]), Dm[:, :k], np.nan), 1)
    with np.errstate(all="ignore"):
        cols_meta = np.column_stack([
            np.log1p(tot), np.log1p(meta.volume.values.astype(float)), tot / np.maximum(meta.volume.values.astype(float), 1),
            meta.center_x.values, meta.center_y.values, rel_x, rel_y,
            sp_dist[:, 0], nanmean_cols(sp_dist, 5), nanmean_cols(sp_dist, 15),
            ex_dist[:, 0], ex_dist[:, :10].mean(1),
            region, ei, segment, ap, gender, mouse, dsets,
        ]).astype(np.float32)
    cols_meta[~np.isfinite(cols_meta)] = np.nan
    X_static = np.hstack([lognorm, P, nb_mean, cols_meta]).astype(np.float32)
    names = [f"g_{c}" for c in U["genes"]] + [f"pc{i}" for i in range(50)] + [f"nbm{i}" for i in range(50)] + META_NAMES
    return dict(X=X_static, names=names, sp_idx=sp_idx, sp_dist=sp_dist, ex_idx=ex_idx, ex_dist=ex_dist,
                ei_known=np.where(np.isnan(ei), -1, ei).astype(np.int8), segment=segment)


def make_X(F, known, sp_k=15, ex_k=25, drop_sp=False, drop_meta=False):
    parts, names = [F["X"]], list(F["names"])
    if not drop_sp:
        h, nm = label_hist(known, F["sp_idx"], F["sp_dist"], sp_k, "sp"); parts.append(h); names += nm
    h, nm = label_hist(known, F["ex_idx"], F["ex_dist"], ex_k, "ex"); parts.append(h); names += nm
    X = np.hstack(parts).astype(np.float32)
    if drop_meta:
        for c in META_LABEL_COLS:
            X[:, names.index(c)] = np.nan
    return X, [n.replace(" ", "_") for n in names]


def sp_labeled_count(F, known, k=15):
    h, _ = label_hist(known, F["sp_idx"], F["sp_dist"], k, "sp")
    return h[:, N_CLASSES]


def postprocess(probs, ei_known, segment, prep):
    """E/I constraint (where observed) + Segment->allowed-labels mask (where observed)."""
    from common import apply_ei
    p = apply_ei(probs, ei_known, np.array(prep["ei_of_label"], np.int8))
    tab = prep["segment_table"]
    M = np.ones_like(p, bool)
    for i, s in enumerate(segment):
        if np.isnan(s) or str(int(s)) not in tab: continue
        M[i] = False; M[i, tab[str(int(s))]] = True
    p2 = np.where(M, p, 0.0); bad = p2.sum(1) <= 0; p2[bad] = p[bad]
    return p2 / p2.sum(1, keepdims=True)
