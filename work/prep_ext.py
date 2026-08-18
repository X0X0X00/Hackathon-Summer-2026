"""Build the EXTENDED universe cache: competition 10k cells + Zenodo reference cells.

Reference = Zenodo 18039571 MERFISH_spinal_cord_resolved_0718.h5ad (146,621 cells, same study),
MINUS all 10,000 competition Cell_IDs MINUS 47 expression-fingerprint duplicates
(external/reference_ids.npy, built and verified on train cells only).

Universe rows = ALL 146,621 cells (needed for kNN structure). Labels:
  - reference cells: their external annotation (normalised to competition names)
  - competition cells: -1, then competition TRAIN labels filled from meta_train.csv
  - competition TEST labels are never read into any array.

Outputs work/cache_ext/static.npz + nbrs.npz with the same keys/feature layout as
work/cache/ so common.py logic can be reused (see common_ext.py).
"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WORK = Path(__file__).resolve().parent
BASE = WORK.parent
CACHE = WORK / "cache_ext"
CACHE.mkdir(exist_ok=True)
K_NBR = 50

def norm_label(s):
    return str(s).replace(" ", "_").replace("-", "_")

# ---------------- competition data ----------------
mtr = pd.read_csv(BASE / "data/meta_train.csv", index_col=0)
mte = pd.read_csv(BASE / "data/meta_test.csv", index_col=0)
ctr = pd.read_csv(BASE / "data/counts_train.csv", index_col=0)
cte = pd.read_csv(BASE / "data/counts_test.csv", index_col=0)
genes = ctr.columns.tolist()
comp_meta = pd.concat([mtr, mte]); comp_meta.index = comp_meta.index.astype(str)
comp_counts = pd.concat([ctr, cte]); comp_counts.index = comp_counts.index.astype(str)
train_ids = mtr.index.astype(str); test_ids = mte.index.astype(str)
labels = sorted(mtr.MERFISH_cell_type_annotation.unique())
lab2code = {l: i for i, l in enumerate(labels)}

# ---------------- external ----------------
a = ad.read_h5ad(WORK / "external/MERFISH_spinal_cord_resolved_0718.h5ad")
ref_ids = set(np.load(WORK / "external/reference_ids.npy", allow_pickle=True).astype(str))
obs = a.obs.copy(); obs.index = obs.index.astype(str)
n = a.n_obs
ids = np.array(obs.index, dtype=str)
pos = pd.Series(np.arange(n), index=ids)
assert set(train_ids) <= set(ids) and set(test_ids) <= set(ids)

# labels: reference only; competition rows -1; then fill train labels
ext_lab = obs["MERFISH cell type annotation"].astype(str).map(norm_label)
y = np.array([lab2code.get(l, -1) for l in ext_lab.values], np.int16)
is_ref = np.array([i in ref_ids for i in ids])
comp_pos = pos.loc[list(train_ids) + list(test_ids)].values
y[comp_pos] = -1                                   # <- competition rows carry no external label
y[pos.loc[train_ids].values] = mtr.MERFISH_cell_type_annotation.map(lab2code).values
y[~is_ref & ~np.isin(np.arange(n), pos.loc[train_ids].values)] = -1   # dupes etc.
del ext_lab
is_train = np.zeros(n, bool); is_train[pos.loc[train_ids].values] = True
is_test = np.zeros(n, bool); is_test[pos.loc[test_ids].values] = True
is_comp = is_train | is_test
print(f"universe={n}  reference(labeled)={(is_ref & (y>=0)).sum()}  train={is_train.sum()}  test={is_test.sum()}  unlabeled_other={((~is_ref)&(~is_comp)).sum()}")
assert (y[is_test] == -1).all()

# ---------------- expression (200 competition genes) ----------------
X200 = a[:, genes].X
X200 = X200.toarray() if sp.issparse(X200) else np.asarray(X200)
X200 = X200.astype(np.float32)
# sanity: competition rows identical to competition csv counts
chk = comp_counts.loc[ids[comp_pos], genes].values.astype(np.float32)
assert np.array_equal(X200[comp_pos], chk), "count mismatch vs competition csv"
tot = X200.sum(1).astype(np.float64)
scale = np.median(tot[is_comp])   # same scale convention as prep.py (competition median)
lognorm = np.log1p(X200 / np.maximum(tot, 1)[:, None] * scale).astype(np.float32)
Z = StandardScaler().fit_transform(lognorm)
pca = PCA(n_components=50, random_state=0)
P = pca.fit_transform(Z).astype(np.float32)
print(f"PCA50 var explained (all cells): {pca.explained_variance_ratio_.sum():.3f}")
del Z

# ---------------- metadata harmonisation ----------------
meta = pd.DataFrame(index=ids)
meta["Section_ID"] = obs["Section ID"].astype(str).values
meta["center_x"] = obs["center_x"].astype(float).values
meta["center_y"] = obs["center_y"].astype(float).values
meta["volume"] = obs["volume"].astype(float).values
meta["Datasets"] = obs["Datasets"].astype(str).values
meta["Gender"] = obs["Gender"].astype(str).values
meta["Mouse_ID"] = obs["Mouse ID"].astype(str).values
meta["Axial"] = obs["Axial level"].astype(str).values
meta["EI_txt"] = obs["Excitatory_vs_Inhibitory"].astype(str).replace("nan", np.nan).values
meta["Region_txt"] = obs["Region"].astype(str).replace("nan", np.nan).values
# competition rows: overwrite with competition meta (identical for train; verified) so that
# Region/EI/Segment/AP encodings for competition cells come from the competition files
cm = comp_meta.loc[ids[comp_pos]]
assert np.allclose(meta.loc[ids[comp_pos], ["center_x", "center_y"]].values, cm[["center_x", "center_y"]].values, atol=1e-3)
assert (meta.loc[ids[comp_pos], "Section_ID"].values == cm.Section_ID.values).all()
# Region: competition numeric 1-5  <->  external text; learn mapping on competition rows
reg_map = (pd.DataFrame({"num": cm.Region.values, "txt": meta.loc[ids[comp_pos], "Region_txt"].values})
           .dropna().drop_duplicates())
assert reg_map.groupby("txt").num.nunique().max() == 1 and reg_map.groupby("num").txt.nunique().max() == 1, reg_map
txt2num = dict(zip(reg_map.txt, reg_map.num)); print("Region map:", txt2num)
region = meta.Region_txt.map(txt2num).astype(np.float32).values
region[comp_pos] = cm.Region.values.astype(np.float32)
# AP: competition 1-4 <-> Axial text
ap_map = (pd.DataFrame({"num": cm.AP_position.values, "txt": meta.loc[ids[comp_pos], "Axial"].values}).dropna().drop_duplicates())
assert ap_map.groupby("txt").num.nunique().max() == 1
ap_txt2num = dict(zip(ap_map.txt, ap_map.num)); print("AP map:", ap_txt2num)
ap = meta.Axial.map(ap_txt2num).astype(np.float32).values
ap[comp_pos] = cm.AP_position.values.astype(np.float32)
ap = ap - 1  # codes 0..3 like prep.py's Categorical codes
# EI
ei = meta.EI_txt.map({"excitatory": 1.0, "inhibitory": 0.0}).astype(np.float32).values
ei[comp_pos] = cm.Excitatory_vs_Inhibitory.map({"excitatory": 1.0, "inhibitory": 0.0}).values.astype(np.float32)
# Segment: competition 'Segment' == external 'Laminae' recoded 1..22 (verified 1:1 on comp rows,
# identical missingness). Learn the text->code map on competition rows, apply to reference rows.
lam = obs["Laminae"].astype(str).replace("nan", np.nan)
seg_map = (pd.DataFrame({"num": cm.Segment.values, "txt": lam.loc[ids[comp_pos]].values}).dropna().drop_duplicates())
assert seg_map.groupby("txt").num.nunique().max() == 1 and seg_map.groupby("num").txt.nunique().max() == 1, seg_map
lam2seg = dict(zip(seg_map.txt, seg_map.num)); print("Segment<-Laminae map:", len(lam2seg), "codes")
segment = lam.map(lam2seg).astype(np.float32).values          # reference rows via Laminae
segment[comp_pos] = cm.Segment.values.astype(np.float32)      # competition rows: official values
assert np.array_equal(np.isnan(segment[comp_pos]), cm.Segment.isna().values)
gender = (meta.Gender == "male").astype(np.float32).values
mouse = pd.Categorical(meta.Mouse_ID).codes.astype(np.float32)
dsets = pd.Categorical(meta.Datasets).codes.astype(np.float32)
# label -> EI mapping from competition train (as prep.py)
lab_ei = mtr.groupby("MERFISH_cell_type_annotation").Excitatory_vs_Inhibitory.first()
ei_of_label = np.array([{"excitatory": 1, "inhibitory": 0}.get(lab_ei.get(l), -1) for l in labels], np.int8)

# ---------------- spatial neighbours (within section, ALL cells) ----------------
coords = meta[["center_x", "center_y"]].values
sp_idx = np.full((n, K_NBR), -1, np.int32); sp_dist = np.full((n, K_NBR), np.inf, np.float32)
sizes = []
for sec, gi in meta.groupby("Section_ID").indices.items():
    gi = np.asarray(gi); sizes.append(len(gi))
    k = min(K_NBR + 1, len(gi))
    d, j = cKDTree(coords[gi]).query(coords[gi], k=k)
    sp_idx[gi, :k - 1] = gi[j[:, 1:]]; sp_dist[gi, :k - 1] = d[:, 1:]
print(f"sections={len(sizes)} size min/med/max={min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")

# ---------------- expression neighbours (PCA50, all cells) ----------------
d, j = cKDTree(P).query(P, k=K_NBR + 1, workers=8)
ex_idx = j[:, 1:].astype(np.int32); ex_dist = d[:, 1:].astype(np.float32)

# ---------------- rel coords, neighbour-mean expression ----------------
g = meta.groupby("Section_ID")
sx = g.center_x.transform("mean").values; sy = g.center_y.transform("mean").values
sxs = g.center_x.transform("std").values; sys_ = g.center_y.transform("std").values
rel_x = ((meta.center_x.values - sx) / np.maximum(sxs, 1)).astype(np.float32)
rel_y = ((meta.center_y.values - sy) / np.maximum(sys_, 1)).astype(np.float32)
nb_mean = np.zeros((n, 50), np.float32); nb_cnt = np.zeros(n, np.float32)
for kk in range(10):
    v = sp_idx[:, kk]; ok = v >= 0
    nb_mean[ok] += P[v[ok]]; nb_cnt[ok] += 1
nb_mean /= np.maximum(nb_cnt, 1)[:, None]

def nanmean_cols(D, k):
    return np.nanmean(np.where(np.isfinite(D[:, :k]), D[:, :k], np.nan), 1)
cols_meta = np.column_stack([
    np.log1p(tot), np.log1p(meta.volume.values), tot / np.maximum(meta.volume.values, 1),
    meta.center_x.values, meta.center_y.values, rel_x, rel_y,
    sp_dist[:, 0], nanmean_cols(sp_dist, 5), nanmean_cols(sp_dist, 15),
    ex_dist[:, 0], ex_dist[:, :10].mean(1),
    region, ei, segment, ap, gender, mouse, dsets,
]).astype(np.float32)
meta_names = ["log_total", "log_volume", "density", "x", "y", "rel_x", "rel_y",
              "sp_d1", "sp_d5", "sp_d15", "ex_d1", "ex_d10",
              "Region", "EI", "Segment", "AP", "gender", "mouse", "dataset"]
X_static = np.hstack([lognorm, P, nb_mean, cols_meta]).astype(np.float32)
names = [f"g_{c}" for c in genes] + [f"pc{i}" for i in range(50)] + [f"nbm{i}" for i in range(50)] + meta_names
assert X_static.shape[1] == len(names)

# ---------------- folds: reuse the ORIGINAL competition folds ----------------
orig = np.load(WORK / "cache/static.npz", allow_pickle=True)
orig_ids = orig["ids"].astype(str); orig_folds = orig["folds"]
folds = np.full(n, -1, np.int8)
folds[pos.loc[orig_ids].values] = orig_folds
assert (folds[is_train] >= 0).all() and (folds[~is_train] == -1).all()

np.savez_compressed(CACHE / "static.npz",
    X=X_static, names=np.array(names), ids=ids, is_train=is_train, is_test=is_test, is_ref=is_ref & (y >= 0),
    y=y, labels=np.array(labels), ei_of_label=ei_of_label, folds=folds, ei_known=ei, region_known=region)
np.savez_compressed(CACHE / "nbrs.npz", sp_idx=sp_idx, sp_dist=sp_dist, ex_idx=ex_idx, ex_dist=ex_dist)
print("saved:", X_static.shape, "-> work/cache_ext/")
