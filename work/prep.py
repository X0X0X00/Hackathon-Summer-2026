"""Build cached features + fixed CV folds for the MERFISH cell-type challenge.

Outputs in work/cache/:
  static.npz   - X_static (float32), static feature names, ids, is_train, y codes,
                 label list, EI codes per cell (observed / masked variants), folds
  nbrs.npz     - spatial + expression kNN neighbor indices/distances (train+test pooled)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
RNG = np.random.default_rng(7)
K_NBR = 50  # neighbors cached per cell (excl. self)

mtr = pd.read_csv(BASE / "data/meta_train.csv", index_col=0)
mte = pd.read_csv(BASE / "data/meta_test.csv", index_col=0)
ctr = pd.read_csv(BASE / "data/counts_train.csv", index_col=0)
cte = pd.read_csv(BASE / "data/counts_test.csv", index_col=0)
assert (mtr.index == ctr.index).all() and (mte.index == cte.index).all()

meta = pd.concat([mtr, mte])
counts = pd.concat([ctr, cte])
is_train = np.r_[np.ones(len(mtr), bool), np.zeros(len(mte), bool)]
n = len(meta)
print(f"cells={n} genes={counts.shape[1]}")
print("train meta missing:\n", mtr.isna().sum()[lambda s: s > 0].to_string() or " none")

# ---- labels ----
labels = sorted(mtr.MERFISH_cell_type_annotation.unique())
lab2code = {l: i for i, l in enumerate(labels)}
y = np.full(n, -1, np.int16)
y[is_train] = mtr.MERFISH_cell_type_annotation.map(lab2code).values
# label -> E/I mapping (verified 1:1 in train)
lab_ei = mtr.groupby("MERFISH_cell_type_annotation").Excitatory_vs_Inhibitory.first()
ei_of_label = np.array([{"excitatory": 1, "inhibitory": 0}.get(lab_ei.get(l), -1) for l in labels], np.int8)

# ---- expression ----
tot = counts.sum(1).values.astype(np.float64)
print(f"total counts: min={tot.min():.0f} median={np.median(tot):.0f}")
scale = np.median(tot)
lognorm = np.log1p(counts.values / np.maximum(tot, 1)[:, None] * scale).astype(np.float32)
Z = StandardScaler().fit_transform(lognorm)
pca = PCA(n_components=50, random_state=0)
P = pca.fit_transform(Z).astype(np.float32)
print(f"PCA50 var explained: {pca.explained_variance_ratio_.sum():.3f}")

# ---- spatial neighbors (within Section_ID, train+test pooled) ----
coords = meta[["center_x", "center_y"]].values
sp_idx = np.full((n, K_NBR), -1, np.int32)
sp_dist = np.full((n, K_NBR), np.inf, np.float32)
sec_sizes = []
for sec, g in meta.groupby("Section_ID"):
    gi = meta.index.get_indexer(g.index)
    sec_sizes.append(len(gi))
    k = min(K_NBR + 1, len(gi))
    tree = cKDTree(coords[gi])
    d, j = tree.query(coords[gi], k=k)
    # drop self (first col), map back to global
    sp_idx[gi, : k - 1] = gi[j[:, 1:]]
    sp_dist[gi, : k - 1] = d[:, 1:]
print(f"sections={len(sec_sizes)} size min/med/max = {min(sec_sizes)}/{int(np.median(sec_sizes))}/{max(sec_sizes)}")

# ---- expression-space neighbors (all cells, PCA50 euclidean) ----
tree = cKDTree(P)
d, j = tree.query(P, k=K_NBR + 1)
ex_idx = j[:, 1:].astype(np.int32)
ex_dist = d[:, 1:].astype(np.float32)

# ---- within-section normalized coords ----
sec_stats = meta.groupby("Section_ID")[["center_x", "center_y"]].agg(["mean", "std"])
sx = meta.Section_ID.map(sec_stats[("center_x", "mean")]).values
sy = meta.Section_ID.map(sec_stats[("center_y", "mean")]).values
sxs = meta.Section_ID.map(sec_stats[("center_x", "std")]).values
sys_ = meta.Section_ID.map(sec_stats[("center_y", "std")]).values
rel_x = ((meta.center_x.values - sx) / np.maximum(sxs, 1)).astype(np.float32)
rel_y = ((meta.center_y.values - sy) / np.maximum(sys_, 1)).astype(np.float32)

# ---- neighbor-mean expression (spatial k=10) ----
nb_mean = np.zeros((n, 50), np.float32)
nb_cnt = np.zeros(n, np.float32)
for kk in range(10):
    v = sp_idx[:, kk]
    ok = v >= 0
    nb_mean[ok] += P[v[ok]]
    nb_cnt[ok] += 1
nb_mean /= np.maximum(nb_cnt, 1)[:, None]

# ---- categorical / meta features (float, NaN = missing) ----
def codes(s):
    c = pd.Categorical(s)
    out = c.codes.astype(np.float32)
    out[out < 0] = np.nan
    return out, list(c.categories)

region, _ = codes(meta.Region)
segment, _ = codes(meta.Segment)
ap, _ = codes(meta.AP_position)
gender = (meta.Gender == "male").astype(np.float32).values
mouse, _ = codes(meta.Mouse_ID)
dsets, _ = codes(meta.Datasets)
ei_obs = meta.Excitatory_vs_Inhibitory.map({"excitatory": 1.0, "inhibitory": 0.0}).values.astype(np.float32)

# train meta has the SAME missingness rates as test (Region/EI ~37% observed,
# Segment ~40%), so no synthetic masking is needed - use observed values as-is.
region_m, ei_m, segment_m = region, ei_obs.copy(), segment
# is Region/EI missing together in test?
both = mte[["Region", "Excitatory_vs_Inhibitory"]].isna()
print("test Region-missing == EI-missing:", (both.Region == both.Excitatory_vs_Inhibitory).all())

# ---- assemble static matrix ----
gene_names = [f"g_{c}" for c in counts.columns]
cols_meta = np.column_stack([
    np.log1p(tot), np.log1p(meta.volume.values), tot / np.maximum(meta.volume.values, 1),
    meta.center_x.values, meta.center_y.values, rel_x, rel_y,
    sp_dist[:, 0], np.nanmean(np.where(np.isfinite(sp_dist[:, :5]), sp_dist[:, :5], np.nan), 1),
    np.nanmean(np.where(np.isfinite(sp_dist[:, :15]), sp_dist[:, :15], np.nan), 1),
    ex_dist[:, 0], ex_dist[:, :10].mean(1),
    region_m, ei_m, segment_m, ap, gender, mouse, dsets,
]).astype(np.float32)
meta_names = ["log_total", "log_volume", "density", "x", "y", "rel_x", "rel_y",
              "sp_d1", "sp_d5", "sp_d15", "ex_d1", "ex_d10",
              "Region", "EI", "Segment", "AP", "gender", "mouse", "dataset"]
X_static = np.hstack([lognorm, P, nb_mean, cols_meta])
names = gene_names + [f"pc{i}" for i in range(50)] + [f"nbm{i}" for i in range(50)] + meta_names
assert X_static.shape[1] == len(names)

# ---- fixed 5-fold stratified CV on train ----
from sklearn.model_selection import StratifiedKFold
folds = np.full(n, -1, np.int8)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
tr_pos = np.where(is_train)[0]
for f, (_, va) in enumerate(skf.split(tr_pos, y[tr_pos])):
    folds[tr_pos[va]] = f

np.savez_compressed(CACHE / "static.npz",
    X=X_static, names=np.array(names), ids=meta.index.values.astype(str),
    is_train=is_train, y=y, labels=np.array(labels), ei_of_label=ei_of_label,
    folds=folds, ei_known=ei_m, region_known=region_m)
np.savez_compressed(CACHE / "nbrs.npz", sp_idx=sp_idx, sp_dist=sp_dist, ex_idx=ex_idx, ex_dist=ex_dist)
print("saved:", X_static.shape, "-> work/cache/")
