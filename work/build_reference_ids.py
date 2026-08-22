"""Build external/reference_ids.npy = deposit cells allowed as REFERENCE.

Rule: all 146,621 cells of Zenodo 18039571 MERFISH_spinal_cord_resolved_0718.h5ad
      MINUS the 10,000 competition Cell_IDs (train + test)
      MINUS rows whose 200-gene count vector exactly equals any competition cell (fingerprint dupes).
Verification uses TRAIN cells only (labels/counts/coords/section identical); test labels are never read.
Expected: 136,574 ids (47 fingerprint duplicates removed).
"""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, hashlib
from pathlib import Path
WORK = Path(__file__).resolve().parent; BASE = WORK.parent
H5 = WORK / "external/MERFISH_spinal_cord_resolved_0718.h5ad"
assert hashlib.md5(H5.read_bytes()).hexdigest() == "ce06f62c0ec4973581dae17bb76f0cd9", "unexpected h5ad"
a = ad.read_h5ad(H5)
mt = pd.read_csv(BASE / "data/meta_train.csv", index_col=0); ms = pd.read_csv(BASE / "data/meta_test.csv", index_col=0)
ct = pd.read_csv(BASE / "data/counts_train.csv", index_col=0); cs = pd.read_csv(BASE / "data/counts_test.csv", index_col=0)
genes = ct.columns.tolist()
comp_ids = set(mt.index.astype(str)) | set(ms.index.astype(str))
ext_ids = np.array(a.obs_names, dtype=str)
assert comp_ids <= set(ext_ids) and set(genes) <= set(a.var_names)
# verify alignment on TRAIN cells only
tr_ids = list(mt.index.astype(str)); sub = a[tr_ids]
norm = lambda s: str(s).replace(" ", "_").replace("-", "_")
assert (sub.obs["MERFISH cell type annotation"].astype(str).map(norm).values == mt.MERFISH_cell_type_annotation.values).all()
Xe = sub[:, genes].X; Xe = Xe.toarray() if sp.issparse(Xe) else np.asarray(Xe)
assert np.array_equal(Xe.astype(np.int64), ct[genes].values.astype(np.int64))
assert np.allclose(sub.obs[["center_x", "center_y"]].values.astype(float), mt[["center_x", "center_y"]].values, atol=1e-3)
ref_mask = ~np.isin(ext_ids, list(comp_ids))
# fingerprint duplicates vs ALL competition count vectors (train+test counts are public inputs)
comp_counts = np.vstack([ct[genes].values, cs[genes].values]).astype(np.int64)
Xr = a[ref_mask][:, genes].X; Xr = (Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)).astype(np.int64)
def rh(M): return pd.util.hash_array(np.ascontiguousarray(M).view(np.dtype((np.void, M.dtype.itemsize * M.shape[1]))).ravel())
dup = np.isin(rh(Xr), list(set(rh(comp_counts))))
keep = ext_ids[ref_mask][~dup]
print(f"deposit={len(ext_ids)} minus competition ids={(~ref_mask).sum()} minus fingerprint dupes={dup.sum()} -> reference={len(keep)}")
np.save(WORK / "external/reference_ids.npy", keep.astype(str))
