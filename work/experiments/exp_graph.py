"""Round 2: upgrade the EXPRESSION kNN graph feeding the ex label histograms.

Variants (all 50 nbrs, train+test pooled, self excluded, exact brute-force kNN):
  pca100  - euclidean in PCA100 of standardized lognorm
  cos50   - cosine distance on PCA50 (row-normalize -> euclidean equivalent)
  corr200 - correlation distance on lognorm genes (standardize genes, center+
            normalize rows -> euclidean equivalent, 200-dim)
  snn     - shared-nearest-neighbor rerank of the cached PCA50 graph:
            dist(i,j) = 1 - |knn(i) & knn(j)| / 50, ties broken by orig rank

Usage:
  python experiments/exp_graph.py build                      # build all graph caches
  python experiments/exp_graph.py screen [folds:0,1] v[:ex_k] ...   # screen variants
  python experiments/exp_graph.py full <variant> <ex_k> <oof_name>  # full 5-fold CV
Graph caches: experiments/nbrs_<variant>.npz  (ex_idx int32, ex_dist float32)
'base' as variant name = original cache/nbrs.npz graph.
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load, save_oof, run_cv, build_X  # noqa: E402

HERE = Path(__file__).resolve().parent
K_NBR = 50
N_JOBS = 4
PARAMS = dict(
    objective="multiclass", num_class=60, learning_rate=0.06,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbosity=-1, num_threads=N_JOBS, seed=0,
)


def fit_predict(Xtr, ytr, Xva, names):
    ds = lgb.Dataset(Xtr, ytr, feature_name=[n.replace(" ", "_") for n in names])
    m = lgb.train(PARAMS, ds, num_boost_round=400)
    return m.predict(Xva)


# ---------------- graph construction ----------------

def knn_excl_self(pts, k=K_NBR):
    """Exact brute-force kNN, self excluded (robust to duplicate points)."""
    from sklearn.neighbors import NearestNeighbors
    n = len(pts)
    nn = NearestNeighbors(n_neighbors=min(k + 2, n), algorithm="brute",
                          metric="euclidean", n_jobs=N_JOBS).fit(pts)
    d, j = nn.kneighbors(pts)
    not_self = j != np.arange(n)[:, None]
    # stable-push self entries to the end, keep first k of the rest
    order = np.argsort(~not_self, axis=1, kind="stable")
    jj = np.take_along_axis(j, order, 1)[:, :k].astype(np.int32)
    dd = np.take_along_axis(d, order, 1)[:, :k].astype(np.float32)
    return jj, dd


def get_lognorm_pca50(D):
    names = D["names"]
    g_cols = [i for i, nm in enumerate(names) if nm.startswith("g_")]
    p_cols = [i for i, nm in enumerate(names) if nm.startswith("pc")]
    assert len(g_cols) == 200 and len(p_cols) == 50
    return D["X"][:, g_cols].astype(np.float64), D["X"][:, p_cols].astype(np.float64)


def build_graphs():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    D = load()
    lognorm, P50 = get_lognorm_pca50(D)
    Z = StandardScaler().fit_transform(lognorm)

    t0 = time.time()
    pca = PCA(n_components=100, random_state=0)
    P100 = pca.fit_transform(Z)
    jj, dd = knn_excl_self(P100)
    np.savez_compressed(HERE / "nbrs_pca100.npz", ex_idx=jj, ex_dist=dd)
    print(f"pca100: var explained {pca.explained_variance_ratio_.sum():.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    U = P50 / np.maximum(np.linalg.norm(P50, axis=1, keepdims=True), 1e-12)
    jj, dd = knn_excl_self(U)
    np.savez_compressed(HERE / "nbrs_cos50.npz", ex_idx=jj, ex_dist=dd)
    print(f"cos50: done ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    R = Z - Z.mean(1, keepdims=True)
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    jj, dd = knn_excl_self(R)
    np.savez_compressed(HERE / "nbrs_corr200.npz", ex_idx=jj, ex_dist=dd)
    print(f"corr200: done ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    import scipy.sparse as sparse
    idx = D["ex_idx"]  # (n, 50) cached PCA50 graph
    n, K = idx.shape
    A = sparse.csr_matrix(
        (np.ones(n * K, np.float32),
         (np.repeat(np.arange(n), K), idx.ravel().astype(np.int64))),
        shape=(n, n))
    snn_d = np.empty((n, K), np.float32)
    for lo in range(0, n, 1000):
        hi = min(lo + 1000, n)
        C = (A[lo:hi] @ A.T).tocsr()  # C[r, j] = |knn(lo+r) & knn(j)|
        rows = np.repeat(np.arange(hi - lo), K)
        ov = np.asarray(C[rows, idx[lo:hi].ravel()]).ravel()
        snn_d[lo:hi] = (1.0 - ov.reshape(hi - lo, K) / K).astype(np.float32)
    order = np.argsort(snn_d, axis=1, kind="stable")  # ties -> orig rank
    jj = np.take_along_axis(idx, order, 1).astype(np.int32)
    dd = np.take_along_axis(snn_d, order, 1)
    np.savez_compressed(HERE / "nbrs_snn.npz", ex_idx=jj, ex_dist=dd)
    print(f"snn: mean dist {dd.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)


# ---------------- harness plumbing ----------------

def with_graph(D, variant):
    if variant == "base":
        return D
    g = np.load(HERE / f"nbrs_{variant}.npz")
    D2 = dict(D)
    D2["ex_idx"] = g["ex_idx"]
    D2["ex_dist"] = g["ex_dist"]
    return D2


def screen(D, specs, use_folds=(0, 1)):
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    tr = np.where(is_tr)[0]
    for spec in specs:
        variant, _, kstr = spec.partition(":")
        ex_k = int(kstr) if kstr else 25
        D2 = with_graph(D, variant)
        t0 = time.time()
        accs = []
        for f in use_folds:
            known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
            X, fn = build_X(D2, known, sp_k=15, ex_k=ex_k)
            fit = tr[folds[tr] != f]
            va = tr[folds[tr] == f]
            probs = fit_predict(X[fit], y[fit], X[va], fn)
            accs.append((probs.argmax(1) == y[va]).mean())
        detail = " ".join(f"f{f}={a:.4f}" for f, a in zip(use_folds, accs))
        line = (f"GRAPH {spec:14s} mean={np.mean(accs):.4f}  {detail}  "
                f"{time.time()-t0:.0f}s")
        print(line, flush=True)
        with open(HERE / "graph_log.txt", "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build_graphs()
    elif cmd == "screen":
        D = load()
        args = sys.argv[2:]
        use_folds = (0, 1)
        if args and args[0].startswith("folds:"):
            use_folds = tuple(int(x) for x in args[0][6:].split(","))
            args = args[1:]
        screen(D, args, use_folds)
    elif cmd == "full":
        variant, ex_k, oof_name = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        D = load()
        res = run_cv(with_graph(D, variant), fit_predict, sp_k=15, ex_k=ex_k)
        print(f"FULLCV {variant}:{ex_k} acc={res['acc']:.4f} acc_ei={res['acc_ei']:.4f}")
        save_oof(oof_name, res)
    else:
        raise SystemExit(f"unknown cmd {cmd}")
