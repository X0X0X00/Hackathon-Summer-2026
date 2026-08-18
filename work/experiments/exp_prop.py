"""Graph label propagation (Zhou et al. label spreading) as a diverse ensemble member.

Combined 10k-cell graph: union of cached spatial kNN (within Section_ID) and
expression kNN (PCA50 euclidean). Per-graph edge weight exp(-d/sigma) with
sigma = sigma_scale * median neighbor distance, mixed with weights ws/we,
symmetrized, then S = D^-1/2 W D^-1/2 and F <- alpha*S*F + (1-alpha)*Y0.

Leakage protocol: predicting fold f => Y0 one-hot only for train cells with
folds != f. Graph structure is unsupervised (all 10k cells) - allowed.

Stages (argv[1]):
  time         quick timing sanity check
  screen       staged hyperparam screen on folds 0+1 (mean acc)
  full         full 5-fold + test with BEST config -> save_oof('prop')
  hybrid       LGBM + 60 prop-prob cols, screen folds 0+1 (vs baseline 0.7725)
  hybrid_full  full 5-fold + test -> save_oof('prop_lgbm')
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import sys
import time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load, build_X, apply_ei, save_oof, N_CLASSES

D = load()
N = len(D["y"])
Y, FOLDS, IS_TR = D["y"], D["folds"], D["is_train"]
TR = np.where(IS_TR)[0]
TE = np.where(~IS_TR)[0]
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# best config from `screen` (folds 0+1 mean 0.5100; see prop_screen.log)
BEST = dict(k_sp=25, k_ex=50, ws=0.1, we=1.0, ssig=1.0, esig=1.0,
            alpha=0.5, n_iter=200, symm="max", cmn=False, ex_src="pca20")

_knn_cache = {}


def ex_graph(src):
    """Expression kNN (idx, dist). 'cache' = the cached PCA50-euclidean graph;
    'pcaD' = kNN on the first D dims of the SAME cached PCA features
    (unsupervised denoising - allowed, no labels involved)."""
    if src == "cache":
        return D["ex_idx"], D["ex_dist"]
    if src in _knn_cache:
        return _knn_cache[src]
    path = os.path.join(EXP_DIR, f"prop_nbrs_{src}.npz")
    if os.path.exists(path):
        z = np.load(path)
        _knn_cache[src] = (z["idx"], z["dist"])
        return _knn_cache[src]
    from scipy.spatial import cKDTree
    dim = int(src.replace("pca", ""))
    P = D["X"][:, 200:200 + dim].astype(np.float64)  # cols 200:250 are pc0..pc49
    d, j = cKDTree(P).query(P, k=51)
    idx, dist = j[:, 1:].astype(np.int32), d[:, 1:].astype(np.float32)
    np.savez_compressed(path, idx=idx, dist=dist)
    _knn_cache[src] = (idx, dist)
    return idx, dist


_graph_cache = {}


def build_S(k_sp, k_ex, ws, we, ssig, esig, symm="avg", ex_src="cache", **_):
    key = (k_sp, k_ex, round(ws, 4), round(we, 4), ssig, esig, symm, ex_src)
    if key in _graph_cache:
        return _graph_cache[key]
    exi, exd = ex_graph(ex_src)
    mats = []
    for idx, dist, k, w, sc in ((D["sp_idx"], D["sp_dist"], k_sp, ws, ssig),
                                (exi, exd, k_ex, we, esig)):
        if w <= 0 or k <= 0:
            continue
        ii = idx[:, :k]
        dd = dist[:, :k]
        ok = (ii >= 0) & np.isfinite(dd)
        sigma = sc * np.median(dd[ok])
        rows = np.repeat(np.arange(N), k)[ok.ravel()]
        cols = ii.ravel()[ok.ravel()]
        vals = (w * np.exp(-dd.ravel()[ok.ravel()] / sigma)).astype(np.float32)
        mats.append(sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr())
    W = mats[0] if len(mats) == 1 else mats[0] + mats[1]
    W = W.maximum(W.T) if symm == "max" else (W + W.T) * 0.5
    deg = np.asarray(W.sum(1)).ravel()
    dinv = sp.diags((1.0 / np.sqrt(np.maximum(deg, 1e-12))).astype(np.float32))
    S = (dinv @ W @ dinv).tocsr().astype(np.float32)
    _graph_cache[key] = S
    if len(_graph_cache) > 6:  # keep memory bounded
        _graph_cache.pop(next(iter(_graph_cache)))
    return S


def propagate(S, known, alpha, n_iter, cmn=False, tol=1e-5, **_):
    """known: (N,) int, -1 = hidden. Returns prob matrix (N, 60), rows renorm."""
    Y0 = np.zeros((N, N_CLASSES), np.float32)
    lab = known >= 0
    Y0[lab, known[lab]] = 1.0
    base = (1.0 - alpha) * Y0
    F = Y0.copy()
    for _t in range(n_iter):
        Fn = alpha * (S @ F) + base
        delta = np.abs(Fn - F).max()
        F = Fn
        if delta < tol:
            break
    if cmn:  # class-mass normalization toward labeled-class priors
        prior = Y0[lab].sum(0) / lab.sum()
        mass = F[~lab].sum(0)
        F = F * (prior / np.maximum(mass / max(mass.sum(), 1e-12), 1e-12))
    z = F.sum(1, keepdims=True)
    out = np.where(z > 1e-12, F / np.maximum(z, 1e-12), 1.0 / N_CLASSES)
    return out.astype(np.float32)


def known_for_fold(f):
    """Labels visible when predicting fold f (f=None => all train visible)."""
    if f is None:
        return np.where(IS_TR, Y, -1).astype(np.int64)
    return np.where(IS_TR & (FOLDS != f), Y, -1).astype(np.int64)


def screen_config(cfg, folds=(0, 1)):
    S = build_S(**cfg)
    accs = []
    for f in folds:
        F = propagate(S, known_for_fold(f), **cfg)
        va = TR[FOLDS[TR] == f]
        accs.append((F[va].argmax(1) == Y[va]).mean())
    return float(np.mean(accs)), accs


def run_full(cfg):
    S = build_S(**cfg)
    oof = np.zeros((N, N_CLASSES), np.float32)
    for f in range(5):
        F = propagate(S, known_for_fold(f), **cfg)
        va = TR[FOLDS[TR] == f]
        oof[va] = F[va]
        print(f"  fold{f}: {(oof[va].argmax(1) == Y[va]).mean():.4f}", flush=True)
    Ffin = propagate(S, known_for_fold(None), **cfg)
    acc = (oof[TR].argmax(1) == Y[TR]).mean()
    oof_ei = apply_ei(oof[TR], D["ei_known"][TR], D["ei_of_label"])
    acc_ei = (oof_ei.argmax(1) == Y[TR]).mean()
    print(f"  OOF acc={acc:.4f}  +EI={acc_ei:.4f}")
    return {"oof": oof[TR], "test": Ffin[TE], "acc": acc, "acc_ei": acc_ei}


# ---------------- hybrid: LGBM + prop-prob features ----------------
import lightgbm as lgb

PARAMS = dict(
    objective="multiclass", num_class=60, learning_rate=0.06,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbosity=-1, num_threads=4, seed=0,
)


def lgbm_fold(f, cfg, S):
    """Leakage-free fold f: baseline features + 60 prop probs, both built
    from labels with fold f hidden (f=None => final test model)."""
    known = known_for_fold(f)
    Xb, names = build_X(D, known, 15, 25)
    P = propagate(S, known, **cfg)
    X = np.hstack([Xb, P]).astype(np.float32)
    names = names + [f"prop{c}" for c in range(N_CLASSES)]
    if f is None:
        fit, va = TR, TE
    else:
        fit = TR[FOLDS[TR] != f]
        va = TR[FOLDS[TR] == f]
    ds = lgb.Dataset(X[fit], Y[fit], feature_name=[n.replace(" ", "_") for n in names])
    m = lgb.train(PARAMS, ds, num_boost_round=400)
    return va, m.predict(X[va])


def hybrid(folds, save_name=None):
    S = build_S(**BEST)
    oof = np.zeros((N, N_CLASSES), np.float32)
    for f in folds:
        t0 = time.time()
        va, p = lgbm_fold(f, BEST, S)
        oof[va] = p
        print(f"  fold{f}: {(p.argmax(1) == Y[va]).mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    accs = [(oof[TR[FOLDS[TR] == f]].argmax(1) == Y[TR[FOLDS[TR] == f]]).mean() for f in folds]
    print(f"  mean over folds {list(folds)}: {np.mean(accs):.4f}")
    if save_name:
        _, test_p = lgbm_fold(None, BEST, S)
        acc = (oof[TR].argmax(1) == Y[TR]).mean()
        oof_ei = apply_ei(oof[TR], D["ei_known"][TR], D["ei_of_label"])
        acc_ei = (oof_ei.argmax(1) == Y[TR]).mean()
        print(f"  OOF acc={acc:.4f}  +EI={acc_ei:.4f}")
        save_oof(save_name, {"oof": oof[TR], "test": test_p, "acc": acc, "acc_ei": acc_ei})


# ---------------- stages ----------------
def stage_time():
    cfg = dict(BEST)
    t0 = time.time()
    S = build_S(**cfg)
    t1 = time.time()
    m, accs = screen_config(cfg)
    print(f"build_S {t1-t0:.1f}s  screen(2 folds) {time.time()-t1:.1f}s  "
          f"mean={m:.4f} f0={accs[0]:.4f} f1={accs[1]:.4f}  nnz={S.nnz}")


def stage_screen():
    results = {}

    def go(tag, **over):
        cfg = {**BEST, **over}
        t0 = time.time()
        m, accs = screen_config(cfg)
        results[tag] = (m, cfg)
        print(f"SCREEN {tag:26s} mean={m:.4f}  f0={accs[0]:.4f} f1={accs[1]:.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)
        return m

    def promote(stage):
        best = max(results, key=lambda t: results[t][0])
        BEST.update(results[best][1])
        print(f"stage {stage} best so far: {best}  {results[best][0]:.4f}")

    print("--- stage A: expression graph source & k (ws=0, alpha=0.5) ---")
    for src in ("cache", "pca15", "pca20"):
        for k in (10, 25, 50):
            go(f"{src}_k{k}", ex_src=src, k_ex=k)
    promote("A")

    print("--- stage B: alpha ---")
    for a in (0.2, 0.3, 0.7, 0.9):
        go(f"alpha{a}", alpha=a)
    promote("B")

    print("--- stage C: spatial admixture ---")
    for wsp in (0.05, 0.1, 0.2, 0.5):
        for ksp in (10, 25):
            go(f"sp{wsp}_k{ksp}", ws=wsp, k_sp=ksp)
    promote("C")

    print("--- stage D: sigma / iters / symm / cmn ---")
    for es in (0.3, 0.5, 2.0):
        go(f"esig{es}", esig=es)
    for it in (3, 5, 10, 20):
        go(f"iter{it}", n_iter=it, tol=0.0)
    go("symm_max", symm="max")
    go("cmn", cmn=True)
    promote("D")
    best = max(results, key=lambda t: results[t][0])
    print(f"\nFINAL BEST: {best}  mean={results[best][0]:.4f}")
    print("cfg =", {k: v for k, v in BEST.items()})


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "time"
    if stage == "time":
        stage_time()
    elif stage == "screen":
        stage_screen()
    elif stage == "full":
        res = run_full(BEST)
        save_oof("prop", res)
    elif stage == "hybrid":
        hybrid((0, 1))
    elif stage == "hybrid_full":
        hybrid(range(5), save_name="prop_lgbm")
    else:
        raise SystemExit(f"unknown stage {stage}")
