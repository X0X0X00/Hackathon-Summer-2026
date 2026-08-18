"""Neighbor-label feature optimization experiments.

Screens variants on fold 0 only (~70s each), then full 5-fold CV for winners.
Leakage protocol identical to common.run_cv: when predicting fold f, known
labels = train cells with folds != f only. LGBM params identical to baseline.

Usage:
  python experiments/exp_nbr.py screen <cfg1> <cfg2> ...   # fold-0 screen
  python experiments/exp_nbr.py full <cfg> <oof_name>      # full 5-fold CV + save
  python experiments/exp_nbr.py list                       # list configs
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load, save_oof, apply_ei, N_CLASSES  # noqa: E402

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


# ---------------- neighbor-label feature machinery ----------------

def labeled_neighbors(known, idx, dist, k):
    """First-k labeled neighbors per cell.

    Returns labels (n,k) pad -1, dists (n,k) pad inf, global ids (n,k) pad -1.
    """
    n, K = idx.shape
    lab = np.full((n, k), -1, np.int32)
    dd = np.full((n, k), np.inf, np.float32)
    gid = np.full((n, k), -1, np.int32)
    taken = np.zeros(n, np.int32)
    rows = np.arange(n)
    for c in range(K):
        j = idx[:, c]
        ok = (j >= 0) & (known[np.maximum(j, 0)] >= 0) & (taken < k) & np.isfinite(dist[:, c])
        r = rows[ok]
        lab[r, taken[ok]] = known[j[ok]].astype(np.int32)
        dd[r, taken[ok]] = dist[ok, c]
        gid[r, taken[ok]] = j[ok]
        taken[ok] += 1
    return lab, dd, gid


def hist_block(known, idx, dist, k, tag, weight="inv", sigma_mult=1.0):
    """Weighted class histogram over first k labeled neighbors.

    weight: 'inv' = 1/(1+d) (baseline), 'uniform', 'gauss' = exp(-(d/sigma)^2)
    with adaptive per-cell sigma = (dist to k-th labeled neighbor) * sigma_mult.
    Returns (features (n, C+2), names). Matches common.label_hist for weight='inv'.
    """
    lab, dd, _ = labeled_neighbors(known, idx, dist, k)
    n = lab.shape[0]
    valid = lab >= 0
    if weight == "uniform":
        w = valid.astype(np.float32)
    elif weight == "inv":
        w = np.where(valid, 1.0 / (1.0 + dd), 0.0).astype(np.float32)
    elif weight == "gauss":
        dmax = np.where(valid, dd, 0.0).max(1)
        sigma = np.maximum(dmax * sigma_mult, 1e-6)
        w = np.where(valid, np.exp(-((dd / sigma[:, None]) ** 2)), 0.0).astype(np.float32)
    else:
        raise ValueError(weight)
    flat = (np.arange(n)[:, None] * N_CLASSES + np.maximum(lab, 0)).ravel()
    hist = np.bincount(flat[valid.ravel()], weights=w.ravel()[valid.ravel()],
                       minlength=n * N_CLASSES).reshape(n, N_CLASSES).astype(np.float32)
    tot = hist.sum(1, keepdims=True)
    hist = hist / np.maximum(tot, 1e-9)
    cnt = valid.sum(1).astype(np.float32)
    dsum = np.where(valid, dd, 0.0).sum(1)
    out = np.hstack([hist, cnt[:, None], (dsum / np.maximum(cnt, 1))[:, None]]).astype(np.float32)
    names = [f"{tag}_h{c}" for c in range(N_CLASSES)] + [f"{tag}_n", f"{tag}_d"]
    return out, names


def class_dist_block(known, idx, dist, tag):
    """Per-class proximity profile over ALL cached neighbors.

    For each class c: similarity 1/(1+d_min) to the nearest labeled neighbor
    of class c among the 50 cached neighbors (0 if none). 60 features.
    """
    n, K = idx.shape
    best = np.full((n, N_CLASSES), np.inf, np.float32)
    for c in range(K):
        j = idx[:, c]
        ok = (j >= 0) & (known[np.maximum(j, 0)] >= 0) & np.isfinite(dist[:, c])
        r = np.where(ok)[0]
        lb = known[j[ok]].astype(np.int64)
        np.minimum.at(best, (r, lb), dist[ok, c])
    sim = np.where(np.isfinite(best), 1.0 / (1.0 + best), 0.0).astype(np.float32)
    return sim, [f"{tag}_c{c}" for c in range(N_CLASSES)]


def twohop_lso_block(known, idx, dist, k, m, tag):
    """Leave-self-out two-hop label smoothing.

    T_i = weighted avg over i's first m neighbors j of j's one-hop histogram
    H_j, with cell i's OWN contribution removed from H_j before normalizing.
    Without the removal, training cells (label known) see their own label
    reflected back through neighbors while masked validation cells do not ->
    train/val feature shift (screened at 0.57 vs 0.759 baseline).
    """
    import scipy.sparse as sparse
    n = idx.shape[0]
    lab, dd, gid = labeled_neighbors(known, idx, dist, k)
    valid = lab >= 0
    w = np.where(valid, 1.0 / (1.0 + dd), 0.0).astype(np.float64)
    flat = (np.arange(n)[:, None] * N_CLASSES + np.maximum(lab, 0)).ravel()
    U = np.bincount(flat[valid.ravel()], weights=w.ravel()[valid.ravel()],
                    minlength=n * N_CLASSES).reshape(n, N_CLASSES)
    tot = w.sum(1)
    # S[j, i] = weight of global cell i inside j's one-hop histogram
    S = sparse.csr_matrix(
        (w.ravel()[valid.ravel()],
         (np.repeat(np.arange(n), lab.shape[1])[valid.ravel()], gid.ravel()[valid.ravel()])),
        shape=(n, n))
    acc = np.zeros((n, N_CLASSES), np.float64)
    wsum = np.zeros(n, np.float64)
    for c in range(m):
        j = idx[:, c]
        ok = (j >= 0) & np.isfinite(dist[:, c])
        i_idx = np.where(ok)[0]
        jj = j[i_idx]
        s = np.asarray(S[jj, i_idx]).ravel()  # my weight in neighbor j's hist (0 if unlabeled)
        Uj = U[jj].copy()
        ki = known[i_idx]
        has = (ki >= 0) & (s > 0)
        hr = np.where(has)[0]
        Uj[hr, ki[has]] = np.maximum(Uj[hr, ki[has]] - s[has], 0.0)
        totj = np.maximum(tot[jj] - np.where(ki >= 0, s, 0.0), 1e-9)
        Hj = Uj / totj[:, None]
        w2 = 1.0 / (1.0 + dist[i_idx, c])
        acc[i_idx] += w2[:, None] * Hj
        wsum[i_idx] += w2
    T = (acc / np.maximum(wsum, 1e-9)[:, None]).astype(np.float32)
    return T, [f"{tag}_t{c}" for c in range(N_CLASSES)]


def twohop_block(hist, idx, dist, m, tag, weight="inv"):
    """Average the one-hop histograms of each cell's first m neighbors.

    hist: (n, C) normalized one-hop histogram (leakage-safe by construction:
    a masked cell's own label never enters any histogram). Neighbors here need
    not be labeled -- their histograms are label-derived but self-excluding.
    """
    n = idx.shape[0]
    acc = np.zeros((n, N_CLASSES), np.float32)
    wsum = np.zeros(n, np.float32)
    for c in range(m):
        j = idx[:, c]
        ok = (j >= 0) & np.isfinite(dist[:, c])
        w = 1.0 / (1.0 + dist[ok, c]) if weight == "inv" else np.ones(ok.sum(), np.float32)
        acc[ok] += w[:, None] * hist[j[ok]]
        wsum[ok] += w
    acc /= np.maximum(wsum, 1e-9)[:, None]
    return acc, [f"{tag}_t{c}" for c in range(N_CLASSES)]


# ---------------- config -> build function ----------------
# spec: list of blocks; each block one of
#   ("sp"|"ex", k, weight, sigma_mult)          one-hop histogram
#   ("th_sp"|"th_ex", k, m)                     two-hop: avg of inv one-hop k-hists over m nbrs

def make_build(spec):
    def build(D, known):
        parts, names = [D["X"]], list(D["names"])
        onehop_cache = {}
        for blk in spec:
            kind = blk[0]
            if kind in ("sp", "ex"):
                _, k, weight, sm = blk
                idx, dist = D[f"{kind}_idx"], D[f"{kind}_dist"]
                tag = f"{kind}{k}{weight[0]}" + (f"{sm}" if weight == "gauss" else "")
                h, nm = hist_block(known, idx, dist, k, tag, weight, sm)
                onehop_cache[(kind, k)] = h[:, :N_CLASSES]
                parts.append(h); names += nm
            elif kind in ("th_sp", "th_ex"):
                _, k, m = blk
                g = kind[3:]
                idx, dist = D[f"{g}_idx"], D[f"{g}_dist"]
                if (g, k) in onehop_cache:
                    h1 = onehop_cache[(g, k)]
                else:
                    hb, _ = hist_block(known, idx, dist, k, "tmp", "inv", 1.0)
                    h1 = hb[:, :N_CLASSES]
                t, nm = twohop_block(h1, idx, dist, m, f"{kind}{k}m{m}")
                parts.append(t); names += nm
            elif kind in ("thl_sp", "thl_ex"):
                _, k, m = blk
                g = kind[4:]
                idx, dist = D[f"{g}_idx"], D[f"{g}_dist"]
                t, nm = twohop_lso_block(known, idx, dist, k, m, f"{kind}{k}m{m}")
                parts.append(t); names += nm
            elif kind in ("cd_sp", "cd_ex"):
                g = kind[3:]
                t, nm = class_dist_block(known, D[f"{g}_idx"], D[f"{g}_dist"], kind)
                parts.append(t); names += nm
            else:
                raise ValueError(blk)
        return np.hstack(parts), names
    return build


CONFIGS = {
    # reference (must reproduce baseline fold0 = 0.7590)
    "base":   [("sp", 15, "inv", 1.0), ("ex", 25, "inv", 1.0)],
    # sp_k sweep (ex fixed 25)
    "sp5":    [("sp", 5, "inv", 1.0), ("ex", 25, "inv", 1.0)],
    "sp10":   [("sp", 10, "inv", 1.0), ("ex", 25, "inv", 1.0)],
    "sp25":   [("sp", 25, "inv", 1.0), ("ex", 25, "inv", 1.0)],
    "sp40":   [("sp", 40, "inv", 1.0), ("ex", 25, "inv", 1.0)],
    # ex_k sweep (sp fixed 15)
    "ex10":   [("sp", 15, "inv", 1.0), ("ex", 10, "inv", 1.0)],
    "ex15":   [("sp", 15, "inv", 1.0), ("ex", 15, "inv", 1.0)],
    "ex40":   [("sp", 15, "inv", 1.0), ("ex", 40, "inv", 1.0)],
    "ex50":   [("sp", 15, "inv", 1.0), ("ex", 50, "inv", 1.0)],
    # weighting schemes at base k
    "unif":   [("sp", 15, "uniform", 1.0), ("ex", 25, "uniform", 1.0)],
    "gauss1": [("sp", 15, "gauss", 1.0), ("ex", 25, "gauss", 1.0)],
    "gauss5": [("sp", 15, "gauss", 0.5), ("ex", 25, "gauss", 0.5)],
    # all 50 cached neighbors, distance-weighted
    "all50i": [("sp", 50, "inv", 1.0), ("ex", 50, "inv", 1.0)],
    "all50g": [("sp", 50, "gauss", 0.5), ("ex", 50, "gauss", 0.5)],
    # multi-k blocks
    "multik": [("sp", 5, "inv", 1.0), ("sp", 15, "inv", 1.0), ("sp", 40, "inv", 1.0),
               ("ex", 10, "inv", 1.0), ("ex", 25, "inv", 1.0), ("ex", 50, "inv", 1.0)],
    # two-hop smoothing added to base
    "twohop": [("sp", 15, "inv", 1.0), ("ex", 25, "inv", 1.0),
               ("th_sp", 15, 10), ("th_ex", 25, 10)],
    # ---- round 2 ----
    # two ex blocks (best single-k was ex10/ex40/ex50)
    "ex1040": [("sp", 15, "inv", 1.0), ("ex", 10, "inv", 1.0), ("ex", 40, "inv", 1.0)],
    # lean 4-block multi-k
    "mk4":    [("sp", 5, "inv", 1.0), ("sp", 25, "inv", 1.0),
               ("ex", 10, "inv", 1.0), ("ex", 40, "inv", 1.0)],
    # leave-self-out two-hop added to base
    "thl":    [("sp", 15, "inv", 1.0), ("ex", 25, "inv", 1.0),
               ("thl_sp", 15, 10), ("thl_ex", 25, 10)],
    # ex_k=40 single (screen tie-winner) + LSO two-hop
    "thl40":  [("sp", 15, "inv", 1.0), ("ex", 40, "inv", 1.0),
               ("thl_sp", 15, 10), ("thl_ex", 40, 10)],
    # ---- round 3: thl ablations ----
    "thl_m5":  [("sp", 15, "inv", 1.0), ("ex", 25, "inv", 1.0),
                ("thl_sp", 15, 5), ("thl_ex", 25, 5)],
    "thl_m25": [("sp", 15, "inv", 1.0), ("ex", 25, "inv", 1.0),
                ("thl_sp", 15, 25), ("thl_ex", 25, 25)],
    "thl_sp":  [("sp", 15, "inv", 1.0), ("ex", 25, "inv", 1.0),
                ("thl_sp", 15, 10)],
    "thl_ex":  [("sp", 15, "inv", 1.0), ("ex", 25, "inv", 1.0),
                ("thl_ex", 25, 10)],
    # ---- round 4: per-class proximity profile ----
    "clsd":    [("sp", 15, "inv", 1.0), ("ex", 25, "inv", 1.0),
                ("cd_sp",), ("cd_ex",)],
    "clsd_ex": [("sp", 15, "inv", 1.0), ("ex", 25, "inv", 1.0), ("cd_ex",)],
}


def screen(D, names, use_folds=(0,)):
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    tr = np.where(is_tr)[0]
    for name in names:
        t0 = time.time()
        accs = []
        for f in use_folds:
            known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
            fit = tr[folds[tr] != f]
            va = tr[folds[tr] == f]
            X, fn = make_build(CONFIGS[name])(D, known)
            probs = fit_predict(X[fit], y[fit], X[va], fn)
            accs.append((probs.argmax(1) == y[va]).mean())
        detail = " ".join(f"f{f}={a:.4f}" for f, a in zip(use_folds, accs))
        line = (f"SCREEN {name:10s} mean={np.mean(accs):.4f}  {detail}  "
                f"nfeat={X.shape[1]}  {time.time()-t0:.0f}s")
        print(line, flush=True)
        with open(Path(__file__).parent / "screen_log.txt", "a") as fh:
            fh.write(line + "\n")


def run_cv_custom(D, build, verbose=True):
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    tr = np.where(is_tr)[0]
    te = np.where(~is_tr)[0]
    oof = np.zeros((len(y), N_CLASSES), np.float32)
    for f in range(5):
        known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
        X, names = build(D, known)
        fit = tr[folds[tr] != f]
        va = tr[folds[tr] == f]
        oof[va] = fit_predict(X[fit], y[fit], X[va], names)
        if verbose:
            print(f"  fold{f}: {(oof[va].argmax(1) == y[va]).mean():.4f}", flush=True)
    acc = (oof[tr].argmax(1) == y[tr]).mean()
    oof_ei = apply_ei(oof[tr], D["ei_known"][tr], D["ei_of_label"])
    acc_ei = (oof_ei.argmax(1) == y[tr]).mean()
    known = np.where(is_tr, y, -1).astype(np.int64)
    X, names = build(D, known)
    test_probs = fit_predict(X[tr], y[tr], X[te], names)
    if verbose:
        print(f"  OOF acc={acc:.4f}  +EI={acc_ei:.4f}")
    return {"oof": oof[tr], "test": test_probs, "acc": acc, "acc_ei": acc_ei}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        for k, v in CONFIGS.items():
            print(k, v)
        sys.exit(0)
    D = load()
    if cmd == "screen":
        args = sys.argv[2:]
        use_folds = (0,)
        if args and args[0].startswith("folds:"):
            use_folds = tuple(int(x) for x in args[0][6:].split(","))
            args = args[1:]
        screen(D, args, use_folds)
    elif cmd == "full":
        cfg, oof_name = sys.argv[2], sys.argv[3]
        res = run_cv_custom(D, make_build(CONFIGS[cfg]))
        save_oof(oof_name, res)
    elif cmd == "finals":
        # full CV on each named config; best -> nbr_best, runner-up -> nbr_alt
        # (only if within 0.005 of best)
        results = []
        for cfg in sys.argv[2:]:
            print(f"FULL {cfg}", flush=True)
            res = run_cv_custom(D, make_build(CONFIGS[cfg]))
            print(f"FULLCV {cfg} acc={res['acc']:.4f} acc_ei={res['acc_ei']:.4f}", flush=True)
            results.append((cfg, res))
        results.sort(key=lambda t: -t[1]["acc"])
        best_cfg, best = results[0]
        print(f"BEST {best_cfg}", flush=True)
        save_oof("nbr_best", best)
        if len(results) > 1:
            alt_cfg, alt = results[1]
            if best["acc"] - alt["acc"] <= 0.005:
                print(f"ALT {alt_cfg}", flush=True)
                save_oof("nbr_alt", alt)
            else:
                print(f"ALT {alt_cfg} skipped (gap {best['acc']-alt['acc']:.4f} > 0.005)", flush=True)
    else:
        raise SystemExit(f"unknown cmd {cmd}")
