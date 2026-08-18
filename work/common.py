"""Shared harness: cached feature loading, fold-aware neighbor-label features,
CV evaluation, E/I constraint, OOF saving.

Fold protocol (fixed, do not change): 5-fold stratified, folds in cache.
For fold f, neighbor-label features may only use labels of train cells with
folds != f. Final test predictions use all train labels.
"""
import numpy as np
from pathlib import Path

WORK = Path(__file__).resolve().parent
CACHE = WORK / "cache"
OOF = WORK / "oof"
N_CLASSES = 60


def load():
    s = np.load(CACHE / "static.npz", allow_pickle=True)
    nb = np.load(CACHE / "nbrs.npz")
    d = {k: s[k] for k in s.files}
    d.update({k: nb[k] for k in nb.files})
    d["names"] = list(d["names"])
    return d


def label_hist(known, idx, dist, k, tag):
    """Weighted class histogram over each cell's first k *labeled* neighbors.

    known: (n,) int codes, -1 = label not visible. idx/dist: (n, K) neighbor
    cache. Returns (features (n, C+2), names). Weights 1/(1+d); self never in idx.
    """
    n, K = idx.shape
    hist = np.zeros((n, N_CLASSES), np.float32)
    cnt = np.zeros(n, np.float32)
    dsum = np.zeros(n, np.float32)
    taken = np.zeros(n, np.int32)
    rows = np.arange(n)
    for c in range(K):
        j = idx[:, c]
        ok = (j >= 0) & (known[np.maximum(j, 0)] >= 0) & (taken < k) & np.isfinite(dist[:, c])
        w = 1.0 / (1.0 + dist[ok, c])
        hist[rows[ok], known[j[ok]]] += w
        cnt[ok] += 1
        dsum[ok] += dist[ok, c]
        taken[ok] += 1
    tot = hist.sum(1, keepdims=True)
    hist = hist / np.maximum(tot, 1e-9)
    out = np.hstack([hist, cnt[:, None], (dsum / np.maximum(cnt, 1))[:, None]]).astype(np.float32)
    names = [f"{tag}_h{c}" for c in range(N_CLASSES)] + [f"{tag}_n", f"{tag}_d"]
    return out, names


def build_X(D, known, sp_k=15, ex_k=25):
    """Static features + fold-aware neighbor-label histograms."""
    parts, names = [D["X"]], list(D["names"])
    if sp_k:
        h, nm = label_hist(known, D["sp_idx"], D["sp_dist"], sp_k, "sp")
        parts.append(h); names += nm
    if ex_k:
        h, nm = label_hist(known, D["ex_idx"], D["ex_dist"], ex_k, "ex")
        parts.append(h); names += nm
    return np.hstack(parts), names


def apply_ei(probs, ei_known, ei_of_label):
    """Zero out classes inconsistent with observed E/I; renormalize.

    Rows the constraint would zero out entirely keep their original probs.
    """
    out = probs.copy()
    for v in (0, 1):
        rows = np.where(ei_known == v)[0]
        if not len(rows):
            continue
        bad_cols = np.where(ei_of_label != v)[0]
        sub = out[rows].copy()
        sub[:, bad_cols] = 0
        z = sub.sum(1)
        alive = z > 0
        out[rows[alive]] = sub[alive] / z[alive, None]
    return out


def run_cv(D, fit_predict, sp_k=15, ex_k=25, verbose=True):
    """5-fold OOF + full-train test prediction.

    fit_predict(Xtr, ytr, Xva, names) -> prob matrix (len(va), 60).
    Returns dict with oof probs (train rows), test probs, acc, acc_ei.
    """
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    tr = np.where(is_tr)[0]
    te = np.where(~is_tr)[0]
    oof = np.zeros((len(D["y"]), N_CLASSES), np.float32)
    for f in range(5):
        known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
        X, names = build_X(D, known, sp_k, ex_k)
        fit = tr[folds[tr] != f]
        va = tr[folds[tr] == f]
        oof[va] = fit_predict(X[fit], y[fit], X[va], names)
        if verbose:
            print(f"  fold{f}: {(oof[va].argmax(1) == y[va]).mean():.4f}", flush=True)
    acc = (oof[tr].argmax(1) == y[tr]).mean()
    oof_ei = apply_ei(oof[tr], D["ei_known"][tr], D["ei_of_label"])
    acc_ei = (oof_ei.argmax(1) == y[tr]).mean()
    # final: all train labels known
    known = np.where(is_tr, y, -1).astype(np.int64)
    X, names = build_X(D, known, sp_k, ex_k)
    test_probs = fit_predict(X[tr], y[tr], X[te], names)
    if verbose:
        print(f"  OOF acc={acc:.4f}  +EI={acc_ei:.4f}")
    return {"oof": oof[tr], "test": test_probs, "acc": acc, "acc_ei": acc_ei}


def save_oof(name, res):
    OOF.mkdir(exist_ok=True)
    np.savez_compressed(OOF / f"{name}.npz", oof=res["oof"], test=res["test"],
                        acc=res["acc"], acc_ei=res["acc_ei"])
    print(f"saved oof/{name}.npz  acc={res['acc']:.4f} +EI={res['acc_ei']:.4f}")
