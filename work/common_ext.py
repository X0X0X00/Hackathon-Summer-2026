"""Harness for the EXTENDED universe (competition + Zenodo reference cells).

Leakage protocol (unchanged in spirit):
  predicting competition fold f  =>  labels of fold-f cells are invisible everywhere.
  Reference-cell labels are ALWAYS visible (they are not competition cells).
  Competition TEST labels do not exist in the cache (prep_ext.py never reads them).

fit modes for run_cv_ext:
  'comp'  - LGBM trained on competition-train cells only (folds != f)   [fast]
  'all'   - LGBM trained on reference cells + competition-train (folds != f)  [big]
"""
import numpy as np
from pathlib import Path
from common import label_hist, apply_ei, N_CLASSES

WORK = Path(__file__).resolve().parent
CACHE = WORK / "cache_ext"
OOF = WORK / "oof_ext"


def load_ext():
    s = np.load(CACHE / "static.npz", allow_pickle=True)
    nb = np.load(CACHE / "nbrs.npz")
    D = {k: s[k] for k in s.files}
    D.update({k: nb[k] for k in nb.files})
    D["names"] = list(D["names"])
    return D


def known_labels(D, fold):
    """Visible label vector: reference always; competition-train except `fold` (fold=None -> all train)."""
    y = D["y"]
    vis = D["is_ref"].copy()
    if fold is None:
        vis |= D["is_train"]
    else:
        vis |= D["is_train"] & (D["folds"] != fold)
    return np.where(vis, y, -1).astype(np.int64)


def build_X_ext(D, known, sp_k=15, ex_k=25, extra_blocks=None):
    """Static + neighbour-label histograms (same layout as common.build_X)."""
    blocks = [D["X"]]
    names = list(D["names"])
    h, hn = label_hist(known, D["sp_idx"], D["sp_dist"], sp_k, f"sp{sp_k}")
    blocks.append(h); names += hn
    h, hn = label_hist(known, D["ex_idx"], D["ex_dist"], ex_k, f"ex{ex_k}")
    blocks.append(h); names += hn
    if extra_blocks:
        for b, bn in extra_blocks:
            blocks.append(b); names += bn
    return np.hstack(blocks).astype(np.float32), names


def run_cv_ext(D, fit_predict, mode="comp", sp_k=15, ex_k=25, ref_frac=1.0, seed=0, verbose=True):
    """5-fold OOF on competition train + refit test prediction.

    fit_predict(Xtr, ytr, Xva, names) -> (len(va), 60) probs.
    mode='comp' trains on competition-train rows only; mode='all' adds reference rows
    (optionally subsampled by ref_frac, stratified-ish via random mask with fixed seed).
    """
    y, folds, is_tr, is_te, is_ref = D["y"], D["folds"], D["is_train"], D["is_test"], D["is_ref"]
    tr = np.where(is_tr)[0]; te = np.where(is_te)[0]
    rng = np.random.default_rng(seed)
    ref_rows = np.where(is_ref)[0]
    if ref_frac < 1.0:
        ref_rows = ref_rows[rng.random(len(ref_rows)) < ref_frac]
    oof = np.zeros((len(y), N_CLASSES), np.float32)
    for f in range(5):
        known = known_labels(D, f)
        X, names = build_X_ext(D, known, sp_k, ex_k)
        fit = tr[folds[tr] != f]
        if mode == "all":
            fit = np.concatenate([ref_rows, fit])
        va = tr[folds[tr] == f]
        oof[va] = fit_predict(X[fit], y[fit], X[va], names)
        if verbose:
            print(f"  fold{f}: {(oof[va].argmax(1) == y[va]).mean():.4f}  (n_fit={len(fit)})", flush=True)
    acc = (oof[tr].argmax(1) == y[tr]).mean()
    oof_ei = apply_ei(oof[tr], D["ei_known"][tr], D["ei_of_label"])
    acc_ei = (oof_ei.argmax(1) == y[tr]).mean()
    known = known_labels(D, None)
    X, names = build_X_ext(D, known, sp_k, ex_k)
    fit = np.concatenate([ref_rows, tr]) if mode == "all" else tr
    test_probs = fit_predict(X[fit], y[fit], X[te], names)
    if verbose:
        print(f"  OOF acc={acc:.4f}  +EI={acc_ei:.4f}")
    return {"oof": oof[tr], "test": test_probs, "acc": acc, "acc_ei": acc_ei}


def save_oof_ext(name, res):
    OOF.mkdir(exist_ok=True)
    np.savez_compressed(OOF / f"{name}.npz", oof=res["oof"], test=res["test"], acc=res["acc"], acc_ei=res["acc_ei"])
    print(f"saved oof_ext/{name}.npz  acc={res['acc']:.4f} +EI={res['acc_ei']:.4f}")


def to_comp_order(D, arr_universe_rows_te):
    """Test probs are produced in universe order of is_test rows; competition files expect
    meta_test.csv order. Return reorder index mapping universe-test-order -> meta_test order."""
    import pandas as pd
    mte = pd.read_csv(WORK.parent / "data/meta_test.csv", index_col=0)
    ids_te = D["ids"][D["is_test"]].astype(str)
    pos = {i: k for k, i in enumerate(ids_te)}
    return np.array([pos[str(i)] for i in mte.index], int)
