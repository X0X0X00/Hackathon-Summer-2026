"""New feature blocks beyond baseline (screen on fold 0, full CV for best stack).

Blocks (all fold-aware ones recompute from `known` labels only -> leakage-free):
  A  : class-prototype correlation of cell lognorm vs 60 per-class mean profiles
  B  : spatially smoothed PCA50 (k=25 plain mean + k=50 gaussian-weighted) [static]
  C  : per-Section visible-label composition (60 fracs + visible count) [fold-aware]
  Cs : section size + distance to section centroid [static]
  Dr : P(class|Region) priors from visible train labels, NaN where Region missing
  E  : raw-count features: n genes detected, top1/top2 gene fraction + identity [static]

Usage:
  python exp_feats.py screen base A B C+Cs Dr E     # fold-0 screening
  python exp_feats.py full A+C+Cs [name]            # 5-fold CV + save oof/<name>.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

WORK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK))
from common import load, build_X, apply_ei, save_oof, N_CLASSES  # noqa: E402

D = load()
n = len(D["y"])
NG = 200  # lognorm gene columns come first in X
LN = D["X"][:, :NG]
P = D["X"][:, NG:NG + 50]
y, folds, is_tr = D["y"], D["folds"], D["is_train"]
tr = np.where(is_tr)[0]
te = np.where(~is_tr)[0]

meta = pd.concat([
    pd.read_csv(WORK.parent / "data/meta_train.csv", index_col=0),
    pd.read_csv(WORK.parent / "data/meta_test.csv", index_col=0),
])
assert (meta.index.astype(str).values == D["ids"]).all()
sec = pd.Categorical(meta.Section_ID).codes.astype(np.int64)
n_sec = int(sec.max()) + 1
coords = meta[["center_x", "center_y"]].values

PARAMS = dict(
    objective="multiclass", num_class=60, learning_rate=0.06,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbosity=-1, num_threads=4, seed=0,
)


def fit_predict(Xtr, ytr, Xva, nms):
    ds = lgb.Dataset(Xtr, ytr, feature_name=[s.replace(" ", "_") for s in nms])
    m = lgb.train(PARAMS, ds, num_boost_round=400)
    return m.predict(Xva)


# ---------------- static blocks (no labels involved) ----------------
def blk_B():
    """Spatially smoothed PCA50: k=25 mean + gaussian-weighted over 50 nbrs."""
    sp_idx, sp_dist = D["sp_idx"], D["sp_dist"]
    mean25 = np.zeros((n, 50), np.float32)
    cnt = np.zeros(n, np.float32)
    for c in range(25):
        j = sp_idx[:, c]
        ok = (j >= 0) & np.isfinite(sp_dist[:, c])
        mean25[ok] += P[j[ok]]
        cnt[ok] += 1
    mean25 /= np.maximum(cnt, 1)[:, None]
    dm = np.where(np.isfinite(sp_dist[:, :25]), sp_dist[:, :25], np.nan)
    sigma = np.nanmedian(dm, 1)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    g = np.zeros((n, 50), np.float32)
    wsum = np.zeros(n, np.float32)
    for c in range(50):
        j = sp_idx[:, c]
        ok = (j >= 0) & np.isfinite(sp_dist[:, c])
        w = np.exp(-0.5 * (sp_dist[ok, c] / sigma[ok]) ** 2).astype(np.float32)
        g[ok] += w[:, None] * P[j[ok]]
        wsum[ok] += w
    g /= np.maximum(wsum, 1e-9)[:, None]
    arr = np.hstack([mean25, g]).astype(np.float32)
    return arr, [f"spm25_{i}" for i in range(50)] + [f"spg_{i}" for i in range(50)]


def blk_Cs():
    """Section size + distance to section centroid."""
    size = np.bincount(sec, minlength=n_sec).astype(np.float64)
    mx = np.zeros(n_sec)
    my = np.zeros(n_sec)
    np.add.at(mx, sec, coords[:, 0])
    np.add.at(my, sec, coords[:, 1])
    mx /= size
    my /= size
    d = np.sqrt((coords[:, 0] - mx[sec]) ** 2 + (coords[:, 1] - my[sec]) ** 2)
    arr = np.column_stack([np.log(size[sec]), d]).astype(np.float32)
    return arr, ["sec_size", "sec_dcent"]


def blk_E():
    """Raw-count style features derived exactly from lognorm."""
    ndet = (LN > 0).sum(1).astype(np.float32)
    e = np.expm1(LN.astype(np.float64))
    frac = e / np.maximum(e.sum(1, keepdims=True), 1e-9)
    order = np.argsort(-frac, 1)
    top1, top2 = order[:, 0], order[:, 1]
    r = np.arange(n)
    arr = np.column_stack([ndet, frac[r, top1], frac[r, top2],
                           top1.astype(np.float32), top2.astype(np.float32)]).astype(np.float32)
    return arr, ["ndet", "topfrac1", "topfrac2", "topg1", "topg2"]


# ---------------- fold-aware blocks (use `known` only) ----------------
LNc = LN - LN.mean(1, keepdims=True)
LNn = LNc / np.maximum(np.linalg.norm(LNc, axis=1, keepdims=True), 1e-9)


def _proto_corr(known):
    vis = known >= 0
    proto = np.zeros((N_CLASSES, NG), np.float64)
    cntc = np.bincount(known[vis], minlength=N_CLASSES).astype(np.float64)
    np.add.at(proto, known[vis], LN[vis])
    proto /= np.maximum(cntc, 1)[:, None]
    pc_ = proto - proto.mean(1, keepdims=True)
    pn = pc_ / np.maximum(np.linalg.norm(pc_, axis=1, keepdims=True), 1e-9)
    return (LNn @ pn.T).astype(np.float32)


def blk_A(known):
    """Correlation of each cell's lognorm profile to 60 visible-class prototypes."""
    corr = _proto_corr(known)
    arr = np.hstack([corr, corr.max(1)[:, None], corr.argmax(1)[:, None].astype(np.float32)])
    return arr.astype(np.float32), [f"pcor_{c}" for c in range(60)] + ["pcor_max", "pcor_arg"]


def blk_Ac(known):
    """Compact prototype-corr summary: top-3 corr values, top-2 class codes, margin."""
    corr = _proto_corr(known)
    order = np.argsort(-corr, 1)
    r = np.arange(n)
    c1, c2, c3 = corr[r, order[:, 0]], corr[r, order[:, 1]], corr[r, order[:, 2]]
    arr = np.column_stack([c1, c2, c3, c1 - c2,
                           order[:, 0].astype(np.float32), order[:, 1].astype(np.float32)])
    return arr.astype(np.float32), ["pc_c1", "pc_c2", "pc_c3", "pc_marg", "pc_a1", "pc_a2"]


def blk_C(known):
    """Per-section composition of visible labels + visible count."""
    vis = known >= 0
    h = np.zeros((n_sec, N_CLASSES), np.float64)
    np.add.at(h, (sec[vis], known[vis]), 1.0)
    cnt = h.sum(1)
    frac = h / np.maximum(cnt, 1)[:, None]
    arr = np.hstack([frac[sec], np.log1p(cnt)[sec][:, None]]).astype(np.float32)
    return arr, [f"secc_{c}" for c in range(60)] + ["secc_n"]


def blk_Dr(known):
    """P(class|Region) from visible labels with observed Region; NaN if missing."""
    reg = D["region_known"]
    vis = (known >= 0) & np.isfinite(reg)
    r = reg[vis].astype(np.int64)
    nr = int(np.nanmax(reg)) + 1
    h = np.zeros((nr, N_CLASSES), np.float64)
    np.add.at(h, (r, known[vis]), 1.0)
    frac = h / np.maximum(h.sum(1, keepdims=True), 1)
    arr = np.full((n, N_CLASSES), np.nan, np.float32)
    ok = np.isfinite(reg)
    arr[ok] = frac[reg[ok].astype(np.int64)]
    return arr, [f"regp_{c}" for c in range(60)]


STATIC = {"B": blk_B, "Cs": blk_Cs, "E": blk_E}
FOLDAW = {"A": blk_A, "Ac": blk_Ac, "C": blk_C, "Dr": blk_Dr}
_static_cache = {}


def assemble(cfg, known, base_X, base_names, fold_cache=None):
    """cfg like 'A+C+Cs'; 'base' = no extras."""
    parts, nms = [base_X], list(base_names)
    if cfg != "base":
        for b in cfg.split("+"):
            if b in STATIC:
                if b not in _static_cache:
                    _static_cache[b] = STATIC[b]()
                arr, an = _static_cache[b]
            elif b in FOLDAW:
                if fold_cache is not None and b in fold_cache:
                    arr, an = fold_cache[b]
                else:
                    arr, an = FOLDAW[b](known)
                    if fold_cache is not None:
                        fold_cache[b] = (arr, an)
            else:
                raise ValueError(f"unknown block {b}")
            parts.append(arr)
            nms += an
    return np.hstack(parts), nms


def screen(cfgs):
    f = 0
    known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
    Xb, nb = build_X(D, known, 15, 25)
    fit = tr[folds[tr] != f]
    va = tr[folds[tr] == f]
    # reference: saved baseline oof, fold 0
    try:
        base = np.load(WORK / "oof/lgbm_base.npz")
        m0 = folds[tr] == 0
        print(f"saved lgbm_base fold0 acc = {(base['oof'][m0].argmax(1) == y[tr][m0]).mean():.4f}")
    except Exception as ex:
        print("no saved baseline:", ex)
    fold_cache = {}
    for cfg in cfgs:
        t0 = time.time()
        X, nms = assemble(cfg, known, Xb, nb, fold_cache)
        pr = fit_predict(X[fit], y[fit], X[va], nms)
        acc = (pr.argmax(1) == y[va]).mean()
        print(f"fold0  {cfg:<14s} acc={acc:.4f}  nfeat={X.shape[1]}  ({time.time()-t0:.0f}s)", flush=True)


def full(cfg, name):
    oof = np.zeros((n, N_CLASSES), np.float32)
    for f in range(5):
        known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
        Xb, nb = build_X(D, known, 15, 25)
        X, nms = assemble(cfg, known, Xb, nb)
        fit = tr[folds[tr] != f]
        va = tr[folds[tr] == f]
        oof[va] = fit_predict(X[fit], y[fit], X[va], nms)
        print(f"  fold{f}: {(oof[va].argmax(1) == y[va]).mean():.4f}", flush=True)
    acc = (oof[tr].argmax(1) == y[tr]).mean()
    oof_ei = apply_ei(oof[tr], D["ei_known"][tr], D["ei_of_label"])
    acc_ei = (oof_ei.argmax(1) == y[tr]).mean()
    # final: all train labels visible
    known = np.where(is_tr, y, -1).astype(np.int64)
    Xb, nb = build_X(D, known, 15, 25)
    X, nms = assemble(cfg, known, Xb, nb)
    test_probs = fit_predict(X[tr], y[tr], X[te], nms)
    print(f"  OOF acc={acc:.4f}  +EI={acc_ei:.4f}")
    res = {"oof": oof[tr], "test": test_probs, "acc": acc, "acc_ei": acc_ei}
    save_oof(name, res)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "screen":
        screen(sys.argv[2:])
    elif mode == "full":
        full(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "feats_best")
    else:
        raise SystemExit("mode must be screen|full")
