"""Cross-section anatomical position features (round 2).

Idea: cells at the same anatomical spot (rel_x, rel_y) in OTHER sections with
the same AP_position share cell-type priors. For each cell, find k nearest
LABELED cells (fold-aware) among train cells in other sections of the same AP
by within-section standardized coords, and add a weighted label histogram
(62 cols, same shape as common.label_hist output).

Neighbor-list variants (precomputed once -> experiments/nbrs_anat.npz):
  an : plain (rel_x, rel_y), all mice, other sections, same AP
  ab : (|rel_x|, rel_y) both sides (bilateral symmetry, orientation-free)
  fl : flip-invariant distance = min(d(q, c), d(q, mirror_x(c)))
  mo : same-mouse-other-section restriction, plain coords
  fm : same-mouse-other-section, flip-invariant distance
  rg : plain coords, but candidates with observed-Region mismatch excluded
       (pairs where both Regions observed and differ -> removed)

Candidates are TRAIN cells only (test cells never carry labels); the query
cell's own section is always excluded, so self-matches are impossible and the
standard fold-aware `known` masking in common.label_hist keeps it leakage-free.

Usage:
  python exp_anat.py build                       # write nbrs_anat.npz
  python exp_anat.py screen an15 ab15 fl15 ...   # folds 0+1 screen (cfg = <var><k>)
  python exp_anat.py full an15 [oof_name]        # 5-fold CV + save oof/<name>.npz
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

WORK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK))
from common import load, build_X, label_hist, apply_ei, save_oof, N_CLASSES  # noqa: E402

CACHE_FILE = Path(__file__).resolve().parent / "nbrs_anat.npz"
LOG_FILE = Path(__file__).resolve().parent / "screen_anat.log"
K_NBR = 50
VARIANTS = ("an", "ab", "fl", "mo", "fm", "rg")

PARAMS = dict(
    objective="multiclass", num_class=60, learning_rate=0.06,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbosity=-1, num_threads=4, seed=0,
)


def fit_predict(Xtr, ytr, Xva, names):
    ds = lgb.Dataset(Xtr, ytr, feature_name=[s.replace(" ", "_") for s in names])
    m = lgb.train(PARAMS, ds, num_boost_round=400)
    return m.predict(Xva)


# ---------------- cache build ----------------

def build_cache(D):
    import pandas as pd
    meta = pd.concat([
        pd.read_csv(WORK.parent / "data/meta_train.csv", index_col=0),
        pd.read_csv(WORK.parent / "data/meta_test.csv", index_col=0),
    ])
    assert (meta.index.astype(str).values == D["ids"]).all()
    n = len(meta)
    sec = pd.Categorical(meta.Section_ID).codes.astype(np.int64)
    ap = meta.AP_position.values.astype(np.int64)
    mouse = pd.Categorical(meta.Mouse_ID).codes.astype(np.int64)
    nm = D["names"]
    rel_x = D["X"][:, nm.index("rel_x")].astype(np.float64)
    rel_y = D["X"][:, nm.index("rel_y")].astype(np.float64)
    region = D["region_known"].astype(np.float64)  # NaN = unobserved
    is_tr = D["is_train"]

    out = {}
    for v in VARIANTS:
        out[f"{v}_idx"] = np.full((n, K_NBR), -1, np.int32)
        out[f"{v}_dist"] = np.full((n, K_NBR), np.inf, np.float32)

    def pair_d2(qx, qy, cx, cy):
        return (qx[:, None] - cx[None, :]) ** 2 + (qy[:, None] - cy[None, :]) ** 2

    for a in np.unique(ap):
        qi = np.where(ap == a)[0]                # all cells at this AP (need feats)
        ci = qi[is_tr[qi]]                       # train candidates at this AP
        same_sec = sec[qi][:, None] == sec[ci][None, :]
        diff_mouse = mouse[qi][:, None] != mouse[ci][None, :]
        rq, rc = region[qi], region[ci]
        reg_mismatch = (np.isfinite(rq)[:, None] & np.isfinite(rc)[None, :]
                        & (rq[:, None] != rc[None, :]))
        d2_plain = pair_d2(rel_x[qi], rel_y[qi], rel_x[ci], rel_y[ci])
        d2_abs = pair_d2(np.abs(rel_x[qi]), rel_y[qi], np.abs(rel_x[ci]), rel_y[ci])
        d2_flip = np.minimum(d2_plain, pair_d2(rel_x[qi], rel_y[qi], -rel_x[ci], rel_y[ci]))
        for v, d2, extra in (("an", d2_plain, None), ("ab", d2_abs, None),
                             ("fl", d2_flip, None), ("mo", d2_plain, diff_mouse),
                             ("fm", d2_flip, diff_mouse), ("rg", d2_plain, reg_mismatch)):
            dm = np.where(same_sec, np.inf, d2)
            if extra is not None:
                dm = np.where(extra, np.inf, dm)
            k = min(K_NBR, dm.shape[1])
            order = np.argsort(dm, axis=1)[:, :k]
            ds = np.sqrt(np.take_along_axis(dm, order, 1))
            valid = np.isfinite(ds)
            out[f"{v}_idx"][qi, :k] = np.where(valid, ci[order], -1)
            out[f"{v}_dist"][qi, :k] = np.where(valid, ds, np.inf).astype(np.float32)
        print(f"AP{a}: queries={len(qi)} candidates={len(ci)}", flush=True)

    np.savez_compressed(CACHE_FILE, **out)
    for v in VARIANTS:
        nlab = (out[f"{v}_idx"] >= 0).sum(1)
        print(f"{v}: valid nbrs per cell min/med/max = "
              f"{nlab.min()}/{int(np.median(nlab))}/{nlab.max()}")
    print(f"saved {CACHE_FILE.name}")


# ---------------- feature assembly ----------------

def parse_cfg(cfg):
    m = re.fullmatch(r"([a-z]+)(\d+)", cfg)
    if not m or m.group(1) not in VARIANTS:
        raise ValueError(f"bad cfg {cfg} (want e.g. an15)")
    return m.group(1), int(m.group(2))


def make_build(cfg):
    nbc = np.load(CACHE_FILE)
    var, k = parse_cfg(cfg)
    a_idx = nbc[f"{var}_idx"]
    a_dist = nbc[f"{var}_dist"]

    def build(D, known):
        X, names = build_X(D, known, 15, 25)  # baseline block set
        h, nm = label_hist(known, a_idx, a_dist, k, f"anat_{var}{k}")
        return np.hstack([X, h]), names + nm
    return build


# ---------------- screen / full ----------------

def screen(D, cfgs, use_folds=(0, 1)):
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    tr = np.where(is_tr)[0]
    for cfg in cfgs:
        t0 = time.time()
        build = make_build(cfg)
        accs = []
        for f in use_folds:
            known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
            fit = tr[folds[tr] != f]
            va = tr[folds[tr] == f]
            X, fn = build(D, known)
            probs = fit_predict(X[fit], y[fit], X[va], fn)
            accs.append((probs.argmax(1) == y[va]).mean())
        detail = " ".join(f"f{f}={a:.4f}" for f, a in zip(use_folds, accs))
        line = (f"SCREEN {cfg:8s} mean={np.mean(accs):.4f}  {detail}  "
                f"nfeat={X.shape[1]}  {time.time()-t0:.0f}s")
        print(line, flush=True)
        with open(LOG_FILE, "a") as fh:
            fh.write(line + "\n")


def full(D, cfg, name):
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    tr = np.where(is_tr)[0]
    te = np.where(~is_tr)[0]
    build = make_build(cfg)
    oof = np.zeros((len(y), N_CLASSES), np.float32)
    for f in range(5):
        known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
        X, names = build(D, known)
        fit = tr[folds[tr] != f]
        va = tr[folds[tr] == f]
        oof[va] = fit_predict(X[fit], y[fit], X[va], names)
        print(f"  fold{f}: {(oof[va].argmax(1) == y[va]).mean():.4f}", flush=True)
    acc = (oof[tr].argmax(1) == y[tr]).mean()
    oof_ei = apply_ei(oof[tr], D["ei_known"][tr], D["ei_of_label"])
    acc_ei = (oof_ei.argmax(1) == y[tr]).mean()
    known = np.where(is_tr, y, -1).astype(np.int64)
    X, names = build(D, known)
    test_probs = fit_predict(X[tr], y[tr], X[te], names)
    print(f"  FULLCV {cfg} acc={acc:.4f}  +EI={acc_ei:.4f}")
    save_oof(name, {"oof": oof[tr], "test": test_probs, "acc": acc, "acc_ei": acc_ei})


if __name__ == "__main__":
    cmd = sys.argv[1]
    D = load()
    if cmd == "build":
        build_cache(D)
    elif cmd == "screen":
        screen(D, sys.argv[2:])
    elif cmd == "full":
        full(D, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "anat_best")
    else:
        raise SystemExit("cmd must be build|screen|full")
