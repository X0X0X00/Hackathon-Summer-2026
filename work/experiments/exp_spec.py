"""Confusion-pair specialist correction (post-processor, NOT an ensemble member).

Idea: the ensemble concentrates its errors in a few confusable class pairs.
For each top confused pair (a, b) train a per-fold binary LGBM specialist on
just the train cells of those two classes (leakage-free: specialist for fold f
is fit on pair rows with folds != f, features from common.build_X with fold-f
labels masked -- the exact X run_cv uses). At inference, cells whose (post-EI)
top-2 classes are exactly {a, b} and whose margin p1-p2 < a per-pair threshold
(tuned on folds 0+1) get their pair probabilities SWAPPED when the specialist
disagrees with the current argmax. Other classes' probs untouched.

Application (if saved): experiments/specialist_correction.npz contains
  oof  (5000, 60)  corrected POST-EI ensemble OOF probs (train rows, D order)
  test (5000, 60)  corrected POST-EI ensemble test probs
  pairs, thresholds, labels for bookkeeping.
The test probs are directly usable by make_submission.py (its apply_ei call is
idempotent on already-EI-applied probs and cannot change the argmax).

Usage:
  python experiments/exp_spec.py pairs        # confusion analysis only (fast)
  python experiments/exp_spec.py run          # full pipeline (~2-4 min)
  python experiments/exp_spec.py variant      # confidence-gate variant (uses cache)

Specialist probs are cached in experiments/spec_cache.npz (keyed by pair list);
delete it to force retraining.
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load, build_X, apply_ei, N_CLASSES, WORK  # noqa: E402

N_JOBS = 4
N_PAIRS = 5          # specialists for the top-N confused pairs
GRID = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.55, 0.75, 1.01]
SPEC_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=15,
    min_data_in_leaf=10, feature_fraction=0.6, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=1.0, max_bin=127, verbosity=-1,
    num_threads=N_JOBS, seed=0,
)
SPEC_ROUNDS = 300


def load_probs(D, tr):
    """Load ensemble + base OOF/test probs, verify shape/order, return post-EI."""
    ens = np.load(WORK / "final_probs.npz", allow_pickle=True)
    base = np.load(WORK / "oof" / "lgbm_base.npz")
    y_tr = D["y"][tr]
    n_te = int((~D["is_train"]).sum())
    assert ens["oof"].shape == (len(tr), N_CLASSES), ens["oof"].shape
    assert ens["test"].shape == (n_te, N_CLASSES), ens["test"].shape
    assert base["oof"].shape == (len(tr), N_CLASSES)
    # row-order check: stored acc of lgbm_base must match a recomputation in D order
    acc_re = (base["oof"].argmax(1) == y_tr).mean()
    assert abs(acc_re - float(base["acc"])) < 1e-6, (acc_re, float(base["acc"]))
    ei_tr, ei_te = D["ei_known"][tr], D["ei_known"][~D["is_train"]]
    P_ens = apply_ei(ens["oof"].astype(np.float64), ei_tr, D["ei_of_label"])
    P_ens_te = apply_ei(ens["test"].astype(np.float64), ei_te, D["ei_of_label"])
    P_base = apply_ei(base["oof"].astype(np.float64), ei_tr, D["ei_of_label"])
    print(f"ensemble OOF acc raw={(ens['oof'].argmax(1) == y_tr).mean():.4f}  "
          f"+EI={(P_ens.argmax(1) == y_tr).mean():.4f}")
    print(f"lgbm_base OOF acc raw={acc_re:.4f}  +EI={(P_base.argmax(1) == y_tr).mean():.4f}")
    return P_ens, P_ens_te, P_base


def top_pairs(P, y_tr, labels, k=10):
    """Symmetric confusion pairs sorted by off-diagonal mass (error counts)."""
    pred = P.argmax(1)
    C = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    np.add.at(C, (y_tr, pred), 1)
    pairs = []
    for a in range(N_CLASSES):
        for b in range(a + 1, N_CLASSES):
            m = C[a, b] + C[b, a]
            if m:
                pairs.append((m, a, b, C[a, b], C[b, a]))
    pairs.sort(reverse=True)
    print(f"\ntop {k} confused pairs (errors a->b + b->a):")
    for m, a, b, ab, ba in pairs[:k]:
        print(f"  {m:3d}  ({a:2d}) {labels[a]:35s} <-> ({b:2d}) {labels[b]:35s} "
              f"[{ab}+{ba}]")
    return [(a, b) for _, a, b, _, _ in pairs]


def top2_margin(P):
    part = np.argpartition(P, -2, axis=1)[:, -2:]
    rows = np.arange(len(P))
    v = P[rows[:, None], part]
    swap = v[:, 0] > v[:, 1]
    t1 = np.where(swap, part[:, 0], part[:, 1])
    t2 = np.where(swap, part[:, 1], part[:, 0])
    return t1, t2, P[rows, t1] - P[rows, t2]


def cand_mask(t1, t2, margin, a, b, thr):
    return (((t1 == a) & (t2 == b)) | ((t1 == b) & (t2 == a))) & (margin < thr)


def apply_correction(P, spec, pairs, thrs, confs=None):
    """Swap pair probs where the specialist disagrees (and is confident).
    confs: per-pair min specialist prob for the winning side (default 0.5).
    Returns corrected copy + override count."""
    out = P.copy()
    t1, t2, margin = top2_margin(P)
    n_override = 0
    if confs is None:
        confs = [0.5] * len(pairs)
    for (a, b), thr, conf in zip(pairs, thrs, confs):
        if thr is None:
            continue
        cand = cand_mask(t1, t2, margin, a, b, thr)
        rows = np.where(cand)[0]
        if not len(rows):
            continue
        pa = spec[(a, b)][rows]
        verdict = np.where(pa >= 0.5, a, b)
        confident = np.maximum(pa, 1 - pa) >= conf
        flip = rows[(verdict != t1[rows]) & confident]
        out[flip, a], out[flip, b] = P[flip, b], P[flip, a]
        n_override += len(flip)
    return out, n_override


def train_specialists(D, tr, te, pairs):
    """Leakage-free per-fold specialist probs P(class=a) for every train row,
    plus full-train specialists for test rows. Cached across runs."""
    cache_f = WORK / "experiments" / "spec_cache.npz"
    key = np.array(pairs, np.int32)
    if cache_f.exists():
        z = np.load(cache_f)
        if z["pairs"].shape == key.shape and (z["pairs"] == key).all():
            print("  (loaded specialists from spec_cache.npz)")
            spec_oof = {p: z[f"oof_{p[0]}_{p[1]}"] for p in pairs}
            spec_test = {p: z[f"test_{p[0]}_{p[1]}"] for p in pairs}
            return spec_oof, spec_test
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    spec_oof = {p: np.full(len(tr), np.nan) for p in pairs}
    spec_test = {p: np.full(len(te), np.nan) for p in pairs}
    for f in range(5):
        t0 = time.time()
        known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
        X, _ = build_X(D, known)
        va_local = np.where(folds[tr] == f)[0]          # positions within tr
        for a, b in pairs:
            fit = tr[(folds[tr] != f) & ((y[tr] == a) | (y[tr] == b))]
            ds = lgb.Dataset(X[fit], (y[fit] == a).astype(np.int8))
            m = lgb.train(SPEC_PARAMS, ds, num_boost_round=SPEC_ROUNDS)
            spec_oof[(a, b)][va_local] = m.predict(X[tr[va_local]])
        print(f"  fold{f} specialists done ({time.time()-t0:.0f}s)", flush=True)
    # final: all train labels known -> predict test rows
    known = np.where(is_tr, y, -1).astype(np.int64)
    X, _ = build_X(D, known)
    for a, b in pairs:
        fit = tr[(y[tr] == a) | (y[tr] == b)]
        ds = lgb.Dataset(X[fit], (y[fit] == a).astype(np.int8))
        m = lgb.train(SPEC_PARAMS, ds, num_boost_round=SPEC_ROUNDS)
        spec_test[(a, b)] = m.predict(X[te])
    print("  test specialists done", flush=True)
    np.savez_compressed(cache_f, pairs=key,
                        **{f"oof_{a}_{b}": spec_oof[(a, b)] for a, b in pairs},
                        **{f"test_{a}_{b}": spec_test[(a, b)] for a, b in pairs})
    return spec_oof, spec_test


CONF_GRID = [0.5, 0.55, 0.6, 0.65, 0.7]


def tune_thresholds(P, spec, pairs, y_tr, fold_of, tune_folds=(0, 1),
                    conf_grid=(0.5,)):
    """Per-pair (margin thr, spec confidence) maximizing net error reduction on
    tune folds. Returns (thrs, confs); thr None = pair dropped."""
    t1, t2, margin = top2_margin(P)
    in_tune = np.isin(fold_of, tune_folds)
    thrs, confs = [], []
    for a, b in pairs:
        best_net, best_thr, best_conf = 0, None, 0.5
        for thr in GRID:
            cand = cand_mask(t1, t2, margin, a, b, thr) & in_tune
            rows = np.where(cand)[0]
            if not len(rows):
                continue
            pa = spec[(a, b)][rows]
            verdict = np.where(pa >= 0.5, a, b)
            for conf in conf_grid:
                use = np.maximum(pa, 1 - pa) >= conf
                # non-confident rows keep the current argmax
                after = np.where(use, verdict == y_tr[rows], t1[rows] == y_tr[rows]).sum()
                net = after - (t1[rows] == y_tr[rows]).sum()
                if net > best_net:
                    best_net, best_thr, best_conf = net, thr, conf
        thrs.append(best_thr)
        confs.append(best_conf)
        print(f"  pair ({a},{b}): thr={best_thr} conf={best_conf} "
              f"net={best_net:+d} on folds {tune_folds}")
    return thrs, confs


def pair_error_report(P_before, P_after, pairs, y_tr, labels):
    for a, b in pairs:
        m = (y_tr == a) | (y_tr == b)
        e0 = (P_before[m].argmax(1) != y_tr[m]).sum()
        e1 = (P_after[m].argmax(1) != y_tr[m]).sum()
        print(f"  ({a:2d},{b:2d}) {labels[a][:26]:26s}/{labels[b][:26]:26s} "
              f"errors {e0:3d} -> {e1:3d} ({e1-e0:+d})")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    D = load()
    tr = np.where(D["is_train"])[0]
    te = np.where(~D["is_train"])[0]
    y_tr = D["y"][tr]
    fold_of = D["folds"][tr]
    labels = D["labels"]
    P_ens, P_ens_te, P_base = load_probs(D, tr)

    all_pairs = top_pairs(P_ens, y_tr, labels, k=12)
    if cmd == "pairs":
        print("\n(base model pairs for comparison)")
        top_pairs(P_base, y_tr, labels, k=12)
        return
    pairs = all_pairs[:N_PAIRS]
    print(f"\ntraining specialists for top {N_PAIRS} pairs: {pairs}", flush=True)
    spec_oof, spec_test = train_specialists(D, tr, te, pairs)
    conf_grid = CONF_GRID if cmd == "variant" else (0.5,)

    print(f"\n--- tuning on folds 0+1 (ensemble), conf_grid={conf_grid} ---")
    thrs_ens, confs_ens = tune_thresholds(P_ens, spec_oof, pairs, y_tr, fold_of,
                                          conf_grid=conf_grid)
    print(f"--- tuning on folds 0+1 (lgbm_base), conf_grid={conf_grid} ---")
    thrs_base, confs_base = tune_thresholds(P_base, spec_oof, pairs, y_tr, fold_of,
                                            conf_grid=conf_grid)

    print("\n=== full-OOF evaluation: ENSEMBLE ===")
    acc0 = (P_ens.argmax(1) == y_tr).mean()
    C_ens, n_ov = apply_correction(P_ens, spec_oof, pairs, thrs_ens, confs_ens)
    acc1 = (C_ens.argmax(1) == y_tr).mean()
    pair_error_report(P_ens, C_ens, pairs, y_tr, labels)
    print(f"ensemble +EI OOF acc {acc0:.4f} -> {acc1:.4f}  "
          f"delta={acc1-acc0:+.4f}  overrides={n_ov}")
    for f in range(5):
        m = fold_of == f
        d = (C_ens[m].argmax(1) == y_tr[m]).sum() - (P_ens[m].argmax(1) == y_tr[m]).sum()
        print(f"  fold{f}: {d:+d} cells", end="")
    print()
    hold = ~np.isin(fold_of, (0, 1))
    h0 = (P_ens[hold].argmax(1) == y_tr[hold]).mean()
    h1 = (C_ens[hold].argmax(1) == y_tr[hold]).mean()
    print(f"holdout folds 2-4 only: {h0:.4f} -> {h1:.4f}  delta={h1-h0:+.4f}")

    print("\n=== full-OOF evaluation: LGBM_BASE ===")
    b0 = (P_base.argmax(1) == y_tr).mean()
    C_base, n_ovb = apply_correction(P_base, spec_oof, pairs, thrs_base, confs_base)
    b1 = (C_base.argmax(1) == y_tr).mean()
    pair_error_report(P_base, C_base, pairs, y_tr, labels)
    print(f"lgbm_base +EI OOF acc {b0:.4f} -> {b1:.4f}  "
          f"delta={b1-b0:+.4f}  overrides={n_ovb}")
    bh0 = (P_base[hold].argmax(1) == y_tr[hold]).mean()
    bh1 = (C_base[hold].argmax(1) == y_tr[hold]).mean()
    print(f"holdout folds 2-4 only: {bh0:.4f} -> {bh1:.4f}  delta={bh1-bh0:+.4f}")

    if cmd == "run" and acc1 > acc0:
        C_te, n_ovt = apply_correction(P_ens_te, spec_test, pairs, thrs_ens, confs_ens)
        out = WORK / "experiments" / "specialist_correction.npz"
        np.savez_compressed(
            out, oof=C_ens.astype(np.float32), test=C_te.astype(np.float32),
            pairs=np.array(pairs, np.int32),
            thresholds=np.array([np.nan if t is None else t for t in thrs_ens]),
            confidences=np.array(confs_ens), labels=labels,
            note="post-EI corrected ensemble probs; feed test to make_submission.py")
        print(f"\nsaved {out}  (test overrides={n_ovt})")
    elif cmd == "run":
        print("\nnet not positive on full OOF -> nothing saved")


if __name__ == "__main__":
    main()
