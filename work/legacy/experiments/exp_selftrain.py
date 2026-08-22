"""Transductive self-training / pseudo-labeling on top of the LGBM baseline.

Round 1: for fold f, train LGBM WITHOUT fold-f labels (exactly like run_cv),
predict all unknown cells (fold-f val + all test cells). Confident predictions
(after applying the observed-E/I constraint, which is a legit test-time feature)
become pseudo-labels that enrich the 'known' set feeding the neighbor-label
histogram features, and optionally are added as extra weighted training rows.
Round 2 retrains on the enriched features and predicts fold-f val.

Leakage protocol: fold-f true labels are never visible to either round —
pseudo-labels for fold-f cells come only from the fold-f-blind round-1 model.
Final test prediction mirrors this with all train labels known (f=None).

Usage:
  python exp_selftrain.py screen                 # fold-0 config screening
  python exp_selftrain.py full THRESH MODE ROUNDS  # e.g. full 0.9 feats 1
    MODE: feats (pseudo-labels feed features only) | rows (also extra train rows)
Saves oof/selftrain.npz in full mode.
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load, build_X, apply_ei, save_oof, N_CLASSES  # noqa: E402

D = load()
Y = D["y"].astype(np.int64)
FOLDS = D["folds"]
IS_TR = D["is_train"]
TR = np.where(IS_TR)[0]
TE = np.where(~IS_TR)[0]

PARAMS = dict(
    objective="multiclass", num_class=60, learning_rate=0.06,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbosity=-1, num_threads=4, seed=0,
)


def fit_predict(Xtr, ytr, Xva, names, w=None):
    ds = lgb.Dataset(Xtr, ytr, weight=w,
                     feature_name=[n.replace(" ", "_") for n in names])
    m = lgb.train(PARAMS, ds, num_boost_round=400)
    return m.predict(Xva)


def fold_setup(f):
    """known0 (labels visible in round 1), fit rows, val rows, unknown rows."""
    if f is None:  # final test prediction: all train labels known
        known0 = np.where(IS_TR, Y, -1).astype(np.int64)
        fit, va, unknown = TR, TE, TE
    else:
        known0 = np.where(IS_TR & (FOLDS != f), Y, -1).astype(np.int64)
        fit = TR[FOLDS[TR] != f]
        va = TR[FOLDS[TR] == f]
        unknown = np.concatenate([va, TE])  # va first, then test
    return known0, fit, va, unknown


def round1(f, sp_k=15, ex_k=25):
    """Baseline-equivalent model; returns probs for the unknown cells."""
    known0, fit, va, unknown = fold_setup(f)
    X, names = build_X(D, known0, sp_k, ex_k)
    return fit_predict(X[fit], Y[fit], X[unknown], names)


def self_train(f, p_unk, thresh, extra_rows, rounds, sp_k=15, ex_k=25):
    """Run `rounds` pseudo-label rounds starting from round-1 probs p_unk.

    Returns (val probs, diagnostics dict from the last round).
    """
    known0, fit, va, unknown = fold_setup(f)
    p = p_unk
    diag = {}
    for _ in range(rounds):
        pe = apply_ei(p, D["ei_known"][unknown], D["ei_of_label"])
        conf = pe.max(1) >= thresh
        pl = pe.argmax(1).astype(np.int64)
        known = known0.copy()
        known[unknown[conf]] = pl[conf]
        X2, names = build_X(D, known, sp_k, ex_k)
        if extra_rows:
            Xfit = np.vstack([X2[fit], X2[unknown[conf]]])
            yfit = np.concatenate([Y[fit], pl[conf]])
            w = np.concatenate([np.ones(len(fit), np.float64),
                                pe.max(1)[conf]])
        else:
            Xfit, yfit, w = X2[fit], Y[fit], None
        p = fit_predict(Xfit, yfit, X2[unknown], names, w)
        # diagnostics (evaluation only; never fed back into training)
        nva = len(va)
        va_conf = conf[:nva] if f is not None else conf
        diag = {"n_pseudo": int(conf.sum()), "frac_pseudo": float(conf.mean())}
        if f is not None and va_conf.sum():
            diag["pseudo_acc_va"] = float(
                (pl[:nva][va_conf] == Y[va][va_conf]).mean())
    return p[:len(va)], diag


def eval_va(pva, va):
    acc = float((pva.argmax(1) == Y[va]).mean())
    pei = apply_ei(pva, D["ei_known"][va], D["ei_of_label"])
    acc_ei = float((pei.argmax(1) == Y[va]).mean())
    return acc, acc_ei


def screen():
    f = 0
    t0 = time.time()
    p1 = round1(f)
    _, _, va, _ = fold_setup(f)
    acc1, acc1_ei = eval_va(p1[:len(va)], va)
    print(f"[{time.time()-t0:5.0f}s] fold0 round1 (baseline): "
          f"acc={acc1:.4f} +EI={acc1_ei:.4f}", flush=True)

    configs = [(0.95, False, 1), (0.90, False, 1), (0.80, False, 1),
               (0.90, True, 1), (0.80, True, 1)]
    results = []
    for thresh, extra, rounds in configs:
        pva, diag = self_train(f, p1, thresh, extra, rounds)
        acc, acc_ei = eval_va(pva, va)
        results.append((acc, acc_ei, thresh, extra, rounds, diag))
        print(f"[{time.time()-t0:5.0f}s] thr={thresh:.2f} "
              f"extra_rows={extra} rounds={rounds}: acc={acc:.4f} "
              f"+EI={acc_ei:.4f}  {diag}", flush=True)
    # extra round on the best 1-round config
    best = max(results, key=lambda r: r[0])
    thresh, extra = best[2], best[3]
    pva, diag = self_train(f, p1, thresh, extra, 2)
    acc, acc_ei = eval_va(pva, va)
    print(f"[{time.time()-t0:5.0f}s] thr={thresh:.2f} extra_rows={extra} "
          f"rounds=2: acc={acc:.4f} +EI={acc_ei:.4f}  {diag}", flush=True)


def full(thresh, extra_rows, rounds):
    t0 = time.time()
    n = len(Y)
    oof = np.zeros((n, N_CLASSES), np.float32)
    for f in range(5):
        p1 = round1(f)
        _, _, va, _ = fold_setup(f)
        pva, diag = self_train(f, p1, thresh, extra_rows, rounds)
        oof[va] = pva
        acc_f = (pva.argmax(1) == Y[va]).mean()
        print(f"[{time.time()-t0:5.0f}s] fold{f}: {acc_f:.4f}  {diag}",
              flush=True)
    acc = float((oof[TR].argmax(1) == Y[TR]).mean())
    oof_ei = apply_ei(oof[TR], D["ei_known"][TR], D["ei_of_label"])
    acc_ei = float((oof_ei.argmax(1) == Y[TR]).mean())
    print(f"OOF acc={acc:.4f}  +EI={acc_ei:.4f}", flush=True)
    # final test prediction (all train labels known in round 1)
    p1 = round1(None)
    ptest, diag = self_train(None, p1, thresh, extra_rows, rounds)
    print(f"[{time.time()-t0:5.0f}s] test done  {diag}", flush=True)
    save_oof("selftrain", {"oof": oof[TR], "test": ptest,
                           "acc": acc, "acc_ei": acc_ei})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if cmd == "screen":
        screen()
    else:
        thresh = float(sys.argv[2])
        extra = sys.argv[3] == "rows"
        rounds = int(sys.argv[4])
        full(thresh, extra, rounds)
