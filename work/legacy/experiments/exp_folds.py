"""Round 3: bagging over DIFFERENT FOLD SCHEMES.

Diversity source = different train subsets + different neighbor-label
visibility masks (each fold scheme hides a different 10%/20% of train labels
when building the sp/ex label histograms), on top of LGBM seed noise.

Runs (baseline params, sp_k=15, ex_k=25, 400 rounds, 4 threads):
  (a) 10-fold StratifiedKFold(shuffle, random_state=11), LGBM seeds 10,11,12
  (b) 5-fold  StratifiedKFold(shuffle, random_state=21), LGBM seeds 20,21
      5-fold  StratifiedKFold(shuffle, random_state=31), LGBM seeds 30,31

run_cv_folds() is a copy of common.run_cv taking an explicit fold array:
predicting fold f => fold-f labels invisible everywhere (known=-1 for those
cells in build_X, and they are not in the fit set). Test prediction = refit
on all train with all train labels visible (identical to common.run_cv).
Every run's OOF is therefore a leakage-free prediction for every train cell,
so OOFs can be averaged across schemes.

Checkpoints: experiments/fold_runs/<tag>.npz (oof, test, acc, acc_ei).
Members saved via common.save_oof:
  fold10bag = mean of the 3 ten-fold runs
  foldmix   = mean of all 7 runs
Report (printed): per-run / member acc, +EI, per-fold on the ORIGINAL
D['folds'], holdout (orig folds 3-4), and (bag8mix + foldmix)/2 vs bag8mix.

Usage: python exp_folds.py            # runs (resumable) + report
       python exp_folds.py --report   # report only from checkpoints
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (load, build_X, save_oof, apply_ei, N_CLASSES,  # noqa: E402
                    OOF)

D = load()
HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "fold_runs"
RUNS_DIR.mkdir(exist_ok=True)
BAG_DIR = HERE / "bag_runs"

TR = np.where(D["is_train"])[0]
TE = np.where(~D["is_train"])[0]
Y_TR = D["y"][TR]
ORIG_F = D["folds"][TR]          # original 5-fold ids for train rows

BASE = dict(
    objective="multiclass", num_class=60, learning_rate=0.06,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbosity=-1, num_threads=4, seed=0,
)


def make_fp(seed):
    p = dict(BASE)
    p["seed"] = seed
    p["bagging_seed"] = seed + 100
    p["feature_fraction_seed"] = seed + 200

    def fit_predict(Xtr, ytr, Xva, names):
        ds = lgb.Dataset(Xtr, ytr, feature_name=[n.replace(" ", "_") for n in names])
        m = lgb.train(p, ds, num_boost_round=400)
        return m.predict(Xva)
    return fit_predict


def make_folds(n_splits, random_state):
    """Fold id per cell (len 10000); -1 for test cells. Stratified on y."""
    folds = np.full(len(D["y"]), -1, np.int64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # rare classes (<n_splits members)
        for f, (_, va) in enumerate(skf.split(np.zeros(len(TR)), Y_TR)):
            folds[TR[va]] = f
    assert (folds[TR] >= 0).all() and (folds[TE] == -1).all()
    return folds


def run_cv_folds(D, fit_predict, folds, n_folds, sp_k=15, ex_k=25, verbose=True):
    """Copy of common.run_cv with an explicit fold array.

    For fold f: labels of fold-f cells are hidden in build_X (known=-1) and
    fold-f cells are excluded from the fit set. Test: refit on all train
    with all train labels visible.
    """
    y, is_tr = D["y"], D["is_train"]
    tr, te = TR, TE
    oof = np.zeros((len(y), N_CLASSES), np.float32)
    for f in range(n_folds):
        known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
        X, names = build_X(D, known, sp_k, ex_k)
        fit = tr[folds[tr] != f]
        va = tr[folds[tr] == f]
        assert (known[va] == -1).all()
        oof[va] = fit_predict(X[fit], y[fit], X[va], names)
        if verbose:
            print(f"  fold{f}: {(oof[va].argmax(1) == y[va]).mean():.4f}", flush=True)
    acc = (oof[tr].argmax(1) == y[tr]).mean()
    oof_ei = apply_ei(oof[tr], D["ei_known"][tr], D["ei_of_label"])
    acc_ei = (oof_ei.argmax(1) == y[tr]).mean()
    known = np.where(is_tr, y, -1).astype(np.int64)
    X, names = build_X(D, known, sp_k, ex_k)
    test_probs = fit_predict(X[tr], y[tr], X[te], names)
    if verbose:
        print(f"  OOF acc={acc:.4f}  +EI={acc_ei:.4f}")
    return {"oof": oof[tr], "test": test_probs, "acc": acc, "acc_ei": acc_ei}


# (tag, n_splits, fold random_state, lgbm seed)
RUNS = [("f10r11_s10", 10, 11, 10),
        ("f10r11_s11", 10, 11, 11),
        ("f10r11_s12", 10, 11, 12),
        ("f5r21_s20", 5, 21, 20),
        ("f5r21_s21", 5, 21, 21),
        ("f5r31_s30", 5, 31, 30),
        ("f5r31_s31", 5, 31, 31)]

_FOLD_CACHE = {}


def get_folds(n_splits, rs):
    key = (n_splits, rs)
    if key not in _FOLD_CACHE:
        _FOLD_CACHE[key] = make_folds(n_splits, rs)
    return _FOLD_CACHE[key]


def run_one(tag, n_splits, rs, seed):
    ckpt = RUNS_DIR / f"{tag}.npz"
    if ckpt.exists():
        z = np.load(ckpt)
        res = {k: z[k] for k in z.files}
        print(f"[cached] {tag}: acc={float(res['acc']):.4f} "
              f"+EI={float(res['acc_ei']):.4f}", flush=True)
        return res
    t0 = time.time()
    folds = get_folds(n_splits, rs)
    res = run_cv_folds(D, make_fp(seed), folds, n_splits, sp_k=15, ex_k=25,
                       verbose=False)
    np.savez_compressed(ckpt, oof=res["oof"], test=res["test"],
                        acc=res["acc"], acc_ei=res["acc_ei"])
    print(f"{tag}: acc={res['acc']:.4f} +EI={res['acc_ei']:.4f} "
          f"({time.time() - t0:.0f}s)  per-orig-fold {perfold_str(res['oof'])}",
          flush=True)
    return res


# ---------------------------------------------------------------- reporting
def perfold(oof):
    pred = oof.argmax(1)
    return np.array([(pred[ORIG_F == f] == Y_TR[ORIG_F == f]).mean() for f in range(5)])


def perfold_str(oof):
    return " ".join(f"{a:.4f}" for a in perfold(oof))


def stats(oof):
    acc = float((oof.argmax(1) == Y_TR).mean())
    ei = apply_ei(oof, D["ei_known"][TR], D["ei_of_label"])
    acc_ei = float((ei.argmax(1) == Y_TR).mean())
    pf = perfold(oof)
    pf_ei = perfold(ei)
    hold = ORIG_F >= 3
    ho = float((oof.argmax(1)[hold] == Y_TR[hold]).mean())
    ho_ei = float((ei.argmax(1)[hold] == Y_TR[hold]).mean())
    dev = ~hold
    dv = float((oof.argmax(1)[dev] == Y_TR[dev]).mean())
    return dict(acc=acc, acc_ei=acc_ei, pf=pf, pf_ei=pf_ei, ho=ho, ho_ei=ho_ei, dev=dv)


def report_line(name, oof):
    s = stats(oof)
    print(f"{name:<28s} acc={s['acc']:.4f} +EI={s['acc_ei']:.4f} | "
          f"folds {perfold_str(oof)} | dev012={s['dev']:.4f} "
          f"HOLDOUT34={s['ho']:.4f} +EI={s['ho_ei']:.4f}", flush=True)
    return s


def mean_res(results):
    avg_oof = np.mean([r["oof"] for r in results], axis=0).astype(np.float32)
    avg_test = np.mean([r["test"] for r in results], axis=0).astype(np.float32)
    avg_oof /= avg_oof.sum(1, keepdims=True)
    avg_test /= avg_test.sum(1, keepdims=True)
    acc = float((avg_oof.argmax(1) == Y_TR).mean())
    ei = apply_ei(avg_oof, D["ei_known"][TR], D["ei_of_label"])
    acc_ei = float((ei.argmax(1) == Y_TR).mean())
    return {"oof": avg_oof, "test": avg_test, "acc": acc, "acc_ei": acc_ei}


def load_member(name):
    z = np.load(OOF / f"{name}.npz")
    return {k: z[k] for k in z.files}


def report(all_res, save=True):
    print("\n=== per-run (per-fold on ORIGINAL D['folds']) ===")
    for (tag, *_), r in zip(RUNS, all_res):
        report_line(tag, r["oof"])
    n_done = len(all_res)
    ten = [r for (t, *_), r in zip(RUNS, all_res) if t.startswith("f10")]
    print("\n=== members ===")
    fold10bag = mean_res(ten)
    report_line(f"fold10bag ({len(ten)} runs)", fold10bag["oof"])
    foldmix = mean_res(all_res)
    report_line(f"foldmix ({n_done} runs)", foldmix["oof"])
    if save and len(ten) == 3:
        save_oof("fold10bag", fold10bag)
    if save and n_done == len(RUNS):
        save_oof("foldmix", foldmix)

    print("\n=== comparison vs bag8mix (all fixed unweighted, no tuning) ===")
    bag8 = load_member("bag8mix")
    s_b = report_line("bag8mix", bag8["oof"])
    blend = mean_res([bag8, foldmix])
    s_bl = report_line("(bag8mix+foldmix)/2", blend["oof"])
    blend10 = mean_res([bag8, fold10bag])
    report_line("(bag8mix+fold10bag)/2", blend10["oof"])
    # all individual runs pooled (8 bag runs + these)
    bag_runs = sorted(BAG_DIR.glob("*.npz"))
    if bag_runs:
        br = []
        for p in bag_runs:
            z = np.load(p)
            br.append({k: z[k] for k in z.files})
        pooled = mean_res(br + all_res)
        s_p = report_line(f"pool{len(br) + n_done} (all runs mean)", pooled["oof"])
        if save and n_done == len(RUNS):
            save_oof(f"pool{len(br) + n_done}", pooled)
    d = s_bl["pf"] - s_b["pf"]
    print(f"\nblend - bag8mix per fold: {' '.join(f'{x:+.4f}' for x in d)}  "
          f"(folds better: {(d > 0).sum()}, worse: {(d < 0).sum()})")
    print(f"blend - bag8mix: full {s_bl['acc'] - s_b['acc']:+.4f}  "
          f"holdout34 {s_bl['ho'] - s_b['ho']:+.4f}  "
          f"holdout34+EI {s_bl['ho_ei'] - s_b['ho_ei']:+.4f}")
    return dict(fold10bag=fold10bag, foldmix=foldmix, blend=blend)


def main():
    report_only = "--report" in sys.argv
    all_res = []
    for r in RUNS:
        ckpt = RUNS_DIR / f"{r[0]}.npz"
        if report_only and not ckpt.exists():
            continue
        all_res.append(run_one(*r))
    report(all_res, save=not report_only or len(all_res) == len(RUNS))


if __name__ == "__main__":
    main()
