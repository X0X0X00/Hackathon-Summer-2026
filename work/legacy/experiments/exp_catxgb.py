"""Round 3: CatBoost + XGBoost members on the baseline features (sp15/ex25).

Modes:
  screen_cat / screen_xgb  -> param variants on folds 0+1 ONLY (per-fold acc, time)
  full                     -> 3 seeds each of the picked config via common.run_cv,
                              checkpointed to experiments/catxgb_runs/<tag>.npz,
                              seed-averages saved as oof/cat_bag3.npz, oof/xgb_bag3.npz
  report                   -> agreement with bag8mix, holdout(3-4) blends

Leakage protocol = common.run_cv (fold-f labels invisible when predicting fold f).
Threads <= 4 everywhere.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load, build_X, run_cv, save_oof, apply_ei, OOF  # noqa: E402

D = load()
RUNS_DIR = Path(__file__).resolve().parent / "catxgb_runs"
RUNS_DIR.mkdir(exist_ok=True)
TR = np.where(D["is_train"])[0]
Y_TR = D["y"][TR]
FOLDS_TR = D["folds"][TR]
NT = 4

# ---------------------------------------------------------------- CatBoost
CAT_BASE = dict(
    loss_function="MultiClass", iterations=800, depth=6, learning_rate=0.08,
    l2_leaf_reg=3, random_strength=1, bootstrap_type="Bernoulli", subsample=0.8,
    rsm=0.6, thread_count=NT, verbose=0, allow_writing_files=False,
    classes_count=60,
)


def make_cat_fp(seed, **over):
    from catboost import CatBoostClassifier
    p = dict(CAT_BASE)
    p.update(over)
    p["random_seed"] = seed

    def fp(Xtr, ytr, Xva, names):
        m = CatBoostClassifier(**p)
        m.fit(Xtr, ytr.astype(np.int64))
        pr = m.predict_proba(Xva)
        cls = np.asarray(m.classes_).astype(int)
        out = np.zeros((len(Xva), 60), np.float64)
        out[:, cls] = pr
        out /= out.sum(1, keepdims=True)
        return out.astype(np.float32)
    return fp


# ---------------------------------------------------------------- XGBoost
XGB_BASE = dict(
    tree_method="hist", objective="multi:softprob", num_class=60,
    max_depth=6, eta=0.06, subsample=0.8, colsample_bytree=0.6,
    min_child_weight=3, reg_lambda=1.0, nthread=NT, verbosity=0,
    max_bin=256,
)
XGB_ROUNDS = 500


def make_xgb_fp(seed, rounds=XGB_ROUNDS, **over):
    import xgboost as xgb
    p = dict(XGB_BASE)
    p.update(over)
    p["seed"] = seed

    def fp(Xtr, ytr, Xva, names):
        dtr = xgb.DMatrix(Xtr, label=ytr.astype(np.int64), nthread=NT)
        dva = xgb.DMatrix(Xva, nthread=NT)
        m = xgb.train(p, dtr, num_boost_round=rounds)
        pr = m.predict(dva)
        pr = pr / pr.sum(1, keepdims=True)
        return pr.astype(np.float32)
    return fp


# ---------------------------------------------------------------- screening
def run_folds(fit_predict, fold_ids, sp_k=15, ex_k=25):
    """Same protocol as common.run_cv but restricted to fold_ids; no test refit."""
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    res = {}
    for f in fold_ids:
        known = np.where(is_tr & (folds != f), y, -1).astype(np.int64)
        X, names = build_X(D, known, sp_k, ex_k)
        fit = TR[folds[TR] != f]
        va = TR[folds[TR] == f]
        t0 = time.time()
        p = fit_predict(X[fit], y[fit], X[va], names)
        acc = float((p.argmax(1) == y[va]).mean())
        res[f] = (acc, time.time() - t0)
        print(f"    fold{f}: {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return res


CAT_VARIANTS = {
    # fold0 screening (see cat_timing*.py / catxgb_screen_cat.log):
    #   symmetric d6 lr.08 800it (default border 254): f0 0.7370, 335s/fit  -> too slow
    #   symmetric d6 lr.15 600it rsm.3 border64:        f0 0.7270,  83s/fit
    #   Depthwise d6 lr.15 600it rsm.3 border64:        f0 0.7400,  84s/fit
    #   Lossguide 63 leaves lr.15 600it rsm.3 border64: f0 0.7470, 122s/fit  <- picked
    "cat_d6_lr08_800": dict(),
    "cat_d8_lr06_800": dict(depth=8, learning_rate=0.06),
    "cat_d6_lr05_1200_l2_5": dict(learning_rate=0.05, iterations=1200, l2_leaf_reg=5),
    "cat_lossguide": dict(learning_rate=0.15, iterations=600, grow_policy="Lossguide",
                          max_leaves=63, depth=10, min_data_in_leaf=10, rsm=0.3,
                          border_count=64),
    "cat_depthwise": dict(learning_rate=0.15, iterations=600, grow_policy="Depthwise",
                          depth=6, rsm=0.3, border_count=64),
}
XGB_VARIANTS = {
    "xgb_d6_eta06_500": dict(),
    "xgb_d8_eta05_500_mcw5": dict(max_depth=8, eta=0.05, min_child_weight=5),
    "xgb_d5_eta06_700_col5": dict(max_depth=5, rounds=700, colsample_bytree=0.5),
}


def screen(lib, only=None):
    variants = CAT_VARIANTS if lib == "cat" else XGB_VARIANTS
    maker = make_cat_fp if lib == "cat" else make_xgb_fp
    for tag, over in variants.items():
        if only and tag not in only:
            continue
        print(f"[{tag}] {over}", flush=True)
        r = run_folds(maker(0, **over), [0, 1])
        m = np.mean([r[f][0] for f in r])
        print(f"  => f0={r[0][0]:.4f} f1={r[1][0]:.4f} mean={m:.4f} "
              f"time/fold={np.mean([r[f][1] for f in r]):.0f}s", flush=True)


# ---------------------------------------------------------------- full runs
def run_one(tag, fp):
    ckpt = RUNS_DIR / f"{tag}.npz"
    if ckpt.exists():
        z = np.load(ckpt)
        res = {k: z[k] for k in z.files}
        print(f"[cached] {tag}: acc={float(res['acc']):.4f} "
              f"+EI={float(res['acc_ei']):.4f}", flush=True)
        return res
    t0 = time.time()
    res = run_cv(D, fp, sp_k=15, ex_k=25, verbose=True)
    np.savez_compressed(ckpt, oof=res["oof"], test=res["test"],
                        acc=res["acc"], acc_ei=res["acc_ei"])
    print(f"{tag}: acc={res['acc']:.4f} +EI={res['acc_ei']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return res


def bag(results, name):
    avg_oof = np.mean([r["oof"] for r in results], axis=0).astype(np.float32)
    avg_test = np.mean([r["test"] for r in results], axis=0).astype(np.float32)
    avg_oof /= avg_oof.sum(1, keepdims=True)
    avg_test /= avg_test.sum(1, keepdims=True)
    acc = float((avg_oof.argmax(1) == Y_TR).mean())
    oof_ei = apply_ei(avg_oof, D["ei_known"][TR], D["ei_of_label"])
    acc_ei = float((oof_ei.argmax(1) == Y_TR).mean())
    res = {"oof": avg_oof, "test": avg_test, "acc": acc, "acc_ei": acc_ei}
    print(f"BAG {name} ({len(results)} runs): acc={acc:.4f} +EI={acc_ei:.4f}",
          flush=True)
    save_oof(name, res)
    return res


def full(lib, cfg_over, seeds=(0, 1, 2), name=None):
    maker = make_cat_fp if lib == "cat" else make_xgb_fp
    results = []
    for s in seeds:
        tag = f"{lib}_s{s}"
        results.append(run_one(tag, maker(s, **cfg_over)))
    return bag(results, name or f"{lib}_bag{len(seeds)}")


# ---------------------------------------------------------------- report
def per_fold(oof):
    pred = oof.argmax(1)
    return {f: float((pred[FOLDS_TR == f] == Y_TR[FOLDS_TR == f]).mean()) for f in range(5)}


def acc_on(oof, mask, ei=False):
    if ei:
        oof = apply_ei(oof, D["ei_known"][TR], D["ei_of_label"])
    return float((oof[mask].argmax(1) == Y_TR[mask]).mean())


def report():
    names = ["bag8mix", "cat_bag3", "xgb_bag3"]
    M = {}
    for n in names:
        p = OOF / f"{n}.npz"
        if p.exists():
            z = np.load(p)
            M[n] = {"oof": z["oof"], "test": z["test"]}
    hold = np.isin(FOLDS_TR, [3, 4])
    dev = ~hold
    allm = np.ones_like(hold)
    print("\n=== per-member ===")
    for n, m in M.items():
        pf = per_fold(m["oof"])
        print(f"{n:10s} full={acc_on(m['oof'], allm):.4f} +EI={acc_on(m['oof'], allm, True):.4f} "
              f"dev(0-2)={acc_on(m['oof'], dev):.4f} hold(3-4)={acc_on(m['oof'], hold):.4f} "
              f"hold+EI={acc_on(m['oof'], hold, True):.4f} folds=" +
              " ".join(f"{pf[f]:.4f}" for f in range(5)))
    if "bag8mix" in M:
        b = M["bag8mix"]["oof"].argmax(1)
        print("\n=== argmax agreement with bag8mix (train OOF, all 5000) ===")
        for n, m in M.items():
            if n == "bag8mix":
                continue
            a = m["oof"].argmax(1)
            print(f"{n:10s} agree={100 * (a == b).mean():.2f}%  "
                  f"agree(hold)={100 * (a[hold] == b[hold]).mean():.2f}%  "
                  f"test-agree={100 * (m['test'].argmax(1) == M['bag8mix']['test'].argmax(1)).mean():.2f}%")
        print("\n=== fixed unweighted blends (no tuning; holdout 3-4 is THE number) ===")
        combos = [("bag8mix",), ("bag8mix", "cat_bag3"), ("bag8mix", "xgb_bag3"),
                  ("bag8mix", "cat_bag3", "xgb_bag3"), ("cat_bag3", "xgb_bag3")]
        for c in combos:
            if not all(n in M for n in c):
                continue
            o = np.mean([M[n]["oof"] for n in c], axis=0)
            pf = per_fold(o)
            print(f"{'+'.join(c):28s} hold(3-4)={acc_on(o, hold):.4f} hold+EI={acc_on(o, hold, True):.4f} | "
                  f"dev(0-2)={acc_on(o, dev):.4f} | full={acc_on(o, allm):.4f} +EI={acc_on(o, allm, True):.4f} | "
                  "folds=" + " ".join(f"{pf[f]:.4f}" for f in range(5)))
        # honest weight tuning: pick weight(s) on dev folds 0-2, report holdout 3-4
        print("\n=== dev-tuned weights (grid on folds 0-2 -> holdout 3-4) ===")
        base = M["bag8mix"]["oof"]
        ws = np.arange(0.0, 0.55, 0.05)
        for n in ("cat_bag3", "xgb_bag3"):
            if n not in M:
                continue
            rows = []
            for w in ws:
                o = (1 - w) * base + w * M[n]["oof"]
                rows.append((w, acc_on(o, dev), acc_on(o, hold), acc_on(o, hold, True)))
            print(f"  bag8mix+{n}: " + " ".join(f"w{r[0]:.2f}:dev{r[1]:.4f}/hold{r[2]:.4f}" for r in rows))
            best = max(rows, key=lambda r: (r[1], -r[0]))
            print(f"    -> best-on-dev w={best[0]:.2f} dev={best[1]:.4f}  HOLD={best[2]:.4f} (+EI {best[3]:.4f})"
                  f"  vs bag8mix hold={acc_on(base, hold):.4f}")
        if "cat_bag3" in M and "xgb_bag3" in M:
            rows = []
            for wc in np.arange(0.0, 0.45, 0.05):
                for wx in np.arange(0.0, 0.45, 0.05):
                    if wc + wx > 0.6:
                        continue
                    o = (1 - wc - wx) * base + wc * M["cat_bag3"]["oof"] + wx * M["xgb_bag3"]["oof"]
                    rows.append((wc, wx, acc_on(o, dev), acc_on(o, hold), acc_on(o, hold, True)))
            best = max(rows, key=lambda r: (r[2], -(r[0] + r[1])))
            print(f"  3-way: best-on-dev wc={best[0]:.2f} wx={best[1]:.2f} dev={best[2]:.4f}  "
                  f"HOLD={best[3]:.4f} (+EI {best[4]:.4f})  vs bag8mix hold={acc_on(base, hold):.4f}")
        # log-space (geometric) blends, fixed weights
        print("\n=== geometric-mean blends (fixed, unweighted) ===")
        for c in combos[1:]:
            if not all(n in M for n in c):
                continue
            o = np.exp(np.mean([np.log(np.clip(M[n]["oof"], 1e-6, 1)) for n in c], axis=0))
            o /= o.sum(1, keepdims=True)
            print(f"{'+'.join(c):28s} hold(3-4)={acc_on(o, hold):.4f} hold+EI={acc_on(o, hold, True):.4f} | "
                  f"dev(0-2)={acc_on(o, dev):.4f} | full={acc_on(o, allm):.4f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "screen_cat":
        screen("cat", sys.argv[2:] or None)
    elif mode == "screen_xgb":
        screen("xgb", sys.argv[2:] or None)
    elif mode == "full_cat":
        # picked config passed as key of CAT_VARIANTS
        full("cat", CAT_VARIANTS[sys.argv[2]], seeds=tuple(int(s) for s in sys.argv[3:]) or (0, 1, 2),
             name="cat_bag3")
    elif mode == "full_xgb":
        full("xgb", XGB_VARIANTS[sys.argv[2]], seeds=tuple(int(s) for s in sys.argv[3:]) or (0, 1, 2),
             name="xgb_bag3")
    elif mode == "run_cat":
        # single seed, checkpoint only (no bag/save) -- lets seeds run in parallel processes
        s = int(sys.argv[3])
        run_one(f"cat_s{s}", make_cat_fp(s, **CAT_VARIANTS[sys.argv[2]]))
    elif mode == "report":
        report()
    else:
        raise SystemExit(f"unknown mode {mode}")
