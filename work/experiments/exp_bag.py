"""Seed bagging for a stronger single member (variance reduction).

Runs the EXACT baseline.py config (sp_k=15, ex_k=25, 400 rounds) as full
5-fold run_cv under 6 different LGBM seed triples (seed, bagging_seed,
feature_fraction_seed), plus 2 jittered runs at (sp_k=15, ex_k=50) with
2 seeds. Averages probabilities:

  oof/bag6.npz    = mean of the 6 baseline-config seed runs
  oof/bag8mix.npz = mean of all 8 runs (6 ex25 + 2 ex50)

Folds are fixed (from cache); only LGBM RNG (and ex_k for the jitter set)
varies, so the protocol is leakage-free per common.run_cv. Each run is
checkpointed to experiments/bag_runs/<tag>.npz so the script resumes after
interruption.

Usage: python exp_bag.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load, run_cv, save_oof, apply_ei  # noqa: E402

D = load()
RUNS_DIR = Path(__file__).resolve().parent / "bag_runs"
RUNS_DIR.mkdir(exist_ok=True)

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


# (tag, seed, ex_k): 6 baseline-config seeds + 2 ex50 jitter seeds
RUNS = [(f"s{s}_ex25", s, 25) for s in range(6)] + \
       [(f"s{s}_ex50", s, 50) for s in (0, 1)]


def run_one(tag, seed, ex_k):
    ckpt = RUNS_DIR / f"{tag}.npz"
    if ckpt.exists():
        z = np.load(ckpt)
        res = {k: z[k] for k in z.files}
        print(f"[cached] {tag}: acc={float(res['acc']):.4f} "
              f"+EI={float(res['acc_ei']):.4f}", flush=True)
        return res
    t0 = time.time()
    res = run_cv(D, make_fp(seed), sp_k=15, ex_k=ex_k, verbose=False)
    np.savez_compressed(ckpt, oof=res["oof"], test=res["test"],
                        acc=res["acc"], acc_ei=res["acc_ei"])
    print(f"{tag}: acc={res['acc']:.4f} +EI={res['acc_ei']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return res


def bag(results, name):
    tr = np.where(D["is_train"])[0]
    y = D["y"][tr]
    avg_oof = np.mean([r["oof"] for r in results], axis=0).astype(np.float32)
    avg_test = np.mean([r["test"] for r in results], axis=0).astype(np.float32)
    acc = float((avg_oof.argmax(1) == y).mean())
    oof_ei = apply_ei(avg_oof, D["ei_known"][tr], D["ei_of_label"])
    acc_ei = float((oof_ei.argmax(1) == y).mean())
    res = {"oof": avg_oof, "test": avg_test, "acc": acc, "acc_ei": acc_ei}
    print(f"BAG {name} ({len(results)} runs): acc={acc:.4f} +EI={acc_ei:.4f}",
          flush=True)
    save_oof(name, res)
    return res


def main():
    all_res = [run_one(*r) for r in RUNS]
    bag(all_res[:6], "bag6")
    bag(all_res, "bag8mix")


if __name__ == "__main__":
    main()
