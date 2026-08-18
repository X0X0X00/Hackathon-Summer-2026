"""E1: baseline LGBM trained on competition-train only, but neighbour-label histograms
built from reference + train labels (dense).  Usage: python ext_e1.py [mode] [threads] [name]
mode: comp | all ; for 'all' optional 4th arg ref_frac (default 1.0)."""
import sys, time
import numpy as np
import lightgbm as lgb
from common_ext import load_ext, run_cv_ext, save_oof_ext

mode = sys.argv[1] if len(sys.argv) > 1 else "comp"
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
name = sys.argv[3] if len(sys.argv) > 3 else f"ext_{mode}"
ref_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0

PARAMS = dict(objective="multiclass", num_class=60, learning_rate=0.06, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, max_bin=127, verbosity=-1, num_threads=threads, seed=seed,
              bagging_seed=seed + 100, feature_fraction_seed=seed + 200)
ROUNDS = 400


def fit_predict(Xtr, ytr, Xva, names):
    t = time.time()
    ds = lgb.Dataset(Xtr, ytr, feature_name=[n.replace(" ", "_") for n in names])
    m = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
    print(f"    fit {Xtr.shape} in {time.time()-t:.0f}s", flush=True)
    return m.predict(Xva)


D = load_ext()
res = run_cv_ext(D, fit_predict, mode=mode, ref_frac=ref_frac, seed=seed)
save_oof_ext(name, res)
