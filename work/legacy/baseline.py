"""Baseline: LightGBM on static + neighbor-label features."""
import sys
import numpy as np
import lightgbm as lgb
from common import load, run_cv, save_oof

D = load()
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 12

PARAMS = dict(
    objective="multiclass", num_class=60, learning_rate=0.06,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbosity=-1, num_threads=N_JOBS, seed=0,
)


def fit_predict(Xtr, ytr, Xva, names):
    ds = lgb.Dataset(Xtr, ytr, feature_name=[n.replace(" ", "_") for n in names])
    m = lgb.train(PARAMS, ds, num_boost_round=400)
    return m.predict(Xva)


res = run_cv(D, fit_predict)
save_oof("lgbm_base", res)
