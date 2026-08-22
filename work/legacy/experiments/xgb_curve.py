import sys, time
sys.path.insert(0, "/Users/zzh/Visual Studio Code/Hackathon_26/work")
import numpy as np, xgboost as xgb
from common import load, build_X
D = load()
tr = np.where(D['is_train'])[0]; y=D['y']; folds=D['folds']
for f in (0, 1):
    known = np.where(D['is_train'] & (folds!=f), y, -1).astype(np.int64)
    X, names = build_X(D, known)
    fit = tr[folds[tr]!=f]; va = tr[folds[tr]==f]
    dtr = xgb.DMatrix(X[fit], label=y[fit].astype(np.int64), nthread=4); dva = xgb.DMatrix(X[va], nthread=4)
    p = dict(tree_method="hist", objective="multi:softprob", num_class=60, max_depth=6, eta=0.06, subsample=0.8,
             colsample_bytree=0.6, min_child_weight=3, reg_lambda=1.0, nthread=4, verbosity=0, seed=0)
    t0=time.time(); m = xgb.train(p, dtr, num_boost_round=1000)
    out = {}
    for r in (200, 300, 400, 500, 600, 800, 1000):
        pr = m.predict(dva, iteration_range=(0, r)); out[r]=round(float((pr.argmax(1)==y[va]).mean()),4)
    print(f"fold{f} eta.06 d6:", out, f"{time.time()-t0:.0f}s", flush=True)
