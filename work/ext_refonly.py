"""Reference-only big LGBM: fit on Zenodo reference cells (disjoint from competition),
evaluate honestly on ALL 5000 competition-train cells (no CV needed), predict test.
Neighbour-label features use known = reference + all competition-train labels (self excluded
by kNN construction), i.e. exactly the test-time condition.
Usage: python ext_refonly.py <name> [ref_frac] [threads] [seed] [lr] [leaves] [rounds] [ff]
"""
import sys, time
import numpy as np
import lightgbm as lgb
from common_ext import load_ext, known_labels, build_X_ext, save_oof_ext
from common import apply_ei

name = sys.argv[1]
ref_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 10
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
leaves = int(sys.argv[6]) if len(sys.argv) > 6 else 127
rounds = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
ff = float(sys.argv[8]) if len(sys.argv) > 8 else 0.5
patience = int(sys.argv[9]) if len(sys.argv) > 9 else 100

D = load_ext()
tr = np.where(D["is_train"])[0]; te = np.where(D["is_test"])[0]
ref = np.where(D["is_ref"])[0]
rng = np.random.default_rng(seed)
if ref_frac < 1.0:
    ref = ref[rng.random(len(ref)) < ref_frac]
known = known_labels(D, None)
X, names = build_X_ext(D, known)
y = D["y"]
print(f"fit rows={len(ref)}  eval rows={len(tr)}  features={X.shape[1]}", flush=True)

PARAMS = dict(objective="multiclass", num_class=60, learning_rate=lr, num_leaves=leaves,
              min_data_in_leaf=30, feature_fraction=ff, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, max_bin=127, verbosity=-1, num_threads=threads, seed=seed,
              bagging_seed=seed + 100, feature_fraction_seed=seed + 200)
fn = [n.replace(" ", "_") for n in names]
dtr = lgb.Dataset(X[ref], y[ref], feature_name=fn)
dva = lgb.Dataset(X[tr], y[tr], reference=dtr)
t0 = time.time()
evals = {}
m = lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["comp_train"],
              callbacks=[lgb.log_evaluation(50), lgb.record_evaluation(evals)] +
                        ([lgb.early_stopping(patience, verbose=True)] if patience < 100000 else []))
best_it = m.best_iteration if (patience < 100000 and m.best_iteration) else rounds
print(f"trained in {time.time()-t0:.0f}s  use_iter={best_it}", flush=True)
p_tr = m.predict(X[tr], num_iteration=best_it)
p_te = m.predict(X[te], num_iteration=best_it)
acc = (p_tr.argmax(1) == y[tr]).mean()
acc_ei = (apply_ei(p_tr, D["ei_known"][tr], D["ei_of_label"]).argmax(1) == y[tr]).mean()
# accuracy trajectory every 100 iters (cheap: use staged prediction on train rows)
for it in range(100, best_it + 1, 100):
    a = (m.predict(X[tr], num_iteration=it).argmax(1) == y[tr]).mean()
    print(f"  iter {it:5d}: comp-train acc={a:.4f}", flush=True)
print(f"FINAL comp-train acc={acc:.4f}  +EI={acc_ei:.4f}", flush=True)
save_oof_ext(name, {"oof": p_tr, "test": p_te, "acc": acc, "acc_ei": acc_ei})
imp = sorted(zip(m.feature_importance("gain"), fn), reverse=True)[:15]
print("top gain features:", [(n, int(g)) for g, n in imp])
