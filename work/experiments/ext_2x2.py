import sys; sys.path.insert(0, "/Users/zzh/Visual Studio Code/Hackathon_26/work")
import numpy as np, lightgbm as lgb, time
from common_ext import load_ext, known_labels, build_X_ext
D = load_ext(); y = D["y"]; folds = D["folds"]
tr = np.where(D["is_train"])[0]; ref = np.where(D["is_ref"])[0]
rng = np.random.default_rng(0); perm = rng.permutation(ref)
ref_hold = perm[:5000]; ref_fit = perm[5000:5000+34000]
f = 0
known = known_labels(D, f); X, names = build_X_ext(D, known)
fit_c = tr[folds[tr] != f]; va = tr[folds[tr] == f]
P = dict(objective="multiclass", num_class=60, learning_rate=0.06, num_leaves=63, min_data_in_leaf=20,
         feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, verbosity=-1, num_threads=10, seed=0)
fn = [n.replace(" ", "_") for n in names]
def acc(m, rows): return (m.predict(X[rows]).argmax(1) == y[rows]).mean()
for tag, rows in [("comp-train 4k", fit_c), ("ref 34k", ref_fit)]:
    t = time.time(); m = lgb.train(P, lgb.Dataset(X[rows], y[rows], feature_name=fn), 400)
    print(f"model={tag:14s}  acc on comp-val(1000)={acc(m, va):.4f}   acc on ref-holdout(5000)={acc(m, ref_hold):.4f}   ({time.time()-t:.0f}s)", flush=True)
    if tag.startswith("ref"):
        # per-class accuracy gap on comp-val vs ref-hold
        pv = m.predict(X[va]).argmax(1); pr = m.predict(X[ref_hold]).argmax(1)
        labs = D["labels"]
        rows_out = []
        for c in range(60):
            a = (pv[y[va]==c]==c).mean() if (y[va]==c).sum()>=10 else np.nan
            b = (pr[y[ref_hold]==c]==c).mean() if (y[ref_hold]==c).sum()>=10 else np.nan
            if not np.isnan(a) and not np.isnan(b): rows_out.append((b-a, labs[c], a, b, int((y[va]==c).sum())))
        rows_out.sort(reverse=True)
        print("classes where ref-holdout acc >> comp-val acc (gap, class, comp_acc, ref_acc, n_compval):")
        for r in rows_out[:12]: print("   %+.3f  %-32s comp=%.3f ref=%.3f n=%d" % r)
