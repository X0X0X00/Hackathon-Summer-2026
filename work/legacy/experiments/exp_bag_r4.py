"""Round 4 (overnight): 20 more LGBM run_cv runs for the final pool, seeds 40..59.

Jitter is drawn from the "known-good" region only (round-3 showed sp10/ff.5/lv95
members are individually ~0.003 weaker):
  sp_k=15, ex_k in {25,35,50}, feature_fraction in {0.6,0.7}, num_leaves in {31,63},
  bagging_fraction in {0.8,0.9}, lr .06, 400 rounds.
  (extra_trees=True was tried for seed 40 and dropped: single-run acc 0.7486, -0.016.)

Checkpoints: experiments/bag_runs/r4_s<seed>.npz (resumable). At the end rebuilds
oof/poolAll.npz = mean of every experiments/bag_runs/*.npz + experiments/fold_runs/*.npz.

Usage: python exp_bag_r4.py [num_threads]
"""
import sys, time, glob, os
from pathlib import Path
import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load, run_cv, apply_ei, OOF  # noqa: E402

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
D = load()
HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "bag_runs"; RUNS_DIR.mkdir(exist_ok=True)

BASE = dict(objective="multiclass", num_class=60, learning_rate=0.06,
            num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
            max_bin=127, verbosity=-1, num_threads=NT, seed=0)
SEEDS = list(range(40, 60))
GRID = dict(ex_k=[25, 35, 50], feature_fraction=[0.6, 0.7], num_leaves=[31, 63],
            bagging_fraction=[0.8, 0.9], extra_trees=[False])
rng = np.random.default_rng(20260818)
CFGS = {s: {k: rng.choice(v).item() for k, v in GRID.items()} for s in SEEDS}


def make_fit(seed, cfg):
    p = dict(BASE); p.update(seed=seed, bagging_seed=seed + 100, feature_fraction_seed=seed + 200,
                             feature_fraction=cfg["feature_fraction"], num_leaves=cfg["num_leaves"],
                             bagging_fraction=cfg["bagging_fraction"], extra_trees=bool(cfg["extra_trees"]))
    def fit_predict(Xtr, ytr, Xva, names):
        m = lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=400)
        return m.predict(Xva)
    return fit_predict


for s in SEEDS:
    out = RUNS_DIR / f"r4_s{s}.npz"
    if out.exists():
        print(f"skip r4_s{s} (exists)"); continue
    cfg = CFGS[s]; t0 = time.time()
    print(f"r4_s{s} {cfg} ...", flush=True)
    res = run_cv(D, make_fit(s, cfg), sp_k=15, ex_k=cfg["ex_k"], verbose=False)
    np.savez_compressed(out, oof=res["oof"], test=res["test"], acc=res["acc"], acc_ei=res["acc_ei"],
                        cfg=str(cfg))
    print(f"r4_s{s}: acc={res['acc']:.4f} +EI={res['acc_ei']:.4f} ({time.time()-t0:.0f}s)", flush=True)

# rebuild the all-runs pool
files = sorted(glob.glob(str(HERE / "bag_runs" / "*.npz"))) + sorted(glob.glob(str(HERE / "fold_runs" / "*.npz")))
oofs, tests = [], []
for f in files:
    z = np.load(f); oofs.append(z["oof"]); tests.append(z["test"])
o = np.mean(oofs, 0); t = np.mean(tests, 0)
o /= o.sum(1, keepdims=True); t /= t.sum(1, keepdims=True)
tr = np.where(D["is_train"])[0]; y = D["y"][tr]
acc = (o.argmax(1) == y).mean()
acc_ei = (apply_ei(o, D["ei_known"][tr], D["ei_of_label"]).argmax(1) == y).mean()
np.savez_compressed(OOF / "poolAll.npz", oof=o, test=t, acc=acc, acc_ei=acc_ei, n_runs=len(files))
print(f"poolAll: {len(files)} runs  acc={acc:.4f} +EI={acc_ei:.4f}  -> oof/poolAll.npz")
