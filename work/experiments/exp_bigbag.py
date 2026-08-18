"""Round 3: grow the LGBM seed bag with DIVERSE jitter (variance reduction).

12 more full 5-fold run_cv LGBM runs, seeds 6..17. Seed triple like exp_bag.py
(seed / bagging_seed=seed+100 / feature_fraction_seed=seed+200). Each run gets a
jitter drawn once from a fixed RNG (np.random.default_rng(20260817)) and logged:

  sp_k in {10,15,20}, ex_k in {25,35,50}, feature_fraction in {0.5,0.6,0.7},
  num_leaves in {31,63,95}, bagging_fraction in {0.7,0.8,0.9}; lr .06, 400 rounds.

Per-run probs are checkpointed to experiments/bag_runs/r3_s<seed>.npz
(oof, test, acc, acc_ei + the jitter config) so the script resumes after
interruption and the orchestrator can remix members. Then:

  oof/bag12new.npz = mean of the 12 new runs
  oof/bag20.npz    = mean of the 12 new runs + the 8 round-2 runs
                     (experiments/bag_runs/s*_ex*.npz)

Folds are fixed (from cache); protocol is leakage-free per common.run_cv.
Prints per-fold accs of bag20 vs bag8mix (oof/bag8mix.npz) at the end.

Usage: python exp_bigbag.py   (num_threads=4)
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load, run_cv, save_oof, apply_ei, OOF  # noqa: E402

D = load()
RUNS_DIR = Path(__file__).resolve().parent / "bag_runs"
RUNS_DIR.mkdir(exist_ok=True)

BASE = dict(
    objective="multiclass", num_class=60, learning_rate=0.06,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbosity=-1, num_threads=4, seed=0,
)
SEEDS = list(range(6, 18))
GRID = dict(sp_k=[10, 15, 20], ex_k=[25, 35, 50],
            feature_fraction=[0.5, 0.6, 0.7], num_leaves=[31, 63, 95],
            bagging_fraction=[0.7, 0.8, 0.9])


def draw_configs():
    """Fixed jitter per seed: one RNG, drawn in seed order (deterministic)."""
    rng = np.random.default_rng(20260817)
    cfgs = {}
    for s in SEEDS:
        cfgs[s] = {k: rng.choice(v).item() for k, v in GRID.items()}
    return cfgs


CFGS = draw_configs()


def make_fp(seed, cfg):
    p = dict(BASE)
    p["seed"] = seed
    p["bagging_seed"] = seed + 100
    p["feature_fraction_seed"] = seed + 200
    p["feature_fraction"] = cfg["feature_fraction"]
    p["num_leaves"] = cfg["num_leaves"]
    p["bagging_fraction"] = cfg["bagging_fraction"]

    def fit_predict(Xtr, ytr, Xva, names):
        ds = lgb.Dataset(Xtr, ytr, feature_name=[n.replace(" ", "_") for n in names])
        m = lgb.train(p, ds, num_boost_round=400)
        return m.predict(Xva)
    return fit_predict


def cfg_str(cfg):
    return (f"sp_k={cfg['sp_k']} ex_k={cfg['ex_k']} ff={cfg['feature_fraction']} "
            f"leaves={cfg['num_leaves']} bf={cfg['bagging_fraction']}")


def load_ckpt(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def run_one(seed):
    cfg = CFGS[seed]
    tag = f"r3_s{seed}"
    ckpt = RUNS_DIR / f"{tag}.npz"
    if ckpt.exists():
        res = load_ckpt(ckpt)
        print(f"[cached] {tag} [{cfg_str(cfg)}]: acc={float(res['acc']):.4f} "
              f"+EI={float(res['acc_ei']):.4f}", flush=True)
        return res
    print(f"{tag} [{cfg_str(cfg)}] ...", flush=True)
    t0 = time.time()
    res = run_cv(D, make_fp(seed, cfg), sp_k=cfg["sp_k"], ex_k=cfg["ex_k"],
                 verbose=True)
    np.savez_compressed(ckpt, oof=res["oof"], test=res["test"],
                        acc=res["acc"], acc_ei=res["acc_ei"], seed=seed,
                        **{k: v for k, v in cfg.items()})
    print(f"{tag} [{cfg_str(cfg)}]: acc={res['acc']:.4f} +EI={res['acc_ei']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return res


def per_fold(probs_tr):
    tr = np.where(D["is_train"])[0]
    y, f = D["y"][tr], D["folds"][tr]
    pred = probs_tr.argmax(1)
    return [float((pred[f == k] == y[f == k]).mean()) for k in range(5)]


def bag(results, name):
    tr = np.where(D["is_train"])[0]
    y = D["y"][tr]
    avg_oof = np.mean([r["oof"] for r in results], axis=0).astype(np.float32)
    avg_test = np.mean([r["test"] for r in results], axis=0).astype(np.float32)
    avg_oof /= avg_oof.sum(1, keepdims=True)
    avg_test /= avg_test.sum(1, keepdims=True)
    acc = float((avg_oof.argmax(1) == y).mean())
    oof_ei = apply_ei(avg_oof, D["ei_known"][tr], D["ei_of_label"])
    acc_ei = float((oof_ei.argmax(1) == y).mean())
    res = {"oof": avg_oof, "test": avg_test, "acc": acc, "acc_ei": acc_ei}
    print(f"BAG {name} ({len(results)} runs): acc={acc:.4f} +EI={acc_ei:.4f}  "
          f"folds={[round(a, 4) for a in per_fold(avg_oof)]}", flush=True)
    save_oof(name, res)
    return res


def main():
    print("jitter configs (fixed per seed):")
    for s in SEEDS:
        print(f"  seed {s}: {cfg_str(CFGS[s])}")
    new = [run_one(s) for s in SEEDS]
    old_paths = sorted(RUNS_DIR.glob("s*_ex*.npz"))
    old = [load_ckpt(p) for p in old_paths]
    print(f"round-2 members: {[p.name for p in old_paths]}")
    print("per-run accs (new):", {s: round(float(r["acc"]), 4) for s, r in zip(SEEDS, new)})
    b12 = bag(new, "bag12new")
    b20 = bag(new + old, "bag20")
    # compare per fold with bag8mix
    b8 = load_ckpt(OOF / "bag8mix.npz")
    pf8, pf20, pf12 = per_fold(b8["oof"]), per_fold(b20["oof"]), per_fold(b12["oof"])
    print("per-fold acc  bag8mix :", [round(a, 4) for a in pf8], " mean", round(float(np.mean(pf8)), 4))
    print("per-fold acc  bag12new:", [round(a, 4) for a in pf12], " mean", round(float(np.mean(pf12)), 4))
    print("per-fold acc  bag20   :", [round(a, 4) for a in pf20], " mean", round(float(np.mean(pf20)), 4))
    print("bag20 - bag8mix per fold:", [round(a - b, 4) for a, b in zip(pf20, pf8)])
    tr = np.where(D["is_train"])[0]
    y = D["y"][tr]
    ei8 = apply_ei(b8["oof"], D["ei_known"][tr], D["ei_of_label"])
    print(f"bag8mix acc={float((b8['oof'].argmax(1)==y).mean()):.4f} "
          f"+EI={float((ei8.argmax(1)==y).mean()):.4f}")


if __name__ == "__main__":
    main()
