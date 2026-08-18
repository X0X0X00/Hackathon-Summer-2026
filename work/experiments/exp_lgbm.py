"""LightGBM param tuning on the EXACT baseline feature set (sp_k=15, ex_k=25).

Usage:
  python exp_lgbm.py screen <stage>   # fold-0 screening (~70s+/config)
  python exp_lgbm.py final <which>    # full 5-fold run_cv + save_oof

Fold-0 screening reuses common.build_X with fold-0 labels masked, trains on
folds 1-4, evaluates on fold 0. Leakage-free: identical masking to run_cv.
The early-stopping probe uses fold-0 labels ONLY to pick a stopping round
(reported for information; final configs use fixed round counts).
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load, build_X, run_cv, save_oof, apply_ei  # noqa: E402

D = load()
SP_K, EX_K = 15, 25  # fixed: baseline feature set

BASE = dict(
    objective="multiclass", num_class=60, learning_rate=0.06,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbosity=-1, num_threads=4, seed=0,
)


def balanced_weights(ytr):
    cnt = np.bincount(ytr, minlength=60).astype(np.float64)
    w = len(ytr) / (60.0 * np.maximum(cnt, 1))
    return w[ytr]


def fold0_data():
    y, folds, is_tr = D["y"], D["folds"], D["is_train"]
    tr = np.where(is_tr)[0]
    known = np.where(is_tr & (folds != 0), y, -1).astype(np.int64)
    X, names = build_X(D, known, SP_K, EX_K)
    fit = tr[folds[tr] != 0]
    va = tr[folds[tr] == 0]
    feat = [n.replace(" ", "_") for n in names]
    return X, feat, fit, va, y


def screen(configs):
    X, feat, fit, va, y = fold0_data()
    for cfg in configs:
        name = cfg["name"]
        p = dict(BASE)
        p.update(cfg.get("params", {}))
        rounds = cfg.get("rounds", 400)
        w = balanced_weights(y[fit]) if cfg.get("balanced") else None
        t0 = time.time()
        ds = lgb.Dataset(X[fit], y[fit], weight=w, feature_name=feat)
        best_it = None
        if cfg.get("early_stop"):
            vs = lgb.Dataset(X[va], y[va], reference=ds)
            m = lgb.train(p, ds, rounds, valid_sets=[vs],
                          callbacks=[lgb.early_stopping(cfg["early_stop"], verbose=False)])
            best_it = m.best_iteration
            probs = m.predict(X[va], num_iteration=best_it)
        else:
            m = lgb.train(p, ds, rounds)
            probs = m.predict(X[va])
        acc = (probs.argmax(1) == y[va]).mean()
        pei = apply_ei(probs, D["ei_known"][va], D["ei_of_label"])
        acc_ei = (pei.argmax(1) == y[va]).mean()
        extra = f" best_it={best_it}" if best_it else ""
        print(f"{name:28s} acc={acc:.4f} +EI={acc_ei:.4f} "
              f"({time.time()-t0:.0f}s){extra}", flush=True)


STAGES = {
    # learning_rate x num_boost_round tradeoff (+ early-stop probe)
    "A": [
        dict(name="base_lr06_r400"),
        dict(name="lr06_r600", rounds=600),
        dict(name="lr04_r800", params=dict(learning_rate=0.04), rounds=800),
        dict(name="es_probe_lr04", params=dict(learning_rate=0.04), rounds=2500,
             early_stop=150),
    ],
    # num_leaves + short-round check (es_probe found best_it~106 @ lr.04)
    "B": [
        dict(name="lr06_r150", rounds=150),
        dict(name="leaves31_r400", params=dict(num_leaves=31), rounds=400),
        dict(name="leaves127_r400", params=dict(num_leaves=127), rounds=400),
    ],
    # min_data_in_leaf
    "C": [
        dict(name="mdl5_r400", params=dict(min_data_in_leaf=5), rounds=400),
        dict(name="mdl10_r400", params=dict(min_data_in_leaf=10), rounds=400),
        dict(name="mdl40_r400", params=dict(min_data_in_leaf=40), rounds=400),
    ],
    # feature_fraction
    "D": [
        dict(name="ff040_r400", params=dict(feature_fraction=0.4), rounds=400),
        dict(name="ff055_r400", params=dict(feature_fraction=0.55), rounds=400),
    ],
    # bagging / lambdas / max_bin / path_smooth
    "E": [
        dict(name="bag06_r400", params=dict(bagging_fraction=0.6), rounds=400),
        dict(name="bag09_r400", params=dict(bagging_fraction=0.9), rounds=400),
        dict(name="l1_05_r400", params=dict(lambda_l1=0.5), rounds=400),
        dict(name="l2_5_r400", params=dict(lambda_l2=5.0), rounds=400),
        dict(name="maxbin255_r400", params=dict(max_bin=255), rounds=400),
        dict(name="maxbin63_r400", params=dict(max_bin=63), rounds=400),
        dict(name="psmooth1_r400", params=dict(path_smooth=1.0), rounds=400),
    ],
    # dart / class-balanced weights
    "F": [
        dict(name="dart_lr06_r400", params=dict(boosting="dart"), rounds=400),
        dict(name="balanced_r400", balanced=True, rounds=400),
    ],
    # combinations of stage winners + seed noise (filled in after A-F)
    "G": [
        dict(name="G_l31_ff055", params=dict(num_leaves=31, feature_fraction=0.55),
             rounds=400),
        dict(name="G_l31_ff055_r600",
             params=dict(num_leaves=31, feature_fraction=0.55), rounds=600),
        dict(name="G_l31_ff055_s1",
             params=dict(num_leaves=31, feature_fraction=0.55, seed=1,
                         bagging_seed=101, feature_fraction_seed=201), rounds=400),
        dict(name="G_l31_s1",
             params=dict(num_leaves=31, seed=1, bagging_seed=101,
                         feature_fraction_seed=201), rounds=400),
        dict(name="G_l31_ff055_mb255_ps1",
             params=dict(num_leaves=31, feature_fraction=0.55, max_bin=255,
                         path_smooth=1.0), rounds=400),
        dict(name="G_dart_l31_ff055",
             params=dict(boosting="dart", num_leaves=31, feature_fraction=0.55),
             rounds=400),
    ],
}


def _fp(params, rounds, seed):
    p = dict(BASE)
    p.update(params)
    p["seed"] = seed
    p["bagging_seed"] = seed + 100
    p["feature_fraction_seed"] = seed + 200

    def fit_predict(Xtr, ytr, Xva, names):
        ds = lgb.Dataset(Xtr, ytr, feature_name=[n.replace(" ", "_") for n in names])
        m = lgb.train(p, ds, num_boost_round=rounds)
        return m.predict(Xva)
    return fit_predict


# Full-CV results (5-fold OOF acc / +EI):
#   baseline seed0 (oof/lgbm_base)        0.7692 / 0.7706
#   baseline seed1                        0.7672 / 0.7682  <- seed noise ~0.002
#   max_bin255+path_smooth1 seed0         0.7678 / 0.7686  -> lgbm_tuned
#   num_leaves31 seed0                    0.7646 / 0.7656
#   num_leaves31 seed1                    0.7656 / 0.7668  -> lgbm_tuned2
#   dart+l31+ff055 seed1                  0.7612 / 0.7626
# Conclusion: baseline params sit on the optimum plateau; every fold-0
# screening delta (all within +-0.004) was noise. No config beat 0.7692.
FINALS = {
    "tuned": dict(params=dict(max_bin=255, path_smooth=1.0), rounds=400,
                  seed=0, save="lgbm_tuned"),
    "tuned2": dict(params=dict(num_leaves=31), rounds=400, seed=1,
                   save="lgbm_tuned2"),
    # earlier candidates, kept for reproducibility of the log above:
    "l31_s0": dict(params=dict(num_leaves=31), rounds=400, seed=0, save=None),
    "dart_l31": dict(params=dict(boosting="dart", num_leaves=31,
                                 feature_fraction=0.55), rounds=400, seed=1,
                     save=None),
    "base_s1": dict(params=dict(), rounds=400, seed=1, save=None),
}


def main():
    mode = sys.argv[1]
    if mode == "screen":
        screen(STAGES[sys.argv[2]])
    elif mode == "final":
        cfg = FINALS[sys.argv[2]]
        res = run_cv(D, _fp(cfg["params"], cfg["rounds"], cfg["seed"]))
        if cfg["save"]:
            save_oof(cfg["save"], res)
        else:
            print(f"(unsaved) acc={res['acc']:.4f} +EI={res['acc_ei']:.4f}")


if __name__ == "__main__":
    main()
