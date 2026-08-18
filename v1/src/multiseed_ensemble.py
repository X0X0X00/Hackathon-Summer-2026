from __future__ import annotations

import itertools
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from train_model import LABEL, N_FOLDS, OFFICIAL, OUT, SEED, expression_features, load_data, prepare_logistic


VARIANTS = [
    {"name": "wide", "seed": 3407, "num_leaves": 32, "min_child_samples": 12, "colsample_bytree": .85, "reg_lambda": 4.0, "reg_alpha": .1},
    {"name": "compact", "seed": 7319, "num_leaves": 16, "min_child_samples": 26, "colsample_bytree": 1.0, "reg_lambda": 2.0, "reg_alpha": 0.0},
]


def metrics(y, prob):
    pred = prob.argmax(axis=1)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro"))}


def main():
    counts_train, counts_test, meta_train, meta_test = load_data()
    encoder = LabelEncoder().fit(meta_train[LABEL].astype(str))
    y = encoder.transform(meta_train[LABEL].astype(str))
    n_classes = len(encoder.classes_)
    expr_train, expr_test = expression_features(counts_train, counts_test)
    x_train, x_test = prepare_logistic(expr_train, expr_test, meta_train, meta_test)
    splitter = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(x_train, y))
    oofs, tests, variant_results = [], [], []

    for variant in VARIANTS:
        oof = np.zeros((len(y), n_classes), dtype=np.float32)
        test = np.zeros((len(counts_test), n_classes), dtype=np.float32)
        folds = []
        for fold, (fit_idx, val_idx) in enumerate(splitter, 1):
            model = LGBMClassifier(
                objective="multiclass", n_estimators=700, learning_rate=.032,
                num_leaves=variant["num_leaves"], min_child_samples=variant["min_child_samples"],
                colsample_bytree=variant["colsample_bytree"], reg_lambda=variant["reg_lambda"],
                reg_alpha=variant["reg_alpha"], random_state=variant["seed"] + fold,
                n_jobs=-1, verbosity=-1,
            )
            model.fit(
                x_train[fit_idx], y[fit_idx], eval_set=[(x_train[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(75, verbose=False)],
            )
            oof[val_idx] = model.predict_proba(x_train[val_idx])
            test += model.predict_proba(x_test) / N_FOLDS
            folds.append({"fold": fold, "accuracy": float(accuracy_score(y[val_idx], oof[val_idx].argmax(axis=1))), "best_iteration": int(model.best_iteration_)})
        oofs.append(oof)
        tests.append(test)
        variant_results.append({"name": variant["name"], **metrics(y, oof), "folds": folds})
        print(json.dumps(variant_results[-1]), flush=True)

    stored = np.load(OUT / "probabilities.npz")
    sources_oof = [stored["oof_logit"], stored["oof_lgbm"], *oofs]
    sources_test = [stored["test_logit"], stored["test_lgbm"], *tests]
    best = (-1.0, None)
    # Coarse, regularized simplex search; 0.1 increments avoid fragile per-class fitting.
    for units in itertools.product(range(11), repeat=4):
        if sum(units) != 10:
            continue
        weights = np.asarray(units, dtype=float) / 10
        probability = sum(weight * source for weight, source in zip(weights, sources_oof))
        candidate = accuracy_score(y, probability.argmax(axis=1))
        if candidate > best[0]:
            best = (candidate, weights)
    weights = best[1]
    ensemble_oof = sum(weight * source for weight, source in zip(weights, sources_oof))
    ensemble_test = sum(weight * source for weight, source in zip(weights, sources_test))
    summary = {
        "source_order": ["logistic", "lightgbm_original", "lightgbm_wide", "lightgbm_compact"],
        "weights": weights.tolist(),
        "ensemble": metrics(y, ensemble_oof),
        "variants": variant_results,
    }
    print(json.dumps(summary, indent=2), flush=True)
    (OUT / "multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "multiseed_probabilities.npz", oof=ensemble_oof, test=ensemble_test, y=y)
    sample = pd.read_csv(OFFICIAL / "prediction" / "prediction.csv")
    sample.iloc[:, 1] = encoder.inverse_transform(ensemble_test.argmax(axis=1))
    sample.to_csv(OUT / "prediction_multiseed.csv", index=False)


if __name__ == "__main__":
    main()
