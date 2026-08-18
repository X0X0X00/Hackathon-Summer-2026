from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


SEED = 2026
N_FOLDS = 4
V1_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = V1_ROOT.parent
OUT = V1_ROOT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
LABEL = "MERFISH_cell_type_annotation"
CAT_COLS = [
    "Datasets", "Region", "Excitatory_vs_Inhibitory", "Segment",
    "Gender", "Mouse_ID", "AP_position", "Section_ID",
]
NUM_META = ["volume", "center_x", "center_y"]


def load_data():
    data = OFFICIAL / "data"
    tr_counts = pd.read_csv(data / "counts_train.csv", index_col=0)
    te_counts = pd.read_csv(data / "counts_test.csv", index_col=0)
    tr_meta = pd.read_csv(data / "meta_train.csv", index_col=0)
    te_meta = pd.read_csv(data / "meta_test.csv", index_col=0)
    assert tr_counts.index.equals(tr_meta.index)
    assert te_counts.index.equals(te_meta.index)
    return tr_counts, te_counts, tr_meta, te_meta


def expression_features(train: pd.DataFrame, test: pd.DataFrame):
    both = pd.concat([train, test])
    values = both.to_numpy(dtype=np.float32)
    totals = values.sum(axis=1, keepdims=True)
    detected = (values > 0).sum(axis=1, keepdims=True).astype(np.float32)
    log_raw = np.log1p(values)
    log_normalized = np.log1p(values / np.maximum(totals, 1) * 100.0)
    features = np.hstack([log_raw, log_normalized, np.log1p(totals), detected])
    return features[: len(train)], features[len(train) :]


def prepare_logistic(expr_train, expr_test, meta_train, meta_test):
    both_meta = pd.concat([meta_train[NUM_META + CAT_COLS], meta_test[NUM_META + CAT_COLS]])
    encoded = pd.get_dummies(both_meta, columns=CAT_COLS, dummy_na=True, dtype=np.float32)
    encoded["volume"] = np.log1p(encoded["volume"].clip(lower=0))
    encoded = encoded.fillna(0)
    meta_values = encoded.to_numpy(dtype=np.float32)
    full = np.hstack([np.vstack([expr_train, expr_test]), meta_values])
    return full[: len(expr_train)], full[len(expr_train) :]


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main():
    counts_train, counts_test, meta_train, meta_test = load_data()
    encoder = LabelEncoder().fit(meta_train[LABEL].astype(str))
    y = encoder.transform(meta_train[LABEL].astype(str))
    n_classes = len(encoder.classes_)
    expr_train, expr_test = expression_features(counts_train, counts_test)
    logit_train, logit_test = prepare_logistic(expr_train, expr_test, meta_train, meta_test)

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_logit = np.zeros((len(y), n_classes), dtype=np.float32)
    oof_lgbm = np.zeros_like(oof_logit)
    test_logit = np.zeros((len(counts_test), n_classes), dtype=np.float32)
    test_lgbm = np.zeros_like(test_logit)
    fold_results = []

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(logit_train, y), 1):
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(logit_train[fit_idx])
        x_val = scaler.transform(logit_train[val_idx])
        x_test = scaler.transform(logit_test)
        logistic = LogisticRegression(C=.5, max_iter=2500, solver="lbfgs")
        logistic.fit(x_fit, y[fit_idx])
        oof_logit[val_idx] = logistic.predict_proba(x_val)
        test_logit += logistic.predict_proba(x_test) / N_FOLDS

        tree = LGBMClassifier(
            objective="multiclass",
            n_estimators=800,
            learning_rate=.035,
            num_leaves=24,
            max_depth=-1,
            min_child_samples=18,
            subsample=.85,
            colsample_bytree=.75,
            reg_lambda=2.0,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        tree.fit(
            logit_train[fit_idx], y[fit_idx],
            eval_set=[(logit_train[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        tree_val = tree.predict_proba(logit_train[val_idx])
        tree_te = tree.predict_proba(logit_test)
        if tree_val.shape[1] != n_classes:
            raise RuntimeError(f"Fold {fold}: LightGBM returned {tree_val.shape[1]} classes, expected {n_classes}")
        oof_lgbm[val_idx] = tree_val
        test_lgbm += tree_te / N_FOLDS
        result = {
            "fold": fold,
            "logistic": metrics(y[val_idx], oof_logit[val_idx].argmax(axis=1)),
            "lightgbm": metrics(y[val_idx], oof_lgbm[val_idx].argmax(axis=1)),
            "lightgbm_best_iteration": int(tree.best_iteration_),
        }
        fold_results.append(result)
        print(json.dumps(result), flush=True)

    candidates = []
    for tree_weight in np.linspace(0, 1, 21):
        blended = tree_weight * oof_lgbm + (1 - tree_weight) * oof_logit
        candidates.append((accuracy_score(y, blended.argmax(axis=1)), float(tree_weight)))
    best_accuracy, best_tree_weight = max(candidates)
    oof_blend = best_tree_weight * oof_lgbm + (1 - best_tree_weight) * oof_logit
    test_blend = best_tree_weight * test_lgbm + (1 - best_tree_weight) * test_logit

    summary = {
        "seed": SEED,
        "folds": N_FOLDS,
        "logistic_oof": metrics(y, oof_logit.argmax(axis=1)),
        "lightgbm_oof": metrics(y, oof_lgbm.argmax(axis=1)),
        "blend_oof": metrics(y, oof_blend.argmax(axis=1)),
        "best_lightgbm_weight": best_tree_weight,
        "fold_results": fold_results,
    }
    print(json.dumps(summary, indent=2), flush=True)
    (OUT / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "probabilities.npz", oof_logit=oof_logit, oof_lgbm=oof_lgbm, test_logit=test_logit, test_lgbm=test_lgbm, y=y)

    sample = pd.read_csv(OFFICIAL / "prediction" / "prediction.csv")
    assert sample.iloc[:, 0].astype(str).tolist() == meta_test.index.astype(str).tolist()
    sample.iloc[:, 1] = encoder.inverse_transform(test_blend.argmax(axis=1))
    sample.to_csv(OUT / "prediction.csv", index=False)

    # Grouped diagnostic uses the faster linear model and holds out entire sections.
    group_splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    group_scores = []
    groups = meta_train["Section_ID"].astype(str).to_numpy()
    for fold, (fit_idx, val_idx) in enumerate(group_splitter.split(logit_train, y, groups), 1):
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(logit_train[fit_idx])
        x_val = scaler.transform(logit_train[val_idx])
        model = LogisticRegression(C=.5, max_iter=2500, solver="lbfgs")
        model.fit(x_fit, y[fit_idx])
        score = metrics(y[val_idx], model.predict(x_val))
        score["fold"] = fold
        group_scores.append(score)
    (OUT / "section_group_cv.json").write_text(json.dumps(group_scores, indent=2), encoding="utf-8")
    print("section_group_cv", json.dumps(group_scores))


if __name__ == "__main__":
    main()
