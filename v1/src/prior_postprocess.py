from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from train_model import LABEL, N_FOLDS, OFFICIAL, OUT, SEED, load_data


GROUPS = [("Section_ID",), ("Region",), ("Segment",), ("Mouse_ID", "AP_position")]


def make_keys(meta, columns):
    return meta.loc[:, list(columns)].fillna("__MISSING__").astype(str).agg("||".join, axis=1).reset_index(drop=True)


def ratio_from_fit(fit_meta, fit_y, apply_meta, columns, n_classes, alpha):
    fit_keys = make_keys(fit_meta, columns)
    apply_keys = make_keys(apply_meta, columns)
    global_prior = np.bincount(fit_y, minlength=n_classes).astype(float)
    global_prior /= global_prior.sum()
    unique = pd.Index(fit_keys.unique())
    lookup = {key: i for i, key in enumerate(unique)}
    group_index = fit_keys.map(lookup).to_numpy()
    table = np.zeros((len(unique), n_classes), dtype=float)
    np.add.at(table, (group_index, fit_y), 1)
    totals = table.sum(axis=1)
    apply_group = apply_keys.map(lookup)
    conditional = np.tile(global_prior, (len(apply_meta), 1))
    seen = apply_group.notna().to_numpy()
    idx = apply_group[seen].astype(int).to_numpy()
    conditional[seen] = (table[idx] + alpha * global_prior) / (totals[idx, None] + alpha)
    return np.clip(conditional / np.maximum(global_prior, 1e-9), .1, 10.0)


def metrics(y, probability):
    pred = probability.argmax(axis=1)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro"))}


def main():
    _, _, meta_train, meta_test = load_data()
    encoder = LabelEncoder().fit(meta_train[LABEL].astype(str))
    y = encoder.transform(meta_train[LABEL].astype(str))
    n_classes = len(encoder.classes_)
    stored = np.load(OUT / "multiseed_probabilities.npz")
    base_oof, base_test = stored["oof"], stored["test"]
    splitter = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(meta_train, y))
    alphas = [12.0, 30.0, 18.0, 25.0]
    oof_ratios, test_ratios = [], []
    for columns, alpha in zip(GROUPS, alphas):
        oof_ratio = np.ones_like(base_oof, dtype=float)
        test_ratio = np.zeros_like(base_test, dtype=float)
        for fit_idx, val_idx in splitter:
            oof_ratio[val_idx] = ratio_from_fit(meta_train.iloc[fit_idx], y[fit_idx], meta_train.iloc[val_idx], columns, n_classes, alpha)
            test_ratio += ratio_from_fit(meta_train.iloc[fit_idx], y[fit_idx], meta_test, columns, n_classes, alpha) / N_FOLDS
        oof_ratios.append(oof_ratio)
        test_ratios.append(test_ratio)

    best = (-1.0, None)
    choices = [0.0, .1, .2, .3, .4, .5]
    for weights in itertools.product(choices, repeat=len(GROUPS)):
        logp = np.log(np.maximum(base_oof, 1e-12))
        for weight, ratio in zip(weights, oof_ratios):
            logp += weight * np.log(ratio)
        prediction = logp.argmax(axis=1)
        acc = accuracy_score(y, prediction)
        if acc > best[0]:
            best = (acc, weights)
    weights = best[1]
    oof_logp = np.log(np.maximum(base_oof, 1e-12))
    test_logp = np.log(np.maximum(base_test, 1e-12))
    for weight, oof_ratio, test_ratio in zip(weights, oof_ratios, test_ratios):
        oof_logp += weight * np.log(oof_ratio)
        test_logp += weight * np.log(test_ratio)
    oof_probability = np.exp(oof_logp - oof_logp.max(axis=1, keepdims=True))
    test_probability = np.exp(test_logp - test_logp.max(axis=1, keepdims=True))
    oof_probability /= oof_probability.sum(axis=1, keepdims=True)
    test_probability /= test_probability.sum(axis=1, keepdims=True)
    summary = {
        "groups": [list(g) for g in GROUPS], "weights": list(weights),
        "baseline": metrics(y, base_oof), "corrected": metrics(y, oof_probability),
    }
    print(json.dumps(summary, indent=2))
    (OUT / "prior_postprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "prior_postprocess_probabilities.npz", oof=oof_probability, test=test_probability, y=y)
    sample = pd.read_csv(OFFICIAL / "prediction" / "prediction.csv")
    sample.iloc[:, 1] = encoder.inverse_transform(test_probability.argmax(axis=1))
    sample.to_csv(OUT / "prediction_prior_corrected.csv", index=False)


if __name__ == "__main__":
    main()
