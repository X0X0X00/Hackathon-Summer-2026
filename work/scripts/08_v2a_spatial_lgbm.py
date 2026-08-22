"""V2-A-SPATIAL-LGBM: competition-only spatial LightGBM on the 5-fold protocol.

Uses official train+test features for unsupervised graphs (no labels).
Hides fold-f labels before any neighbor-label histogram is built.
Does not write an official submission. Does not start the external-reference phase.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

_WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORK / "src"))
from merfish60.io import repo_root  # noqa: E402
ROOT = repo_root()

from merfish60.io import N_CLASSES, load_dataset, validate_contract  # noqa: E402
from merfish60.metrics import summarize_oof  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.neighbor_labels import (  # noqa: E402
    EX_K,
    SP_K,
    apply_ei,
    build_X,
    encode_train_labels,
    visible_label_codes,
)
from merfish60.official_contract import (  # noqa: E402
    allowed_labels,
    git_commit,
    manifest_sha256,
    verify_official_manifest,
)
from merfish60.registry_v2 import RegistryV2Error, append_registry_v2_row  # noqa: E402
from merfish60.spatial_features import (  # noqa: E402
    K_NBR,
    N_SPATIAL_MEAN,
    PCA_DIM,
    build_spatial_universe,
    ei_of_label_from_train,
)
from merfish60.team_cv import (  # noqa: E402
    TEAM_CV_PROTOCOL,
    TEAM_CV_RANDOM_STATE,
    TEAM_FOLD_VALUES,
    ensure_team_folds,
    team_folds_sha256,
)
from merfish60.v2_metrics import (  # noqa: E402
    slice_metrics,
    universe_fold_ids,
    write_confusion,
    write_json,
    write_oof,
    write_proba,
)


RUN_ID = "V2-A-SPATIAL-LGBM"
COMPLIANCE = "EXTERNAL_REFERENCE_PERMISSION_REQUIRED"
LGBM_SEED = 0
NUM_BOOST_ROUND = 400


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lgbm_params(num_threads: int) -> dict:
    return dict(
        objective="multiclass",
        num_class=N_CLASSES,
        learning_rate=0.06,
        num_leaves=63,
        min_data_in_leaf=20,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=127,
        verbosity=-1,
        num_threads=int(num_threads),
        seed=LGBM_SEED,
    )


def fit_predict(X_train, y_train, X_val, names, params, num_boost_round: int):
    feature_names = [str(name).replace(" ", "_") for name in names]
    dataset = lgb.Dataset(X_train, y_train, feature_name=feature_names)
    model = lgb.train(params, dataset, num_boost_round=num_boost_round)
    return np.asarray(model.predict(X_val), dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--skip-test-proba", action="store_true")
    args = parser.parse_args()

    print("Verifying official data manifest...", flush=True)
    for line in verify_official_manifest(ROOT):
        print(" - {}".format(line), flush=True)

    data = load_dataset(ROOT)
    for line in validate_contract(data):
        print(" - {}".format(line), flush=True)

    class_names = allowed_labels(ROOT)
    folds, fold_messages = ensure_team_folds(
        data.counts_train.index,
        data.counts_test.index,
        data.y_train,
        root=ROOT,
    )
    print("Team-compatible 5-fold protocol:", TEAM_CV_PROTOCOL, flush=True)
    for line in fold_messages:
        print(" - {}".format(line), flush=True)

    t0 = time.perf_counter()
    print("Building transductive spatial/expression universe...", flush=True)
    universe = build_spatial_universe(data)
    print(
        "  static={} pca_var={:.4f} sections={} {}".format(
            universe.X_static.shape,
            universe.pca_var_explained,
            len(universe.section_sizes),
            universe.transductive_note,
        ),
        flush=True,
    )

    y_train = data.y_train.astype(str)
    train_ids = [str(v) for v in y_train.index.tolist()]
    y_codes = encode_train_labels(
        y_train.tolist(), class_names, len(universe.cell_ids), universe.is_train
    )
    fold_universe = universe_fold_ids(folds, universe.cell_ids, universe.is_train)
    ei_of_label = ei_of_label_from_train(data.meta_train, class_names)
    params = lgbm_params(args.num_threads)

    n_train = int(universe.is_train.sum())
    oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    train_pos = np.where(universe.is_train)[0]
    y_train_codes = y_codes[train_pos]
    fold_train = fold_universe[train_pos]

    for fold_id in TEAM_FOLD_VALUES:
        known = visible_label_codes(y_codes, universe.is_train, fold_universe, fold_id)
        features, names = build_X(universe, known, sp_k=SP_K, ex_k=EX_K)
        fit_idx = train_pos[fold_train != fold_id]
        val_idx = train_pos[fold_train == fold_id]
        print(
            "  training fold {} n_fit={} n_val={} n_features={}".format(
                fold_id, len(fit_idx), len(val_idx), features.shape[1]
            ),
            flush=True,
        )
        oof_fold = fit_predict(
            features[fit_idx],
            y_train_codes[fold_train != fold_id],
            features[val_idx],
            names,
            params,
            NUM_BOOST_ROUND,
        )
        oof[fold_train == fold_id] = oof_fold
        pred_fold = argmax_labels(oof_fold, class_names)
        true_fold = np.asarray(class_names)[y_train_codes[fold_train == fold_id]]
        print(
            "  fold {}: {:.4f}".format(fold_id, float(np.mean(pred_fold == true_fold))),
            flush=True,
        )

    assert_probability_rows(oof)
    pred_raw = argmax_labels(oof, class_names)
    true = y_train.to_numpy()
    metrics_raw = summarize_oof(true, pred_raw, fold_train, labels=class_names)
    oof_ei = apply_ei(oof, universe.ei_known[train_pos], ei_of_label)
    assert_probability_rows(oof_ei)
    pred_ei = argmax_labels(oof_ei, class_names)
    metrics_ei = summarize_oof(true, pred_ei, fold_train, labels=class_names)
    slices_raw = slice_metrics(true, pred_raw, data.meta_train, class_names, ei_of_label)
    slices_ei = slice_metrics(true, pred_ei, data.meta_train, class_names, ei_of_label)

    test_rel = None
    if not args.skip_test_proba:
        print("Fitting full-train test probabilities (all train labels visible)...", flush=True)
        known_test = visible_label_codes(y_codes, universe.is_train, fold_universe, None)
        features, names = build_X(universe, known_test, sp_k=SP_K, ex_k=EX_K)
        test_idx = np.where(~universe.is_train)[0]
        test_probs = fit_predict(
            features[train_pos],
            y_train_codes,
            features[test_idx],
            names,
            params,
            NUM_BOOST_ROUND,
        )
        test_ei = apply_ei(test_probs, universe.ei_known[test_idx], ei_of_label)
        test_rel = "work/outputs/probabilities/{}_test_probabilities.csv.gz".format(RUN_ID)
        test_ids = [str(v) for v in universe.cell_ids[test_idx]]
        write_proba(ROOT / test_rel, test_ids, test_ei, class_names)

    runtime = time.perf_counter() - t0

    oof_rel = "work/outputs/oof/{}_oof.csv".format(RUN_ID)
    proba_rel = "work/outputs/probabilities/{}_oof_probabilities.csv.gz".format(RUN_ID)
    proba_ei_rel = "work/outputs/probabilities/{}_oof_probabilities_ei.csv.gz".format(RUN_ID)
    metrics_rel = "work/outputs/metrics/{}_metrics.json".format(RUN_ID)
    cm_rel = "work/outputs/metrics/{}_confusion.csv".format(RUN_ID)

    write_oof(ROOT / oof_rel, train_ids, true, pred_ei, fold_train)
    write_proba(ROOT / proba_rel, train_ids, oof, class_names)
    write_proba(ROOT / proba_ei_rel, train_ids, oof_ei, class_names)
    write_confusion(ROOT / cm_rel, true, pred_ei, class_names)

    feature_blocks = {
        "genes_lognorm": "library-size log1p (median total) of 200 genes",
        "pca50": "PCA50 of standardized lognorm genes, random_state=0, train+test",
        "spatial_knn": "within-Section_ID spatial kNN, K={} cached".format(K_NBR),
        "expression_knn": "PCA50 Euclidean kNN across all cells, K={}".format(K_NBR),
        "neighbor_mean_pca": "mean PCA50 of {} spatial neighbors".format(N_SPATIAL_MEAN),
        "metadata": (
            "log total, log volume, density, x/y, within-section rel x/y, "
            "spatial/expression distances, Region/EI/Segment/AP/gender/mouse/dataset codes"
        ),
        "neighbor_label_hist": (
            "fold-safe 1/(1+d) histograms of {} spatial and {} expression labeled neighbors".format(
                SP_K, EX_K
            )
        ),
    }
    payload = {
        "run_id": RUN_ID,
        "model": "competition-only spatial LightGBM (single seed)",
        "feature_set": "; ".join(
            "{}: {}".format(key, value) for key, value in feature_blocks.items()
        ),
        "feature_blocks": feature_blocks,
        "transductive_unsupervised": True,
        "transductive_note": universe.transductive_note,
        "cv_protocol": TEAM_CV_PROTOCOL,
        "random_seed": LGBM_SEED,
        "lgbm_params": params,
        "num_boost_round": NUM_BOOST_ROUND,
        "n_seeds": 1,
        "class_order": class_names,
        "fold_accuracy_raw": {str(k): v for k, v in metrics_raw["fold_accuracy"].items()},
        "fold_accuracy_ei": {str(k): v for k, v in metrics_ei["fold_accuracy"].items()},
        "oof_accuracy_raw": metrics_raw["oof_accuracy"],
        "oof_accuracy_ei": metrics_ei["oof_accuracy"],
        "macro_f1_raw": metrics_raw["macro_f1"],
        "macro_f1_ei": metrics_ei["macro_f1"],
        "slices_raw": slices_raw,
        "slices_ei": slices_ei,
        "hard_bucket_accuracy_raw": slices_raw["hard_bucket_accuracy"],
        "hard_bucket_accuracy_ei": slices_ei["hard_bucket_accuracy"],
        "neuron_accuracy_raw": slices_raw["neuron_accuracy"],
        "neuron_accuracy_ei": slices_ei["neuron_accuracy"],
        "glial_accuracy_raw": slices_raw["glial_accuracy"],
        "glial_accuracy_ei": slices_ei["glial_accuracy"],
        "confusion_pairs_top10_raw": slices_raw["confusion_pairs_top10"],
        "confusion_pairs_top10_ei": slices_ei["confusion_pairs_top10"],
        "pca_var_explained": universe.pca_var_explained,
        "n_static_features": int(universe.X_static.shape[1]),
        "n_sections": len(universe.section_sizes),
        "runtime_seconds": runtime,
        "oof_path": oof_rel,
        "proba_path": proba_rel,
        "proba_ei_path": proba_ei_rel,
        "test_proba_path": test_rel,
        "metrics_path": metrics_rel,
        "confusion_path": cm_rel,
        "compliance_status": COMPLIANCE,
        "git_commit": git_commit(ROOT),
        "manifest_sha256": manifest_sha256(ROOT),
        "team_folds_sha256": team_folds_sha256(ROOT),
        "timestamp": utc_now(),
        "same_team_benchmark": "single spatial LGBM OOF ~0.762-0.769; not a required match",
        "no_47_model_bag": True,
    }
    write_json(ROOT / metrics_rel, payload)

    try:
        append_registry_v2_row(
            {
                "run_id": RUN_ID,
                "timestamp": payload["timestamp"],
                "git_commit": payload["git_commit"],
                "model": payload["model"],
                "feature_set": payload["feature_set"],
                "cv_protocol": TEAM_CV_PROTOCOL,
                "random_seed": LGBM_SEED,
                "fold_0_accuracy": payload["fold_accuracy_raw"].get("0"),
                "fold_1_accuracy": payload["fold_accuracy_raw"].get("1"),
                "fold_2_accuracy": payload["fold_accuracy_raw"].get("2"),
                "fold_3_accuracy": payload["fold_accuracy_raw"].get("3"),
                "fold_4_accuracy": payload["fold_accuracy_raw"].get("4"),
                "oof_accuracy": payload["oof_accuracy_raw"],
                "oof_accuracy_ei": payload["oof_accuracy_ei"],
                "hard_bucket_accuracy": payload["hard_bucket_accuracy_ei"],
                "neuron_accuracy": payload["neuron_accuracy_ei"],
                "glial_accuracy": payload["glial_accuracy_ei"],
                "macro_f1": payload["macro_f1_ei"],
                "runtime_seconds": runtime,
                "status": "completed",
                "notes": "competition-only; single seed; E/I applied after fold prediction",
                "manifest_sha256": payload["manifest_sha256"],
                "team_folds_sha256": payload["team_folds_sha256"],
                "oof_path": oof_rel,
                "proba_path": proba_rel,
                "metrics_path": metrics_rel,
                "compliance_status": COMPLIANCE,
                "warnings": "none",
            },
            root=ROOT,
            overwrite=args.overwrite,
        )
    except RegistryV2Error as exc:
        print("registry_v2: {}".format(exc), flush=True)
        return 1

    print(
        "V2-A raw OOF={:.6f} +EI={:.6f}".format(
            payload["oof_accuracy_raw"], payload["oof_accuracy_ei"]
        ),
        flush=True,
    )
    print("wrote {}".format(metrics_rel), flush=True)
    print("compliance: {}".format(COMPLIANCE), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
