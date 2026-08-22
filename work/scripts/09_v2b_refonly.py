"""V2-B-REFONLY: same-team reference-only LightGBM on honest 5-fold OOF.

Fit LightGBM on usable Zenodo reference cells only. For each competition
fold f, hide fold-f train labels everywhere (histograms, fitting, masks).
Reference labels remain visible. Does not write an official submission.
Does not start V2-C blending.
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
import pandas as pd

_WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORK / "src"))
from merfish60.io import repo_root  # noqa: E402
ROOT = repo_root()

from merfish60.ext_universe import build_or_load_ext_universe  # noqa: E402
from merfish60.io import N_CLASSES, load_dataset, validate_contract  # noqa: E402
from merfish60.metrics import summarize_oof  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.neighbor_labels import (  # noqa: E402
    EX_K,
    SP_K,
    apply_ei,
    apply_segment_mask,
    build_X,
    segment_allowed_from_reference,
    visible_ext_label_codes,
)
from merfish60.official_contract import (  # noqa: E402
    allowed_labels,
    git_commit,
    manifest_sha256,
    verify_official_manifest,
)
from merfish60.reference import audit_reference  # noqa: E402
from merfish60.registry_v2 import RegistryV2Error, append_registry_v2_row  # noqa: E402
from merfish60.spatial_features import ei_of_label_from_train  # noqa: E402
from merfish60.team_cv import (  # noqa: E402
    TEAM_CV_PROTOCOL,
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


RUN_ID = "V2-B-REFONLY"
COMPLIANCE = "EXTERNAL_REFERENCE_PERMITTED"
LGBM_SEED = 0
NUM_BOOST_ROUND = 700
BRIDGE_OOF = 0.7596
V2A_EI_OOF = 0.7690


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lgbm_params(num_threads: int) -> dict:
    return dict(
        objective="multiclass",
        num_class=N_CLASSES,
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=30,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=127,
        verbosity=-1,
        num_threads=int(num_threads),
        seed=LGBM_SEED,
        bagging_seed=LGBM_SEED + 100,
        feature_fraction_seed=LGBM_SEED + 200,
    )


def fit_predict(X_train, y_train, X_val, names, params, num_boost_round: int):
    feature_names = [str(name).replace(" ", "_") for name in names]
    dataset = lgb.Dataset(X_train, y_train, feature_name=feature_names, free_raw_data=True)
    model = lgb.train(params, dataset, num_boost_round=num_boost_round)
    return np.asarray(model.predict(X_val), dtype=np.float64)


def _cell_delta(pred_a, pred_b, true) -> dict:
    pred_a = np.asarray(pred_a, dtype=object)
    pred_b = np.asarray(pred_b, dtype=object)
    true = np.asarray(true, dtype=object)
    a_ok = pred_a == true
    b_ok = pred_b == true
    return {
        "wrong_to_correct": int((~a_ok & b_ok).sum()),
        "correct_to_wrong": int((a_ok & ~b_ok).sum()),
        "net_gain": int(b_ok.sum() - a_ok.sum()),
        "n": int(len(true)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--skip-test-proba", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
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

    print("Auditing approved external reference...", flush=True)
    audit = audit_reference(data, class_names, root=ROOT)
    if audit["md5"] != "ce06f62c0ec4973581dae17bb76f0cd9":
        print("STOP: reference MD5 mismatch", flush=True)
        return 2
    print(
        "  shape={} md5={} usable={} train_id={} test_id={} vector_dupes={}".format(
            (audit["raw_n_obs"], audit["raw_n_vars"]),
            audit["md5"],
            audit["n_usable_reference_rows"],
            audit["n_train_id_overlaps_removed"],
            audit["n_test_id_overlaps_removed"],
            audit["n_exact_vector_duplicates_removed"],
        ),
        flush=True,
    )
    print(
        "  labels raw_unique={} coverage={}/60 unmapped={} segment_map_valid={}".format(
            audit["n_raw_unique_labels"],
            audit["reference_label_coverage"],
            audit["unmapped_labels"],
            audit["segment_map_valid"],
        ),
        flush=True,
    )

    write_json(ROOT / "work/outputs/metrics/V2-B-REFONLY_reference_provenance.json", audit)
    write_json(
        ROOT / "work/outputs/metrics/V2-B-REFONLY_exclusion_manifest.json",
        {
            "raw_rows": audit["raw_n_obs"],
            "train_cell_ids_removed": audit["n_train_id_overlaps_removed"],
            "test_cell_ids_removed": audit["n_test_id_overlaps_removed"],
            "exact_vector_duplicates_removed": audit["n_exact_vector_duplicates_removed"],
            "final_usable_reference_rows": audit["n_usable_reference_rows"],
            "historical_usable_rows": audit["historical_usable_rows"],
            "historical_vector_dupes": audit["historical_vector_dupes"],
            "usable_delta_vs_historical": audit["usable_delta_vs_historical"],
            "md5": audit["md5"],
            "filename": audit["filename"],
        },
    )
    write_json(
        ROOT / "work/experiments/V2-B-REFONLY_exclusion_manifest.json",
        {
            "raw_rows": audit["raw_n_obs"],
            "train_cell_ids_removed": audit["n_train_id_overlaps_removed"],
            "test_cell_ids_removed": audit["n_test_id_overlaps_removed"],
            "exact_vector_duplicates_removed": audit["n_exact_vector_duplicates_removed"],
            "final_usable_reference_rows": audit["n_usable_reference_rows"],
            "md5": audit["md5"],
        },
    )
    write_json(
        ROOT / "work/outputs/metrics/V2-B-REFONLY_alignment_audit.json",
        {
            "n_competition_genes": audit["n_competition_genes"],
            "gene_order_ok": audit["gene_order_ok"],
            "gene_order_sha256": audit["gene_order_sha256"],
            "competition_genes_aligned": audit["competition_genes_aligned"],
            "n_raw_unique_labels": audit["n_raw_unique_labels"],
            "raw_unique_labels": audit["raw_unique_labels"],
            "n_normalized_labels_matching_taxonomy": audit["n_normalized_labels_matching_taxonomy"],
            "unmapped_labels": audit["unmapped_labels"],
            "competition_classes_missing_from_reference": audit[
                "competition_classes_missing_from_reference"
            ],
            "reference_label_coverage": audit["reference_label_coverage"],
            "segment_map_valid": audit["segment_map_valid"],
            "laminae_segment_train_comparable": audit["laminae_segment_train_comparable"],
            "laminae_segment_train_matches": audit["laminae_segment_train_matches"],
            "laminae_segment_map_n": audit["laminae_segment_map_n"],
            "region_map_valid": audit["region_map_valid"],
            "ap_map_valid": audit["ap_map_valid"],
            "train_count_alignment_ok": audit["train_count_alignment_ok"],
            "train_label_alignment_ok": audit["train_label_alignment_ok"],
            "train_coordinate_alignment_ok": audit["train_coordinate_alignment_ok"],
            "test_target_used": False,
        },
    )
    write_json(
        ROOT / "work/reports/V2-B-REFONLY_methodology.json",
        {
            "read_only_source": "team/main via git show",
            "scripts": [
                "work/README.md",
                "work/build_reference_ids.py",
                "work/prep_ext.py",
                "work/common_ext.py",
                "work/ext_refonly.py",
                "work/ext_post.py",
            ],
            "exclusion": (
                "drop all competition train Cell_IDs, drop all competition test "
                "Cell_IDs, align 200 genes in official column order, drop exact "
                "200-gene count-vector duplicates versus train+test counts"
            ),
            "label_column": "obs['MERFISH cell type annotation']",
            "label_normalization": "replace space and hyphen with underscore; map to official 60",
            "features": (
                "prep_ext: 200 lognorm genes, PCA50 on full deposit, spatial kNN "
                "within Section ID, expression kNN on PCA50, neighbor-mean PCA k=10, "
                "Region/EI/Segment/AP/gender/mouse/dataset; Segment from Laminae if 1:1"
            ),
            "fit": "LightGBM on usable reference rows only; not competition train",
            "fold_protocol": (
                "work/experiments/team_folds_5_seed42.csv; fold-f competition labels hidden "
                "everywhere; reference labels always visible; other train-fold labels "
                "may enter neighbor histograms"
            ),
            "lgbm": {
                "learning_rate": 0.05,
                "num_leaves": 127,
                "min_data_in_leaf": 30,
                "feature_fraction": 0.5,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l2": 1.0,
                "max_bin": 127,
                "seed": 0,
                "num_boost_round": NUM_BOOST_ROUND,
                "early_stopping": False,
            },
            "postprocess_order": ["raw", "+E/I", "+Segment mask if Laminae audit passes"],
            "not_reproduced": (
                "team ext_refonly.py evaluates all 5000 train with all train labels "
                "visible; V2-B uses honest 5-fold instead. Fixed 700 rounds from "
                "README refonly_full, not CLI default 2000+early-stop."
            ),
        },
    )

    if audit["exclusion_materially_different"]:
        print(
            "STOP: exclusion counts differ materially from historical "
            "136574 usable / 47 vector duplicates "
            "(usable={} dupes={}).".format(
                audit["n_usable_reference_rows"],
                audit["n_exact_vector_duplicates_removed"],
            ),
            flush=True,
        )
        return 2
    if not audit["gene_order_ok"]:
        print("STOP: gene order alignment failed", flush=True)
        return 2
    if args.audit_only:
        print("audit-only complete", flush=True)
        return 0

    t0 = time.perf_counter()
    print("Building or loading extended universe cache...", flush=True)
    universe = build_or_load_ext_universe(audit, data, class_names, root=ROOT)
    audit.pop("_adata", None)
    print(
        "  static={} pca_var={:.4f} sections={} ref={} {}".format(
            universe.X_static.shape,
            universe.pca_var_explained,
            len(universe.section_sizes),
            int(universe.is_ref.sum()),
            universe.transductive_note,
        ),
        flush=True,
    )

    train_ids = [str(v) for v in data.counts_train.index.tolist()]
    id_pos = {str(cid): i for i, cid in enumerate(universe.cell_ids.tolist())}
    train_pos = np.array([id_pos[cid] for cid in train_ids], dtype=np.int64)
    test_ids = [str(v) for v in data.counts_test.index.tolist()]
    test_pos = np.array([id_pos[cid] for cid in test_ids], dtype=np.int64)
    fold_universe = universe_fold_ids(folds, universe.cell_ids, universe.is_train)
    fold_train = fold_universe[train_pos]
    y_train = data.y_train.astype(str)
    true = y_train.loc[train_ids].to_numpy()
    ei_of_label = ei_of_label_from_train(data.meta_train, class_names)
    params = lgbm_params(args.num_threads)
    ref_idx = np.where(universe.is_ref)[0]
    if (universe.y_codes[ref_idx] < 0).any():
        print("STOP: unlabeled usable reference rows", flush=True)
        return 2

    n_train = len(train_ids)
    oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    for fold_id in TEAM_FOLD_VALUES:
        known = visible_ext_label_codes(
            universe.y_codes,
            universe.is_train,
            universe.is_ref,
            fold_universe,
            fold_id,
        )
        if (known[train_pos][fold_train == fold_id] != -1).any():
            print("STOP: held-out fold labels leaked into known", flush=True)
            return 2
        if (known[test_pos] != -1).any():
            print("STOP: test labels visible", flush=True)
            return 2
        features, names = build_X(universe, known, sp_k=SP_K, ex_k=EX_K)
        val_idx = train_pos[fold_train == fold_id]
        print(
            "  training fold {} n_fit={} n_val={} n_features={}".format(
                fold_id, len(ref_idx), len(val_idx), features.shape[1]
            ),
            flush=True,
        )
        fold_t0 = time.perf_counter()
        oof_fold = fit_predict(
            features[ref_idx],
            universe.y_codes[ref_idx],
            features[val_idx],
            names,
            params,
            NUM_BOOST_ROUND,
        )
        oof[fold_train == fold_id] = oof_fold
        pred_fold = argmax_labels(oof_fold, class_names)
        true_fold = true[fold_train == fold_id]
        print(
            "  fold {}: {:.4f}  ({:.0f}s)".format(
                fold_id,
                float(np.mean(pred_fold == true_fold)),
                time.perf_counter() - fold_t0,
            ),
            flush=True,
        )
        del features

    assert_probability_rows(oof)
    pred_raw = argmax_labels(oof, class_names)
    metrics_raw = summarize_oof(true, pred_raw, fold_train, labels=class_names)
    oof_ei = apply_ei(oof, universe.ei_known[train_pos], ei_of_label)
    assert_probability_rows(oof_ei)
    pred_ei = argmax_labels(oof_ei, class_names)
    metrics_ei = summarize_oof(true, pred_ei, fold_train, labels=class_names)
    slices_raw = slice_metrics(true, pred_raw, data.meta_train.loc[train_ids], class_names, ei_of_label)
    slices_ei = slice_metrics(true, pred_ei, data.meta_train.loc[train_ids], class_names, ei_of_label)

    segment_valid = bool(audit["segment_map_valid"])
    oof_seg = None
    pred_seg = None
    metrics_seg = None
    slices_seg = None
    allowed_seg = None
    if segment_valid:
        allowed_seg = segment_allowed_from_reference(
            universe.segment, universe.is_ref, universe.y_codes
        )
        oof_seg = apply_segment_mask(oof_ei, universe.segment[train_pos], allowed_seg)
        assert_probability_rows(oof_seg)
        pred_seg = argmax_labels(oof_seg, class_names)
        metrics_seg = summarize_oof(true, pred_seg, fold_train, labels=class_names)
        slices_seg = slice_metrics(
            true, pred_seg, data.meta_train.loc[train_ids], class_names, ei_of_label
        )

    test_rel = None
    test_ei_rel = None
    test_seg_rel = None
    if not args.skip_test_proba:
        print("Fitting full-reference test probabilities (all train labels visible)...", flush=True)
        known_test = visible_ext_label_codes(
            universe.y_codes,
            universe.is_train,
            universe.is_ref,
            fold_universe,
            None,
        )
        if (known_test[test_pos] != -1).any():
            print("STOP: test labels visible at test-time", flush=True)
            return 2
        features, names = build_X(universe, known_test, sp_k=SP_K, ex_k=EX_K)
        test_probs = fit_predict(
            features[ref_idx],
            universe.y_codes[ref_idx],
            features[test_pos],
            names,
            params,
            NUM_BOOST_ROUND,
        )
        test_ei = apply_ei(test_probs, universe.ei_known[test_pos], ei_of_label)
        test_rel = "work/outputs/probabilities/{}_test_probabilities.csv.gz".format(RUN_ID)
        test_ei_rel = "work/outputs/probabilities/{}_test_probabilities_ei.csv.gz".format(RUN_ID)
        write_proba(ROOT / test_rel, test_ids, test_probs, class_names)
        write_proba(ROOT / test_ei_rel, test_ids, test_ei, class_names)
        if segment_valid and allowed_seg is not None:
            test_seg = apply_segment_mask(test_ei, universe.segment[test_pos], allowed_seg)
            test_seg_rel = "work/outputs/probabilities/{}_test_probabilities_seg.csv.gz".format(RUN_ID)
            write_proba(ROOT / test_seg_rel, test_ids, test_seg, class_names)
        del features

    runtime = time.perf_counter() - t0
    best_name = "ei"
    best_pred = pred_ei
    best_metrics = metrics_ei
    best_slices = slices_ei
    best_acc = metrics_ei["oof_accuracy"]
    if metrics_seg is not None and metrics_seg["oof_accuracy"] >= best_acc:
        best_name = "segment"
        best_pred = pred_seg
        best_metrics = metrics_seg
        best_slices = slices_seg
        best_acc = metrics_seg["oof_accuracy"]

    oof_rel = "work/outputs/oof/{}_oof.csv".format(RUN_ID)
    proba_rel = "work/outputs/probabilities/{}_oof_probabilities.csv.gz".format(RUN_ID)
    proba_ei_rel = "work/outputs/probabilities/{}_oof_probabilities_ei.csv.gz".format(RUN_ID)
    metrics_rel = "work/outputs/metrics/{}_metrics.json".format(RUN_ID)
    cm_rel = "work/outputs/metrics/{}_confusion.csv".format(RUN_ID)
    write_oof(ROOT / oof_rel, train_ids, true, best_pred, fold_train)
    write_proba(ROOT / proba_rel, train_ids, oof, class_names)
    write_proba(ROOT / proba_ei_rel, train_ids, oof_ei, class_names)
    if oof_seg is not None:
        write_proba(
            ROOT / "work/outputs/probabilities/{}_oof_probabilities_seg.csv.gz".format(RUN_ID),
            train_ids,
            oof_seg,
            class_names,
        )
    write_confusion(ROOT / cm_rel, true, best_pred, class_names)

    v2a_path = ROOT / "work/outputs/oof/V2-A-SPATIAL-LGBM_oof.csv"
    vs_v2a = None
    if v2a_path.is_file():
        v2a = pd.read_csv(v2a_path, dtype={"Cell_ID": str, "predicted_label": str})
        merged = pd.DataFrame({"Cell_ID": train_ids, "true": true, "v2b": best_pred}).merge(
            v2a[["Cell_ID", "predicted_label"]], on="Cell_ID"
        )
        vs_v2a = _cell_delta(merged["predicted_label"], merged["v2b"], merged["true"])

    payload = {
        "run_id": RUN_ID,
        "model": "reference-only LightGBM (same-team ext_refonly family, honest 5-fold)",
        "feature_set": universe.transductive_note,
        "cv_protocol": TEAM_CV_PROTOCOL,
        "random_seed": LGBM_SEED,
        "lgbm_params": params,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping": False,
        "n_usable_reference_rows": audit["n_usable_reference_rows"],
        "reference_md5": audit["md5"],
        "segment_map_valid": segment_valid,
        "segment_mask_applied": bool(metrics_seg is not None),
        "best_postprocess": best_name,
        "class_order": class_names,
        "fold_accuracy_raw": {str(k): v for k, v in metrics_raw["fold_accuracy"].items()},
        "fold_accuracy_ei": {str(k): v for k, v in metrics_ei["fold_accuracy"].items()},
        "fold_accuracy_seg": (
            {str(k): v for k, v in metrics_seg["fold_accuracy"].items()} if metrics_seg else None
        ),
        "oof_accuracy_raw": metrics_raw["oof_accuracy"],
        "oof_accuracy_ei": metrics_ei["oof_accuracy"],
        "oof_accuracy_seg": None if metrics_seg is None else metrics_seg["oof_accuracy"],
        "macro_f1_raw": metrics_raw["macro_f1"],
        "macro_f1_ei": metrics_ei["macro_f1"],
        "macro_f1_seg": None if metrics_seg is None else metrics_seg["macro_f1"],
        "correct_cells_raw": int(round(metrics_raw["oof_accuracy"] * 5000)),
        "correct_cells_ei": int(round(metrics_ei["oof_accuracy"] * 5000)),
        "correct_cells_seg": (
            None if metrics_seg is None else int(round(metrics_seg["oof_accuracy"] * 5000))
        ),
        "slices_raw": slices_raw,
        "slices_ei": slices_ei,
        "slices_seg": slices_seg,
        "hard_bucket_accuracy_raw": slices_raw["hard_bucket_accuracy"],
        "hard_bucket_accuracy_ei": slices_ei["hard_bucket_accuracy"],
        "hard_bucket_accuracy_seg": None if slices_seg is None else slices_seg["hard_bucket_accuracy"],
        "neuron_accuracy_ei": slices_ei["neuron_accuracy"],
        "glial_accuracy_ei": slices_ei["glial_accuracy"],
        "confusion_pairs_top10_ei": slices_ei["confusion_pairs_top10"],
        "confusion_pairs_top10_best": best_slices["confusion_pairs_top10"],
        "bridge_oof": BRIDGE_OOF,
        "v2a_ei_oof": V2A_EI_OOF,
        "delta_best_minus_bridge": best_acc - BRIDGE_OOF,
        "delta_best_minus_v2a": best_acc - V2A_EI_OOF,
        "v2b_vs_v2a_cells": vs_v2a,
        "pca_var_explained": universe.pca_var_explained,
        "n_static_features": int(universe.X_static.shape[1]),
        "n_sections": len(universe.section_sizes),
        "runtime_seconds": runtime,
        "oof_path": oof_rel,
        "proba_path": proba_rel,
        "proba_ei_path": proba_ei_rel,
        "test_proba_path": test_rel,
        "test_proba_ei_path": test_ei_rel,
        "test_proba_seg_path": test_seg_rel,
        "metrics_path": metrics_rel,
        "confusion_path": cm_rel,
        "compliance_status": COMPLIANCE,
        "git_commit": git_commit(ROOT),
        "manifest_sha256": manifest_sha256(ROOT),
        "team_folds_sha256": team_folds_sha256(ROOT),
        "timestamp": utc_now(),
        "quality_gate_target": 0.80,
        "quality_gate_pass": bool(best_acc >= 0.80),
        "do_not_start_v2c": True,
        "test_target_used": False,
    }
    write_json(ROOT / metrics_rel, payload)
    write_json(
        ROOT / "work/outputs/metrics/V2-B-REFONLY_comparison.json",
        {
            "bridge_oof": BRIDGE_OOF,
            "v2a_ei_oof": V2A_EI_OOF,
            "v2b_raw": metrics_raw["oof_accuracy"],
            "v2b_ei": metrics_ei["oof_accuracy"],
            "v2b_seg": None if metrics_seg is None else metrics_seg["oof_accuracy"],
            "best_postprocess": best_name,
            "best_oof": best_acc,
            "delta_vs_bridge": best_acc - BRIDGE_OOF,
            "delta_vs_v2a": best_acc - V2A_EI_OOF,
            "v2b_vs_v2a_cells": vs_v2a,
            "quality_gate_pass": bool(best_acc >= 0.80),
        },
    )

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
                "fold_0_accuracy": payload["fold_accuracy_ei"].get("0"),
                "fold_1_accuracy": payload["fold_accuracy_ei"].get("1"),
                "fold_2_accuracy": payload["fold_accuracy_ei"].get("2"),
                "fold_3_accuracy": payload["fold_accuracy_ei"].get("3"),
                "fold_4_accuracy": payload["fold_accuracy_ei"].get("4"),
                "oof_accuracy": payload["oof_accuracy_raw"],
                "oof_accuracy_ei": payload["oof_accuracy_ei"],
                "hard_bucket_accuracy": payload["hard_bucket_accuracy_ei"],
                "neuron_accuracy": payload["neuron_accuracy_ei"],
                "glial_accuracy": payload["glial_accuracy_ei"],
                "macro_f1": payload["macro_f1_ei"],
                "runtime_seconds": runtime,
                "status": "completed",
                "notes": "reference-only; 700 fixed rounds; honest 5-fold; best={}".format(
                    best_name
                ),
                "manifest_sha256": payload["manifest_sha256"],
                "team_folds_sha256": payload["team_folds_sha256"],
                "oof_path": oof_rel,
                "proba_path": proba_rel,
                "metrics_path": metrics_rel,
                "compliance_status": COMPLIANCE,
                "warnings": "none" if payload["quality_gate_pass"] else "below_0.80_quality_gate",
            },
            root=ROOT,
            overwrite=args.overwrite,
        )
    except RegistryV2Error as exc:
        print("registry_v2: {}".format(exc), flush=True)
        return 1

    print(
        "V2-B raw OOF={:.6f} +EI={:.6f} +Seg={} best={:.6f} ({})".format(
            payload["oof_accuracy_raw"],
            payload["oof_accuracy_ei"],
            payload["oof_accuracy_seg"],
            best_acc,
            best_name,
        ),
        flush=True,
    )
    print("wrote {}".format(metrics_rel), flush=True)
    print("quality gate: {}".format("PASS" if payload["quality_gate_pass"] else "STOP"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
