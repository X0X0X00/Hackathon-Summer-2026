"""V2-C: score five predeclared blends of saved expert probabilities.

Does not train LightGBM, logistic regression, or any other predictor.
Does not search blend weights. Does not rewrite the official example submission.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from merfish60.io import load_dataset, validate_contract  # noqa: E402
from merfish60.metrics import summarize_oof  # noqa: E402
from merfish60.models import argmax_labels, assert_probability_rows  # noqa: E402
from merfish60.v2c_blend import (  # noqa: E402
    BLEND_IDS,
    EXPERT_A_OOF,
    EXPERT_A_TEST,
    EXPERT_B_OOF,
    EXPERT_B_TEST,
    EXPERT_C_OOF,
    EXPERT_C_TEST,
    FIXED_BLENDS,
    V2B_CORRECT_TARGET,
    V2B_OOF_TARGET,
    V2CAlignmentError,
    assert_oof_alignment,
    assert_test_alignment,
    cell_delta,
    complementarity,
    confusion_shift,
    fold_deltas,
    load_proba_frame,
    mix_fixed,
    score_blend,
    select_model_v2,
    write_candidate_csv,
)
from merfish60.official_contract import (  # noqa: E402
    allowed_labels,
    expected_test_cell_ids,
    git_commit,
    manifest_sha256,
    verify_official_manifest,
)
from merfish60.spatial_features import ei_of_label_from_train  # noqa: E402
from merfish60.team_cv import team_folds_path, load_team_folds, team_folds_sha256  # noqa: E402
from merfish60.v2_metrics import (  # noqa: E402
    hard_bucket_mask,
    neuron_glial_masks,
    write_confusion,
    write_json,
    write_oof,
    write_proba,
)
from merfish60.validate_submission import SubmissionContractError, validate_submission  # noqa: E402


BRIDGE_OOF = 0.7596
V2A_EI_OOF = 0.7690
MODEL_V1_OOF = 0.7598
CANDIDATE_REL = "outputs/submissions/model_v2_candidate.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _public_score(row: dict) -> dict:
    return {
        "weights": row["weights"],
        "fold_accuracy": row["fold_accuracy"],
        "oof_accuracy": row["oof_accuracy"],
        "macro_f1": row["macro_f1"],
        "correct": row["correct"],
        "wrong": row["wrong"],
        "hard_bucket_accuracy": row["slices"]["hard_bucket_accuracy"],
        "neuron_accuracy": row["slices"]["neuron_accuracy"],
        "glial_accuracy": row["slices"]["glial_accuracy"],
        "delta_vs_bridge": row["oof_accuracy"] - BRIDGE_OOF,
        "delta_vs_v2a": row["oof_accuracy"] - V2A_EI_OOF,
        "delta_vs_v2b": row["oof_accuracy"] - V2B_OOF_TARGET,
        "vs_c0": row.get("vs_c0"),
        "stability": row.get("stability"),
        "selection_eligible": row.get("selection_eligible"),
        "confusion_pairs_top10": row["confusion_pairs_top10"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    print("Verifying official data manifest...", flush=True)
    for line in verify_official_manifest(ROOT):
        print(" - {}".format(line), flush=True)

    data = load_dataset(ROOT)
    for line in validate_contract(data):
        print(" - {}".format(line), flush=True)
    class_names = allowed_labels(ROOT)
    folds = load_team_folds(team_folds_path(ROOT))
    train_ids = [str(v) for v in data.counts_train.index.tolist()]
    test_ids = expected_test_cell_ids(ROOT)
    y_true = data.y_train.loc[train_ids].astype(str).to_numpy()
    fold_ids = folds.set_index("Cell_ID").loc[train_ids, "fold"].to_numpy()
    ei_of_label = ei_of_label_from_train(data.meta_train, class_names)
    meta_train = data.meta_train.loc[train_ids]

    oof_frames = {
        "A": load_proba_frame(ROOT / EXPERT_A_OOF, class_names),
        "B": load_proba_frame(ROOT / EXPERT_B_OOF, class_names),
        "C": load_proba_frame(ROOT / EXPERT_C_OOF, class_names),
    }
    test_frames = {
        "A": load_proba_frame(ROOT / EXPERT_A_TEST, class_names),
        "B": load_proba_frame(ROOT / EXPERT_B_TEST, class_names),
        "C": load_proba_frame(ROOT / EXPERT_C_TEST, class_names),
    }
    try:
        oof_msgs = assert_oof_alignment(oof_frames, train_ids, folds, class_names)
        test_msgs = assert_test_alignment(test_frames, test_ids, class_names)
    except V2CAlignmentError as exc:
        print("STOP: alignment failed: {}".format(exc), flush=True)
        return 2
    for line in oof_msgs + test_msgs:
        print(" - {}".format(line), flush=True)

    pa = oof_frames["A"][class_names].to_numpy(dtype=np.float64)
    pb = oof_frames["B"][class_names].to_numpy(dtype=np.float64)
    pc = oof_frames["C"][class_names].to_numpy(dtype=np.float64)
    ta = test_frames["A"][class_names].to_numpy(dtype=np.float64)
    tb = test_frames["B"][class_names].to_numpy(dtype=np.float64)
    tc = test_frames["C"][class_names].to_numpy(dtype=np.float64)

    pred_a = argmax_labels(pa, class_names)
    pred_b = argmax_labels(pb, class_names)
    pred_c = argmax_labels(pc, class_names)
    ok_a = pred_a == y_true
    ok_b = pred_b == y_true
    ok_c = pred_c == y_true

    experts = {}
    for key, pred, ok in (("A", pred_a, ok_a), ("B", pred_b, ok_b), ("C", pred_c, ok_c)):
        metrics = summarize_oof(y_true, pred, fold_ids, labels=class_names)
        experts[key] = {
            "oof_accuracy": metrics["oof_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "correct": int(ok.sum()),
            "wrong": int((~ok).sum()),
            "fold_accuracy": {str(k): v for k, v in metrics["fold_accuracy"].items()},
        }
    if abs(experts["C"]["oof_accuracy"] - V2B_OOF_TARGET) > 1e-6:
        print(
            "STOP: expert C OOF {} != validated V2-B {}".format(
                experts["C"]["oof_accuracy"], V2B_OOF_TARGET
            ),
            flush=True,
        )
        return 2

    hard_mask = hard_bucket_mask(meta_train)
    neuron_mask, glial_mask = neuron_glial_masks(y_true, class_names, ei_of_label)
    comp_all = complementarity(ok_a, ok_b, ok_c, pred_a, pred_b, pred_c)
    comp_slices = {}
    for slice_name, mask in (
        ("hard_bucket", hard_mask),
        ("neuron", neuron_mask),
        ("glial", glial_mask),
    ):
        comp_slices[slice_name] = complementarity(
            ok_a[mask], ok_b[mask], ok_c[mask], pred_a[mask], pred_b[mask], pred_c[mask]
        )
        comp_slices[slice_name]["n_slice"] = int(mask.sum())

    scoreboard = {}
    blend_probas = {}
    for cand_id in BLEND_IDS:
        weights = FIXED_BLENDS[cand_id]
        mixed = mix_fixed(pa, pb, pc, weights)
        assert_probability_rows(mixed)
        row = score_blend(mixed, y_true, fold_ids, meta_train, class_names, ei_of_label)
        row["weights"] = {"A": weights[0], "B": weights[1], "C": weights[2]}
        blend_probas[cand_id] = mixed
        scoreboard[cand_id] = row
    c0 = scoreboard["C0"]
    if abs(c0["oof_accuracy"] - V2B_OOF_TARGET) > 1e-6 or c0["correct"] != V2B_CORRECT_TARGET:
        print(
            "STOP: C0 did not reproduce V2-B (acc={} correct={})".format(
                c0["oof_accuracy"], c0["correct"]
            ),
            flush=True,
        )
        return 2
    c0_folds = {int(k): v for k, v in c0["fold_accuracy"].items()}
    for cand_id in ("C1", "C2", "C3", "C4"):
        row = scoreboard[cand_id]
        row["vs_c0"] = cell_delta(c0["pred"], row["pred"], y_true)
        row["stability"] = fold_deltas(
            {int(k): v for k, v in row["fold_accuracy"].items()}, c0_folds
        )

    selection = select_model_v2(scoreboard)
    selected_id = selection["selected_id"]
    selected = scoreboard[selected_id]
    print("selected {} OOF={:.4f}".format(selected_id, selected["oof_accuracy"]), flush=True)
    print("reason: {}".format(selection["reason"]), flush=True)

    strongest_blend = max(
        ("C1", "C2", "C3", "C4"), key=lambda cid: scoreboard[cid]["oof_accuracy"]
    )
    error = confusion_shift(y_true, c0["pred"], scoreboard[strongest_blend]["pred"])

    weights = FIXED_BLENDS[selected_id]
    test_mixed = mix_fixed(ta, tb, tc, weights)
    assert_probability_rows(test_mixed)
    test_pred = argmax_labels(test_mixed, class_names)
    candidate_path = ROOT / CANDIDATE_REL
    write_candidate_csv(candidate_path, test_ids, test_pred)
    try:
        val_msgs = validate_submission(candidate_path, ROOT)
    except SubmissionContractError as exc:
        print("STOP: candidate failed submission contract", flush=True)
        for line in exc.violations:
            print(" - {}".format(line), flush=True)
        return 2

    write_oof(
        ROOT / "outputs/oof/MODEL-V2_oof.csv",
        train_ids,
        y_true,
        selected["pred"],
        fold_ids,
    )
    write_proba(
        ROOT / "outputs/probabilities/MODEL-V2_oof_probabilities.csv.gz",
        train_ids,
        blend_probas[selected_id],
        class_names,
    )
    write_proba(
        ROOT / "outputs/probabilities/MODEL-V2_test_probabilities.csv.gz",
        test_ids,
        test_mixed,
        class_names,
    )
    write_confusion(
        ROOT / "outputs/metrics/MODEL-V2_confusion.csv",
        y_true,
        selected["pred"],
        class_names,
    )

    public_board = {cid: _public_score(scoreboard[cid]) for cid in BLEND_IDS}
    write_json(
        ROOT / "outputs/metrics/V2-C-scoreboard.json",
        {
            "expert_files": {
                "A_oof": EXPERT_A_OOF,
                "A_test": EXPERT_A_TEST,
                "B_oof": EXPERT_B_OOF,
                "B_test": EXPERT_B_TEST,
                "C_oof": EXPERT_C_OOF,
                "C_test": EXPERT_C_TEST,
            },
            "fixed_blends": {cid: list(FIXED_BLENDS[cid]) for cid in BLEND_IDS},
            "experts": experts,
            "scoreboard": public_board,
            "selected_id": selected_id,
            "selection_reason": selection["reason"],
            "no_weight_search": True,
            "timestamp": utc_now(),
        },
    )
    write_json(
        ROOT / "outputs/metrics/V2-C-complementarity.json",
        {
            "headroom_only": True,
            "not_a_model_score": True,
            "all": comp_all,
            "slices": comp_slices,
            "error_analysis_c0_vs_strongest_blend": {
                "strongest_blend": strongest_blend,
                **error,
            },
        },
    )
    write_json(
        ROOT / "outputs/metrics/model_v2_selection.json",
        {
            "model_name": "MODEL V2",
            "selected_id": selected_id,
            "architecture": (
                "V2-B reference-only LightGBM"
                if selected_id == "C0"
                else "fixed-weight blend of BRIDGE + V2-A + V2-B"
            ),
            "weights": selected["weights"],
            "component_probability_files": {
                "A_oof": EXPERT_A_OOF,
                "A_test": EXPERT_A_TEST if selected["weights"]["A"] else None,
                "B_oof": EXPERT_B_OOF,
                "B_test": EXPERT_B_TEST if selected["weights"]["B"] else None,
                "C_oof": EXPERT_C_OOF,
                "C_test": EXPERT_C_TEST,
            },
            "cv_protocol": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "oof_accuracy": selected["oof_accuracy"],
            "correct": selected["correct"],
            "macro_f1": selected["macro_f1"],
            "fold_accuracy": selected["fold_accuracy"],
            "hard_bucket_accuracy": selected["slices"]["hard_bucket_accuracy"],
            "neuron_accuracy": selected["slices"]["neuron_accuracy"],
            "glial_accuracy": selected["slices"]["glial_accuracy"],
            "delta_vs_model_v1": selected["oof_accuracy"] - MODEL_V1_OOF,
            "delta_vs_bridge": selected["oof_accuracy"] - BRIDGE_OOF,
            "delta_vs_v2a": selected["oof_accuracy"] - V2A_EI_OOF,
            "delta_vs_v2b": selected["oof_accuracy"] - V2B_OOF_TARGET,
            "official_leaderboard_score": "Not submitted",
            "candidate_path": CANDIDATE_REL,
            "git_commit": git_commit(ROOT),
            "manifest_sha256": manifest_sha256(ROOT),
            "team_folds_sha256": team_folds_sha256(ROOT),
            "selection_reason": selection["reason"],
            "no_weight_search": True,
            "timestamp": utc_now(),
        },
    )

    print("candidate: {}".format(CANDIDATE_REL), flush=True)
    for line in val_msgs:
        print(" - {}".format(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
