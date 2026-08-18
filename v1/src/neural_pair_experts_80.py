from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from hierarchical_glia import build_internal_features, redistribute_family
from neuronal_experts import fit_family_expert, metrics
from pair_experts_80 import strict_gate
from train_model import LABEL, N_FOLDS, OFFICIAL, OUT, SEED, load_data


BASE_PROBABILITIES = OUT / "ventral_experts_80_probabilities.npz"
EXTERNAL_REFERENCE = Path(__file__).resolve().parents[1] / "cache" / "external_neuronal_reference.npz"
PAIR_SPECS = [
    ("dh_in_klhl14_vs_cdh3", ["DH_in_Klhl14", "DH_in_Cdh3"]),
    ("dh_ex_maf_slc17a8_vs_maf_cck", ["DH_ex_Maf/Slc17a8", "DH_ex_Maf/Cck"]),
    ("dh_ex_grp_vs_prkcg_rxfp1", ["DH_ex_Grp", "DH_ex_Prkcg/Rxfp1"]),
    ("dh_ex_prkcg_cck_vs_nts", ["DH_ex_Prkcg/Cck", "DH_ex_Prkcg/Nts"]),
    ("dh_ex_gpr83_vs_grpr", ["DH_ex_Gpr83", "DH_ex_Grpr"]),
    ("dh_ex_rreb1_vs_maf_cck", ["DH_ex_Rreb1", "DH_ex_Maf/Cck"]),
    ("dh_ex_tac2_vs_prkcg_nts", ["DH_ex_Tac2", "DH_ex_Prkcg/Nts"]),
    ("dh_ex_cpne4_vs_reln_nmur2", ["DH_ex_Cpne4", "DH_ex_Reln/Nmur2"]),
    ("mv_in_chrna2_vs_gm26673", ["MV_in_Chrna2", "MV_in_Gm26673"]),
    ("motor_alpha_vs_beta", ["alpha_motoneuron", "beta_motoneuron"]),
    ("motor_gamma_vs_alpha", ["gamma_motoneuron", "alpha_motoneuron"]),
]


def main() -> None:
    counts_train, counts_test, meta_train, meta_test = load_data()
    encoder = LabelEncoder().fit(meta_train[LABEL].astype(str))
    y = encoder.transform(meta_train[LABEL].astype(str))
    expr_train, expr_test, full_train, full_test = build_internal_features(
        counts_train, counts_test, meta_train, meta_test
    )
    splits = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(full_train, y)
    )
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, validation) in enumerate(splits):
        fold_ids[validation] = fold

    external = np.load(EXTERNAL_REFERENCE, allow_pickle=False)
    external_features = external["features"].astype(np.float32)
    external_labels = external["labels"].astype(str)
    stored = np.load(BASE_PROBABILITIES)
    current_oof = stored["oof"].astype(np.float32).copy()
    current_test = stored["test"].astype(np.float32).copy()
    baseline_oof = current_oof.copy()
    baseline_test_prediction = current_test.argmax(axis=1)
    report: dict = {"baseline": metrics(y, current_oof), "pairs": {}}

    for pair_name, labels in PAIR_SPECS:
        specialist_oof, specialist_test, family_indices, training = fit_family_expert(
            pair_name,
            labels,
            encoder,
            y,
            expr_train,
            expr_test,
            full_train,
            full_test,
            splits,
            external_features,
            external_labels,
        )
        gate, current_oof = strict_gate(
            current_oof, specialist_oof, family_indices, y, fold_ids
        )
        current_test, selected_test = redistribute_family(
            current_test,
            specialist_test,
            family_indices,
            gate["gate"],
            gate["confidence_cap"],
            gate["weight"],
        )
        report["pairs"][pair_name] = {
            "labels": labels,
            "training": training,
            "gate": gate,
            "test_cells_selected": int(selected_test.sum()),
            "cumulative_metrics": metrics(y, current_oof),
        }
        print(json.dumps({"stage": "neural_pair_selected", "pair": pair_name, **report["pairs"][pair_name]}, indent=2), flush=True)

    baseline_prediction = baseline_oof.argmax(axis=1)
    final_prediction = current_oof.argmax(axis=1)
    report["final"] = metrics(y, current_oof)
    report["delta_correct"] = int(
        (final_prediction == y).sum() - (baseline_prediction == y).sum()
    )
    report["fold_delta_correct"] = [
        int(
            (final_prediction[fold_ids == fold] == y[fold_ids == fold]).sum()
            - (baseline_prediction[fold_ids == fold] == y[fold_ids == fold]).sum()
        )
        for fold in range(N_FOLDS)
    ]
    report["test_label_changes"] = int(
        (current_test.argmax(axis=1) != baseline_test_prediction).sum()
    )
    (OUT / "neural_pair_experts_80_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "neural_pair_experts_80_probabilities.npz",
        oof=current_oof,
        test=current_test,
        y=y,
    )
    sample = pd.read_csv(OFFICIAL / "prediction" / "prediction.csv")
    sample.iloc[:, 1] = encoder.inverse_transform(current_test.argmax(axis=1))
    sample.to_csv(OUT / "prediction_neural_pair_experts_80.csv", index=False)
    print(json.dumps({"stage": "final", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
