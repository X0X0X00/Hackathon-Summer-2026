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


BASE_PROBABILITIES = OUT / "pair_experts_80_probabilities.npz"
EXTERNAL_REFERENCE = Path(__file__).resolve().parents[1] / "cache" / "external_neuronal_reference.npz"


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

    families = {
        "midventral_inhibitory": [name for name in encoder.classes_ if name.startswith("MV_in_")],
        "midventral_excitatory": [name for name in encoder.classes_ if name.startswith("MV_ex_")],
        "medial_excitatory": [name for name in encoder.classes_ if name.startswith("M_ex_")],
        "cholinergic_pair": ["VH_in_Chat", "cholinergic_interneuron"],
    }
    external = np.load(EXTERNAL_REFERENCE, allow_pickle=False)
    external_features = external["features"].astype(np.float32)
    external_labels = external["labels"].astype(str)
    stored = np.load(BASE_PROBABILITIES)
    current_oof = stored["oof"].astype(np.float32).copy()
    current_test = stored["test"].astype(np.float32).copy()
    baseline_oof = current_oof.copy()
    baseline_test_prediction = current_test.argmax(axis=1)
    report: dict = {"baseline": metrics(y, current_oof), "families": {}}

    for family_name, family_labels in families.items():
        specialist_oof, specialist_test, family_indices, training = fit_family_expert(
            family_name,
            family_labels,
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
        report["families"][family_name] = {
            "labels": family_labels,
            "training": training,
            "gate": gate,
            "test_cells_selected": int(selected_test.sum()),
            "cumulative_metrics": metrics(y, current_oof),
        }
        print(json.dumps({"stage": "ventral_selected", "family": family_name, **report["families"][family_name]}, indent=2), flush=True)

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
    (OUT / "ventral_experts_80_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "ventral_experts_80_probabilities.npz", oof=current_oof, test=current_test, y=y
    )
    sample = pd.read_csv(OFFICIAL / "prediction" / "prediction.csv")
    sample.iloc[:, 1] = encoder.inverse_transform(current_test.argmax(axis=1))
    sample.to_csv(OUT / "prediction_ventral_experts_80.csv", index=False)
    print(json.dumps({"stage": "final", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
