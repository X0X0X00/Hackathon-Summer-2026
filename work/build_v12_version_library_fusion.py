"""Build a conservative V12 router from the historical model library.

V9 remains the anchor.  Older/alternative models may only change an anchor
prediction through an ordered source->target rule that is rediscovered across
outer folds.  This avoids treating an in-sample OOF rule sweep as proof of a
generalizable gain.
"""
from __future__ import annotations

import json
import sys
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
sys.path.insert(0, str(WORK))

from common import load  # noqa: E402
from common_ext import load_ext  # noqa: E402


DEFAULT_LZH_MODEL = ROOT / "submission_graph_stacker_82_50_20260820" / "model"


def norm(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    return values / values.sum(axis=1, keepdims=True)


def metrics(probabilities: np.ndarray, y: np.ndarray, folds: np.ndarray) -> dict:
    pred = probabilities.argmax(1)
    return {
        "correct": int(np.sum(pred == y)),
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, probabilities, labels=np.arange(60))),
        "fold_correct": [
            int(np.sum(pred[folds == fold] == y[folds == fold])) for fold in range(5)
        ],
    }


def load_library(
    data: dict,
    extended: dict,
    lzh_model: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train = np.where(data["is_train"])[0]
    test = np.where(~data["is_train"])[0]
    ext_train = np.where(extended["is_train"])[0]
    ext_test = np.where(extended["is_test"])[0]
    train_position = {
        str(cell_id): pos for pos, cell_id in enumerate(extended["ids"][ext_train])
    }
    test_position = {
        str(cell_id): pos for pos, cell_id in enumerate(extended["ids"][ext_test])
    }
    train_map = np.asarray([train_position[str(v)] for v in data["ids"][train]])
    test_map = np.asarray([test_position[str(v)] for v in data["ids"][test]])

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add_npz(name: str, path: Path, align_extended: bool = False) -> None:
        archive = np.load(path, allow_pickle=True)
        oof, test_prob = archive["oof"], archive["test"]
        if align_extended:
            oof, test_prob = oof[train_map], test_prob[test_map]
        result[name] = (norm(oof), norm(test_prob))

    add_npz("v1", WORK / "oof_ext" / "yhh_v1.npz")
    add_npz("bag8mix", WORK / "oof" / "bag8mix.npz")
    add_npz("refonly_full", WORK / "oof_ext" / "refonly_full.npz", True)
    add_npz("mlp3", WORK / "oof_ext" / "mlp3.npz", True)
    add_npz("v6", WORK / "final_probs.npz")
    add_npz("v7", WORK / "v7_final_probs.npz")
    add_npz("v8", WORK / "v8_final_probs.npz")
    add_npz("v85", WORK / "v85_final_probs.npz")
    add_npz("v85_classwise", WORK / "v85_classwise_stacker_probs.npz")
    add_npz("v85_knn", WORK / "v85_knn_probs.npz")
    add_npz("v9_pair_sweep", WORK / "v9_pair_sweep_probs.npz")
    add_npz("section_groupcv", WORK / "v11_section_groupcv_baseline_probs.npz")

    if lzh_model.is_dir():
        result["lzh"] = (
            norm(pd.read_csv(lzh_model / "oof_probabilities.csv").iloc[:, 1:].to_numpy()),
            norm(pd.read_csv(lzh_model / "test_probabilities.csv").iloc[:, 1:].to_numpy()),
        )

    # Probability-level consensus sources reduce single-model seed noise.
    consensus_members = {
        "consensus_versions": ["v6", "v7", "v8", "v85"],
        "consensus_strong": ["v7", "v8", "v85", "lzh"],
        "consensus_diverse": ["v1", "refonly_full", "mlp3", "lzh"],
    }
    for name, members in consensus_members.items():
        available = [member for member in members if member in result]
        result[name] = tuple(
            norm(np.mean([result[member][part] for member in available], axis=0))
            for part in (0, 1)
        )
    return result


PMINS = (0.0, 0.30, 0.40, 0.50, 0.60)
MARGIN_CAPS = (0.03, 0.05, 0.08, 0.12, 0.20, 0.40, 1.0)


def anchor_margin(probabilities: np.ndarray) -> np.ndarray:
    top = np.partition(probabilities, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def rule_mask(
    anchor: np.ndarray,
    expert: np.ndarray,
    source: int,
    target: int,
    pmin: float,
    margin_cap: float,
) -> np.ndarray:
    return (
        (anchor.argmax(1) == source)
        & (expert.argmax(1) == target)
        & (expert[:, target] >= pmin)
        & (anchor_margin(anchor) <= margin_cap)
    )


def enumerate_candidates(
    anchor: np.ndarray,
    experts: dict[str, np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    fit_folds: list[int],
    min_rate: float = 0.25,
) -> list[dict]:
    anchor_pred = anchor.argmax(1)
    margin = anchor_margin(anchor)
    fit = np.isin(folds, fit_folds)
    candidates = []
    for expert_name, expert in experts.items():
        expert_pred = expert.argmax(1)
        pairs = np.unique(np.column_stack([anchor_pred[fit], expert_pred[fit]]), axis=0)
        for source, target in pairs:
            source, target = int(source), int(target)
            if source == target:
                continue
            base = (anchor_pred == source) & (expert_pred == target)
            for pmin in PMINS:
                for margin_cap in MARGIN_CAPS:
                    mask = (
                        base
                        & (expert[:, target] >= pmin)
                        & (margin <= margin_cap)
                    )
                    selected = int(np.sum(mask & fit))
                    if selected < 3:
                        continue
                    fold_delta = []
                    for fold in fit_folds:
                        rows = mask & (folds == fold)
                        fold_delta.append(
                            int(np.sum(expert_pred[rows] == y[rows]))
                            - int(np.sum(anchor_pred[rows] == y[rows]))
                        )
                    gain = int(sum(fold_delta))
                    if (
                        gain >= 2
                        and min(fold_delta) >= 0
                        and sum(value > 0 for value in fold_delta) >= 2
                        and gain / selected >= min_rate
                    ):
                        candidates.append(
                            {
                                "expert": expert_name,
                                "source": source,
                                "target": target,
                                "pmin": pmin,
                                "margin_cap": margin_cap,
                                "gain": gain,
                                "selected": selected,
                                "fold_delta": fold_delta,
                                "rate": gain / selected,
                            }
                        )
    candidates.sort(
        key=lambda row: (row["gain"], row["rate"], -row["selected"]), reverse=True
    )
    # One threshold configuration per expert/source/target pair.
    best = {}
    for row in candidates:
        key = (row["expert"], row["source"], row["target"])
        best.setdefault(key, row)
    return list(best.values())


def apply_rules(
    anchor: np.ndarray,
    experts: dict[str, np.ndarray],
    rules: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = anchor.copy()
    selected = np.zeros(len(anchor), dtype=bool)
    selected_rule = np.full(len(anchor), -1, dtype=int)
    anchor_pred = anchor.argmax(1)
    margin = anchor_margin(anchor)
    expert_pred = {name: values.argmax(1) for name, values in experts.items()}
    for rule_index, rule in enumerate(rules):
        expert = experts[rule["expert"]]
        mask = (
            (anchor_pred == rule["source"])
            & (expert_pred[rule["expert"]] == rule["target"])
            & (expert[:, rule["target"]] >= rule["pmin"])
            & (margin <= rule["margin_cap"])
        )
        take = mask & ~selected
        # Route the full probability vector so submission argmax and confidence agree.
        out[take] = experts[rule["expert"]][take]
        selected[take] = True
        selected_rule[take] = rule_index
    return norm(out), selected, selected_rule


def greedy_rules(
    anchor: np.ndarray,
    experts: dict[str, np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    fit_folds: list[int],
    allowed_keys: set[tuple[str, int, int]] | None = None,
    max_rules: int = 12,
    min_rate: float = 0.25,
) -> list[dict]:
    candidates = enumerate_candidates(
        anchor, experts, y, folds, fit_folds, min_rate=min_rate
    )
    if allowed_keys is not None:
        candidates = [
            row
            for row in candidates
            if (row["expert"], row["source"], row["target"]) in allowed_keys
        ]
    chosen: list[dict] = []
    current = anchor.copy()
    current_pred = current.argmax(1)
    fit = np.isin(folds, fit_folds)
    for row in candidates:
        trial, selected, _ = apply_rules(anchor, experts, chosen + [row])
        trial_pred = trial.argmax(1)
        incremental = int(np.sum(trial_pred[fit] == y[fit])) - int(
            np.sum(current_pred[fit] == y[fit])
        )
        fold_incremental = [
            int(np.sum(trial_pred[folds == fold] == y[folds == fold]))
            - int(np.sum(current_pred[folds == fold] == y[folds == fold]))
            for fold in fit_folds
        ]
        if incremental >= 2 and min(fold_incremental) >= 0 and selected.any():
            chosen.append(row)
            current, current_pred = trial, trial_pred
        if len(chosen) >= max_rules:
            break
    return chosen


def serialize_rule(rule: dict, labels: np.ndarray) -> dict:
    result = dict(rule)
    result["source"] = str(labels[rule["source"]])
    result["target"] = str(labels[rule["target"]])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lzh-model-dir",
        type=Path,
        default=DEFAULT_LZH_MODEL,
        help="directory containing lzh oof_probabilities.csv and test_probabilities.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, extended = load(), load_ext()
    train = np.where(data["is_train"])[0]
    test = np.where(~data["is_train"])[0]
    y = data["y"][train].astype(int)
    folds = data["folds"][train].astype(int)
    labels = np.asarray(data["labels"]).astype(str)
    anchor_archive = np.load(WORK / "v9_final_probs.npz")
    anchor_oof, anchor_test = norm(anchor_archive["oof"]), norm(anchor_archive["test"])
    library = load_library(data, extended, args.lzh_model_dir)
    expert_oof = {name: values[0] for name, values in library.items()}
    expert_test = {name: values[1] for name, values in library.items()}

    inventory = {
        name: {
            **metrics(oof, y, folds),
            "test_disagreement_vs_v9": int(
                np.sum(test_prob.argmax(1) != anchor_test.argmax(1))
            ),
        }
        for name, (oof, test_prob) in library.items()
    }

    # Honest outer-fold audit: rules are learned on four folds and applied to the fifth.
    nested = anchor_oof.copy()
    nested_rule_keys: list[tuple[str, int, int]] = []
    key_holdout_delta: dict[tuple[str, int, int], list[int]] = {}
    outer_rules: list[list[dict]] = []
    nested_rules_by_holdout = []
    for holdout in range(5):
        fit_folds = [fold for fold in range(5) if fold != holdout]
        rules = greedy_rules(anchor_oof, expert_oof, y, folds, fit_folds)
        outer_rules.append(rules)
        routed, _, _ = apply_rules(anchor_oof, expert_oof, rules)
        nested[folds == holdout] = routed[folds == holdout]
        nested_rule_keys.extend(
            (row["expert"], row["source"], row["target"]) for row in rules
        )
        holdout_rows = folds == holdout
        for row in rules:
            key = (row["expert"], row["source"], row["target"])
            single, _, _ = apply_rules(anchor_oof, expert_oof, [row])
            delta = int(np.sum(single.argmax(1)[holdout_rows] == y[holdout_rows])) - int(
                np.sum(anchor_oof.argmax(1)[holdout_rows] == y[holdout_rows])
            )
            key_holdout_delta.setdefault(key, []).append(delta)
        nested_rules_by_holdout.append(
            {
                "holdout": holdout,
                "rule_count": len(rules),
                "rules": [serialize_rule(row, labels) for row in rules],
            }
        )

    key_counts = Counter(nested_rule_keys)
    # A key must be repeatedly selected and must also work on the folds that did
    # not participate in selecting it.  This removes the large full-OOF mirage.
    stable_keys = {
        key
        for key, count in key_counts.items()
        if count >= 3
        and sum(key_holdout_delta.get(key, [])) > 0
        and min(key_holdout_delta.get(key, [0])) >= 0
    }
    stable_nested = anchor_oof.copy()
    stable_nested_rule_counts = []
    for holdout, rules in enumerate(outer_rules):
        stable_rules = [
            row
            for row in rules
            if (row["expert"], row["source"], row["target"]) in stable_keys
        ]
        routed, _, _ = apply_rules(anchor_oof, expert_oof, stable_rules)
        stable_nested[folds == holdout] = routed[folds == holdout]
        stable_nested_rule_counts.append(len(stable_rules))
    final_rules = greedy_rules(
        anchor_oof,
        expert_oof,
        y,
        folds,
        list(range(5)),
        allowed_keys=stable_keys,
        min_rate=0.20,
    )
    final_oof, selected_oof, selected_rule = apply_rules(
        anchor_oof, expert_oof, final_rules
    )
    final_test, selected_test, selected_test_rule = apply_rules(
        anchor_test, expert_test, final_rules
    )
    probability_path = WORK / "v12_final_probs.npz"
    np.savez_compressed(probability_path, oof=final_oof, test=final_test)

    anchor_metric = metrics(anchor_oof, y, folds)
    nested_metric = metrics(nested, y, folds)
    stable_nested_metric = metrics(stable_nested, y, folds)
    final_metric = metrics(final_oof, y, folds)
    nested_metric["delta_correct"] = nested_metric["correct"] - anchor_metric["correct"]
    nested_metric["fold_delta"] = [
        nested_metric["fold_correct"][fold] - anchor_metric["fold_correct"][fold]
        for fold in range(5)
    ]
    stable_nested_metric["delta_correct"] = (
        stable_nested_metric["correct"] - anchor_metric["correct"]
    )
    stable_nested_metric["fold_delta"] = [
        stable_nested_metric["fold_correct"][fold]
        - anchor_metric["fold_correct"][fold]
        for fold in range(5)
    ]
    stable_nested_metric["rule_count_by_holdout"] = stable_nested_rule_counts
    final_metric["delta_correct"] = final_metric["correct"] - anchor_metric["correct"]
    final_metric["fold_delta"] = [
        final_metric["fold_correct"][fold] - anchor_metric["fold_correct"][fold]
        for fold in range(5)
    ]

    target_column = pd.read_csv(
        ROOT / "prediction" / "prediction_v9.csv", nrows=1
    ).columns[1]
    output_path = ROOT / "prediction" / "prediction_v12_version_router.csv"
    pd.DataFrame(
        {
            "Cell_ID": data["ids"][test].astype(str),
            target_column: labels[final_test.argmax(1)],
        }
    ).to_csv(output_path, index=False)

    changes = []
    for row in np.where(selected_test)[0]:
        rule = final_rules[selected_test_rule[row]]
        changes.append(
            {
                "Cell_ID": str(data["ids"][test][row]),
                "v9": str(labels[anchor_test[row].argmax()]),
                "v12": str(labels[final_test[row].argmax()]),
                "rule": serialize_rule(rule, labels),
            }
        )

    report = {
        "name": "V12 historical version library router",
        "protocol": (
            "V9 anchor; outer-fold rule discovery; final rules restricted to ordered "
            "expert/source/target keys rediscovered in at least 3 of 5 outer fits."
        ),
        "anchor_v9": anchor_metric,
        "inventory": inventory,
        "nested_exploratory_audit": nested_metric,
        "nested_stable_rule_audit": stable_nested_metric,
        "nested_rules_by_holdout": nested_rules_by_holdout,
        "stable_key_counts": {
            f"{expert}|{labels[source]}|{labels[target]}": {
                "selection_count": count,
                "holdout_delta": key_holdout_delta.get((expert, source, target), []),
            }
            for (expert, source, target), count in sorted(key_counts.items())
            if (expert, source, target) in stable_keys
        },
        "candidate": final_metric,
        "selected_oof": int(selected_oof.sum()),
        "selected_test": int(selected_test.sum()),
        "test_changes_vs_v9": int(
            np.sum(final_test.argmax(1) != anchor_test.argmax(1))
        ),
        "final_rules": [serialize_rule(row, labels) for row in final_rules],
        "changes": changes,
        "output": str(output_path),
        "probabilities": str(probability_path),
        "caveat": (
            "The official test labels are hidden. Nested OOF stability is evidence, not a "
            "guarantee of leaderboard improvement."
        ),
    }
    report_path = WORK / "v12_version_library_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "anchor": anchor_metric,
        "nested": nested_metric,
        "stable_nested": stable_nested_metric,
        "candidate": final_metric,
        "stable_keys": len(stable_keys),
        "rules": report["final_rules"],
        "test_changes": report["test_changes_vs_v9"],
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
