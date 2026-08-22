"""Sprint 3 OOF comparison of YW-002/003/004/005/007. Does not refit models."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

_WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORK / "src"))
from merfish60.io import repo_root  # noqa: E402
ROOT = repo_root()

from merfish60.io import load_dataset  # noqa: E402
from merfish60.official_contract import allowed_labels  # noqa: E402
from merfish60.signatures import missing_bucket_key, signatures_from_meta  # noqa: E402

YW004_OOF_REF = 0.7598
COMPARE_RUNS = ["YW-002", "YW-003", "YW-004", "YW-005", "YW-007"]
FORMAL_SELECTION_RUNS = ["YW-002", "YW-003", "YW-004", "YW-007"]
METHODS = {
    "YW-002": "gene + fold-safe Region/E-I/Segment one-hot + LR",
    "YW-003": "gene-only global LR + fold-safe candidate masking",
    "YW-004": "fold-safe per-signature logistic specialists",
    "YW-005": "exploratory ensemble diagnostic (not nested; excluded from selection)",
    "YW-007": "hybrid: YW-004 outside missing bucket; YW-006 inside",
}


def load_oof(run_id: str) -> pd.DataFrame:
    path = ROOT / "work" / "outputs" / "oof" / "{}_oof.csv".format(run_id)
    return pd.read_csv(
        path,
        dtype={"Cell_ID": str, "true_label": str, "predicted_label": str, "fold": int},
    )


def load_metrics(run_id: str) -> dict:
    path = ROOT / "work" / "outputs" / "metrics" / "{}_metrics.json".format(run_id)
    return json.loads(path.read_text())


def total_correct(oof: pd.DataFrame) -> int:
    return int((oof["true_label"].astype(str) == oof["predicted_label"].astype(str)).sum())


def missing_bucket_stats(oof: pd.DataFrame, sigs: pd.Series) -> dict:
    key = missing_bucket_key()
    aligned = sigs.loc[oof["Cell_ID"].astype(str)]
    mask = aligned.to_numpy() == key
    true = oof["true_label"].astype(str).to_numpy()
    pred = oof["predicted_label"].astype(str).to_numpy()
    inside_err = int(((true != pred) & mask).sum())
    outside_err = int(((true != pred) & ~mask).sum())
    inside_n = int(mask.sum())
    outside_n = int((~mask).sum())
    inside_acc = float((true[mask] == pred[mask]).mean()) if inside_n else None
    outside_acc = float((true[~mask] == pred[~mask]).mean()) if outside_n else None
    return {
        "signature": key,
        "n_inside": inside_n,
        "n_outside": outside_n,
        "errors_inside": inside_err,
        "errors_outside": outside_err,
        "accuracy_inside": inside_acc,
        "accuracy_outside": outside_acc,
    }


def confusion_top(oof: pd.DataFrame, k: int = 5) -> List[dict]:
    counts = Counter()
    for t, p in zip(oof["true_label"].astype(str), oof["predicted_label"].astype(str)):
        if t != p:
            counts[(t, p)] += 1
    return [
        {"true_label": t, "predicted_label": p, "n": int(n)}
        for (t, p), n in counts.most_common(k)
    ]


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Sprint 3 OOF comparison",
        "",
        "Frozen folds: `work/experiments/folds.csv`. Primary metric: overall accuracy.",
        "Sprint 2 runs YW-002/YW-003/YW-004 were not refit.",
        "YW-005 is an exploratory ensemble diagnostic (not nested; excluded from formal selection).",
        "YW-006 is a valid hard-bucket negative ablation, not a 5000-cell candidate.",
        "YW-007 is a valid hybrid that exactly duplicates YW-004 and is not selected.",
        "Formal Model V1 architecture: YW-004 (OOF accuracy 0.7598). Model V1 is not written in this sprint.",
        "",
        "| Run | Method | Fold 0 | Fold 1 | Fold 2 | OOF Accuracy | Delta vs YW-004 | Macro-F1 | Total Correct | Runtime | Eligible |",
        "| --- | ------ | -----: | -----: | -----: | -----------: | --------------: | -------: | ------------: | ------: | -------- |",
    ]
    for row in payload["table"]:
        lines.append(
            "| {run} | {method} | {f0:.4f} | {f1:.4f} | {f2:.4f} | {oof:.4f} | {delta:+.4f} | {f1m:.4f} | {tc} | {rt:.3f}s | {elig} |".format(
                run=row["run_id"],
                method=row["method"],
                f0=row["fold_0"],
                f1=row["fold_1"],
                f2=row["fold_2"],
                oof=row["oof_accuracy"],
                delta=row["delta_vs_YW-004"],
                f1m=row["macro_f1"],
                tc=row["total_correct"],
                rt=row["runtime_seconds"],
                elig="yes" if row.get("formal_selection_eligible", True) else "no",
            )
        )
    best = payload["formal_model_v1_decision"]
    miss = payload["remaining_errors"]
    lines.extend(
        [
            "",
            "## Formal Model V1 decision",
            "",
            "- selected architecture: {}".format(best["selected_run_id"]),
            "- selected OOF accuracy: {:.6f}".format(best["selected_oof_accuracy"]),
            "- YW-007 not selected: {}".format(best["excluded"]["YW-007"]),
            "- YW-005 not selected: {}".format(best["excluded"]["YW-005"]),
            "- YW-006 retained as: {}".format(best["excluded"]["YW-006"]),
            "- remaining total errors: {}".format(payload["best_experiment"]["remaining_total_errors"]),
            "- errors inside missing bucket: {}".format(payload["best_experiment"]["errors_inside_missing_bucket"]),
            "- errors outside missing bucket: {}".format(payload["best_experiment"]["errors_outside_missing_bucket"]),
            "",
            "## YW-005 exploratory diagnostic",
            "",
            "- role: exploratory ensemble diagnostic; excluded from formal selection",
            "- nested independence: false",
            "- equal-weight OOF accuracy: {}".format(
                payload["yw005_equal_weight_oof_accuracy"]
            ),
            "- cross-fitted OOF accuracy (exploratory): {}".format(
                payload["yw005_cross_fitted_oof_accuracy"]
            ),
            "- provenance: {}".format(payload["yw005_provenance_summary"]),
            "",
            "## YW-006 hard-bucket specialist (valid negative ablation)",
            "",
            "- hard-bucket OOF accuracy: {}".format(payload["yw006"]["oof_accuracy"]),
            "- selected model by outer fold: {}".format(payload["yw006"]["selected_model_by_outer_fold"]),
            "- delta vs YW-004 hard bucket: {}".format(payload["yw006"]["delta_vs_YW-004_hard_bucket"]),
            "",
            "## Remaining errors (YW-004)",
            "",
            "- from run: {}".format(miss["from_run"]),
            "- remaining total errors: {}".format(miss["remaining_total_errors"]),
            "- errors inside missing bucket: {}".format(miss["errors_inside_missing_bucket"]),
            "- errors outside missing bucket: {}".format(miss["errors_outside_missing_bucket"]),
            "",
            "### Five largest confusion pairs",
            "",
        ]
    )
    for item in miss["confusion_pairs_top5"]:
        lines.append(
            "- true={} pred={} n={}".format(item["true_label"], item["predicted_label"], item["n"])
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    data = load_dataset(ROOT)
    class_names = allowed_labels(ROOT)
    sigs = signatures_from_meta(data.meta_train)
    oofs: Dict[str, pd.DataFrame] = {rid: load_oof(rid) for rid in COMPARE_RUNS}
    mets = {rid: load_metrics(rid) for rid in COMPARE_RUNS}
    table_rows = []
    for rid in COMPARE_RUNS:
        m = mets[rid]
        oof_acc = float(m["oof_accuracy"])
        table_rows.append(
            {
                "run_id": rid,
                "method": METHODS[rid],
                "fold_0": float(m["fold_accuracy"]["0"]),
                "fold_1": float(m["fold_accuracy"]["1"]),
                "fold_2": float(m["fold_accuracy"]["2"]),
                "oof_accuracy": oof_acc,
                "delta_vs_YW-004": oof_acc - YW004_OOF_REF,
                "macro_f1": float(m["macro_f1"]),
                "total_correct": total_correct(oofs[rid]),
                "runtime_seconds": float(m["runtime_seconds"]),
                "formal_selection_eligible": rid in FORMAL_SELECTION_RUNS,
            }
        )
    # Formal selection is YW-004. YW-007 ties numerically but is a more complex duplicate.
    # YW-005 is excluded from ranking.
    best_id = "YW-004"
    best_row = next(r for r in table_rows if r["run_id"] == best_id)
    best_oof = oofs[best_id]
    bucket = missing_bucket_stats(best_oof, sigs)
    remaining = int((best_oof["true_label"] != best_oof["predicted_label"]).sum())
    yw004_correct = total_correct(oofs["YW-004"])
    m005 = mets["YW-005"]
    m006 = load_metrics("YW-006")
    payload = {
        "yw004_oof_accuracy_reference": YW004_OOF_REF,
        "n_classes": len(class_names),
        "table": table_rows,
        "yw005_equal_weight_oof_accuracy": m005["equal_weight"]["oof_accuracy"],
        "yw005_cross_fitted_oof_accuracy": m005["oof_accuracy"],
        "yw005_provenance_summary": (
            m005.get("provenance", {}).get("reason")
            or "saved first-level OOF files; not nested"
        ),
        "yw005_formal_selection_eligible": bool(m005.get("formal_selection_eligible", False)),
        "yw006": {
            "oof_accuracy": m006["oof_accuracy"],
            "selected_model_by_outer_fold": m006["selected_model_by_outer_fold"],
            "outer_fold_bucket_accuracy": m006["outer_fold_bucket_accuracy"],
            "delta_vs_YW-004_hard_bucket": m006["delta_vs_YW-004_hard_bucket"],
            "n_bucket_cells": m006["n_bucket_cells"],
            "role": m006.get("role", "valid_negative_ablation"),
        },
        "formal_model_v1_decision": {
            "selected_run_id": "YW-004",
            "selected_oof_accuracy": YW004_OOF_REF,
            "model_v1_written": False,
            "excluded": {
                "YW-005": (
                    "exploratory ensemble diagnostic; not nested; underperforms YW-004 "
                    "(OOF 0.7592 vs 0.7598)"
                ),
                "YW-006": "valid negative ablation; hard-bucket only; selected log1p_lr matching YW-004",
                "YW-007": "exact more-complex duplicate of YW-004; n_changed=0; not selected",
            },
        },
        "best_experiment": {
            "run_id": best_id,
            "method": METHODS[best_id],
            "oof_accuracy": best_row["oof_accuracy"],
            "improvement_over_0_7598": best_row["oof_accuracy"] - YW004_OOF_REF,
            "additional_correct_cells_vs_YW-004": best_row["total_correct"] - yw004_correct,
            "remaining_total_errors": remaining,
            "errors_inside_missing_bucket": bucket["errors_inside"],
            "errors_outside_missing_bucket": bucket["errors_outside"],
        },
        "remaining_errors": {
            "from_run": best_id,
            "remaining_total_errors": remaining,
            "errors_inside_missing_bucket": bucket["errors_inside"],
            "errors_outside_missing_bucket": bucket["errors_outside"],
            "confusion_pairs_top5": confusion_top(best_oof, k=5),
            "missing_bucket": bucket,
        },
    }
    out_json = ROOT / "work" / "outputs" / "metrics" / "sprint3_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(ROOT / "work" / "reports" / "sprint3_comparison.md", payload)
    print("Wrote {}".format(out_json))
    print("Wrote reports/sprint3_comparison.md")
    print("best={}".format(best_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
