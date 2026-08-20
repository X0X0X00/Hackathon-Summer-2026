"""Paired OOF comparison of YW-001 through YW-004. Does not refit models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from merfish60.io import load_dataset  # noqa: E402
from merfish60.official_contract import allowed_labels  # noqa: E402
from merfish60.signatures import missing_bucket_key, signatures_from_meta  # noqa: E402


def load_oof(run_id: str) -> pd.DataFrame:
    path = ROOT / "outputs" / "oof" / "{}_oof.csv".format(run_id)
    df = pd.read_csv(path, dtype={"Cell_ID": str, "true_label": str, "predicted_label": str, "fold": int})
    return df


def load_metrics(run_id: str) -> dict:
    path = ROOT / "outputs" / "metrics" / "{}_metrics.json".format(run_id)
    return json.loads(path.read_text())


def paired(base: pd.DataFrame, other: pd.DataFrame) -> dict:
    merged = base.merge(other, on="Cell_ID", suffixes=("_base", "_new"))
    if len(merged) != 5000:
        raise SystemExit("paired comparison n={} != 5000".format(len(merged)))
    true = merged["true_label_base"].astype(str)
    a = merged["predicted_label_base"].astype(str)
    b = merged["predicted_label_new"].astype(str)
    both_correct = int(((a == true) & (b == true)).sum())
    both_wrong = int(((a != true) & (b != true)).sum())
    w2c = int(((a != true) & (b == true)).sum())
    c2w = int(((a == true) & (b != true)).sum())
    acc_base = float((a == true).mean())
    acc_new = float((b == true).mean())
    return {
        "n": int(len(merged)),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "base_wrong_to_new_correct": w2c,
        "base_correct_to_new_wrong": c2w,
        "net_additional_correct": w2c - c2w,
        "base_accuracy": acc_base,
        "new_accuracy": acc_new,
        "delta_accuracy": acc_new - acc_base,
    }


def fold_direction(base_m: dict, new_m: dict) -> Dict[str, str]:
    out = {}
    for k in ("0", "1", "2"):
        b = float(base_m["fold_accuracy"][k])
        n = float(new_m["fold_accuracy"][k])
        if n > b:
            out[k] = "improved"
        elif n < b:
            out[k] = "worsened"
        else:
            out[k] = "unchanged"
    return out


def missing_bucket_acc(oof: pd.DataFrame, sigs: pd.Series) -> dict:
    key = missing_bucket_key()
    aligned = sigs.loc[oof["Cell_ID"].astype(str)]
    mask = aligned.to_numpy() == key
    if not mask.any():
        return {"n": 0, "accuracy": None}
    acc = float((oof.loc[mask, "true_label"] == oof.loc[mask, "predicted_label"]).mean())
    return {"n": int(mask.sum()), "accuracy": acc, "signature": key}


def write_markdown(path: Path, table_rows: List[dict], payload: dict) -> None:
    lines = [
        "# Sprint 2 OOF comparison",
        "",
        "Frozen folds: `experiments/folds.csv`. Primary metric: overall accuracy.",
        "YW-001 was not refit. YW-002/003/004 are experiments, not formal model versions.",
        "",
        "| Run | Method | Fold 0 | Fold 1 | Fold 2 | OOF Accuracy | Delta vs YW-001 | Macro-F1 | Runtime |",
        "| --- | ------ | -----: | -----: | -----: | -----------: | --------------: | -------: | ------: |",
    ]
    for row in table_rows:
        lines.append(
            "| {run} | {method} | {f0:.4f} | {f1:.4f} | {f2:.4f} | {oof:.4f} | {delta:+.4f} | {f1m:.4f} | {rt:.3f}s |".format(
                run=row["run_id"],
                method=row["method"],
                f0=row["fold_0"],
                f1=row["fold_1"],
                f2=row["fold_2"],
                oof=row["oof_accuracy"],
                delta=row["delta_vs_yw001"],
                f1m=row["macro_f1"],
                rt=row["runtime_seconds"],
            )
        )
    lines.extend(["", "## Paired comparisons versus YW-001", ""])
    for run_id, stats in payload["paired_vs_YW-001"].items():
        lines.append("### {}".format(run_id))
        lines.append("")
        lines.append("- both correct: {}".format(stats["both_correct"]))
        lines.append("- both wrong: {}".format(stats["both_wrong"]))
        lines.append("- YW-001 wrong → new correct: {}".format(stats["base_wrong_to_new_correct"]))
        lines.append("- YW-001 correct → new wrong: {}".format(stats["base_correct_to_new_wrong"]))
        lines.append("- net additional correct cells: {}".format(stats["net_additional_correct"]))
        lines.append("- exact paired delta in accuracy: {:+.6f}".format(stats["delta_accuracy"]))
        lines.append("- fold direction vs YW-001: {}".format(payload["fold_direction"][run_id]))
        lines.append("")
    best = payload["best_experiment"]
    lines.extend(
        [
            "## Best experiment",
            "",
            "- run: {}".format(best["run_id"]),
            "- OOF accuracy: {:.6f}".format(best["oof_accuracy"]),
            "- gain over 0.5500: {:+.6f}".format(best["gain_over_0_5500"]),
            "- net additional correct cells vs YW-001: {}".format(best["net_additional_correct"]),
            "- all three folds improved vs YW-001: {}".format(best["all_folds_improved"]),
            "",
            "## Missing/missing/missing bucket",
            "",
        ]
    )
    for run_id, stats in payload["missing_bucket"].items():
        lines.append("- {}: n={}, accuracy={}".format(run_id, stats["n"], stats["accuracy"]))
    lines.extend(["", "## Largest remaining error groups", ""])
    for item in payload["remaining_errors"]["confusion_pairs_top5"]:
        lines.append(
            "- true={} pred={} n={}".format(item["true_label"], item["predicted_label"], item["n"])
        )
    lines.extend(["", "## Hardest signatures (lowest accuracy, n>=20)", ""])
    for item in payload["remaining_errors"]["hardest_signatures"]:
        lines.append(
            "- {} n={} accuracy={:.4f}".format(item["signature"], item["n"], item["accuracy"])
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    data = load_dataset(ROOT)
    class_names = allowed_labels(ROOT)
    sigs = signatures_from_meta(data.meta_train)
    oofs = {rid: load_oof(rid) for rid in ["YW-001", "YW-002", "YW-003", "YW-004"]}
    mets = {rid: load_metrics(rid) for rid in ["YW-001", "YW-002", "YW-003", "YW-004"]}
    methods = {
        "YW-001": "log1p 200 genes + multinomial LR",
        "YW-002": "log1p genes + fold-safe one-hot Region/E/I/Segment + LR",
        "YW-003": "YW-001 gene-only LR + fold-safe candidate masking",
        "YW-004": "fold-safe per-signature gene LR specialists",
    }
    table_rows = []
    for rid in ["YW-001", "YW-002", "YW-003", "YW-004"]:
        m = mets[rid]
        oof_acc = float(m["oof_accuracy"])
        table_rows.append(
            {
                "run_id": rid,
                "method": methods[rid],
                "fold_0": float(m["fold_accuracy"]["0"]),
                "fold_1": float(m["fold_accuracy"]["1"]),
                "fold_2": float(m["fold_accuracy"]["2"]),
                "oof_accuracy": oof_acc,
                "delta_vs_yw001": oof_acc - 0.55,
                "macro_f1": float(m["macro_f1"]),
                "runtime_seconds": float(m["runtime_seconds"]),
            }
        )
    paired_stats = {}
    directions = {}
    for rid in ["YW-002", "YW-003", "YW-004"]:
        paired_stats[rid] = paired(oofs["YW-001"], oofs[rid])
        directions[rid] = fold_direction(mets["YW-001"], mets[rid])
    ranked = sorted(table_rows, key=lambda r: r["oof_accuracy"], reverse=True)
    best_row = ranked[0]
    best_id = best_row["run_id"]
    all_up = False
    net = 0
    if best_id == "YW-001":
        net = 0
        all_up = False
    else:
        net = paired_stats[best_id]["net_additional_correct"]
        all_up = all(v == "improved" for v in directions[best_id].values())
    miss = {rid: missing_bucket_acc(oofs[rid], sigs) for rid in oofs}
    best_oof = oofs[best_id]
    merged_best = best_oof.copy()
    merged_best["signature"] = sigs.loc[merged_best["Cell_ID"].astype(str)].to_numpy()
    conf = []
    counts = {}
    for t, p in zip(merged_best["true_label"], merged_best["predicted_label"]):
        if t != p:
            counts[(t, p)] = counts.get((t, p), 0) + 1
    for (t, p), n in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
        conf.append({"true_label": t, "predicted_label": p, "n": int(n)})
    sig_stats = []
    for sig, g in merged_best.groupby("signature"):
        n = int(len(g))
        if n < 20:
            continue
        acc = float((g["true_label"] == g["predicted_label"]).mean())
        sig_stats.append({"signature": sig, "n": n, "accuracy": acc})
    hardest = sorted(sig_stats, key=lambda x: (x["accuracy"], -x["n"]))[:8]
    payload = {
        "yw001_oof_accuracy_reference": 0.55,
        "table": table_rows,
        "paired_vs_YW-001": paired_stats,
        "fold_direction": directions,
        "best_experiment": {
            "run_id": best_id,
            "oof_accuracy": best_row["oof_accuracy"],
            "gain_over_0_5500": best_row["oof_accuracy"] - 0.55,
            "net_additional_correct": net,
            "all_folds_improved": all_up,
            "fold_direction": directions.get(best_id, {"0": "reference", "1": "reference", "2": "reference"}),
        },
        "missing_bucket": miss,
        "remaining_errors": {
            "from_run": best_id,
            "confusion_pairs_top5": conf,
            "hardest_signatures": hardest,
        },
        "n_classes": len(class_names),
    }
    out_json = ROOT / "outputs" / "metrics" / "sprint2_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(ROOT / "reports" / "sprint2_comparison.md", table_rows, payload)
    print("Wrote {}".format(out_json))
    print("Wrote reports/sprint2_comparison.md")
    print("best={}".format(best_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
