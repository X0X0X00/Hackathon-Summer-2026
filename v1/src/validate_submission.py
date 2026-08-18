from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


V1_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V1_ROOT.parent
LABEL = "MERFISH_cell_type_annotation"


def main() -> None:
    prediction_path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else V1_ROOT / "submission" / "prediction.csv"
    )
    prediction = pd.read_csv(prediction_path)
    template = pd.read_csv(REPO_ROOT / "prediction" / "prediction.csv")
    meta_train = pd.read_csv(REPO_ROOT / "data" / "meta_train.csv", index_col=0)
    meta_test = pd.read_csv(REPO_ROOT / "data" / "meta_test.csv", index_col=0)

    if prediction.columns.tolist() != template.columns.tolist():
        raise ValueError("prediction columns do not match the official template")
    if len(prediction) != len(meta_test):
        raise ValueError(f"expected {len(meta_test)} rows, found {len(prediction)}")
    if prediction.iloc[:, 0].astype(str).tolist() != meta_test.index.astype(str).tolist():
        raise ValueError("Cell_ID order does not match meta_test.csv")
    if prediction.iloc[:, 0].duplicated().any():
        raise ValueError("duplicate Cell_ID values found")
    if prediction.isna().any().any():
        raise ValueError("null values found")
    valid_labels = set(meta_train[LABEL].astype(str))
    invalid = sorted(set(prediction.iloc[:, 1].astype(str)) - valid_labels)
    if invalid:
        raise ValueError(f"invalid labels found: {invalid}")
    print(
        f"valid submission: rows={len(prediction)}, "
        f"unique_ids={prediction.iloc[:, 0].nunique()}, "
        f"predicted_classes={prediction.iloc[:, 1].nunique()}"
    )


if __name__ == "__main__":
    main()

