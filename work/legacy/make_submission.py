"""Turn final test probs into prediction/prediction.csv (exact template format).

Applies the E/I hard constraint using observed test metadata before argmax.
Usage: python make_submission.py [probs_npz]  (default work/final_probs.npz)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from common import load, apply_ei, WORK

D = load()
te = np.where(~D["is_train"])[0]
probs_file = Path(sys.argv[1]) if len(sys.argv) > 1 else WORK / "final_probs.npz"
test_probs = np.load(probs_file)["test"]
assert test_probs.shape == (len(te), 60)

test_probs = apply_ei(test_probs, D["ei_known"][te], D["ei_of_label"])
pred_codes = test_probs.argmax(1)
labels = D["labels"]
pred_labels = labels[pred_codes]

BASE = WORK.parent
template = pd.read_csv(BASE / "prediction/prediction.csv")
ids = D["ids"][te]
assert (template.iloc[:, 0].astype(str).values == ids).all(), "row order mismatch vs template!"

out = pd.DataFrame({template.columns[0]: ids, template.columns[1]: pred_labels})
out.to_csv(BASE / "prediction/prediction.csv", index=False)
print(f"wrote prediction/prediction.csv  ({len(out)} rows)")
print("prediction distribution (top 10):")
print(pd.Series(pred_labels).value_counts().head(10).to_string())
# sanity: all predicted labels are valid training labels
assert set(pred_labels) <= set(labels)
print("all labels valid:", True)
