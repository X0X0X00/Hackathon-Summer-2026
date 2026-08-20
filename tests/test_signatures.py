"""Fold-safe signature and candidate-map leakage guards."""

import numpy as np
import pandas as pd
import pytest

from merfish60.models import train_val_masks
from merfish60.signatures import (
    MISSING_TOKEN,
    build_candidate_map,
    canonicalize_value,
    mask_and_renormalize,
    missing_bucket_key,
    signature_key,
    signature_tuple,
)


def test_missing_value_canonicalization():
    assert canonicalize_value(None) == MISSING_TOKEN
    assert canonicalize_value(float("nan")) == MISSING_TOKEN
    assert canonicalize_value("NA") == MISSING_TOKEN
    assert canonicalize_value("nan") == MISSING_TOKEN
    assert canonicalize_value(1.0) == "1"
    assert canonicalize_value("4.0") == "4"
    assert canonicalize_value("excitatory") == "excitatory"
    assert missing_bucket_key() == "__MISSING__/__MISSING__/__MISSING__"
    assert signature_key(signature_tuple("NA", None, 22)) == "__MISSING__/__MISSING__/22"


def test_candidate_map_from_training_rows_only():
    train_sig = ["A", "A", "B"]
    train_y = ["t1", "t1", "t2"]
    val_y = ["SHOULD_NOT_ENTER"]
    cmap = build_candidate_map(train_sig, train_y)
    observed = set().union(*cmap.values())
    assert "SHOULD_NOT_ENTER" not in observed
    assert cmap["A"] == {"t1"}
    assert cmap["B"] == {"t2"}
    assert val_y[0] not in cmap.get("A", set())


def test_train_validation_masks_disjoint():
    folds = pd.Series([0, 0, 1, 1, 2, 2], dtype=int)
    for fold_id in (0, 1, 2):
        train, val = train_val_masks(folds, fold_id)
        assert not np.any(train & val)
        assert np.any(train) and np.any(val)
        assert int(val.sum()) == 2


def test_mask_renormalize_and_unseen_fallback():
    classes = ["a", "b", "c"]
    row = np.array([0.2, 0.5, 0.3])
    masked, action = mask_and_renormalize(row, classes, {"a", "c"})
    assert action == "masked"
    assert masked[1] == 0.0
    assert pytest.approx(masked.sum(), abs=1e-12) == 1.0
    assert pytest.approx(masked[0], abs=1e-12) == 0.2 / 0.5
    kept, action2 = mask_and_renormalize(row, classes, None)
    assert action2 == "keep_unmasked"
    assert np.allclose(kept, row)
