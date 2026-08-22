"""Fold-safe metadata signatures and candidate maps.

Missing Region / Excitatory_vs_Inhibitory / Segment values are encoded as
the canonical token __MISSING__. Maps must be built from training-fold rows
only; this module does not inspect validation labels.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from merfish60.official_contract import YW002_METADATA_FIELDS


MISSING_TOKEN = "__MISSING__"
NA_TOKENS = {"", "NA", "NaN", "nan", "None", "<NA>", "NULL"}
SIGNATURE_FIELDS = list(YW002_METADATA_FIELDS)


def canonicalize_value(value) -> str:
    if value is None:
        return MISSING_TOKEN
    try:
        if pd.isna(value):
            return MISSING_TOKEN
    except (ValueError, TypeError):
        pass
    text = str(value).strip()
    if text in NA_TOKENS:
        return MISSING_TOKEN
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def signature_tuple(region, excitatory_vs_inhibitory, segment) -> Tuple[str, str, str]:
    return (
        canonicalize_value(region),
        canonicalize_value(excitatory_vs_inhibitory),
        canonicalize_value(segment),
    )


def signature_key(sig: Sequence[str]) -> str:
    return "{}/{}/{}".format(sig[0], sig[1], sig[2])


def signatures_from_meta(meta: pd.DataFrame) -> pd.Series:
    missing = [c for c in SIGNATURE_FIELDS if c not in meta.columns]
    if missing:
        raise KeyError("meta missing signature fields: {}".format(missing))
    keys = [
        signature_key(
            signature_tuple(row["Region"], row["Excitatory_vs_Inhibitory"], row["Segment"])
        )
        for row in meta[SIGNATURE_FIELDS].to_dict("records")
    ]
    return pd.Series(keys, index=meta.index, name="signature")


def missing_bucket_key() -> str:
    return signature_key((MISSING_TOKEN, MISSING_TOKEN, MISSING_TOKEN))


def build_candidate_map(
    signatures: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Set[str]]:
    """Map signature -> candidate classes using only the provided rows.

    Callers must pass training-fold signatures and training-fold labels.
    """
    if len(signatures) != len(labels):
        raise ValueError("signatures and labels must have the same length")
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for sig, lab in zip(signatures, labels):
        mapping[str(sig)].add(str(lab))
    return {k: set(v) for k, v in mapping.items()}


def is_deterministic_map(candidates: Set[str]) -> bool:
    return len(candidates) == 1


def mask_and_renormalize(
    proba_row: np.ndarray,
    class_names: Sequence[str],
    candidates: Optional[Iterable[str]],
) -> Tuple[np.ndarray, str]:
    """Zero non-candidate mass and renormalize.

    If candidates is None (unseen signature), the original vector is kept.
    If remaining mass is numerically zero, use uniform mass on candidates.
    Returns (new_row, action) where action is keep_unmasked | masked | uniform_candidates.
    """
    out = np.asarray(proba_row, dtype=np.float64).copy()
    if candidates is None:
        return out, "keep_unmasked"
    cand = set(str(c) for c in candidates)
    if not cand:
        return out, "keep_unmasked"
    keep = np.array([name in cand for name in class_names], dtype=bool)
    masked = out.copy()
    masked[~keep] = 0.0
    total = float(masked.sum())
    if total > 1e-15:
        return masked / total, "masked"
    uniform = np.zeros_like(out)
    uniform[keep] = 1.0 / float(keep.sum())
    return uniform, "uniform_candidates"
