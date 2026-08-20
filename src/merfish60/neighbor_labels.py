"""Fold-safe neighbor-label histograms and E/I post-constraint.

When predicting fold f, hide ALL competition labels from fold f before any
label-derived feature is built. Test labels are never visible.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from merfish60.io import N_CLASSES
from merfish60.spatial_features import SpatialUniverse


SP_K = 15
EX_K = 25


def encode_train_labels(
    y_train: Sequence[str],
    class_names: Sequence[str],
    n_universe: int,
    is_train: np.ndarray,
) -> np.ndarray:
    """Integer codes for the universe: train labels in class order, test = -1."""
    index = {str(name): i for i, name in enumerate(class_names)}
    y = np.full(n_universe, -1, dtype=np.int64)
    codes = np.array([index[str(v)] for v in y_train], dtype=np.int64)
    y[np.asarray(is_train)] = codes
    return y


def visible_label_codes(
    y_codes: np.ndarray,
    is_train: np.ndarray,
    fold_ids_universe: np.ndarray,
    holdout_fold: Optional[int],
) -> np.ndarray:
    """Return known codes with holdout-fold and non-train labels set to -1.

    fold_ids_universe is length n_universe; train rows hold 0..4, test rows -1.
    If holdout_fold is None, all training labels are visible (test-time).
    """
    known = np.asarray(y_codes, dtype=np.int64).copy()
    known[~np.asarray(is_train)] = -1
    if holdout_fold is not None:
        hide = np.asarray(is_train) & (np.asarray(fold_ids_universe) == int(holdout_fold))
        known[hide] = -1
    return known


def visible_ext_label_codes(
    y_codes: np.ndarray,
    is_train: np.ndarray,
    is_ref: np.ndarray,
    fold_ids_universe: np.ndarray,
    holdout_fold: Optional[int],
) -> np.ndarray:
    """Visible labels for the extended universe.

    Reference labels are always visible. Competition fold-f labels are hidden.
    Test labels are never visible. Fingerprint-duplicate rows stay unlabeled.
    """
    known = np.full(len(y_codes), -1, dtype=np.int64)
    is_ref = np.asarray(is_ref)
    is_train = np.asarray(is_train)
    y_codes = np.asarray(y_codes, dtype=np.int64)
    known[is_ref] = y_codes[is_ref]
    if holdout_fold is None:
        vis_train = is_train
    else:
        vis_train = is_train & (np.asarray(fold_ids_universe) != int(holdout_fold))
    known[vis_train] = y_codes[vis_train]
    known[~is_ref & ~is_train] = -1
    return known


def apply_segment_mask(
    probs: np.ndarray,
    segment_values: np.ndarray,
    allowed_by_segment: dict,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    """Zero classes never observed with Segment s among reference cells; renormalize.

    Rows the mask would zero out entirely keep their incoming probabilities.
    """
    source = np.asarray(probs, dtype=np.float64)
    out = source.copy()
    segv = np.asarray(segment_values, dtype=np.float64)
    allowed = np.ones((len(segv), n_classes), dtype=bool)
    for i, value in enumerate(segv):
        if not np.isfinite(value):
            continue
        key = int(value)
        if key not in allowed_by_segment:
            continue
        allowed[i] = False
        cols = allowed_by_segment[key]
        allowed[i, np.asarray(cols, dtype=np.int64)] = True
    masked = np.where(allowed, out, 0.0)
    dead = masked.sum(axis=1) <= 0
    masked[dead] = source[dead]
    totals = masked.sum(axis=1, keepdims=True)
    return masked / np.maximum(totals, 1e-12)


def segment_allowed_from_reference(
    segment_values: np.ndarray,
    is_ref: np.ndarray,
    y_codes: np.ndarray,
) -> dict:
    """Segment -> allowed class codes, built from reference labels only."""
    seg = np.asarray(segment_values, dtype=np.float64)
    ref = np.asarray(is_ref)
    y = np.asarray(y_codes, dtype=np.int64)
    ok = ref & np.isfinite(seg) & (y >= 0)
    table = {}
    for value in np.unique(seg[ok]):
        table[int(value)] = np.unique(y[ok & (seg == value)]).astype(int).tolist()
    return table


def label_hist(
    known: np.ndarray,
    idx: np.ndarray,
    dist: np.ndarray,
    k: int,
    tag: str,
    n_classes: int = N_CLASSES,
) -> Tuple[np.ndarray, List[str]]:
    """Weighted class histogram over each cell's first k *labeled* neighbors.

    known: (n,) int codes, -1 = label not visible.
    Weights are 1/(1+d). Self is never present in idx.
    """
    n, k_cached = idx.shape
    hist = np.zeros((n, n_classes), dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    dsum = np.zeros(n, dtype=np.float32)
    taken = np.zeros(n, dtype=np.int32)
    rows = np.arange(n)
    known = np.asarray(known, dtype=np.int64)
    for col in range(k_cached):
        j = idx[:, col]
        ok = (
            (j >= 0)
            & (known[np.maximum(j, 0)] >= 0)
            & (taken < k)
            & np.isfinite(dist[:, col])
        )
        w = 1.0 / (1.0 + dist[ok, col])
        hist[rows[ok], known[j[ok]]] += w
        cnt[ok] += 1
        dsum[ok] += dist[ok, col]
        taken[ok] += 1
    tot = hist.sum(axis=1, keepdims=True)
    hist = hist / np.maximum(tot, 1e-9)
    out = np.hstack([hist, cnt[:, None], (dsum / np.maximum(cnt, 1.0))[:, None]]).astype(
        np.float32
    )
    names = ["{}_h{}".format(tag, c) for c in range(n_classes)] + [
        "{}_n".format(tag),
        "{}_d".format(tag),
    ]
    return out, names


def build_X(
    universe: SpatialUniverse,
    known: np.ndarray,
    sp_k: int = SP_K,
    ex_k: int = EX_K,
) -> Tuple[np.ndarray, List[str]]:
    """Static features + fold-aware neighbor-label histograms."""
    parts = [universe.X_static]
    names = list(universe.names)
    if sp_k:
        hist, hist_names = label_hist(
            known, universe.sp_idx, universe.sp_dist, sp_k, "sp", n_classes=N_CLASSES
        )
        parts.append(hist)
        names.extend(hist_names)
    if ex_k:
        hist, hist_names = label_hist(
            known, universe.ex_idx, universe.ex_dist, ex_k, "ex", n_classes=N_CLASSES
        )
        parts.append(hist)
        names.extend(hist_names)
    return np.hstack(parts), names


def apply_ei(probs: np.ndarray, ei_known: np.ndarray, ei_of_label: np.ndarray) -> np.ndarray:
    """Zero classes inconsistent with observed E/I; renormalize.

    Rows the constraint would zero out entirely keep their original probabilities.
    """
    out = np.asarray(probs, dtype=np.float64).copy()
    ei_known = np.asarray(ei_known)
    ei_of_label = np.asarray(ei_of_label)
    for value in (0, 1):
        rows = np.where(ei_known == value)[0]
        if len(rows) == 0:
            continue
        bad_cols = np.where(ei_of_label != value)[0]
        sub = out[rows].copy()
        sub[:, bad_cols] = 0.0
        z = sub.sum(axis=1)
        alive = z > 0
        out[rows[alive]] = sub[alive] / z[alive, None]
    return out
