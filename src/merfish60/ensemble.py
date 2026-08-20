"""Cross-fitted probability ensembling over saved first-level OOF files.

This is NOT a fully nested / fully leakage-safe ensemble.

Weight selection for held-out fold f does not read fold-f labels directly.
It does use saved OOF probabilities for the other two folds. Those
probabilities were produced by YW-002/YW-003/YW-004 models whose training
sets included fold f (standard three-fold OOF: a fold-g prediction is
trained on all folds except g, which includes f whenever f != g).

Changing fold-f labels and regenerating the dependent base probabilities
could therefore change the weights chosen for fold f. Treat YW-005 as an
exploratory ensemble diagnostic, not as a nested stacking estimator.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from merfish60.cv import FOLD_VALUES
from merfish60.models import argmax_labels


WEIGHT_STEP = 0.1


def convex_weight_grid(step: float = WEIGHT_STEP) -> List[Tuple[float, float, float]]:
    """Nonnegative weights summing to 1 on a coarse simplex grid."""
    n = int(round(1.0 / step))
    grid = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            k = n - i - j
            grid.append((round(i * step, 1), round(j * step, 1), round(k * step, 1)))
    return grid


def mix_probas(
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    w2, w3, w4 = (float(weights[0]), float(weights[1]), float(weights[2]))
    mixed = w2 * p2 + w3 * p3 + w4 * p4
    totals = mixed.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1e-15)
    return mixed / totals


def accuracy_from_proba(
    proba: np.ndarray, y_true: np.ndarray, class_names: Sequence[str]
) -> float:
    pred = argmax_labels(proba, class_names)
    return float(np.mean(pred == y_true))


def select_ensemble_weights(
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    y_true: np.ndarray,
    eligible_mask: np.ndarray,
    class_names: Sequence[str],
    grid: Iterable[Tuple[float, float, float]] = None,
) -> Tuple[Tuple[float, float, float], float]:
    """Choose convex weights using ONLY eligible rows of saved OOF files.

    `eligible_mask` must exclude the held-out fold. Held-out labels are
    never read because y_true is indexed by this mask. This does not make
    the procedure nested: eligible-row probabilities were still produced
    by base models trained on the held-out fold.
    """
    if grid is None:
        grid = convex_weight_grid()
    y = np.asarray(y_true)[eligible_mask]
    best = None
    best_acc = -1.0
    equal = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    for weights in grid:
        mixed = mix_probas(p2[eligible_mask], p3[eligible_mask], p4[eligible_mask], weights)
        acc = accuracy_from_proba(mixed, y, class_names)
        dist = sum((a - b) ** 2 for a, b in zip(weights, equal))
        key = (acc, -dist, -weights[0], -weights[1], -weights[2])
        if best is None or key > best[0]:
            best = (key, weights, acc)
            best_acc = acc
    return best[1], best_acc


def cross_fitted_ensemble(
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """Blend saved OOF probabilities with fold-wise weights chosen off-fold.

    Off-fold means labels from fold f are not used to score the grid.
    Base models that produced the other folds' OOF probabilities were
    still trained with fold f in their training sets.
    """
    y_true = np.asarray(y_true)
    folds = np.asarray(folds)
    out = np.zeros_like(p2, dtype=np.float64)
    details = {}
    for fold_id in FOLD_VALUES:
        held = folds == fold_id
        eligible = ~held
        weights, select_acc = select_ensemble_weights(
            p2, p3, p4, y_true, eligible, class_names
        )
        out[held] = mix_probas(p2[held], p3[held], p4[held], weights)
        details[str(fold_id)] = {
            "weights": {"YW-002": weights[0], "YW-003": weights[1], "YW-004": weights[2]},
            "selection_accuracy_on_other_folds": select_acc,
            "n_selection_cells": int(eligible.sum()),
            "n_held_out_cells": int(held.sum()),
        }
    return out, details
