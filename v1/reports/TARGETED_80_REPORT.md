# Targeted 80% experiments

## Outcome

The two requested modules were implemented and evaluated on the frozen 0.7910 four-fold OOF probabilities.

- Baseline: 0.7910 accuracy, 0.7450 macro-F1, 3,955/5,000 correct.
- Pairwise oligodendrocyte experts: 0.7914 accuracy, 0.7451 macro-F1, +2 correct.
- Pairwise experts plus Section-aware kNN prior: **0.7932 accuracy**, 0.7458 macro-F1, 3,966/5,000 correct.
- Total gain: +11 correct cells.
- Fold-level gains: +3, +3, +3, +2.

The result is stable across all four folds, but it does not reach 80%. It remains 34 correct cells short of 4,000/5,000.

## 1. Pairwise oligodendrocyte experts

Three binary experts were trained with fold-safe internal data and the filtered external MERFISH reference:

1. `oligodendrocyte_progenitor_2` vs `oligodendrocyte_1`;
2. `oligodendrocyte_progenitor_2` vs `oligodendrocyte_2`;
3. `oligodendrocyte_progenitor_1` vs `oligodendrocyte_precursor_cell`.

Each expert blends a regularized LightGBM model, a scaled logistic model, and an external-expression model. Corrections are applied only when pair mass, model confidence, and fold-stability gates pass.

Only the first expert was accepted. It added two correct cells with fold deltas +1, +1, 0, 0. The other two experts were rejected because they did not improve the full 60-class pipeline, despite reasonable within-pair accuracy.

## 2. Section-aware kNN prior

For every validation fold, neighbors were drawn only from the other three training folds within the same `Section_ID`; validation labels were never used as neighbor labels. Test neighbors use all labeled training cells.

The search compared spatial coordinates, whitened and non-whitened expression PCA, raw/log-normalized expression cosine distance, and joint expression-spatial embeddings over several values of k.

The selected configuration is:

- Section-constrained log-expression cosine neighbors;
- k = 20;
- distance-weighted label distribution with 1.5 prior smoothing;
- multiplicative probability correction with weight 0.05;
- base-confidence cap 0.55;
- neighbor-confidence gate 0.25.

This module added nine correct OOF cells with fold deltas +2, +2, +3, +2.

## Candidate behavior

- 25 OOF labels changed.
- 16 previously wrong predictions were fixed.
- 5 previously correct predictions were broken.
- 4 predictions changed but remained wrong.
- Net change: +11 correct.
- The candidate changes 44 of the 5,000 submitted test labels.

## Files

- Implementation: `targeted_80.py`
- Full experiment report: `artifacts/targeted_80_summary.json`
- OOF/test probabilities: `artifacts/targeted_80_probabilities.npz`
- Validated candidate submission: `artifacts/prediction_targeted_80.csv`

The candidate has not replaced or been pushed over the current `test1` submission.
