# Sprint 3 OOF comparison

Frozen folds: `experiments/folds.csv`. Primary metric: overall accuracy.
Sprint 2 runs YW-002/YW-003/YW-004 were not refit.
YW-005 is an exploratory ensemble diagnostic (not nested; excluded from formal selection).
YW-006 is a valid hard-bucket negative ablation, not a 5000-cell candidate.
YW-007 is a valid hybrid that exactly duplicates YW-004 and is not selected.
Formal Model V1 architecture: YW-004 (OOF accuracy 0.7598). Model V1 is not written in this sprint.

| Run | Method | Fold 0 | Fold 1 | Fold 2 | OOF Accuracy | Delta vs YW-004 | Macro-F1 | Total Correct | Runtime | Eligible |
| --- | ------ | -----: | -----: | -----: | -----------: | --------------: | -------: | ------------: | ------: | -------- |
| YW-002 | gene + fold-safe Region/E-I/Segment one-hot + LR | 0.7540 | 0.7570 | 0.7593 | 0.7568 | -0.0030 | 0.6799 | 3784 | 1.630s | yes |
| YW-003 | gene-only global LR + fold-safe candidate masking | 0.7510 | 0.7510 | 0.7569 | 0.7530 | -0.0068 | 0.7130 | 3765 | 3.178s | yes |
| YW-004 | fold-safe per-signature logistic specialists | 0.7594 | 0.7564 | 0.7635 | 0.7598 | +0.0000 | 0.7053 | 3799 | 2.566s | yes |
| YW-005 | exploratory ensemble diagnostic (not nested; excluded from selection) | 0.7594 | 0.7570 | 0.7611 | 0.7592 | -0.0006 | 0.7072 | 3796 | 0.418s | no |
| YW-007 | hybrid: YW-004 outside missing bucket; YW-006 inside | 0.7594 | 0.7564 | 0.7635 | 0.7598 | +0.0000 | 0.7053 | 3799 | 0.001s | yes |

## Formal Model V1 decision

- selected architecture: YW-004
- selected OOF accuracy: 0.759800
- YW-007 not selected: exact more-complex duplicate of YW-004; n_changed=0; not selected
- YW-005 not selected: exploratory ensemble diagnostic; not nested; underperforms YW-004 (OOF 0.7592 vs 0.7598)
- YW-006 retained as: valid negative ablation; hard-bucket only; selected log1p_lr matching YW-004
- remaining total errors: 1201
- errors inside missing bucket: 942
- errors outside missing bucket: 259

## YW-005 exploratory diagnostic

- role: exploratory ensemble diagnostic; excluded from formal selection
- nested independence: false
- equal-weight OOF accuracy: 0.7598
- cross-fitted OOF accuracy (exploratory): 0.7592
- provenance: Standard 3-fold OOF: a prediction for fold g is fit on folds != g. When selecting weights for held-out fold f, the eligible folds are g != f, so those base models were trained with fold f in the training set. Regenerating base probabilities after changing fold-f labels could change the selected weights.

## YW-006 hard-bucket specialist (valid negative ablation)

- hard-bucket OOF accuracy: 0.6815415821501014
- selected model by outer fold: {'0': 'log1p_lr', '1': 'log1p_lr', '2': 'log1p_lr'}
- delta vs YW-004 hard bucket: 0.0

## Remaining errors (YW-004)

- from run: YW-004
- remaining total errors: 1201
- errors inside missing bucket: 942
- errors outside missing bucket: 259

### Five largest confusion pairs

- true=oligodendrocyte_1 pred=oligodendrocyte_progenitor_2 n=111
- true=oligodendrocyte_progenitor_2 pred=oligodendrocyte_1 n=64
- true=oligodendrocyte_progenitor_2 pred=oligodendrocyte_2 n=56
- true=oligodendrocyte_2 pred=oligodendrocyte_progenitor_2 n=50
- true=astrocyte_2 pred=astrocyte_1 n=42
