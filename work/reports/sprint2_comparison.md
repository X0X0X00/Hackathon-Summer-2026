# Sprint 2 OOF comparison

Frozen folds: `experiments/folds.csv`. Primary metric: overall accuracy.
YW-001 was not refit. YW-002/003/004 are experiments, not formal model versions.

| Run | Method | Fold 0 | Fold 1 | Fold 2 | OOF Accuracy | Delta vs YW-001 | Macro-F1 | Runtime |
| --- | ------ | -----: | -----: | -----: | -----------: | --------------: | -------: | ------: |
| YW-001 | log1p 200 genes + multinomial LR | 0.5429 | 0.5579 | 0.5492 | 0.5500 | +0.0000 | 0.3894 | 2.672s |
| YW-002 | log1p genes + fold-safe one-hot Region/E/I/Segment + LR | 0.7540 | 0.7570 | 0.7593 | 0.7568 | +0.2068 | 0.6799 | 1.630s |
| YW-003 | YW-001 gene-only LR + fold-safe candidate masking | 0.7510 | 0.7510 | 0.7569 | 0.7530 | +0.2030 | 0.7130 | 3.178s |
| YW-004 | fold-safe per-signature gene LR specialists | 0.7594 | 0.7564 | 0.7635 | 0.7598 | +0.2098 | 0.7053 | 2.566s |

## Paired comparisons versus YW-001

### YW-002

- both correct: 2699
- both wrong: 1165
- YW-001 wrong → new correct: 1085
- YW-001 correct → new wrong: 51
- net additional correct cells: 1034
- exact paired delta in accuracy: +0.206800
- fold direction vs YW-001: {'0': 'improved', '1': 'improved', '2': 'improved'}

### YW-003

- both correct: 2750
- both wrong: 1235
- YW-001 wrong → new correct: 1015
- YW-001 correct → new wrong: 0
- net additional correct cells: 1015
- exact paired delta in accuracy: +0.203000
- fold direction vs YW-001: {'0': 'improved', '1': 'improved', '2': 'improved'}

### YW-004

- both correct: 2696
- both wrong: 1147
- YW-001 wrong → new correct: 1103
- YW-001 correct → new wrong: 54
- net additional correct cells: 1049
- exact paired delta in accuracy: +0.209800
- fold direction vs YW-001: {'0': 'improved', '1': 'improved', '2': 'improved'}

## Best experiment

- run: YW-004
- OOF accuracy: 0.759800
- gain over 0.5500: +0.209800
- net additional correct cells vs YW-001: 1049
- all three folds improved vs YW-001: True

## Missing/missing/missing bucket

- YW-001: n=2958, accuracy=0.6612576064908722
- YW-002: n=2958, accuracy=0.6812035158891143
- YW-003: n=2958, accuracy=0.6778228532792427
- YW-004: n=2958, accuracy=0.6815415821501014

## Largest remaining error groups

- true=oligodendrocyte_1 pred=oligodendrocyte_progenitor_2 n=111
- true=oligodendrocyte_progenitor_2 pred=oligodendrocyte_1 n=64
- true=oligodendrocyte_progenitor_2 pred=oligodendrocyte_2 n=56
- true=oligodendrocyte_2 pred=oligodendrocyte_progenitor_2 n=50
- true=astrocyte_2 pred=astrocyte_1 n=42

## Hardest signatures (lowest accuracy, n>=20)

- 1/excitatory/6 n=89 accuracy=0.6292
- 1/excitatory/10 n=96 accuracy=0.6458
- 1/excitatory/8 n=181 accuracy=0.6685
- __MISSING__/__MISSING__/__MISSING__ n=2958 accuracy=0.6815
- 1/excitatory/2 n=64 accuracy=0.7188
- __MISSING__/__MISSING__/22 n=169 accuracy=0.7456
- 1/inhibitory/9 n=46 accuracy=0.8043
- 1/excitatory/3 n=93 accuracy=0.8495
