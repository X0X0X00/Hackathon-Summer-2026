# V3-E05A — Asymmetric Directional Expert Complementarity Audit

Research codename: **ADE-AUDIT**. This experiment is ANALYSIS ONLY. It does not train a model, router, or ensemble, does not search thresholds or weights, does not add a dataset, and does not freeze MODEL V3.

## 1. Research Question

Can observable predicted-class patterns identify asymmetric confusion directions where S0 or SNI provides stable, low-risk correction value?

A future deployable concept, if supported, would **not** be "choose the most confident expert" or "average all experts". It would be:

strong anchor prediction pattern → specific predefined confusion direction? → abstain and keep the strong prediction, or consult a designated directional expert with a fixed patch.

V3-E05A only evaluates whether such directional structure exists. It does **not** implement a final patch.

Oracle values below are diagnostic coverage ceilings. **ORACLE != DEPLOYABLE ACCURACY.**

## 2. Frozen Four-Expert Evidence

Reproduced from the four-expert canonical registry before any directional analysis:

| Quantity | Frozen | Reproduced |
|---|---:|---:|
| LZH Prior-H | 4133 / 5000 = 0.8266 | 4133 / 5000 = 0.8266 |
| WYH MODEL V2 | 4106 / 5000 = 0.8212 | 4106 / 5000 = 0.8212 |
| S0 hard-label 200-gene MLP | 3151 / 5000 = 0.6302 | 3151 / 5000 = 0.6302 |
| SNI-only source-diverse expert | 2841 / 5000 = 0.5682 | 2841 / 5000 = 0.5682 |
| LZH + WYH oracle | 4215 / 5000 = 0.8430 | 4215 / 5000 = 0.8430 |
| LZH + WYH + S0 oracle | 4311 / 5000 = 0.8622 | 4311 / 5000 = 0.8622 |
| LZH + WYH + S0 + SNI oracle | 4364 / 5000 = 0.8728 | 4364 / 5000 = 0.8728 |
| S0 unique recoveries beyond LZH+WYH | 96 | 96 |
| SNI unique recoveries beyond LZH+WYH+S0 | 53 | 53 |
| Remaining all-four-wrong | 636 | 636 |
| Four-expert oracle folds 0-2 | 2627 / 3000 = 0.8757 | 2627 / 3000 = 0.8757 |
| Four-expert oracle folds 3-4 | 1737 / 2000 = 0.8685 | 1737 / 2000 = 0.8685 |

These oracles are **diagnostic coverage ceilings**, not deployable accuracy.

The question is no longer whether another expert can raise the oracle. The question is whether a simple, interpretable, predefined directional mechanism can convert some of this coverage into real net corrections.

## 3. Why Confidence Routing Was Rejected

V3-E03A found that S0 confidence does not identify the 96 unique S0 rescues: high S0 top1/margin deciles contained almost none of them, and P vs N1 AUROC for S0 top1 was 0.4142.

V3-E04S found that inside the 689 all-three-wrong cells, SNI-correct cases had **lower** mean top1/margin than SNI-wrong cases (top1 AUROC 0.4195).

Neither weak expert has globally useful confidence-based rescue behavior. Confidence is therefore **not** the primary E05A hypothesis. It appears only as a secondary descriptive diagnostic inside the predeclared triggers.

## 4. Directional-Expertise Hypothesis

S0 and SNI are weak global classifiers (0.6302 and 0.5682) but contribute 96 and 53 unique recoveries. Frozen confusion-pair evidence was asymmetric:

- SNI rescued true `oligodendrocyte_progenitor_2` when strong experts predicted `oligodendrocyte_1` (20 / 36 shared errors).
- SNI rescued true `oligodendrocyte_progenitor_2` when LZH predicted `oligodendrocyte_2` (8 / 35).
- SNI rescued **0** of the reverse failure true `oligodendrocyte_1` / strong `oligodendrocyte_progenitor_2`.
- S0 previously contributed in that reverse oligo-1 direction.

Hypothesis: weak/source-diverse experts may possess **direction-specific expertise** rather than globally useful reliability.

A future rule may use only inference-time observables (predicted classes, probabilities, agreement, test-safe metadata). It must never condition on true class, correctness flags, oracle membership, or rescue flags.

## 5. Integrity Checks

Four-expert registry rows: **5000**. Unique 19-digit `Cell_ID` strings: **5000**. Labels aligned to official `meta_train`. Joins used `Cell_ID`, never row position.

All predeclared counts reproduced exactly:

| Check | Expected | Reproduced |
| --- | --- | --- |
| LZH correct | 4133 | 4133 |
| WYH correct | 4106 | 4106 |
| S0 correct | 3151 | 3151 |
| SNI correct | 2841 | 2841 |
| two-expert oracle | 4215 | 4215 |
| three-expert oracle | 4311 | 4311 |
| four-expert oracle | 4364 | 4364 |
| all-four-wrong | 636 | 636 |
| S0 incremental unique recoveries | 96 | 96 |
| SNI incremental unique recoveries | 53 | 53 |

If any of these had failed, the experiment would have stopped with `E05A REGISTRY INTEGRITY FAILURE`.

H1/H2/H3 trigger functions use predictions only: **yes**.

## 6. Predeclared H1 / H2 / H3

Trigger definitions were frozen from prior E03A/E04S observations **before** E05A accounting. They were not modified after seeing E05A results. No confidence thresholds were added. Newly discovered full-data patterns are **not** promoted to candidate rules here.

Canonical folds 3-4 have been viewed in prior research stages. They are a **RETROSPECTIVE STABILITY PARTITION**, not an untouched holdout. E05A does not claim an unbiased final MODEL V3 result.

### H1

Observable trigger: `LZH_pred == oligodendrocyte_1 AND WYH_pred == oligodendrocyte_1 AND SNI_pred == oligodendrocyte_progenitor_2`.
Hypothetical action: replace strong consensus with `SNI` prediction `oligodendrocyte_progenitor_2`.
The trigger uses **predictions only**. It does not condition on true class.

| split | support | wrong→correct | correct→wrong | precision | net | strong acc | candidate acc | oracle acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 88 | 25 | 55 | 0.3125 | -30 | 0.6250 | 0.2841 | 0.9091 |
| folds 0-2 (development) | 50 | 15 | 32 | 0.3191 | -17 | 0.6400 | 0.3000 | 0.9400 |
| folds 3-4 (retrospective stability) | 38 | 10 | 23 | 0.3030 | -13 | 0.6053 | 0.2632 | 0.8684 |

Actionable under predeclared criteria: **no**.
Failed checks: precision_ge_065, net_gt_0, folds_0_2_net_gt_0, folds_3_4_net_gt_0, folds_0_2_precision_ge_060, folds_3_4_precision_ge_055.
Sparsity flag: **not sparse** (none).
Sections represented: 51. Max section fraction: 0.0568.
Support by fold: 16/19/15/24/14.
Retrospective true-class mix (diagnostic only; not a trigger input): oligodendrocyte_1=55, oligodendrocyte_progenitor_2=25, endothelial=4, microglia=2, astrocyte_1=1, pericyte=1.

### H2

Observable trigger: `LZH_pred == oligodendrocyte_progenitor_2 AND WYH_pred == oligodendrocyte_progenitor_2 AND S0_pred == oligodendrocyte_1`.
Hypothetical action: replace strong consensus with `S0` prediction `oligodendrocyte_1`.
The trigger uses **predictions only**. It does not condition on true class.

| split | support | wrong→correct | correct→wrong | precision | net | strong acc | candidate acc | oracle acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 68 | 25 | 33 | 0.4310 | -8 | 0.4853 | 0.3676 | 0.8529 |
| folds 0-2 (development) | 47 | 19 | 23 | 0.4524 | -4 | 0.4894 | 0.4043 | 0.8936 |
| folds 3-4 (retrospective stability) | 21 | 6 | 10 | 0.3750 | -4 | 0.4762 | 0.2857 | 0.7619 |

Actionable under predeclared criteria: **no**.
Failed checks: precision_ge_065, net_gt_0, folds_0_2_net_gt_0, folds_3_4_net_gt_0, folds_0_2_precision_ge_060, folds_3_4_precision_ge_055.
Sparsity flag: **not sparse** (none).
Sections represented: 46. Max section fraction: 0.0588.
Support by fold: 17/18/12/11/10.
Retrospective true-class mix (diagnostic only; not a trigger input): oligodendrocyte_progenitor_2=33, oligodendrocyte_1=25, oligodendrocyte_2=4, astrocyte_2=3, endothelial=2, astrocyte_1=1.

### H3

Observable trigger: `LZH_pred == oligodendrocyte_2 AND WYH_pred == oligodendrocyte_2 AND SNI_pred == oligodendrocyte_progenitor_2`.
Hypothetical action: replace strong consensus with `SNI` prediction `oligodendrocyte_progenitor_2`.
The trigger uses **predictions only**. It does not condition on true class.

| split | support | wrong→correct | correct→wrong | precision | net | strong acc | candidate acc | oracle acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 35 | 9 | 23 | 0.2812 | -14 | 0.6571 | 0.2571 | 0.9143 |
| folds 0-2 (development) | 19 | 5 | 11 | 0.3125 | -6 | 0.5789 | 0.2632 | 0.8421 |
| folds 3-4 (retrospective stability) | 16 | 4 | 12 | 0.2500 | -8 | 0.7500 | 0.2500 | 1.0000 |

Actionable under predeclared criteria: **no**.
Failed checks: wrong_to_correct_ge_10, precision_ge_065, net_gt_0, folds_0_2_net_gt_0, folds_3_4_net_gt_0, folds_0_2_precision_ge_060, folds_3_4_precision_ge_055.
Sparsity flag: **not sparse** (none).
Sections represented: 24. Max section fraction: 0.0857.
Support by fold: 9/9/1/5/11.
Retrospective true-class mix (diagnostic only; not a trigger input): oligodendrocyte_2=23, oligodendrocyte_progenitor_2=9, astrocyte_2=1, oligodendrocyte_1=1, oligodendrocyte_precursor_cell=1.

## 7. Oligo / OPC Direction Matrix

Matrix uses only cells with observable strong consensus (`LZH_pred == WYH_pred`) and ordered pairs among the frozen oligo/OPC family:

`oligodendrocyte_1`, `oligodendrocyte_2`, `oligodendrocyte_precursor_cell`, `oligodendrocyte_progenitor_1`, `oligodendrocyte_progenitor_2`.

This table is **descriptive**. Non-predeclared rows are **not** candidate rules. Highlighted H1/H2/H3 rows are the only directions eligible for E05A actionability.

Rows with support 0 are omitted below; the full 5×5 × two-expert matrix is in `outputs/v3/v3_e05a_oligo_direction_matrix.csv`.

| expert | strong consensus | proposed | support | w→c | c→w | net | precision | strong acc | cand acc | predeclared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0 | oligodendrocyte_1 | oligodendrocyte_1 | 130 | 0 | 0 | 0 | n/a | 0.6462 | 0.6462 |  |
| s0 | oligodendrocyte_1 | oligodendrocyte_2 | 2 | 0 | 1 | -1 | 0.0000 | 0.5000 | 0.0000 |  |
| s0 | oligodendrocyte_1 | oligodendrocyte_progenitor_2 | 14 | 5 | 8 | -3 | 0.3846 | 0.5714 | 0.3571 |  |
| s0 | oligodendrocyte_2 | oligodendrocyte_1 | 5 | 1 | 3 | -2 | 0.2500 | 0.6000 | 0.2000 |  |
| s0 | oligodendrocyte_2 | oligodendrocyte_2 | 410 | 0 | 0 | 0 | n/a | 0.8585 | 0.8585 |  |
| s0 | oligodendrocyte_2 | oligodendrocyte_precursor_cell | 1 | 1 | 0 | 1 | 1.0000 | 0.0000 | 1.0000 |  |
| s0 | oligodendrocyte_2 | oligodendrocyte_progenitor_2 | 16 | 4 | 10 | -6 | 0.2857 | 0.6250 | 0.2500 |  |
| s0 | oligodendrocyte_precursor_cell | oligodendrocyte_1 | 1 | 0 | 0 | 0 | n/a | 0.0000 | 0.0000 |  |
| s0 | oligodendrocyte_precursor_cell | oligodendrocyte_precursor_cell | 92 | 0 | 0 | 0 | n/a | 0.6957 | 0.6957 |  |
| s0 | oligodendrocyte_precursor_cell | oligodendrocyte_progenitor_1 | 2 | 1 | 1 | 0 | 0.5000 | 0.5000 | 0.5000 |  |
| s0 | oligodendrocyte_precursor_cell | oligodendrocyte_progenitor_2 | 1 | 0 | 1 | -1 | 0.0000 | 1.0000 | 0.0000 |  |
| s0 | oligodendrocyte_progenitor_1 | oligodendrocyte_precursor_cell | 2 | 1 | 1 | 0 | 0.5000 | 0.5000 | 0.5000 |  |
| s0 | oligodendrocyte_progenitor_1 | oligodendrocyte_progenitor_1 | 17 | 0 | 0 | 0 | n/a | 0.8824 | 0.8824 |  |
| s0 | oligodendrocyte_progenitor_2 | oligodendrocyte_1 | 68 | 25 | 33 | -8 | 0.4310 | 0.4853 | 0.3676 | H2 |
| s0 | oligodendrocyte_progenitor_2 | oligodendrocyte_2 | 12 | 2 | 8 | -6 | 0.2000 | 0.6667 | 0.1667 |  |
| s0 | oligodendrocyte_progenitor_2 | oligodendrocyte_precursor_cell | 1 | 1 | 0 | 1 | 1.0000 | 0.0000 | 1.0000 |  |
| s0 | oligodendrocyte_progenitor_2 | oligodendrocyte_progenitor_2 | 579 | 0 | 0 | 0 | n/a | 0.8066 | 0.8066 |  |
| sni | oligodendrocyte_1 | oligodendrocyte_1 | 49 | 0 | 0 | 0 | n/a | 0.5510 | 0.5510 |  |
| sni | oligodendrocyte_1 | oligodendrocyte_2 | 4 | 0 | 2 | -2 | 0.0000 | 0.5000 | 0.0000 |  |
| sni | oligodendrocyte_1 | oligodendrocyte_progenitor_2 | 88 | 25 | 55 | -30 | 0.3125 | 0.6250 | 0.2841 | H1 |
| sni | oligodendrocyte_2 | oligodendrocyte_2 | 393 | 0 | 0 | 0 | n/a | 0.8626 | 0.8626 |  |
| sni | oligodendrocyte_2 | oligodendrocyte_progenitor_2 | 35 | 9 | 23 | -14 | 0.2812 | 0.6571 | 0.2571 | H3 |
| sni | oligodendrocyte_precursor_cell | oligodendrocyte_1 | 4 | 0 | 2 | -2 | 0.0000 | 0.5000 | 0.0000 |  |
| sni | oligodendrocyte_precursor_cell | oligodendrocyte_precursor_cell | 81 | 0 | 0 | 0 | n/a | 0.7037 | 0.7037 |  |
| sni | oligodendrocyte_precursor_cell | oligodendrocyte_progenitor_1 | 1 | 1 | 0 | 1 | 1.0000 | 0.0000 | 1.0000 |  |
| sni | oligodendrocyte_precursor_cell | oligodendrocyte_progenitor_2 | 4 | 0 | 3 | -3 | 0.0000 | 0.7500 | 0.0000 |  |
| sni | oligodendrocyte_progenitor_1 | oligodendrocyte_precursor_cell | 16 | 1 | 15 | -14 | 0.0625 | 0.9375 | 0.0625 |  |
| sni | oligodendrocyte_progenitor_1 | oligodendrocyte_progenitor_1 | 1 | 0 | 0 | 0 | n/a | 1.0000 | 1.0000 |  |
| sni | oligodendrocyte_progenitor_1 | oligodendrocyte_progenitor_2 | 1 | 0 | 1 | -1 | 0.0000 | 1.0000 | 0.0000 |  |
| sni | oligodendrocyte_progenitor_2 | oligodendrocyte_1 | 8 | 2 | 5 | -3 | 0.2857 | 0.6250 | 0.2500 |  |
| sni | oligodendrocyte_progenitor_2 | oligodendrocyte_2 | 12 | 1 | 10 | -9 | 0.0909 | 0.8333 | 0.0833 |  |
| sni | oligodendrocyte_progenitor_2 | oligodendrocyte_progenitor_2 | 646 | 0 | 0 | 0 | n/a | 0.7647 | 0.7647 |  |

Any non-H1/H2/H3 pair that looks numerically attractive is recorded only as a **HYPOTHESIS FOR FUTURE WORK**, not as validated E05A evidence.

No non-predeclared oligo/OPC consensus→candidate pair met the descriptive filter (support ≥ 15, net ≥ 5, precision ≥ 0.60). Nothing is promoted.

## 8. S0 vs SNI Asymmetric Expertise

On the same observable H1/H2/H3 cells, are S0 and SNI complementary in opposite directions? Standalone accuracy is not the question.

| H | support | S0 correct | SNI correct | both correct | both wrong | S0-only | SNI-only | S0==SNI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | 88 | 49 | 25 | 5 | 19 | 44 | 20 | 14 |
| H2 | 68 | 25 | 33 | 2 | 12 | 23 | 31 | 4 |
| H3 | 35 | 18 | 9 | 1 | 9 | 17 | 8 | 7 |

On H1 cells, the designated SNI expert is not clearly superior to S0 (SNI-only 20, S0-only 44, net -30). On H2 cells, the designated S0 expert is not clearly superior to SNI (S0-only 23, SNI-only 31, net -8). H3 net=-14. Opposite-direction complementarity between S0 and SNI is not cleanly supported.

## 9. Strong-Consensus Failure Analysis

E03A established that weak-expert rescues often occur even when `LZH == WYH`. Strong agreement does not imply correctness on specific biological boundaries.

| Quantity | Value |
|---|---:|
| LZH == WYH cells | 4764 |
| Strong-consensus accuracy | 0.8447 |
| Strong-consensus errors | 740 |
| Rescued by S0 | 89 |
| Rescued by SNI | 81 |
| Rescued by either | 140 |
| Rescued by both | 30 |

Breakdown by strong consensus class (oligo/OPC classes plus any class with at least one consensus error):

| consensus class | n | errors | acc | S0 rescues | SNI rescues | either | both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oligodendrocyte_progenitor_2 | 707 | 173 | 0.7553 | 35 | 5 | 36 | 4 |
| astrocyte_1 | 597 | 112 | 0.8124 | 9 | 4 | 12 | 1 |
| oligodendrocyte_1 | 183 | 76 | 0.5847 | 8 | 28 | 29 | 7 |
| oligodendrocyte_2 | 438 | 71 | 0.8379 | 8 | 12 | 18 | 2 |
| endothelial | 298 | 50 | 0.8322 | 4 | 5 | 7 | 2 |
| oligodendrocyte_precursor_cell | 103 | 34 | 0.6699 | 3 | 2 | 3 | 2 |
| meninges_1 | 84 | 29 | 0.6548 | 6 | 6 | 9 | 3 |
| DH_in_Cdh3 | 141 | 21 | 0.8511 | 0 | 0 | 0 | 0 |
| DH_ex_Maf/Cck | 76 | 21 | 0.7237 | 1 | 0 | 1 | 0 |
| astrocyte_2 | 164 | 19 | 0.8841 | 1 | 4 | 4 | 1 |
| alpha_motoneuron | 116 | 16 | 0.8621 | 0 | 0 | 0 | 0 |
| Schwann_cell | 59 | 12 | 0.7966 | 1 | 1 | 1 | 1 |
| DH_ex_Tac2 | 36 | 12 | 0.6667 | 2 | 1 | 3 | 0 |
| DH_ex_Prkcg/Rxfp1 | 35 | 11 | 0.6857 | 0 | 0 | 0 | 0 |
| DH_ex_Prkcg/Nts | 46 | 10 | 0.7826 | 0 | 0 | 0 | 0 |
| DH_ex_Prkcg/Cck | 24 | 8 | 0.6667 | 1 | 3 | 3 | 1 |
| microglia | 51 | 7 | 0.8627 | 0 | 0 | 0 | 0 |
| DH_in_Npy2r | 39 | 5 | 0.8718 | 0 | 0 | 0 | 0 |
| meninges_3 | 13 | 5 | 0.6154 | 0 | 0 | 0 | 0 |
| beta_motoneuron | 12 | 5 | 0.5833 | 1 | 3 | 3 | 1 |
| DH_ex_Rreb1 | 5 | 5 | 0.0000 | 0 | 0 | 0 | 0 |
| DH_ex_Nmu/Tac2 | 66 | 4 | 0.9394 | 1 | 0 | 1 | 0 |
| DH_ex_Gpr83 | 31 | 4 | 0.8710 | 1 | 0 | 1 | 0 |
| DH_ex_Grpr | 29 | 4 | 0.8621 | 0 | 0 | 0 | 0 |
| M_ex_Vsx2/Shox2 | 25 | 3 | 0.8800 | 0 | 0 | 0 | 0 |

## 10. Non-Oligo Controls

These summaries test whether directional rescues of strong-consensus failures are concentrated in oligo/OPC rather than generic weak-expert complementarity. **No candidate patch rules are invented for these families.**

| family | n true | strong-consensus n | consensus errors | S0 rescues | SNI rescues | either |
| --- | --- | --- | --- | --- | --- | --- |
| oligodendrocyte_opc | 1552 | 1443 | 348 | 45 | 47 | 80 |
| astrocyte | 769 | 732 | 102 | 13 | 10 | 18 |
| vascular | 363 | 349 | 86 | 9 | 4 | 11 |
| meningeal | 128 | 109 | 32 | 6 | 6 | 8 |
| microglia | 64 | 60 | 16 | 6 | 4 | 7 |
| remaining_glial_non_neuronal | 82 | 77 | 13 | 1 | 1 | 1 |
| neuronal_or_other | 2042 | 1994 | 143 | 9 | 9 | 15 |

Oligo/OPC share of S0 strong-consensus rescues: 45 / 89.
Oligo/OPC share of SNI strong-consensus rescues: 47 / 81.

Oligo/OPC accounts for 348 / 740 strong-consensus errors (47.0%). S0 strong-consensus rescues in that family: 45 / 89 (50.6%). SNI strong-consensus rescues in that family: 47 / 81 (58.0%). Absolute unique-rescue mass is largest in oligo/OPC, but S0 is close to the error base rate and non-oligo families still contribute. This is family-skewed unique coverage, not a license to patch non-oligo directions. No non-oligo patch rule is proposed.

## 11. Confidence Secondary Diagnostics

Confidence is **not** used to modify H1/H2/H3. No threshold is chosen. Means are descriptive only, comparing wrong→correct versus correct→wrong cells inside each predeclared trigger.

| H | group | n | cand top1 | cand margin | cand entropy | LZH margin | WYH margin | LZH entropy | WYH entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | wrong_to_correct | 25 | 0.8043 | 0.6999 | 0.6167 | 0.2672 | 0.5103 | 0.9967 | 0.5607 |
| H1 | correct_to_wrong | 55 | 0.7896 | 0.6519 | 0.6235 | 0.3317 | 0.5176 | 0.9412 | 0.5859 |
| H2 | wrong_to_correct | 25 | 0.4496 | 0.1301 | 1.4473 | 0.3666 | 0.4814 | 0.9695 | 0.5993 |
| H2 | correct_to_wrong | 33 | 0.4888 | 0.1570 | 1.2965 | 0.3393 | 0.4014 | 1.0110 | 0.6832 |
| H3 | wrong_to_correct | 9 | 0.7437 | 0.5387 | 0.6900 | 0.4513 | 0.6197 | 0.9258 | 0.4923 |
| H3 | correct_to_wrong | 23 | 0.6670 | 0.3875 | 0.7323 | 0.4296 | 0.6267 | 0.8782 | 0.4646 |

H1 candidate margin mean is 0.700 on wrong→correct vs 0.652 on correct→wrong (difference +0.048); this is descriptive only and does not justify a threshold. H2 candidate margin mean is 0.130 on wrong→correct vs 0.157 on correct→wrong (difference -0.027); this is descriptive only and does not justify a threshold. H3 candidate margin mean is 0.539 on wrong→correct vs 0.388 on correct→wrong (difference +0.151); this is descriptive only and does not justify a threshold.

## 12. Hard-Bucket Analysis

Diagnostic only. No hard-bucket threshold or rule is created.

| H | bucket | support | w→c | c→w | net | precision |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | hard | 88 | 25 | 55 | -30 | 0.3125 |
| H1 | not-hard | 0 | 0 | 0 | 0 | n/a |
| H2 | hard | 68 | 25 | 33 | -8 | 0.4310 |
| H2 | not-hard | 0 | 0 | 0 | 0 | n/a |
| H3 | hard | 35 | 9 | 23 | -14 | 0.2812 |
| H3 | not-hard | 0 | 0 | 0 | 0 | n/a |

## 13. Retrospective Fixed-Patch Diagnostics D0-D3

**D0-D3 ARE NOT UNBIASED FINAL OOF RESULTS.**

They are **RETROSPECTIVE FIXED-PATCH DIAGNOSTICS**. H1/H2/H3 were motivated by prior full-data error analysis, so these numbers estimate whether the directional mechanism is promising enough to justify a separately implemented cross-fitted/predeclared E05B evaluation. They are **not** MODEL V3 and must not be frozen as a version score.

| System | Rule |
|---|---|
| D0 | LZH prediction only |
| D1 | LZH baseline + apply H1 only |
| D2 | LZH baseline + apply H1 and H2 |
| D3 | LZH baseline + apply H1, H2, and H3 |

No combination was optimized. No additional pattern was added to improve D3.

| system | changed | w→c | c→w | net | correct | accuracy | macro-F1 | folds 0-2 | folds 3-4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | 0 | 0 | 0 | 0 | 4133 | 0.8266 | 0.7977 | 0.8303 | 0.8210 |
| D1 | 88 | 25 | 55 | -30 | 4103 | 0.8206 | 0.7947 | 0.8247 | 0.8145 |
| D2 | 156 | 50 | 88 | -38 | 4095 | 0.8190 | 0.7956 | 0.8233 | 0.8125 |
| D3 | 191 | 59 | 111 | -52 | 4081 | 0.8162 | 0.7952 | 0.8213 | 0.8085 |

## 14. Sparsity / Stability Audit

Directional rules can overfit if trigger support is tiny or confined to one fold or one Section.

| H | support | by fold | n sections | n true classes | hard frac | max section frac | flag | reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | 88 | 16/19/15/24/14 | 51 | 6 | 1.0000 | 0.0568 | ok | none |
| H2 | 68 | 17/18/12/11/10 | 46 | 6 | 1.0000 | 0.0588 | ok | none |
| H3 | 35 | 9/9/1/5/11 | 24 | 5 | 1.0000 | 0.0857 | ok | none |

None of H1/H2/H3 is flagged as a single-fold or single-section artifact under the predeclared sparsity rules, though absolute support remains limited.

## 15. Leakage / Selection-Bias Audit

H1/H2/H3 were motivated by **prior frozen full-data analyses** (V3-E03A confusion-pair rescues and V3-E04S shared-failure recoveries). Those prior looks used the same 5000 training cells, including canonical folds 3-4. Therefore:

- E05A cannot claim an unbiased final OOF estimate for a directional patch.
- Canonical folds 3-4 are a retrospective stability partition, not a pristine holdout.
- D0-D3 must not be presented as MODEL V3 performance.

True labels were used only retrospectively to score predefined observable prediction patterns.

- competition test labels used: False
- true labels used to define H1/H2/H3 triggers: False
- true labels used only for retrospective scoring: True
- learned router/classifier trained: False
- threshold optimized: False
- ensemble weights optimized: False
- post-hoc pattern mining promoted to a candidate rule: False
- new external dataset added: False
- expert probabilities altered: False
- S0 / SNI / LZH / WYH retrained: False
- Spatial-ID started: False
- MODEL V3 created: False
- submission generated: False
- prediction/prediction.csv modified: False
- V3-E00T / E02D / E03A / E04S artifacts modified: False

## 16. What a Future E05B May Use

Only observable, predeclared prediction-direction patterns, for example:

- LZH predicted class
- WYH predicted class
- S0 predicted class
- SNI predicted class
- whether LZH and WYH agree
- the H1/H2/H3 triggers exactly as frozen here
- test-safe metadata as covariates or stratum descriptors, not as newly searched rules

E05A does **not** by itself justify starting E05B. If a later experiment still implements a directional patch, it should be cross-fitted or otherwise predeclared so that trigger evaluation is not scored on the same full-data look that motivated H1/H2/H3.

## 17. What a Future E05B Must NOT Use

- true labels at inference
- expert correctness flags
- oracle membership
- rescue flags
- a confusion direction defined using the true class ("if the true cell is progenitor_2")
- post-hoc mining of prediction tuples on full OOF
- confidence thresholds selected on full data
- arbitrary ensemble-weight search
- a learned router fit on the same cells used to report the score

## 18. Decision

**DIRECTIONAL SIGNAL WEAK**

None of H1/H2/H3 provides stable positive net correction under the predeclared actionable criteria. Each observable trigger mixes both sides of an oligo/OPC confusion: H1 net -30 (precision 0.3125), H2 net -8 (precision 0.4310), H3 net -14 (precision 0.2812). Unconditional replacement of strong consensus therefore harms more cells than it rescues on both canonical folds 0-2 and the retrospective stability partition.

Recommended next action: **Move to a final conservative model-selection / integration strategy. Do not train a learned router.**

Do not start that experiment in this task. MODEL V3 is not frozen. No `docs/versions/model_v3.md` and no submission candidate were created.
