# V3-E03A — Weak-but-Diverse Expert Rescue Audit

Research codename: **CER-AUDIT**. This is a MODEL V3 research experiment, not a formal MODEL V3 freeze.

## 1. Research Question

Can test-safe reliability, confidence, disagreement, biological-family, and metadata signals identify the small subset of cells where the weak-but-diverse S0 expert should be trusted, while abstaining everywhere else?

S0 is not treated as a globally competitive classifier. It is a **WEAK-BUT-DIVERSE RESCUE EXPERT**. This audit does not implement a deployment gate.

Diagnostic oracle accuracy is **not** deployable model accuracy.

## 2. Frozen Starting Evidence

| Quantity | Value |
|---|---|
| LZH Prior-H | 4133 / 5000 = 0.8266 |
| WYH MODEL V2 | 4106 / 5000 = 0.8212 |
| LZH + WYH diagnostic oracle | 4215 / 5000 = 0.8430 |
| Both LZH and WYH wrong | 785 |
| S0 hard-label 200-gene reference MLP | 3151 / 5000 = 0.6302 |
| S0 unique recoveries | 96 |
| LZH + WYH + S0 diagnostic oracle | 4311 / 5000 = 0.8622 |
| All three wrong | 689 |

S0 unique recoveries by canonical fold: 25 / 28 / 16 / 17 / 10. Folds 0-2: 69. Locked folds 3-4: 27.

`0.8622` is a diagnostic oracle, **NOT** deployable accuracy.

V3-E00T and V3-E02D artifacts were read only. Distillation was not continued. S0 was not retrained.

## 3. Why S0 Is Not a Global Classifier Candidate

S0 standalone accuracy is **0.6302**, far below LZH (0.8266) and WYH MODEL V2 (0.8212). Unconstrained replacement of either strong expert by S0 would destroy accuracy.

S0's project value is complementarity: it is uniquely correct on **96** cells that both strong experts miss. The research question is whether those 96 cells can be recognized with test-safe signals.

## 4. Integrity Reproduction

All predeclared counts reproduced exactly from the joined three-expert registry:

| Check | Expected | Reproduced |
|---|---:|---:|
| LZH correct | 4133 | 4133 |
| WYH correct | 4106 | 4106 |
| S0 correct | 3151 | 3151 |
| LZH + WYH oracle | 4215 | 4215 |
| both strong wrong | 785 | 785 |
| positive S0 rescues | 96 | 96 |
| three-expert oracle | 4311 | 4311 |
| all three wrong | 689 | 689 |
| fold rescues | 25/28/16/17/10 | 25/28/16/17/10 |

Registry rows: **5000**. Unique 19-digit `Cell_ID` strings: **5000**. Labels aligned to official `meta_train`.

S0 60-class probabilities were recovered by deterministic CPU inference from frozen `work/v3_e02d/s0.pt` (SHA256 `5451c7a53ff3ca88dd4d83f7cebcc2920d7bd4bbc38073a6326f048f92111fd0`). Frozen S0 hard labels matched on 5000 / 5000 cells. No retraining occurred.

## 5. The 96 Unique S0 Rescues

GROUP P is defined as LZH wrong AND WYH wrong AND S0 correct. Count: **96**.

| true class | n | family |
| --- | --- | --- |
| oligodendrocyte_1 | 31 | oligodendrocyte_opc |
| oligodendrocyte_progenitor_2 | 11 | oligodendrocyte_opc |
| astrocyte_1 | 9 | astrocyte |
| endothelial | 7 | vascular |
| microglia | 7 | microglia |
| astrocyte_2 | 5 | astrocyte |
| meninges_3 | 4 | meningeal |
| pericyte | 3 | vascular |
| DH_ex_Prkcg/Nts | 3 | neuronal_or_other |
| oligodendrocyte_precursor_cell | 3 | oligodendrocyte_opc |
| oligodendrocyte_2 | 2 | oligodendrocyte_opc |
| alpha_motoneuron | 2 | neuronal_or_other |
| oligodendrocyte_progenitor_1 | 2 | oligodendrocyte_opc |
| DH_ex_Cpne4 | 1 | neuronal_or_other |
| meninges_2 | 1 | meningeal |

Hard-bucket fraction of the 96: **0.9062** (87 / 96).

Canonical-fold counts: 25/28/16/17/10.

## 6. Dangerous Override Populations

| Group | Definition | Count |
|---|---|---:|
| N1 | LZH wrong AND WYH wrong AND S0 wrong | 689 |
| N2 | LZH correct AND WYH correct AND S0 wrong | 1043 |
| N3 | at least one of LZH/WYH correct AND S0 wrong | 1160 |

N1 is the failed-rescue-opportunity set. N2 is the strongest dangerous-override set. N3 is the full dangerous-override set for any future S0 replacement of a currently correct strong expert.

S0 wrong total = N1 + N3 = 1849.

## 7. LZH / WYH Agreement-State Analysis

| state | n | LZH acc | WYH acc | S0 acc | P count | P rate | N3 | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: LZH == WYH | 4764 | 0.8447 | 0.8447 | 0.6444 | 89 | 0.0187 | 1043 | 0.8634 |
| B: LZH != WYH | 236 | 0.4619 | 0.3475 | 0.3432 | 7 | 0.0297 | 117 | 0.8390 |

When LZH == WYH AND both are wrong: S0 is correct on **89** / **740** cells (0.1203).

When LZH != WYH: S0 is correct on **81** / **236** cells (0.3432). Positive S0 rescues in this state: **7**.

When LZH != WYH: at least one strong expert is already correct on **191** / **236** cells (0.8093).

Interpretation: Most S0 unique rescues are not concentrated in strong-expert disagreement. Future rescue, if any, must handle rare shared-confidence failures rather than only LZH/WYH disagreement.

## 8. Hard-Bucket Analysis

| hard_bucket | n | LZH acc | WYH acc | S0 acc | P count | P rate | N3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| True | 2958 | 0.7620 | 0.7569 | 0.7049 | 87 | 0.0294 | 322 |
| False | 2042 | 0.9202 | 0.9143 | 0.5220 | 9 | 0.0044 | 838 |

Fraction of the 96 positive rescues in the hard bucket: **0.9062**.

## 9. Biological Family Rescue Map

E03A family mapping keeps the frozen E02D oligo/OPC, astrocyte, vascular, and meningeal definitions, then splits `microglia` out of the remaining glial/non-neuronal group.

E02D-mapping reproduction of S0 unique recoveries: oligodendrocyte_opc **49**, vascular **10**.

| family | n | shared errors | S0 rescues | rescue rate | N3 |
| --- | --- | --- | --- | --- | --- |
| oligodendrocyte_opc | 1552 | 365 | 49 | 0.1342 | 162 |
| astrocyte | 769 | 114 | 14 | 0.1228 | 99 |
| vascular | 363 | 92 | 10 | 0.1087 | 28 |
| meningeal | 128 | 36 | 6 | 0.1667 | 24 |
| microglia | 64 | 18 | 7 | 0.3889 | 4 |
| remaining_glial_non_neuronal | 82 | 13 | 1 | 0.0769 | 5 |
| neuronal_or_other | 2042 | 147 | 9 | 0.0612 | 838 |

## 10. Confusion-Pair Rescue Map

Among the 96 rescues, LZH_pred == WYH_pred (both wrong, S0 correct): **89**.

LZH_pred != WYH_pred and S0 correct: **7**.

Top true → LZH_pred pairs among GROUP P:

| true | LZH pred | n |
| --- | --- | --- |
| oligodendrocyte_1 | oligodendrocyte_progenitor_2 | 26 |
| oligodendrocyte_progenitor_2 | oligodendrocyte_1 | 5 |
| microglia | oligodendrocyte_progenitor_2 | 4 |
| astrocyte_2 | astrocyte_1 | 4 |
| oligodendrocyte_progenitor_2 | oligodendrocyte_2 | 4 |
| oligodendrocyte_1 | oligodendrocyte_2 | 4 |
| meninges_3 | meninges_1 | 3 |
| astrocyte_1 | endothelial | 3 |

Top true → WYH_pred pairs among GROUP P:

| true | WYH pred | n |
| --- | --- | --- |
| oligodendrocyte_1 | oligodendrocyte_progenitor_2 | 28 |
| oligodendrocyte_progenitor_2 | oligodendrocyte_1 | 5 |
| astrocyte_2 | astrocyte_1 | 4 |
| oligodendrocyte_progenitor_2 | oligodendrocyte_2 | 4 |
| meninges_3 | meninges_1 | 3 |
| astrocyte_1 | endothelial | 3 |
| microglia | oligodendrocyte_progenitor_2 | 3 |
| endothelial | oligodendrocyte_1 | 2 |

Interpretation: Most unique S0 rescues occur when LZH and WYH make the same wrong prediction. That favors a global or family-aware rescue of shared-confidence failures over a pure disagreement router.

## 11. Test-Safe Reliability Features

Candidate test-safe features were restricted to quantities observable without held-out labels: expert confidence, predicted-class agreement, probability advantage, JS divergence, library size, detected-gene count, hard-bucket / metadata missingness, and the single predeclared composite margin score.

True labels, correctness flags, and rescue flags are **DIAGNOSTIC-ONLY**. They are not future deployment features.

LZH Prior-H eligibility, graph degree, and reliable-gene count remain **UNAVAILABLE** at cell level, as in V3-E00T.

S0 60-class probabilities were recovered by deterministic CPU inference from frozen `work/v3_e02d/s0.pt` (SHA256 `5451c7a53ff3ca88dd4d83f7cebcc2920d7bd4bbc38073a6326f048f92111fd0`). Frozen S0 hard labels matched on 5000 / 5000 cells. No retraining occurred.

## 12. Diagnostic Separability

AUROC values below are diagnostic only. No classifier was fit and no threshold was chosen.

| feature | P mean | N1 mean | N3 mean | AUROC P vs N1 | AUROC P vs N3 |
| --- | --- | --- | --- | --- | --- |
| s0_top1 | 0.4786 | 0.5441 | 0.4740 | 0.4142 | 0.5263 |
| s0_margin | 0.1956 | 0.3412 | 0.2711 | 0.3355 | 0.4250 |
| s0_entropy | 1.3933 | 1.4080 | 1.6551 | 0.5002 | 0.3766 |
| lzh_top1 | 0.6075 | 0.7048 | 0.8383 | 0.3214 | 0.1816 |
| lzh_margin | 0.3509 | 0.5286 | 0.7243 | 0.3022 | 0.1910 |
| lzh_entropy | 1.0319 | 0.8636 | 0.4314 | 0.6365 | 0.8394 |
| wyh_top1 | 0.7283 | 0.8235 | 0.9022 | 0.3243 | 0.1798 |
| wyh_margin | 0.5119 | 0.6902 | 0.8238 | 0.3177 | 0.1810 |
| wyh_entropy | 0.6479 | 0.4688 | 0.2348 | 0.6573 | 0.8256 |
| lzh_wyh_agree | 0.9271 | 0.9448 | 0.8991 | 0.4911 | 0.5140 |
| lzh_s0_agree | 0.0000 | 0.6880 | 0.0336 | 0.1560 | 0.4832 |
| wyh_s0_agree | 0.0000 | 0.6749 | 0.0164 | 0.1626 | 0.4918 |
| all_three_agree | 0.0000 | 0.6604 | 0.0000 | 0.1698 | 0.5000 |
| all_three_different | 0.0729 | 0.0131 | 0.0509 | 0.5299 | 0.5110 |
| n_experts_supporting_s0 | 1.0000 | 2.3628 | 1.0500 | 0.1488 | 0.4750 |
| n_strong_supporting_s0 | 0.0000 | 1.3628 | 0.0500 | 0.1488 | 0.4750 |
| s0_prob_advantage_lzh | -0.1289 | -0.1607 | -0.3643 | 0.5263 | 0.7627 |
| s0_prob_advantage_wyh | -0.2497 | -0.2794 | -0.4282 | 0.5301 | 0.7180 |
| js_s0_lzh | 0.1946 | 0.2390 | 0.6159 | 0.5660 | 0.1261 |
| js_s0_wyh | 0.2667 | 0.2846 | 0.6453 | 0.5651 | 0.1285 |
| js_lzh_wyh | 0.0707 | 0.0608 | 0.0376 | 0.5255 | 0.7330 |
| n_detected | 10.7812 | 11.2961 | 15.6759 | 0.4628 | 0.3308 |
| library_size | 22.7917 | 22.5022 | 31.0560 | 0.5352 | 0.4244 |
| hard_bucket | 0.9062 | 0.7997 | 0.2776 | 0.5533 | 0.8143 |
| rescue_evidence_score | 0.4804 | 0.2504 | -0.5779 | 0.5608 | 0.7511 |

Summary: P vs N3 (dangerous override of a currently correct strong expert) is where LZH/WYH low-confidence and hard-bucket missingness show the strongest diagnostic AUROCs. P vs N1 (both strong experts already wrong, S0 also wrong) is the operational bottleneck: S0 top1/margin do not identify the 96 unique rescues among the 785 shared errors, and high S0 confidence deciles contain almost none of the 96 rescues. A cue that finds shared-confidence failures still leaves about a 12% S0 hit rate on those cells, which is unique-oracle headroom rather than a high-precision rescue regime.

## 13. Quantile Diagnostics

All 5000 cells were split into fixed feature-value deciles. Deciles were not optimized and are not candidate rules.

### s0_top1

| decile | n | rescues | rescue rate | N3 | N3 rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 10 | 0.0200 | 275 | 0.5500 |
| 2 | 500 | 20 | 0.0400 | 227 | 0.4540 |
| 3 | 500 | 30 | 0.0600 | 163 | 0.3260 |
| 4 | 500 | 15 | 0.0300 | 151 | 0.3020 |
| 5 | 500 | 11 | 0.0220 | 114 | 0.2280 |
| 6 | 500 | 6 | 0.0120 | 111 | 0.2220 |
| 7 | 500 | 3 | 0.0060 | 65 | 0.1300 |
| 8 | 500 | 1 | 0.0020 | 32 | 0.0640 |
| 9 | 500 | 0 | 0.0000 | 15 | 0.0300 |
| 10 | 500 | 0 | 0.0000 | 7 | 0.0140 |

### s0_margin

| decile | n | rescues | rescue rate | N3 | N3 rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 23 | 0.0460 | 254 | 0.5080 |
| 2 | 500 | 25 | 0.0500 | 224 | 0.4480 |
| 3 | 500 | 16 | 0.0320 | 157 | 0.3140 |
| 4 | 500 | 15 | 0.0300 | 140 | 0.2800 |
| 5 | 500 | 8 | 0.0160 | 149 | 0.2980 |
| 6 | 500 | 5 | 0.0100 | 97 | 0.1940 |
| 7 | 500 | 3 | 0.0060 | 83 | 0.1660 |
| 8 | 500 | 1 | 0.0020 | 33 | 0.0660 |
| 9 | 500 | 0 | 0.0000 | 15 | 0.0300 |
| 10 | 500 | 0 | 0.0000 | 8 | 0.0160 |

### s0_entropy

| decile | n | rescues | rescue rate | N3 | N3 rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 0 | 0.0000 | 5 | 0.0100 |
| 2 | 500 | 0 | 0.0000 | 16 | 0.0320 |
| 3 | 500 | 1 | 0.0020 | 24 | 0.0480 |
| 4 | 500 | 11 | 0.0220 | 73 | 0.1460 |
| 5 | 500 | 9 | 0.0180 | 106 | 0.2120 |
| 6 | 500 | 21 | 0.0420 | 126 | 0.2520 |
| 7 | 500 | 16 | 0.0320 | 153 | 0.3060 |
| 8 | 500 | 19 | 0.0380 | 162 | 0.3240 |
| 9 | 500 | 12 | 0.0240 | 228 | 0.4560 |
| 10 | 500 | 7 | 0.0140 | 267 | 0.5340 |

### lzh_margin

| decile | n | rescues | rescue rate | N3 | N3 rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 40 | 0.0800 | 186 | 0.3720 |
| 2 | 500 | 30 | 0.0600 | 112 | 0.2240 |
| 3 | 500 | 17 | 0.0340 | 82 | 0.1640 |
| 4 | 500 | 7 | 0.0140 | 79 | 0.1580 |
| 5 | 500 | 2 | 0.0040 | 66 | 0.1320 |
| 6 | 500 | 0 | 0.0000 | 80 | 0.1600 |
| 7 | 500 | 0 | 0.0000 | 70 | 0.1400 |
| 8 | 500 | 0 | 0.0000 | 84 | 0.1680 |
| 9 | 946 | 0 | 0.0000 | 400 | 0.4228 |
| 10 | 54 | 0 | 0.0000 | 1 | 0.0185 |

### wyh_margin

| decile | n | rescues | rescue rate | N3 | N3 rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 41 | 0.0820 | 168 | 0.3360 |
| 2 | 500 | 33 | 0.0660 | 121 | 0.2420 |
| 3 | 500 | 12 | 0.0240 | 97 | 0.1940 |
| 4 | 500 | 8 | 0.0160 | 71 | 0.1420 |
| 5 | 500 | 2 | 0.0040 | 77 | 0.1540 |
| 6 | 500 | 0 | 0.0000 | 74 | 0.1480 |
| 7 | 500 | 0 | 0.0000 | 66 | 0.1320 |
| 8 | 500 | 0 | 0.0000 | 91 | 0.1820 |
| 9 | 1000 | 0 | 0.0000 | 395 | 0.3950 |

### rescue_evidence_score

| decile | n | rescues | rescue rate | N3 | N3 rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 0 | 0.0000 | 310 | 0.6200 |
| 2 | 500 | 4 | 0.0080 | 220 | 0.4400 |
| 3 | 500 | 10 | 0.0200 | 145 | 0.2900 |
| 4 | 500 | 14 | 0.0280 | 98 | 0.1960 |
| 5 | 500 | 13 | 0.0260 | 77 | 0.1540 |
| 6 | 500 | 9 | 0.0180 | 61 | 0.1220 |
| 7 | 500 | 7 | 0.0140 | 27 | 0.0540 |
| 8 | 500 | 1 | 0.0020 | 33 | 0.0660 |
| 9 | 500 | 13 | 0.0260 | 42 | 0.0840 |
| 10 | 500 | 25 | 0.0500 | 147 | 0.2940 |

## 14. Canonical Fold Stability

Development analysis uses canonical folds 0-2 (69 rescues). Locked confirmation uses folds 3-4 (27 rescues). Nothing was tuned on folds 3-4.

| split | n | P | N1 | N3 | S0 acc | oracle |
| --- | --- | --- | --- | --- | --- | --- |
| folds 0-2 | 3000 | 69 | 399 | 682 | 0.6397 | 0.8670 |
| folds 3-4 | 2000 | 27 | 290 | 478 | 0.6160 | 0.8550 |

Direction stability of the strongest available scalar signals: Folds 0-2 vs 3-4 keep the unique-rescue identity 69 vs 27 and oligodendrocyte/OPC remains the largest rescue family by count. The oligo/OPC rescue rate among shared errors is not stable (0.176 on folds 0-2 vs 0.069 on folds 3-4). LZH/WYH low-confidence vs N3 keeps the same direction on both splits; S0 high-confidence does not isolate GROUP P on either split.

## 15. Rescue Opportunity Ceilings

These are oracle-style diagnostic ceilings, **not** deployable scores.

| restriction | unique recoveries | note |
| --- | --- | --- |
| A | 96 | all positive unique S0 rescues |
| B | 7 | only LZH != WYH |
| C | 89 | only LZH == WYH |
| D | 87 | only hard-bucket cells |
| E | 49 | only oligodendrocyte / OPC |
| F | 14 | only astrocyte |
| G | 10 | only vascular / endothelial |
| H | 11 | S0 top1 > LZH top1 and S0 top1 > WYH top1 |
| I | 13 | S0 margin > LZH margin and S0 margin > WYH margin |

Restriction J is not a count. LZH/WYH confidence among GROUP P versus the 5000-cell population:

GROUP P LZH margin mean 0.3509 (population 0.7387); WYH margin mean 0.5119 (population 0.8603). No threshold was chosen.

## 16. What a Future Gate Should and Should Not Use

**Should consider only if a later experiment is justified:** LZH/WYH low confidence and metadata-missingness as negative controls against overriding a currently correct strong expert. Do not use high S0 confidence as a rescue trigger: the 96 unique rescues sit in low-to-mid S0 confidence. Do not treat LZH/WYH disagreement as the primary rescue state: 89 / 96 unique rescues occur when LZH == WYH.

**Should not use:** true labels; correctness / rescue flags; any threshold chosen on full OOF; blend-weight search; S0 as a global replacement; LZH Prior-H eligibility / graph degree / reliable-gene count until cell-level OOF fields exist; folds 3-4 for redesign.

## 17. Leakage Audit

- competition test labels used: False
- true labels used only for retrospective diagnostics: True
- learned gate trained: False
- threshold optimized: False
- weights optimized: False
- leaderboard feedback used: False
- S0 retrained: False
- folds 3-4 used for redesign: False
- submission generated: False
- V3-E00T / V3-E02D numerical artifacts modified: False
- prediction/prediction.csv modified: False

## 18. Limitations

- S0 is an external-reference MLP, not competition OOF; its errors are not fold-exchangeable with LZH/WYH in the same protocol
- LZH uses a native 3-fold protocol; canonical seed42 folds are a meta-analysis partition
- YHH and current team-main remain unavailable for honest cell-level OOF
- Diagnostic AUROC on 96 vs 689 or 96 vs N3 is a small-N characterization
- A diagnostic oracle of 0.8622 is not a model score
- No spatial, graph-degree, or Prior-H eligibility cell-level fields were available

## 19. Decision

**RESCUE SIGNAL WEAK**

S0 adds unique oracle headroom (96 cells; diagnostic three-expert oracle 0.8622), but available test-safe signals do not isolate those rescues from dangerous overrides with fold-stable high precision.

Recommended next experiment: **Create another independent expert rather than a complex S0 rescue router. Do not start V3-E03B or V3-E03F on the current S0 reliability evidence.**

Do not start that experiment in this task. MODEL V3 is not frozen. No `docs/versions/model_v3.md` and no submission candidate were created.
