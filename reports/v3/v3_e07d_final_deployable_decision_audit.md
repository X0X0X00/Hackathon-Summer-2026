# V3-E07D — Final Deployable Candidate Decision Audit

Research codename: **FDC-AUDIT**. This experiment is ANALYSIS ONLY. It does not train a model, router, or stacker; does not blend experts; does not optimize ensemble weights, thresholds, or source weights; does not add a dataset; does not start Spatial-ID; does not create M3; does not modify test predictions; and does not freeze MODEL V3.

## 1. Decision Question

Which currently auditable standalone expert is strongest for deployment? Does M2 provide enough stable evidence to replace frozen personal MODEL V2? Is promotion to MODEL V3 scientifically justified? What should be frozen as the final conclusion of the current V3 research program?

Oracle values below are diagnostic coverage ceilings. **ORACLE != DEPLOYABLE ACCURACY.**

## 2. Frozen V3 Research Context

The current program is a sequence of controlled experiments, not a version freeze:

| Experiment | Result |
|---|---|
| V3-E00T Team Canonical Expert & Complementarity Audit | ROUTING-ONLY INSUFFICIENT for the then-auditable LZH+WYH pool; diagnostic two-expert oracle 0.8430 |
| V3-E02D Privileged-Gene Dual-Level Distillation | PROMISING BUT INSUFFICIENT; 500-gene teacher signal exists, but the 200-gene student was not a mature third expert |
| V3-E03A Weak-but-Diverse Expert Rescue Audit | RESCUE SIGNAL WEAK; S0 unique recoveries exist but test-safe confidence cannot identify them |
| V3-E04S Source-Diverse SNI Expert | STRONG SOURCE-DIVERSE EXPERT as a diversity/oracle member (standalone 0.5682); 53 unique recoveries |
| V3-E05A Asymmetric Directional Complementarity Audit | DIRECTIONAL SIGNAL WEAK; predeclared S0/SNI directional overrides had negative net correction |
| V3-E06M Source-Balanced Multi-Reference Transfer | PROMISING SOURCE-BALANCED TRANSFER, not STRONG; M2 = 4109 vs M0 = 4106, net +3, folds 3-4 net -9 |

Frozen personal versions remain MODEL V1 (historical) and MODEL V2 (current deployable). MODEL V3 is not defined.

## 3. Candidate Eligibility

Primary personal comparison: frozen WYH MODEL V2 (**M0**) versus V3-E06M source-balanced MERFISH+SNI (**M2**).

Team-level auditable standalone comparison additionally includes **LZH Prior-H**.

Excluded from standalone ranking:

| Candidate | Why excluded |
|---|---|
| S0 (accuracy 0.6302) | Diversity / oracle-analysis expert only; not a deployable standalone candidate |
| SNI (accuracy 0.5682) | Diversity / oracle-analysis expert only; not a deployable standalone candidate |
| team main v9 | Test/submission-oriented without the same auditable cell-level OOF protocol |
| YHH | No previously unavailable auditable 5000-cell YHH probability or prediction artifact is present and provenance-verified. YHH is excluded; it is not reconstructed. |

No YHH reconstruction was attempted.

## 4. Integrity Reproduction

All frozen identity checks reproduced exactly before any decision audit:

| Check | Expected | Reproduced |
|---|---|---|
| M0 / MODEL V2 correct | 4106 | 4106 |
| M0 accuracy | 0.8212 | 0.8212 |
| M2 correct | 4109 | 4109 |
| M2 accuracy | 0.8218 | 0.8218 |
| M2 vs M0 wrong→correct | 105 | 105 |
| M2 vs M0 correct→wrong | 102 | 102 |
| M2 vs M0 net | 3 | 3 |
| Folds 0-2 net | 12 | 12 |
| Folds 3-4 net | -9 | -9 |
| LZH correct | 4133 | 4133 |
| Four-expert oracle | 4364 | 4364 |
| Five-expert oracle | 4393 | 4393 |
| M2 new unique recoveries | 29 | 29 |

Re-evaluating the frozen E06M STRONG classifier on these reproduced numbers returns **PROMISING SOURCE-BALANCED TRANSFER**, not STRONG SOURCE-BALANCED TRANSFER. Failed STRONG checks: net_ge_25, acc_ge_0.8262, folds_3_4_net_positive, sni_capture_ge_15.

E07D does not lower the promotion bar after observing E06M. The frozen STRONG thresholds remain net correction ≥ 25, accuracy ≥ 0.8262, positive net on both folds 0-2 and folds 3-4, and the remaining predeclared SNI-capture / new-unique / macro-F1 conditions.

## 5. Overall Standalone Performance

| Model | Correct | Accuracy | Macro-F1 | Log loss |
|---|---|---|---|---|
| M0 / MODEL V2 | 4106 | 0.8212 | 0.7936 | 0.6416 |
| M2 / V3-E06M | 4109 | 0.8218 | 0.7955 | 0.6222 |
| LZH Prior-H | 4133 | 0.8266 | 0.7977 | 0.5411 |

LZH Prior-H has the highest auditable standalone accuracy. M2 is +3 cells versus M0 and remains below LZH.

## 6. Fold Stability

Canonical analysis partition: `experiments/team_folds_5_seed42.csv`. Folds 3-4 are a retrospective stability partition, not an untouched holdout.

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Folds 0-2 | Folds 3-4 | Mean | Std | Min | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 0.822 | 0.831 | 0.812 | 0.821 | 0.820 | 0.8217 | 0.8205 | 0.8212 | 0.0068 | 0.8120 | 0.8310 |
| M2 | 0.827 | 0.829 | 0.821 | 0.816 | 0.816 | 0.8257 | 0.8160 | 0.8218 | 0.0061 | 0.8160 | 0.8290 |
| LZH | 0.828 | 0.836 | 0.827 | 0.824 | 0.818 | 0.8303 | 0.8210 | 0.8266 | 0.0065 | 0.8180 | 0.8360 |

M2 improves folds 0-2 versus M0 (net +12 cells) and regresses folds 3-4 (net -9 cells). LZH remains ahead of both personal candidates on folds 0-2 and does not share M2's folds 3-4 regression versus M0.

## 7. Standard Deployment Slices

Slice comparison is deployment characterization only. No model is chosen from a post-hoc favorable slice.

| Slice | n | M0 | M2 | LZH | M2−M0 | LZH−M0 | LZH−M2 |
|---|---|---|---|---|---|---|---|
| hard bucket | 2958 | 0.7569 | 0.7569 | 0.7620 | +0.0000 | +0.0051 | +0.0051 |
| non-hard bucket | 2042 | 0.9143 | 0.9158 | 0.9202 | +0.0015 | +0.0059 | +0.0044 |
| neuron | 1858 | 0.9198 | 0.9214 | 0.9252 | +0.0016 | +0.0054 | +0.0038 |
| glial / non-neuronal | 3142 | 0.7629 | 0.7629 | 0.7683 | +0.0000 | +0.0054 | +0.0054 |
| oligodendrocyte / OPC | 1552 | 0.7326 | 0.7326 | 0.7378 | +0.0000 | +0.0052 | +0.0052 |
| astrocyte | 769 | 0.8336 | 0.8375 | 0.8375 | +0.0039 | +0.0039 | +0.0000 |
| vascular / endothelial | 363 | 0.7355 | 0.7273 | 0.7355 | -0.0083 | +0.0000 | +0.0083 |
| meningeal | 128 | 0.6641 | 0.6250 | 0.6562 | -0.0391 | -0.0078 | +0.0312 |
| microglia | 64 | 0.6875 | 0.7344 | 0.7188 | +0.0469 | +0.0312 | -0.0156 |
| remaining glial | 82 | 0.7927 | 0.8171 | 0.8293 | +0.0244 | +0.0366 | +0.0122 |
| neuronal / other | 2042 | 0.9143 | 0.9158 | 0.9202 | +0.0015 | +0.0059 | +0.0044 |

## 8. M2 vs M0 Paired Cell-Level Audit

This is the primary personal-version comparison.

- Changed predictions: **252**
- M2 wrong → correct: **105**
- M2 correct → wrong: **102**
- Net: **+3**
- Discordant correctness count: **207**
- Exact two-sided McNemar p-value (`scipy.stats.binomtest` on discordant cells): **0.8895**

A non-significant result does **not** prove equivalence. It means the observed +3-cell advantage does not provide strong paired evidence that M2 is superior. Offset errors remain almost as large as the recoveries: 105 vs 102.

## 9. Paired Bootstrap Uncertainty

Deterministic seed `20260819`, 10000 paired cell-level bootstrap resamples of accuracy(M2) − accuracy(M0). Diagnostic uncertainty only. Not used to invent a new promotion threshold.

| Quantity | Value |
|---|---|
| Mean delta | +0.000611 |
| Median delta | +0.000600 |
| 2.5th percentile | -0.005000 |
| 97.5th percentile | +0.006200 |
| Fraction delta > 0 | 0.5717 |
| Fraction delta = 0 | 0.0247 |
| Fraction delta < 0 | 0.4036 |

The 95% percentile interval includes 0. Bootstrap agreement with a tiny positive mean does not convert PROMISING E06M evidence into STRONG promotion evidence.

## 10. Section-Level Robustness

Cells are grouped by `Section_ID`. Per-section M2−M0 accuracy deltas:

| Quantity | Value |
|---|---|
| Sections | 108 |
| M2 > M0 | 31 |
| M2 = M0 | 45 |
| M2 < M0 | 32 |
| Median per-section accuracy delta | +0.000000 |
| IQR | 0.026760 |
| Minimum | -0.166667 |
| Maximum | +0.142857 |

Section-cluster bootstrap (seed `20260819`, 10000 resamples of Sections with replacement):

| Quantity | Value |
|---|---|
| Mean delta | +0.000621 |
| Median delta | +0.000572 |
| 2.5th percentile | -0.005371 |
| 97.5th percentile | +0.006870 |
| Fraction > 0 | 0.5594 |
| Fraction < 0 | 0.4153 |

M2's overall +3-cell gain is **not a broad, section-stable improvement**. Wins and losses both exist; the cluster-bootstrap interval includes 0. The gain is small and compatible with concentrated / unstable section-level fluctuation.

## 11. Classwise / Rare-Class Stability

No class-specific routing or patch rule is created.

Classes where M2 improves over M0: **18**. Tied: **28**. Worsens: **14**.

Top 10 M2 recall gains vs M0:

| class | support | M0 recall | M2 recall | delta |
|---|---|---|---|---|
| meninges_3 | 19 | 0.4737 | 0.6316 | +0.1579 |
| DH_ex_Rreb1 | 9 | 0.1111 | 0.2222 | +0.1111 |
| ependymal | 10 | 0.7000 | 0.8000 | +0.1000 |
| DH_ex_Maf/Cck | 64 | 0.8750 | 0.9531 | +0.0781 |
| beta_motoneuron | 14 | 0.5714 | 0.6429 | +0.0714 |
| microglia | 64 | 0.6875 | 0.7344 | +0.0469 |
| DH_ex_Cpne4 | 26 | 0.8846 | 0.9231 | +0.0385 |
| M_ex_Vsx2 | 26 | 0.8846 | 0.9231 | +0.0385 |
| DH_ex_Gpr83 | 33 | 0.8182 | 0.8485 | +0.0303 |
| DH_in_Npy2r | 38 | 0.8947 | 0.9211 | +0.0263 |

Top 10 M2 recall losses vs M0:

| class | support | M0 recall | M2 recall | delta |
|---|---|---|---|---|
| pericyte | 40 | 0.4250 | 0.3250 | -0.1000 |
| meninges_1 | 67 | 0.8955 | 0.8060 | -0.0896 |
| DH_ex_Grp | 26 | 0.6154 | 0.5385 | -0.0769 |
| DH_ex_Grpr | 31 | 0.8387 | 0.7742 | -0.0645 |
| oligodendrocyte_progenitor_1 | 49 | 0.4286 | 0.3673 | -0.0612 |
| meninges_2 | 42 | 0.3810 | 0.3333 | -0.0476 |
| MV_in_Chrna2 | 22 | 0.9545 | 0.9091 | -0.0455 |
| DH_ex_Prkcg/Nts | 52 | 0.6923 | 0.6538 | -0.0385 |
| DH_ex_Prkcg/Rxfp1 | 30 | 0.8333 | 0.8000 | -0.0333 |
| DH_ex_Tac2 | 32 | 0.7812 | 0.7500 | -0.0312 |

Macro-F1: M0 0.7936; M2 0.7955; LZH 0.7977. M2 does not show a material macro-F1 regression versus M0.

Rare-class buckets (true-class support):

| Support bucket | n classes | n cells | M0 mean recall | M2 mean recall | M0 mean F1 | M2 mean F1 |
|---|---|---|---|---|---|---|
| <25 | 15 | 201 | 0.5976 | 0.6239 | 0.6238 | 0.6528 |
| 25-49 | 21 | 751 | 0.8080 | 0.7956 | 0.8301 | 0.8136 |
| 50-99 | 11 | 721 | 0.9158 | 0.9169 | 0.8989 | 0.8995 |
| >=100 | 13 | 3327 | 0.8489 | 0.8499 | 0.8414 | 0.8431 |

The +3-cell overall result did not come from a large, obvious minority-class collapse, but neither did it produce a robust rare-class gain that could justify replacing MODEL V2.

## 12. Calibration Characterization

No temperature scaling, confidence threshold, or calibration tuning was performed.

| Model | Log loss | Mean top1 | Mean margin | Mean entropy | ECE (10 equal-width bins) |
|---|---:|---:|---:|---:|---:|
| M0 | 0.6416 | 0.9224 | 0.8603 | 0.2026 | 0.1019 |
| M2 | 0.6222 | 0.9119 | 0.8423 | 0.2284 | 0.0902 |

M2 has lower log loss than M0. That is a characterization result only. It is not a promotion criterion and is not used to set thresholds.

Accuracy by confidence decile is in `outputs/v3/v3_e07d_calibration_audit.csv`.

## 13. Test-Side Descriptive Shift

**NO TEST LABELS WERE USED.** Hidden test labels were not read, inferred, or scored.

Frozen M0 and M2 test probabilities are compared descriptively with validation probabilities.

| Quantity | M0 | M2 |
|---|---:|---:|
| Validation mean top1 | 0.9224 | 0.9119 |
| Test mean top1 | 0.9181 | 0.9101 |
| Validation mean entropy | 0.2026 | 0.2284 |
| Test mean entropy | 0.2095 | 0.2330 |
| Validation vs test predicted-class JS divergence | 0.004153 | 0.003704 |

M0 vs M2 test prediction agreement: **0.9470** (4735 / 5000). Changed test predictions: **265**.

These figures are not test accuracy. They are not used to choose thresholds, weights, classes, or post-processing.

## 14. LZH vs M0 / M2

No LZH blend, router, or stacker is formed.

### LZH vs M0

| Quantity | Value |
|---|---:|
| Both correct | 4024 |
| LZH-only correct | 109 |
| M0-only correct | 82 |
| Both wrong | 785 |
| Net (LZH − M0) | +27 |
| Pair oracle | 4215 / 5000 = 0.8430 |
| Exact McNemar p-value | 0.0596 |
| Folds 0-2 net | +26 |
| Folds 3-4 net | +1 |

Pair oracle is diagnostic only.

### LZH vs M2

| Quantity | Value |
|---|---:|
| Both correct | 4008 |
| LZH-only correct | 125 |
| M2-only correct | 101 |
| Both wrong | 766 |
| Net (LZH − M2) | +24 |
| Pair oracle | 4234 / 5000 = 0.8468 |
| Exact McNemar p-value | 0.1258 |
| Folds 0-2 net | +14 |
| Folds 3-4 net | +10 |

LZH remains ahead of both personal candidates overall. M2 does not overtake LZH, and pairing M2 with LZH is not a deployment action in this audit.

## 15. Strongest Auditable Standalone Expert

Standalone ranking uses overall accuracy, paired evidence, fold stability, and slice characterization. Diagnostic oracle coverage is **not** used.

**LZH REMAINS STRONGEST AUDITABLE STANDALONE EXPERT**

LZH Prior-H is the strongest currently auditable standalone expert: 4133 / 5000 = 0.8266, with a positive paired net versus both M0 and M2. M2 is not a standalone replacement for LZH. M0 remains the frozen personal deployable model, not the strongest team-auditable expert.

## 16. Personal MODEL V3 Promotion Decision

Personal version decision is separate from the team standalone ranking.

Frozen E06M STRONG promotion evidence required, among other conditions: M2 net correction versus M0 ≥ +25 cells, M2 accuracy ≥ 0.8262, positive net on both folds 0-2 and folds 3-4, no material macro-F1 regression, and sufficient SNI-signal / new-unique capture.

Frozen and reproduced E06M result: M2 = 4109, M0 = 4106, net +3, folds 3-4 net -9, label **PROMISING SOURCE-BALANCED TRANSFER**.

**MODEL V3 PROMOTION NOT JUSTIFIED**

This is preregistration discipline, not a subjective downgrade of a 4109 vs 4106 difference. E07D does not retroactively lower the criterion because a three-cell gain was observed.

## 17. Why Diagnostic Oracle Is Not Deployable Accuracy

Frozen four-expert diagnostic oracle: **4364 / 5000 = 0.8728**.

Frozen five-expert diagnostic oracle including M2: **4393 / 5000 = 0.8786**.

M2 new unique recoveries beyond the frozen four-expert pool: **29**.

Identity: 4364 + 29 = 4393. These numbers measure coverage if a perfect selector existed. They are not achievable OOF scores. V3-E03A and V3-E05A showed that converting complementary coverage into net-correct cells is the actual bottleneck: weak-expert confidence rescue and directional overrides both produced offsetting errors. V3-E06M internalized some source diversity into one model and still converted that coverage into only +3 net cells, with folds 3-4 negative.

Do not call 0.8786 an achievable OOF score.

## 18. Final V3 Research Synthesis

The current bottleneck is not lack of expert coverage. The auditable expert pool reaches a **0.8786 diagnostic oracle**, but multiple controlled experiments show that complementary information is difficult to convert into stable deployable gains without causing offsetting errors.

Evidence chain, without exaggeration:

- E00T: LZH and MODEL V2 are complementary (pair oracle 0.8430), but routing-only over that two-expert pool was insufficient.
- E02D: privileged 500-gene information exists, but dual-level distillation did not yield a mature deployable third expert.
- E03A: S0 contributes 96 unique recoveries, but available test-safe signals cannot isolate them at usable precision.
- E04S: SNI is a weak global classifier and a real source-diversity expert (53 unique recoveries; four-expert oracle 0.8728).
- E05A: the obvious directional patches over that complementarity have negative net correction.
- E06M: source-balanced training is better than naive pooling and slightly better than M0 overall, but not STRONG, and folds 3-4 regress.
- E07D: paired, bootstrap, section-cluster, and slice audits do not convert that +3-cell result into a justified personal version freeze.

Deployable accuracy and diagnostic oracle coverage remain distinct quantities. MODEL V2 stays the frozen personal deployable model because the V3 program did not produce a robust replacement.

## 19. Limitations

- Canonical folds 3-4 have been viewed in prior V3 stages and are retrospective.
- LZH Prior-H uses a different original 3-fold protocol; canonical 5-fold numbers are a common evaluation partition, not LZH's native validation.
- LZH 0.8266 is also a selection metric on that OOF bundle (MEDIUM validation confidence from V3-E00T).
- YHH and team main v9 remain unavailable for honest cell-level standalone comparison.
- SNI labels are consensus `voting` labels, not a wet-lab gold standard.
- Bootstrap and McNemar assume the resampled units; the section-cluster bootstrap is the dependence-aware check, not a new selection rule.
- No temperature scaling or threshold was fit; calibration numbers are descriptive.
- Test-side JS divergence is a predicted-class distribution comparison only. It is not test accuracy.

## 20. Final Decision

TEAM STANDALONE DECISION:
LZH REMAINS STRONGEST AUDITABLE STANDALONE EXPERT

PERSONAL VERSION DECISION:
MODEL V3 PROMOTION NOT JUSTIFIED

Recommended next action (do not start it automatically):

Close the personal V3 research program without creating MODEL V3. Keep frozen MODEL V2 as the personal deployable candidate. Treat LZH Prior-H as the strongest currently auditable standalone team expert for any later separately reviewed team-selection discussion. Do not blend M2, train a router, tune source weights, add Spatial-ID, or create a model-v3 tag.

Project state if V3 is not promoted, which is the decision above:

- MODEL V1: frozen historical version
- MODEL V2: current frozen personal deployable model
- V3 Research Program: completed controlled research program
- V3-E00T: expert complementarity audit
- V3-E02D: privileged-gene transfer negative result
- V3-E03A: weak-expert confidence rescue negative result
- V3-E04S: strong positive source-diversity discovery
- V3-E05A: directional routing negative result
- V3-E06M: promising but insufficient source-balanced transfer

Do not create a weaker or statistically indistinguishable MODEL V3 merely for version numbering.
