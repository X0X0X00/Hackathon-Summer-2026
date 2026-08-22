# V3-E06M — Source-Balanced Multi-Reference Transfer

Research codename: **SBMR**. This is a MODEL V3 research experiment, not a formal MODEL V3 freeze.

## 1. Research Question

Can SNI source information be internalized into a stronger single 200-gene reference model during training, by pooling cleaned MERFISH and cleaned SNI rows while explicitly preventing either source from dominating the objective?

The experiment does not route between source-specific experts at inference time. It trains two predeclared candidates only: naive pooling (M1, control) and source-balanced pooling (M2, primary).

## 2. Motivation from V3-E04S / E05A

Frozen auditable pool:

- LZH Prior-H: 4133 / 5000 = 0.8266
- WYH MODEL V2 / M0: 4106 / 5000 = 0.8212
- S0: 3151 / 5000 = 0.6302
- SNI-only expert: 2841 / 5000 = 0.5682
- LZH + WYH oracle: 4215 / 5000 = 0.8430
- LZH + WYH + S0 oracle: 4311 / 5000 = 0.8622
- LZH + WYH + S0 + SNI oracle: 4364 / 5000 = 0.8728
- SNI unique recoveries beyond LZH+WYH+S0: **53**
- remaining all-four-wrong: **636**

V3-E04S established that SNI is a weak global classifier but contributes 53 genuinely new source-diverse correct cells. V3-E05A established that directional prediction rules using that complementarity have negative net correction.

## 3. Why Test-Time Routing Was Rejected

V3-E03A found that confidence-based rescue of the weak S0 expert is unsafe. V3-E05A found that predeclared directional overrides of strong-expert consensus by S0 or SNI all have negative net correction. Weak source-specific experts are therefore not used as test-time rescue members here. The alternative is to integrate SNI during training inside a single deployable 200-gene LightGBM.

## 4. Project-Level Duplication Check

git fetch team --prune succeeded. Inspected team/main, team/yhh, team/lzh, team/wyh, and team/revert-1-lzh. No auditable equivalent of MERFISH + SNI + explicit source-balanced sample weighting + single deployable 200-gene reference model was found. YHH sample_weight is class-balancing / competition-vs-MERFISH mixing. LZH source_weights downweight rare-class MERFISH reference rows in binary glial heads. V3-E04S is SNI-only. This is not a global first-ever novelty claim.

The valid project-level claim is: **Source-balanced multi-reference training is a new controlled modeling direction within the current auditable project.**.

Nearby but non-equivalent methods:

- WYH MODEL V2 is MERFISH-reference-only LightGBM with no SNI rows and no sample weights
- V3-E04S is an isolated SNI-only expert, not a combined source-balanced model
- YHH `sample_weight` is class-balancing / competition-vs-MERFISH mixing for family specialists, not MERFISH+SNI 0.5/0.5 source mass
- LZH `source_weights` downweight rare-class MERFISH-reference rows inside binary glial heads, not a 60-class source-balanced MERFISH+SNI LightGBM
- Generic ensembles and separate reference experts are not equivalent to explicit source-balanced single-model training

No auditable equivalent of MERFISH + SNI + explicit source-balanced sample weighting + single deployable 200-gene reference model was found.

## 5. Reference Provenance

| Source | Path | MD5 | Raw shape | Label column |
|---|---|---|---|---|
| MERFISH | `/Users/yyl/Documents/Hackathon-Summer-2026/work/external/MERFISH_spinal_cord_resolved_0718.h5ad` | `ce06f62c0ec4973581dae17bb76f0cd9` | 146621 × 500 | MERFISH cell type annotation |
| SNI | `/Users/yyl/Documents/Hackathon-Summer-2026/work/external/SNI_merged_0917.h5ad` | `7e90a801ee57b8fec06cd03c8630f01b` | 55331 × 500 | voting + `norm_label` |

Official 200-gene SHA256: `e3301724038990aa2db237026316aaa5fd265a11231c343bea733f8106ab06f5`. Matches frozen V2: True.

## 6. Source Cleaning / Exclusion Audit

MERFISH exclusions reproduced from the frozen V2 contract: 5000 train IDs, 5000 test IDs, 47 exact 200-gene competition duplicates. Usable MERFISH: **136574**.

SNI exclusions reproduced from V3-E04S: 0 train ID overlaps, 0 test ID overlaps, 10 train exact-vector duplicates, 15 test exact-vector duplicates, 116 cleaned-MERFISH exact-vector duplicates. Usable SNI after E04S contract: **55193**.

Additional remaining cross-source exact 200-gene duplicates after that contract, excluded from SNI while retaining MERFISH: **0**.

## 7. Combined Reference Population

| Quantity | Count |
|---|---:|
| Cleaned MERFISH | 136574 |
| Cleaned SNI after E04S | 55193 |
| Combined before extra cross-source dups | 191767 |
| Extra SNI copies excluded | 0 |
| Final combined training rows | 191767 |
| MERFISH class coverage | 60/60 |
| SNI class coverage | 60/60 |

`reference_source ∈ {MERFISH, SNI}` is stored for weighting/audit only and is **not** a model input.

Combined-row identity SHA256: `a817c6ea48e9fe327031bfe5dd1cfbd702b6156d23b068b15b7a96916f832c4f`.

## 8. Frozen V2 Modeling Contract

E06M reuses V2-B-REFONLY:

- official 200-gene order and library-size log1p with the median total of the 10,000 competition cells
- PCA50 (`random_state=0`) fitted on the frozen 146,621-cell MERFISH deposit, **not** refit on SNI
- within-section spatial kNN, PCA50 expression kNN, neighbor-mean PCA, metadata, fold-safe neighbor-label histograms
- LightGBM multiclass, 700 fixed rounds, seed 0, `num_threads=8`, no early stopping
- E/I then Segment post-processing from the frozen V2 leakage-safe maps
- competition labels never enter the boosting objective
- held-out fold labels remain invisible in histograms, masks, and fitting

Competition-cell and MERFISH-reference features stay in the frozen MERFISH universe. SNI training rows receive the same feature schema, with SNI-internal graphs and SNI labels in histograms. Missing SNI metadata (Region, E/I, Segment, gender) is NaN. Unseen SNI mouse/dataset codes are NaN so they cannot become a source identifier.

Feature-schema audit: **PASS**. Frozen V2 feature names, PCA50 reconstruction, LightGBM hyperparameters, 700 rounds, seed 0, E/I+Segment post-processing, and M0 accuracy 0.8212 were reproduced. The only intended differences are the combined reference population and M2 sample weights.

M0 was not retrained. Frozen MODEL V2 artifacts were reused.

## 9. Candidate Definitions

### M0

Frozen MERFISH-only MODEL V2 / V2-B-REFONLY. Accuracy 0.8212 (4106 / 5000). Macro-F1 0.7936.

### M1 — Naive multi-reference pool (CONTROL)

Same combined rows as M2. Frozen V2 base sample-weight behavior: every external row has weight 1. Source contribution is proportional to row counts. M1 is not the primary innovation.

### M2 — Source-balanced multi-reference (PRIMARY)

Exactly the same rows, features, labels, folds, rounds, and LightGBM settings as M1. Only sample weighting differs: each source receives total mass 0.5, then one common rescale makes mean(weight)=1.

## 10. Source-Weight Audit

| Candidate | MERFISH n | SNI n | MERFISH effective sum | SNI effective sum | ratio MERFISH/SNI | mean | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 136574 | 55193 | 136574.000000 | 55193.000000 | 2.474480 | 1.000000 | 1.000000 | 1.000000 |
| M2 | 136574 | 55193 | 95883.500000 | 95883.500000 | 1.000000 | 1.000000 | 0.702063 | 1.737240 |

M2 required effective source-weight ratio ≈ 1.0: **True**.

No 0.6/0.4 through 0.9/0.1 grid was tested.

## 11. Overall Validation Results

Predictions on the 5000 competition-train cells are honest external-validation predictions from models whose boosting objective uses only external-reference labels. Fold-safe neighbor-label histograms may use non-held-out competition-train labels exactly as in frozen V2. This is not a model that is independent of all competition-train labels, and it is not conventional competition-label OOF for the boosting target.

| Model | Correct | Accuracy | Macro-F1 | Log loss |
|---|---:|---:|---:|---:|
| M0 | 4106 | 0.8212 | 0.7936 | 0.6416 |
| M1 | 4086 | 0.8172 | 0.7906 | 0.6313 |
| M2 | 4109 | 0.8218 | 0.7955 | 0.6222 |

## 12. Fold / Stability Results

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Folds 0-2 | Folds 3-4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.822 | 0.831 | 0.812 | 0.821 | 0.820 | 0.8217 | 0.8205 |
| M1 | 0.818 | 0.823 | 0.812 | 0.820 | 0.813 | 0.8177 | 0.8165 |
| M2 | 0.827 | 0.829 | 0.821 | 0.816 | 0.816 | 0.8257 | 0.8160 |

Canonical folds 3-4 remain a retrospective stability partition, not an untouched holdout.

## 13. Slice / Biological-Family Results

| Slice | M0 | M1 | M2 |
|---|---:|---:|---:|
| Hard bucket | 0.7569 | 0.7498 | 0.7569 |
| Non-hard bucket | 0.9143 | 0.9148 | 0.9158 |
| Neuron | 0.9198 | 0.9214 | 0.9214 |
| Glial / non-neuronal | 0.7629 | 0.7556 | 0.7629 |

M0 families:

| oligodendrocyte / OPC | 1552 | 0.7326 |
| astrocyte | 769 | 0.8336 |
| vascular / endothelial | 363 | 0.7355 |
| meningeal | 128 | 0.6641 |
| microglia | 64 | 0.6875 |
| remaining glial/non-neuronal | 82 | 0.7927 |
| neuronal / other | 2042 | 0.9143 |

M1 families:

| oligodendrocyte / OPC | 1552 | 0.7216 |
| astrocyte | 769 | 0.8349 |
| vascular / endothelial | 363 | 0.7273 |
| meningeal | 128 | 0.6484 |
| microglia | 64 | 0.7031 |
| remaining glial/non-neuronal | 82 | 0.7805 |
| neuronal / other | 2042 | 0.9148 |

M2 families:

| oligodendrocyte / OPC | 1552 | 0.7326 |
| astrocyte | 769 | 0.8375 |
| vascular / endothelial | 363 | 0.7273 |
| meningeal | 128 | 0.6250 |
| microglia | 64 | 0.7344 |
| remaining glial/non-neuronal | 82 | 0.8171 |
| neuronal / other | 2042 | 0.9158 |

## 14. M1 and M2 vs M0 Cell-Level Deltas

| Comparison | Split | Changed | Wrong→correct | Correct→wrong | Net |
|---|---|---:|---:|---:|---:|
| M1 vs M0 | overall | 234 | 84 | 104 | -20 |
| M1 vs M0 | folds 0-2 | 140 | 49 | 61 | -12 |
| M1 vs M0 | folds 3-4 | 94 | 35 | 43 | -8 |
| M2 vs M0 | overall | 252 | 105 | 102 | 3 |
| M2 vs M0 | folds 0-2 | 145 | 68 | 56 | 12 |
| M2 vs M0 | folds 3-4 | 107 | 37 | 46 | -9 |

## 15. M2 vs M1 Source-Balancing Ablation

| Quantity | Value |
|---|---:|
| M1 accuracy | 0.8172 |
| M2 accuracy | 0.8218 |
| M2 correct − M1 correct | 23 |
| M2 wrong→correct vs M1 | 104 |
| M2 correct→wrong vs M1 | 81 |
| Net M2 gain over M1 | 23 |
| Macro-F1 delta | 0.004896 |
| Folds 0-2 delta (accuracy) | 0.008000 |
| Folds 3-4 delta (accuracy) | -0.000500 |
| Does balancing help? | True |

M1 remains a control. It is not promoted as MODEL V3 from this experiment.

## 16. SNI Unique-Signal Capture

Frozen V3-E04S unique recoveries: 53 cells where LZH, WYH, and S0 are wrong and SNI is correct.

| Candidate | Captured / 53 | Capture fraction | M0 wrong → candidate correct on the 53 | Agrees with SNI | Remains equal to M0 |
|---|---:|---:|---:|---:|---:|
| M1 | 5 | 0.0943 | 5 | 5 | 45 |
| M2 | 4 | 0.0755 | 4 | 4 | 47 |

## 17. MERFISH Anchor Retention

| Candidate | M0-correct retained | M0-correct lost | Retention rate | M0-wrong recovered | Net transfer |
|---|---:|---:|---:|---:|---:|
| M1 | 4002 | 104 | 0.9747 | 84 | -20 |
| M2 | 4004 | 102 | 0.9752 | 105 | 3 |

Source integration is unsuccessful if it captures SNI cases but destroys more frozen MERFISH-anchor correct cells than it recovers.

## 18. Pairwise Complementarity

**ORACLE != DEPLOYABLE ACCURACY.**

| Candidate | Existing | Both correct | Candidate-only | Existing-only | Both wrong | Oracle n | Oracle acc |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 | lzh_prior_h | 3997 | 89 | 136 | 778 | 4222 | 0.8444 |
| M1 | wyh_model_v2 | 4002 | 84 | 104 | 810 | 4190 | 0.8380 |
| M1 | s0 | 2998 | 1088 | 153 | 761 | 4239 | 0.8478 |
| M1 | sni | 2705 | 1381 | 136 | 778 | 4222 | 0.8444 |
| M1 | m0 | 4002 | 84 | 104 | 810 | 4190 | 0.8380 |
| M2 | lzh_prior_h | 4008 | 101 | 125 | 766 | 4234 | 0.8468 |
| M2 | wyh_model_v2 | 4004 | 105 | 102 | 789 | 4211 | 0.8422 |
| M2 | s0 | 3014 | 1095 | 137 | 754 | 4246 | 0.8492 |
| M2 | sni | 2719 | 1390 | 122 | 769 | 4231 | 0.8462 |
| M2 | m0 | 4004 | 105 | 102 | 789 | 4211 | 0.8422 |

No blend or weight was optimized.

## 19. New Recoveries Beyond the Four-Expert Pool

New unique recoveries are cells where LZH, WYH, S0, and SNI are all wrong and the candidate is correct.

| Candidate | New unique recoveries | Five-expert oracle | Five-expert accuracy |
|---|---:|---:|---:|
| M1 | 21 | 4385 | 0.8770 |
| M2 | 29 | 4393 | 0.8786 |

Identity check: five-expert oracle = 4364 + new unique recoveries. M2 identity: **True**.

M2 new-unique families:

| oligodendrocyte / OPC | 11 |
| astrocyte | 3 |
| vascular / endothelial | 4 |
| meningeal | 0 |
| microglia | 1 |
| remaining glial/non-neuronal | 0 |
| neuronal / other | 10 |

## 20. Five-Expert Diagnostic Oracle

**ORACLE != DEPLOYABLE ACCURACY.**

M2 five-expert diagnostic coverage:

| Split | Correct | Accuracy | Remaining all-five-wrong |
|---|---:|---:|---:|
| Overall | 4393 | 0.8786 | 607 |
| Folds 0-2 | 2645 | 0.8817 | 355 |
| Folds 3-4 | 1748 | 0.8740 | 252 |

These numbers are diagnostic coverage ceilings, not a deployable MODEL V3 score.

## 21. Leakage Audit

- competition test labels used: False
- competition train labels enter boosting objective: False
- held-out fold labels remain invisible: True
- all external competition ID overlaps excluded: True
- all prohibited exact-vector duplicates excluded: True
- hyperparameter search: False
- seed search: False
- source-weight search: False
- leaderboard tuning: False
- post-hoc class-rule tuning: False
- test probabilities used in selection: False
- prediction/prediction.csv modified: False
- MODEL V1 / V2 / V3-E00T / E02D / E03A / E04S / E05A modified: False
- `reference_source` used as a model input: False

## 22. Limitations

- SNI labels are the file's `voting` consensus after `norm_label`, not a wet-lab gold standard.
- SNI mixes Sham and SNI injury conditions; this experiment treats the file as one source.
- SNI rows lack Region / E/I / Segment / gender, so those V2 metadata channels are missing for the SNI half of the objective.
- SNI graphs are source-internal; competition cells do not gain SNI spatial neighbors. That preserves the frozen V2 competition feature contract.
- Canonical folds 3-4 have been viewed in prior V3 stages and are retrospective.
- Diagnostic oracle coverage is not deployable accuracy.
- Only the predeclared 0.5/0.5 source mass was tested.

## 23. Decision

**PROMISING SOURCE-BALANCED TRANSFER**

M2 has positive overall net correction, folds 3-4 are not a material regression, and some SNI/new-unique signal transferred, but not all STRONG criteria were met.

Recommended next experiment/action:

Keep M2 as a documented base-model candidate and next compare it against LZH Prior-H and frozen MODEL V2 under the same folds without blending or routing. Do not add another dataset or start Spatial-ID.
