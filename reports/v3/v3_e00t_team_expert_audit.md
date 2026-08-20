# V3-E00T — Team Canonical Expert & Complementarity Audit

## 1. Objective

Quantify whether currently available **honest, cell-level OOF** team experts have enough complementary headroom to justify a MODEL V3 reliability-routing experiment. This audit does **not** train a new model, search blend weights, tune thresholds, or produce a submission.

Research hypothesis: selective expert correction may be more useful than another globally averaged ensemble, **if** a small pool of methodologically different experts has real oracle headroom on a locked canonical partition.

Oracle accuracy is diagnostic headroom only. **ORACLE != DEPLOYABLE MODEL ACCURACY.**

## 2. Team Repository Snapshot

Fetched `team --prune` at audit start. Remote-tracking branches were inspected with `git show` / `git ls-tree` only. No teammate branch was checked out.

| Branch | SHA | Date | Latest subject / selected candidate |
|---|---|---|---|
| `team/main` | `c34feed7a03d1a1421c55a6fe4c40765aeff2b8b` | 2026-08-20T07:51:14-04:00 | Update predictions (team blend v9) (test-only blend v9) |
| `team/yhh` | `9afc72774e2a6e186a04c2557110992d0befe164` | 2026-08-19T18:14:10+08:00 | merge V6 and V7 model updates (V7 hierarchical specialists, reported 0.8320) |
| `team/lzh` | `af51ce78846063f2ec9bede33b6b23b7a75d3492` | 2026-08-20T19:37:13+08:00 | Upload submission_graph_stacker_V3_prior_H (selected `depth_masked_prior_h_anchor`, 0.8266) |
| `team/wyh` | `da127c6ca86693f8bad5941aa26cc24ba129e1af` | 2026-08-20T07:53:30-04:00 | docs(wyh): document Model V1 and Model V2 releases (mature WYH delivery; personal frozen V2 used instead) |
| `team/revert-1-lzh` | `bf4b34bbb988505341ec2caec237de6a9587207d` | 2026-08-20T02:00:05+08:00 | Revert "Upload files to lzh branch" (historical provenance only) |

Personal development branch: `ywan/ml-pipeline` @ `420e7a6afd9aa0d50b21f385ea7f8c324cee3c84`. Frozen MODEL V2 tag: `model-v2`.

## 3. Expert Artifact Manifest

| Expert | Owner | Model | OOF artifacts | Protocol | Level | Confidence |
|---|---|---|---|---|---|---|
| `wyh_model_v2` | WYH | V2-B-REFONLY | `outputs/oof/MODEL-V2_oof.csv` + `outputs/probabilities/V2-B-REFONLY_oof_probabilities_seg.csv.gz` | 5-fold seed42 | A | HIGH |
| `lzh_prior_h` | LZH | `depth_masked_prior_h_anchor` | `submission_graph_stacker_V3_prior_H_20260820/model/oof_probabilities_final.csv` | 3-fold LZH OOF | A | MEDIUM |
| `yhh_v7` | YHH | V7 hierarchical specialists | **missing cell-level OOF** (gitignored npz) | claimed 5-fold, tune 0-2 / holdout 3-4 | C | LOW |
| `team_main_v9` | zzh / main | team blend v9 | test `prediction/prediction.csv` only | unknown for v9 | C | LOW |
| `lzh_graph_stacker_v2_historical` | LZH | 82.50 graph stacker | committed OOF probabilities | 3-fold | A | MEDIUM |

`lzh_graph_stacker_v2_historical` is inventoried but **excluded from the predeclared oracle pool** because it is a superseded model on the same LZH line, not an independent owner-line expert.

Exact paths, SHAs, and eligibility reasons: `outputs/v3/v3_e00t_expert_manifest.json`.

## 4. Integrity Checks

- Canonical population: **5000** unique 19-digit `Cell_ID` strings from official `meta_train`.
- Labels aligned from `MERFISH_cell_type_annotation`; no missing labels.
- Joins used `Cell_ID`, never row position as the primary key.
- Official class order: `allowed_labels()` (60 sorted training labels). WYH probability columns already match. LZH columns use `p__<class>` prefixes and match the same 60 names.
- Probability rows are finite, non-negative, and sum to 1 within `atol=1e-4`.
- LZH final OOF probabilities match `oof_probabilities_prior_h_anchor.csv` (selected model is the Prior-H Anchor, not the newly trained stacker).
- Competition test labels were not used. Test probability files were inventoried for Cell_ID/class-order only.
- `prediction/prediction.csv` was not modified.
- `pyarrow==19.0.1` was added only to support reproducible Parquet artifact I/O for V3-E00T, not as a modeling dependency.

## 5. Individual Expert Metrics

Reported vs reproduced overall accuracy on the canonical 5000-cell population.

| Expert | Level | Reported | Reproduced | Status | Canonical 0-2 | Canonical 3-4 |
|---|---|---:|---:|---|---:|---:|
| `wyh_model_v2` | A | 0.8212 | 0.8212 | VERIFIED | 0.8217 | 0.8205 |
| `lzh_prior_h` | A | 0.8266 | 0.8266 | VERIFIED | 0.8303 | 0.8210 |

YHH V7 reported **0.8320** (4160 / 5000) and holdout **0.8230**, but this cannot be reproduced from committed cell-level artifacts. Team main recorded a non-reproduction on zzh member files (`0.8252` / holdout `0.8170`). Current team main v9 has **no reported OOF**.

## 6. Canonical Partition Metrics

Canonical analysis partition: `experiments/team_folds_5_seed42.csv` (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

This is a **meta-analysis partition**. It is the original protocol for WYH MODEL V2. It is **not** LZH's original 3-fold.

WYH MODEL V2 canonical fold accuracies: {"0": 0.822, "1": 0.831, "2": 0.812, "3": 0.821, "4": 0.82}

LZH Prior-H canonical fold accuracies: {"0": 0.828, "1": 0.836, "2": 0.827, "3": 0.824, "4": 0.818}

LZH original 3-fold accuracies (bundle `oof_gate.csv`, descriptive only): {"0": {"n": 1667, "accuracy": 0.8290341931613677}, "1": {"n": 1667, "accuracy": 0.8320335932813437}, "2": {"n": 1666, "accuracy": 0.8187274909963985}}

## 7. Pairwise Complementarity

| A | B | A acc | B acc | Disagree | Pair oracle | Headroom vs stronger |
|---|---|---:|---:|---:|---:|---:|
| lzh_prior_h | wyh_model_v2 | 0.8266 | 0.8212 | 236 | 0.8430 | 0.0164 |

Pairwise oracle on canonical folds 3-4:

- `lzh_prior_h` vs `wyh_model_v2`: oracle 0.8415, headroom 0.0205

## 8. Multi-Expert Oracle

**ORACLE != DEPLOYABLE MODEL ACCURACY.**

Currently auditable honest expert pool: LZH `depth_masked_prior_h_anchor` and WYH MODEL V2.

YHH V7 and current team-main are unavailable for honest cell-level OOF comparison, so they are omitted. The 0.8430 oracle below applies **only** to this auditable pair. It is **not** evidence that the entire team's model pool has an oracle below 0.85. The full-team oracle remains unknown.

| Combo | Split | Oracle correct | Oracle acc | All wrong |
|---|---|---:|---:|---:|
| LZH + WYH | overall | 4215 | 0.8430 | 785 |
| LZH + WYH | canonical folds 0-2 | 2532 | 0.8440 | 468 |
| LZH + WYH | canonical folds 3-4 | 1683 | 0.8415 | 317 |

Best 3-way oracle: **not available** (only two LEVEL A/B experts in the predeclared pool).
Best 4-way oracle: **not available**.

Number of correct experts overall: {"0": 785, "1": 191, "2": 4024}

## 9. Recommended Anchor

**Recommended anchor: `lzh_prior_h`**

Reason: LZH Prior-H has honest Cell_ID-aligned OOF probabilities, the highest reproduced overall accuracy among eligible experts (0.8266 vs WYH 0.8212), and canonical folds 3-4 accuracy 0.8210 vs WYH 0.8205. YHH V7 and team-main v9 cannot be anchors because they lack honest OOF artifacts. Limitation: LZH uses a 3-fold protocol and selected this model on the same OOF used for reporting (MEDIUM confidence).

This recommendation is for the next V3 diagnostic experiment. It is not a formal MODEL V3 freeze and does not generate predictions.

## 10. Anchor Error Recoverability

Anchor errors: **867 / 5000**.

| Pattern | Count | Fraction of anchor errors |
|---|---:|---:|
| Recoverable by WYH MODEL V2 | 82 | 0.0946 |
| Recoverable by any alternative expert | 82 | 0.0946 |
| Recoverable by 2+ agreeing alternatives | 0 | n/a with one alternative |
| A. both alternatives agree on the correct replacement | 0 | n/a |
| B. alternatives disagree with anchor and with each other | 0 | n/a |
| C. anchor correct, alternatives agree on the same wrong class | 0 | n/a |
| D. anchor correct, exactly one alternative disagrees | 109 | 0.0264 |

False-correction risk among cells where the single alternative disagrees with a correct anchor: 109 cells have a correct anchor and a disagreeing alternative (0.0264 of correct-anchor cells). An unconstrained override of every disagreement would convert these into new errors..

## 11. Fixed Consensus Diagnostics

Predeclared diagnostic rules only. No threshold or weight tuning. These are **not** MODEL V3 candidates.

With one eligible alternative, D1 (two-alternative agreement) cannot fire. D2 reduces to "replace whenever the single alternative disagrees". D3 majority-with-anchor-tie-break keeps the anchor on every 1–1 disagreement, so D3 equals D0.

| Rule | Changed | Wrong→correct | Correct→wrong | Net | Precision | Final acc | Canonical 3-4 acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0 | 0 | 0 | 0 | n/a | 0.8266 | 0.8210 |
| D1 | 0 | 0 | 0 | 0 | n/a | 0.8266 | 0.8210 |
| D2 | 236 | 82 | 109 | -27 | 0.347 | 0.8212 | 0.8205 |
| D3 | 0 | 0 | 0 | 0 | n/a | 0.8266 | 0.8210 |

## 12. Confidence / Reliability Findings

Anchor confidence by outcome (top1 / margin / entropy):

| Subset | n | top1 mean | margin mean | entropy mean |
|---|---:|---:|---:|---:|
| A. anchor correct | 4133 | 0.8744 | 0.7937 | 0.4118 |
| B. anchor wrong | 867 | 0.6772 | 0.4766 | 0.9014 |
| C. wrong and recoverable | 82 | 0.5271 | 0.1871 | 1.0670 |
| D. wrong and unrecoverable | 785 | 0.6929 | 0.5068 | 0.8841 |
| F. correct but alternative disagrees | 109 | 0.5439 | 0.2322 | 1.0464 |

Diagnostic AUROC (not a deployed selector):

- P(anchor wrong) from `1 - top1`: 0.8220
- P(anchor wrong) from entropy: 0.8158
- P(anchor wrong AND recoverable) from `1 - top1`: 0.9135
- Mean JS divergence (base 2) on cells where predictions disagree: 0.1231

Anchor wrong cells have lower top1 / margin and higher entropy than correct cells, but recoverable vs unrecoverable errors overlap substantially. A threshold on confidence alone is unlikely to isolate safe corrections without a second expert signal.

## 13. Test-Safe Reliability Feature Inventory

See `outputs/v3/v3_e00t_tables/reliability_feature_inventory.csv`.

Summary:

- **Diagnostic-only:** true label, expert_correct, oracle flags, recoverable flags, LZH original 3-fold id used as provenance.
- **Test-safe and available now:** anchor top1 / margin / entropy; alternative top1 / margin / entropy; predicted-class disagreement; JS divergence; probability advantage; library size; n_detected; Region / E/I / Segment / missingness / hard_bucket / Section_ID.
- **Documented by LZH but not available as committed cell-level OOF fields:** Prior-H eligibility, strict graph degree, reliable-gene count. Only aggregate test-route counts are in `prior_h_route_audit.json`.
- **Not test-safe:** anything derived from held-out labels, including this audit's recoverable flags.

## 14. Biological Error Families

Anchor errors by true-class family (recoverable = alternative expert already predicts the true class).

- `oligodendrocyte_opc`: 407 errors, 42 recoverable (0.103)
- `astrocyte`: 125 errors, 11 recoverable (0.088)
- `vascular`: 96 errors, 4 recoverable (0.042)
- `meningeal`: 44 errors, 8 recoverable (0.182)
- `other_glial_non_neuronal`: 32 errors, 1 recoverable (0.031)
- `neuronal_or_other`: 163 errors, 16 recoverable (0.098)

Top true → anchor_pred confusion pairs:

| True | Anchor pred | Errors | Recoverable | Fraction | Rescued by |
|---|---|---:|---:|---:|---|
| oligodendrocyte_1 | oligodendrocyte_progenitor_2 | 92 | 8 | 0.087 | wyh_model_v2 |
| oligodendrocyte_progenitor_2 | oligodendrocyte_2 | 53 | 14 | 0.264 | wyh_model_v2 |
| oligodendrocyte_progenitor_2 | oligodendrocyte_1 | 47 | 6 | 0.128 | wyh_model_v2 |
| oligodendrocyte_2 | oligodendrocyte_progenitor_2 | 31 | 2 | 0.065 | wyh_model_v2 |
| endothelial | astrocyte_1 | 30 | 2 | 0.067 | wyh_model_v2 |
| oligodendrocyte_1 | astrocyte_1 | 30 | 6 | 0.200 | wyh_model_v2 |
| oligodendrocyte_progenitor_2 | astrocyte_1 | 27 | 0 | 0.000 | none |
| astrocyte_1 | oligodendrocyte_progenitor_2 | 24 | 2 | 0.083 | wyh_model_v2 |
| oligodendrocyte_progenitor_1 | oligodendrocyte_precursor_cell | 23 | 0 | 0.000 | none |
| DH_in_Klhl14 | DH_in_Cdh3 | 21 | 0 | 0.000 | none |
| meninges_2 | meninges_1 | 18 | 1 | 0.056 | wyh_model_v2 |
| astrocyte_1 | endothelial | 18 | 0 | 0.000 | none |
| endothelial | oligodendrocyte_1 | 18 | 0 | 0.000 | none |
| astrocyte_1 | oligodendrocyte_1 | 17 | 2 | 0.118 | wyh_model_v2 |
| oligodendrocyte_1 | oligodendrocyte_2 | 17 | 1 | 0.059 | wyh_model_v2 |

Interpretation for the next design: recoverable errors are not confined to a single family, but oligodendrocyte / OPC and other glial/non-neuronal confusions remain the largest absolute buckets. A purely family-specific rule would miss a non-trivial neuronal remainder. The evidence is closer to **global reliability + optional family-aware features** than to a single-family patch.

## 15. Unique Value of Each Expert

| Expert | Standalone acc | Only this expert correct | Anchor-wrong and this expert correct |
|---|---:|---:|---:|
| `lzh_prior_h` | 0.8266 | 109 | 0 |
| `wyh_model_v2` | 0.8212 | 82 | 82 |

**Does WYH MODEL V2 add unique team value despite lower standalone accuracy?** Yes. WYH MODEL V2 is uniquely correct on 82 cells that LZH Prior-H misses, which is the entire incremental oracle of the eligible two-expert pool. That is meaningful unique value despite a lower standalone accuracy (0.8212 vs 0.8266). It is not enough, by itself, to create a 3-expert team oracle, and unconstrained replacement of the anchor by WYH is unsafe because WYH also introduces 109 false corrections.

## 16. Leakage / Validation Audit

- No competition test labels were used for scoring, selection, or thresholding.
- No blend-weight search, stacking, or learned router was fit.
- WYH MODEL V2 OOF is the frozen 5-fold seed42 protocol. MODEL V2 architecture was not retuned here.
- LZH Prior-H OOF is a different 3-fold protocol. Cell-level predictions can still be used descriptively if each cell is genuinely held out of that 3-fold, but canonical folds 3-4 are **not** LZH's original holdout.
- LZH selected Prior-H over the graph stacker using the same full 3-fold OOF, so the 0.8266 number is a selection metric as well as an OOF metric (MEDIUM validation confidence).
- YHH V7 used folds 0-2 for gating and folds 3-4 as holdout, which is a stronger selection protocol, but the cell-level OOF artifact is absent.
- Team main v9 has no OOF counterpart; test prediction changes cannot be treated as validation.

## 17. Current Team Main Reproducibility Status

**Current `team/main` HEAD is prediction-only.** Commit `c34feed7a03d1a1421c55a6fe4c40765aeff2b8b` updates `prediction/prediction.csv` and does not add OOF predictions, OOF probabilities, or generation metadata sufficient to reconstruct an honest OOF score.

No OOF metric is reported for team blend v9.

The latest historically documented reproducible ensemble on that line is V6 (equal-weight `refonly_full + ext_all25 + mlp3 + yhh_v1 + poolAll`, reported ~0.8248 / holdout ~0.8175). Those member probability `npz` files are gitignored and were not available for this audit.

## 18. MODEL V3 Strategic Decision

**ROUTING-ONLY INSUFFICIENT**

Best honest multi-expert oracle for the **currently auditable** pool (LZH `depth_masked_prior_h_anchor` + WYH MODEL V2) = **0.8430** overall (4215 / 5000), and **0.8415** on canonical folds 3-4.

This 0.8430 figure does **not** describe the entire team's model pool. YHH V7 and current team-main are unavailable for honest cell-level OOF comparison, so the full-team oracle remains unknown.

Routing-only is insufficient for the CURRENTLY AUDITABLE honest expert pool. A new independent expert should be developed now rather than tuning a router over only LZH + WYH. If YHH/main OOF artifacts later become available, rerun the canonical oracle audit.

Recommended next primary experiment: **V3-E02D — 500-to-200 Gene Privileged Distillation**

A new independent expert should be developed now rather than tuning a router over only LZH + WYH. The approved same-study 500-gene reference is the smallest domain-gap source of additional signal not already fully used by the 200-gene reference-only LightGBM.

Do not start that experiment in this task. MODEL V3 is not frozen. No `docs/versions/model_v3.md` and no submission candidate were created.
