# V3-E04S — Source-Diverse SNI Expert

Research codename: **SDE-SNI**. This is a MODEL V3 research experiment, not a formal MODEL V3 freeze.

## 1. Research Question

Can a biologically distinct SNI reference source produce complementary cell-type information that the existing MERFISH-reference, biology/graph, and weak-MLP expert families do not contain?

The first SNI model isolates the **source** effect:

SNI-only + official 200 competition genes + a LightGBM recipe matched to frozen V2-B-REFONLY.

## 2. Motivation from V3-E00T / E02D / E03A

Frozen auditable pool:

- LZH Prior-H: 4133 / 5000 = 0.8266
- WYH MODEL V2: 4106 / 5000 = 0.8212
- S0 hard-label 200-gene reference MLP: 3151 / 5000 = 0.6302
- LZH + WYH diagnostic oracle: 4215 / 5000 = 0.8430
- LZH + WYH + S0 diagnostic oracle: 4311 / 5000 = 0.8622
- remaining all-three-wrong population: **689**

V3-E03A found that S0 contributes 96 genuine unique recoveries, but available test-safe signals cannot identify those cases at high enough precision. An S0 rescue router is not justified. The new objective is an independent expert that can recover cells from the remaining 689.

## 3. Why This Is Not Duplicate Team Work

git fetch team --prune and git grep on team/main, team/yhh, team/lzh, team/wyh, team/revert-1-lzh, and the personal tree found no auditable SNI_merged_0917.h5ad isolated expert. This is not a global-novelty claim.

The correct claim is: **new source-diversity experiment within the current team/project**.

Nearby but non-equivalent methods:

- WYH MODEL V2 is MERFISH-reference-only LightGBM
- LZH Prior-H uses biology priors / graph evidence
- YHH uses hierarchical specialists
- V3-E02D/S0 uses the same MERFISH reference as a neural student
- team main uses MERFISH reference / spatial / ensemble members

No auditable equivalent isolated `SNI_merged_0917.h5ad` expert was found.

## 4. SNI Provenance

| Item | Value |
|---|---|
| Path | `/Users/yyl/Documents/Hackathon-Summer-2026/work/external/SNI_merged_0917.h5ad` |
| Filename | `SNI_merged_0917.h5ad` |
| Size | 115901575 bytes |
| AnnData shape | 55331 cells × 500 genes |
| Label column used | `voting` |
| Condition field | {'Sham': 31498, 'SNI': 23833} |
| Datasets | 5 distinct run identifiers |
| Gene identifiers | `AnnData.var_names` |

obs columns: Datasets, volume, center_x, center_y, batch, voting, tangram, singler, seurat, rctd, Condition, Mouse ID, Axial level, Side, Section ID, Custom Cells groups

No undocumented biological meaning was inferred from field names.

## 5. Checksum Verification

| Item | Value |
|---|---|
| MD5 | `7e90a801ee57b8fec06cd03c8630f01b` |
| Expected MD5 | `7e90a801ee57b8fec06cd03c8630f01b` |
| MD5 match | True |
| SHA256 | `30fcfed7daa1d5bc690b444a2a4e4fe1801fcad47e1417800ef9117252086226` |
| MERFISH reference MD5 | `ce06f62c0ec4973581dae17bb76f0cd9` (unchanged) |

## 6. Competition Cell_ID Overlap Audit

Cell_ID remained a lossless 19-digit string.

| Comparison | Overlap |
|---|---:|
| SNI vs competition train IDs | 0 |
| SNI vs competition test IDs | 0 |

Overlapping IDs, if any, are excluded before fitting.

## 7. MERFISH-Reference Overlap Audit

| Comparison | Overlap |
|---|---:|
| SNI vs MERFISH-reference Cell_IDs | 0 |
| SNI exact 200-gene vectors vs cleaned MERFISH reference | 116 |

Direct duplicated source identities are excluded from SNI training.

## 8. Official 200-Gene Alignment

| Item | Value |
|---|---|
| Official genes present | 200 / 200 |
| Exact 200 / 200 | True |
| Aligned to official MODEL V2 order | True |
| Ordered 200-gene SHA256 | `e3301724038990aa2db237026316aaa5fd265a11231c343bea733f8106ab06f5` |
| Matches frozen contract | True |

Missing official genes were not imputed. Genes were not alphabetically reordered.

## 9. Taxonomy Mapping

**CASE B.** Mapping uses the existing project function `merfish60.reference.norm_label` on SNI `obs["voting"]`. This is not a newly invented biological map.

| Item | Value |
|---|---|
| Raw SNI label count | 60 |
| Mapped competition classes | 60 / 60 |
| Missing competition classes | none |
| Cells retained by taxonomy | 55331 |
| Cells excluded by taxonomy | 0 |
| Per-class min / median / max | 18 / 379.5 / 7448 |

`tangram`, `seurat`, `singler`, and `rctd` were inspected as label candidates. `voting` is the complete consensus column covering all 60 classes with zero missing values after `norm_label`. No class was mapped merely because names looked similar.

## 10. Exact-Vector Duplicate Audit

Exclude competition-train/test ID overlaps, competition exact 200-gene vectors, and exact 200-gene vectors that reproduce the cleaned MERFISH reference so that source diversity is not artificially inflated.

| Comparison | Count |
|---|---:|
| SNI exact vectors matching competition train | 10 |
| SNI exact vectors matching competition test | 15 |
| SNI exact vectors matching cleaned MERFISH reference | 116 |
| Usable SNI cells after all exclusions | 55193 |

## 11. Source Distribution Comparison

Unlabeled/test-safe summaries. Competition labels were not used to fit the SNI model.

| Quantity | SNI usable | Competition train | Cleaned MERFISH |
|---|---:|---:|---:|
| Library-size mean | 37.709 | 31.317 | 31.329 |
| Library-size median | 20.000 | 21.000 | 21.000 |
| n_detected mean | 15.524 | 14.859 | 14.949 |
| n_detected median | 11.000 | 12.000 | 12.000 |

Per-gene mean correlation, SNI vs competition train: 0.9744311404594553

Per-gene variance correlation, SNI vs competition train: 0.952808857603493

Unlabeled 2-PC mean Euclidean separation (SNI vs competition train): 0.12052429467439651

No domain-adaptation model was trained.

## 12. SNI-Only Expert Design

Experimental ID: **V3-E04S-SNI**. This is not MODEL V3.

Controlled variable: **reference source** (SNI vs frozen MERFISH MODEL V2).

Reused from V2-B-REFONLY:

- official 200-gene order
- library-size log1p using the median total of the 10,000 unlabeled competition cells
- StandardScaler + PCA50 (`random_state=0`)
- LightGBM objective, hyperparameters, 700 rounds, seed 0
- canonical 60-class probability order

Intentionally omitted because they would break source isolation or require competition-label fitting:

- MERFISH-reference cells
- spatial neighbors / expression graphs / neighbor-label histograms
- neural nets, family specialists, S0
- E/I and Segment post-masks (SNI has no Laminae/Segment; V2 masks need MERFISH-reference Segment or train-derived E/I maps)

PCA/scaler are fit on usable SNI cells only, then applied to competition matrices. That is a necessary source-isolation difference versus V2's transductive PCA on the MERFISH deposit.

## 13. Honest Competition External-Validation Performance

The SNI model is trained only on external SNI labels. Predictions on the 5000 competition train cells are **honest external-validation predictions**, not competition-label OOF.

| Metric | Value |
|---|---|
| Accuracy | 0.5682 |
| Correct | 2841 / 5000 |
| Macro-F1 | 0.4217 |
| Log loss | 1.9809 |
| Canonical folds 0-2 | 0.5777 |
| Canonical folds 3-4 | 0.5540 |

Fold accuracies: {'0': 0.574, '1': 0.583, '2': 0.576, '3': 0.559, '4': 0.549}

## 14. Canonical Partition Stability

Canonical analysis partition: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Folds 3-4 were not used to redesign the model.

| Split | SNI accuracy | sni_new_unique_recoveries | four-expert oracle |
|---|---:|---:|---:|
| overall | 0.5682 | 53 | 0.8728 |
| folds 0-2 | 0.5777 | 26 | 0.8757 |
| folds 3-4 | 0.5540 | 27 | 0.8685 |

## 15. Pairwise Complementarity

**ORACLE != DEPLOYABLE ACCURACY.**

| A | B | A acc | B acc | Disagree | Pair oracle n | Pair oracle acc | Headroom vs stronger |
|---|---|---:|---:|---:|---:|---:|---:|
| lzh_prior_h | sni | 0.8266 | 0.5682 | 1771 | 4238 | 0.8476 | 0.0210 |
| wyh_model_v2 | sni | 0.8212 | 0.5682 | 1789 | 4233 | 0.8466 | 0.0254 |
| s0 | sni | 0.6302 | 0.5682 | 1410 | 3445 | 0.6890 | 0.0588 |

## 16. Recovery of the Remaining 689 Shared Failures

Primary metric:

`sni_new_unique_recoveries` = LZH wrong AND WYH wrong AND S0 wrong AND SNI correct

**sni_new_unique_recoveries = 53**

Shared-failure registry rows: 689 (expected 689).

## 17. Four-Expert Diagnostic Oracle

four_expert_oracle_correct = 4311 + sni_new_unique_recoveries = **4364**

four_expert_oracle_accuracy = 4364 / 5000 = **0.8728**

Remaining all-four-wrong cells: **636**

**ORACLE != DEPLOYABLE ACCURACY.**

## 18. Biological Family Recovery

SNI standalone family accuracy:

| family | n | accuracy |
|---|---:|---:|
| oligodendrocyte_opc | 1552 | 0.6456 |
| astrocyte | 769 | 0.7321 |
| vascular | 363 | 0.6226 |
| meningeal | 128 | 0.4922 |
| microglia | 64 | 0.7031 |
| remaining_glial_non_neuronal | 82 | 0.7683 |
| neuronal_or_other | 2042 | 0.4305 |

SNI new unique recoveries by family, compared with S0's previous 96 rescues (oligo/OPC, vascular, astrocyte-heavy):

| family | SNI new unique recoveries |
|---|---:|
| oligodendrocyte_opc | 36 |
| astrocyte | 5 |
| vascular | 2 |
| meningeal | 3 |
| microglia | 1 |
| remaining_glial_non_neuronal | 0 |
| neuronal_or_other | 6 |

S0 previous unique-rescue family counts: oligodendrocyte_opc 49, astrocyte 14, vascular 10, meningeal 6, microglia 7, remaining_glial_non_neuronal 1, neuronal_or_other 9.

## 19. Shared Confusion Recovery

Among the 689 all-three-wrong cells, ranked true_label → LZH_pred pairs newly resolved by SNI:

| true_label | LZH pred | errors | SNI rescues | fraction | family |
|---|---|---:|---:|---:|---|
| oligodendrocyte_1 | oligodendrocyte_progenitor_2 | 58 | 0 | 0.0000 | oligodendrocyte_opc |
| oligodendrocyte_progenitor_2 | oligodendrocyte_1 | 36 | 20 | 0.5556 | oligodendrocyte_opc |
| oligodendrocyte_progenitor_2 | oligodendrocyte_2 | 35 | 8 | 0.2286 | oligodendrocyte_opc |
| oligodendrocyte_2 | oligodendrocyte_progenitor_2 | 27 | 0 | 0.0000 | oligodendrocyte_opc |
| endothelial | astrocyte_1 | 26 | 0 | 0.0000 | vascular |
| oligodendrocyte_progenitor_2 | astrocyte_1 | 26 | 2 | 0.0769 | oligodendrocyte_opc |
| oligodendrocyte_1 | astrocyte_1 | 23 | 1 | 0.0435 | oligodendrocyte_opc |
| oligodendrocyte_progenitor_1 | oligodendrocyte_precursor_cell | 22 | 0 | 0.0000 | oligodendrocyte_opc |
| astrocyte_1 | oligodendrocyte_progenitor_2 | 21 | 1 | 0.0476 | astrocyte |
| DH_in_Klhl14 | DH_in_Cdh3 | 21 | 0 | 0.0000 | neuronal_or_other |
| meninges_2 | meninges_1 | 16 | 1 | 0.0625 | meningeal |
| endothelial | oligodendrocyte_1 | 16 | 0 | 0.0000 | vascular |

This analysis was not used to retune the model.

## 20. SNI Confidence Diagnostics

Within the 689 shared failures, SNI-correct vs SNI-wrong. Diagnostic only; no threshold was optimized.

| group | n | top1 mean | margin mean | entropy mean |
|---|---:|---:|---:|---:|
| SNI-correct | 53 | 0.710464434467053 | 0.5407700898776464 | 0.8169651469991901 |
| SNI-wrong | 636 | 0.7617040289780668 | 0.6286718137446309 | 0.7242200820325654 |

Diagnostic AUROC for SNI-correct vs SNI-wrong:

- top1: 0.4194553221787113
- margin: 0.41091135635457454
- negative entropy: 0.4368695858549899

## 21. Leakage Audit

- competition test labels used: False
- competition train labels used to fit SNI: False
- external competition ID overlaps excluded: True
- exact competition vector duplicates excluded: True
- MERFISH-reference exact-vector duplicates excluded: True
- leaderboard tuning: False
- hyperparameter search: False
- seed search: False
- post-hoc blend optimization: False
- canonical folds 3-4 used for redesign: False
- test probabilities used for model selection: False
- submission generated: False
- prediction/prediction.csv modified: False
- MODEL V1 / V2 / V3-E00T / E02D / E03A modified: False

## 22. Limitations

- SNI labels come from the file's `voting` consensus, not a wet-lab gold standard.
- The SNI matrix mixes Sham and SNI injury conditions; this experiment treats the file as one independent source rather than splitting condition.
- Spatial/graph features were omitted by design, so this is not a full reimplementation of every V2-B feature, only the source-isolated gene LightGBM analog.
- Diagnostic oracle headroom is not deployable accuracy.
- Rare classes remain rare (usable min count 18).

## 23. Decision

**STRONG SOURCE-DIVERSE EXPERT**

SNI recovered at least 40 all-three-wrong cells, four-expert oracle reached 0.87, and locked folds 3-4 also contribute.

Recommended next experiment/action:

Admit this isolated SNI-only expert into the future V3 expert pool as a frozen source-diverse member, then run a separate controlled experiment on how to use it. Do not concatenate SNI with the MERFISH reference, train a router, or create MODEL V3 yet.
