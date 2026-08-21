# V3-E02D — Privileged-Gene Dual-Level Distillation

Research codename: **PGD-200**. This is a MODEL V3 research experiment, not a formal MODEL V3 freeze.

## 1. Research Question

Can biological information in the 300 reference-only genes of the approved same-study 500-gene MERFISH deposit be transferred into a **deployable student that sees only the official 200 genes** at competition inference time?

The proposed transfer uses a 500-gene privileged teacher and two aligned targets:

1. 60-class soft probability structure (temperature-scaled KL)
2. class-aligned latent biological relation structure (cosine similarity to 60 named class prototypes)

Primary goal: recover cells that **both LZH Prior-H and WYH MODEL V2 currently miss**, not merely raise standalone accuracy.

## 2. Motivation from V3-E00T

V3-E00T established the currently auditable honest expert pool:

- LZH `depth_masked_prior_h_anchor`: 0.8266 (4133 / 5000)
- WYH MODEL V2: 0.8212 (4106 / 5000)
- two-expert diagnostic oracle: **4215 / 5000 = 0.8430**
- both-wrong population: **785**
- 85% diagnostic target: 4250 / 5000, so a new expert must recover **at least 35** of those 785 cells

YHH V7 and current team-main were not used: they lack auditable cell-level OOF artifacts. Their reported scores were not inferred.

Diagnostic oracle is **not** deployable model accuracy.

## 3. Novelty Relative to Existing Team Work

This is a **new methodological direction within the team/project**. It is **not** a claim that knowledge distillation is globally novel.

No auditable team method currently implements the same mechanism:

`500-gene privileged teacher → 200-gene deployable student → soft-logit distillation + class-aligned latent-relation distillation`

Nearby but non-equivalent methods:

- WYH MODEL V2 uses only the official 200 genes from the same 500-gene deposit
- LZH Prior-H / graph stacker use biology priors and stacking on competition-scale features
- LZH gene-token KL is same-input clean-vs-corrupt consistency, not 500-to-200 privileged transfer
- YHH / team-main combine reference LightGBM, specialists, and spatial/expression neighbors

## 4. External Reference Provenance

| Item | Value |
|---|---|
| Source | Zenodo record 18039571 |
| File | `work/external/MERFISH_spinal_cord_resolved_0718.h5ad` |
| MD5 | `ce06f62c0ec4973581dae17bb76f0cd9` |
| Raw shape | 146621 cells × 500 genes |
| Usable reference | 136574 |
| Official 200-gene SHA256 | `e3301724038990aa2db237026316aaa5fd265a11231c343bea733f8106ab06f5` |
| Teacher 500-gene SHA256 | `0ca0e36cc93f8dcb47322214d0f0c35490f2eeb0dbcb2ba661db151f4e52b74a` |
| 200 ⊂ 500 | True |

## 5. Exclusion / Leakage Audit

Removed 5000 competition train Cell_IDs, 5000 competition test Cell_IDs, and 47 exact aligned 200-gene count-vector duplicates. Historical targets (5000 / 5000 / 47 / 136574) reproduced exactly.

Usable reference Cell_IDs are disjoint from competition train and test IDs. Competition test labels were not used. Students are trained only on approved external-reference cells; competition-train scores are **honest external-validation predictions**, not conventional competition OOF.

## 6. Stage A — 200 vs 500 Gene Signal

A200 (200 genes) validation accuracy **0.6193**, macro-F1 **0.5068**, log loss **1.2275**.
A500 (500 genes) validation accuracy **0.8737**, macro-F1 **0.8360**, log loss **0.3725**.

Deltas (A500 − A200): accuracy **0.2544**, macro-F1 **0.3292**, difficult-family mean recall **0.3225**.

| Family | A200 recall | A500 recall | delta |
|---|---:|---:|---:|
| oligodendrocyte_opc | 0.5807 | 0.8718 | 0.2911 |
| astrocyte | 0.6895 | 0.9128 | 0.2233 |
| vascular | 0.5604 | 0.8966 | 0.3362 |
| meningeal | 0.4721 | 0.9115 | 0.4394 |
| other_glial_non_neuronal | 0.7626 | 0.9064 | 0.1438 |
| difficult_family_mean_recall | 0.5757 | 0.8982 | 0.3225 |

Decision: **PRIVILEGED SIGNAL SUPPORTED**.
Criteria: A (acc +0.005)=True, B (macro-F1 +0.005)=True, C (difficult-family recall +0.010)=True. Material collapse=False.


## 7. 500-Gene Cross-Fitted Teacher

Protocol: StratifiedKFold(n_splits=3, shuffle=True, random_state=20260820) on the usable reference. Cross-fitted teacher accuracy on held-out reference cells: **0.8713** (118998 / 136574), macro-F1 **0.8305**, log loss **0.3924**. Relation targets have shape (n_usable, 60).

## 8. Class-Aligned Relational Representation

Raw 128-d latents from independently trained fold-teachers are **not** matched. Independently trained networks can rotate or permute latent coordinates, so a direct SmoothL1 / MSE on those vectors would mix incompatible axes.

Instead, each teacher fold:

1. embeds its teacher-training reference cells
2. computes one 128-d prototype per official class
3. stores held-out cosine similarities to those 60 named prototypes

The resulting 60-d relation target is aligned across folds because each dimension is a named cell class in canonical order.

## 9. Student Ablations

All students see **only official 200 genes**.

- **S0** hard-label CE student (architecture control)
- **S1** CE + λ_kd T² KL(teacher_T || student_T), T=2.0, λ_kd=0.50
- **S2** S1 plus λ_rel SmoothL1 on the class-aligned relation head, λ_rel=0.25

No coefficient, seed, or architecture search was performed. Competition prediction uses only the student classification head.

## 10. Competition External-Validation Results

S0 honest external-validation accuracy **0.6302** (3151 / 5000), macro-F1 **0.5196**, log loss **1.1979**. Canonical folds 0-2: **0.6397**. Locked folds 3-4: **0.6160**.

S1 honest external-validation accuracy **0.6294** (3147 / 5000), macro-F1 **0.5088**, log loss **1.2641**. Canonical folds 0-2: **0.6357**. Locked folds 3-4: **0.6200**.

S2 honest external-validation accuracy **0.6292** (3146 / 5000), macro-F1 **0.5089**, log loss **1.2642**. Canonical folds 0-2: **0.6363**. Locked folds 3-4: **0.6185**.

These numbers are honest external validation: the student never trained on competition labels.

## 11. Canonical Folds 0-2 vs Locked Folds 3-4

Canonical seed42 folds remain analysis partitions only. Model definitions were not changed after seeing folds 3-4.

| Variant | folds 0-2 | folds 3-4 |
|---|---:|---:|
| S0 | 0.6397 | 0.6160 |
| S1 | 0.6357 | 0.6200 |
| S2 | 0.6363 | 0.6185 |

## 12. Complementarity with LZH and WYH

Reverified LZH + WYH oracle: 4215 / 5000 = 0.8430; both-wrong = 785.

| Combo | Oracle correct | Oracle accuracy |
|---|---:|---:|
| LZH + S0 | 4247 | 0.8494 |
| WYH + S0 | 4258 | 0.8516 |
| LZH + WYH + S0 | 4311 | 0.8622 |
| LZH + S2 | 4242 | 0.8484 |
| WYH + S2 | 4245 | 0.8490 |
| LZH + WYH + S2 | 4303 | 0.8606 |

Derived, not trained:

LZH + WYH oracle = 4215 / 5000

S0 `new_unique_recoveries` = 96

LZH + WYH + S0 diagnostic oracle = **4311 / 5000 = 0.8622**

This is a derived oracle result, **NOT** a deployable model score.

## 13. Recovery of the 785 Shared Errors

`new_unique_recoveries` is defined exactly as LZH wrong AND WYH wrong AND student correct.

- S0 unique recoveries: **96**
- S1 unique recoveries: **88**
- S2 unique recoveries: **88**

4215 + S0 unique recoveries = **4311**.

4215 + S2 unique recoveries = **4303**.

## 14. Three-Expert Diagnostic Oracle

LZH + WYH + S0 diagnostic oracle: **4311 / 5000 = 0.8622**.

LZH + WYH + S2 diagnostic oracle: **4303 / 5000 = 0.8606**.

Canonical folds 3-4 LZH + WYH + S2 oracle: **1705 / 2000 = 0.8525**.

Does the overall three-expert diagnostic oracle reach ≥ 0.85? **YES** (S0 0.8622; S2 0.8606).

Remaining all-three-wrong cells with S2: **697**. Remaining with S0: **689**.

**DIAGNOSTIC ORACLE != DEPLOYABLE MODEL ACCURACY.**

Although privileged distillation failed to outperform the hard-label student, S0 provides stronger unique complementarity than S1/S2 and therefore becomes the preferred candidate third expert for the next rescue audit.

## 15. Biological / Confusion-Family Analysis

S2 unique recoveries by true family: oligodendrocyte_opc (n=47), astrocyte (n=14), neuronal_or_other (n=9), other_glial_non_neuronal (n=7), vascular (n=6), meningeal (n=5)

Top newly recovered true classes: oligodendrocyte_progenitor_2 (n=24), oligodendrocyte_1 (n=16), astrocyte_1 (n=10), microglia (n=6), endothelial (n=5), astrocyte_2 (n=4), meninges_3 (n=3), oligodendrocyte_2 (n=3)

This is the evidence for whether privileged representation transfer helped the current shared-hard cases, especially oligodendrocyte / OPC, astrocyte, vascular, and meningeal families.

## 16. Evidence for or Against Dual-Level Distillation

Stage A privileged signal: **PRIVILEGED SIGNAL SUPPORTED**.

Unique-recovery ordering S0 / S1 / S2: 96 / 88 / 88.

The strongest project-level evidence would have been S0 < S1 < S2 unique recoveries and a three-expert oracle ≥ 0.85. That pattern is **not supported by the observed unique-recovery ordering**.

Although privileged distillation failed to outperform the hard-label student, S0 provides stronger unique complementarity than S1/S2 and therefore becomes the preferred candidate third expert for the next rescue audit. LZH + WYH + S0 diagnostic oracle = **4311 / 5000 = 0.8622**. This is a derived oracle result, not a deployable model score.

## 17. Leakage Audit

{
  "canonical_folds_3_4_used_for_tuning": false,
  "cell_id_dtype": "lossless_string",
  "competition_test_labels_used": false,
  "competition_train_labels_used_for_student_training": false,
  "hyperparameter_search": false,
  "model_v1_modified": false,
  "model_v2_modified": false,
  "prediction_csv_modified": false,
  "privileged_500_genes_at_deployment": false,
  "reference_excludes_train_and_test_ids": true,
  "seed_search": false,
  "student_inference_input_dim": 200,
  "v3_e00t_metrics_modified": false
}

## 18. Limitations

- Teacher and student are modest MLPs with one predeclared budget; this is not an exhaustive neural architecture search
- Early stopping uses a fixed 80/20 reference split; students are then retrained on the full usable reference for the selected epoch count
- Canonical folds 3-4 are a locked confirmation partition for this experiment, not LZH's original 3-fold protocol
- YHH and team-main remain unavailable for honest complementarity
- A diagnostic oracle is not a deployable ensemble

## 19. Experiment Decision

**PROMISING BUT INSUFFICIENT**

Privileged 500-gene signal is supported and the 200-gene student recovers some LZH+WYH shared errors, but S2 did not improve unique recoveries over S0/S1. Dual-level distillation is therefore not a mature third expert.

Recommended next action (not started): Do not freeze MODEL V3 and do not start routing on S2. The next experiment should audit the simpler 200-gene hard-label reference MLP (S0) as a candidate third expert, because S0 recovered more shared errors than S1/S2 and dual-level distillation did not transfer the Stage A 500-gene advantage.

Reproducibility: Python 3.12.13, torch 2.13.0, device mps, seed 20260820, reference MD5 ce06f62c0ec4973581dae17bb76f0cd9.
