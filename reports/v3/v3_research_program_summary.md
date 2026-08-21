# WYH V3 Research Program — Final Summary

This document closes the WYH V3 research program. It is a research-program
summary, not a model freeze. MODEL V3 was not created.

Frozen decision checkpoint: `e7f94fd`
(`exp(v3): finalize deployable candidate decision audit`).

Oracle values below are diagnostic coverage ceilings. They ask, retrospectively
and using true labels, whether at least one expert was already correct. They
are not deployable accuracy, not OOF accuracy, not test accuracy, and not
leaderboard accuracy.

---

## 1. Executive Summary

The competition task is 60-class MERFISH cell-type annotation on 5000 labeled
training cells and 5000 hidden-label test cells, using 200 official genes and
overall accuracy.

Frozen personal deployable model:

- **MODEL V2** (V2-B-REFONLY): **0.8212** (**4106 / 5000**)
- Official score: **Not submitted**

Strongest currently auditable team standalone expert:

- **LZH Prior-H**, used here as a frozen team comparator: **0.8266**
  (**4133 / 5000**)

The V3 program asked whether a methodologically distinct improvement path
existed beyond MODEL V2 without violating leakage, fold, or promotion
discipline. Seven controlled experiments (E00T–E07D) established that
complementary expert coverage is real, then showed that converting that
coverage into a stable standalone gain was the actual bottleneck.

Final five-expert diagnostic oracle (LZH Prior-H + WYH MODEL V2 + S0 + SNI +
experimental M2): **4393 / 5000 = 0.8786**. This is a coverage ceiling, not a
model score. Do not report 0.8786 as deployable or OOF accuracy.

Final personal decision: **MODEL V3 PROMOTION NOT JUSTIFIED.**

The experimental source-balanced candidate M2 reached **0.8218**
(**4109 / 5000**), a net of **+3** cells versus MODEL V2. Exact McNemar
p-value **0.8895**. Canonical folds 3–4 net **−9**. Cell-level bootstrap 95%
interval **[−0.00500, +0.00620]**, which includes 0. M2 is not MODEL V3.

Central research insight: the principal bottleneck was not lack of
complementary expert coverage. The auditable expert pool reached a 0.8786
diagnostic oracle, but multiple controlled experiments showed that the
complementary information could not be converted into a stable deployable
improvement without producing offsetting errors.

---

## 2. Problem Definition

University of Rochester Biomedical Data Science Hackathon Summer 2026.

| Item | Value |
|---|---|
| Target | `MERFISH_cell_type_annotation` |
| Official metric | overall accuracy |
| Labeled training cells | 5000 |
| Hidden-label test cells | 5000 |
| Official genes | 200 |
| Classes | 60 |
| Cell_ID | lossless 19-digit string |

Official files: `data/counts_train.csv`, `data/counts_test.csv`,
`data/meta_train.csv`, `data/meta_test.csv`. Test labels remain hidden.

MODEL V1 (hierarchical signature specialists; frozen 3-fold OOF **0.7598**)
is historical. MODEL V2 (reference-only LightGBM on the cleaned Zenodo
MERFISH spinal-cord deposit) is the current frozen personal deployable model.

---

## 3. Research Objective

V3 aimed to find a methodologically distinct improvement path beyond frozen
MODEL V2 and existing team approaches, while preserving:

- fixed canonical fold definitions
- leakage control
- artifact traceability
- predeclared success/failure criteria
- negative-result preservation

The program was not a search for a higher single OOF number by blending,
threshold tuning, seed search, or informal router training.

MODEL V3 is a formal version identity. It is created only if a candidate
meets frozen promotion evidence. That bar was not met.

---

## 4. Validation / Evidence Discipline

Canonical analysis partition for V3 comparisons:

`experiments/team_folds_5_seed42.csv`

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

This is the original protocol for WYH MODEL V2. Canonical folds 3–4 were
viewed across research stages and are a **retrospective stability partition**,
not a pristine untouched holdout.

Leakage and integrity controls used throughout:

- competition test labels unused
- Cell_ID stored as lossless 19-digit strings
- official 60-class order from training labels
- approved Zenodo MERFISH exclusions: 5000 train IDs, 5000 test IDs, 47 exact
  200-gene vector duplicates; usable reference 136,574
- SNI exclusions reproduced from E04S; usable 55,193
- no leaderboard tuning
- no lucky-seed search
- no post-hoc blend-weight search
- frozen Git checkpoints for each E-stage
- predeclared STRONG / PROMISING / negative-result labels
- `prediction/prediction.csv` unmodified
- MODEL V1 and MODEL V2 artifacts unmodified

External-reference students and SNI/M1/M2 candidates are scored as
**honest external-validation predictions** on the 5000 competition-train
cells unless a frozen report states otherwise. Competition labels do not
enter those boosting or student-training objectives. That is not identical
to conventional competition-label OOF.

---

## 5. Starting Point

At program start, two honest cell-level experts were auditable:

| Expert | Owner | Accuracy | Correct |
|---|---|---:|---:|
| LZH Prior-H (`depth_masked_prior_h_anchor`) | LZH (frozen comparator) | 0.8266 | 4133 |
| WYH MODEL V2 | WYH | 0.8212 | 4106 |

LZH + WYH diagnostic oracle: **4215 / 5000 = 0.8430**. Both wrong: **785**.

YHH V7 and current team-main v9 lacked committed cell-level OOF artifacts and
were omitted from the honest oracle pool. The 0.8430 figure describes only
the then-auditable pair.

An 85% diagnostic coverage target is 4250 / 5000. The two-expert pool was
35 unique recoveries short of that ceiling, and unconstrained replacement of
LZH by MODEL V2 had net **−27** (82 recoveries vs 109 false corrections).
That motivated independent information acquisition rather than a router over
LZH + WYH alone.

---

## 6. Experiment Timeline

| Experiment | Research question | Method | Key quantitative result | Decision | Commit |
|---|---|---|---|---|---|
| V3-E00T | Does the auditable team pool have enough complementary coverage to justify integration? | Canonical Cell_ID-aligned expert audit; no training | LZH 0.8266; WYH 0.8212; pair oracle 4215 / 5000 = 0.8430 | ROUTING-ONLY INSUFFICIENT | `f117b29` |
| V3-E02D | Can the extra 300 reference-only genes transfer into a 200-gene student? | 500-gene teacher vs 200-gene students S0/S1/S2 | Reference A200 0.6193 vs A500 0.8737 (Δ +0.2544); S0/S1/S2 = 0.6302 / 0.6294 / 0.6292 | PROMISING BUT INSUFFICIENT; privileged signal real, KD unsuccessful | `718395d` |
| V3-E03A | Can S0's complementary errors be identified with test-safe confidence? | Rescue audit of frozen S0; no gate trained | S0 unique recoveries 96; three-expert oracle 4311 / 5000 = 0.8622; high S0 confidence does not isolate rescues | RESCUE SIGNAL WEAK | `a683a22` |
| V3-E04S | Does a new biological source add genuinely independent recoveries? | Isolated SNI-only LightGBM on `SNI_merged_0917.h5ad` | SNI 2841 / 5000 = 0.5682; 53 new unique recoveries; four-expert oracle 4364 / 5000 = 0.8728 | STRONG SOURCE-DIVERSE EXPERT | `61f6194` |
| V3-E05A | Can S0/SNI rescue become observable class-direction rules? | Predeclared H1/H2/H3 prediction-only patches | H1 net −30; H2 net −8; H3 net −14; D3 net −52 | DIRECTIONAL SIGNAL WEAK | `4adf802` |
| V3-E06M | Can SNI diversity be internalized at train time? | Naive pool M1 vs source-balanced M2 | M1 0.8172; M2 0.8218 (4109); net +3 vs M0; 4 / 53 SNI unique captured; 29 new recoveries; five-expert oracle 4393 / 5000 = 0.8786 | PROMISING SOURCE-BALANCED TRANSFER, not STRONG | `a189841` |
| V3-E07D | Does M2 meet frozen promotion evidence versus MODEL V2? | Paired McNemar, bootstrap, section/class stability; no retraining | McNemar p 0.8895; bootstrap 95% CI includes 0; sections 31 / 45 / 32; folds 3–4 net −9 | MODEL V3 PROMOTION NOT JUSTIFIED | `e7f94fd` |

Failed experiments are retained. They are part of the evidence.

---

## Research Story in One Figure / Flow

Deployable scores are standalone model accuracy. Oracle ceilings are
retrospective coverage, not deployable OOF.

```text
MODEL V2  82.12%   (4106 / 5000)     [deployable]
        |
        v
E00T  coverage audit
      LZH 82.66%  |  pair oracle 84.30%   [diagnostic]
        |
        v
E02D  privileged 500-gene signal exists (0.6193 -> 0.8737 on reference)
      KD / dual-level distillation fails (S0 0.6302 >= S1/S2)
        |
        v
E03A  weak expert S0 adds 96 answers
      three-expert oracle 86.22%   [diagnostic]
      confidence / margin / entropy gate fails
        |
        v
E04S  new source SNI is a weak classifier (56.82%)
      but adds 53 unique recoveries
      four-expert oracle 87.28%   [diagnostic]
        |
        v
E05A  directional class-patch rules fail (H1/H2/H3 nets -30 / -8 / -14)
        |
        v
E06M  source balancing > naive pool (82.18% vs 81.72%)
      M2 vs MODEL V2: only +3 deployable cells
      five-expert oracle 87.86%   [diagnostic]
        |
        v
E07D  statistical / stability audit rejects promotion
      McNemar p 0.8895; bootstrap CI crosses 0; folds 3-4 net -9
        |
        v
MODEL V2 retained as personal deployable model
V3 research program closed
MODEL V3 not created
```

---

## 7. Positive Findings

### A. Privileged 500-gene signal is real

On the cleaned Zenodo reference (MD5 `ce06f62c0ec4973581dae17bb76f0cd9`),
a 200-gene MLP reached reference-domain accuracy **0.6193** and a 500-gene
teacher reached **0.8737** (delta **+0.2544**). Cross-fitted 500-gene teacher
accuracy **0.8713** (118998 / 136574). Difficult-family mean recall rose
**+0.3225**. Label: **PRIVILEGED SIGNAL SUPPORTED**.

### B. Weak neural model adds genuine complementary errors

S0 (hard-label 200-gene reference MLP) has standalone external-validation
accuracy **0.6302**, far below the strong experts, but is uniquely correct on
**96** cells that both LZH Prior-H and WYH MODEL V2 miss. Three-expert
diagnostic oracle: **4311 / 5000 = 0.8622**.

### C. SNI provides genuine source-diverse information

`SNI_merged_0917.h5ad` (MD5 `7e90a801ee57b8fec06cd03c8630f01b`; raw
55331 × 500; usable 55193; official genes 200 / 200; taxonomy 60 / 60) is a
weak global classifier (**2841 / 5000 = 0.5682**) and a strong diversity
member: **53** cells where LZH, WYH, and S0 are all wrong and SNI is correct.
Four-expert diagnostic oracle: **4364 / 5000 = 0.8728**. Folds 0–2: **0.8757**.
Folds 3–4: **0.8685**. SNI did not itself achieve 87.28%.

### D. Explicit source balancing is better than naive pooling

M1 (every combined row weight 1) **0.8172** (4086 / 5000), net **−20** versus
MODEL V2. M2 (explicit 0.5 / 0.5 source mass) **0.8218** (4109 / 5000),
**+23** correct cells versus M1. Balancing is the operative control, not
mere concatenation.

### E. M2 adds 29 new expert-pool recoveries even though standalone promotion fails

M2 is uniquely correct on **29** cells that LZH, WYH, S0, and SNI all miss.
Five-expert diagnostic oracle: **4393 / 5000 = 0.8786**. Identity:
4364 + 29 = 4393. Those 29 cells are coverage evidence, not a promotion
argument.

---

## 8. Negative Findings

### A. Privileged-gene KD did not outperform S0

Competition-train external-validation: S0 **0.6302**, S1 **0.6294**,
S2 **0.6292**. Unique recoveries: S0 **96**, S1 **88**, S2 **88**. Distillation
did not transfer the Stage A 500-gene advantage into a better 200-gene
student. Distillation is unsuccessful under the tested mechanism.

### B. Weak-expert confidence gating was unsafe

S0 top1 AUROC for unique rescues versus remaining shared errors (P vs N1):
**0.4142**. High S0 top1/margin deciles contained almost none of the 96
rescues. Most unique rescues (89 / 96) occur when LZH and WYH already agree.
No high-precision, fold-stable gate was supported. E03A is not a deployable
improvement.

### C. Directional rescue rules produced negative net gains

Predeclared observable patches:

- H1 net **−30** (precision 0.3125)
- H2 net **−8** (precision 0.4310)
- H3 net **−14** (precision 0.2812)
- D3 combined retrospective patch net **−52**

True-class-conditioned rescue asymmetry does not become a safe inference-time
rule. Post-hoc class patches were not promoted.

### D. Naive MERFISH+SNI pooling regressed

M1 **0.8172**, net **−20** versus MODEL V2 (84 recoveries, 104 new errors).
Unweighted concatenation is not a free source-diversity gain.

### E. Source-balanced M2 produced only +3 standalone cells and late-fold regression

M2 vs M0: **105** wrong→correct, **102** correct→wrong, net **+3**.
Folds 0–2 net **+12**. Folds 3–4 net **−9**. SNI's original 53 unique
recoveries captured by M2: **4 / 53**. Frozen STRONG criteria (net ≥ 25,
accuracy ≥ 0.8262, positive net on folds 3–4, SNI capture ≥ 15) all failed.

---

## 9. Deployable Accuracy vs Oracle Coverage

| System | Value | Correct | Kind |
|---|---:|---:|---|
| MODEL V2 | 82.12% | 4106 | deployable standalone |
| M2 (E06M experimental candidate) | 82.18% | 4109 | experimental standalone; **not MODEL V3** |
| LZH Prior-H (frozen team comparator) | 82.66% | 4133 | auditable standalone |
| 2-expert oracle (LZH + WYH) | 84.30% | 4215 | diagnostic coverage |
| 3-expert oracle (+ S0) | 86.22% | 4311 | diagnostic coverage |
| 4-expert oracle (+ SNI) | 87.28% | 4364 | diagnostic coverage |
| 5-expert oracle (+ M2) | 87.86% | 4393 | diagnostic coverage |

Oracle values are diagnostic coverage ceilings, not deployable OOF scores.
The frozen auditable expert pool reached a 87.86% diagnostic oracle coverage ceiling. That ceiling is not a V3 OOF result.

---

## 10. Why MODEL V3 Was Not Promoted

Personal promotion compared M2 with frozen MODEL V2 under criteria frozen
before E07D and not lowered after seeing a three-cell gain.

Evidence against promotion:

- net **+3** cells (105 vs 102 discordant correctness)
- exact McNemar p **0.8895**
- folds 3–4 net **−9**
- cell-level bootstrap mean delta **+0.00061**; 95% interval
  **[−0.00500, +0.00620]** includes 0
- 108 sections: M2 wins 31, ties 45, loses 32
- section-cluster bootstrap 95% interval **[−0.00537, +0.00687]** includes 0
- failed frozen STRONG checks: `net_ge_25`, `acc_ge_0.8262`,
  `folds_3_4_net_positive`, `sni_capture_ge_15`

LZH Prior-H remains the strongest currently auditable standalone expert
(4133 / 5000 = 0.8266). M2 does not overtake LZH and is not a justified
replacement for MODEL V2.

Preserving the promotion bar is part of reproducible research. A weaker or
statistically indistinguishable MODEL V3 was not created for version
numbering.

---

## 11. Final Model State

| Identity | Status |
|---|---|
| MODEL V1 | historical frozen model (0.7598, 3-fold) |
| MODEL V2 | current frozen personal deployable model (0.8212, 5-fold) |
| MODEL V3 | **NOT CREATED** |
| V3 research program | **COMPLETED** |
| LZH Prior-H | strongest currently auditable team standalone expert (comparator) |
| M2 / V3-E06M | experimental source-balanced candidate (0.8218); not a version |

Official personal score remains **Not submitted**.

---

## 12. Research Interpretation

The project exposed a **coverage-versus-utilization gap** at the project
level.

Progression of the scientific question:

1. coverage discovery (E00T)
2. privileged-information test (E02D)
3. weak-expert rescue test (E03A)
4. independent-source discovery (E04S)
5. directional-rule test (E05A)
6. train-time source integration (E06M)
7. final paired / stability audit (E07D)

The question moved from “Can another expert answer difficult cells?” to
“Can complementary answers be identified or internalized reliably without
damaging already-correct predictions?”

Supported by frozen evidence:

- information availability: **YES**
- safe utilization: **UNRESOLVED / INSUFFICIENT UNDER TESTED METHODS**

This is a project-level empirical finding. It is not a claim of global
methodological novelty, and it does not claim that no possible method could
solve utilization. It claims that the tested controlled methods did not.

---

## 13. Limitations

- Only 5000 labeled competition cells are available for canonical evaluation.
- Canonical folds 3–4 were viewed across research stages and are retrospective
  stability partitions rather than a pristine untouched holdout.
- LZH Prior-H uses a native 3-fold protocol; canonical 5-fold numbers are a
  common evaluation partition. The 0.8266 figure is also a selection metric
  on that OOF bundle (MEDIUM validation confidence in E00T).
- External sources have domain shift. SNI labels are `voting` consensus after
  `norm_label`, not a wet-lab gold standard. SNI mixes Sham and SNI injury
  conditions.
- No official hidden-test evaluation exists for personal experimental V3
  candidates. Test-side figures in E07D are predicted-class / confidence
  shift only.
- Diagnostic oracle depends on true labels and assumes a perfect selector.
- Limited time prevented exploration of every possible utilization mechanism
  (learned cross-fitted routers, additional sources, Spatial-ID, architecture
  search).
- YHH and current team-main remain unavailable for honest cell-level
  standalone comparison.

This document does **not** claim that 85% deployable accuracy is impossible.

---

## 14. Reproducibility Map

| Stage | Commit | Script | Report | Primary output artifact |
|---|---|---|---|---|
| E00T | `f117b29` | `experiments/v3/v3_e00t_team_expert_audit.py` | `reports/v3/v3_e00t_team_expert_audit.md` | `outputs/v3/v3_e00t_metrics.json` |
| E02D | `718395d` | `experiments/v3/v3_e02d_privileged_gene_distillation.py` | `reports/v3/v3_e02d_privileged_gene_distillation.md` | `outputs/v3/v3_e02d_student_comparison.csv` |
| E03A | `a683a22` | `experiments/v3/v3_e03a_rescue_audit.py` | `reports/v3/v3_e03a_rescue_audit.md` | `outputs/v3/v3_e03a_rescue_metrics.json` |
| E04S | `61f6194` | `experiments/v3/v3_e04s_sni_source_expert.py` | `reports/v3/v3_e04s_sni_source_expert.md` | `outputs/v3/v3_e04s_complementarity.json` |
| E05A | `4adf802` | `experiments/v3/v3_e05a_directional_complementarity_audit.py` | `reports/v3/v3_e05a_directional_complementarity_audit.md` | `outputs/v3/v3_e05a_directional_metrics.json` |
| E06M | `a189841` | `experiments/v3/v3_e06m_source_balanced_multireference.py` | `reports/v3/v3_e06m_source_balanced_multireference.md` | `outputs/v3/v3_e06m_metrics.json` |
| E07D | `e7f94fd` | `experiments/v3/v3_e07d_final_deployable_decision_audit.py` | `reports/v3/v3_e07d_final_deployable_decision_audit.md` | `outputs/v3/v3_e07d_decision.json` |

Supporting frozen identities:

- MODEL V2 metrics: `outputs/metrics/model_v2_metrics.json`
- MODEL V2 OOF: `outputs/oof/MODEL-V2_oof.csv`
- MODEL V2 tag: `model-v2`
- Closure manifest: `outputs/v3/v3_research_program_manifest.json`
- Closure metrics table: `outputs/v3/v3_research_program_metrics.csv`

---

## 15. Final Takeaway

The WYH V3 program showed that complementary answers to hard MERFISH cells
already exist in an auditable expert pool whose diagnostic coverage ceiling
is 87.86%. The same program showed, under predeclared protocols, that
privileged-gene distillation, confidence-gated rescue, directional class
patches, naive multi-reference pooling, and source-balanced transfer did not
convert that coverage into a stable personal-model replacement for MODEL V2.

MODEL V2 remains the frozen personal deployable model at 82.12%.
MODEL V3 was not created. The research program is complete.
