# WYH Contribution Record — MERFISH Hackathon V3 Research Program

This record documents work owned on the personal development branch
`ywan/ml-pipeline`. It is an attribution and evidence map, not a MODEL V3
freeze and not a leaderboard claim.

Oracle values are diagnostic coverage ceilings. They are not personal
accuracy scores.

---

## 1. Scope of My Work

Supported by the personal repository history:

- reproducible validation pipeline foundation (official contracts, Cell_ID
  integrity, frozen fold files, submission validators)
- personal **MODEL V1** (hierarchical signature specialists; frozen tag
  `model-v1`; OOF 0.7598)
- personal **MODEL V2** (reference-only LightGBM on the approved Zenodo
  MERFISH deposit; frozen tag `model-v2`; 0.8212, 4106 / 5000)
- external-reference modeling, provenance, and duplicate-exclusion work for
  the approved Zenodo file
  (`ce06f62c0ec4973581dae17bb76f0cd9`; usable 136,574)
- V3 research program E00T–E07D
- provenance / leakage / artifact discipline for those experiments
- source-diverse SNI investigation (`SNI_merged_0917.h5ad`)
- source-balanced multi-reference experiment (M1/M2)
- final statistical model-promotion audit (E07D)

Teammate models are used only as frozen comparators or excluded when
cell-level artifacts are unavailable. They are not re-attributed here.

---

## 2. Research Questions I Investigated

1. Is current team expert complementarity sufficient to justify routing or
   blending over the auditable pool?
2. Can privileged genes in the 500-gene same-study reference transfer into
   the official 200-gene task?
3. Can a weak but complementary expert rescue strong-model errors using
   test-safe confidence signals?
4. Does a new biological source contribute genuinely independent information?
5. Can that information be used through prediction-direction rules at
   inference time?
6. Can it instead be incorporated during training via source-balanced
   multi-reference learning?
7. Does the resulting candidate meet a predeclared evidence threshold for
   formal version promotion?

---

## 3. Methods I Implemented / Evaluated

### Personally implemented / evaluated

- MODEL V2 external-reference LightGBM (V2-B-REFONLY)
- V3 analysis and modeling scripts under `experiments/v3/`
- S0 / privileged-gene teacher–student evaluation (E02D)
- weak-expert rescue audit of frozen S0 (E03A; no gate trained)
- SNI source expert (E04S)
- directional complementarity audit (E05A; analysis only)
- source-balanced M1/M2 experiment (E06M)
- statistical final audit (E07D; analysis only)

### Team methods used as frozen comparators

- **LZH Prior-H** (`depth_masked_prior_h_anchor`): used here as a frozen team comparator. I did not develop this path.
- **YHH**: mentioned only where required for audit context. Cell-level
  auditable OOF artifacts were unavailable; YHH outputs are not claimed and
  were not reconstructed.
- **team main**: treated as team/integration output, not as my personal
  model.

---

## 4. Quantitative Contributions

Defensible numbers only. Diagnostic oracles are not personal accuracy.

| Contribution | Value | Kind |
|---|---|---|
| Frozen personal MODEL V2 | 0.8212 (4106 / 5000) | deployable standalone |
| Experimental M2 candidate | 0.8218 (4109 / 5000) | experimental; not MODEL V3 |
| SNI new unique recoveries beyond LZH + WYH + S0 | 53 | diagnostic complementarity |
| Four-expert diagnostic oracle | 0.8728 (4364 / 5000) | coverage ceiling |
| M2 new unique recoveries beyond four-expert pool | 29 | diagnostic complementarity |
| Five-expert diagnostic oracle | 0.8786 (4393 / 5000) | coverage ceiling |

Correct wording:

My source-diverse and source-balanced experiments expanded the auditable
expert-pool diagnostic coverage ceiling to 87.86%, although this did not
translate into a stable standalone model improvement.

Incorrect wording, not used:

- treating the 87.86% diagnostic oracle as personal accuracy
- calling the experimental 82.18% candidate a MODEL V3 score
- describing LZH Prior-H as a WYH-owned model

---

## 5. Negative Results I Preserved

| Result | Frozen label | Why it matters |
|---|---|---|
| Privileged-gene KD / dual-level distillation did not beat S0 | PROMISING BUT INSUFFICIENT | Privileged signal exists; transfer mechanism failed |
| S0 confidence / margin / entropy gating was unsafe | RESCUE SIGNAL WEAK | Complementary answers are not automatically selectable |
| Directional H1/H2/H3 patches had negative net correction | DIRECTIONAL SIGNAL WEAK | True-class asymmetry ≠ inference-time rule |
| M2 failed frozen STRONG promotion criteria | MODEL V3 PROMOTION NOT JUSTIFIED | +3 cells is not a version |

Preserving these results keeps the promotion decision auditable. Hiding them
would make the 0.8786 oracle look like a deployable score.

---

## 6. Reproducibility / Engineering Contributions

- fixed validation contracts and official data-integrity checks
- deterministic personal 3-fold and team-compatible 5-fold files
- Cell_ID lossless-string handling
- checksum provenance (Zenodo MD5, SNI MD5, gene-order SHA256)
- duplicate and ID-overlap exclusion for external references
- machine-readable experiment outputs (JSON, CSV, Parquet)
- unit tests for each E-stage and for this closure package
- experiment reports with leakage audits
- frozen Git checkpoints and annotated tags `model-v1`, `model-v2`
- explicit STRONG / PROMISING / rejection criteria that were not lowered
  after seeing results

---

## 7. Team Contribution Boundaries

**LZH.** Owns / developed the Prior-H path as reflected in team artifacts
(`team/lzh`; selected `depth_masked_prior_h_anchor`, reproduced 0.8266).
WYH uses LZH only as a frozen auditable comparator. This document does not claim ownership of LZH Prior-H.

**YHH.** Separate teammate specialist work. Reported scores were not
reproduced from committed cell-level OOF in E00T. They are not attributed
to WYH and were not used as oracle members.

**Team main.** Shared integration / ensemble space. Current HEAD at the
E00T snapshot was prediction-only (team blend v9). All team-main content
is not attributed to one person, and not to WYH.

**WYH.** Owns the personal-branch experiments and artifacts documented by
the personal Git history on `ywan/ml-pipeline`, including MODEL V1, MODEL V2,
and V3-E00T through V3-E07D.

Statements that cannot be verified from committed artifacts are omitted.

---

## 8. Final Personal Result

MODEL V2 remains the personal frozen deployable model (0.8212).

MODEL V3 was not created because M2 failed the predeclared promotion
threshold (net +3, McNemar p 0.8895, folds 3–4 net −9, bootstrap interval
crossing 0, failed STRONG criteria).

This is a deliberate evidence-based decision, not an unfinished experiment.

---

## 9. Reproducibility / Commit Map

| Identity | Commit / tag |
|---|---|
| MODEL V1 | tag `model-v1` |
| MODEL V2 | tag `model-v2` (`2a25ee2`) |
| V3-E00T | `f117b29` |
| V3-E02D | `718395d` |
| V3-E03A | `a683a22` |
| V3-E04S | `61f6194` |
| V3-E05A | `4adf802` |
| V3-E06M | `a189841` |
| V3-E07D | `e7f94fd` |

Scripts live under `work/experiments/v3/`. Reports live under `work/reports/v3/`.
Machine-readable summaries live under `work/experiments/evidence/v3/`.

---

## 10. Suggested External Description

### A. One-sentence resume / portfolio description

Conducted a seven-stage MERFISH cell-type annotation research program that
introduced source-diverse external-reference experts and expanded the
auditable team expert pool’s diagnostic oracle coverage to 87.86%, while
retaining an 82.12% frozen personal model after a predeclared statistical
promotion audit rejected an unstable +3-cell successor.

### B. Three-bullet technical project description

- Built and froze a reference-only LightGBM personal model (MODEL V2) at
  82.12% team-compatible five-fold fold-safe validation accuracy using a
  provenance-audited Zenodo MERFISH reference with 136,574 usable cells
  after competition-ID and exact-vector duplicate exclusion.
- Showed that a biologically distinct SNI source is a weak standalone
  classifier (56.82%) but contributes 53 unique recoveries, and that
  explicit 0.5/0.5 source-balanced training beats naive pooling without
  becoming a stable MODEL V2 replacement.
- Preserved negative results (privileged-gene distillation, confidence
  gating, directional patches, and MODEL V3 promotion) with frozen Git
  checkpoints, leakage audits, and machine-readable artifacts.

### C. 90-second interview explanation

I worked on 60-class MERFISH cell-type annotation. My frozen personal model
is a reference-only LightGBM, MODEL V2, at 82.12% on a locked 5-fold
protocol. A teammate model, LZH Prior-H, is stronger as a standalone
comparator at 82.66%; I did not build that model. The V3 program asked
whether complementary experts could raise deployable accuracy. We found
real extra information: privileged 500-gene signal on the same-study
reference, a weak neural model that uniquely recovers 96 hard cells, and a
new SNI source that uniquely recovers 53 more. Together the auditable pool
reaches a 87.86% diagnostic oracle, which is a coverage ceiling using true
labels, not a model score. Distillation, confidence gating, and class-direction
patches all failed to use that coverage safely. Train-time source balancing
beat naive pooling but gained only three net cells, with late-fold
regression and a McNemar p-value of 0.89. I therefore did not create
MODEL V3. The contribution is the experimental sequence, the source-diversity
work, and the decision not to over-claim a statistically unsupported version.
