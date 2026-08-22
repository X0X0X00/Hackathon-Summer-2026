# WYH — MERFISH Cell-Type Annotation

A reproducible modeling track for 60-class MERFISH cell-type annotation,
focused on external-reference transfer, leakage-safe validation, and
evidence-based model selection.

## Results

| Model | Method | Validation Accuracy | Status |
| --- | --- | ---: | --- |
| MODEL V1 | Hierarchical Signature Specialists | 75.98% | Historical baseline |
| **MODEL V2** | Reference-only LightGBM | **82.12%** | **Selected model** |
| E06M M2 | Source-balanced multi-reference LightGBM | 82.18% | Experimental; not selected |

## Selected Model — MODEL V2

MODEL V2 is the selected WYH model. It uses a cleaned external MERFISH
spinal-cord reference aligned to the official 200-gene panel and a
LightGBM classifier evaluated under the fixed team-compatible five-fold
protocol.

- Validation accuracy: **82.12%** (4106 / 5000)
- External reference cells: **136,574**
- Improvement over MODEL V1: **+6.14 percentage points**
- Model documentation: [`docs/versions/model_v2.md`](docs/versions/model_v2.md)
- Candidate predictions: [`outputs/submissions/model_v2_candidate.csv`](outputs/submissions/model_v2_candidate.csv)

A predeclared V2-C blend reached 82.24% but was not selected because the gain
was only six net cells, later folds regressed, and macro-F1 decreased.

## Post-V2 Research

After MODEL V2, a sequence of controlled experiments tested whether additional
biological information and complementary experts could produce a stable
improvement.

| Direction | Main Finding |
| --- | --- |
| Privileged-gene distillation | The 500-gene reference contained substantially more predictive information, but the tested 500→200 distillation methods did not improve the 200-gene student |
| Weak-expert complementarity | S0 added 96 unique recoveries beyond the two strong experts, but confidence-based rescue was not reliable |
| SNI source diversity | The independent SNI source added 53 further unique recoveries despite weak standalone accuracy |
| Source-balanced multi-reference transfer | Explicit source balancing outperformed naive pooling; E06M M2 reached 82.18%, but the gain over MODEL V2 was not stable enough for selection |
| Final model audit | MODEL V2 was retained after paired, fold, bootstrap, and section-level evaluation |

### Model Selection Decision

E06M M2 achieved 82.18% (4109 / 5000), only three net correct cells above
MODEL V2. The gain was not stable across folds, sections, or paired
statistical evaluation, so MODEL V2 was retained as the selected model.

See [`reports/v3/v3_e07d_final_deployable_decision_audit.md`](reports/v3/v3_e07d_final_deployable_decision_audit.md).

### Main Finding — Coverage vs. Utilization

The combined expert pool reached an **87.86% retrospective diagnostic oracle**,
showing that substantial complementary information exists across the evaluated
experts. However, the tested distillation, confidence-gating, directional
correction, and source-balanced transfer methods could not convert that
coverage into a stable standalone improvement without introducing offsetting
errors.

> **87.86% is a retrospective diagnostic oracle, not a deployable model
> accuracy.**

See [`reports/v3/v3_research_program_summary.md`](reports/v3/v3_research_program_summary.md).

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/merfish60/` | Reusable modeling, feature, and validation utilities |
| `scripts/` | Released-model and pipeline entry points |
| `experiments/` | Controlled modeling and research experiments |
| `outputs/` | Metrics, validation predictions, probabilities, and selected artifacts |
| `reports/` | Detailed experiment reports and decision audits |
| `docs/` | Model documentation and methodology notes |
| `tests/` | Reproducibility and integrity tests |
| `data/`, `prediction/` | Challenge data and prediction interfaces |

## Reproducibility

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/09_v2b_refonly.py
.venv/bin/pytest -q tests/
```

MODEL V2 reproduction requires the approved external reference described in
[`docs/versions/model_v2.md`](docs/versions/model_v2.md).

## Challenge

Developed for the University of Rochester Biomedical Data Science Hackathon
Summer 2026.

See [`Data.Description.md`](Data.Description.md) for the task/data description
and the [upstream repository](https://github.com/Rochester-Biomedical-DS/Hackathon-Summer-2026)
for the original challenge information.
