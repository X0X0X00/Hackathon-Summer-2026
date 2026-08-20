# MODEL V1 — Hierarchical Signature Specialists

This document describes the frozen MODEL V1 release. It is not an official captain-repository submission.

## 1. Status

| Item | Value |
|---|---|
| Identity | MODEL V1 candidate |
| Architecture | Frozen YW-004 full-train per-signature specialists |
| Selection metric | 3-fold out-of-fold (OOF) overall accuracy |
| Primary OOF accuracy | **0.7598** (3799 / 5000) |
| Official hidden-test score | **Not submitted** |
| Official leaderboard score | **Not submitted** |
| Submission candidate | `outputs/submissions/model_v1.csv` (5000 rows; official contract) |

MODEL V1 is **frozen**. The selected architecture is YW-004 hierarchical specialists. The recorded OOF accuracy is a local validation number on persisted 3-fold assignments. MODEL V1 was not submitted for official scoring, so no official hidden-test or leaderboard score is available.

The candidate has exactly 5000 test predictions in `meta_test.csv` order, uses the official columns `Cell_ID,MERFISH_cell_type_annotation.y`, and is checked by the in-repo submission validator.

## 2. Problem Setup

University of Rochester Biomedical Data Science Hackathon Summer 2026: 60-class MERFISH cell-type classification.

| Item | Value |
|---|---|
| Labeled training cells | 5000 |
| Test cells | 5000 (labels fully hidden) |
| Genes | 200 official competition genes |
| Classes | 60, from `meta_train` only |
| Target | `MERFISH_cell_type_annotation` |
| Official metric | overall accuracy (correct / total) |
| Submission prediction column | `MERFISH_cell_type_annotation.y` |

Official inputs are `data/counts_train.csv`, `data/counts_test.csv`, `data/meta_train.csv`, and `data/meta_test.csv`. Cell_IDs are 19-digit integers and are stored as strings.

The official train/test split is a within-section mixed split. Cells are **not** treated as IID.

## 3. Validation Protocol

All MODEL V1 selection numbers use the frozen 3-fold protocol:

```text
StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=20260819
)
```

Assignments are persisted in `experiments/folds.csv` (`Cell_ID`, `fold` in `{0,1,2}`). Every training Cell_ID appears once. Test Cell_IDs are excluded. Later runs **load** this file; they do not regenerate folds.

For each training cell, the recorded prediction comes from a model fit on the other two folds only. Concatenating the three validation folds yields 5000 OOF predictions. OOF accuracy is overall accuracy on that vector.

Primary selection metric: overall accuracy. Macro-F1 and per-class recall are diagnostics only.

## 4. Experiment Evolution

Evidence: `experiments/registry.csv`, `outputs/metrics/YW-00*_metrics.json`, `reports/sprint2_comparison.md`, `reports/sprint3_comparison.md`.

| Run | Method | Fold 0 | Fold 1 | Fold 2 | OOF accuracy | Macro-F1 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| YW-000 | Per-training-fold majority class (no features) | 0.1404 | 0.1404 | 0.1411 | 0.1406 | 0.0041 | Sanity floor. Majority class each fold: `oligodendrocyte_progenitor_2`. |
| YW-001 | `log1p` of 200 raw gene counts + multinomial L2 logistic regression | 0.5429 | 0.5579 | 0.5492 | 0.5500 | 0.3894 | Gene-only baseline. No metadata, coordinates, spatial features, PCA, or class weights. |
| YW-002 | YW-001 genes + fold-safe one-hot `Region` / `Excitatory_vs_Inhibitory` / `Segment` | 0.7540 | 0.7570 | 0.7593 | 0.7568 | 0.6799 | Metadata one-hot is the large jump vs genes (+0.2068 OOF; net +1034 cells vs YW-001). |
| YW-003 | YW-001 gene LR + fold-safe signature candidate masking | 0.7510 | 0.7510 | 0.7569 | 0.7530 | 0.7130 | Masking uses metadata only to restrict classes. Net +1015 vs YW-001; 0 cells that YW-001 got right were broken. |
| **YW-004** | **Per-signature gene-only L2 logistic specialists + masked global fallback** | **0.7594** | **0.7564** | **0.7635** | **0.7598** | **0.7053** | **Selected MODEL V1 architecture.** +0.2098 vs YW-001 (net +1049). Highest eligible OOF. |

YW-000 and YW-001 are Sprint 1 baselines. YW-002/003/004 are Sprint 2 experiments, not separate model versions. The selected architecture is YW-004.

Sprint 3 did not replace YW-004 (see §6).

## 5. MODEL V1 Architecture

MODEL V1 is the YW-004 architecture fit on **all 5000 labeled training cells**, then applied to the official test IDs.

Signature: `(Region, Excitatory_vs_Inhibitory, Segment)`. Missing values are the token `__MISSING__`.

Routing:

1. **Single-class signature** → deterministic one-hot prediction of that class.
2. **Ambiguous signature** (≥2 classes in training) → specialist multinomial L2 logistic regression on `log1p` of the 200 gene counts for cells with that signature, then mask-and-renormalize to the signature's candidate classes.
3. **Unseen signature or specialist fit failure** → global gene-only logistic regression fallback (unmasked if unseen; masked to candidates if the specialist failed).

Hyperparameters (frozen `LR_KWARGS`, same as YW-004):

```text
penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
class_weight=None, random_state=20260819
```

Specialists do not take metadata as features. Metadata is used only for routing and candidate sets.

### Full-train routing (`outputs/metrics/model_v1_metrics.json`)

| Quantity | Count |
|---|---:|
| Training cells | 5000 |
| Test cells | 5000 |
| Full-train signatures | 28 |
| Single-class (deterministic) signatures | 15 |
| Ambiguous signatures / specialists trained | 13 |
| Specialist fit failures | 0 |
| Test cells routed deterministically | 797 |
| Test cells routed to specialists | 4203 |
| Test fallbacks | 0 |

The largest signature is `__MISSING__/__MISSING__/__MISSING__` (2958 training cells, 16 glial / vascular / non-neuronal classes) and is an ambiguous specialist.

OOF routing under the 3-fold protocol (`outputs/metrics/YW-004_metrics.json`): deterministic signatures 842 cells at accuracy 1.0; ambiguous signatures 4158 cells at accuracy 0.7112; fallbacks 0.

## 6. Why YW-004 Was Selected

YW-004 was selected because it had the highest **eligible** 3-fold OOF accuracy among leakage-safe experiments: **0.7598**.

Relative to the gene-only baseline YW-001 (0.5500):

- all three folds improved
- net +1049 correct cells
- OOF +0.2098

Relative to other Sprint 2 models:

- YW-002 (genes + metadata one-hot): 0.7568 (−0.0030 vs YW-004)
- YW-003 (genes + candidate masking): 0.7530 (−0.0068 vs YW-004)
- YW-003 has higher macro-F1 (0.7130 vs 0.7053) but lower official-metric accuracy, so it was not selected

Sprint 3 did not replace YW-004:

| Run | Role | OOF | Decision |
|---|---|---:|---|
| YW-005 | Exploratory convex blend of saved YW-002/003/004 OOF probabilities | 0.7592 (cross-fit); equal-weight 0.7598 | **Excluded.** Not nested: weight selection for fold *f* used other-fold OOF that had been trained with fold *f* visible. |
| YW-006 | Inner-CV specialist search inside the missing-metadata bucket only | bucket 0.6815 | **Negative ablation.** Selected `log1p_lr` on every outer fold; identical to the YW-004 bucket specialist. |
| YW-007 | Hybrid: YW-004 outside the bucket, YW-006 inside | 0.7598 | **Not selected.** Exact duplicate of YW-004 (`n_changed=0`). |

No statistical-significance test is recorded. Selection is the recorded OOF comparison, not a p-value.

## 7. Error Analysis

YW-004 remaining errors: **1201 / 5000**.

| Slice | n | Errors | Accuracy |
|---|---:|---:|---:|
| Missing-metadata bucket `__MISSING__/__MISSING__/__MISSING__` | 2958 | 942 | 0.6815 |
| Outside that bucket | 2042 | 259 | — |
| Deterministic signatures (OOF) | 842 | 0 | 1.0000 |
| Ambiguous signatures (OOF) | 4158 | — | 0.7112 |

Most remaining errors are in the large metadata-missing regime (942 / 1201). That bucket contains 16 glial / vascular / non-neuronal types.

Largest OOF confusion pairs (`outputs/metrics/YW-004_metrics.json`, `reports/sprint3_comparison.md`):

| True | Predicted | n |
|---|---|---:|
| oligodendrocyte_1 | oligodendrocyte_progenitor_2 | 111 |
| oligodendrocyte_progenitor_2 | oligodendrocyte_1 | 64 |
| oligodendrocyte_progenitor_2 | oligodendrocyte_2 | 56 |
| oligodendrocyte_2 | oligodendrocyte_progenitor_2 | 50 |
| astrocyte_2 | astrocyte_1 | 42 |

The residual problem is overlapping glial subtypes when Region / E-I / Segment are missing, not the deterministic neuronal signatures.

## 8. Leakage and Integrity Controls

- Test target is never used. `meta_test.MERFISH_cell_type_annotation` is fully hidden. MODEL V1 sources do not read `y_test` or write `prediction/prediction.csv`.
- Cell_IDs are 19-digit strings (`load_dataset()` / `dtype=str`). They are not cast through float64.
- Train and test Cell_IDs are disjoint (data contract).
- Official CSVs are hashed in `experiments/official_data_manifest.json` and verified before V1 training (`scripts/10_official_manifest.py --verify`).
- During OOF, signature → candidate maps are built from the training fold only. Full-train MODEL V1 maps use the 5000 labeled training cells only.
- Submission candidate CSV must match official test Cell_ID order and the 60 allowed labels (`merfish60.validate_submission`).
- `tests/test_model_v1.py` checks: all 5000 train IDs used; no test IDs in fitting; gene-only specialists; hyperparameters match YW-004; 5000-row submission; label set; probability row sums; `prediction/prediction.csv` unmodified; metrics record OOF selection and do **not** record test/leaderboard accuracy.

These tests are part of the frozen V1 implementation. This document does not invent a historical pytest pass count beyond what the repository records.

## 9. Submission Artifact

Submission-ready MODEL V1 candidate (not an official leaderboard file):

| Artifact | Path |
|---|---|
| Submission candidate | `outputs/submissions/model_v1.csv` |
| Test probabilities (60-class order from `allowed_labels()`) | `outputs/probabilities/model_v1_test_probabilities.csv.gz` |
| Run metrics | `outputs/metrics/model_v1_metrics.json` |
| Class order | `outputs/metrics/model_v1_class_order.json` |
| Signature routing table | `outputs/metrics/model_v1_signature_summary.csv` |

Do not copy this file to `prediction/prediction.csv` as part of MODEL V1. Official team submission is a separate captain-repository action and was not performed for this MODEL V1 candidate.

## 10. Reproduction

Requires the project virtualenv (`pandas`, `numpy`, `scikit-learn`, `scipy`, `pytest` as pinned in `requirements.txt` at the V1 freeze). Official CSVs must match the saved manifest.

```sh
# verify official inputs
.venv/bin/python scripts/10_official_manifest.py --verify
.venv/bin/python scripts/00_validate_data.py

# frozen 3-fold YW-004 OOF (does not regenerate experiments/folds.csv)
.venv/bin/python scripts/02_sprint2_experiments.py

# full-train MODEL V1 candidate (refuses to overwrite unless passed)
.venv/bin/python scripts/06_model_v1.py
# existing artifacts: .venv/bin/python scripts/06_model_v1.py --overwrite

# submission contract
.venv/bin/python scripts/90_validate_submission.py outputs/submissions/model_v1.csv

# V1 tests
.venv/bin/python -m pytest tests/test_model_v1.py tests/test_folds.py tests/test_official_contract.py tests/test_submission_contract.py -q
```

Expected selection number after a faithful OOF rerun: **0.7598**. MODEL V1 was not submitted for official scoring, so no official hidden-test or leaderboard score is available.

## 11. Repository Artifacts

| Location | Role |
|---|---|
| `src/merfish60/model_v1.py` | Frozen full-train YW-004 implementation |
| `src/merfish60/signatures.py` | Signature keys, candidate maps, mask-and-renormalize |
| `src/merfish60/models.py` | Shared `LR_KWARGS` and `log1p` helpers |
| `src/merfish60/validate_submission.py` | Official CSV contract |
| `scripts/06_model_v1.py` | Train / predict / write submission candidate |
| `scripts/02_sprint2_experiments.py` | YW-002/003/004 OOF |
| `scripts/01_baseline.py` | YW-000/001 OOF |
| `scripts/04_sprint3_experiments.py` | YW-005/006/007 (not selected) |
| `scripts/90_validate_submission.py` | CLI validator |
| `scripts/10_official_manifest.py` | Manifest verify |
| `tests/test_model_v1.py` | V1 leakage and contract tests |
| `experiments/folds.csv` | Frozen 3-fold assignments |
| `experiments/registry.csv` | Experiment log (YW-000..YW-007) |
| `experiments/official_data_manifest.json` | Official CSV hashes |
| `outputs/metrics/` | Per-run JSON, including `model_v1_metrics.json` |
| `outputs/oof/` | OOF prediction tables |
| `outputs/probabilities/` | OOF and test probability matrices |
| `outputs/submissions/model_v1.csv` | Submission candidate |
| `reports/sprint2_comparison.md` | Sprint 2 comparison |
| `reports/sprint3_comparison.md` | Sprint 3 comparison |
| `docs/validation_protocol.md` | 3-fold protocol note |

## 12. Version / Provenance

Recorded git history in the development pipeline:

| Record | SHA | Message |
|---|---|---|
| Validation pipeline | `6ca7c4b` | chore: establish reproducible MERFISH validation pipeline |
| Sprint 2 hierarchy | `ec8f87b` | feat: add fold-safe metadata hierarchy pipeline |
| Sprint 2 registry | `62d436d` | exp: record reproducible benchmarks through 75.98% OOF |
| Sprint 3 experiments | `299225a` | feat: add hard-bucket benchmarking and ensemble diagnostics |
| Sprint 3 decision | `4ae6aab` | exp: document no-gain Sprint 3 ablations at 75.98% OOF |
| MODEL V1 implementation | `a4aa972` | feat: add full-train YW-004 Model V1 pipeline |
| MODEL V1 release commit (existing) | `a3722c0` | release: create submission-ready Model V1 |

Local tag `model-v1` currently points at `a3722c0`. A later documentation commit will follow this SHA, so **do not treat `a3722c0` as the final tagged documentation SHA**.

`outputs/metrics/model_v1_metrics.json` records `current_git_commit` = `a4aa972` (the implementation commit at artifact write time), `selected_from_run` = `YW-004`, `selected_oof_accuracy` = `0.7598`.

The tag had not been pushed at the time this document was written.

## 13. Limitations and Next Step

Gene/signature specialists still struggle in the hard glial / metadata-missing regime: 2958 of 5000 training cells have no Region / E-I / Segment, OOF accuracy there is 0.6815, and 942 of 1201 errors fall in that bucket. Dominant confusions are oligodendrocyte ↔ progenitor pairs and astrocyte_2 → astrocyte_1.

Spatial structure and external-reference modeling belong to **MODEL V2 development** and are **not** part of MODEL V1.
