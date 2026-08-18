# Hierarchical glial specialist model

## Result

The selected pipeline improves four-fold out-of-fold accuracy from **0.7708** to **0.7910** and macro-F1 from **0.7247** to **0.7450**. This is 101 additional correct cells out of 5,000 training cells.

The final validated submission is `prediction_hierarchical_glia.csv`.

## Model design

1. Start from the 0.7708 multistructure LightGBM/logistic ensemble with fold-safe Region and Segment priors.
2. Train a coarse glial router for oligodendrocyte-lineage, astrocyte, vascular/meningeal, immune, peripheral-glia, and ependymal families.
3. Train separate within-family specialists for:
   - five oligodendrocyte-lineage labels;
   - two astrocyte labels;
   - endothelial, pericyte, and three meningeal labels;
   - peripheral glia and Schwann cells.
4. Blend LightGBM and logistic internal specialists with an external-expression specialist.
5. Apply corrections only when the total family probability passes a learned gate. High-confidence non-glial predictions are left unchanged.

## External reference and leakage controls

The public reference is the full mouse spinal-cord MERFISH dataset deposited at <https://doi.org/10.5281/zenodo.18039571>.

- Source file MD5: `ce06f62c0ec4973581dae17bb76f0cd9`.
- The source contains 146,621 cells, 500 genes, and the same 60-label taxonomy.
- All 10,000 competition train/test Cell_IDs were removed before model fitting.
- An additional 35 remaining rows with a 200-gene expression fingerprint matching a competition cell were removed conservatively.
- The reference was capped at 2,500 cells per target class.
- Only the 200 competition genes were retained.
- No competition test label was queried or used.

The processed external reference contains 27,112 non-competition glial cells. Raw reference data and processed reference arrays must not be committed.

## OOF contribution by stage

| Stage | Additional correct cells | Fold deltas |
|---|---:|---|
| Coarse glial router | +44 | +12, +18, +6, +8 |
| Oligodendrocyte specialist | +21 | +1, +12, +8, 0 |
| Astrocyte specialist | +21 | +3, +12, +6, 0 |
| Vascular/meningeal specialist | +15 | +4, +7, +2, +2 |
| Peripheral-glia specialist | 0; skipped | 0, 0, 0, 0 |

All selected corrections have non-negative fold-level deltas. The peripheral-glia correction was automatically rejected because it did not add at least two correct cells after the preceding stages.

## Reproduction

1. Download `MERFISH_spinal_cord_resolved_0718.h5ad` from the DOI above and place it beside the scripts as `external_MERFISH_spinal_cord.h5ad`.
2. Run `prepare_external_reference.py` to verify the MD5, remove all competition IDs and fingerprints, align the 200 genes, and create `external_glia_reference.npz`.
3. Run `hierarchical_glia.py` after the baseline probability artifacts have been generated.
4. Validate `artifacts/prediction_hierarchical_glia.csv` against the official sample submission.

## Competition caution

This method uses a public dataset from the same underlying study. Even though exact competition cells are excluded, the team should obtain written organizer confirmation that external reference training is allowed before submitting this version. If external data is not permitted, use the internal-only hierarchical version, which reached 0.7784 OOF accuracy.
