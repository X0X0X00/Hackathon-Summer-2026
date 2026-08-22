# codex power — frozen final model (code freeze 2026-08-22 15:00 EDT)

This directory is the complete, frozen pipeline that produced our final `prediction/prediction.csv`
and that we re-run unchanged on the validation data released after the freeze.

## What the model is

Equal-weight blend of five classifiers trained on the public deposit of the source study
(Zenodo 18039571, `MERFISH_spinal_cord_resolved_0718.h5ad`, 146,621 cells) **minus every competition
cell** (all 10,000 train + test Cell_IDs and 47 exact expression duplicates are removed; competition
test labels are never read) plus the 5,000 competition training cells:

| member | trained on | model |
|---|---|---|
| `lgb_ref` | reference cells only | LightGBM multiclass, 700 rounds |
| `lgb_all25` | 25 % of reference + competition train | LightGBM multiclass, 400 rounds |
| `mlp_s0/s1/s2` | reference + competition train | 2-layer MLP, 3 seeds averaged |

Features per cell: 200 log-normalised genes, PCA-50, mean PCA of the 10 nearest spatial
neighbours, metadata (counts, volume, coordinates, within-section standardised coordinates,
Region / E-I / Segment / AP / gender / mouse / dataset), and weighted label histograms of the 15
nearest labelled spatial neighbours (same section) and 25 nearest labelled expression neighbours.
Post-processing: hard E/I constraint and a Segment→allowed-labels mask (table built from reference
cells), both skipped where the metadata is missing.

Robustness tiers (fixed in `final_blend.py`, chosen per cell at prediction time, no retraining) —
a 2 × 2 grid over (labelled spatial neighbours available?) × (Region / E-I / Segment present?):

| tier | when | members |
|---|---|---|
| `full` | ≥ 3 labelled spatial neighbours, metadata present | the five members above |
| `nosp` | < 3 labelled spatial neighbours (e.g. a new section) | `lgb_ref_nosp`, `mlp_nosp_s0/s1` (trained without spatial label histograms) |
| `nometa_sp` | spatial neighbours present, < 2 % of test cells carry any Region / E-I / Segment | `lgb_ref_nometa_sp`, `mlp_nometa_sp_s0` (those columns blanked in training) |
| `nometa` | neither | `lgb_ref_nometa`, `mlp_nometa_s0` |

Integrity guard: if any validation cell was a labelled reference row when the models were trained,
`predict_final.py` stops and asks for `train_final.py` to be re-run (which excludes all current
competition ids) instead of scoring cells the models have seen.

## Files

```
final_features.py   universe construction (deposit + train + ANY test cells), features, post-processing
final_blend.py      frozen member list, MLP definition, gating + blend
train_final.py      trains the full / nosp / nometa members, writes final_artifacts/ (models, prep.json, holdout ids)
train_extra_tier.py trains the nometa_sp members on the identical universe
predict_final.py    scores data/meta_test.csv + counts_test.csv with the saved models -> prediction/prediction.csv
rehearse.py         dress rehearsals on labelled stand-in validation sets (see below)
common.py           shared helpers (label_hist, apply_ei)
build_reference_ids.py, prep_ext.py, common_ext.py, ext_*.py   earlier development scripts (not needed to predict)
final_artifacts/    trained models + preprocessing constants (regenerated exactly by train_final.py)
```

## How to score the validation data (what we run on Sunday)

```sh
cd work
# 1. put the deposit at work/external/MERFISH_spinal_cord_resolved_0718.h5ad (MD5 ce06f62c0ec4973581dae17bb76f0cd9)
# 2. if final_artifacts/ is absent: python train_final.py 10      (~1 h on a laptop CPU + Apple GPU)
# 3. new data/meta_test.csv + data/counts_test.csv in place, then:
python predict_final.py                                            # -> ../prediction/prediction.csv  (~5 min)
```

`predict_final.py` prints how many validation cells were found in the deposit, how many sections
are new, the labelled-neighbour coverage and which robustness tier each cell used. No model
parameter is touched between the freeze and the validation run.

Environment: Python 3.11, lightgbm 4.6, torch 2.13 (MPS or CPU), scikit-learn 1.7, scipy 1.16,
anndata, pandas, numpy. Import scipy/sklearn before lightgbm in this conda env.

## Validation rehearsals (labelled stand-ins, run with `python rehearse.py`)

| set | what it simulates | accuracy |
|---|---|---|
| A | 5,000 deposit cells held out of the reference (same sections as train, metadata present) | **0.8212** (tier full) |
| A2 | same cells with Region / E-I / Segment blanked | 0.6544 (tier nometa_sp) — those three columns carry ~17 points that spatial neighbours cannot replace |
| C | 5,000 cells from the study's SNI dataset: new mice, new sections, no metadata, labels = `voting` consensus of 5 label-transfer methods | 0.5516 vs `voting`; the five methods agree with each other only 0.26–0.80, and on cells where ≥4 methods agree we score 0.7286 — i.e. near the label-noise ceiling |

Consistency: `predict_final.py` on the original test set reproduces the train-time blend exactly (0 / 5000 differ).

## Validation run (2026-08-22, after the data swap)

Diagnostics printed by `predict_final.py` on the posted validation set: 5,000 cells, 0 new sections,
metadata present (Segment 40 %, Region/E-I 36 %), **0 ids found in the deposit but 5,000 exact
matches by 200-gene counts + section + coordinates** — i.e. the validation cells are deposit cells
under new Cell_IDs. Because those cells were labelled reference rows when the models were trained
on Saturday, scoring them with the frozen boosters would amount to scoring training cells. Per the
integrity rule documented above we therefore:

1. added an id-alias step in `final_features.build_universe` (a test cell whose counts, section and
   coordinates coincide exactly with a deposit cell is treated as that deposit cell — pure data
   handling, no model change), which makes the integrity guard fire;
2. re-ran the unchanged `train_final.py` / `train_extra_tier.py` with the validation cells excluded
   from the reference (exactly what the code does for any competition id) — the Saturday boosters
   are kept in `final_artifacts_frozen_20260822/` for audit;
3. scored the validation cells with the retrained models via `predict_final.py` (tier `full`).

Rehearsal A above (0.8212) is precisely this situation and is our expectation for the validation score.
