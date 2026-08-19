# MERFISH cell-type prediction — pipeline notes

Local-only pipeline for the Rochester Biomedical DS Summer 2026 hackathon (60-class
cell-type prediction on MERFISH spinal-cord cells; metric = accuracy on 5000 test cells).
Everything runs on CPU in a few minutes per model; no GPU needed.

Python env: `~/miniconda3/envs/ML/bin/python` (lightgbm 4.6, sklearn 1.7, scipy, pandas,
catboost 1.2, xgboost 3.0).

## Reproduce the submission

```sh
cd work
python prep.py                          # -> cache/static.npz, cache/nbrs.npz  (~1 min)
python experiments/exp_bag.py           # 8 LGBM runs      -> experiments/bag_runs/s*_ex*.npz, oof/bag8mix.npz
python experiments/exp_bigbag.py        # 12 jittered runs -> experiments/bag_runs/r3_s*.npz
python experiments/exp_folds.py         # 7 alt-fold runs  -> experiments/fold_runs/*.npz
python experiments/exp_bag_r4.py 8      # 20 more runs     -> experiments/bag_runs/r4_s*.npz, oof/poolAll.npz
# pool = mean of every run in bag_runs/ + fold_runs/  (v3 used the first 27 = oof/poolB27.npz)
python -c "import numpy as np,shutil; z=np.load('oof/poolAll.npz'); np.savez_compressed('final_probs.npz', oof=z['oof'], test=z['test'])"
python experiments/exp_spec.py run      # glial-pair specialist post-correction -> experiments/specialist_correction.npz
python make_submission.py experiments/specialist_correction.npz   # -> ../prediction/prediction.csv
```

`make_submission.py` applies the E/I hard constraint, argmaxes, and writes the CSV in the
exact template format (row order = `data/meta_test.csv`; header
`Cell_ID,MERFISH_cell_type_annotation.y`; asserts every label is one of the 60).

## v5+ : external reference (Zenodo 18039571) — what changed on 2026-08-18

The competition data is a 10k-cell / 200-gene subset of the study's public deposit
(Wang … Meltzer, bioRxiv 2026.01.10.698734; Zenodo 18039571,
`MERFISH_spinal_cord_resolved_0718.h5ad`, 146,621 cells × 500 genes, same 60 labels,
MD5 `ce06f62c0ec4973581dae17bb76f0cd9`). Team decision (2026-08-18): organizers permit it.

**Rules we enforce in code (`prep_ext.py`, `external/reference_ids.npy`):**
- Reference = the deposit MINUS all 10,000 competition Cell_IDs MINUS 47 rows whose 200-gene
  count vector exactly equals a competition cell → **136,574 reference cells**.
- Competition rows are kept in the universe only for kNN structure; their label slot is set to
  -1 before anything else, then competition-TRAIN labels are filled from `meta_train.csv`.
  Competition TEST labels are never read into any array (asserted).
- Alignment verified on TRAIN cells only: label / counts / coordinates / section 100 % identical.

**Pipeline** (`prep_ext.py` → `cache_ext/`, `common_ext.py` harness, ~4 min build):
same feature layout as `prep.py`, but spatial kNN and rel-coords over the FULL sections and
neighbour-label histograms from reference + visible train labels (density 50 % → 96 %).
Fold protocol unchanged: predicting competition fold f hides fold-f labels everywhere; reference
labels are always visible. Reference cells are indistinguishable from competition cells
(adversarial AUC 0.50 once `Segment`, absent from the deposit, is dropped).

**The key discovery (2026-08-18 evening): competition `Segment` == deposit `Laminae`.**
`Segment` (1–22, observed for 40 % = the neurons) is a 1:1 recoding of the deposit's `Laminae`
column, and Laminae is a per-*subtype* annotation — every label maps to essentially one Segment
(purity ≥ 0.96) and 9 Segment values map to a single label. It is by far the strongest neuron
feature (fold-0 neuron acc 0.89 with it vs 0.64 without). The deposit lacks a `Segment` column,
so reference-trained models were crippled (0.73–0.78) and transferred badly both ways (2×2 test in
`experiments/ext_2x2.py`). `prep_ext.py` now fills `Segment` for reference rows from Laminae, and
`ext_post.py` adds a Segment→allowed-labels **hard mask** built from reference labels only
(0 true-label violations on train; +0.5 pt on any blend). Region mask adds nothing.

| member (`oof_ext/`, Segment-enabled cache) | how | OOF+EI | +Seg mask | folds 3–4 |
|---|---|---|---|---|
| refonly_full | LGBM on 136k reference only, 700 rounds (`ext_refonly.py`), eval on all 5000 train | 0.8188 | 0.8188 | 0.8130 |
| ext_all25 | LGBM on 25 % reference + comp-train (`ext_e1.py all 8 ext_all25 0.25`) | 0.8120 | 0.8120 | 0.8020 |
| mlp3 | torch/MPS MLP on 100 % reference + comp-train, mean of 3 seeds (`ext_mlp.py`) | 0.8096 | 0.8134 | 0.8035 |
| yhh_v1 | teammate's Codex pipeline reproduced in `../Hackathon_26_yhh` (OOF-selected → optimistic) | 0.8070 | 0.8074 | 0.7950 |
| poolAll | v4 competition-only 47-run pool | 0.7712 | 0.7720 | 0.7700 |
| ext_comp | LGBM on comp-train with dense nbr labels — dominated, not used | 0.7756 | 0.7764 | 0.7670 |

| blend (equal weights, no search) → EI → Segment mask | all OOF | folds 3–4 |
|---|---|---|
| v5 (poolAll + ext_comp + ext_all25 + mlp3, pre-Segment members) | 0.8106 | 0.8040 |
| refonly + mlp3 | 0.8222 | 0.8175 |
| **v6 = refonly + ext_all25 + mlp3 + yhh_v1 + poolAll** | **0.8248** | **0.8175** |

All strong-member blends sit at 0.822–0.825 (differences are noise); v6 is the "one member per
family, equal weight" choice. `ext_blend.py` aligns universe-order members with competition-order
ones by Cell_ID. Reproduce v6: `prep_ext.py` → the four member scripts → reproduce yhh_v1 per its
README (`v1/artifacts/align_oof.py` writes `oof_ext/yhh_v1.npz`) → build the 5-way mean into
`final_probs.npz` (see the snippet in git history of this commit) → `make_submission.py --ext`.

Negative results on the reference (do not repeat): early-stopping a reference-only LGBM on
multi_logloss stops at ~85 rounds (over-confidence, accuracy still climbing) — use fixed rounds;
before the Segment fix, adding reference rows to one LGBM gave only +0.5 pt and blending was the
only gain — that was the missing-Segment artefact, not a property of the reference.

## Method (v1–v4, competition data only)

**Features** (`prep.py` -> `common.build_X`, ~450 cols):

- 200 log-normalised genes, PCA50 of them, mean PCA50 of the 15 spatial neighbours (`nbm*`),
  metadata (log total counts, log volume, density, absolute + within-section standardised
  coords, Region / E-I / Segment codes with missing kept as a code, AP position, gender,
  mouse, dataset).
- **Neighbour-label histograms** — the strongest block. For each cell, a 1/(1+d)-weighted
  histogram of the *visible* labels of its 15 nearest spatial neighbours (same section) and
  its 25 nearest expression neighbours (PCA50 euclidean, all sections), plus the number of
  labelled neighbours found. Test cells naturally get this from the labelled train cells
  around them (~50 % of all cells are labelled).

**Model**: LightGBM multiclass, 400 rounds, lr 0.06, 63 leaves, ff 0.7, bf 0.8, l2 1,
max_bin 127 — averaged over **27 (v3) → 47 (v4) full-CV runs** with different seeds, mild
hyper-parameter jitter, and different fold schemes. Test probs = mean over the runs' refit
models (each refit on all 5000 train cells with all train labels visible to the histograms).

**Post-processing**: (1) hard E/I constraint from observed metadata; (2) binary LGBM
specialists for the three most confused glial pairs (oligo_1↔OPC_2, oligo_2↔OPC_2,
astro_1↔endothelial) override the top-1 when the top-2 are exactly that pair and the margin is
below a threshold tuned on folds 0+1 (`experiments/exp_spec.py`).

## Leakage protocol (do not weaken)

Fixed 5-fold stratified folds live in `cache/static.npz`. When predicting fold *f*, the
labels of fold *f* are invisible **everywhere** — model training set, neighbour-label
histograms, pseudo-labels, prototypes, specialists. `common.run_cv` implements this; every
experiment either calls it or copies it verbatim. OOF numbers below are therefore honest
(slightly pessimistic: CV histograms see 40 % of cells labelled vs 50 % at test time).

## Scoreboard (5-fold OOF accuracy, +EI = after E/I constraint)

| what | OOF | +EI | verdict |
|---|---|---|---|
| majority class / LogReg on genes | 0.141 / 0.47 | | sanity floors |
| LGBM, static features only | ~0.60 | | |
| + neighbour-label histograms (single run) | 0.762–0.769 | | seed noise ±0.003; the 0.7692 "baseline" was a lucky seed |
| bag8mix (8 runs) | 0.7704 | 0.7716 | v2 base |
| poolB27 (27 runs) | 0.7734 | 0.7742 | v3 base — a lucky subset: random 27-of-47 subsets give 0.7712 ± 0.0009 (max 0.7732), and it contains 3 ten-fold runs whose OOF is inflated by 45 % label density |
| poolAll (47 runs) | 0.7700 | 0.7712 | **v4 base** (pre-registered "average everything"; test argmax differs from poolB27 on 32 cells) |
| + glial specialists | | 0.7746 | honest holdout gain ≈ 0 to +0.002 |

Ensemble oracle (any member right) is 0.783 — but no weighted/greedy/stacked ensemble
beats a plain average on held-out folds; the weighting just overfits the OOF. Pool-vs-pool OOF
differences of ±0.003 are measurement noise — do not pick pools by OOF.

Submissions pushed to `X0X0X00/Hackathon-Summer-2026` main: v1 d28495f (greedy ensemble),
v2 1d700f0 (bag8mix), v3 9404824 (poolB27), v4 0c9249c (poolAll) — all on 2026-08-17.

## What was tried and did NOT help (so nobody repeats it)

Screened on folds 0+1 (noise ±0.006 on the mean; fold-0 alone ±0.013), promoted only on full
5-fold CV. Full logs in `experiments/*.log`.

| direction | script | result |
|---|---|---|
| k / weighting / multi-k / two-hop of neighbour histograms | `exp_nbr.py` | sp15/ex25 is a local optimum; two-hop causes train/val shift |
| LGBM params (leaves, lr, rounds, ff, bf, l1/l2, dart, class weights) | `exp_lgbm.py` | saturated (~100 effective rounds); class weights hurt |
| self-training / pseudo-labels into histograms | `exp_selftrain.py` | wash (−0.0006) |
| new feature blocks (prototype corr, section composition, region priors, smoothed PCA, raw-count stats) | `exp_feats.py` | only a 5-col raw-count block ties; wide blocks dilute |
| better expression graphs (PCA100, cosine, correlation-200, SNN rerank) | `exp_graph.py` | SNN ties, rest worse |
| cross-section anatomical position matching (needs bilateral mirroring) | `exp_anat.py` | −0.002 |
| graph label propagation | `exp_prop.py` | 0.51 alone; label signal alone caps ~0.5 |
| CatBoost / XGBoost bags | `exp_catxgb.py` | 0.763 / 0.760; blending with the LGBM bag **hurts** on holdout |
| top-2 reranker (generalised specialist) | `exp_rerank.py` | ≤ 0 on holdout |
| HGB / MLP / LogReg / kNN as diversity members | `run_alt.py` | too weak to help (HGB needs l2 ≥ 1 or it diverges) |
| LGBM `extra_trees=True` as bag members | `exp_bag_r4.py` | single run 0.749 (−0.016) — dropped |
| batch-correcting the expression graph | (checked only) | expression-kNN same-mouse enrichment is 1.14×, same-dataset 1.21× — too weak to be worth correcting |
| greedy / Nelder-Mead / stacked ensembles of all members | `../ensemble.py` | +0.3 pt on OOF, −0.1 pt on holdout → overfit |

## Layout

```
work/
  prep.py            data -> cache/ (features, kNN graphs, folds)
  common.py          load / build_X / label_hist / run_cv / apply_ei / save_oof
  baseline.py        single LGBM run (reference)
  ensemble.py        (round-1 greedy ensemble; superseded, kept for reference)
  make_submission.py final probs -> ../prediction/prediction.csv
  final_probs.npz    current base test/oof probs
  cache/             static.npz, nbrs.npz   (never edit; delete + rerun prep.py to rebuild)
  oof/               one npz per member: oof (5000,60) train rows, test (5000,60), acc, acc_ei
  experiments/       one script per direction + logs; bag_runs/, fold_runs/ hold per-run probs
```
