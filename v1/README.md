# v1 — 80.68% OOF specialist pipeline

This directory archives the first reproducible version of our MERFISH cell-type pipeline that exceeded 80% local out-of-fold accuracy.

## Result

| Metric | Value |
|---|---:|
| Four-fold OOF accuracy | **0.8068** |
| Correct OOF predictions | **4,034 / 5,000** |
| OOF macro-F1 | **0.7864** |
| Gain over the 0.7932 candidate | **+68 correct** |
| Per-fold gain over 0.7932 | **+13 / +15 / +22 / +18** |

These are training-set OOF results, not test labels or a leaderboard score. The folds are stratified random folds; strict grouped CV by sample, mouse, or section remains a separate audit.

## Contents

```text
v1/
├── README.md
├── requirements.txt
├── run_pipeline.ps1
├── src/                  # complete training and validation chain
├── reports/              # OOF summaries, method reports, provenance
└── submission/
    └── prediction.csv    # archived v1 candidate; not the scored root path
```

Generated `cache/` and `artifacts/` directories are intentionally ignored by Git.

## Pipeline order

1. Global LightGBM and multinomial logistic baseline.
2. Multi-seed LightGBM ensemble and conservative metadata-prior correction.
3. External-reference hierarchical glial routing.
4. Oligodendrocyte pair specialists and section-constrained spatial kNN.
5. Dorsal-horn and motor-neuron family specialists.
6. Meningeal and peripheral-glia pair specialists.
7. Mid-ventral inhibitory family specialist.
8. Strictly gated residual neural pair specialists.

The final residual stage accepts a pair expert only when no fold loses correct predictions and at least two folds improve.

## Environment

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r v1\requirements.txt
```

## External reference

The external specialist path uses the public MERFISH spinal-cord data associated with [Zenodo record 18039571](https://doi.org/10.5281/zenodo.18039571).

Download the H5AD file separately and place it at:

```text
v1/cache/external_MERFISH_spinal_cord.h5ad
```

Expected MD5:

```text
ce06f62c0ec4973581dae17bb76f0cd9
```

The external file is not committed. Reference preparation removes all 10,000 competition Cell IDs and additional expression-fingerprint matches before fitting specialists. See `reports/*provenance.json` for details.

Before using this candidate as the official submission, confirm that the organizers permit this external reference.

## Reproduce

With the environment active and the external H5AD in `v1/cache/`:

```powershell
powershell -ExecutionPolicy Bypass -File v1\run_pipeline.ps1 -Python python
```

The script runs every stage in order, writes intermediate outputs to `v1/artifacts/`, updates `v1/submission/prediction.csv`, and validates the final CSV.

## Validate the archived candidate

Validation does not require the external H5AD:

```powershell
python v1\src\validate_submission.py v1\submission\prediction.csv
```

Expected output begins with:

```text
valid submission: rows=5000, unique_ids=5000
```

## Submission safety

The competition scores only the repository-root file at `prediction/prediction.csv`. The v1 candidate is archived under `v1/submission/` so publishing this folder does not silently replace the captain's current official submission.

## Detailed reports

- `reports/FINAL_80_REPORT.md`: complete optimization path and final validation.
- `reports/HIERARCHICAL_GLIA_REPORT.md`: hierarchical glial model.
- `reports/TARGETED_80_REPORT.md`: oligodendrocyte pair and spatial kNN stage.
- `reports/NEURONAL_EXPERTS_REPORT.md`: neuronal family experts.
- `reports/*summary.json`: machine-readable fold metrics and gates.
- `reports/*provenance.json`: external reference filtering records.
