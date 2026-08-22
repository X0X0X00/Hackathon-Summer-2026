# WYH work directory

This directory is the active WYH research and reproduction tree. Official
competition inputs stay at the repository root in `data/`. The current official
team prediction stays at `prediction/prediction.csv`. That file is **not** the
MODEL V2 candidate (`experiments/evidence/submissions/model_v2_candidate.csv`).

## Directory map

```text
work/
├── src/merfish60/     # active modeling library
├── scripts/           # reproduction and validation entry points
├── experiments/       # folds, manifests, V3 experiment code, retained evidence
├── docs/              # model documentation and contribution record
├── reports/           # experiment reports and decision audits
├── tests/             # contract, leakage, and integrity tests
├── requirements.txt   # pinned dependencies
├── external/          # local-only raw reference data (not committed)
├── outputs/           # generated rerun artifacts (not committed)
└── legacy/            # superseded early local pipeline (not active)
```

## What is current

**Active implementation:** `work/src/merfish60/` plus `work/scripts/`.

**Selected research model:** MODEL V2 / V2-B-REFONLY, documented in
[`docs/versions/model_v2.md`](docs/versions/model_v2.md).

**Final reproduction entry point:**

```bash
python work/scripts/09_v2b_refonly.py
```

This reconstructs MODEL V2 validation artifacts under `work/outputs/`. It does
not write `prediction/prediction.csv`.

**Official prediction validator:**

```bash
python work/scripts/90_validate_submission.py prediction/prediction.csv
```

**Tests:**

```bash
python -m pytest -q work/tests
```

Some tests skip when regenerable probability dumps or the local `.h5ad`
reference are absent. That is expected after delivery cleanup.

## Experiments and evidence

- `work/experiments/folds.csv` — frozen personal 3-fold protocol used by MODEL V1
- `work/experiments/team_folds_5_seed42.csv` — team-compatible 5-fold protocol used by MODEL V2
- `work/experiments/official_data_manifest.json` — official CSV hashes
- `work/experiments/v3/` — V3 research-program scripts (E00T–E07D)
- `work/experiments/evidence/` — compact retained metrics, candidate CSVs, and V3 summaries

Large probability arrays, OOF dumps, and parquet registries were removed from
the delivery tree because they are regenerable. Compact JSON/CSV evidence that
supports the reports was retained.

## External data

Place approved local files here. Do not commit them.

| File | MD5 | Used by |
| --- | --- | --- |
| `work/external/MERFISH_spinal_cord_resolved_0718.h5ad` | `ce06f62c0ec4973581dae17bb76f0cd9` | MODEL V2 / V3 MERFISH experiments |
| `work/external/SNI_merged_0917.h5ad` | `7e90a801ee57b8fec06cd03c8630f01b` | V3-E04S / E06M only |

The MERFISH file is the approved Zenodo record **18039571** deposit. This
repository records the record id, filename, local path, and MD5, but not a
download URL.

Caches created from these files live under `work/external/cache/` and are
gitignored.

## Legacy material

[`legacy/`](legacy/) is the earlier local bagging pipeline that predates the
`merfish60` track. It is retained only as historical contribution evidence. It
is **not** the active implementation and must not be used to overwrite
`prediction/prediction.csv`.
