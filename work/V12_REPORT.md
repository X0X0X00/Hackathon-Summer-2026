# V12 Historical-Version Router

## Result

V12 keeps V9 as the default prediction and uses the historical model library
only for cross-fold-stable corrections.

| Candidate | OOF correct | OOF accuracy | Macro-F1 | Log loss |
|---|---:|---:|---:|---:|
| V9 anchor | 4,169 / 5,000 | 0.8338 | 0.8037 | 0.50148 |
| Stable-rule nested audit | 4,174 / 5,000 | 0.8348 | 0.8049 | 0.50085 |
| V12 final refit | 4,178 / 5,000 | 0.8356 | 0.8051 | 0.50070 |

The strict nested audit learns rules on four folds and applies them only to the
untouched fifth fold. Its fold deltas over V9 are `[+1,+1,+1,+1,+1]`. The final
all-OOF refit has fold deltas `[+5,+1,+1,+1,+1]`. The final number is therefore
the more optimistic diagnostic; `0.8348` is the better estimate of the routing
idea's repeatability.

The official hidden-test accuracy is unknown until the leaderboard evaluates
the CSV. The last confirmed official score remains V8.5's `0.8148`.

## Version audit

Two historical naming systems exist in this repository. The commit sequence
uses V1-V9, while `v1/` is a later reproducible 80.68% specialist package. They
must not be treated as the same model.

| Version | Main idea | Auditable local OOF |
|---|---|---:|
| Commit V1 | First ensemble | 0.773 (commit message) |
| V2 | Bagged LightGBM plus specialist correction | No preserved standalone OOF probability artifact |
| V3 | 27-model LightGBM bag plus specialist | No preserved standalone OOF probability artifact |
| V4 | 47-model LightGBM bag plus specialist | No preserved standalone OOF probability artifact |
| V5 | Blend with reference-trained members | No preserved standalone OOF probability artifact |
| V6 | Diverse five-member probability ensemble | 0.8250 |
| V7 | Hierarchical family specialists and routers | 0.8320 |
| V8 | Robustness-screened specialist correction | 0.8300 |
| V8.5 | Conservative pair/family correction | 0.8328 |
| V9 | Guarded complementary-model fusion | 0.8338 |

The archived `v1/` specialist package scores `0.8068` and is retained as a
diverse historical expert. Its lower global accuracy does not prevent it from
being useful on one narrow, repeatedly validated confusion pair.

## What was rejected

- Direct averaging of older models: the alternatives are weaker and highly
  correlated, so global blending generally lowers accuracy.
- A broad rule sweep: it appeared to reach 0.8396 on the same OOF rows used to
  choose rules, but lost three cells in untouched-fold evaluation. This was a
  selection-bias artifact and was rejected.
- Family-prior, domain-prior, glial-specialist, SSL-autoencoder, and direct
  section-GroupCV replacements: the existing V10/V11 audits did not show a
  stable global gain.

## Accepted routing rules

1. When V9 predicts `oligodendrocyte_progenitor_2`, the classwise V8.5 expert
   predicts `oligodendrocyte_1`, and the V9 top-two margin is at most 0.20,
   route to the expert. Full OOF gain: 6 correct cells; no fold loses.
2. When V9 predicts `alpha_motoneuron`, the archived specialist predicts
   `gamma_motoneuron`, and the V9 top-two margin is at most 0.40, route to the
   specialist. Full OOF gain: 3 correct cells; no fold loses.

Together these rules change 29 test predictions relative to V9:

- 28 `oligodendrocyte_progenitor_2 -> oligodendrocyte_1`
- 1 `alpha_motoneuron -> gamma_motoneuron`

## Files

- `build_v12_version_library_fusion.py`: reproducible loader, alignment,
  nested rule discovery, final refit, and submission writer.
- `v12_version_library_report.json`: full inventory, fold audit, selected rules,
  and changed test rows.
- `v12_final_probs.npz`: V12 OOF and test probabilities.
- `../prediction/prediction_v12_version_router.csv`: competition-format candidate.

The CSV contains 5,000 unique IDs, two required columns, no missing values, and
has SHA-256 `1093970edb3b8548937d1658933be1f03768d2b7cbd1648b2c0b7faaa6d931fc`.
