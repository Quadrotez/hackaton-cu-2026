# Frozen bulk-volume A/B

Run from the repo root: `.venv/bin/python -B work/sprint3_bulk_expand/reproduce.py`.
The script writes three OOF prediction JSONs and `report.json` next to itself.

The two heads use the same weighted log-volume ridge on metric projected area
and relative height, estimated against a robust local inverse-depth fit in a
ring around the mask. The frozen volume gets 75% weight in log space; the final
volume ratio is capped to `[1/1.35, 1.35]`. The only arm difference is the
train-only erosion/dilation of real GT masks. An `object_id` contributes total
weight 1 in both arms, distributed equally across its valid clean annotations
before splitting each annotation's weight between the original and its variants.
No synthetic or test data are used.

The existing outer manifest is preserved, with no overlap in training and
validation object IDs, frames, or inferred positive capture bursts (the data
have no explicit scene ID; this is not session-held-out validation). Geometry
at evaluation comes solely from verified frozen detector OOF masks, including
the actual l960 masks for added hypotheses. The baseline export is byte-identical
to its input. Only `volume` on predicted `bulk` rows may change.

| Slice | Exact baseline | Real-only Δ vs baseline | Expanded Δ vs baseline | Expanded Δ vs real-only |
|---|---:|---:|---:|---:|
| Pooled | 72.780196 | +0.012861 | +0.062936 | +0.050075 |
| Fold 1 | 76.688814 | -0.900948 | -0.857092 | +0.043856 |
| Fold 2 | 68.330873 | +0.485655 | +0.498134 | +0.012479 |
| Fold 3 | 73.241578 | -0.250871 | -0.240342 | +0.010529 |
| Fold 4 | 76.689586 | +0.993118 | +1.235700 | +0.242582 |
| Fold 5 | 74.799211 | -0.148253 | -0.148779 | -0.000526 |

Pooled volume credit `V`: **0.637222 → 0.637736 → 0.639739**.
Against GT bulk only (including misses): **0.470030 → 0.470447 → 0.473988**.
GT-solid credit can move because official volume matching ignores predicted
form; all predicted-solid records stayed identical. `D`, `M`, and `S` are
unchanged exactly, pooled and in every fold. The expansion increment is small
and concentrated in fold 4. Expanded beats the frozen baseline on only two
folds; not enough evidence to replace the anchor.
