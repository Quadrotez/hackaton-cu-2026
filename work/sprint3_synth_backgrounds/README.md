# Synthetic background donor filter

Run from the repo root: `.venv/bin/python -B work/sprint3_synth_backgrounds/reproduce.py`.
Requires NumPy and OpenCV. All outputs are written next to the script.

- `allowed_synth.json`: `{"1": ["s_...", ...], ..., "5": [...]}`. Each sorted list
  is a subset of that fold's existing `train_image_ids`. It excludes mixed-object
  frames and any synth whose donor is in that fold's `val_real_empty_image_ids`.
  Consumers should apply this list to the synthetic portion only, keeping their
  real training IDs and weighting unchanged.
- `synth_to_real_empty.json`: one-to-one lookup by synth ID, e.g. `{"s_0001": "r_..."}`.
  Multiple synth IDs may share a donor.
- `synth_to_real_empty.csv`: `synth_id,real_empty_id,donor_val_fold`.
- `audit.json`: counts, removed IDs per fold, shortlist diagnostics and input hashes.

Matching starts with strided depth samples across 24 disjoint tiles. Only pixels
with instance mask value 0 participate. Each surviving candidate is then checked
for exact equality on **all** mask-0 depth pixels; the script requires exactly
one donor per synth image. No RGB similarity threshold or filename prior is used.

| Fold | Original train synth | Removed for held-out donor | Allowed |
|---|---:|---:|---:|
| 1 | 472 | 173 | 299 |
| 2 | 471 | 0 | 471 |
| 3 | 462 | 0 | 462 |
| 4 | 466 | 292 | 174 |
| 5 | 466 | 0 | 466 |

Mixed-object frames were already absent from the original training lists.
