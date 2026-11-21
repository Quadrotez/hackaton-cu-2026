# Sprint 5: CPU A/B on the gated baseline

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 work/venv_yolo/bin/python -u work/sprint5_cpu/run.py > work/sprint5_cpu/run.log 2>&1
PYTHONDONTWRITEBYTECODE=1 work/venv_yolo/bin/python -u work/sprint5_cpu/reapply.py > work/sprint5_cpu/reapply.log 2>&1
```

`base_7486069_oof.json` replays the actual Sprint 4 volume gate via its Python functions, including fold-local geometric scaling and preserved per-frame detection order. It covers 263 frames, 377 predictions and 214 GT objects. OOF SHA256: `2387085171e4c23a5d7d482e495e7c76f68e054afbfe17561fc44ee65722be41`. Score **74.86069298423111**, D/M/V/S = .861649/.661919/.641869/.740642. All 44 heavy and 170 light GT have matches, with **38/44 heavy** and **105/170 light** take decisions correct. The reconstructed test prediction strings exactly equal the original gated CSV (`e9c38fe9349c95fac7a6191c55fb8b936f30c7302d356d709ac666b36a65c49f`, historical public **81.7147**, not measured again).

| Variant vs exact base | OOF | Δ pooled | Δ by fold 1–5 | Min object-removal Δ |
| --- | ---: | ---: | --- | ---: |
| Extra rescue, frozen take=0 | 75.21363416 | +.352941 | 0 / +.182927 / 0 / +.250000 / +1.184211 | +.093168 |
| Prune alternates, final conf < .05 | 74.76038167 | −.100311 | 0 / 0 / 0 / 0 / −.770975 | −.109898 |
| Prune alternates, final conf < .10 | 74.43852193 | −.422171 | 0 / 0 / −.400000 / 0 / −2.464097 | −.448759 |
| Prune alternates, final conf < .20 | 74.21896104 | −.641732 | 0 / −.617669 / −.800000 / 0 / −2.464097 | −.678068 |
| Original rescue reapplied from pre-rescue take | 74.86069298 | 0 | 0 / 0 / 0 / 0 / 0 | 0 |

Object removal drops every real OOF frame containing that object, then calculates paired Score with retained predictions unchanged; this is not a prior-refit jackknife. Promotion required positive pooled gain, at least three positive folds, positive minimum object-removal gain, and the relevant invariant components.

- **Extra rescue:** No threshold tuning for this baseline. One comparison (`off` vs the shipped `rescue-small` threshold, `p_light >= .85` and `0 < gated_volume <= 300 ml`). On each outer split, fit from real labels in its manifest train only, one median log-density per physical object, then standard form/material similarity weights. No read of `dataset/densities.csv`. Ten of 11 eligible predictions flip (8 distinct frames; 8 matched light, 2 unmatched, 0 heavy), taking light correctness to **113/170** and leaving heavy at **38/44**. D/M/V and match assignments unchanged. None of these 10 volumes was modified by the gate; the gains are from the additional take rule, not from the geometry update. The final re-score for the only passing branch is the same rescue-only OOF; evaluated against the exact base CSV with an all-real-labelled prior, it changes **0** test tokens.
- **Prune:** Only added l960 alternates, never anchor rows. Three fixed cuts, 2/4/8× the historic .025 admission floor, on the final reranked score. In this AP implementation, deleting FPs strictly after all TPs cannot increase D: it leaves all TP precision/recall coordinates and their precision envelope unchanged. Earlier FPs can hurt AP, but a weak alternate can itself be a form TP even when an anchor already takes the same GT under class-blind matching. The cuts removed 11/25/34 predictions, including **1/4/6 AP TPs**, and reduced D; M/V/S and class-blind match assignments were unchanged.
- **Original reapplication:** Recovered original fold rules from the shipped nested inner tables restricted to that outer train and matched them against both historical reports (`.80/350`, `.85/250`, `.80/300`, `.90/300`, `.90/300`). Rebuilt exact historical post-rescue n640 OOF (25 anchor flips), and replayed the exact post-rescue fusion CSV (36 anchor flips). As in the original pipeline, alternate rows inherit take from their verified anchor donor; the rescue is applied on that donor's gated volume and pre-rescue seed, so a `1→0` can only undo a verified historical `0→1`, never change a CatBoost positive. Reapplying yields **0 OOF edits** and **1 test-only `1→0`** (`r_0300`, slot 1) with no OOF support.

**No new CSV**: extra rescue is a test no-op; prune and original reapplication did not pass OOF promotion. `report.json` and `reapply_report.json` record component metrics/AP for each fold, mass-specific accuracy, activation/drop details, all leave-one-object-out deltas, and input hashes. All OOF variants and logs are in this directory.
