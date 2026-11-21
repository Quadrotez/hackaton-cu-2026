# Frozen form rerank

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 work/venv_yolo/bin/python work/sprint4_form_rank/run.py
```

Frozen ViT-B/16 DINO and ViT-B/32 CLIP; the 892 GT crop embeddings are joined by
annotation ID. OOF evaluation and deployment cache newly encoded masked donor
crops. `eval_cache.json` identifies the ordered pixel crops, RGB sources, encoder
weights, Torch, and device. Use `--device cpu` if MPS is unavailable. Each outer
fit uses only that manifest's train images/objects and polygon masks; standardization
and three C=1 logistic probes are fitted within the fold. Class-balanced, inverse
views-per-object weights; fixed .4 DINO / .4 CLIP / .2 silhouette posterior.

The posterior estimates *bulk from pixels* without predicted form/attributes as
input. The added box asks the same posterior about its own assigned form: both OOF
and test reuse the nearest overlapping opposite-form anchor's feature vector, not
its predicted form; the test anchor links back to its verified WBF mask. This
treats the added row as an alternate label for the same physical crop.

Candidate A/B use `logit(base score) + strength * (2 * P(assigned form) - 1)`
with strengths .7 / 1.4. One per-frame shrink factor preserves the entire stable
prediction order (not only overlap pairs), hence greedy `_match` cannot change.
Promotion requires positive pooled Score and D, ≥3 positive folds, minimum
leave-one-object-out gain >0, and zero attribution churn, including matched slot.
Pick the highest passing pooled score; otherwise copy frozen files byte-for-byte.
No box, form or attribute token is changed.

`report.json`: pooled and per-fold official components and AP50 per form, match
churn, all real physical-object removals (all views removed), CSV token checks and
input/output hashes. `selected_oof.json` and `submission_selected.csv` are the
selected handoff. No leaderboard submission was made.
