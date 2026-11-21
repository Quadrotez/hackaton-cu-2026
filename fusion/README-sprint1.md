# Sprint 1: class-agnostic WBF and the exact 78.97 baseline

```bash
work/venv_yolo/bin/python fusion/sprint1_wbf_oof.py oof
work/venv_yolo/bin/python fusion/sprint1_wbf_oof.py test
python3 -m unittest discover -s fusion -v
```

`oof` checks the detector image manifest against `cv_folds.json` and the current
labels, holds out entire physical objects and scores all 263 real frames once.
Only checkpoints tied to the split can vote. The frozen n640 OOF predictions
are the anchor; l960 predictions use the matching checkpoint and fold-local
heads. Weights from the m960 dataset have fold numbers but lack training
manifests, and the legacy s640 checkpoint has no shared five-fold provenance.
They are valid **test inputs**, not OOF inputs. Inference on a cache miss is
opt-in (`--allow-inference`); it does not qualify an unverified checkpoint.

Cached 960 boxes and bit-packed masks are reused in `work/oof_cache_960/`.
The completed run's `work/sprint1_agnostic_oof_v2/mask_index.json` records the archive and SHA-256 per
fold/source. In `per_model_predictions.json`, each `mask_index` points to the
original mask slot for that image. `report.json` contains D/M/V/S/Score, misses,
duplicate counts and deltas for all methods per fold and pooled; `metrics.csv`
provides the same scores in tabular form. `--out` can select a fresh directory
for reruns. WBF uses
one confidence-weighted vote per checkpoint, selects the form by summed vote,
then suppresses remaining overlaps regardless of form at IoU ≥ 0.55. The OOF
ablation measures WBF and final NMS separately before adding `rescue-small`.

On the verified n640+l960 inputs, class-aware fusion plus `rescue-small` scores
72.5480 (+1.9761 over n640 plus `rescue-small`, positive on all five folds).
Class-agnostic WBF+NMS plus `rescue-small` scores 69.4768. There are zero
geometry misses in both; the class-agnostic variant removes the only correct
form hypothesis (at IoU ≥ 0.5) for 26 labeled instances. Final NMS, isolated
after class-agnostic WBF, makes no difference on this OOF.

The two test-only 960+s640 exports are in `work/sprint1_agnostic_test/`.
The exact old fusion baseline is Git blob
`a52cf92:candidates/submission_fused_max_7897.csv` (`origin/best-submissions`,
SHA-256 `e07d09498834d5b4a0d2b3ed47dc35e016b9e74f34b7c3adddd1502f8bd6b2a1`).
`origin/submission-fusion` documents the recipe; running it with the local
WBF6 and Grisha single CSVs reproduces the blob byte for byte. For the isolated
take ablation, on a clean output directory:

```bash
mkdir -p work/sprint1_grisha
git show a52cf92:candidates/submission_fused_max_7897.csv > work/sprint1_grisha/submission_fused_max_7897.csv
python3 fusion/attribute_postprocess.py --mode best \
  --input work/sprint1_grisha/submission_fused_max_7897.csv \
  --out work/sprint1_grisha/submission_fused_max_7897_rescue_small.csv
work/venv_yolo/bin/python work/submission_tools.py \
  work/sprint1_grisha/submission_fused_max_7897.csv \
  work/sprint1_grisha/submission_fused_max_7897_rescue_small.csv
```
