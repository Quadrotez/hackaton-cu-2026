# Astramis: reproducible metric-improvement pipeline

## Goal

Improve the hidden-test score without using hidden-test labels or choosing a model on test predictions.  The baseline currently scores 60.01 on its held-out objects; prior experiments recorded a 78.4 public score, but the implementation and artefacts were not preserved in the repository.

## Constraints

- Split, fold assignment, calibration and model selection use `object_id`, never random frames.
- `real/test` is inference-only.
- `score.py` remains unchanged and is the authoritative evaluator.
- All generated or augmented imagery is train-fold-only.  It is never copied into validation folds.
- RGB generation is not used to supervise volume, mass, `heavy_500`, or `take`: those labels require real depth and physical annotations.

## Data interpretation

Each material prediction is made for a detected instance crop.  An instance can be `solid` or `bulk`; a bulk instance may visually contain many small physical pieces.  Therefore multi-piece images are valid only when they represent one annotated `bulk` instance with a single material, not multiple unrelated objects.

The `~/generated` set is not Gemini output: the Gemini image call failed with HTTP 429 and the files were created by augmentation of Astramis training images.  It adds no genuinely new object identities.  It will be treated as an optional training-only regularizer and must prove a gain on real held-out objects.

## Architecture

1. **Detection.** Retain an RGB YOLO-seg detector for `bulk` and `solid`; use its box, mask and confidence as the interface to downstream models.
2. **Material.** Extract masked crop features from real RGB frames using colour/shape/depth features and pretrained image embeddings when available.  Train a fold-local classifier.  Compare no augmentation vs. the `~/generated` augmentation set, then retain only the winning real-OOF version.
3. **Volume.** Estimate raw volume from depth and the floor plane.  Learn a fold-local residual/calibration model with form, predicted material, mask geometry and depth-quality features.  Use a deterministic fallback when depth is invalid.
4. **Heavy/take.** Predict `heavy_500 = mass_g > 500` from calibrated volume, material probabilities, form and geometry; turn this into `take = 1 - heavy_500`.  Do not train on synthetic RGB-only mass labels.
5. **Selection.** Save out-of-fold predictions for every component, score them with `score.py`, and choose the final configuration by pooled OOF Score.  Retrain selected components on all train data before producing test predictions.

## Evaluation

Use five `GroupKFold` folds over `object_id`.  For every candidate, record `D`, `M`, `V`, `S`, `Score`, class-wise material recall, volume error by form/material and heavy/light accuracy.  A candidate is accepted only when its pooled real-OOF score improves on the current selected configuration; generated-image variants must also avoid a material-recall regression on the weak classes.

## Deliverables

- A single reproducible train/evaluate/predict entry point.
- A `work/` report containing fold membership, component metrics and the selected configuration.
- A test `submission.csv` generated only after model selection on OOF evidence.
