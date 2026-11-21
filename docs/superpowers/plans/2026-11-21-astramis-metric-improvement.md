# Astramis Metric Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible object-level-CV pipeline that improves material, volume and heavy/take predictions and emits a test submission.

**Architecture:** `astramis_improve.py` will turn annotated train instances into masked RGB/depth geometry features, fit fold-local models, save OOF predictions and score them with the supplied metric.  The same feature/model interfaces will support final training and YOLO detection inference; generated imagery is an optional train-fold-only material augmentation, never validation data.

**Tech Stack:** Python 3.12+, NumPy, pandas, OpenCV, scikit-learn, Ultralytics, supplied `score.py`.

## Global Constraints

- All folds use `GroupKFold(n_splits=5)` over `attributes.object_id`.
- Test frames are inference-only; `score.py` is not changed.
- `~/generated` is considered optional train-fold-only RGB augmentation and cannot supply volume/mass/take labels.
- A `bulk` instance can contain multiple physical pieces; an instance is defined by its one COCO annotation/mask.
- Save models, OOF files and submissions under ignored `work/`; commit source and tests only.

---

### Task 1: Instance extraction and geometry feature contract

**Files:**
- Create: `astramis_improve.py`
- Create: `tests/test_astramis_improve.py`

**Interfaces:**
- Produces `build_instances(data_dir: Path, labels_file: str) -> pandas.DataFrame`.
- Produces `extract_features(instance: Mapping, camera: Mapping) -> dict[str, float]`.
- Each record includes `image_id`, `ann_id`, `object_id`, `form`, `material`, `volume_ml`, `mass_g`, `take`, `bbox`, `segmentation`, and feature columns.

- [ ] **Step 1: Write the failing parser test**

```python
def test_build_instances_preserves_one_row_per_annotation(tmp_path):
    data = make_tiny_astramis_dataset(tmp_path)
    rows = ai.build_instances(data, "real_train.json")
    assert rows[["ann_id", "object_id", "material", "take"]].to_dict("records") == [
        {"ann_id": 7, "object_id": "o1", "material": "metal", "take": 0}
    ]
```

- [ ] **Step 2: Run the test and confirm it fails because `build_instances` is missing.**

Run: `python -m pytest tests/test_astramis_improve.py -q`

- [ ] **Step 3: Implement parsing, polygon-mask extraction, colour statistics, mask geometry, depth quality and raw floor-plane volume.**

```python
def build_instances(data_dir, labels_file):
    info = json.loads((Path(data_dir) / "labels" / labels_file).read_text())
    images = {im["id"]: im for im in info["images"]}
    records = []
    for ann in info["annotations"]:
        record = instance_record(Path(data_dir), images[ann["image_id"]], ann)
        record["take"] = int(record["mass_g"] <= 500.0)
        records.append(record)
    return pd.DataFrame(records)

def raw_volume_ml(depth_mm, mask, plane, camera, cell=0.004):
    v, u = np.nonzero(mask & (depth_mm > 0))
    if len(v) < 30:
        return float("nan")
    # Convert valid pixels to 3-D, project them onto the floor basis, and sum
    # max(height, 0) * cell**2 over occupied floor cells.
    return integrate_floor_heights(v, u, depth_mm, plane, camera, cell)
```

- [ ] **Step 4: Run the parser test and a real-data smoke check.**

Run: `python -m pytest tests/test_astramis_improve.py -q && python astramis_improve.py features --data ../hackathon/astramis_case --out work/features.csv`

- [ ] **Step 5: Commit.**

```bash
git add astramis_improve.py tests/test_astramis_improve.py
git commit -m "feat: extract Astramis instance features"
```

### Task 2: Fold-local OOF models

**Files:**
- Modify: `astramis_improve.py`
- Modify: `tests/test_astramis_improve.py`

**Interfaces:**
- Produces `make_oof_predictions(rows: DataFrame, generated_dir: Path | None) -> DataFrame`.
- Returned rows contain `material_pred`, `material_conf`, `volume_pred`, `heavy_pred`, `take_pred` and `fold`.

- [ ] **Step 1: Write failing fold-isolation tests.**

```python
def test_group_folds_never_share_object_id():
    rows = pd.DataFrame({"object_id": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]})
    for train_idx, val_idx in ai.group_folds(rows):
        assert set(rows.object_id.iloc[train_idx]).isdisjoint(rows.object_id.iloc[val_idx])
```

- [ ] **Step 2: Run the test and confirm it fails because `group_folds` is missing.**

Run: `python -m pytest tests/test_astramis_improve.py -q`

- [ ] **Step 3: Implement fold-local preprocessing and models.**

```python
def make_oof_predictions(rows, generated_dir=None):
    oof = rows[["ann_id", "image_id", "object_id", "form"]].copy()
    for fold, (tr, va) in enumerate(group_folds(rows)):
        train, valid = rows.iloc[tr], rows.iloc[va]
        material = fit_material_model(train, generated_dir)
        volume = fit_log_volume_model(train)
        heavy = fit_heavy_model(train)
        oof.loc[va, "material_pred"] = material.predict(valid[FEATURE_COLUMNS])
        oof.loc[va, "volume_pred"] = np.exp(volume.predict(valid[FEATURE_COLUMNS]))
        oof.loc[va, "heavy_pred"] = heavy.predict(valid[FEATURE_COLUMNS])
        oof.loc[va, "take_pred"] = 1 - oof.loc[va, "heavy_pred"]
        oof.loc[va, "fold"] = fold
    return oof
```

- [ ] **Step 4: Run OOF and score oracle boxes with predicted attributes.**

Run: `python astramis_improve.py oof --features work/features.csv --out work/oof.csv --report work/oof_report.json`

- [ ] **Step 5: Commit.**

```bash
git add astramis_improve.py tests/test_astramis_improve.py
git commit -m "feat: add grouped OOF attribute models"
```

### Task 3: Augmentation ablation and final detector inference

**Files:**
- Modify: `astramis_improve.py`
- Create: `tests/test_submission_format.py`

**Interfaces:**
- Produces `write_submission(predictions: dict[str, list[dict]], out: Path) -> None`.
- CLI commands `compare-augmentation` and `predict` write reports under `work/`.

- [ ] **Step 1: Write the failing CSV-format test.**

```python
def test_write_submission_uses_none_for_empty_frame(tmp_path):
    out = tmp_path / "submission.csv"
    ai.write_submission({"r_0001": []}, out)
    assert pd.read_csv(out).iloc[0].prediction == "none"
```

- [ ] **Step 2: Run the test and confirm it fails because `write_submission` is missing.**

Run: `python -m pytest tests/test_submission_format.py -q`

- [ ] **Step 3: Implement score-compatible serialization, compare baseline vs. generated-image material augmentation, then select by real OOF Score.**

```python
def choose_variant(baseline_report, augmented_report):
    return "augmented" if augmented_report["Score"] > baseline_report["Score"] else "baseline"
```

- [ ] **Step 4: Retrain selected models on all real train data, run YOLO inference, generate the 137-row test CSV, and validate its schema.**

Run: `python astramis_improve.py predict --data ../hackathon/astramis_case --model work/seg/weights/best.pt --out work/submission.csv && python -c "import pandas as p; assert len(p.read_csv('work/submission.csv')) == 137"`

- [ ] **Step 5: Commit.**

```bash
git add astramis_improve.py tests/test_submission_format.py
git commit -m "feat: select augmentation and write submission"
```
