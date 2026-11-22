# Astramis Automated Synthetic Data Expansion Pipeline

Automated pipeline for synthesizing augmented training samples for the Astramis recycling competition.
Specially designed to resolve class imbalances in under-represented categories (**glass**, **paper**, **bulk metal**, **bulk plastic**) without any manual labeling.

---

## 1. Pipeline Architecture & Methodology

The pipeline performs 100% automated cut-and-paste and soft-feathered depth-aware compositing:
1. **Source Library (`library.py`)**:
   - Indexes 600 synthetic object scenes in `dataset/synth/train/` with ground-truth instance masks (`mask.png`).
   - Extracts 49 real empty conveyor donor background frames from `dataset/real/train/` spanning all illumination regimes (`dim`, `side`, `bright`).
2. **Realistic Compositing & Augmentation Engine (`blender.py`)**:
   - **Planar Transforms**: Isotropic scaling ($s \in [0.82, 1.18]$), full $360^\circ$ planar rotations, and horizontal flip ($p=0.5$).
   - **Conveyor Bounding**: All items are placed within the valid conveyor region ($X \in [220, 1060], Y \in [180, 620]$), avoiding edge artifacts.
   - **Photometric Jitter**: Dynamic brightness adjustment ($\times [0.88, 1.14]$), contrast modulation ($\times [0.90, 1.12]$), and subtle HSV color shifts.
   - **Soft Gaussian Blur Edge Feathering**: Blends object boundary pixels using Gaussian-blurred alpha masks ($\sigma = 0.9, 5 \times 5$ kernel) to eliminate sharp digital cut-and-paste boundaries.
3. **Rigorous Physics Engine**:
   - Volume scales isotropically with physical geometry: $V_{\text{new}} = \text{round}(V_{\text{orig}} \cdot s^3, 1)$.
   - Mass is strictly computed from `dataset/densities.csv`: $\text{mass}_{\text{new}} = \text{round}(V_{\text{new}} \cdot \rho, 1)$.
   - Take decision strictly enforced: $\text{take} = 1 \text{ if } \text{mass} \le 500.0\,\text{g} \text{ else } 0$.
4. **Isolated Multimodal Crops (`extract_crops_for_multimodal.py`)**:
   - Context-padded RGB crops.
   - Segmentation-masked RGBA crops with transparent backgrounds (per Sprint 9 findings, eliminating table-surface leakage in DINO/CLIP ViT self-attention).

---

## 2. Directory Structure

```
work/synth_expansion/
├── README.md                          # Pipeline documentation
├── config.py                          # Pipeline configuration, bounds, and physical densities
├── library.py                         # Dataset library and background indexer
├── blender.py                         # Compositing and augmentation engine
├── extract_crops_for_multimodal.py    # DINO/CLIP crop extractor
├── generate_synthetic_expansion.py   # Main generation script
├── verify_expansion.py                # Comprehensive verification and compliance test suite
├── annotations.json                   # COCO-format ground truth annotations
├── dataset_manifest.csv               # CSV index of all synthesized frames
├── crops_metadata.csv                 # Detailed metadata for extracted crops
├── data.yaml                          # Ultralytics YOLO Form dataset config
├── data_materials.yaml                # Ultralytics YOLO Material dataset config
├── images/                            # Synthesized full-resolution blended images (PNG)
├── masks/                             # Instance segmentation masks (PNG)
├── labels/                            # YOLO Form segmentation txt files
├── labels_material/                   # YOLO Material segmentation txt files
└── crops/
    ├── rgb/                           # RGB crops with 10% context padding
    └── masked/                        # Alpha-masked crops (background removed)
```

---

## 3. How to Run

### Run Generation
To generate the full synthetic expansion dataset:
```bash
/home/conexl/Code/hackathon/hackaton-cu-2026/.venv/bin/python3 work/synth_expansion/generate_synthetic_expansion.py --seed 42
```

### Run Automated Verification
To verify data validity, image dimensions, mask matching, YOLO label parsing, and physical consistency:
```bash
/home/conexl/Code/hackathon/hackaton-cu-2026/.venv/bin/python3 work/synth_expansion/verify_expansion.py
```

---

## 4. Downstream Model Training Integration

### YOLO Training (Ultralytics)
To train a YOLO segmentation model on the expanded dataset:
```python
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")
model.train(
    data="/home/conexl/Code/hackathon/hackaton-cu-2026/work/synth_expansion/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu"  # or 0 for GPU
)
```

### DINO / CLIP Feature Retraining
Load crops directly from `crops_metadata.csv`:
```python
import pandas as pd
from PIL import Image

df = pd.read_csv("/home/conexl/Code/hackathon/hackaton-cu-2026/work/synth_expansion/crops_metadata.csv")
for _, row in df.iterrows():
    # Use masked crop for zero background leakage
    img = Image.open(row["masked_crop_path"]).convert("RGBA")
    # Feed into ViT encoder...
```
