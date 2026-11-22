"""
Configuration and constants for Astramis Synthetic Expansion Pipeline.
"""
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path("/home/conexl/Code/hackathon/hackaton-cu-2026")
WORK_DIR = BASE_DIR / "work"
EXPANSION_DIR = WORK_DIR / "synth_expansion"

DATA_DIR = BASE_DIR / "dataset"
SYNTH_TRAIN_DIR = DATA_DIR / "synth" / "train"
REAL_TRAIN_DIR = DATA_DIR / "real" / "train"
SYNTH_JSON_PATH = DATA_DIR / "labels" / "synth_train.json"
REAL_JSON_PATH = DATA_DIR / "labels" / "real_train.json"
DENSITIES_PATH = DATA_DIR / "densities.csv"

# Output directories
IMAGES_DIR = EXPANSION_DIR / "images"
MASKS_DIR = EXPANSION_DIR / "masks"
LABELS_YOLO_DIR = EXPANSION_DIR / "labels"
LABELS_MATERIAL_DIR = EXPANSION_DIR / "labels_material"
CROPS_RGB_DIR = EXPANSION_DIR / "crops" / "rgb"
CROPS_MASKED_DIR = EXPANSION_DIR / "crops" / "masked"

# Canonical categories
FORMS = ["bulk", "solid"]
FORM_TO_ID = {"bulk": 1, "solid": 2}  # COCO 1-indexed
FORM_TO_YOLO_ID = {"bulk": 0, "solid": 1}  # YOLO 0-indexed

MATERIALS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
MATERIAL_TO_ID = {mat: i for i, mat in enumerate(MATERIALS)}

# Image specifications
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Valid conveyor placement region (safely inside conveyor belt, avoiding border cameras/mounts)
CONVEYOR_BOUNDS = {
    "x_min": 220,
    "x_max": 1060,
    "y_min": 180,
    "y_max": 620,
}

# Augmentation ranges
AUGMENTATION_CONFIG = {
    "scale_min": 0.82,
    "scale_max": 1.18,
    "rotation_min_deg": -180.0,
    "rotation_max_deg": 180.0,
    "flip_prob": 0.5,
    "brightness_min": 0.88,
    "brightness_max": 1.14,
    "contrast_min": 0.90,
    "contrast_max": 1.12,
    "hue_shift_max": 4,
    "sat_scale_min": 0.92,
    "sat_scale_max": 1.08,
    "edge_blur_kernel": 5,
    "edge_blur_sigma": 0.9,
}

# Crop export settings for DINO/CLIP
CROP_CONFIG = {
    "padding_ratio": 0.10,  # 10% context padding around bounding box
    "target_size": (224, 224),  # Standard ViT resolution
}

# Load densities dictionary: (form, material) -> dict(density, min, max)
def load_densities_table():
    df = pd.read_csv(DENSITIES_PATH)
    densities = {}
    for _, row in df.iterrows():
        densities[(row["form"], row["material"])] = {
            "density_g_per_ml": float(row["density_g_per_ml"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "n_objects": int(row["n_objects"]),
        }
    return densities

DENSITIES = load_densities_table()
