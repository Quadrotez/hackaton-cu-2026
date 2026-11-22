"""
Extracts tight RGB and segmentation-masked crops for DINO/CLIP multimodal feature extraction.
Eliminates background table contamination per Sprint 9 research findings.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from config import (
    CROPS_RGB_DIR,
    CROPS_MASKED_DIR,
    CROP_CONFIG,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)


def extract_object_crops(
    blended_bgr,
    instance_mask,
    bbox,
    crop_id,
    padding_ratio=CROP_CONFIG["padding_ratio"],
):
    """
    Extracts:
      1. Context-padded RGB crop
      2. Segmentation-masked RGBA crop (transparent background)
    """
    x, y, w, h = bbox
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(IMAGE_WIDTH, x + w + pad_w)
    y2 = min(IMAGE_HEIGHT, y + h + pad_h)

    # 1. RGB crop
    rgb_crop = blended_bgr[y1:y2, x1:x2].copy()

    # 2. Masked RGBA crop
    mask_roi = instance_mask[y1:y2, x1:x2]
    # Create RGBA where alpha is mask
    b, g, r = cv2.split(rgb_crop)
    alpha = (mask_roi > 0).astype(np.uint8) * 255
    rgba_crop = cv2.merge([b, g, r, alpha])

    return rgb_crop, rgba_crop, (x1, y1, x2, y2)
