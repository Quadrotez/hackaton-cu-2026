"""
Compositing and augmentation engine for Astramis synthetic expansion.
Handles geometric transformations, photometric jitter, soft Gaussian edge feathering,
and rigorous physical attribute computations.
"""
import math
import cv2
import numpy as np

from config import (
    AUGMENTATION_CONFIG,
    CONVEYOR_BOUNDS,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    DENSITIES,
    FORM_TO_ID,
    FORM_TO_YOLO_ID,
    MATERIAL_TO_ID,
)


def transform_object(crop_rgb, crop_mask, scale, angle_deg, flip_h=False):
    """
    Applies isotropic scaling, 2D planar rotation, and optional horizontal flipping.
    Calculates exact bounding canvas to prevent any edge clipping.
    """
    img = crop_rgb.copy()
    mask = crop_mask.copy()

    if flip_h:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Compute rotation matrix with scale
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    # Compute new canvas dimensions so object is completely unclipped
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(math.ceil(h * sin + w * cos))
    new_h = int(math.ceil(h * cos + w * sin))

    # Adjust translation offset to canvas center
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    transformed_img = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    transformed_mask = cv2.warpAffine(
        mask, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    # Clean binary mask
    transformed_mask = (transformed_mask > 127).astype(np.uint8) * 255

    return transformed_img, transformed_mask


def apply_photometric_jitter(img, mask, rng, config=AUGMENTATION_CONFIG):
    """
    Applies brightness, contrast, and subtle HSV color jitter to object pixels.
    """
    out = img.astype(np.float32)

    # 1. Brightness jitter
    brightness = rng.uniform(config["brightness_min"], config["brightness_max"])
    out *= brightness

    # 2. Contrast jitter around mean gray
    contrast = rng.uniform(config["contrast_min"], config["contrast_max"])
    out = (out - 128.0) * contrast + 128.0
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    # 3. Subtle HSV shift
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = rng.uniform(-config["hue_shift_max"], config["hue_shift_max"])
    sat_scale = rng.uniform(config["sat_scale_min"], config["sat_scale_max"])

    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0.0, 255.0)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def extract_contours_and_bbox(binary_mask):
    """
    Extracts polygon contours and enclosing bounding box [x, y, w, h].
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 15:
            continue
        # Approximate contour slightly to produce compact polygons
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True)
        pts = approx.reshape(-1, 2)
        if len(pts) >= 3:
            polygons.append(pts.ravel().tolist())

    y_indices, x_indices = np.where(binary_mask > 0)
    if len(y_indices) == 0:
        return [], [0, 0, 0, 0]

    x1, x2 = int(x_indices.min()), int(x_indices.max())
    y1, y2 = int(y_indices.min()), int(y_indices.max())
    bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

    return polygons, bbox


def blend_object_onto_background(
    bg_img,
    composite_mask,
    obj_inst,
    rng,
    config=AUGMENTATION_CONFIG,
    bounds=CONVEYOR_BOUNDS,
    existing_masks=None
):
    """
    Renders an object onto the conveyor background.
    Returns:
        dict containing annotation ground truth, or None if placement failed.
    """
    if existing_masks is None:
        existing_masks = []

    # 1. Sample geometric augmentation parameters
    scale = rng.uniform(config["scale_min"], config["scale_max"])
    angle_deg = rng.uniform(config["rotation_min_deg"], config["rotation_max_deg"])
    flip_h = rng.random() < config["flip_prob"]

    # 2. Transform crop and mask
    rot_img, rot_mask = transform_object(
        obj_inst.crop_rgb, obj_inst.crop_mask, scale, angle_deg, flip_h
    )

    # Tighten transformed object bounding box
    nz_y, nz_x = np.where(rot_mask > 0)
    if len(nz_y) < 50:
        return None

    crop_y1, crop_y2 = nz_y.min(), nz_y.max()
    crop_x1, crop_x2 = nz_x.min(), nz_x.max()
    rot_img = rot_img[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1]
    rot_mask = rot_mask[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1]

    h_obj, w_obj = rot_img.shape[:2]

    # Check that object fits inside conveyor bounds
    max_x = bounds["x_max"] - w_obj
    max_y = bounds["y_max"] - h_obj
    min_x = bounds["x_min"]
    min_y = bounds["y_min"]

    if max_x <= min_x or max_y <= min_y:
        return None

    # 3. Find non-overlapping placement
    placed = False
    pos_x, pos_y = 0, 0
    for _ in range(25):
        cand_x = rng.integers(min_x, max_x + 1)
        cand_y = rng.integers(min_y, max_y + 1)

        # Check overlap with existing masks on the frame
        overlap = False
        for em in existing_masks:
            roi_em = em[cand_y : cand_y + h_obj, cand_x : cand_x + w_obj]
            if np.count_nonzero(np.logical_and(roi_em > 0, rot_mask > 0)) > 20:
                overlap = True
                break
        if not overlap:
            pos_x, pos_y = cand_x, cand_y
            placed = True
            break

    if not placed:
        return None

    # 4. Photometric jitter
    jittered_img = apply_photometric_jitter(rot_img, rot_mask, rng, config)

    # 5. Soft Gaussian edge feathering for physically realistic blending
    ksize = config["edge_blur_kernel"]
    sigma = config["edge_blur_sigma"]
    alpha = (rot_mask > 0).astype(np.float32)
    alpha_blurred = cv2.GaussianBlur(alpha, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    alpha_3d = np.expand_dims(alpha_blurred, -1)

    # Blend onto background ROI
    roi_bg = bg_img[pos_y : pos_y + h_obj, pos_x : pos_x + w_obj]
    blended_roi = (jittered_img.astype(np.float32) * alpha_3d + roi_bg.astype(np.float32) * (1.0 - alpha_3d))
    bg_img[pos_y : pos_y + h_obj, pos_x : pos_x + w_obj] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    # 6. Record instance mask
    instance_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    instance_mask[pos_y : pos_y + h_obj, pos_x : pos_x + w_obj] = rot_mask
    composite_mask[pos_y : pos_y + h_obj, pos_x : pos_x + w_obj] = np.maximum(
        composite_mask[pos_y : pos_y + h_obj, pos_x : pos_x + w_obj], rot_mask
    )

    # 7. Extract exact contours and bbox
    polygons, bbox = extract_contours_and_bbox(instance_mask)
    if not polygons or bbox[2] < 5 or bbox[3] < 5:
        return None

    # 8. Compute physics attributes strictly following densities.csv
    # Volume scales isotropically with scale^3
    aug_volume_ml = round(float(obj_inst.volume_ml * (scale ** 3)), 1)
    aug_volume_ml = max(1.0, aug_volume_ml)

    # Mass is computed using physical density
    density = obj_inst.object_density
    aug_mass_g = round(float(aug_volume_ml * density), 1)

    # Take rule strictly from competition specification (mass <= 500g -> take 1 else 0)
    take = 1 if aug_mass_g <= 500.0 else 0

    ann_info = {
        "form": obj_inst.form,
        "material": obj_inst.material,
        "object_id": obj_inst.obj_id,
        "volume_ml": aug_volume_ml,
        "mass_g": aug_mass_g,
        "take": take,
        "density_g_per_ml": round(density, 4),
        "scale": round(scale, 4),
        "rotation_deg": round(angle_deg, 2),
        "flip_h": bool(flip_h),
        "bbox": bbox,  # [x, y, w, h]
        "segmentation": polygons,
        "area": float(np.count_nonzero(instance_mask)),
        "instance_mask": instance_mask,
    }
    return ann_info
