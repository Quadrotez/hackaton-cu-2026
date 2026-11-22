"""
5-Fold CV Evaluation of Multi-Scale TTA + WBF + Conveyor Boundary Filtering
"""
import os
import sys
import time
import json
import math
import numpy as np
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/home/conexl/Code/hackathon/hackaton-cu-2026")
sys.path.insert(0, str(ROOT))
import score as metric
import validate_offline

WORK = ROOT / "work"
WEIGHTS_PATH = ROOT / "weights" / "grisha_yolo.pt"
FOLDS_PATH = WORK / "detector_folds_v1.json"
LABELS_PATH = ROOT / "dataset" / "labels" / "real_train.json"
REAL_TRAIN_DIR = ROOT / "dataset" / "real" / "train"

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def is_conveyor_artifact(box, mask=None, img_w=1280, img_h=720):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    # 1. Top camera mount artifact (bracket top-right or top-left)
    if y1 < 25 and y2 < 220 and (x2 > 950 or x1 < 360):
        return True, "top_camera_mount"
        
    # 2. Left conveyor side rail
    if x1 < 20 and x2 < 220:
        return True, "left_side_rail"
        
    # 3. Right conveyor side rail / table corner
    if x2 > 1260 and x1 > 850 and y2 > 450:
        return True, "right_corner_rail"
        
    # 4. Extreme thin sliver along image boundaries
    if (x1 < 5 and w < 30) or (x2 > img_w - 5 and w < 30) or (y1 < 5 and h < 30) or (y2 > img_h - 5 and h < 30):
        return True, "boundary_sliver"
        
    # 5. Mask area filter: specks/dust/reflection glares < 2500 px
    if mask is not None:
        mask_area = float(mask.sum())
        if mask_area < 2500:
            return True, "tiny_mask_glare"
    elif (w * h) < 3000:
        return True, "tiny_box_glare"
        
    return False, "clean"

def refine_box_from_mask(box, mask, pad=2.0):
    """Refine box coordinates using tight mask bounds if available."""
    if mask is None or mask.sum() == 0:
        return box
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return box
    m_x1 = max(0.0, float(xs.min()) - pad)
    m_y1 = max(0.0, float(ys.min()) - pad)
    m_x2 = min(1280.0, float(xs.max()) + 1.0 + pad)
    m_y2 = min(720.0, float(ys.max()) + 1.0 + pad)
    
    # Blend: 80% mask bounds, 20% box bounds
    b_x1 = 0.8 * m_x1 + 0.2 * box[0]
    b_y1 = 0.8 * m_y1 + 0.2 * box[1]
    b_x2 = 0.8 * m_x2 + 0.2 * box[2]
    b_y2 = 0.8 * m_y2 + 0.2 * box[3]
    return [b_x1, b_y1, b_x2, b_y2]

def weighted_box_fusion(boxes_list, scores_list, classes_list, masks_list=None, weights=None, iou_thr=0.55):
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models
    weights = np.array(weights, dtype=np.float32)
    total_weight = float(np.sum(weights))

    all_boxes = []
    all_scores = []
    all_classes = []
    all_model_idx = []
    all_masks = []

    for m_idx in range(n_models):
        for i in range(len(boxes_list[m_idx])):
            all_boxes.append(boxes_list[m_idx][i])
            all_scores.append(scores_list[m_idx][i])
            all_classes.append(classes_list[m_idx][i])
            all_model_idx.append(m_idx)
            if masks_list is not None and masks_list[m_idx] is not None:
                all_masks.append(masks_list[m_idx][i])

    if len(all_boxes) == 0:
        return [], [], [], []

    unique_classes = sorted(list(set(all_classes)))
    fused_boxes = []
    fused_scores = []
    fused_classes = []
    fused_masks = []

    for cls in unique_classes:
        cls_indices = [i for i, c in enumerate(all_classes) if c == cls]
        cls_indices.sort(key=lambda idx: all_scores[idx], reverse=True)

        clusters = []
        for idx in cls_indices:
            b = all_boxes[idx]
            best_iou = 0.0
            best_cluster_idx = -1
            for c_idx, cluster in enumerate(clusters):
                iou = box_iou(b, cluster['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_cluster_idx = c_idx

            if best_iou >= iou_thr:
                cluster = clusters[best_cluster_idx]
                cluster['indices'].append(idx)
                c_boxes = np.array([all_boxes[i] for i in cluster['indices']])
                c_weights = np.array([weights[all_model_idx[i]] * all_scores[i] for i in cluster['indices']])
                cluster['box'] = (np.sum(c_boxes * c_weights[:, None], axis=0) / np.sum(c_weights)).tolist()
            else:
                clusters.append({'indices': [idx], 'box': list(b)})

        for cluster in clusters:
            c_indices = cluster['indices']
            c_boxes = np.array([all_boxes[i] for i in c_indices])
            c_scores = np.array([all_scores[i] for i in c_indices])
            c_m_idx = [all_model_idx[i] for i in c_indices]
            c_weights = np.array([weights[m] * s for m, s in zip(c_m_idx, c_scores)])
            
            f_box = (np.sum(c_boxes * c_weights[:, None], axis=0) / np.sum(c_weights)).tolist()
            
            # Consensus score: weighted score normalized by cluster count vs total models
            # Single-pass transient noise gets suppressed; consensus detections stay high
            f_score = float(np.sum([weights[m] * s for m, s in zip(c_m_idx, c_scores)]) / total_weight)
            # Boost consensus if detected by multiple scales
            if len(c_indices) > 1:
                f_score = max(f_score, float(np.mean(c_scores)))
            
            f_mask = None
            if len(all_masks) > 0:
                c_masks = np.array([all_masks[i] for i in c_indices], dtype=np.float32)
                weighted_mask = np.sum(c_masks * c_weights[:, None, None], axis=0) / np.sum(c_weights)
                f_mask = (weighted_mask >= 0.5).astype(np.uint8)
                # Refine box with mask
                f_box = refine_box_from_mask(f_box, f_mask)

            fused_boxes.append(f_box)
            fused_scores.append(f_score)
            fused_classes.append(cls)
            if f_mask is not None:
                fused_masks.append(f_mask)

    sort_order = np.argsort(fused_scores)[::-1]
    fused_boxes = [fused_boxes[i] for i in sort_order]
    fused_scores = [fused_scores[i] for i in sort_order]
    fused_classes = [fused_classes[i] for i in sort_order]
    fused_masks = [fused_masks[i] for i in sort_order] if fused_masks else []

    return fused_boxes, fused_scores, fused_classes, fused_masks

print("Loaded cv_evaluate_tta helper module successfully.")
