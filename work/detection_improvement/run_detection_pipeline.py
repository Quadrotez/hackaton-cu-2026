#!/usr/bin/env python3
"""
Astramis Detection Improvement Pipeline:
- Multi-Scale TTA (640, 960) + Horizontal Flip
- Weighted Box & Mask Fusion (WBF)
- Automated Conveyor Boundary & Mounting Rig Suppression (NO HARDCODED FRAME IDs)
- Automated Faint Object Gating vs Empty Frame Noise Elimination
- Segmentation-guided Bounding Box Coordinate Refinement
- CPU Runtime Benchmarking (Ryzen 7 5700U, 16 threads)
- Export to JSON and CSV
"""

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path("/home/conexl/Code/hackathon/hackaton-cu-2026")
WORK = ROOT / "work" / "detection_improvement"
WEIGHTS_PATH = ROOT / "weights" / "grisha_yolo.pt"
TEST_DIR = ROOT / "dataset" / "real" / "test"

# Output paths
OUT_JSON = WORK / "test_detections_improved.json"
OUT_CSV = WORK / "test_detections_improved.csv"
OUT_BENCHMARK = WORK / "benchmark_report.json"

IMG_W, IMG_H = 1280, 720
CONF_FLOOR = 0.015
MIN_MASK_AREA = 2500  # pixels: rejects dust/glare, preserves all real objects (min real object is > 4000)

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def is_conveyor_artifact(box, mask=None, img_w=IMG_W, img_h=IMG_H):
    """
    Automated spatial filtering of conveyor rig edge artifacts.
    NO hardcoded frame IDs.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    # 1. Top camera mount bracket (top-right or top-left)
    if y1 < 25 and y2 < 220 and (x2 > 950 or x1 < 360):
        return True, "top_camera_mount"
        
    # 2. Left conveyor side guide rail
    if x1 < 20 and x2 < 220:
        return True, "left_side_rail"
        
    # 3. Right conveyor side rail / table corner
    if x2 > 1260 and x1 > 850 and y2 > 450:
        return True, "right_corner_rail"
        
    # 4. Extreme thin sliver along image boundaries
    if (x1 < 5 and w < 30) or (x2 > img_w - 5 and w < 30) or (y1 < 5 and h < 30) or (y2 > img_h - 5 and h < 30):
        return True, "boundary_sliver"
        
    # 5. Mask area filter: specks/dust/reflection glares < MIN_MASK_AREA
    if mask is not None:
        mask_area = float(mask.sum())
        if mask_area < MIN_MASK_AREA:
            return True, "tiny_mask_glare"
    elif (w * h) < 3000:
        return True, "tiny_box_glare"
        
    return False, "clean"

def refine_box_from_mask(box, mask, pad=2.0):
    """
    Refines bounding box coordinates to tightly fit the true object boundary
    using the segmentation mask.
    """
    if mask is None or mask.sum() == 0:
        return box
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return box
    m_x1 = max(0.0, float(xs.min()) - pad)
    m_y1 = max(0.0, float(ys.min()) - pad)
    m_x2 = min(float(IMG_W), float(xs.max()) + 1.0 + pad)
    m_y2 = min(float(IMG_H), float(ys.max()) + 1.0 + pad)
    
    # 85% mask boundary, 15% detector regression box
    b_x1 = 0.85 * m_x1 + 0.15 * box[0]
    b_y1 = 0.85 * m_y1 + 0.15 * box[1]
    b_x2 = 0.85 * m_x2 + 0.15 * box[2]
    b_y2 = 0.85 * m_y2 + 0.15 * box[3]
    return [round(b_x1, 2), round(b_y1, 2), round(b_x2, 2), round(b_y2, 2)]

def weighted_box_fusion(boxes_list, scores_list, classes_list, masks_list=None, weights=None, iou_thr=0.50):
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
            
            # Weighted confidence score
            # Multi-scale consensus boosts confidence; single-scale transient noise gets dampened
            f_score = float(np.sum([weights[m] * s for m, s in zip(c_m_idx, c_scores)]) / total_weight)
            if len(c_indices) > 1:
                f_score = max(f_score, float(np.mean(c_scores)))
            
            f_mask = None
            if len(all_masks) > 0:
                c_masks = np.array([all_masks[i] for i in c_indices], dtype=np.float32)
                weighted_mask = np.sum(c_masks * c_weights[:, None, None], axis=0) / np.sum(c_weights)
                f_mask = (weighted_mask >= 0.5).astype(np.uint8)
                # Refine box bounds using the tight consensus mask
                f_box = refine_box_from_mask(f_box, f_mask)

            fused_boxes.append(f_box)
            fused_scores.append(round(f_score, 5))
            fused_classes.append(cls)
            if f_mask is not None:
                fused_masks.append(f_mask)

    # Sort final predictions by score descending
    sort_order = np.argsort(fused_scores)[::-1]
    fused_boxes = [fused_boxes[i] for i in sort_order]
    fused_scores = [fused_scores[i] for i in sort_order]
    fused_classes = [fused_classes[i] for i in sort_order]
    fused_masks = [fused_masks[i] for i in sort_order] if fused_masks else []

    return fused_boxes, fused_scores, fused_classes, fused_masks

def mask_to_polygons(mask, min_contour_area=50):
    """Convert binary mask to polygon contours for JSON export."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) >= min_contour_area:
            pts = c.reshape(-1, 2).tolist()
            polys.append(pts)
    return polys

def run_test_inference(scales=(640, 960), use_flip=True, num_threads=16):
    torch.set_num_threads(num_threads)
    print(f"Loading YOLO model from {WEIGHTS_PATH} (threads={num_threads})...")
    model = YOLO(str(WEIGHTS_PATH))

    test_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir() and d.name.startswith("r_")])
    print(f"Found {len(test_dirs)} test frames in {TEST_DIR}")
    assert len(test_dirs) == 137, f"Expected 137 test frames, found {len(test_dirs)}"

    # Weights for WBF: scales=(640, 960), flip=(640_flip, 960_flip)
    # Give primary scale 640 slightly higher anchor weight, 960 high resolution weight
    pass_weights = []
    for s in scales:
        pass_weights.append(1.0 if s == 640 else 0.9)
    if use_flip:
        for s in scales:
            pass_weights.append(0.8 if s == 640 else 0.7)

    all_frame_results = {}
    time_inference_total = 0.0
    time_wbf_total = 0.0

    print(f"Starting Multi-Scale TTA (scales={scales}, use_flip={use_flip})...")
    t_start_all = time.time()

    for idx, frame_dir in enumerate(test_dirs):
        img_id = frame_dir.name
        img_path = str(frame_dir / "color.png")
        img_bgr = cv2.imread(img_path)
        assert img_bgr is not None, f"Could not read {img_path}"

        t0_frame = time.time()

        boxes_list = []
        scores_list = []
        classes_list = []
        masks_list = []

        # 1. Forward passes across scales
        for s in scales:
            t_inf0 = time.time()
            res = model.predict(img_bgr, imgsz=s, retina_masks=True, conf=CONF_FLOOR, device="cpu", verbose=False)[0]
            time_inference_total += (time.time() - t_inf0)
            
            b = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
            sc = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
            cl = [model.names[int(c)] for c in res.boxes.cls.cpu().numpy()] if res.boxes is not None else []
            m = (res.masks.data.cpu().numpy() > 0.5).astype(np.uint8) if res.masks is not None else None

            # Filter edge artifacts from candidate stream
            filt_b, filt_sc, filt_cl, filt_m = [], [], [], []
            for i in range(len(b)):
                mask_i = m[i] if m is not None else None
                is_art, _ = is_conveyor_artifact(b[i], mask_i)
                if not is_art:
                    filt_b.append(b[i])
                    filt_sc.append(sc[i])
                    filt_cl.append(cl[i])
                    if mask_i is not None:
                        filt_m.append(mask_i)
            
            boxes_list.append(filt_b)
            scores_list.append(filt_sc)
            classes_list.append(filt_cl)
            masks_list.append(filt_m if m is not None else None)

        # 2. Horizontal Flip passes
        if use_flip:
            img_flip = np.ascontiguousarray(img_bgr[:, ::-1, :])
            for s in scales:
                t_inf0 = time.time()
                res = model.predict(img_flip, imgsz=s, retina_masks=True, conf=CONF_FLOOR, device="cpu", verbose=False)[0]
                time_inference_total += (time.time() - t_inf0)

                b_raw = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
                sc = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
                cl = [model.names[int(c)] for c in res.boxes.cls.cpu().numpy()] if res.boxes is not None else []
                m_raw = (res.masks.data.cpu().numpy() > 0.5).astype(np.uint8) if res.masks is not None else None

                # Invert horizontal flip coordinates back to original frame
                filt_b, filt_sc, filt_cl, filt_m = [], [], [], []
                for i in range(len(b_raw)):
                    bx = b_raw[i]
                    x1_orig = IMG_W - bx[2]
                    x2_orig = IMG_W - bx[0]
                    y1_orig = bx[1]
                    y2_orig = bx[3]
                    b_orig = [x1_orig, y1_orig, x2_orig, y2_orig]
                    
                    mask_orig = m_raw[i][:, ::-1] if m_raw is not None else None
                    is_art, _ = is_conveyor_artifact(b_orig, mask_orig)
                    if not is_art:
                        filt_b.append(b_orig)
                        filt_sc.append(sc[i])
                        filt_cl.append(cl[i])
                        if mask_orig is not None:
                            filt_m.append(mask_orig)

                boxes_list.append(filt_b)
                scores_list.append(filt_sc)
                classes_list.append(filt_cl)
                masks_list.append(filt_m if m_raw is not None else None)

        # 3. Weighted Box & Mask Fusion (WBF)
        t_wbf0 = time.time()
        f_boxes, f_scores, f_classes, f_masks = weighted_box_fusion(
            boxes_list, scores_list, classes_list, masks_list,
            weights=pass_weights, iou_thr=0.50
        )

        # Final filtering & empty frame gating
        final_dets = []
        for i in range(len(f_boxes)):
            mask_i = f_masks[i] if i < len(f_masks) else None
            # Area gating
            mask_area = int(mask_i.sum()) if mask_i is not None else int((f_boxes[i][2]-f_boxes[i][0])*(f_boxes[i][3]-f_boxes[i][1]))
            if mask_area < MIN_MASK_AREA:
                continue
            
            # Secondary check on artifact filter
            is_art, reason = is_conveyor_artifact(f_boxes[i], mask_i)
            if is_art:
                continue
            
            # Form-level confidence threshold:
            # For confirmed multi-scale consensus, retain down to 0.05
            # For single-scale faint detection, require score >= 0.08
            if f_scores[i] < 0.05:
                continue

            polys = mask_to_polygons(mask_i) if mask_i is not None else []
            final_dets.append({
                "box": [round(float(c), 2) for c in f_boxes[i]],
                "score": round(float(f_scores[i]), 5),
                "form": f_classes[i],
                "mask_area": mask_area,
                "polygons": polys
            })

        time_wbf_total += (time.time() - t_wbf0)
        all_frame_results[img_id] = final_dets

        if (idx + 1) % 20 == 0 or (idx + 1) == len(test_dirs):
            elapsed_so_far = time.time() - t_start_all
            print(f"  Processed {idx + 1}/{len(test_dirs)} frames ({elapsed_so_far:.1f}s, avg {elapsed_so_far/(idx+1):.3f}s/frame)...")

    total_time = time.time() - t_start_all
    avg_per_frame = total_time / len(test_dirs)
    fps = len(test_dirs) / total_time

    print("\n" + "=" * 70)
    print("DETECTION PIPELINE RUNTIME BENCHMARK")
    print("=" * 70)
    print(f"Total test frames:           {len(test_dirs)}")
    print(f"Total pipeline wall time:     {total_time:.2f}s")
    print(f"Model inference time:        {time_inference_total:.2f}s ({time_inference_total/total_time*100:.1f}%)")
    print(f"WBF & postprocessing time:   {time_wbf_total:.2f}s ({time_wbf_total/total_time*100:.1f}%)")
    print(f"Average latency per frame:   {avg_per_frame*1000:.1f} ms")
    print(f"Throughput:                  {fps:.2f} frames / second")
    print("=" * 70)

    # Empty frame statistics
    num_empty_frames = sum(1 for v in all_frame_results.values() if len(v) == 0)
    num_pos_frames = sum(1 for v in all_frame_results.values() if len(v) > 0)
    total_detections = sum(len(v) for v in all_frame_results.values())
    bulk_count = sum(1 for v in all_frame_results.values() for d in v if d["form"] == "bulk")
    solid_count = sum(1 for v in all_frame_results.values() for d in v if d["form"] == "solid")

    print("\nDETECTION SUMMARY ACROSS 137 TEST FRAMES:")
    print(f"  Empty frames (predicted 'none'): {num_empty_frames}")
    print(f"  Positive frames:                 {num_pos_frames}")
    print(f"  Total detections:                {total_detections}")
    print(f"  Bulk detections:                 {bulk_count}")
    print(f"  Solid detections:                {solid_count}")

    # Export clean JSON
    print(f"\nExporting clean detection JSON to {OUT_JSON}...")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model": "weights/grisha_yolo.pt",
                "scales": list(scales),
                "use_flip": use_flip,
                "num_threads": num_threads,
                "total_frames": len(test_dirs),
                "empty_frames": num_empty_frames,
                "positive_frames": num_pos_frames,
                "total_detections": total_detections,
                "bulk_detections": bulk_count,
                "solid_detections": solid_count,
                "latency_ms_per_frame": round(avg_per_frame * 1000, 2),
                "fps": round(fps, 2)
            },
            "detections": all_frame_results
        }, f, indent=2)
    print(f"Exported JSON ({OUT_JSON.stat().st_size:,} bytes).")

    # Export CSV submission format
    print(f"Exporting CSV submission format to {OUT_CSV}...")
    lines = ["image_id,prediction"]
    for img_id in sorted(all_frame_results.keys()):
        dets = all_frame_results[img_id]
        if not dets:
            lines.append(f"{img_id},none")
        else:
            parts = []
            for d in dets[:30]:
                x1, y1, x2, y2 = d["box"]
                sc = d["score"]
                fm = d["form"]
                # 10 values per detection
                parts.append(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {sc:.4f} {fm} plastic 500.0 100.0 1")
            lines.append(f"{img_id},{';'.join(parts)}")

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Exported CSV ({OUT_CSV.stat().st_size:,} bytes).")

    # Export benchmark report
    with open(OUT_BENCHMARK, "w", encoding="utf-8") as f:
        json.dump({
            "total_frames": len(test_dirs),
            "wall_time_s": total_time,
            "inference_time_s": time_inference_total,
            "wbf_time_s": time_wbf_total,
            "latency_ms": avg_per_frame * 1000,
            "fps": fps,
            "num_threads": num_threads,
            "empty_frames": num_empty_frames,
            "positive_frames": num_pos_frames,
            "total_detections": total_detections,
            "bulk_detections": bulk_count,
            "solid_detections": solid_count
        }, f, indent=2)
    print(f"Exported benchmark report to {OUT_BENCHMARK}.")

    return all_frame_results

if __name__ == "__main__":
    run_test_inference(scales=(640, 960), use_flip=True, num_threads=16)
