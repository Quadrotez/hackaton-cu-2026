"""
Standalone Local Cross-Validation and Metric Evaluation for Astramis.

Uses official score.py components() against dataset/labels/real_train.json.
Guarantees object-disjoint evaluation across the 5 canonical folds
defined in work/detector_folds_v1.json.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import score as metric

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
LABELS_PATH = DATA / "labels" / "real_train.json"
FOLDS_PATH = ROOT / "work" / "detector_folds_v1.json"


def load_ground_truth(labels_path=LABELS_PATH):
    """Load real_train.json into {image_id: [objects]}."""
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    anns_by_img = {}
    for a in labels["annotations"]:
        x, y, w, h = a["bbox"]
        at = a["attributes"]
        anns_by_img.setdefault(a["image_id"], []).append({
            "ann_id": a["id"],
            "object_id": at.get("object_id", ""),
            "box": (x, y, x + w, y + h),
            "form": at["form"],
            "material": at["material"],
            "volume": float(at["volume_ml"]),
            "mass": float(at.get("mass_g", 0.0)),
            "take": int(at["take"]),
            "score": 1.0,
        })

    gt_by_frame = {}
    for im in labels["images"]:
        frame_name = Path(im["file_name"]).parent.name
        gt_by_frame[frame_name] = anns_by_img.get(im["id"], [])
    return gt_by_frame


def evaluate_predictions(gt_by_frame, pred_by_frame, frame_subset=None):
    """
    Computes official D, M, V, S and Score on a subset of frames (or all).
    """
    if frame_subset is not None:
        target_frames = set(frame_subset)
        gt_sub = {k: v for k, v in gt_by_frame.items() if k in target_frames}
        pred_sub = {k: pred_by_frame.get(k, []) for k in target_frames}
    else:
        gt_sub = gt_by_frame
        pred_sub = {k: pred_by_frame.get(k, []) for k in gt_by_frame}

    res = metric.components(gt_sub, pred_sub)
    
    # Detailed diagnostics
    aps = {f: metric._average_precision(gt_sub, pred_sub, f) for f in metric.FORMS}
    
    # Material breakdown
    mat_total = {m: 0 for m in metric.MATERIALS}
    mat_correct = {m: 0 for m in metric.MATERIALS}
    vol_scores = []
    heavy_correct, heavy_total = 0, 0
    light_correct, light_total = 0, 0
    
    for img, objs in gt_sub.items():
        matched = metric._match(objs, pred_sub.get(img, []))
        for g, p in zip(objs, matched):
            m = g["material"]
            mat_total[m] += 1
            if p is not None and p["material"] == m:
                mat_correct[m] += 1
            if p is not None and p["volume"] > 0 and g["volume"] > 0:
                vol_scores.append(max(0.0, 1.0 - abs(math.log(p["volume"] / g["volume"])) / math.log(3.0)))
            else:
                vol_scores.append(0.0)
            if g["take"] == 0:
                heavy_total += 1
                if p is not None and p["take"] == 0:
                    heavy_correct += 1
            else:
                light_total += 1
                if p is not None and p["take"] == 1:
                    light_correct += 1

    mat_recall = {m: (mat_correct[m] / mat_total[m] if mat_total[m] > 0 else 0.0) for m in metric.MATERIALS}
    heavy_acc = heavy_correct / heavy_total if heavy_total > 0 else 0.0
    light_acc = light_correct / light_total if light_total > 0 else 0.0

    return {
        "Score": res["Score"],
        "D": res["D"],
        "M": res["M"],
        "V": res["V"],
        "S": res["S"],
        "AP": aps,
        "Material_Recall": mat_recall,
        "Heavy_Accuracy": heavy_acc,
        "Light_Accuracy": light_acc,
        "Total_Objects": sum(mat_total.values()),
        "Frames_Evaluated": len(gt_sub),
    }


def parse_csv_predictions(csv_path):
    """Parses a submission or OOF CSV file into {image_id: [objects]}."""
    df = pd.read_csv(csv_path)
    pred_by_frame = {}
    for _, row in df.iterrows():
        img = str(row["image_id"]).strip()
        cell = row["prediction"]
        pred_by_frame[img] = metric._parse(cell, with_score=True, image_id=img)
    return pred_by_frame


def run_cv_audit(pred_by_frame):
    """Runs 5-fold cross-validation audit using detector_folds_v1.json."""
    gt_all = load_ground_truth()
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]

    print("=" * 80)
    print("ASTRAMIS 5-FOLD OBJECT-DISJOINT OFFLINE VALIDATION AUDIT")
    print("=" * 80)

    fold_scores = []
    for f in folds:
        fid = f["fold"]
        val_frames = f["val_real_image_ids"]
        stats = evaluate_predictions(gt_all, pred_by_frame, frame_subset=val_frames)
        fold_scores.append(stats)
        print(f"Fold {fid} ({len(val_frames)} frames, {stats['Total_Objects']} objects): "
              f"Score = {stats['Score']:.4f} | D={stats['D']:.4f} M={stats['M']:.4f} V={stats['V']:.4f} S={stats['S']:.4f}")

    pooled = evaluate_predictions(gt_all, pred_by_frame, frame_subset=gt_all.keys())
    print("-" * 80)
    print(f"POOLED 5-FOLD CV SCORE: {pooled['Score']:.4f}")
    print(f"  -> D (Form mAP50):        {pooled['D']:.4f}  (Bulk AP: {pooled['AP'].get('bulk', 0):.4f}, Solid AP: {pooled['AP'].get('solid', 0):.4f})")
    print(f"  -> M (Material Recall):   {pooled['M']:.4f}")
    for m, rec in pooled["Material_Recall"].items():
        print(f"     * {m:<8s}: {rec*100:5.1f}%")
    print(f"  -> V (Volume Accuracy):   {pooled['V']:.4f}")
    print(f"  -> S (Take Balanced Acc): {pooled['S']:.4f}  (Heavy: {pooled['Heavy_Accuracy']*100:.1f}%, Light: {pooled['Light_Accuracy']*100:.1f}%)")
    print("=" * 80)
    return pooled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Astramis Offline Validation")
    parser.add_argument("--csv", help="Path to predictions CSV file to evaluate across 5 folds")
    args = parser.parse_args()

    if args.csv:
        preds = parse_csv_predictions(args.csv)
        run_cv_audit(preds)
    else:
        print("Run with --csv <path_to_oof.csv> or import in python.")
