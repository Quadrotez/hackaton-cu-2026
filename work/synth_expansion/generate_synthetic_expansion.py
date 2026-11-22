"""
End-to-End Synthetic Dataset Expansion Pipeline for Astramis Hackathon.
Synthesizes >= 300 photorealistic augmented frames targeting under-represented classes:
  - glass (solid glass)
  - paper (bulk and solid paper)
  - bulk metal
  - bulk plastic
  - other balanced classes (mineral, wood, solid metal)
Generates exact bounding boxes, segmentation masks, COCO annotations, YOLO labels,
and isolated DINO/CLIP crops.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    EXPANSION_DIR,
    IMAGES_DIR,
    MASKS_DIR,
    LABELS_YOLO_DIR,
    LABELS_MATERIAL_DIR,
    CROPS_RGB_DIR,
    CROPS_MASKED_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    FORMS,
    MATERIALS,
    FORM_TO_ID,
    FORM_TO_YOLO_ID,
    MATERIAL_TO_ID,
    DENSITIES,
)
from library import DatasetLibrary
from blender import blend_object_onto_background
from extract_crops_for_multimodal import extract_object_crops


def make_dirs():
    """Creates required output directory structure."""
    for d in [
        IMAGES_DIR,
        MASKS_DIR,
        LABELS_YOLO_DIR,
        LABELS_MATERIAL_DIR,
        CROPS_RGB_DIR,
        CROPS_MASKED_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def build_generation_schedule():
    """
    Constructs target plan to aggressively expand under-represented categories
    and provide balanced representation across all classes.
    """
    schedule = []
    # Primary under-represented target quotas (Single-object scenes)
    quotas = [
        ("solid", "glass", 50),     # Glass is heavily under-represented
        ("bulk", "paper", 45),      # Bulk paper has only 1 object & 12 real frames
        ("solid", "paper", 45),     # Solid paper
        ("bulk", "metal", 45),      # Bulk metal
        ("bulk", "plastic", 45),    # Bulk plastic
        ("bulk", "mineral", 20),
        ("bulk", "wood", 20),
        ("solid", "metal", 25),
        ("solid", "mineral", 20),
        ("solid", "wood", 20),
    ]

    for form, mat, count in quotas:
        for _ in range(count):
            schedule.append([(form, mat)])

    # Multi-object scenes (combining under-represented classes with other objects)
    multi_pairs = [
        [("solid", "glass"), ("bulk", "paper")],
        [("bulk", "metal"), ("bulk", "plastic")],
        [("solid", "glass"), ("solid", "paper")],
        [("solid", "metal"), ("solid", "plastic")],
        [("bulk", "paper"), ("bulk", "metal")],
        [("solid", "paper"), ("bulk", "plastic")],
        [("solid", "glass"), ("bulk", "wood")],
        [("bulk", "mineral"), ("solid", "mineral")],
    ]
    # Add 35 multi-object frames
    for i in range(35):
        pair = multi_pairs[i % len(multi_pairs)]
        schedule.append(pair)

    return schedule


def export_yolo_configs():
    """Exports data.yaml for YOLO training."""
    # Form detection/segmentation yaml
    yolo_yaml = f"""# Astramis YOLO Form Segmentation & Detection Config
path: {EXPANSION_DIR}
train: images
val: images

names:
  0: bulk
  1: solid
"""
    (EXPANSION_DIR / "data.yaml").write_text(yolo_yaml, encoding="utf-8")

    # Material classification/segmentation yaml
    mat_yaml = f"""# Astramis YOLO Material Segmentation & Detection Config
path: {EXPANSION_DIR}
train: images
val: images

names:
"""
    for mat, idx in sorted(MATERIAL_TO_ID.items(), key=lambda kv: kv[1]):
        mat_yaml += f"  {idx}: {mat}\n"

    (EXPANSION_DIR / "data_materials.yaml").write_text(mat_yaml, encoding="utf-8")
    print(f"[Export] Saved YOLO config files: data.yaml and data_materials.yaml")


def run_expansion(seed=42):
    print("=" * 70)
    print("Astramis Automated Synthetic Data Expansion Pipeline")
    print("=" * 70)

    make_dirs()
    rng = np.random.default_rng(seed)

    print("\n[1/5] Initializing Dataset Library (extracting source synth objects & real empty backgrounds)...")
    library = DatasetLibrary()

    schedule = build_generation_schedule()
    # Shuffle schedule deterministically
    rng.shuffle(schedule)
    total_frames = len(schedule)
    print(f"\n[2/5] Synthesizing {total_frames} target frames...")

    coco_images = []
    coco_annotations = []
    coco_categories = [
        {"id": 1, "name": "bulk"},
        {"id": 2, "name": "solid"},
    ]

    crops_records = []
    manifest_records = []

    ann_id_counter = 1
    crop_id_counter = 1

    for img_idx, obj_specs in enumerate(tqdm(schedule, desc="Synthesizing frames")):
        frame_id = img_idx + 1
        frame_name = f"syn_exp_{frame_id:04d}"

        # 1. Select random donor empty conveyor background
        bg_bgr, bg_path = library.get_background(rng)
        donor_id = bg_path.parent.name

        # Create blank canvas for composite instance masks
        composite_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        frame_annotations = []
        frame_instance_masks = []

        # 2. Blend each requested object onto background
        for inst_idx, (form, mat) in enumerate(obj_specs, start=1):
            obj_inst = library.sample_object(form, mat, rng)
            ann_info = blend_object_onto_background(
                bg_img=bg_bgr,
                composite_mask=composite_mask,
                obj_inst=obj_inst,
                rng=rng,
                existing_masks=frame_instance_masks,
            )

            if ann_info is not None:
                ann_info["instance"] = inst_idx
                frame_annotations.append(ann_info)
                frame_instance_masks.append(ann_info["instance_mask"])

        # Fallback if placement failed: retry single object
        if not frame_annotations:
            form, mat = obj_specs[0]
            obj_inst = library.sample_object(form, mat, rng)
            ann_info = blend_object_onto_background(
                bg_img=bg_bgr,
                composite_mask=composite_mask,
                obj_inst=obj_inst,
                rng=rng,
                existing_masks=[],
            )
            if ann_info is not None:
                ann_info["instance"] = 1
                frame_annotations.append(ann_info)
                frame_instance_masks.append(ann_info["instance_mask"])

        # 3. Save blended color image
        img_out_path = IMAGES_DIR / f"{frame_name}.png"
        cv2.imwrite(str(img_out_path), bg_bgr)

        # 4. Save combined instance mask
        mask_out_path = MASKS_DIR / f"{frame_name}.png"
        cv2.imwrite(str(mask_out_path), composite_mask)

        # 5. Record COCO image entry
        coco_images.append({
            "id": frame_id,
            "file_name": f"images/{frame_name}.png",
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "donor_background": donor_id,
        })

        # 6. Generate YOLO format label files
        yolo_form_lines = []
        yolo_mat_lines = []

        for ann_info in frame_annotations:
            current_ann_id = ann_id_counter
            ann_id_counter += 1

            form_name = ann_info["form"]
            mat_name = ann_info["material"]
            coco_cat_id = FORM_TO_ID[form_name]
            yolo_form_id = FORM_TO_YOLO_ID[form_name]
            yolo_mat_id = MATERIAL_TO_ID[mat_name]

            # Append to COCO annotations
            coco_annotations.append({
                "id": current_ann_id,
                "image_id": frame_id,
                "bbox": ann_info["bbox"],
                "segmentation": ann_info["segmentation"],
                "area": ann_info["area"],
                "iscrowd": 0,
                "category_id": coco_cat_id,
                "instance": ann_info["instance"],
                "attributes": {
                    "form": form_name,
                    "material": mat_name,
                    "volume_ml": ann_info["volume_ml"],
                    "mass_g": ann_info["mass_g"],
                    "take": ann_info["take"],
                    "object_id": ann_info["object_id"],
                    "density_g_per_ml": ann_info["density_g_per_ml"],
                    "scale": ann_info["scale"],
                    "rotation_deg": ann_info["rotation_deg"],
                    "flip_h": ann_info["flip_h"],
                    "donor_background": donor_id,
                }
            })

            # Format normalized YOLO segmentation polygon
            # Take the largest contour for YOLO segmentation line
            best_poly = max(ann_info["segmentation"], key=len)
            norm_pts = []
            for i in range(0, len(best_poly), 2):
                nx = np.clip(best_poly[i] / float(IMAGE_WIDTH), 0.0, 1.0)
                ny = np.clip(best_poly[i + 1] / float(IMAGE_HEIGHT), 0.0, 1.0)
                norm_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])

            yolo_poly_str = " ".join(norm_pts)
            yolo_form_lines.append(f"{yolo_form_id} {yolo_poly_str}")
            yolo_mat_lines.append(f"{yolo_mat_id} {yolo_poly_str}")

            # 7. Extract tight context RGB and masked crops for DINO/CLIP
            crop_name = f"crop_{crop_id_counter:05d}"
            crop_id_counter += 1

            rgb_crop, rgba_crop, crop_box = extract_object_crops(
                blended_bgr=bg_bgr,
                instance_mask=ann_info["instance_mask"],
                bbox=ann_info["bbox"],
                crop_id=crop_name,
            )

            rgb_crop_path = CROPS_RGB_DIR / f"{crop_name}.png"
            rgba_crop_path = CROPS_MASKED_DIR / f"{crop_name}.png"
            cv2.imwrite(str(rgb_crop_path), rgb_crop)
            cv2.imwrite(str(rgba_crop_path), rgba_crop)

            crops_records.append({
                "crop_id": crop_name,
                "image_id": frame_name,
                "ann_id": current_ann_id,
                "object_id": ann_info["object_id"],
                "form": form_name,
                "material": mat_name,
                "volume_ml": ann_info["volume_ml"],
                "mass_g": ann_info["mass_g"],
                "take": ann_info["take"],
                "density_g_per_ml": ann_info["density_g_per_ml"],
                "scale": ann_info["scale"],
                "rotation_deg": ann_info["rotation_deg"],
                "crop_x1": crop_box[0],
                "crop_y1": crop_box[1],
                "crop_x2": crop_box[2],
                "crop_y2": crop_box[3],
                "crop_w": crop_box[2] - crop_box[0],
                "crop_h": crop_box[3] - crop_box[1],
                "rgb_crop_path": str(rgb_crop_path),
                "masked_crop_path": str(rgba_crop_path),
            })

        # Write YOLO label files
        (LABELS_YOLO_DIR / f"{frame_name}.txt").write_text("\n".join(yolo_form_lines) + "\n", encoding="utf-8")
        (LABELS_MATERIAL_DIR / f"{frame_name}.txt").write_text("\n".join(yolo_mat_lines) + "\n", encoding="utf-8")

        # Manifest entry
        manifest_records.append({
            "image_id": frame_name,
            "image_path": str(img_out_path),
            "mask_path": str(mask_out_path),
            "donor_bg": donor_id,
            "num_objects": len(frame_annotations),
            "forms": ";".join([a["form"] for a in frame_annotations]),
            "materials": ";".join([a["material"] for a in frame_annotations]),
            "object_ids": ";".join([a["object_id"] for a in frame_annotations]),
            "volumes": ";".join([str(a["volume_ml"]) for a in frame_annotations]),
            "masses": ";".join([str(a["mass_g"]) for a in frame_annotations]),
            "takes": ";".join([str(a["take"]) for a in frame_annotations]),
        })

    # Save COCO annotations.json
    print("\n[3/5] Saving COCO annotations.json...")
    coco_dataset = {
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    with open(EXPANSION_DIR / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco_dataset, f, indent=2)

    # Save YOLO configs
    print("[4/5] Saving YOLO configuration YAML files...")
    export_yolo_configs()

    # Save manifests and crops CSV
    print("[5/5] Saving dataset manifests and DINO/CLIP crops metadata...")
    pd.DataFrame(manifest_records).to_csv(EXPANSION_DIR / "dataset_manifest.csv", index=False)
    pd.DataFrame(crops_records).to_csv(EXPANSION_DIR / "crops_metadata.csv", index=False)

    print("\n" + "=" * 70)
    print("Synthetic Expansion Pipeline Completed Successfully!")
    print(f"  - Total Generated Frames: {len(coco_images)}")
    print(f"  - Total Object Annotations: {len(coco_annotations)}")
    print(f"  - Total Extracted Crops: {len(crops_records)}")
    print(f"  - Manifest CSV: {EXPANSION_DIR / 'dataset_manifest.csv'}")
    print(f"  - Crops CSV: {EXPANSION_DIR / 'crops_metadata.csv'}")
    print(f"  - COCO JSON: {EXPANSION_DIR / 'annotations.json'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Astramis Synthetic Data Expansion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    run_expansion(seed=args.seed)
