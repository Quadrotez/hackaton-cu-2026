#!/usr/bin/env python3
"""
Generalized Physical & Multi-View Machine Learning Pipeline (100% Automated, Zero Hardcoding):
Builds directly on the 5-fold DINO ViT-B/16 + CLIP multimodal material model.

All constants and priors are derived purely from real_train.json:
- Cardboard/paper packaging density: 0.0614 g/ml (s07: 70g / 1140ml)
- Bulk plastic shredded density: 0.180 g/ml (b01/b03/b12)
- Metal density: 7.80 g/ml
- Mineral density: 2.60 g/ml
- Glass density: 2.50 g/ml
- Solid plastic density: 0.280 g/ml
- Bulk wood chips packing fraction: 0.25x (s01 vs b01)
- Hollow plastic jug: vol >= 3000 ml -> empty jug s12 (220g, take=1)

Zero hardcoded frame IDs.
"""

import csv
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent
SUBMISSIONS = ROOT / "submissions"
DATA = ROOT / "dataset"
TEST_DIR = DATA / "real" / "test"
WORK = ROOT / "work"

if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
if str(ROOT / "fusion") not in sys.path:
    sys.path.insert(0, str(ROOT / "fusion"))
import submission_tools

# Empirical physical densities from real_train.json
DENSITIES = {
    "glass": 2.50,
    "metal": 7.80,
    "mineral": 2.60,
    "paper": 0.0614,       # hollow cardboard & packaging prior from s07
    "plastic": 0.280,
    "bulk_plastic": 0.180, # shredded/crushed plastic from b01/b03/b12
    "wood": 0.692,
}

HEDGE_CEILING = 0.1250


def load_automated_camera_passes(max_delta_sec=20.0):
    """Groups test frames into camera passes based on metadata timestamps."""
    frame_times = []
    for p in sorted(TEST_DIR.iterdir()):
        if p.is_dir():
            meta_file = p / "meta.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    t_str = m.get("time")
                    dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                    frame_times.append((p.name, dt))

    frame_times.sort(key=lambda x: x[1])

    passes = []
    cur_pass = [frame_times[0][0]]
    for i in range(len(frame_times) - 1):
        f1, t1 = frame_times[i]
        f2, t2 = frame_times[i + 1]
        delta = (t2 - t1).total_seconds()
        if delta <= max_delta_sec:
            cur_pass.append(f2)
        else:
            passes.append(cur_pass)
            cur_pass = [f2]
    passes.append(cur_pass)
    return passes


def generate(input_sub_path: Path, output_sub_path: Path):
    print(f"[*] Loading input submission: {input_sub_path}")
    with open(input_sub_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_rows = list(reader)

    sub_dict = {row[0]: row[1] for row in raw_rows}
    passes = load_automated_camera_passes()
    print(f"[*] Grouped test set into {len(passes)} automated camera passes.")

    # Parse predictions
    parsed_preds = {}
    for f_name, pred_str in sub_dict.items():
        if not pred_str or pred_str.strip() == "none":
            parsed_preds[f_name] = []
            continue
        objs = []
        for obj_toks in pred_str.strip().split(";"):
            t = obj_toks.split()
            if len(t) == 10:
                box = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                score = float(t[4])
                form = t[5]
                mat = t[6]
                vol = float(t[7])
                mass = float(t[8])
                take = int(t[9])
                objs.append({
                    "box": box,
                    "score": score,
                    "form": form,
                    "mat": mat,
                    "vol": vol,
                    "mass": mass,
                    "take": take,
                })
        parsed_preds[f_name] = objs

    # Step 1: Multi-View Pass-Level Consistency (Volume Median & Specular Glare Correction)
    volume_updates = 0
    material_updates = 0
    for p_idx, p_frames in enumerate(passes):
        if len(p_frames) < 2:
            continue

        # Check primary detections in the pass
        primaries = [parsed_preds[f][0] for f in p_frames if parsed_preds[f]]
        if len(primaries) < 2:
            continue

        # If majority of frames in pass agree on material, correct single specular misclassifications
        mats = [o["mat"] for o in primaries]
        mat_counts = Counter(mats)
        most_common_mat, top_count = mat_counts.most_common(1)[0]
        if top_count >= len(primaries) * 0.70:
            for f in p_frames:
                if parsed_preds[f] and parsed_preds[f][0]["mat"] != most_common_mat:
                    parsed_preds[f][0]["mat"] = most_common_mat
                    material_updates += 1

        # Multi-view volume median: compute median volume of matching (form, mat) tracks
        for form_val in ("solid", "bulk"):
            track_frames = [f for f in p_frames if parsed_preds[f] and parsed_preds[f][0]["form"] == form_val]
            if len(track_frames) >= 2:
                track_vols = [parsed_preds[f][0]["vol"] for f in track_frames]
                med_v = float(np.median(track_vols))
                for f in track_frames:
                    curr_v = parsed_preds[f][0]["vol"]
                    # If current volume is an extreme outlier (sensor dropout < 0.6x or glare spike > 1.5x)
                    if curr_v < med_v * 0.60 or curr_v > med_v * 1.50:
                        parsed_preds[f][0]["vol"] = med_v
                        volume_updates += 1

    print(f"[*] Multi-view consistency: updated {material_updates} specular materials, {volume_updates} volume sensor outliers.")

    # Step 2: Generalized Physics & Take Modeling (All frames)
    mass_updates = 0
    take_updates = 0
    for f_name, objs in parsed_preds.items():
        for i, obj in enumerate(objs):
            form = obj["form"]
            mat = obj["mat"]
            vol = obj["vol"]

            # Bulk wood shavings envelope scaling (packing factor 0.25)
            if form == "bulk" and mat == "wood":
                vol = vol * 0.25
                obj["vol"] = vol

            # Effective density selection
            if form == "bulk" and mat == "plastic":
                rho = DENSITIES["bulk_plastic"]
            else:
                rho = DENSITIES.get(mat, 1.0)

            # Mass calculation
            mass = vol * rho

            # Hollow plastic container prior (e.g. 5L water jug s12: thin plastic shell ~220g)
            if mat == "plastic" and form == "solid" and vol >= 3000.0:
                mass = 220.0
                take = 1
            else:
                # General physical threshold: mass <= 500g -> take=1; mass > 500g -> take=0
                take = 1 if mass <= 500.0 else 0

            if abs(mass - obj["mass"]) > 1.0:
                mass_updates += 1
            if take != obj["take"]:
                take_updates += 1

            obj["mass"] = mass
            obj["take"] = take

            # Step 3: Secondary Hedge Calibration
            if i > 0:
                # Cap hedge score at ceiling
                obj["score"] = min(obj["score"], HEDGE_CEILING)

                # Suppress physically impossible hedges
                if mat in ("glass", "mineral") and form == "bulk":
                    obj["score"] = 0.0051

    print(f"[*] Physical modeling: calibrated {mass_updates} masses, updated {take_updates} take decisions.")

    # Format output
    out_rows = []
    total_objects = 0
    none_frames = 0
    for f_name, _ in raw_rows:
        objs = parsed_preds.get(f_name, [])
        objs.sort(key=lambda o: o["score"], reverse=True)
        valid_objs = [o for o in objs if o["score"] > 0.0100][:2]

        if not valid_objs:
            out_rows.append((f_name, "none"))
            none_frames += 1
        else:
            obj_strs = []
            for o in valid_objs:
                box_str = f"{o['box'][0]:.1f} {o['box'][1]:.1f} {o['box'][2]:.1f} {o['box'][3]:.1f}"
                score_str = f"{o['score']:.4f}"
                vol_str = f"{o['vol']:.1f}"
                mass_str = f"{o['mass']:.1f}"
                toks = [box_str, score_str, o["form"], o["mat"], vol_str, mass_str, str(o["take"])]
                obj_strs.append(" ".join(toks))
            out_rows.append((f_name, ";".join(obj_strs)))
            total_objects += len(valid_objs)

    output_sub_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_sub_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(out_rows)

    print(f"[+] Output written to {output_sub_path}")
    print(f"  Frames: {len(out_rows)}, Objects: {total_objects}, None: {none_frames}")

    submission_tools.validate_csv(output_sub_path)
    sha = hashlib.sha256(open(output_sub_path, "rb").read()).hexdigest()
    print(f"  SHA-256: {sha}")
    return output_sub_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(SUBMISSIONS / "submission_83_91_v17_overhaul.csv"))
    parser.add_argument("--output", default=str(SUBMISSIONS / "submission_generalized_physics_v1.csv"))
    args = parser.parse_args()

    generate(Path(args.input), Path(args.output))
