#!/usr/bin/env python3
"""
Generate Candidate v29:
Unified Champion (v28c) + Adaptive Margin Hedging + Border Noise Pruning.

Sprint 9 Technical Improvements:
1. Adaptive Margin Hedging:
   - In v28c, the 2nd hypothesis was set to flat 0.30 across all frames.
   - For frames that are unambiguously solid (e.g. p_bulk=0.001), setting a fake bulk
     hypothesis to 0.30 pollutes the bulk precision curve and causes false positive penalties.
   - Instead, adapt the 2nd hypothesis score based on classification margin:
     margin = |p_bulk - (1 - p_bulk)|
     s_hedge = min(top_score - 0.001, max(0.12, 0.33 - 0.21 * margin))
   - On ambiguous frames (margin < 0.2), s_hedge stays high (~0.30-0.33) to rescue recall.
   - On certain frames (margin > 0.9), s_hedge drops to ~0.12-0.14, staying below all real objects.

2. Table Edge / Border Noise Pruning:
   - Filters out camera mount artifacts (x2 < 120 and y2 < 220).

3. Preserves all verified physics invariants from v28c:
   - DINO+CLIP material reclassifications
   - Bulk wood 0.25x volume scaling
   - Glass strictly solid rank priority
   - Strict 500g physical take consistency
"""

import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SUBMISSIONS = ROOT / "submissions"
WORK = ROOT / "work"

if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
if str(ROOT / "fusion") not in sys.path:
    sys.path.insert(0, str(ROOT / "fusion"))
import submission_tools

BASE_V28C = SUBMISSIONS / "submission_v28c_unified_hedge_optimal.csv"
OUT_V29 = SUBMISSIONS / "submission_v29_adaptive_hedge.csv"
PROBS_CSV = WORK / "champion_probs" / "champion_test_probs.csv"
TELEGRAM_DIR = Path("/home/conexl/Downloads/Telegram Desktop")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_probs():
    probs = {}
    if PROBS_CSV.exists():
        with open(PROBS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                probs[(row["image_id"], int(row["hyp_idx"]))] = float(row["p_bulk"])
    return probs


def generate():
    probs = load_probs()
    print(f"Loaded {len(probs)} probability entries from {PROBS_CSV}")

    with open(BASE_V28C, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    out_rows = []
    adapted_count = 0
    pruned_count = 0

    for img_id, pred_str in rows:
        pred_str = pred_str.strip()
        if not pred_str or pred_str == "none":
            out_rows.append((img_id, "none"))
            continue

        raw_objs = [obj.split() for obj in pred_str.split(";")]
        valid_objs = []
        for obj in raw_objs:
            if len(obj) == 10:
                x1, y1, x2, y2 = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])
                # Filter camera mount artifacts
                if x2 < 120 and y2 < 220:
                    pruned_count += 1
                    continue
                valid_objs.append(obj)

        if not valid_objs:
            out_rows.append((img_id, "none"))
            continue

        valid_objs.sort(key=lambda s: float(s[4]), reverse=True)

        if len(valid_objs) >= 2:
            top_score = float(valid_objs[0][4])
            p_b = probs.get((img_id, 0), 0.5)
            margin = abs(p_b - (1.0 - p_b))
            
            # Adaptive hedge score
            adaptive_target = round(min(top_score - 0.001, max(0.12, 0.33 - 0.21 * margin)), 4)
            valid_objs[1][4] = f"{adaptive_target:.4f}"
            adapted_count += 1

        # Keep at most 2 hypotheses
        valid_objs = valid_objs[:2]
        pred_out = ";".join(" ".join(o) for o in valid_objs)
        out_rows.append((img_id, pred_out))

    with open(OUT_V29, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(out_rows)

    print(f"Generated {OUT_V29}")
    print(f"  Frames: {len(out_rows)}")
    print(f"  Empty frames: {sum(1 for _, p in out_rows if p == 'none')}")
    print(f"  Adapted hedge frames: {adapted_count}")
    print(f"  Pruned border noise: {pruned_count}")

    # Validate schema
    submission_tools.validate_csv(OUT_V29)
    sha = sha256_file(OUT_V29)
    print(f"  SHA-256: {sha}")

    if TELEGRAM_DIR.exists():
        t_out = TELEGRAM_DIR / "submission_v29_adaptive_hedge.csv"
        with open(t_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(out_rows)
        print(f"  Copied to Telegram: {t_out}")

    return OUT_V29


if __name__ == "__main__":
    generate()
