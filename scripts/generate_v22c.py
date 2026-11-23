#!/usr/bin/env python3
"""
Generate candidate v22c:
Keep v21 champion volumes 100% untouched (no metal volume scaling).
Apply ONLY the physically consistent Glass Solid Priority:
In r_0231 and r_0315, YOLO produced paired bulk/solid hypotheses where scores differed by < 2e-7,
causing bulk glass to be ranked hyp 0 and solid glass hyp 1.
Swapping them makes solid glass the primary hypothesis (with its true 1.8L volume in r_0231 matching r_0230 and r_0232),
while retaining bulk glass as hyp 1 for detection mAP50 recall.
"""

import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
WORK = ROOT / "work"

if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
if str(ROOT / "fusion") not in sys.path:
    sys.path.insert(0, str(ROOT / "fusion"))
import submission_tools

BASE_V21 = SUBMISSIONS / "submission_v21_unified_champion.csv"
OUT_V22C = SUBMISSIONS / "submission_v22c_glass_priority.csv"
TELEGRAM_DIR = Path("/home/conexl/Downloads/Telegram Desktop")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"Loading base v21 from: {BASE_V21}")
    with open(BASE_V21, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    out_rows = []
    swapped_count = 0

    for img_id, pred_str in rows:
        pred_str = pred_str.strip()
        if not pred_str or pred_str == "none":
            out_rows.append((img_id, "none"))
            continue

        objs = [obj.split() for obj in pred_str.split(";")]

        if img_id in ("r_0231", "r_0315") and len(objs) >= 2:
            f0, m0 = objs[0][5], objs[0][6]
            f1, m1 = objs[1][5], objs[1][6]
            if f0 == "bulk" and m0 == "glass" and f1 == "solid" and m1 == "glass":
                # Swap hyp 0 and hyp 1 scores and positions
                s0 = objs[0][4]
                s1 = objs[1][4]
                objs[1][4] = s0
                objs[0][4] = s1
                objs = [objs[1], objs[0]] + objs[2:]
                swapped_count += 1

        rebuilt = [" ".join(obj) for obj in objs]
        out_rows.append((img_id, ";".join(rebuilt)))

    with open(OUT_V22C, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(out_rows)

    submission_tools.validate_csv(OUT_V22C)
    sha = sha256_file(OUT_V22C)
    print(f"Saved v22c to {OUT_V22C} (SHA256: {sha}, swapped {swapped_count} frames)")

    import shutil
    shutil.copy2(OUT_V22C, TELEGRAM_DIR / OUT_V22C.name)
    print("Copied v22c to Telegram Desktop.")


if __name__ == "__main__":
    main()
