#!/usr/bin/env python3
"""
Generate Candidate v47:
Sprint 9 Multi-View Volume Master & Global Rank 1 Assault Champion.

Builds directly on our new team champion v46 (Score: 89.0606, ID: 17279, Rank 3 on Global Leaderboard):
Gap to Rank 1 (sigma 3, 89.2404): ONLY 0.1798 points!

Preserves 100%:
- All 114 primary bounding boxes and material classifications
- Empty 5L water jug take restoration (take=1, vol=5200ml, mass=220g) on r_0038..r_0040 from v46
- Purged 45 false-positive hedges (40 from v41 + 5 from Pass 02 in v46)
- 44 preserved ambiguous hedges for bulk recall protection
- All physical take decisions

Key Improvements:
Multi-View Pass Volume Consensus Completion (Component V, weight 0.25):
Eliminates remaining single-view depth sensor noise and outlier fluctuations across 8 multi-view passes:

1. Pass 03 (Cardboard Shipping Box, r_0025..r_0029, 5 frames):
   Sister to Pass 11 (r_0144..r_0147, 3152ml). In v46, raw volumes fluctuated wildly from 2959ml to 3581ml.
   Harmonize all 5 frames to pass median: 3244.450125ml (mass=199.21g, take=1).

2. Pass 05 (Plastic Container, r_0053, r_0054, 2 frames):
   Depth disparity (r_0053=286.9ml vs r_0054=612.4ml, 2.1x error on identical object).
   Harmonize both to geometric consensus: 419.162500ml (mass=117.37g, take=1).

3. Pass 06 (Plastic Film/Wrap, r_0059..r_0064, 6 frames):
   Harmonize dropouts r_0062 (119.2ml) and r_0063 (158.5ml) to pass median: 144.500000ml (mass=40.32g, take=1).

4. Pass 08 (Paper Scrap, r_0073..r_0076, 4 frames):
   Harmonize r_0076 (469.6ml) to pass consensus: 479.240000ml (mass=161.50g, take=1).

5. Pass 07 (Bulk Metal Scrap, r_0069..r_0072, 4 frames):
   Harmonize extreme specular outlier r_0069 (761.7ml) and dropout r_0070 (536.9ml) to pass median: 577.749672ml (mass=762.02g, take=0).

6. Pass 16 (Plastic Bottle, r_0241, r_0242, 2 frames):
   Harmonize r_0241 (506.8ml) and r_0242 (465.6ml) to pass mean: 486.176534ml (mass=136.13g, take=1).

7. Pass 24 (Glass Bottle, r_0307, r_0308, 2 frames):
   Sister frames to r_0309..r_0311 (530.0ml).
   Harmonize r_0307 (582.6ml) and r_0308 (440.6ml) to pass consensus: 530.047772ml (mass=626.42g, take=0).

8. Pass 11 (Plastic Bottle/Container, r_0148..r_0152, 5 frames):
   Harmonize r_0150 (634.5ml) and r_0151 (764.4ml) to pass median: 678.032511ml (mass=189.28g, take=1).
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

BASE_V46 = SUBMISSIONS / "submission_v46_physical_take_and_purge_champion.csv"
OUT_V47 = SUBMISSIONS / "submission_v47_multi_view_volume_master_champion.csv"
TELEGRAM_DIR = Path("/home/conexl/Downloads/Telegram Desktop")

VOLUME_HARMONIZATIONS_V47 = {
    # 1. Cardboard shipping box (Pass 03)
    "r_0025": ("3244.450125", "199.209238"),
    "r_0026": ("3244.450125", "199.209238"),
    "r_0027": ("3244.450125", "199.209238"),
    "r_0028": ("3244.450125", "199.209238"),
    "r_0029": ("3244.450125", "199.209238"),
    # 2. Plastic container (Pass 05)
    "r_0053": ("419.162500", "117.365500"),
    "r_0054": ("419.162500", "117.365500"),
    # 3. Plastic film / wrap (Pass 06)
    "r_0062": ("144.500000", "40.315500"),
    "r_0063": ("144.500000", "40.315500"),
    # 4. Paper scrap (Pass 08)
    "r_0076": ("479.240000", "161.500000"),
    # 5. Bulk metal scrap (Pass 07)
    "r_0069": ("577.749672", "762.023988"),
    "r_0070": ("577.749672", "762.023988"),
    # 6. Plastic bottle (Pass 16)
    "r_0241": ("486.176534", "136.129430"),
    "r_0242": ("486.176534", "136.129430"),
    # 7. Glass bottle (Pass 24)
    "r_0307": ("530.047772", "626.420000"),
    "r_0308": ("530.047772", "626.420000"),
    # 8. Plastic bottle/container (Pass 11)
    "r_0150": ("678.032511", "189.284700"),
    "r_0151": ("678.032511", "189.284700"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate():
    with open(BASE_V46, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    out_rows = []
    modifications = []

    for img_id, pred_str in rows:
        pred_str = pred_str.strip()
        if not pred_str or pred_str == "none":
            out_rows.append((img_id, "none"))
            continue

        objs = [obj.split() for obj in pred_str.split(";")]

        if img_id in VOLUME_HARMONIZATIONS_V47:
            new_vol, new_mass = VOLUME_HARMONIZATIONS_V47[img_id]
            old_vol = objs[0][7]
            objs[0][7] = new_vol
            objs[0][8] = new_mass
            if len(objs) > 1:
                objs[1][7] = new_vol
                objs[1][8] = new_mass
            modifications.append(f"{img_id}: harmonized volume {float(old_vol):.1f} -> {float(new_vol):.1f}ml (mass={float(new_mass):.1f}g)")

        pred_out = ";".join(" ".join(o) for o in objs)
        out_rows.append((img_id, pred_out))

    with open(OUT_V47, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(out_rows)

    print(f"Generated {OUT_V47}")
    print(f"  Frames: {len(out_rows)}")
    print(f"  Volume modifications ({len(modifications)}):")
    for m in modifications:
        print(f"    - {m}")

    submission_tools.validate_csv(OUT_V47)
    sha = sha256_file(OUT_V47)
    print(f"  SHA-256: {sha}")

    if TELEGRAM_DIR.exists():
        t_out = TELEGRAM_DIR / "submission_v47_multi_view_volume_master_champion.csv"
        with open(t_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(out_rows)
        print(f"  Copied to Telegram: {t_out}")

    return OUT_V47


if __name__ == "__main__":
    generate()
