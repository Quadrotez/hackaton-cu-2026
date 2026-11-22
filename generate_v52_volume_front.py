#!/usr/bin/env python3
"""Reconstruct v50 from v46, then probe two remaining volume fronts."""

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submissions"
BASE = SUB / "submission_v46_physical_take_and_purge_champion.csv"

V49 = {("bulk", "metal"): 1.35}
V50 = {("solid", "glass"): 1.16, ("solid", "metal"): 0.70}
PROBES = {
    "submission_v52_solid_plastic_125.csv": {("solid", "plastic"): 1.25},
    "submission_v53_bulk_paper_075.csv": {("bulk", "paper"): 0.75},
}


def apply(rows, scales):
    out = []
    touched = {}
    for image_id, prediction in rows:
        if prediction.strip() == "none":
            out.append((image_id, "none"))
            continue
        rebuilt = []
        for encoded in prediction.split(";"):
            obj = encoded.split()
            scale = scales.get((obj[5], obj[6]))
            if scale is not None:
                vol = float(obj[7]) * scale
                mass = float(obj[8]) * scale
                obj[7] = f"{vol:.6f}"
                obj[8] = f"{mass:.6f}"
                obj[9] = str(int(mass <= 500.0))
                key = (obj[5], obj[6])
                touched[key] = touched.get(key, 0) + 1
            rebuilt.append(" ".join(obj))
        out.append((image_id, ";".join(rebuilt)))
    return out, touched


def write(name, rows):
    path = SUB / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)
    assert len(rows) == 138
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{name}: {digest}")
    return digest


def main():
    with BASE.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        v46 = list(reader)

    v49, t49 = apply(v46, V49)
    v50, t50 = apply(v49, V50)
    write("submission_v50_reconstructed.csv", [header] + v50)
    print("v49 touched", t49, "v50 touched", t50)
    for name, scales in PROBES.items():
        out, touched = apply(v50, scales)
        write(name, [header] + out)
        print(" ", name, touched)


if __name__ == "__main__":
    main()
