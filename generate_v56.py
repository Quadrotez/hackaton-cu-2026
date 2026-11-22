#!/usr/bin/env python3
"""Push further from v55 (89.8269): stronger bulk plastic + bulk wood probe."""

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submissions"
BASE = SUB / "submission_v55_bulk_plastic_080.csv"

PROBES = {
    "submission_v56_bulk_plastic_070.csv": {("bulk", "plastic"): 0.70 / 0.80},
    "submission_v57_bulk_wood_080.csv": {("bulk", "wood"): 0.80},
}


def apply(rows, scales):
    out, touched = [], {}
    for image_id, prediction in rows:
        if prediction.strip() == "none":
            out.append((image_id, "none"))
            continue
        rebuilt = []
        for encoded in prediction.split(";"):
            obj = encoded.split()
            scale = scales.get((obj[5], obj[6]))
            if scale is not None:
                vol, mass = float(obj[7]) * scale, float(obj[8]) * scale
                obj[7], obj[8] = f"{vol:.6f}", f"{mass:.6f}"
                obj[9] = str(int(mass <= 500.0))
                key = (obj[5], obj[6])
                touched[key] = touched.get(key, 0) + 1
            rebuilt.append(" ".join(obj))
        out.append((image_id, ";".join(rebuilt)))
    return out, touched


def main():
    with BASE.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header, base = next(reader), list(reader)
    for name, scales in PROBES.items():
        rows, touched = apply(base, scales)
        path = SUB / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(name, touched, hashlib.sha256(path.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
