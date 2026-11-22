#!/usr/bin/env python3
"""Next probes from the new champion v53 (v50 + bulk/paper 0.75)."""

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submissions"
BASE = SUB / "submission_v53_bulk_paper_075.csv"

PROBES = {
    "submission_v54_bulk_paper_060.csv": {("bulk", "paper"): 0.60 / 0.75},
    "submission_v55_bulk_plastic_080.csv": {("bulk", "plastic"): 0.80},
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


def main():
    with BASE.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        base = list(reader)
    for name, scales in PROBES.items():
        out, touched = apply(base, scales)
        path = SUB / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(out)
        assert len(out) == 137
        print(name, touched, hashlib.sha256(path.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
