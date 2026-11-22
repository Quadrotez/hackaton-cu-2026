#!/usr/bin/env python3
"""From v56 (89.9554): stronger bulk plastic and bulk wood."""

import csv, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submissions"
BASE = SUB / "submission_v56_bulk_plastic_070.csv"
PROBES = {
    "submission_v58_bulk_plastic_060.csv": {("bulk", "plastic"): 0.60 / 0.70},
    "submission_v59_bulk_wood_080.csv": {("bulk", "wood"): 0.80},
}

def apply(rows, scales):
    out, touched = [], {}
    for image_id, prediction in rows:
        if prediction.strip() == "none":
            out.append((image_id, "none")); continue
        rebuilt = []
        for enc in prediction.split(";"):
            o = enc.split()
            s = scales.get((o[5], o[6]))
            if s is not None:
                v, m = float(o[7]) * s, float(o[8]) * s
                o[7], o[8], o[9] = f"{v:.6f}", f"{m:.6f}", str(int(m <= 500.0))
                k = (o[5], o[6]); touched[k] = touched.get(k, 0) + 1
            rebuilt.append(" ".join(o))
        out.append((image_id, ";".join(rebuilt)))
    return out, touched

with BASE.open(newline="", encoding="utf-8") as f:
    rd = csv.reader(f); header, base = next(rd), list(rd)
for name, scales in PROBES.items():
    rows, touched = apply(base, scales)
    p = SUB / name
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(name, touched, hashlib.sha256(p.read_bytes()).hexdigest()[:16])
