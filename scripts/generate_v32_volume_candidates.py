#!/usr/bin/env python3
"""Volume-calibration candidates derived from the scored v31 champion.

These variants preserve every v31 detection, material/form label and confidence.
Only real-data category volume ratios are applied; bulk wood is deliberately
excluded because v31 already includes its established x0.25 correction.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions" / "submission_v31_grand_champion.csv"

NARROW = {("bulk", "metal"): 1.35}
WIDE = {
    ("bulk", "metal"): 1.35,
    ("solid", "glass"): 1.16,
    ("solid", "metal"): 0.70,
    ("solid", "wood"): 0.32,
}


def write_variant(name: str, scales: dict[tuple[str, str], float], physical: bool) -> None:
    destination = ROOT / "submissions" / name
    changed = take_changed = 0
    with BASE.open(newline="", encoding="utf-8") as source, destination.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader, writer = csv.reader(source), csv.writer(target)
        writer.writerow(next(reader))
        for image_id, prediction in reader:
            output = []
            for encoded in prediction.split(";"):
                item = encoded.split()
                scale = scales.get((item[5], item[6])) if len(item) == 10 else None
                if scale is not None:
                    item[7] = repr(float(item[7]) * scale)
                    if physical:
                        mass = float(item[8]) * scale
                        new_take = int(mass <= 500.0)
                        take_changed += new_take != int(item[9])
                        item[8], item[9] = repr(mass), str(new_take)
                    changed += 1
                output.append(" ".join(item))
            writer.writerow((image_id, ";".join(output)))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"{name}: changed={changed}, take_changed={take_changed}, sha256={digest}")


def main() -> None:
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    write_variant("submission_v32a_v31_bulk_metal_135.csv", NARROW, True)
    write_variant("submission_v32b_v31_multiclass_physical.csv", WIDE, True)
    write_variant("submission_v32c_v31_multiclass_volume_only.csv", WIDE, False)


if __name__ == "__main__":
    main()
