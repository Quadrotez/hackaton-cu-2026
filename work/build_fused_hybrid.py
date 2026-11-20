"""Build a verified, offline fused/hybrid test CSV without overwriting prior submissions."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import pandas as pd
from threadpoolctl import threadpool_limits

from cv_volume_v2 import GEOMETRY_PATH
from form_refine import select_model
from material_split import DATA, WORK
from predict_pipeline import DEFAULT_BUNDLE, PredictPipeline
from submission_tools import HEADER, test_ids, validate_csv
from volume_hybrid_eval import audit_holdout, collect, variant_predictions
from v2_data import ALL_FEATURES


def serialize(objects):
    return ";".join(
        f"{' '.join(f'{x:.8g}' for x in p['box'])} {p['score']:.8g} {p['form']} {p['material']} "
        f"{p['volume']:.8g} {p['mass']:.8g} {p['take']}"
        for p in objects
    ) or "none"


def compare(base, hybrid, reference):
    counts = Counter()
    with reference.open(newline="", encoding="utf-8") as f:
        expected = {row["image_id"]: row["prediction"] for row in csv.DictReader(f)}
    if base.keys() != hybrid.keys() or base.keys() != expected.keys():
        raise ValueError("Frame IDs differ from the prior fused export")
    for i, pp in base.items():
        if serialize(pp) != expected[i]:
            raise ValueError(f"{i}: the reproduced fused/CB baseline differs from its export")
        for p, q in zip(pp, hybrid[i], strict=True):
            if any(p[k] != q[k] for k in ("box", "score", "form", "material")):
                raise ValueError(f"{i}: the hybrid must not change detection or material")
            if not (q["volume"] > 0 and q["mass"] > 0):
                raise ValueError(f"{i}: nonpositive mass or volume")
            changed = any(p[k] != q[k] for k in ("volume", "mass", "take"))
            if changed and p["form"] != "solid":
                raise ValueError(f"{i}: the hybrid modified a bulk object")
            counts[p["form"]] += 1
            counts["changed_solid"] += changed
            counts["changed_take"] += p["take"] != q["take"]
    if not counts["changed_solid"]:
        raise ValueError("Hybrid predictions did not change any solid objects")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "submission_fused_hybrid.csv")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    weights = WORK / "weights" / "best.pt"
    _, _, train_form, _ = audit_holdout(weights, DEFAULT_BUNDLE.with_name("compact.holdout.pkl"))
    labeled = pd.read_csv(ALL_FEATURES)
    train_volume = labeled.merge(pd.read_csv(GEOMETRY_PATH),
                                 on=["image_id", "ann_id"], validate="one_to_one")
    if len(train_volume) != len(labeled):
        raise ValueError("Missing labeled training geometry")

    cv2.setNumThreads(1)
    frames = [(i, DATA / "real" / "test" / i, ()) for i in test_ids()]
    with threadpool_limits(3):
        form_model = select_model(train_form)
        pipeline = PredictPipeline(weights=weights)
        rows, found = collect(pipeline, frames)
        all_predictions = dict(variant_predictions(pipeline, rows, found, train_volume, form_model))
    hybrid = all_predictions["hybrid"]
    stats = compare(all_predictions["CB-residual-d3"], hybrid,
                    WORK / "form_candidates" / "submission_fused.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    f = NamedTemporaryFile(dir=args.out.parent, mode="w", newline="", encoding="utf-8", delete=False)
    temp = Path(f.name)
    try:
        with f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for i, _, _ in frames:
                writer.writerow([i, serialize(hybrid[i])])
        validate_csv(temp)
        args.out.hardlink_to(temp)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Fused baseline reproduced exactly; compared {dict(stats)}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
