"""Shared holdout and strict format checks for the local submissions."""

import argparse
import csv
import math
import sys
from pathlib import Path

from material_split import DATA, WORK, load_frames

sys.path.append(str(WORK.parent))
import score as metric


HEADER = ["image_id", "prediction"]
SAMPLE = WORK.parent / "sample_submission.csv"


def baseline_holdout():
    real = [(image_id, (DATA / image["file_name"]).parent, anns)
            for source, image_id, image, anns in load_frames() if source == "real"]
    real = [(image_id, path, anns) for image_id, path, anns in real
            if (path / "depth.png").is_file()
            and any((path / f"color.{suffix}").is_file() for suffix in ("png", "jpg"))]
    objects = sorted({a["attributes"]["object_id"] for _, _, anns in real for a in anns})
    val_ids = set(objects[::5])  # Same selection as the unmodified YOLO baseline.
    return [(image_id, path, anns) for image_id, path, anns in real
            if any(a["attributes"]["object_id"] in val_ids for a in anns)]


def ground_truth(holdout):
    gt = {}
    for image_id, _, anns in holdout:
        gt[image_id] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            at = ann["attributes"]
            gt[image_id].append(dict(box=(x, y, x + w, y + h), form=at["form"],
                                     material=at["material"], volume=at["volume_ml"],
                                     take=int(at["mass_g"] <= metric.HEAVY_G)))
    return gt


def test_ids():
    with SAMPLE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError(f"Unexpected header in {SAMPLE}: {reader.fieldnames}")
        ids = [row["image_id"] for row in reader]
    paths = {p.name for p in (DATA / "real" / "test").iterdir() if p.is_dir()}
    if len(ids) != 137 or len(set(ids)) != 137 or set(ids) != paths:
        raise ValueError("Expected exactly 137 unique test frames matching the sample")
    return ids


def parse_prediction(image_id, text):
    if text == "none":
        return []
    for obj in text.split(";"):
        fields = obj.split()
        if len(fields) != 10:
            raise ValueError(f"{image_id}: expected 10 fields per object or 'none'")
        conf, volume, mass = map(float, (fields[4], fields[7], fields[8]))
        if not all(map(math.isfinite, (conf, volume, mass))) or not 0 <= conf <= 1:
            raise ValueError(f"{image_id}: invalid confidence, volume or mass")
        if volume < 0 or mass < 0 or fields[9] not in ("0", "1"):
            raise ValueError(f"{image_id}: invalid volume, mass or take")
    return metric._parse(text, True, image_id)


def validate_csv(path):
    expected = set(test_ids())
    predictions = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if next(reader, None) != HEADER:
            raise ValueError(f"{path}: expected image_id,prediction header")
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"{path}:{reader.line_num}: expected exactly two columns")
            image_id, text = row
            if image_id not in expected or image_id in predictions:
                raise ValueError(f"{path}: unexpected or duplicate image_id {image_id!r}")
            predictions[image_id] = parse_prediction(image_id, text)
    if missing := expected - predictions.keys():
        raise ValueError(f"{path}: {len(missing)} missing frames, e.g. {sorted(missing)[:5]}")
    empty = sum(not objects for objects in predictions.values())
    objects = sum(map(len, predictions.values()))
    print(f"OK: {path}: {len(predictions)} frames, {objects} objects, {empty} none")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a test submission")
    parser.add_argument("csv", type=Path, nargs="+")
    for path in parser.parse_args().csv:
        validate_csv(path)
