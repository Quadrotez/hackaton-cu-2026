import json
import math
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from material_split import DATA, PARTS, SPLIT_PATH, WORK, load_frames


TARGET_COLUMNS = ("object_id", "form", "material", "volume_ml", "mass_g", "take")
METADATA_COLUMNS = ("split", "image_id", "ann_id", *TARGET_COLUMNS, "source")
CROP_SIZE = (128, 128)


def bounds(bbox, width, height, padding=0.0):
    x, y, w, h = bbox
    return (
        max(0, math.floor(x - w * padding)),
        max(0, math.floor(y - h * padding)),
        min(width, math.ceil(x + w * (1 + padding))),
        min(height, math.ceil(y + h * (1 + padding))),
    )


def polygon_mask(segmentation, width, height, x0, y0):
    mask = np.zeros((height, width), dtype=np.uint8)
    offset = np.array([x0, y0], dtype=np.int32)
    for polygon in segmentation:
        points = np.rint(np.array(polygon).reshape(-1, 2)).astype(np.int32) - offset
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def color_features(bgr, mask):
    rgb = bgr[mask][:, ::-1]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[mask]
    features = {}
    for values, names in ((rgb, "rgb"), (hsv, "hsv")):
        for channel, mean, std in zip(names, values.mean(axis=0), values.std(axis=0)):
            features[f"{channel}_mean"] = mean
            features[f"{channel}_std"] = std
    for values, key, bins, upper in zip(hsv.T, "hsv", (18, 8, 8), (180, 256, 256)):
        hist, _ = np.histogram(values, bins=bins, range=(0, upper))
        features.update({f"{key}_hist_{i:02d}": count / values.size for i, count in enumerate(hist)})
    return features


def summary(values, prefix):
    keys = ("mean", "std", "p10", "p50", "p90")
    if values.size:
        stats = [values.mean(), values.std(), *np.percentile(values, [10, 50, 90])]
    else:
        stats = [np.nan] * len(keys)
    return {f"{prefix}_{key}": value for key, value in zip(keys, stats)}


def depth_features(depth_mm, mask, plane, camera, x0, y0):
    depth = depth_mm.astype(np.float32)
    valid = mask & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[v, u] / 1000.0
    nx, ny, nz, d = plane
    x = (u + x0 - camera["cx"]) / camera["fx"] * z
    y = (v + y0 - camera["cy"]) / camera["fy"] * z
    height_m = nx * x + ny * y + nz * z + d

    horizontal = valid[:, :-1] & valid[:, 1:]
    vertical = valid[:-1, :] & valid[1:, :]
    gradients = np.concatenate((
        np.abs(np.diff(depth, axis=1))[horizontal],
        np.abs(np.diff(depth, axis=0))[vertical],
    ))
    return {
        "depth_zero_share": 1.0 - valid.sum() / mask.sum(),
        **summary(height_m, "height_m"),
        **summary(gradients, "depth_gradient_mm"),
    }


def region_features(color, depth, bbox, mask, meta, camera, x0, y0):
    height, width = color.shape[:2]
    y1, x1 = y0 + mask.shape[0], x0 + mask.shape[1]
    bx0, by0, bx1, by1 = bounds(bbox, width, height)
    tight = mask[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
    return {
        **color_features(color[y0:y1, x0:x1], mask),
        **depth_features(depth[y0:y1, x0:x1], mask, meta["floor_plane"], camera, x0, y0),
        "aspect_ratio": bbox[2] / bbox[3],
        "mask_area_px": int(mask.sum()),
        "mask_bbox_share": tight.mean(),
    }


def extract_annotation(color, depth, ann, meta, camera):
    height, width = color.shape[:2]
    x0, y0, x1, y1 = bounds(ann["bbox"], width, height, padding=0.1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Annotation {ann['id']} has an empty padded bbox")
    mask = polygon_mask(ann["segmentation"], x1 - x0, y1 - y0, x0, y0)
    if not mask.any():
        raise ValueError(f"Annotation {ann['id']} has an empty polygon mask")

    bgr = color[y0:y1, x0:x1]
    masked_color = np.where(mask[..., None], bgr, np.uint8(0))
    interpolation = cv2.INTER_AREA if max(mask.shape) > 128 else cv2.INTER_LINEAR
    crop = cv2.resize(masked_color, CROP_SIZE, interpolation=interpolation)

    return crop, region_features(color, depth, ann["bbox"], mask, meta, camera, x0, y0)


def build_features():
    cv2.setNumThreads(1)
    plan = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    image_parts = {image_id: p for p in PARTS for image_id in plan[p]["image_ids"]}
    objects = {p: set(plan[p]["object_ids"]) for p in PARTS}
    excluded = {
        (a["image_id"], a["ann_id"]): a["object_id"] for a in plan["excluded_annotations"]
    }
    camera = json.loads((DATA / "camera.json").read_text(encoding="utf-8"))
    rows = []
    for source, image_id, image, anns in load_frames():
        part = image_parts[image_id]
        selected = []
        for ann in anns:
            obj = ann["attributes"]["object_id"]
            key = image_id, ann["id"]
            if key in excluded:
                assert part == "val" and obj in objects["train"] and excluded.pop(key) == obj
            else:
                assert obj in objects[part], (image_id, ann["id"], obj, part)
                selected.append(ann)
        if not selected:
            continue

        color_path = DATA / image["file_name"]
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(color_path.with_name("depth.png")), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None or color.shape[:2] != depth.shape or depth.dtype != np.uint16:
            raise ValueError(f"Invalid color/depth pair for {image_id}")
        meta = json.loads(color_path.with_name("meta.json").read_text(encoding="utf-8"))
        for ann in selected:
            attrs = ann["attributes"]
            crop, features = extract_annotation(color, depth, ann, meta, camera)
            crop_dir = WORK / "crops" / part / attrs["material"]
            crop_dir.mkdir(parents=True, exist_ok=True)
            path = crop_dir / f"{image_id}_{ann['id']}.png"
            if not cv2.imwrite(str(path), crop):
                raise OSError(f"Could not write {path}")
            rows.append({
                "split": part,
                "image_id": image_id,
                "ann_id": ann["id"],
                **{key: attrs[key] for key in TARGET_COLUMNS},
                "source": source,
                **features,
            })

    assert not excluded, excluded
    df = pd.DataFrame(rows)
    assert not df.duplicated(["image_id", "ann_id"]).any()
    for part in PARTS:
        subset = df.loc[df["split"] == part]
        assert set(subset["object_id"]) == objects[part]
        expected = {s: c["annotations"] for s, c in plan[part]["counts_by_source"].items()}
        assert Counter(subset["source"]) == expected
        print(f"{part}: {len(subset)} crops; materials: {dict(Counter(subset['material']))}")
    df.to_csv(WORK / "features.csv", index=False)
    print(f"Features: {len(df)} rows, {len(df.columns) - len(METADATA_COLUMNS)} numeric predictors")
    print(f"Masks without any depth: {(df['depth_zero_share'] == 1).sum()}")


if __name__ == "__main__":
    build_features()
