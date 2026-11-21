import json

import cv2
import numpy as np
import pandas as pd

from material_features import polygon_mask
from material_split import DATA, WORK, load_frames


def raw_volume_ml(depth_mm, mask, plane, camera, cell=0.004):
    # Same projection and floor-grid integration as the baseline.
    v, u = np.nonzero(mask & (depth_mm > 0))
    if len(v) < 30:
        return None
    z = depth_mm[v, u] / 1000.0
    pts = np.stack([
        (u - camera["cx"]) / camera["fx"] * z,
        (v - camera["cy"]) / camera["fy"] * z,
        z,
    ], axis=1)
    n, d = np.array(plane[:3]), plane[3]
    height = pts @ n + d
    e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    keys = np.floor(pts @ e1 / cell).astype(np.int64) * 100000 + np.floor(pts @ e2 / cell).astype(np.int64)
    order = np.argsort(keys)
    keys, height = keys[order], height[order]
    starts = np.r_[0, np.nonzero(np.diff(keys))[0] + 1]
    tops = np.maximum.reduceat(height, starts)
    return float(np.clip(tops, 0, None).sum() * cell * cell * 1e6)


def build_raw_volumes(features):
    cv2.setNumThreads(1)
    wanted = set(zip(features["image_id"], features["ann_id"]))
    camera = json.loads((DATA / "camera.json").read_text(encoding="utf-8"))
    raw = []
    for _, image_id, image, anns in load_frames():
        chosen = [a for a in anns if (image_id, a["id"]) in wanted]
        if not chosen:
            continue
        directory = (DATA / image["file_name"]).parent
        depth = cv2.imread(str(directory / "depth.png"), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.shape != (image["height"], image["width"]):
            raise ValueError(f"Invalid depth map for {image_id}")
        plane = json.loads((directory / "meta.json").read_text(encoding="utf-8"))["floor_plane"]
        for ann in chosen:
            mask = polygon_mask(ann["segmentation"], image["width"], image["height"], 0, 0)
            value = raw_volume_ml(depth, mask, plane, camera)
            raw.append((image_id, ann["id"], value))
    assert wanted == {(image_id, ann_id) for image_id, ann_id, _ in raw}
    result = pd.DataFrame(raw, columns=["image_id", "ann_id", "raw_volume_ml"])
    return features.merge(result, on=["image_id", "ann_id"], validate="one_to_one")


def group_key(form, material):
    return f"{form}/{material}"


def volume_summary(rows):
    valid = rows["raw_volume_ml"].gt(0)
    return {
        "n": len(rows),
        "objects": int(rows["object_id"].nunique()),
        "n_raw": int(valid.sum()),
        "raw_median_ml": float(rows.loc[valid, "raw_volume_ml"].median()) if valid.any() else None,
        "true_median_ml": float(rows["volume_ml"].median()),
        "ratio_median": (
            float((rows.loc[valid, "volume_ml"] / rows.loc[valid, "raw_volume_ml"]).median())
            if valid.any() else None
        ),
    }


def fit_volume(train):
    assert train["split"].eq("train").all() and train["volume_ml"].gt(0).all()
    low, high = train["volume_ml"].agg(["min", "max"])
    return {
        "clip_ml": [float(low), float(high)],
        "global": volume_summary(train),
        "groups": {
            group_key(form, mat): volume_summary(group)
            for (form, mat), group in train.groupby(["form", "material"])
        },
    }


def predict_volume(rows, materials, calibration):
    fallback = calibration["global"]
    preds = []
    for form, mat, raw in zip(rows["form"], materials, rows["raw_volume_ml"]):
        stats = calibration["groups"].get(group_key(form, mat), fallback)
        ratio = stats["ratio_median"] or fallback["ratio_median"]
        preds.append(raw * ratio if pd.notna(raw) and raw > 0 and ratio else stats["true_median_ml"])
    return np.clip(preds, *calibration["clip_ml"])


def volume_score(pred, true):
    pred, true = np.broadcast_arrays(np.asarray(pred, dtype=float), np.asarray(true, dtype=float))
    valid = (pred > 0) & (true > 0)
    per_object = np.zeros(pred.shape)
    per_object[valid] = np.maximum(0, 1 - np.abs(np.log(pred[valid] / true[valid])) / np.log(3))
    return float(per_object.mean())


def build_calibration():
    data = build_raw_volumes(pd.read_csv(WORK / "features.csv"))
    train = data.loc[data["split"].eq("train")]
    val = data.loc[data["split"].eq("val")]
    fit = fit_volume(train)
    data[["image_id", "ann_id", "raw_volume_ml"]].to_csv(WORK / "raw_volumes.csv", index=False)
    table = pd.DataFrame.from_dict(fit["groups"], orient="index").rename_axis("form/material")
    table.to_csv(WORK / "volume_by_group.csv")
    (WORK / "volume_calibration.json").write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
    print(table.round(4).to_string())
    print(f"Train: {len(train)} annotations, {train['object_id'].nunique()} objects; "
          f"raw missing or nonpositive: {len(train) - fit['global']['n_raw']}")
    print(f"Global ratio = {fit['global']['ratio_median']:.6f}; clip_ml = {fit['clip_ml']}")
    for scope, rows in (("all val", val), ("real val", val[val["source"].eq("real")])):
        calibrated = predict_volume(rows, rows["material"], fit)
        raw_v = volume_score(rows["raw_volume_ml"], rows["volume_ml"])
        calibrated_v = volume_score(calibrated, rows["volume_ml"])
        print(f"{scope} ({len(rows)}): raw V={raw_v:.6f}; calibrated with GT material V={calibrated_v:.6f}")


if __name__ == "__main__":
    build_calibration()
