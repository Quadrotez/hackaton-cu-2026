"""Feature extraction and grouped evaluation utilities for Astramis."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


HEAVY_G = 500.0


def color_path(frame_dir: Path) -> Path:
    png = frame_dir / "color.png"
    return png if png.exists() else frame_dir / "color.jpg"


def polygon_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def raw_volume_ml(depth_mm: np.ndarray, mask: np.ndarray, plane: list, camera: dict, cell: float = 0.004) -> float:
    v, u = np.nonzero(mask & (depth_mm > 0))
    if len(v) < 30:
        return float("nan")
    z = depth_mm[v, u].astype(float) / 1000.0
    points = np.stack([
        (u - camera["cx"]) / camera["fx"] * z,
        (v - camera["cy"]) / camera["fy"] * z,
        z,
    ], axis=1)
    normal = np.asarray(plane[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    height = points @ normal + float(plane[3])
    e1 = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross(normal, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    floor = points @ np.stack([e1, e2], axis=1)
    keys = np.floor(floor[:, 0] / cell).astype(np.int64) * 1_000_000 + np.floor(floor[:, 1] / cell).astype(np.int64)
    order = np.argsort(keys)
    starts = np.r_[0, np.nonzero(np.diff(keys[order]))[0] + 1]
    tops = np.maximum.reduceat(height[order], starts)
    return float(np.clip(tops, 0, None).sum() * cell * cell * 1e6)


def instance_record(data_dir: Path, image: dict, annotation: dict, camera: dict) -> dict:
    frame_dir = data_dir / Path(image["file_name"]).parent
    colour = cv2.imread(str(color_path(frame_dir)), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(frame_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if colour is None or depth is None:
        raise ValueError(f"unreadable frame: {frame_dir}")
    plane = json.loads((frame_dir / "meta.json").read_text(encoding="utf-8"))["floor_plane"]
    mask = polygon_mask(annotation["segmentation"], image["height"], image["width"])
    pixels_bgr = colour[mask]
    pixels_hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)[mask]
    depth_pixels = depth[mask]
    x, y, w, h = annotation["bbox"]
    attrs = annotation["attributes"]
    record = {
        "ann_id": annotation["id"], "image_id": Path(image["file_name"]).parent.name,
        "object_id": attrs["object_id"], "form": attrs["form"], "material": attrs["material"],
        "volume_ml": float(attrs["volume_ml"]), "mass_g": float(attrs["mass_g"]),
        "take": int(float(attrs["mass_g"]) <= HEAVY_G), "bbox_x": float(x), "bbox_y": float(y),
        "bbox_w": float(w), "bbox_h": float(h), "mask_area": int(mask.sum()),
        "mask_share": float(mask.mean()), "aspect_ratio": float(w / max(h, 1e-6)),
        "depth_zero_share": float((depth_pixels == 0).mean()),
        "depth_median_mm": float(np.median(depth_pixels[depth_pixels > 0])) if np.any(depth_pixels > 0) else np.nan,
        "raw_volume_ml": raw_volume_ml(depth, mask, plane, camera),
    }
    for prefix, pixels in (("bgr", pixels_bgr), ("hsv", pixels_hsv)):
        for index, name in enumerate(("c0", "c1", "c2")):
            record[f"{prefix}_{name}_mean"] = float(pixels[:, index].mean())
            record[f"{prefix}_{name}_std"] = float(pixels[:, index].std())
    return record


def build_instances(data_dir: Path, labels_file: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    info = json.loads((data_dir / "labels" / labels_file).read_text(encoding="utf-8"))
    camera = json.loads((data_dir / "camera.json").read_text(encoding="utf-8"))
    images = {image["id"]: image for image in info["images"]}
    return pd.DataFrame([instance_record(data_dir, images[ann["image_id"]], ann, camera) for ann in info["annotations"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    features = sub.add_parser("features")
    features.add_argument("--data", required=True, type=Path)
    features.add_argument("--labels", default="real_train.json")
    features.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "features":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rows = build_instances(args.data, args.labels)
        rows.to_csv(args.out, index=False)
        print(f"wrote {len(rows)} instances to {args.out}")


if __name__ == "__main__":
    main()
