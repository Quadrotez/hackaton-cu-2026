"""Feature extraction and grouped evaluation utilities for Astramis."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import score as metric


HEAVY_G = 500.0
MATERIALS = ("metal", "wood", "plastic", "glass", "paper", "mineral")
MATERIAL_CODE = {name: index for index, name in enumerate(MATERIALS)}
NON_FEATURE_COLUMNS = {
    "ann_id", "image_id", "object_id", "form", "material", "volume_ml", "mass_g", "take",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
}


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


def instance_crop(data_dir: Path, image: dict, annotation: dict, size: int = 224, padding: float = 0.10) -> np.ndarray:
    frame_dir = Path(data_dir) / Path(image["file_name"]).parent
    colour = cv2.imread(str(color_path(frame_dir)), cv2.IMREAD_COLOR)
    if colour is None:
        raise ValueError(f"unreadable colour frame: {frame_dir}")
    height, width = colour.shape[:2]
    x, y, box_w, box_h = annotation["bbox"]
    pad = padding * max(box_w, box_h)
    x1, y1 = max(0, int(np.floor(x - pad))), max(0, int(np.floor(y - pad)))
    x2, y2 = min(width, int(np.ceil(x + box_w + pad))), min(height, int(np.ceil(y + box_h + pad)))
    crop = colour[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"empty crop for annotation {annotation['id']}")
    return cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def resnet18_embeddings(crops: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    encoded = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            tensors = [transform(torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0)
                       for crop in crops[start:start + batch_size]]
            encoded.append(model(torch.stack(tensors)).cpu().numpy())
    return np.concatenate(encoded, axis=0)


def embedding_table(data_dir: Path, labels_file: str, encoder=resnet18_embeddings) -> pd.DataFrame:
    data_dir = Path(data_dir)
    info = json.loads((data_dir / "labels" / labels_file).read_text(encoding="utf-8"))
    images = {image["id"]: image for image in info["images"]}
    annotations = info["annotations"]
    embeddings = encoder([instance_crop(data_dir, images[ann["image_id"]], ann) for ann in annotations])
    table = pd.DataFrame(embeddings, columns=[f"emb_{index:03d}" for index in range(embeddings.shape[1])])
    table.insert(0, "ann_id", [ann["id"] for ann in annotations])
    return table


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


def group_folds(rows: pd.DataFrame):
    groups = rows["object_id"].astype(str)
    splitter = GroupKFold(n_splits=5)
    return splitter.split(rows, groups=groups)


def numeric_features(rows: pd.DataFrame, material: pd.Series | None = None) -> pd.DataFrame:
    columns = [column for column in rows.columns if column not in NON_FEATURE_COLUMNS]
    features = rows[columns].copy()
    features["form_code"] = rows["form"].map({"bulk": 0, "solid": 1}).astype(float)
    if material is not None:
        features["material_code"] = material.map(MATERIAL_CODE).fillna(-1).astype(float)
    return features.replace([np.inf, -np.inf], np.nan)


def make_oof_predictions(rows: pd.DataFrame, embeddings: pd.DataFrame | None = None) -> pd.DataFrame:
    oof = rows[["ann_id", "image_id", "object_id", "form"]].copy()
    oof["fold"] = -1
    oof["material_pred"] = "plastic"
    oof["volume_pred"] = np.nan
    oof["heavy_pred"] = 0
    base_features = numeric_features(rows)
    visual_features = None
    if embeddings is not None:
        visual_features = rows[["ann_id"]].merge(embeddings, on="ann_id", how="left", validate="one_to_one").drop(columns="ann_id")
        if visual_features.isna().any().any():
            raise ValueError("missing visual embedding for an annotation")
    for fold, (train_idx, valid_idx) in enumerate(group_folds(rows)):
        train, valid = rows.iloc[train_idx], rows.iloc[valid_idx]
        tabular_material_model = CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced", iterations=300, depth=4,
            learning_rate=0.05, random_seed=17, verbose=False, allow_writing_files=False,
        )
        tabular_material_model.fit(base_features.iloc[train_idx], train["material"])
        tabular_material_pred = pd.Series(
            tabular_material_model.predict(base_features.iloc[valid_idx]).reshape(-1), index=valid.index,
        )
        if visual_features is None:
            material_pred = tabular_material_pred
        else:
            material_model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=17),
            )
            material_model.fit(visual_features.iloc[train_idx], train["material"])
            material_pred = pd.Series(material_model.predict(visual_features.iloc[valid_idx]), index=valid.index)
        oof.loc[valid_idx, "material_pred"] = material_pred.to_numpy()

        volume_train = numeric_features(train, train["material"])
        volume_valid = numeric_features(valid, tabular_material_pred)
        volume_model = CatBoostRegressor(
            loss_function="RMSE", iterations=300, depth=4, learning_rate=0.05,
            random_seed=17, verbose=False, allow_writing_files=False,
        )
        volume_model.fit(volume_train, np.log(train["volume_ml"].clip(lower=1)))
        volume_pred = np.exp(volume_model.predict(volume_valid)).clip(1, 10000)
        oof.loc[valid_idx, "volume_pred"] = volume_pred

        heavy_model = CatBoostClassifier(
            loss_function="Logloss", auto_class_weights="Balanced", iterations=200, depth=3,
            learning_rate=0.05, random_seed=17, verbose=False, allow_writing_files=False,
        )
        heavy_model.fit(volume_train, (train["mass_g"] > HEAVY_G).astype(int))
        oof.loc[valid_idx, "heavy_pred"] = heavy_model.predict(volume_valid).reshape(-1).astype(int)
        oof.loc[valid_idx, "fold"] = fold
    oof["take_pred"] = 1 - oof["heavy_pred"].astype(int)
    return oof


def oracle_score(rows: pd.DataFrame, oof: pd.DataFrame) -> dict:
    merged = rows.merge(oof, on=["ann_id", "image_id", "object_id", "form"], validate="one_to_one")
    gt, pred = {}, {}
    for image_id, part in merged.groupby("image_id", sort=False):
        gt[image_id], pred[image_id] = [], []
        for row in part.itertuples(index=False):
            box = (row.bbox_x, row.bbox_y, row.bbox_x + row.bbox_w, row.bbox_y + row.bbox_h)
            gt[image_id].append({"box": box, "form": row.form, "material": row.material, "volume": row.volume_ml, "take": row.take})
            pred[image_id].append({"box": box, "score": 1.0, "form": row.form, "material": row.material_pred,
                                   "volume": float(row.volume_pred), "take": int(row.take_pred)})
    return metric.components(gt, pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    features = sub.add_parser("features")
    features.add_argument("--data", required=True, type=Path)
    features.add_argument("--labels", default="real_train.json")
    features.add_argument("--out", required=True, type=Path)
    oof = sub.add_parser("oof")
    oof.add_argument("--features", required=True, type=Path)
    oof.add_argument("--out", required=True, type=Path)
    oof.add_argument("--report", required=True, type=Path)
    oof.add_argument("--embeddings", type=Path)
    embeddings = sub.add_parser("embeddings")
    embeddings.add_argument("--data", required=True, type=Path)
    embeddings.add_argument("--labels", default="real_train.json")
    embeddings.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "features":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rows = build_instances(args.data, args.labels)
        rows.to_csv(args.out, index=False)
        print(f"wrote {len(rows)} instances to {args.out}")
    if args.command == "oof":
        rows = pd.read_csv(args.features)
        visual = pd.read_csv(args.embeddings) if args.embeddings else None
        predictions = make_oof_predictions(rows, visual)
        report = oracle_score(rows, predictions)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.out, index=False)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("OOF " + " ".join(f"{key}={value:.4f}" for key, value in report.items()))
    if args.command == "embeddings":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table = embedding_table(args.data, args.labels)
        table.to_csv(args.out, index=False)
        print(f"wrote {len(table)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
