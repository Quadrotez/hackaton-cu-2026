import json
import pickle

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

from material_features import METADATA_COLUMNS, TARGET_COLUMNS, extract_annotation, polygon_mask
from material_split import DATA, PARTS, SPLIT_PATH, WORK, build_split, load_frames
from material_train import balanced_accuracy
from volume_calibration import raw_volume_ml


V2_PATH = WORK / "split_v2.json"
V2_IDS = ("b02", "b12", "b21", "s02", "s07", "s18", "s36", "s42")
ALL_FEATURES = WORK / "features_all_v2.csv"
AUX_COLUMNS = ("raw_volume_ml", "surface", "lighting")
UNKNOWN = "__unknown__"


def build_all_data():
    cv2.setNumThreads(1)
    camera = json.loads((DATA / "camera.json").read_text(encoding="utf-8"))
    rows = []
    for source, image_id, image, anns in load_frames():
        if not anns:
            continue
        color_path = DATA / image["file_name"]
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(color_path.with_name("depth.png")), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None or color.shape[:2] != depth.shape or depth.dtype != np.uint16:
            raise ValueError(f"Invalid color/depth pair for {image_id}")
        meta = json.loads(color_path.with_name("meta.json").read_text(encoding="utf-8"))
        for ann in anns:
            _, features = extract_annotation(color, depth, ann, meta, camera)
            mask = polygon_mask(ann["segmentation"], image["width"], image["height"], 0, 0)
            rows.append({
                "image_id": image_id, "ann_id": ann["id"], "source": source,
                **{key: ann["attributes"][key] for key in TARGET_COLUMNS},
                **features,
                "raw_volume_ml": raw_volume_ml(depth, mask, meta["floor_plane"], camera),
                "surface": meta.get("surface", UNKNOWN), "lighting": meta.get("lighting", UNKNOWN),
            })
    data = pd.DataFrame(rows)
    assert not data.duplicated(["image_id", "ann_id"]).any()
    data.to_csv(ALL_FEATURES, index=False)
    return pd.read_csv(ALL_FEATURES)


def get_data(version, all_data=None):
    if version not in ("primary", "v2"):
        raise ValueError(f"Unknown split {version}")
    plan_path = SPLIT_PATH if version == "primary" else V2_PATH
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    all_data = pd.read_csv(ALL_FEATURES) if all_data is None else all_data
    data = []
    for part in PARTS:
        segment = all_data.loc[
            all_data["image_id"].isin(plan[part]["image_ids"]) &
            all_data["object_id"].isin(plan[part]["object_ids"])
        ].copy()
        segment["split"] = part
        assert set(segment["object_id"]) == set(plan[part]["object_ids"])
        assert segment.groupby("source").size().to_dict() == {
            s: c["annotations"] for s, c in plan[part]["counts_by_source"].items()
        }
        data.append(segment)
    train, val = data
    assert set(train["object_id"]).isdisjoint(val["object_id"])
    assert set(train["image_id"]).isdisjoint(val["image_id"])
    steps3 = [c for c in all_data if c not in (*METADATA_COLUMNS, *AUX_COLUMNS)]
    return train, val, steps3


def save_model(model, filename):
    with (WORK / filename).open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_material_model(fast=False):
    # The full version is the frozen v1 CatBoost recipe, retrained independently.
    return CatBoostClassifier(
        iterations=300 if fast else 600, depth=6, learning_rate=.05,
        auto_class_weights="Balanced", loss_function="MultiClass", random_seed=42,
        thread_count=4, allow_writing_files=False, verbose=False,
    )


def diagnostic_material():
    build_split(V2_IDS, V2_PATH)
    data = build_all_data()
    train, val, steps3 = get_data("v2", data)
    model = make_material_model().fit(train[steps3], train["material"])
    save_model(model, "material_v2.pkl")
    predicted = np.asarray(model.predict(val[steps3])).ravel()
    labels = sorted(train["material"].unique())
    metrics = {}
    subsets = (("real", val["source"].eq("real")), ("all", val["split"].eq("val")))
    for scope, mask in subsets:
        y, p = val.loc[mask, "material"], predicted[mask]
        assert set(y) == set(labels)
        matrix = confusion_matrix(y, p, labels=labels)
        score = balanced_accuracy(y, p)
        metrics[scope] = {
            "M": score, "n": len(y), "labels": labels, "confusion": matrix.tolist(),
            "recall": dict(zip(labels, (matrix.diagonal() / matrix.sum(axis=1)).tolist())),
        }
        print(f"\nv2 {scope}: M={score:.6f}, n={len(y)}")
        print(classification_report(y, p, labels=labels, digits=4, zero_division=0))
        print(pd.DataFrame(matrix, index=labels, columns=labels))
    (WORK / "material_v2_results.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    diagnostic_material()
