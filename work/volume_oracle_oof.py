"""Object-disjoint OOF check for geometry-derived volume on real Astramis data.

This is deliberately an oracle-localisation experiment: GT masks isolate the
volume head from detector errors.  It must not be exported as a submission.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def group_folds(groups, n_splits=5):
    """Deterministic balanced assignment of physical objects to folds."""
    counts = Counter(groups)
    bins = [[] for _ in range(n_splits)]
    sizes = [0] * n_splits
    for group, size in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        i = min(range(n_splits), key=lambda j: (sizes[j], j))
        bins[i].append(group)
        sizes[i] += size
    return [set(x) for x in bins]


def polygon_mask(parts, shape):
    mask = Image.new("1", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    for poly in parts:
        draw.polygon(list(zip(poly[::2], poly[1::2])), fill=1)
    return np.asarray(mask, dtype=bool)


def raw_volume(depth_mm, mask, plane, camera, cell=0.004):
    v, u = np.nonzero(mask & (depth_mm > 0))
    if len(v) < 30:
        return np.nan
    z = depth_mm[v, u].astype(float) / 1000.0
    pts = np.stack(((u - camera["cx"]) / camera["fx"] * z,
                    (v - camera["cy"]) / camera["fy"] * z, z), axis=1)
    n = np.asarray(plane[:3], dtype=float)
    n /= np.linalg.norm(n)
    height = pts @ n + float(plane[3])
    e1 = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    key = (np.floor(pts @ e1 / cell).astype(np.int64) * 100000
           + np.floor(pts @ e2 / cell).astype(np.int64))
    order = np.argsort(key)
    key, height = key[order], height[order]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    return float(np.clip(np.maximum.reduceat(height, starts), 0, None).sum() * cell * cell * 1e6)


def extract(data):
    labels = json.loads((data / "labels/real_train.json").read_text())
    camera = json.loads((data / "camera.json").read_text())
    images = {x["id"]: x for x in labels["images"]}
    rows = []
    for ann in labels["annotations"]:
        im = images[ann["image_id"]]
        frame = data / Path(im["file_name"]).parent
        depth = np.asarray(Image.open(frame / "depth.png"))
        meta = json.loads((frame / "meta.json").read_text())
        mask = polygon_mask(ann["segmentation"], depth.shape)
        values = depth[mask & (depth > 0)].astype(float)
        x, y, w, h = ann["bbox"]
        at = ann["attributes"]
        rows.append(dict(
            image_id=frame.name, object_id=at["object_id"], form=at["form"],
            material=at["material"], surface=meta["surface"], volume_ml=at["volume_ml"],
            raw_volume_ml=raw_volume(depth, mask, meta["floor_plane"], camera),
            log_mask_area=math.log(max(1, int(mask.sum()))),
            log_bbox_area=math.log(max(1.0, w * h)), aspect=math.log(max(w, 1e-3) / max(h, 1e-3)),
            depth_median=float(np.median(values)) if len(values) else np.nan,
            depth_iqr=float(np.percentile(values, 75) - np.percentile(values, 25)) if len(values) else np.nan,
        ))
    return pd.DataFrame(rows)


NUMERIC = ("log_raw", "log_mask_area", "log_bbox_area", "aspect", "depth_median", "depth_iqr")
CATEGORICAL = ("form", "surface")


def matrix(train, rows):
    tr = train.copy()
    tr["log_raw"] = np.log(tr.raw_volume_ml.clip(lower=1e-3))
    med = tr[list(NUMERIC)].median()
    def numeric(df):
        x = df.copy()
        x["log_raw"] = np.log(x.raw_volume_ml.clip(lower=1e-3))
        return x[list(NUMERIC)].fillna(med).to_numpy(float)
    levels = {c: sorted(tr[c].unique()) for c in CATEGORICAL}
    def build(df):
        blocks = [np.ones((len(df), 1)), numeric(df)]
        for c in CATEGORICAL:
            blocks.append(np.column_stack([(df[c].to_numpy() == x).astype(float) for x in levels[c]]))
        return np.column_stack(blocks)
    return build(train), build(rows)


def ridge_predict(train, valid, alpha, residual=False):
    x, xv = matrix(train, valid)
    y = np.log(train.volume_ml.to_numpy(float))
    if residual:
        y -= np.log(train.raw_volume_ml.clip(lower=1e-3).to_numpy(float))
    count = train.object_id.value_counts()
    weight = train.object_id.map(lambda x: 1 / count[x]).to_numpy(float)
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T @ (x * weight[:, None]) + penalty, x.T @ (y * weight))
    logv = xv @ beta
    if residual:
        logv += np.log(valid.raw_volume_ml.clip(lower=1e-3).to_numpy(float))
    lo, hi = np.log(train.volume_ml.min() / 3), np.log(train.volume_ml.max() * 3)
    return np.exp(np.clip(logv, lo, hi))


def v_score(pred, truth):
    return float(np.mean(np.maximum(0, 1 - np.abs(np.log(np.asarray(pred) / np.asarray(truth))) / math.log(3))))


def medians(train, valid, cols):
    grouped = train.groupby(list(cols)).volume_ml.median()
    fallback = train.volume_ml.median()
    return np.asarray([grouped.get(tuple(x), fallback) for x in valid.loc[:, cols].itertuples(index=False, name=None)])


def run(data, out):
    df = extract(data)
    if len(df) != 214 or df.raw_volume_ml.notna().sum() < 200:
        raise RuntimeError(f"Unexpected extraction coverage: {df.shape}, raw={df.raw_volume_ml.notna().sum()}")
    folds = group_folds(df.object_id)
    variants = ("raw", "median_form", "median_form_material", "ridge_a1", "ridge_a10", "ridge_a100",
                "residual_a1", "residual_a10", "residual_a100")
    result = {x: np.full(len(df), np.nan) for x in variants}
    per_fold = defaultdict(dict)
    for fold, held in enumerate(folds, 1):
        va = df.object_id.isin(held)
        tr, valid = df.loc[~va], df.loc[va]
        result["raw"][va] = valid.raw_volume_ml.fillna(tr.volume_ml.median())
        result["median_form"][va] = medians(tr, valid, ("form",))
        result["median_form_material"][va] = medians(tr, valid, ("form", "material"))
        for alpha in (1, 10, 100):
            result[f"ridge_a{alpha}"][va] = ridge_predict(tr, valid, alpha)
            result[f"residual_a{alpha}"][va] = ridge_predict(tr, valid, alpha, residual=True)
        for name in variants:
            per_fold[name][str(fold)] = v_score(result[name][va], valid.volume_ml)
    report = {"rows": len(df), "objects": int(df.object_id.nunique()),
              "raw_coverage": int(df.raw_volume_ml.notna().sum()),
              "pooled_V": {k: v_score(v, df.volume_ml) for k, v in result.items()},
              "fold_V": per_fold}
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"image_id": df.image_id, "object_id": df.object_id, "truth_volume_ml": df.volume_ml,
                  "raw_volume_ml": df.raw_volume_ml, **result}).to_csv(out / "oof.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
