"""Paired volume-head A/B on frozen, object-disjoint n640+l960 detections."""

import argparse
import json
import sys
from pathlib import Path

from l960_anchor import CACHE, WORK, additions
from export_joint04 import tail
from yolo960_ab import ROOT, file_sha, fuse_frame

sys.path.insert(0, str(WORK))
import astra_joint04 as joint
import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

FEATURES = {"original": (), "y_bottom": ("y_bottom",), "depth_ring": ("depth_ring",),
            "y_bottom+depth_ring": ("y_bottom", "depth_ring")}


def depth_ring(depth, mask, bbox, margin=8):
    l, t, r, b = joint.bounds(bbox, depth.shape[1], depth.shape[0], 0)
    l, t = max(0, l - margin), max(0, t - margin)
    r, b = min(depth.shape[1], r + margin), min(depth.shape[0], b + margin)
    local = mask[t:b, l:r].astype("uint8")
    zz = depth[t:b, l:r]
    inner = zz[(local != 0) & (zz > 0)]
    expanded = cv2.dilate(local, np.ones((2 * margin - 1, 2 * margin - 1), "uint8"))
    ring = zz[(expanded > local) & (zz > 0)]
    if min(len(inner), len(ring)) < 30:
        return np.nan
    return (float(np.median(ring)) - float(np.median(inner))) / 1000.


def train_extras(frames, include_depth):
    from cv_volume_v2 import polygon_mask

    rows = []
    for _, fid, im, anns in frames:
        if not anns:
            continue
        h, w = im["height"], im["width"]
        dp = (joint.DATA / im["file_name"]).with_name("depth.png")
        depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED) if include_depth else None
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            ring = np.nan
            if include_depth:
                mask = polygon_mask(ann["segmentation"], w, h, 0, 0)
                ring = depth_ring(depth, mask, (x, y, bw, bh))
            rows.append({"image_id": fid, "ann_id": ann["id"], "y_bottom": (y + bh) / h,
                         "depth_ring": ring})
    return pd.DataFrame(rows)


def detector_features(index, frames, anchor, include_depth, *, with_crops=False):
    ds = joint.Detections()
    extras = []
    for spec in index["folds"]:
        directory = WORK / "detector_oof_v1" / f"fold_{spec['fold']:02d}" / "evaluation"
        tag = f"n640_fold_{spec['fold']:02d}"
        meta = spec["sources"][tag]
        for name, key in (("base_masks.npz", "masks_sha256"), ("base_detections.jsonl", "detections_sha256")):
            if meta[key] != file_sha(directory / name):
                raise ValueError(f"Changed cache: {directory / name}")
        with np.load(directory / "base_masks.npz") as masks:
            items = {p["image_id"]: p for p in map(json.loads, (directory / "base_detections.jsonl").read_text().splitlines())}
            for fid in spec["frames"]:
                item = items[fid]
                h, w = item["shape"]
                predictions = anchor[fid]
                packed = masks[fid]
                if len(packed) != len(predictions):
                    raise ValueError(f"{fid}: mask count mismatch")
                for found, pred in zip(item["detections"], predictions, strict=True):
                    if any(found[k] != pred[k] for k in ("box", "score", "form")):
                        raise ValueError(f"{fid}: changed detector prediction")
                unpacked = np.unpackbits(packed, axis=-1, count=w).astype(bool)
                path = frames[fid]
                ds.frame("volume", fid, path, list(zip(predictions, unpacked, strict=True)), spec["fold"])
                depth = cv2.imread(str(path / "depth.png"), cv2.IMREAD_UNCHANGED) if include_depth and predictions else None
                for idx, (p, mask) in enumerate(zip(predictions, unpacked, strict=True)):
                    x, y, r, b = p["box"]
                    extras.append((fid, idx, p["material"], p["volume"], b / h,
                                   depth_ring(depth, mask, (x, y, r - x, b - y)) if include_depth else np.nan))
    keys = ["image_id", "idx", "material_pred", "baseline_volume", "y_bottom", "depth_ring"]
    result = pd.concat([pd.DataFrame(ds.rows), pd.DataFrame(extras, columns=keys)], axis=1)
    return (result, ds.crops) if with_crops else result


def fit_predict(train, hold, extra, *, direct=False):
    from cv_volume_v2 import RIDGE_SIZE, RIDGE_X, inputs
    from volume_v2 import VOLUME_CATS, VOLUME_NUMERIC, volume_inputs, weights_by_object

    m = train["volume_ml"].median()

    def cb_x(rows, mat):
        return volume_inputs(rows, rows[mat]).assign(**{k: rows[k] for k in extra})

    x1, x2 = cb_x(train, "material"), cb_x(hold, "material_pred")
    assert set(x1) == set(VOLUME_NUMERIC + VOLUME_CATS + extra)
    y = np.log(train.volume_ml)
    if not direct:
        y -= np.log(x1.raw_volume_ml.fillna(m))
    cb = CatBoostRegressor(iterations=400, depth=3, learning_rate=.04, l2_leaf_reg=6, loss_function="MAE",
                           one_hot_max_size=16, cat_features=list(VOLUME_CATS), thread_count=3, random_seed=42,
                           verbose=False, allow_writing_files=False)
    weights = weights_by_object(train)
    cb.fit(x1, y, sample_weight=weights)
    logv = cb.predict(x2)
    if not direct:
        logv += np.log(x2.raw_volume_ml.fillna(m))
    result = np.exp(np.clip(logv,
                            np.log(train.volume_ml.min()), np.log(train.volume_ml.max())))
    solid = hold.form.eq("solid")
    if solid.any():
        cats = [k for k in VOLUME_CATS if k != "surface" or train.surface.nunique() > 1]
        numbers = [*RIDGE_X, *extra]
        prep = joint.ColumnTransformer([
            ("number", joint.make_pipeline(joint.SimpleImputer(strategy="median"), joint.StandardScaler()), numbers),
            ("category", joint.OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ])
        ridge = joint.make_pipeline(prep, joint.Ridge(alpha=15.))
        ridge.fit(inputs(train, numbers, VOLUME_CATS, heldout=False, log_sizes=RIDGE_SIZE),
                  np.log(train.volume_ml), ridge__sample_weight=weights)
        s = hold.loc[solid].assign(material=hold.loc[solid, "material_pred"])
        v = ridge.predict(inputs(s, numbers, VOLUME_CATS, heldout=True, log_sizes=RIDGE_SIZE))
        result[solid] = np.exp(np.clip(v, np.log(train.volume_ml.min() / 2), np.log(train.volume_ml.max() * 2)))
    return np.asarray(result)


def run(out, include_depth):
    index = json.loads((CACHE / "mask_index.json").read_text())
    manifests = WORK / "detector_folds_v1.json"
    if file_sha(manifests) != index["fold_manifest_sha256"]:
        raise ValueError("OOF manifest mismatch")
    specs = {d["fold"]: d for d in json.loads(manifests.read_text())["folds"]}
    anchor = json.loads((CACHE / "predictions.json").read_text())["n640_rescue"]
    sources = json.loads((CACHE / "per_model_predictions.json").read_text())
    frames = list(joint.load_frames())
    frames_real = {i: (joint.DATA / im["file_name"]).parent for src, i, im, _ in frames if src == "real"}
    extras = train_extras(frames, include_depth)
    data, _ = joint.read_data()
    data = data.merge(pd.read_csv(joint.GEOMETRY_PATH), on=["image_id", "ann_id"], validate="one_to_one")
    data = data.merge(extras, on=["image_id", "ann_id"], validate="one_to_one")
    if len(data) != 892:
        raise ValueError("Training feature count changed")
    val = detector_features(index, frames_real, anchor, include_depth)
    gt = {r.image_id: joint.metric._parse(r.objects, False, r.image_id) for r in
          joint.metric.labels_to_solution(joint.DATA / "labels" / "real_train.json").itertuples(index=False)}
    modes = {k: v for k, v in FEATURES.items() if include_depth or "depth_ring" not in v}
    for f, spec in specs.items():
        tr = data.loc[data.image_id.isin(spec["train_image_ids"])]
        va = val.loc[val.fold.eq(f)].copy()
        if set(tr.object_id) != set(spec["train_object_ids"]) or set(tr.object_id) & set(spec["val_object_ids"]):
            raise ValueError(f"Fold {f}: invalid training split")
        with threadpool_limits(4):
            for k, v in modes.items():
                val.loc[va.index, k] = fit_predict(tr, va, v)
            if include_depth:
                val.loc[va.index, "bulk_direct"] = fit_predict(tr, va, (), direct=True)
                val.loc[va.index, "bulk_direct_depth"] = fit_predict(tr, va, ("depth_ring",), direct=True)
        print(f"Fold {f}: heads fitted", flush=True)
    # Compare both the refit head and a relative adjustment on the frozen volumes.
    variants = ["frozen", *modes, *(f"delta:{k}" for k in modes if k != "original"), "bulk:y_bottom"]
    if include_depth:
        variants += ["bulk:depth_ring", "bulk:bulk_direct", "bulk:bulk_direct_depth",
                     "bulk:blend_direct", "bulk:blend_direct_depth"]
    all_preds = {k: {} for k in variants}
    for f, spec in specs.items():
        model = f"yolo11l_fold_{f:02d}"
        for fid in spec["val_real_image_ids"]:
            base = anchor[fid]
            dets = [dict(x, tail=tail(x)) for x in sources[fid][model]]
            nearby = joint.metric._parse(fuse_frame([dets], [1.], ()), True, fid)
            add = list(additions(base, nearby))
            frame = val.loc[val.image_id.eq(fid)].set_index("idx")
            for k in variants:
                def adjust(p, idx):
                    v = p["volume"]
                    if k.startswith("delta:"):
                        v *= frame.loc[idx, k[6:]] / frame.loc[idx, "original"]
                    elif k.startswith("bulk:"):
                        if base[idx]["form"] == "bulk":
                            name = k[5:]
                            if name.startswith("blend_"):
                                v = np.sqrt(v * frame.loc[idx, "bulk_" + name[6:]])
                            else:
                                v = frame.loc[idx, name]
                    elif k != "frozen":
                        v = frame.loc[idx, k]
                    return dict(p, volume=float(v))
                all_preds[k][fid] = [adjust(p, j) for j, p in enumerate(base)] + [
                    adjust(dict(p, **{key: base[j][key] for key in ("material", "volume", "take")}), j)
                    for p, j in add]
    pooled = {k: joint.metric.components(gt, p) for k, p in all_preds.items()}
    folds = {f: {k: joint.metric.components({i: gt[i] for i in s["val_real_image_ids"]},
                                         {i: p[i] for i in s["val_real_image_ids"]})
                 for k, p in all_preds.items()} for f, s in specs.items()}
    for k in variants:
        if any(pooled[k][m] != pooled["frozen"][m] for m in ("D", "M", "S")):
            raise ValueError("Volume-only experiment changed other components")
        print(f"{k:<24s} {pooled[k]['Score']:.4f}  V={pooled[k]['V']:.5f}  "
              f"{[round(v[k]['Score'] - v['frozen']['Score'], 4) for v in folds.values()]}")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "report.json").open("x") as stream:
        json.dump({"pooled": pooled, "folds": folds}, stream, indent=2)
    with (out / "bulk_y_bottom_oof.json").open("x") as stream:
        json.dump(all_preds["bulk:y_bottom"], stream, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "sprint2_volume_feature_ab")
    parser.add_argument("--depth", action="store_true", help="Also test depth-ring contrast alone and with y_bottom")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists")
    cv2.setNumThreads(1)
    run(args.out, args.depth)
