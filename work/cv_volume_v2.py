"""Volume-only ablation on the frozen grouped CV folds and material OOF outputs."""

import json
import math

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from material_features import polygon_mask
from material_split import DATA, WORK, load_frames
from v2_data import ALL_FEATURES
from volume_calibration import volume_score
from volume_v2 import VOLUME_CATS, VOLUME_NUMERIC, weights_by_object


THREADS = 4
KEY = ["image_id", "ann_id"]
GEOMETRY_PATH = WORK / "cv_volume_geometry_v2.csv"
BASE = list(VOLUME_NUMERIC) + ["tilt_deg"]
SILHOUETTE = ["tilt_deg", "mask_area_px", "aspect_ratio", "mask_bbox_share", "sil_area_m2", "sil_volume_ml"]
GEOMETRY = [
    "valid_share", "plane_offset_m", "z_p50_m", "h_adj_p10_m", "h_adj_p50_m", "h_adj_p90_m",
    "h_adj_p98_m", "floor_hull_m2", "footprint_length_m", "footprint_width_m",
    "floor_envelope_ml", "adjusted_raw_ml", "floor_hull_fill",
]
EXTRA = list(dict.fromkeys(BASE + SILHOUETTE + GEOMETRY))
RIDGE_SIZE = [
    "raw_volume_ml", "sil_area_m2", "sil_volume_ml", "z_p50_m", "floor_hull_m2",
    "footprint_length_m", "footprint_width_m", "floor_envelope_ml", "adjusted_raw_ml",
]
RIDGE_X = RIDGE_SIZE + [
    "valid_share", "h_adj_p50_m", "h_adj_p90_m", "h_adj_p98_m", "floor_hull_fill",
    "aspect_ratio", "mask_bbox_share", "tilt_deg",
]
COMPACT_LOG = ["raw_volume_ml", "adjusted_raw_ml", "floor_envelope_ml", "sil_volume_ml"]
COMPACT = COMPACT_LOG + ["valid_share", "mask_bbox_share", "aspect_ratio", "tilt_deg"]
NO_MAT = [c for c in VOLUME_CATS if c != "material"]
MIN_POINTS = 30
MIN_COVERAGE = .70


def rays(u, v, cam):
    return np.column_stack(((u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones(len(u))))


def geometry_one(depth_mm, ann, meta, cam):
    x, y, w, h = ann["bbox"]
    width, height = depth_mm.shape[1], depth_mm.shape[0]
    left, top = max(0, math.floor(x - 16)), max(0, math.floor(y - 16))
    right, bottom = min(width, math.ceil(x + w + 16)), min(height, math.ceil(y + h + 16))
    mask = polygon_mask(ann["segmentation"], right - left, bottom - top, left, top)
    local = depth_mm[top:bottom, left:right].astype(float) / 1000
    n = np.asarray(meta["floor_plane"][:3])
    d = float(meta["floor_plane"][3])
    n_mag = np.linalg.norm(n)
    n, d = n / n_mag, d / n_mag
    v_all, u_all = np.nonzero(mask)
    center_ray = rays(u_all + left, v_all + top, cam).mean(axis=0)
    floor_z = float(np.clip(-d / np.dot(center_ray, n), .05, 3.0))
    sil_area = len(u_all) * floor_z**2 / (cam["fx"] * cam["fy"])
    valid = mask & (local > 0)
    v, u = np.nonzero(valid)
    result = dict(
        tilt_deg=meta["tilt_deg"], depth_points=len(u), valid_share=float(valid.sum() / mask.sum()),
        sil_area_m2=sil_area, sil_volume_ml=sil_area**1.5 * 1e6,
    )
    if len(u) < MIN_POINTS:
        return result
    base = cv2.dilate(mask.astype("uint8"), np.ones((13, 13), dtype="uint8")).astype(bool)
    v_r, u_r = np.nonzero(base & ~mask & (local > 0))
    ring_h = (rays(u_r + left, v_r + top, cam) @ n) * local[v_r, u_r] + d
    offset = float(np.median(ring_h)) if len(ring_h) >= 60 else 0
    offset = float(np.clip(offset, -.02, .02))
    z = local[v, u]
    xyz = rays(u + left, v + top, cam) * z[:, None]
    heights = xyz @ n + d - offset
    h10, h50, h90, h98 = np.quantile(heights, [.1, .5, .9, .98])
    e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    uv = xyz @ np.column_stack((e1, e2))
    low, high = np.quantile(uv, [.01, .99], axis=0)
    keep = ((uv >= low) & (uv <= high)).all(axis=1)
    if keep.sum() < 3:
        keep = np.ones(len(uv), dtype=bool)
    hull = cv2.convexHull(np.ascontiguousarray(uv[keep], dtype=np.float32))
    area = cv2.contourArea(hull)
    _, rect, _ = cv2.minAreaRect(hull)
    cell = .004
    uv_cell = np.floor(uv / cell).astype(np.int64)
    keys = uv_cell[:, 0] * 100000 + uv_cell[:, 1]
    order = np.argsort(keys)
    starts = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1]
    tops = np.maximum.reduceat(heights[order], starts)
    adjusted_raw = np.maximum(0, tops).sum() * cell**2 * 1e6
    return {
        **result, "plane_offset_m": offset, "z_p50_m": np.median(z),
        "h_adj_p10_m": h10, "h_adj_p50_m": h50, "h_adj_p90_m": h90, "h_adj_p98_m": h98,
        "floor_hull_m2": area, "footprint_length_m": max(rect), "footprint_width_m": min(rect),
        "floor_envelope_ml": area * max(h90, .001) * 1e6, "adjusted_raw_ml": adjusted_raw,
        "floor_hull_fill": np.clip(len(starts) * cell**2 / max(area, cell**2), 0, 2),
    }


def get_geometry():
    if GEOMETRY_PATH.exists():
        return pd.read_csv(GEOMETRY_PATH)
    cv2.setNumThreads(1)
    camera = json.loads((DATA / "camera.json").read_text(encoding="utf-8"))
    result = []
    for _, image_id, image, anns in load_frames():
        if not anns:
            continue
        path = (DATA / image["file_name"]).parent
        depth = cv2.imread(str(path / "depth.png"), cv2.IMREAD_UNCHANGED)
        assert depth is not None and depth.dtype == np.uint16
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        for ann in anns:
            result.append({
                "image_id": image_id, "ann_id": ann["id"],
                "meta_surface": meta["surface"], "meta_lighting": meta["lighting"],
                **geometry_one(depth, ann, meta, camera),
            })
    geo = pd.DataFrame(result)
    assert len(geo) == 892 and not geo.duplicated(KEY).any()
    geo.to_csv(GEOMETRY_PATH, index=False)
    return pd.read_csv(GEOMETRY_PATH)


def inputs(rows, numeric, cats, *, heldout, log_sizes=()):
    x = rows[list(numeric) + list(cats)].copy()
    if heldout and "material" in cats:
        x["material"] = rows["material_pred"]
    if log_sizes:
        for c in log_sizes:
            values = x[c].astype(float)
            x[c] = np.log(values.where(values.gt(0)))
    assert not {"mass_g", "take", "volume_ml", "object_id", "image_id", "ann_id", "fold"}.intersection(x)
    return x


def anchor_values(rows, median, adjusted):
    raw = rows["raw_volume_ml"].where(rows["raw_volume_ml"].gt(0), median)
    if adjusted:
        return rows["adjusted_raw_ml"].where(rows["adjusted_raw_ml"].gt(0), raw)
    return raw


def fit_cb(train, numeric, cats, mode, depth=3, loosen=False, adjusted=False):
    x = inputs(train, numeric, cats, heldout=False)
    y = np.log(train["volume_ml"])
    anchor = float(train["volume_ml"].median())
    if mode == "residual":
        y -= np.log(anchor_values(train, anchor, adjusted))
    model = CatBoostRegressor(
        iterations=400, depth=depth, learning_rate=.04, l2_leaf_reg=6,
        loss_function="MAE", one_hot_max_size=16, cat_features=list(cats),
        thread_count=THREADS, random_seed=42, verbose=False, allow_writing_files=False,
    )
    model.fit(x, y, sample_weight=weights_by_object(train))
    clip = np.log([train["volume_ml"].min() / (2 if loosen else 1), train["volume_ml"].max() * (2 if loosen else 1)])
    return model, numeric, cats, mode, anchor, clip, adjusted


def cb_predict(fit, val):
    model, numeric, cats, mode, anchor, clip, adjusted = fit
    x = inputs(val, numeric, cats, heldout=True)
    pred = model.predict(x, thread_count=THREADS)
    if mode == "residual":
        pred += np.log(anchor_values(val, anchor, adjusted))
    return np.clip(pred, *clip)


def ridge_fit_predict(
    train, val, cats, mode, alpha=15., domain=None, silhouette=False,
    compact=False, robust=False, adjusted=False,
):
    numeric, logs = (SILHOUETTE, ["sil_area_m2", "sil_volume_ml"]) if silhouette else (
        (COMPACT, COMPACT_LOG) if compact else (RIDGE_X, RIDGE_SIZE)
    )
    x = inputs(train, numeric, cats, heldout=False, log_sizes=logs)
    x_val = inputs(val, numeric, cats, heldout=True, log_sizes=logs)
    for c in ("surface",):
        if c in cats and x[c].nunique() == 1:
            cats = [k for k in cats if k != c]
    representation = ColumnTransformer([
        ("number", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cats)),
    ])
    reg = HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=1000) if robust else Ridge(alpha=alpha)
    model = make_pipeline(representation, reg)
    weights = weights_by_object(train)
    if domain == "real":
        weights *= np.where(train["source"].eq("real"), 3., 1.)
        weights /= weights.mean()
    y = np.log(train["volume_ml"])
    median = train["volume_ml"].median()
    raw = anchor_values(train, median, adjusted)
    raw_val = anchor_values(val, median, adjusted)
    if mode == "residual":
        y -= np.log(raw)
    weight_key = "huberregressor__sample_weight" if robust else "ridge__sample_weight"
    model.fit(x, y, **{weight_key: weights})
    pred = model.predict(x_val)
    if mode == "residual":
        pred += np.log(raw_val)
    return np.clip(pred, *np.log([train["volume_ml"].min() / 2, train["volume_ml"].max() * 2]))


def frame_split(data, split):
    train_ids = set(split["train_object_ids"])
    val_ids = set(split["val_object_ids"])
    assert train_ids.isdisjoint(val_ids)
    val = data[data["object_id"].isin(val_ids)]
    train = data[data["object_id"].isin(train_ids) & ~data["image_id"].isin(val["image_id"])]
    assert set(train["object_id"]) == train_ids and set(val["object_id"]) == val_ids
    assert val["fold_frozen"].eq(split["fold"]).all()
    return train, val


def candidate_predictions(train, val):
    cats = list(VOLUME_CATS)
    saved = val["volume_baseline_ml"].to_numpy()
    base = cb_predict(fit_cb(train, list(VOLUME_NUMERIC), cats, "residual"), val)
    np.testing.assert_allclose(np.exp(base), saved, rtol=1e-7, atol=1e-7)
    new = {
        "CB-tilt": cb_predict(fit_cb(train, BASE, cats, "residual"), val),
        "CB-geometry": cb_predict(fit_cb(train, EXTRA, cats, "residual", loosen=True), val),
        "CB-geometry-no-material": cb_predict(fit_cb(train, EXTRA, NO_MAT, "residual", loosen=True), val),
        "CB-meta-no-material": cb_predict(fit_cb(train, BASE, NO_MAT, "residual", loosen=True), val),
        "Ridge-geometry": ridge_fit_predict(train, val, cats, "direct"),
        "Ridge-geometry-no-material": ridge_fit_predict(train, val, NO_MAT, "direct"),
        "Ridge-residual": ridge_fit_predict(train, val, cats, "residual"),
        "Ridge-residual-no-material": ridge_fit_predict(train, val, NO_MAT, "residual"),
        "Ridge-residual-real-upweight": ridge_fit_predict(train, val, NO_MAT, "residual", domain="real"),
    }
    silhouette = ridge_fit_predict(train, val, NO_MAT, "direct", silhouette=True)
    new["silhouette-only"] = silhouette
    new["CB-geometry-material-blend"] = (new["CB-geometry"] + new["CB-geometry-no-material"]) / 2
    new["geometry-ensemble"] = (new["CB-geometry-material-blend"] + new["Ridge-residual-no-material"]) / 2

    # Stage 1: measured depth availability; no classifier is trained on an all-positive target.
    # Stage 2: one depth-based head and one independently fitted image/plane-only fallback.
    present = val["depth_points"].ge(MIN_POINTS) & val["raw_volume_ml"].gt(0)
    new["two-stage-presence"] = np.where(present, new["geometry-ensemble"], silhouette)
    weak = val["valid_share"].lt(MIN_COVERAGE)
    soft = np.where(weak, (2 * new["geometry-ensemble"] + silhouette) / 3, new["geometry-ensemble"])
    new["two-stage-coverage"] = np.where(present, soft, silhouette)
    soft_hard = np.where(weak, (new["geometry-ensemble"] + silhouette) / 2, new["geometry-ensemble"])
    new["two-stage-coverage50"] = np.where(present, soft_hard, silhouette)
    new["Ridge-geometry-plus-baseline"] = (new["Ridge-geometry"] + base) / 2
    new["CB-residual-adjusted-compact"] = cb_predict(
        fit_cb(train, COMPACT, cats, "residual", loosen=True, adjusted=True), val
    )
    new["Ridge-compact"] = ridge_fit_predict(train, val, cats, "direct", compact=True)
    new["Ridge-compact-real-upweight"] = ridge_fit_predict(train, val, cats, "direct", compact=True, domain="real")
    new["Huber-compact"] = ridge_fit_predict(train, val, cats, "direct", alpha=2., compact=True, robust=True)
    new["Huber-adjusted-residual"] = ridge_fit_predict(
        train, val, cats, "residual", alpha=2., compact=True, robust=True, adjusted=True,
    )
    new["two-stage-ridge"] = np.where(
        present, np.where(weak, (2 * new["Ridge-geometry"] + silhouette) / 3, new["Ridge-geometry"]), silhouette,
    )
    return {"baseline": saved, **{name: np.exp(value) for name, value in new.items()}}


def metrics_table(oof, recipes):
    rows = []
    for name in recipes:
        for scope, data in (("real", oof[oof["source"].eq("real")]), ("all", oof)):
            scores = []
            base_scores = []
            for fold, group in data.groupby("fold", sort=True):
                base = volume_score(group["baseline"], group["volume_ml"])
                value = volume_score(group[name], group["volume_ml"])
                rows.append((name, scope, str(fold), len(group), value, base))
                scores.append(value)
                base_scores.append(base)
            rows.append((name, scope, "pooled", len(data), volume_score(data[name], data["volume_ml"]),
                         volume_score(data["baseline"], data["volume_ml"])))
            rows.append((name, scope, "fold_mean", len(data), np.mean(scores), np.mean(base_scores)))
    df = pd.DataFrame(rows, columns=["candidate", "scope", "fold", "n_annotations", "V", "baseline_V"])
    return df.assign(delta_V=df["V"] - df["baseline_V"])


def main():
    data = pd.read_csv(ALL_FEATURES)
    oof_ref = pd.read_csv(WORK / "cv_oof.csv")
    split = json.loads((WORK / "cv_folds.json").read_text(encoding="utf-8"))
    data = data.merge(
        oof_ref[KEY + ["object_id", "source", "material_pred", "fold", "volume_pred_ml"]]
        .rename(columns={"volume_pred_ml": "volume_baseline_ml", "fold": "fold_frozen"}),
        on=[*KEY, "object_id", "source"], validate="one_to_one",
    )
    data = data.merge(get_geometry(), on=KEY, validate="one_to_one")
    assert len(data) == len(oof_ref) == 892
    assert data["meta_surface"].eq(data["surface"]).all() and data["meta_lighting"].eq(data["lighting"]).all()
    assert data["tilt_deg"].notna().all()
    near = data["source"].eq("real")
    print(f"Real meta surface={data.loc[near, 'surface'].value_counts().to_dict()}, "
          f"lighting={data.loc[near, 'lighting'].value_counts().to_dict()}, tilt={data.loc[near, 'tilt_deg'].value_counts().to_dict()}")
    print(f"Depth missing: {(data['depth_points'] < MIN_POINTS).sum()}/{len(data)}; "
          f"coverage <{MIN_COVERAGE:.0%}: {(data.loc[near, 'valid_share'] < MIN_COVERAGE).sum()}/{near.sum()} real")
    print("Only volume refit; frozen out-of-fold material predictions and unchanged real/synth grouped folds.")
    folds = []
    for entry in split:
        train, val = frame_split(data, entry)
        preds = candidate_predictions(train, val)
        frame = val[[*KEY, "object_id", "source", "volume_ml", "depth_points", "valid_share"]].assign(fold=entry["fold"])
        folds.append(frame.assign(**preds))
        print(f"Fold {entry['fold']}: train={len(train)}, val={len(val)} including {val['source'].eq('real').sum()} real")
    full = pd.concat(folds, ignore_index=True)
    assert not full.duplicated(KEY).any() and len(full) == 892
    recipes = list(preds)
    result = metrics_table(full, recipes)
    summary = result[result["scope"].eq("real") & result["fold"].eq("pooled")].sort_values("V", ascending=False)
    best = summary.iloc[0]["candidate"]
    result["best_new_pooled_real"] = result["candidate"].eq(summary[~summary["candidate"].eq("baseline")].iloc[0]["candidate"])
    result.to_csv(WORK / "cv_volume_v2.csv", index=False)
    full.to_csv(WORK / "cv_volume_oof_v2.csv", index=False)
    print("\nPooled real (214 OOF annotations):\n" + summary[["candidate", "V", "delta_V"]].round(6).to_string(index=False))
    names = list(dict.fromkeys(["baseline", "CB-tilt", "two-stage-presence", "two-stage-coverage", "two-stage-ridge", best]))
    print("\nReal metrics by fold (pooled = all 214 observations, not 5-fold mean):")
    print(result.loc[result["scope"].eq("real") & result["candidate"].isin(names)]
          .pivot(index="fold", columns="candidate", values="V").round(6).to_string())
    print(f"Best observed pooled real recipe: {best}; comparison on this CV, not an external blind estimate.")


if __name__ == "__main__":
    main()
