"""Evaluate a train-only RGB-D shape head on the actual masks from frozen YOLO."""

import argparse
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cv_grouped import read_data
from material_split import WORK, load_frames
from predict_pipeline import DEFAULT_BUNDLE, PredictPipeline, write_submission
from submission_tools import baseline_holdout, ground_truth, metric, parse_prediction
from volume_v2 import weights_by_object


FEATURES = (
    "mask_bbox_share", "depth_zero_share", "height_m_mean", "height_m_std",
    "height_m_p10", "height_m_p50", "height_m_p90", "depth_gradient_mm_mean",
    "depth_gradient_mm_std", "depth_gradient_mm_p10", "depth_gradient_mm_p50",
    "depth_gradient_mm_p90",
)


def inputs(rows):
    x = rows[list(FEATURES)].astype(float).copy()
    x["log_aspect"] = np.abs(np.log(rows["aspect_ratio"].where(rows["aspect_ratio"] > 0)))
    x["log_area"] = np.log1p(rows["mask_area_px"])
    x["log_raw_volume"] = np.log1p(rows["raw_volume_ml"].astype(float))
    x["height_90_10"] = rows["height_m_p90"] - rows["height_m_p10"]
    x["height_relief"] = x["height_90_10"] / rows["height_m_p90"].abs().clip(lower=.005)
    x["height_skew"] = (rows["height_m_p90"] - rows["height_m_p50"]) / x["height_90_10"].clip(lower=.005)
    return x.replace([np.inf, -np.inf], np.nan)


def make_model(name):
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=.3, max_iter=1000))
    return CatBoostClassifier(
        iterations=250, depth=3, learning_rate=.05, verbose=False, random_seed=42,
        allow_writing_files=False, thread_count=3,
    )


RECIPES = (("LR", 1), ("LR", 2), ("CatBoost", 1), ("CatBoost", 2))
POLICIES = {"fused": 1., "conservative": 2., "geometry_only": 0.}


def fit(rows, name, real_weight):
    w = weights_by_object(rows) * np.where(rows["source"].eq("real"), real_weight, 1.)
    w *= rows["form"].map(1. / rows.groupby("form")["object_id"].nunique())
    w /= w.mean()
    model = make_model(name)
    weight_key = "logisticregression__sample_weight" if name == "LR" else "sample_weight"
    return model.fit(inputs(rows), rows["form"].eq("solid"), **{weight_key: w})


def predict_solid(model, rows):
    return model.predict_proba(inputs(rows))[:, 1]


def group_folds(rows):
    objects = rows.groupby("object_id")["form"].first()
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for _, indices in splitter.split(objects, objects):
        val = rows[rows["object_id"].isin(objects.index[indices])]
        train = rows[~rows["object_id"].isin(val["object_id"]) & ~rows["image_id"].isin(val["image_id"])]
        assert set(train["object_id"]).isdisjoint(val["object_id"])
        assert set(train["image_id"]).isdisjoint(val["image_id"])
        yield train, val


def select_model(train):
    metrics = {}
    real = train["source"].eq("real")
    for recipe in RECIPES:
        oof = pd.concat([
            pd.Series(predict_solid(fit(a, *recipe), b), index=b.index)
            for a, b in group_folds(train)
        ], verify_integrity=True).reindex(train.index)
        assert oof.notna().all()
        scores = []
        for rows in (train[real], train):
            scores.append(balanced_accuracy_score(
                rows["form"].eq("solid"), oof.loc[rows.index].ge(.5),
                sample_weight=weights_by_object(rows),
            ))
        metrics[recipe] = .75 * scores[0] + .25 * scores[1]
        print(f"Train OOF {recipe}: real BA={scores[0]:.4f}, all BA={scores[1]:.4f}")
    recipe = max(RECIPES, key=metrics.get)
    print(f"Selected on train only: {recipe}", flush=True)
    return fit(train, *recipe)


def fused_forms(rows, scores, model, prior_strength):
    p = predict_solid(model, rows).clip(.001, .999)
    geometry = np.log(p / (1. - p))
    c = np.clip(np.asarray(scores), 0., .99)
    yolo = np.where(rows["form"].eq("solid"), 1., -1.) * np.log1p(2 * c / (1 - c))
    valid = rows["height_m_p90"].notna() & rows["depth_zero_share"].le(.75)
    solid = geometry + prior_strength * yolo > 0
    return np.where(valid, np.where(solid, "solid", "bulk"), rows["form"])


def yolo_eval(holdout, model, weights, conf):
    cv2.setNumThreads(1)
    pipeline = PredictPipeline(weights=weights, bundle_path=DEFAULT_BUNDLE.with_name("compact.holdout.pkl"))
    gt = ground_truth(holdout)
    # 1.0 is the main candidate. Other weights are sensitivity checks, not model selection.
    variants = {"yolo": None, **POLICIES}
    outputs = {name: {} for name in variants}
    changes = defaultdict(Counter)
    for image_id, path, anns in holdout:
        frame = pipeline.detect_frame(path, conf=conf)
        for name, weight in variants.items():
            pipeline.form_refiner = (None if weight is None else
                                     partial(fused_forms, model=model, prior_strength=weight))
            out = parse_prediction(image_id, pipeline.predict_from_detections(*frame))
            outputs[name][image_id] = out
            if name == "yolo":
                continue
            before = metric._match(gt[image_id], outputs["yolo"][image_id])
            after = metric._match(gt[image_id], out)
            changes[name]["all_flips"] += sum(
                p["form"] != q["form"] for p, q in zip(outputs["yolo"][image_id], out, strict=True)
            )
            for ann, g, p, q in zip(anns, gt[image_id], before, after, strict=True):
                if p is None or q is None or p["form"] == q["form"]:
                    continue
                change = "fixed" if q["form"] == g["form"] else "broke"
                changes[name][change] += 1
                changes[name][f'{change}:{ann["attributes"]["object_id"]}'] += 1
    for name, pred in outputs.items():
        r = metric.components(gt, pred)
        print(f"{name:16} Score {r['Score']:.2f} | D {r['D']:.3f} M {r['M']:.3f} "
              f"V {r['V']:.3f} S {r['S']:.3f} | {dict(changes[name])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=WORK / "weights" / "best.pt")
    parser.add_argument("--conf", type=float, default=.025)
    parser.add_argument("--out-dir", type=Path, help="Write separate test candidates instead of running validation")
    args = parser.parse_args()
    holdout = baseline_holdout()
    val_objects = {a["attributes"]["object_id"] for _, _, anns in holdout for a in anns}
    excluded = {i for _, i, _, anns in load_frames()
                if any(a["attributes"]["object_id"] in val_objects for a in anns)}
    data, _ = read_data()
    train = data[~data["image_id"].isin(excluded)]
    assert val_objects.isdisjoint(train["object_id"])
    print(f"Train: {len(train)} annotations, {train.object_id.nunique()} objects; "
          f"held out: {len(holdout)} real frames, {len(val_objects)} objects")
    with threadpool_limits(3):
        model = select_model(train)
        if args.out_dir is None:
            yolo_eval(holdout, model, args.weights, args.conf)
        else:
            cv2.setNumThreads(1)
            pipeline = PredictPipeline(weights=args.weights)
            for name in ("fused", "geometry_only"):
                pipeline.form_refiner = partial(fused_forms, model=model, prior_strength=POLICIES[name])
                write_submission(pipeline, args.out_dir / f"submission_{name}.csv", {"conf": args.conf})


if __name__ == "__main__":
    main()
