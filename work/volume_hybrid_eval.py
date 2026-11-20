"""Audit the frozen holdout and compare a predicted-form hybrid volume model."""

import argparse
import hashlib
import json
from pathlib import Path
from types import FunctionType

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cv_volume_v2 import GEOMETRY_PATH, RIDGE_X, geometry_one, ridge_fit_predict
from form_refine import POLICIES, fused_forms, select_model
from material_features import bounds, region_features
from material_split import DATA, WORK, load_frames
from predict_pipeline import DEFAULT_BUNDLE, PredictPipeline
from submission_tools import baseline_holdout, ground_truth, metric
from take_train import estimate_mass, predict_take
from take_v2 import take_input
from v2_data import ALL_FEATURES, UNKNOWN
from volume_calibration import raw_volume_ml
from volume_v2 import VOLUME_CATS, predict_volume_v2


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit_holdout(weights, bundle):
    run = WORK / "yolo_run"
    detector_hash = sha256(weights)
    if detector_hash != sha256(run / "seg" / "weights" / "best.pt"):
        raise ValueError("The YOLO training manifest only applies to the original 10-epoch run")
    yolo_train = {p.stem for p in (run / "yolo" / "images" / "train").iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    if len(yolo_train) != 683:
        raise ValueError(f"Expected 683 original YOLO training frames, found {len(yolo_train)}")

    frames = list(load_frames())
    real = baseline_holdout()
    held_objects = {a["attributes"]["object_id"] for _, _, anns in real for a in anns}
    excluded_frames = {i for _, i, _, anns in frames
                       if any(a["attributes"]["object_id"] in held_objects for a in anns)}
    features = pd.read_csv(ALL_FEATURES)
    # Exactly the training selection of fit_submission_models.py for the holdout bundle.
    train = features.loc[~features["image_id"].isin(excluded_frames)].assign(split="train")
    cb_train = set(train["image_id"])
    cb_objects = set(train["object_id"])
    if len(train) != 685 or cb_objects & held_objects or not cb_train <= yolo_train:
        raise ValueError("Unexpected CB-residual-d3 training membership")

    synth, mixed, used_empty_real = [], [], []
    for source, image_id, im, anns in frames:
        objects = {a["attributes"]["object_id"] for a in anns}
        if source == "real" and not anns and image_id in yolo_train:
            used_empty_real.append(image_id)
        if source != "synth" or not objects.intersection(held_objects):
            continue
        if not objects <= held_objects:
            mixed.append(image_id)
            continue
        path = (DATA / im["file_name"]).parent
        if not all((path / file).is_file() for file in ("meta.json", "depth.png")) or not any(
            (path / file).is_file() for file in ("color.png", "color.jpg")
        ):
            raise ValueError(f"Missing data in eligible frame {path}")
        synth.append((image_id, path, anns))

    if len(used_empty_real) != sum(src == "real" and not anns for src, _, _, anns in frames):
        raise ValueError("The dataset has previously unaudited empty real validation frames")
    selected = {i for i, _, _ in real + synth}
    if len(selected) != len(real) + len(synth) or selected & (cb_train | yolo_train):
        raise ValueError("Train and validation images overlap")

    def manifest_rows(selection):
        return [{"image_id": i, "path": str(path.relative_to(DATA)),
                 "object_ids": sorted({a["attributes"]["object_id"] for a in anns}),
                 "annotation_count": len(anns)} for i, path, anns in selection]

    manifest = {
        "detector_sha256": detector_hash,
        "compact_holdout_bundle_sha256": sha256(bundle),
        "yolo_train_frames": len(yolo_train),
        "cb_train_annotations": len(train),
        "cb_train_frames": len(cb_train),
        "cb_train_objects": sorted(cb_objects),
        "held_out_objects": sorted(held_objects),
        "validation_real": manifest_rows(real),
        "validation_synth": manifest_rows(synth),
        "excluded_mixed_synth": sorted(mixed),
        "excluded_real_empty_seen_by_yolo": sorted(used_empty_real),
    }
    return real, synth, train, manifest


def masked_geometry(depth, mask, bbox, meta, camera):
    # Use the same calculation as the trained Ridge recipe, adapting only its
    # polygon rasterizer to slice the original YOLO mask (holes and islands retained).
    def raster(_, width, height, left, top):
        return mask[top:top + height, left:left + width]

    geo = FunctionType(geometry_one.__code__, dict(geometry_one.__globals__, polygon_mask=raster))
    return geo(depth, {"bbox": bbox, "segmentation": None}, meta, camera)


def collect(pipeline, frames):
    rows, crops, found = [], [], {}
    for image_id, path, _ in frames:
        color, depth, meta, detections = pipeline.detect_frame(path, conf=.025, device="cpu")
        h, w = depth.shape
        found[image_id] = []
        for det in sorted(detections, key=lambda d: -d.score):
            x, y, r, b = np.clip(det.box, (0, 0, 0, 0), (w, h, w, h))
            if r <= x or b <= y:
                continue
            bbox = (x, y, r - x, b - y)
            left, top, right, bottom = bounds(bbox, w, h, .1)
            mask = det.mask[top:bottom, left:right]
            if not mask.any():
                continue
            found[image_id].append({"idx": len(rows), "box": (x, y, r, b), "score": det.score})
            rows.append({
                **region_features(color, depth, bbox, mask, meta, pipeline.camera, left, top),
                **masked_geometry(depth, det.mask, bbox, meta, pipeline.camera),
                "raw_volume_ml": raw_volume_ml(depth, det.mask, meta["floor_plane"], pipeline.camera),
                "form": det.form, "score": det.score, "image_id": image_id,
                "surface": meta.get("surface", UNKNOWN), "lighting": meta.get("lighting", UNKNOWN),
            })
            crops.append(pipeline.crop_for_dino(color[top:bottom, left:right], mask))
    rows = pd.DataFrame(rows)
    for name in set(RIDGE_X) - set(rows):
        rows[name] = np.nan
    rows["material_pred"] = pipeline.dino.predict(crops)
    return rows, found


def variant_predictions(pipeline, rows, found, train, form_model):
    rows = rows.copy()
    rows["material"] = rows["material_pred"]
    if form_model is not None:
        rows["form"] = fused_forms(rows, rows["score"], form_model, POLICIES["fused"])
    base = predict_volume_v2(pipeline.volume, rows, rows["material_pred"])
    ridge_log = ridge_fit_predict(train, rows, list(VOLUME_CATS), "direct", alpha=15.)
    blend = np.exp((np.log(base) + ridge_log) / 2)
    methods = {
        "CB-residual-d3": base,
        "Ridge-geometry-plus-baseline": blend,
        "hybrid": np.where(rows["form"].eq("solid"), blend, base),
    }
    for name, volume in methods.items():
        up = pd.DataFrame({
            "volume_pred_ml": volume, "material_pred": rows["material_pred"],
            "mass_pred_g": estimate_mass(rows["form"], rows["material_pred"], volume, pipeline.take["density"]),
        }, index=rows.index)
        take = predict_take(pipeline.take, take_input(rows, pipeline.take["steps3"], up))
        annotations = [
            {"form": form, "material": mat, "volume": float(v), "mass": float(m), "take": int(t)}
            for form, mat, v, m, t
            in zip(rows["form"], rows["material_pred"], volume, up["mass_pred_g"], take, strict=True)
        ]
        yield name, {
            i: [dict(p, **annotations[p["idx"]]) for p in detections] for i, detections in found.items()
        }


def summarize(gt, pred, tags, metrics, by_form):
    res = metric.components(gt, pred)
    metrics.append({**tags, "frames": len(gt), **res})
    pairs = [(g, p) for i, gs in gt.items() for g, p in zip(gs, metric._match(gs, pred[i]), strict=True)]
    for form in metric.FORMS:
        group = [(g, p) for g, p in pairs if g["form"] == form]
        if not group:
            continue
        matched = [(g, p) for g, p in group if p is not None]
        ratios = np.asarray([p["volume"] / g["volume"] for g, p in matched])
        by_form.append({
            **tags, "true_form": form, "matched": len(matched), "objects": len(group),
            "predicted_as_other_form": sum(p["form"] != form for _, p in matched),
            "median_pred_over_true": float(np.median(ratios)) if len(ratios) else np.nan,
            "V": np.maximum(0, 1 - np.abs(np.log(ratios)) / np.log(3)).sum() / len(group),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=WORK / "volume_hybrid_eval_01")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    weights = WORK / "weights" / "best.pt"
    bundle = DEFAULT_BUNDLE.with_name("compact.holdout.pkl")
    real, synth, train, manifest = audit_holdout(weights, bundle)
    print(f"Leakage audit: {len(real)} real + {len(synth)} synth frames on the same "
          f"{len(manifest['held_out_objects'])} objects; 0 overlapping train frames")
    cv2.setNumThreads(1)
    with threadpool_limits(3):
        model = select_model(train)
        train = train.merge(pd.read_csv(GEOMETRY_PATH), on=["image_id", "ann_id"], validate="one_to_one")
        if len(train) != manifest["cb_train_annotations"]:
            raise ValueError("Missing training geometry")
        pipeline = PredictPipeline(weights=weights, bundle_path=bundle)
        rows, found = collect(pipeline, real + synth)
        metrics, by_form = [], []
        truth = {name: ground_truth(frames)
                 for name, frames in (("real", real), ("synth", synth), ("extended", real + synth))}
        for name, form_model in (("YOLO", None), ("fused", model)):
            for method, predictions in variant_predictions(pipeline, rows, found, train, form_model):
                for subset, gt in truth.items():
                    for conf in (.025, .10):
                        if conf == .10 and (subset != "real" or name != "YOLO"):
                            continue  # Reproduce Astra's setup on the original 49 frames.
                        pred = {i: [p for p in predictions[i] if p["score"] >= conf] for i in gt}
                        summarize(gt, pred, {"subset": subset, "conf": conf, "form": name, "volume": method},
                                  metrics, by_form)
    results = pd.DataFrame(metrics).sort_values(["subset", "conf", "form", "volume"])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "split.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results.to_csv(args.out_dir / "metrics.csv", index=False)
    pd.DataFrame(by_form).to_csv(args.out_dir / "by_form.csv", index=False)
    print(results.round(5).to_string(index=False))
    print(f"Saved {args.out_dir}")


if __name__ == "__main__":
    main()
