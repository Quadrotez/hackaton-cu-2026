"""Export fixed out-of-fold masks and evaluate shape/ranking on open-set outer folds."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from build_detector_folds import DEFAULT_OUT
from form_refine import fit, fused_forms
from material_features import bounds, depth_features
from material_split import DATA, WORK
from predict_pipeline import unpack_yolo
from submission_tools import metric
from train_detector_folds import frame_map, image_size, read_manifest
from volume_calibration import raw_volume_ml


CONF = .025
VIEWS = ("base", "flip")
VARIANTS = ("yolo", "yolo_rerank", "shape_only", "shape_fused", "shape_fused_rerank")


def ground_truth(ids, frames):
    return {
        i: [
            {"box": (x, y, x + w, y + h), "form": a["attributes"]["form"], "ann_id": a["id"],
             "object_id": a["attributes"]["object_id"]}
            for a in frames[i][3] for x, y, w, h in [a["bbox"]]
        ]
        for i in ids
    }


def frame_data(path):
    if not path.is_file():
        path = next(p for p in (path.with_name("color.png"), path.with_name("color.jpg")) if p.is_file())
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(path.with_name("depth.png")), cv2.IMREAD_UNCHANGED)
    meta = json.loads(path.with_name("meta.json").read_text(encoding="utf-8"))
    if rgb is None or depth is None or depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
        raise ValueError(f"Invalid RGB-D frame {path}")
    return rgb, depth, meta["floor_plane"]


def detect(model, color, device, *, imgsz, flip=False):
    height, width = color.shape[:2]
    result = model.predict(source=cv2.flip(color, 1) if flip else color,
                           conf=CONF, imgsz=imgsz,
                           max_det=30, retina_masks=True, device=device, verbose=False)[0]
    found = []
    for d in unpack_yolo(result, (height, width)):
        if flip:
            x0, y0, x1, y1 = d.box
            box, mask = (width - x1, y0, width - x0, y1), d.mask[:, ::-1].copy()
        else:
            box, mask = d.box, d.mask
        x0, y0, x1, y1 = np.clip(box, (0, 0, 0, 0), (width, height, width, height)).tolist()
        if x1 <= x0 or y1 <= y0:
            continue
        padded = bounds((x0, y0, x1 - x0, y1 - y0), width, height, .1)
        l, t, r, b = padded
        if not mask[t:b, l:r].any():
            continue
        found.append(({"box": (x0, y0, x1, y1), "score": d.score, "form": d.form}, mask))
    return sorted(found, key=lambda pair: -pair[0]["score"])


def mask_features(p, mask, depth, plane, camera):
    x0, y0, x1, y1 = p["box"]
    bbox = (x0, y0, x1 - x0, y1 - y0)
    height, width = depth.shape
    l, t, r, b = bounds(bbox, width, height, .1)
    bl, bt, br, bb = bounds(bbox, width, height)
    local = mask[t:b, l:r]
    raw = raw_volume_ml(depth, mask, plane, camera)
    return {
        **depth_features(depth[t:b, l:r], local, plane, camera, l, t),
        "aspect_ratio": bbox[2] / bbox[3],
        "mask_area_px": int(local.sum()),
        "mask_bbox_share": mask[bt:bb, bl:br].mean(),
        "raw_volume_ml": np.nan if raw is None else raw,
        "form": p["form"],
    }


def training_features(model, ids, frames, camera, device, *, imgsz):
    gt = ground_truth(ids, frames)
    records = []
    for i, gs in gt.items():
        if not gs:
            continue
        source, path, _, _ = frames[i]
        color, depth, plane = frame_data(path)
        ds = detect(model, color, device, imgsz=imgsz)
        by_index = {id(p): m for p, m in ds}
        pairs = zip(gs, metric._match(gs, [p for p, _ in ds]), strict=True)
        for g, p in pairs:
            if p is None:
                continue
            row = mask_features(p, by_index[id(p)], depth, plane, camera)
            records.append({**row, "image_id": i, "ann_id": g["ann_id"], "object_id": g["object_id"],
                            "source": source, "yolo_form": p["form"], "form": g["form"]})
    return pd.DataFrame.from_records(records)


def rerank(primary, *views):
    predictions = []
    for p in primary:
        support = 0.0
        for view in views:
            match = next((q for q in view if metric._iou(p["box"], q["box"]) >= .55), None)
            if match is not None:
                support += match["score"] if match["form"] == p["form"] else -match["score"]
        predictions.append(dict(p, score=max(.001, min(.999, p["score"] + .20 * support))))
    return predictions


def apply_form(shape, found, rows, strength):
    if not found:
        return []
    forms = fused_forms(pd.DataFrame.from_records(rows), [p["score"] for p in found], shape, strength)
    return [dict(p, form=str(f)) for p, f in zip(found, forms, strict=True)]


def named_ablations(found, features, shape):
    main, flip = (found[v] for v in VIEWS)
    if shape is None:
        return {"yolo": main, "yolo_rerank": rerank(main, flip)}
    fused = [apply_form(shape, found[v], features[v], 1.0) for v in VIEWS]
    return {
        "yolo": main,
        "yolo_rerank": rerank(main, flip),
        "shape_only": apply_form(shape, main, features["base"], 0.0),
        "shape_fused": fused[0],
        "shape_fused_rerank": rerank(*fused),
    }


def measure(gt, predictions):
    common = list(gt)
    pred = {i: predictions[i] for i in common}
    ap = {f: metric._average_precision(gt, pred, f) for f in metric.FORMS}
    aps = [x for x in ap.values() if x is not None]
    n_gt = sum(map(len, gt.values()))
    n_pred = sum(map(len, pred.values()))
    n_empty = sum(not gs for gs in gt.values())
    tp = form_errors = duplicate = unmatched_background = 0
    for i, gs in gt.items():
        pairs = zip(gs, metric._match(gs, pred[i]), strict=True)
        used = set()
        for g, p in pairs:
            if p is not None:
                tp += 1
                used.add(id(p))
                form_errors += p["form"] != g["form"]
        for p in pred[i]:
            if id(p) not in used:
                if any(metric._iou(g["box"], p["box"]) >= .5 for g in gs):
                    duplicate += 1
                elif not any(metric._iou(g["box"], p["box"]) > 0 for g in gs):
                    unmatched_background += 1
    empty_hits = sum(bool(pred[i]) for i in common if not gt[i])
    empty_fp = sum(len(pred[i]) for i in common if not gt[i])
    return {
        "D": (d := float(np.mean(aps)) if aps else None),
        "D_points": 40 * d if d is not None else None,
        "AP50_by_form": ap, "frames": len(common), "gt_boxes": n_gt, "predictions": n_pred,
        "box_recall": tp / n_gt if n_gt else None, "matched": tp, "missed": n_gt - tp,
        "false_positives": n_pred - tp, "duplicates": duplicate, "unmatched_background": unmatched_background,
        "form_errors_among_matched": form_errors,
        "empty_frames": n_empty, "empty_with_predictions": empty_hits, "false_positives_on_empty": empty_fp,
        "false_positives_per_empty_frame": empty_fp / n_empty if n_empty else None,
    }


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def val_views(model, val, frames, camera, device, export_dir=None, *, imgsz):
    """Read each val frame once; publish primary detection masks before any shape fitting."""
    found, features, packed = {}, {}, {}
    for i in val:
        color, depth, plane = frame_data(frames[i][1])
        found[i], features[i] = {}, {}
        for view in VIEWS:
            ds = detect(model, color, device, imgsz=imgsz, flip=view == "flip")
            found[i][view] = [p for p, _ in ds]
            features[i][view] = [mask_features(p, mask, depth, plane, camera) for p, mask in ds]
            if view == "base" and export_dir is not None:
                h, w = depth.shape
                packed[i] = (np.stack([np.packbits(mask, axis=1) for _, mask in ds]) if ds
                             else np.empty((0, h, (w + 7) // 8), dtype=np.uint8))
    if export_dir is not None:
        with (export_dir / "base_masks.npz").open("xb") as f:
            np.savez_compressed(f, **packed)
        with (export_dir / "base_detections.jsonl").open("x", encoding="utf-8") as f:
            for i in val:
                im = frames[i][2]
                f.write(json.dumps({"image_id": i, "source": frames[i][0], "shape": [im["height"], im["width"]],
                                    "detections": found[i]["base"]}, allow_nan=False) + "\n")
        print(f"FIXED DETECTIONS READY: {export_dir / 'base_detections.jsonl'} and base_masks.npz", flush=True)
    return found, features


def _run_eval(model, train, val, frames, device, out, spec=None, *, imgsz):
    camera = json.loads((DATA / "camera.json").read_text(encoding="utf-8"))
    print(f"EXPORT {out}: {len(val)} frames at imgsz={imgsz} (base and flip)", flush=True)
    with threadpool_limits(3):
        found, features = val_views(model, val, frames, camera, device, out, imgsz=imgsz)
        rows = training_features(model, train, frames, camera, device, imgsz=imgsz)
        if rows.empty or set(rows["form"]) != set(metric.FORMS):
            raise ValueError(f"Insufficient predicted-mask form training support at {out}")
        if spec is not None and set(rows["object_id"]) & set(spec["val_object_ids"]):
            raise ValueError("Held-out physical object in form-head training")
        rows.to_csv(out / "shape_train_predicted_features.csv", index=False, mode="x")
        print(f"FIT {out}: {len(rows)} matched predicted masks; "
              f"shape support {dict(Counter(rows['form']))} across {rows['object_id'].nunique()} train objects",
              flush=True)
        shape = fit(rows, "CatBoost", 1)
        combined = {i: named_ablations(found[i], features[i], shape) for i in val}
    preds = {v: {i: combined[i][v] for i in val} for v in VARIANTS}
    write_json(out / "ablation_predictions.json", preds)
    real = [i for i in val if frames[i][0] == "real"]
    sets = {"real": ground_truth(real, frames), "all": ground_truth(val, frames)}
    out_scores = {s: {name: measure(g, p) for name, p in preds.items()} for s, g in sets.items()}
    write_json(out / "ablation_metrics.json", out_scores)
    print(f"REAL D-points {out}: " + ", ".join(f"{k}={m['D_points']:.3f}"
          for k, m in out_scores["real"].items()), flush=True)
    return out_scores


def evaluation_size(model, imgsz):
    if imgsz is not None:
        return imgsz
    try:
        return image_size(model.ckpt["train_args"]["imgsz"])
    except (KeyError, TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise ValueError("Checkpoint has no valid evaluation size; pass --imgsz") from exc


def evaluate_fold(fold_root, checkpoint, spec, frames, device="cpu", *, imgsz=None):
    from ultralytics import YOLO

    train = spec["train_image_ids"]
    val = spec["val_real_image_ids"] + spec["val_synth_image_ids"]
    if set(train) & set(val) or len(val) != len(set(val)):
        raise ValueError(f"Overlapping validation and training frames in fold {spec['fold']}")
    model = YOLO(str(checkpoint.resolve()))
    size = evaluation_size(model, imgsz)
    out = fold_root / "evaluation"
    out.mkdir(exist_ok=False)
    _run_eval(model, train, val, frames, device, out, spec, imgsz=size)


def summarize_pooled(root, specs, frames):
    pooled = {v: {} for v in VARIANTS}
    held_series = {}
    for spec in specs:
        p = root / f"fold_{spec['fold']:02d}" / "evaluation" / "ablation_predictions.json"
        preds = json.loads(p.read_text(encoding="utf-8"))
        ids = spec["val_real_image_ids"]
        for v in VARIANTS:
            if set(ids) & pooled[v].keys():
                raise ValueError("Duplicate real image in OOF predictions")
            pooled[v].update({i: preds[v][i] for i in ids})
        if spec["val_real_empty_image_ids"]:
            g = ground_truth(spec["val_real_empty_image_ids"], frames)
            held_series[",".join(spec["val_empty_series_ids"])] = {
                v: measure(g, preds[v]) for v in VARIANTS
            }
    g = ground_truth(sorted(pooled["yolo"]), frames)
    scores = {v: measure(g, p) for v, p in pooled.items()}
    write_json(root / "pooled_real_ablation.json", {"folds": [f["fold"] for f in specs],
               "variants": scores, "empty_series": held_series})
    print(f"POOLED REAL ({len(g)} unique frames): " + ", ".join(f"{v}={m['D_points']:.3f} D-points"
          for v, m in scores.items()), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--diagnostic", action="store_true", help="Use old checkpoint on original 49 real, never as OOF")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--weights", type=Path, default=WORK / "weights" / "best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=image_size, help="Override the checkpoint's training resolution")
    args = ap.parse_args()
    if args.diagnostic == (args.fold is not None):
        ap.error("Choose --diagnostic or --fold")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    frames = frame_map()
    if args.diagnostic:
        from submission_tools import baseline_holdout
        from ultralytics import YOLO

        held = baseline_holdout()
        val = [i for i, _, _ in held]
        objects = {a["attributes"]["object_id"] for _, _, anns in held for a in anns}
        train = [i for i, (_, _, _, anns) in frames.items()
                 if not any(a["attributes"]["object_id"] in objects for a in anns)]
        model = YOLO(str(args.weights.resolve()))
        size = evaluation_size(model, args.imgsz)
        args.out.mkdir(parents=True, exist_ok=False)
        _run_eval(model, train, val, frames, args.device, args.out, imgsz=size)
    else:
        spec = next(f for f in read_manifest(args.manifest)["folds"] if f["fold"] == args.fold)
        args.out.mkdir(parents=True, exist_ok=False)
        evaluate_fold(args.out, args.weights, spec, frames, args.device, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
