"""Fixed paired bulk-volume experiment. Run: .venv/bin/python -B work/sprint3_bulk_expand/reproduce.py"""

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import score

WORK, DATA = ROOT / "work", ROOT / "dataset"
BASE = WORK / "sprint3_bulk_y_frozen" / "bulk_y_bottom_oof.json"
BASE_SHA = "f0eb64df263afff0130b4e4daf19591031e0f9092374fb9fa6eef1882f1e807e"
FOLDS = WORK / "detector_folds_v1.json"
INDEX = WORK / "sprint1_agnostic_oof_v2" / "mask_index.json"
LABELS = DATA / "labels/real_train.json"
CAM = DATA / "camera.json"
BLEND, MAX_FACTOR = .25, 1.35


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def shape_feature(mask, depth, camera):
    """Size + thickness, with local robust inverse-depth plane and size-only fallback."""
    ys, xs = np.nonzero(mask)
    n = len(xs)
    if n < 30:
        return None
    fx, fy, cx, cy = (camera[k] for k in ("fx", "fy", "cx", "cy"))
    zz = depth[ys, xs] * .001
    good = (zz >= .12) & (zz <= 3.)
    if good.sum() < max(30, .15 * n):
        return None
    z = zz[good]
    ys, xs = ys[good], xs[good]
    # Apparent metric area, no dependence on absolute image position as a predictor.
    area = n * np.median(z) ** 2 / (fx * fy)
    size = math.sqrt(area)
    radius = int(np.clip(round(.14 * math.sqrt(n)), 10, 42))
    h, w = mask.shape
    x0, x1 = max(0, xs.min() - radius), min(w, xs.max() + radius + 1)
    y0, y1 = max(0, ys.min() - radius), min(h, ys.max() + radius + 1)
    m = mask[y0:y1, x0:x1]

    def dilate(r):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        return cv2.dilate(m, kernel)

    bg = (dilate(radius) > 0) & (dilate(radius // 3) == 0)
    yy, xx = np.nonzero(bg)
    zb = depth[yy + y0, xx + x0] * .001
    keep = (zb >= .12) & (zb <= 3.)
    yy, xx, zb = yy[keep] + y0, xx[keep] + x0, zb[keep]
    shape, local = 0., False
    if len(zb) >= 120:
        stride = max(1, math.ceil(len(zb) / 3500))
        yy, xx, zb = yy[::stride], xx[::stride], zb[::stride]

        def rays(y, x):
            return np.column_stack((np.ones(len(y)), (x-cx)/fx, (y-cy)/fy))

        A = rays(yy, xx)
        target = 1 / zb
        fit = np.linalg.lstsq(A, target, rcond=None)[0]
        for _ in range(5):
            residual = target - A @ fit
            sigma = max(.008, 1.4826 * np.median(np.abs(residual - np.median(residual))))
            # Downweight foreground contamination more aggressively than farther pixels.
            weights = np.sqrt(np.minimum(1, sigma * np.where(residual > 0, 1, 2)
                                         / np.maximum(np.abs(residual), 1e-9)))
            fit = np.linalg.lstsq(A * weights[:, None], target * weights, rcond=None)[0]
        error = np.median(np.abs(target - A @ fit)) / np.median(target)
        if error <= .025 and np.linalg.cond(A) <= 500:
            stride = max(1, math.ceil(len(z) / 5000))
            inv = rays(ys[::stride], xs[::stride]) @ fit
            thickness = (1 - z[::stride] * inv) / np.linalg.norm(fit)
            # Robust aggregate: zero negative heights; cap isolated spikes.
            if np.quantile(inv, .05) > 0 and np.median(thickness) >= -.01:
                height = np.mean(np.clip(thickness, 0., .7 * size))
                shape = math.log((height / size + .03) / .13)
                local = True
    return np.array([1., math.log(area / .02), shape]), local


def fit(rows):
    X = np.array([x for x, _, _ in rows])
    y = np.array([v for _, v, _ in rows])
    w = np.array([v for _, _, v in rows])
    # Fixed ridge penalty: one parent-equivalent unit on both slopes.
    P = np.diag([0., 1., 1.])
    return np.linalg.solve(X.T @ (w[:, None] * X) + P, X.T @ (w * y) + P @ [0., 1.5, .5])


def core(p):
    return tuple(p["box"]), p["score"], p["form"]


def rounded(p):
    return tuple(float(f"{x:.1f}") for x in p["box"]), float(f"{p['score']:.6f}"), p["form"]


def oof_features(spec, baseline, images, camera, input_hashes):
    f = spec["fold"]
    n = spec["sources"][f"n640_fold_{f:02d}"]
    l = spec["sources"][f"yolo11l_fold_{f:02d}"]
    nm, nd, lm = [ROOT / p for p in (n["masks"], n["detections"], l["archive"])]
    for path, expected in ((nm, n["masks_sha256"]), (nd, n["detections_sha256"]),
                           (lm, l["archive_sha256"])):
        input_hashes[str(path.relative_to(ROOT))] = sha(path)
        check(input_hashes[str(path.relative_to(ROOT))] == expected, f"Changed OOF mask source {path}")
    nano_meta = {x["image_id"]: x for x in map(json.loads, nd.read_text().splitlines())}
    features = {}
    count = Counter()
    with np.load(nm, allow_pickle=False) as nano, np.load(lm, allow_pickle=False) as large:
        header = json.loads(large["raw_header"].item())
        check(header["key"]["checkpoint_sha256"] == l["checkpoint_sha256"], "l960 checkpoint mismatch")
        large_meta = {x["image_id"]: x for x in header["frames"]}
        for fid in spec["frames"]:
            if not any(p["form"] == "bulk" for p in baseline[fid]):
                continue
            h, w = images[fid]["height"], images[fid]["width"]
            dp = (DATA / images[fid]["file_name"]).with_name("depth.png")
            depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            check(depth is not None and depth.shape == (h, w), f"Invalid depth {fid}")
            nr = nano_meta[fid]
            lr = large_meta[fid]
            check(nr["shape"] == lr["shape"] == [h, w], f"Invalid cached shape {fid}")
            npack, lpack = nano[fid], large[fid]
            check(len(npack) == len(nr["detections"]) and len(lpack) == len(lr["detections"]),
                  f"Invalid cached mask count {fid}")
            for j, p in enumerate(baseline[fid]):
                if p["form"] != "bulk":
                    continue
                if j < len(npack):
                    check(core(p) == core(nr["detections"][j]), f"n640 mismatch {fid}/{j}")
                    packed = npack[j]
                    count["n640_bulk"] += 1
                else:
                    hits = [k for k, q in enumerate(lr["detections"]) if rounded(q) == core(p)]
                    check(len(hits) == 1, f"l960 rounded match not unique {fid}/{j}: {hits}")
                    packed = lpack[hits[0]]
                    count["l960_bulk"] += 1
                x = shape_feature(np.unpackbits(packed, axis=-1, count=w), depth, camera)
                if x is None or not (p["volume"] > 0):
                    count["frozen_fallback"] += 1
                    continue
                count["local_plane"] += int(x[1])
                count["area_only"] += int(not x[1])
                features[fid, j] = x[0]
    return features, dict(count)


def run():
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (BASE, FOLDS, INDEX, LABELS, CAM)}
    check(sha(BASE) == BASE_SHA, "Frozen prediction SHA is not exact")
    manifest, index = read(FOLDS), read(INDEX)
    check(sha(FOLDS) == index["fold_manifest_sha256"], "Fold manifest mismatch")
    check(sha(LABELS) == manifest["labels_sha256"]["labels/real_train.json"], "Real label mismatch")
    baseline, labels, camera = read(BASE), read(LABELS), read(CAM)
    images = {Path(im["file_name"]).parent.name: im for im in labels["images"]}
    check(all(im["file_name"].startswith("real/train/") for im in images.values()), "Not train-only")
    fid_lookup = {im["id"]: fid for fid, im in images.items()}
    anns, ann_by_id = defaultdict(list), {}
    for a in labels["annotations"]:
        anns[fid_lookup[a["image_id"]]].append(a)
        ann_by_id[a["id"]] = a
    specs = {spec["fold"]: spec for spec in index["folds"]}
    ids = [fid for spec in specs.values() for fid in spec["frames"]]
    check(len(ids) == 263 and set(ids) == set(baseline) == set(images), "OOF frame coverage mismatch")

    # No provided scene ID: proxy = chronological single-scene capture burst of same object
    # composition and surface (break on object changes, date changes or >120s gap).
    captures = []
    for fid in sorted(anns):
        meta = read((DATA / images[fid]["file_name"]).with_name("meta.json"))
        objects = tuple(sorted(a["attributes"]["object_id"] for a in anns[fid]))
        captures.append((datetime.fromisoformat(meta["time"]), fid, objects, meta["surface"]))
    scenes, previous, scene = {}, None, 0
    for t, fid, objects, surface in sorted(captures):
        signature = (t.date(), objects, surface)
        if previous is None or signature != previous[0] or (t - previous[1]).total_seconds() > 120:
            scene += 1
        scenes[fid] = scene
        previous = signature, t

    @lru_cache(None)
    def training_features(ann_id):
        a = ann_by_id[ann_id]
        fid = fid_lookup[a["image_id"]]
        im = images[fid]
        dp = (DATA / im["file_name"]).with_name("depth.png")
        dep = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        check(dep is not None and dep.shape == (im["height"], im["width"]), f"Invalid train depth {fid}")
        mask = np.zeros(dep.shape, np.uint8)
        for polygon in a["segmentation"]:
            cv2.fillPoly(mask, [np.rint(np.array(polygon).reshape(-1, 2)).astype(np.int32)], 1)
        r = int(np.clip(round(.05 * math.sqrt(mask.sum())), 3, 16))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        masks = [mask, cv2.erode(mask, kernel), cv2.dilate(mask, kernel)]
        return [shape_feature(m, dep, camera) for m in masks]

    preds = {arm: copy.deepcopy(baseline) for arm in ("real_only", "expanded")}
    diagnostics = {}
    for spec in manifest["folds"]:
        f = spec["fold"]
        tr = set(spec["train_image_ids"]) & images.keys()
        va = set(spec["val_real_image_ids"])
        check(not tr & va and set(specs[f]["frames"]) == va, f"Invalid frames fold {f}")
        tr_objects = {a["attributes"]["object_id"] for fid in tr for a in anns.get(fid, [])}
        va_objects = {a["attributes"]["object_id"] for fid in va for a in anns.get(fid, [])}
        check(tr_objects <= set(spec["train_object_ids"]) and va_objects <= set(spec["val_object_ids"])
              and not tr_objects & va_objects, f"Invalid object groups fold {f}")
        tr_scenes, va_scenes = ({scenes[fid] for fid in group if fid in scenes} for group in (tr, va))
        check(not tr_scenes & va_scenes, f"Scene overlap fold {f}")
        parents, missing = defaultdict(list), 0
        for fid in sorted(tr):
            for a in anns.get(fid, []):
                at = a["attributes"]
                if at["form"] != "bulk" or at["volume_ml"] <= 0:
                    continue
                clean, eroded, dilated = training_features(a["id"])
                if clean is None:
                    missing += 1
                    continue
                parents[at["object_id"]].append((clean, eroded, dilated, math.log(at["volume_ml"])))
        check(parents and not parents.keys() & va_objects, f"Invalid parents fold {f}")
        rows = {arm: [] for arm in preds}
        train_stats = Counter()
        for family in parents.values():
            for clean, eroded, dilated, target in family:
                w = 1 / len(family)
                rows["real_only"].append((clean[0], target, w))
                train_stats["clean_local_plane"] += int(clean[1])
                train_stats["clean_area_only"] += int(not clean[1])
                clean_weight = .5
                for x in (eroded, dilated):
                    if x is None:
                        clean_weight += .25
                        train_stats["invalid_augments"] += 1
                    else:
                        rows["expanded"].append((x[0], target, w * .25))
                        train_stats["augments"] += 1
                rows["expanded"].append((clean[0], target, w * clean_weight))
        coefficients = {arm: fit(rows[arm]) for arm in preds}
        features, oof_stats = oof_features(specs[f], baseline, images, camera, input_hashes)
        cap_count = Counter()
        for arm, coefs in coefficients.items():
            for (fid, j), x in features.items():
                old = baseline[fid][j]["volume"]
                residual = BLEND * (float(x @ coefs) - math.log(old))
                ceiling = math.log(MAX_FACTOR)
                cap_count[arm] += int(abs(residual) >= ceiling)
                preds[arm][fid][j]["volume"] = old * math.exp(float(np.clip(residual, -ceiling, ceiling)))
        diagnostics[f] = {
            "train_parent_ids": sorted(parents), "val_object_ids": sorted(va_objects),
            "train_scenes": len(tr_scenes), "val_scenes": len(va_scenes), "scene_overlap": 0,
            "eligible_train_rows": {k: len(v) for k, v in rows.items()},
            "weight_sum": {k: sum(row[2] for row in v) for k, v in rows.items()},
            "train_clean_missing": missing, "train_geometry": dict(train_stats),
            "coefficients": {k: v.tolist() for k, v in coefficients.items()},
            "oof_geometry": oof_stats, "capped_adjustments": dict(cap_count),
        }
        print(f"fold {f} fitted: {len(parents)} parents, {len(features)} OOF bulk masks", flush=True)

    # Only now score held-out outcomes, once, with the official evaluator.
    gt = {r.image_id: score._parse(r.objects, False, r.image_id) for r in
          score.labels_to_solution(LABELS).itertuples(index=False)}
    for arm, pp in preds.items():
        for fid in baseline:
            for before, after in zip(baseline[fid], pp[fid], strict=True):
                if before["form"] == "bulk":
                    check(before == dict(after, volume=before["volume"]), f"Non-volume mutation {arm}/{fid}")
                else:
                    check(before == after, f"Non-bulk mutation {arm}/{fid}")
    all_preds = {"exact_baseline": baseline, **preds}

    def vol_credit(g, p):
        if p is None or p["volume"] <= 0 or g["volume"] <= 0:
            return 0.
        return max(0., 1. - abs(math.log(p["volume"] / g["volume"])) / math.log(3.))

    def evaluate(keys, expect):
        y = {i: obj for i, obj in gt.items() if i in keys}
        result = {}
        for arm, p in all_preds.items():
            pp = {i: obj for i, obj in p.items() if i in keys}
            result[arm] = score.components(y, pp)
            form_credits = defaultdict(list)
            for fid, objs in y.items():
                for g, pred in zip(objs, score._match(objs, pp[fid]), strict=True):
                    form_credits[g["form"]].append(vol_credit(g, pred))
            for form, credits in form_credits.items():
                result[arm][f"V_gt_{form}"] = float(np.mean(credits))
        ref = result["exact_baseline"]
        check(all(math.isclose(ref[k], v, rel_tol=0, abs_tol=1e-12) for k, v in expect.items()),
              "Exact baseline does not reproduce frozen components")
        for arm in preds:
            check(all(result[arm][k] == ref[k] for k in ("D", "M", "S")), "Other components changed")
        delta = lambda a, b: {k: a[k] - b[k] for k in a}
        result["delta_vs_baseline"] = {k: delta(result[k], ref) for k in preds}
        result["delta_expanded_vs_real_only"] = delta(result["expanded"], result["real_only"])
        return result

    ref_report = read(BASE.with_name("report.json"))
    pooled = evaluate(set(ids), ref_report["pooled"]["bulk:y_bottom"])
    folds = {f: evaluate(set(spec["frames"]), ref_report["folds"][str(f)]["bulk:y_bottom"])
             for f, spec in specs.items()}
    (OUT / "exact_baseline_oof.json").write_bytes(BASE.read_bytes())
    for arm, pred in preds.items():
        (OUT / f"{arm}_oof.json").write_text(json.dumps(pred, allow_nan=False) + "\n")
    report = {
        "protocol": {
            "input": "real train GT polygons/volumes only; fixed existing detector OOF masks at validation; no test",
            "head": "weighted log-volume ridge on apparent metric area and local inverse-depth thickness; slope priors (1.5, 0.5), alpha=1 per slope",
            "expanded": "clean/erosion/dilation masks at fixed per-annotation weights 0.5/0.25/0.25; invalid augments return weight to clean; each object_id sums to weight 1 in both arms",
            "scene": "same UTC/local date, object composition and surface in consecutive positive captures with gaps <=120s, inferred from metadata; not session/surface-held-out",
            "blend": BLEND, "max_volume_ratio": MAX_FACTOR, "no_outer_selection": True,
            "cpu_only": True,
        },
        "inputs_sha256": input_hashes, "training_and_oof": diagnostics, "pooled": pooled, "folds": folds,
        "outputs_sha256": {arm: sha(OUT / f"{arm}_oof.json") for arm in all_preds},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for key, result in [("pooled", pooled), *folds.items()]:
        print(f"{key}: " + "  ".join(f"{arm}={result[arm]['Score']:.6f}" for arm in all_preds)
              + f"  expansion delta={result['delta_expanded_vs_real_only']['Score']:+.6f}")


if __name__ == "__main__":
    run()
