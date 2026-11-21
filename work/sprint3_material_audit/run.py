"""Fixed train-only RGB-D material override and no-jitter ablation."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "fusion")]
import score as metric
from export_joint04 import tail
from l960_anchor import additions
from yolo960_ab import fuse_frame

PAIR = ("metal", "plastic")
LABEL = ROOT / "dataset/labels/real_train.json"
MANIFEST = ROOT / "work/detector_folds_v1.json"
BASE = ROOT / "work/sprint3_bulk_y_frozen/bulk_y_bottom_oof.json"
CACHE = ROOT / "work/sprint1_agnostic_oof_v2"
PINNED = {
    BASE: "f0eb64df263afff0130b4e4daf19591031e0f9092374fb9fa6eef1882f1e807e",
    CACHE / "mask_index.json": "90fe9cd8059b981effbce1977c2dc1a8c29c6e066ab3f627992046fca504595c",
    CACHE / "predictions.json": "0bfafd6e15a0619f67478a9a6eabed6c04966118618bd9aa6efcbdbb350d35f9",
    CACHE / "per_model_predictions.json": "3eb7fda8e79e9f858473a7c15983094f16ab8089b3ff01c4a3a3883815ad5c5d",
}
FEATURES = (
    "s_p20", "s_p50", "s_p90", "v_p20", "v_p50", "v_p95", "v_p95_minus_v_p50",
    "bright_neutral_190_50", "bright_neutral_230_65",
    "gray_sobel_p50", "gray_sobel_p90", "gray_sobel_p99",
    "no_depth", "no_depth_interior", "no_depth_boundary", "no_depth_interior_minus_boundary",
    "no_depth_on_bright_neutral", "no_depth_on_bright_neutral_minus_all",
    "depth_local_residual_p50_mm", "depth_local_residual_p90_mm", "depth_local_residual_p99_mm",
    "depth_local_residual_share_gt_20mm",
)
C = 1.0
MIN_FULL = .85
MIN_JACKKNIFE = .70
VARIANTS = {"rgbd_jitter": True, "rgbd_no_jitter": False}
# In BGR order. Pixel-space five-way mask/RGB perturbation; base + four equally weighted views.
VIEWS = (
    ((1., 1., 1.), 0., 0),
    ((1.08, 1.08, 1.08), 5., 0),
    ((.92, .92, .92), -5., 0),
    ((1.05, 1., .95), 0., -1),
    ((.95, 1., 1.05), 0., 1),
)
K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, msg):
    if not condition:
        raise ValueError(msg)


def read_rgbd(image):
    path = ROOT / "dataset" / image["file_name"]
    require(path.is_relative_to(ROOT / "dataset/real/train"), f"Not a real training image: {path}")
    rgb = cv2.imread(str(path))
    depth = cv2.imread(str(path.with_name("depth.png")), cv2.IMREAD_UNCHANGED)
    require(rgb is not None and depth is not None and depth.dtype == np.uint16 and
            rgb.shape[:2] == depth.shape == (image["height"], image["width"]), f"Invalid RGB-D: {path}")
    return rgb, depth


def bounds(box, shape):
    x1, y1, x2, y2 = box
    h, w = shape
    # Coordinates define only the pixel selection. They never enter the model.
    return (max(0, int(np.floor(x1)) - 8), max(0, int(np.floor(y1)) - 8),
            min(w, int(np.ceil(x2)) + 8), min(h, int(np.ceil(y2)) + 8))


def describe(im, z, mask):
    require(bool(mask.any()), "Empty RGB-D region")
    interior = cv2.erode(mask.astype(np.uint8), K5, borderType=cv2.BORDER_CONSTANT,
                         borderValue=0).astype(bool)
    if not interior.any():
        interior = mask
    boundary = mask & ~interior
    _, sat, val = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[mask].T
    s, v = np.percentile(sat, [20, 50, 90]) / 255, np.percentile(val, [20, 50, 95]) / 255
    bright = (val >= 190) & (sat <= 50)
    bright_hi = (val >= 230) & (sat <= 65)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    text = np.log1p(np.percentile(cv2.magnitude(gx, gy)[interior], [50, 90, 99]))
    invalid = z == 0
    all_invalid = invalid[mask].mean()
    ct_invalid = invalid[interior].mean()
    edge_invalid = invalid[boundary].mean() if boundary.any() else all_invalid
    bright_invalid = invalid[mask][bright].mean() if bright.any() else all_invalid
    zmed = cv2.medianBlur(z, 3)
    valid = interior & (z > 0) & (zmed > 0)
    residual = np.abs(z[valid].astype(np.int32) - zmed[valid].astype(np.int32))
    dr = np.log1p(np.percentile(residual, [50, 90, 99])) if residual.size else np.zeros(3)
    row = np.array([*s, *v, v[2]-v[1], bright.mean(), bright_hi.mean(), *text,
                    all_invalid, ct_invalid, edge_invalid, ct_invalid-edge_invalid,
                    bright_invalid, bright_invalid-all_invalid, *dr,
                    np.mean(residual > 20) if residual.size else 0.])
    require(len(row) == len(FEATURES) and np.isfinite(row).all(), "Invalid feature vector")
    return row


def train_rows(frames, ids):
    for fid in sorted(ids):
        image, anns = frames[fid]
        pair = [a for a in anns if a["attributes"]["material"] in PAIR]
        if not pair:
            continue
        im, depth = read_rgbd(image)
        for a in pair:
            x, y, w, h = a["bbox"]
            l, t, r, b = bounds((x, y, x+w, y+h), depth.shape)
            m = np.zeros((b-t, r-l), np.uint8)
            segments = [np.rint(np.asarray(p).reshape(-1, 2) - (l, t)).astype(np.int32)
                        for p in a["segmentation"]]
            for p in segments:
                cv2.fillPoly(m, [p], 1)
            roi, z = im[t:b, l:r], depth[t:b, l:r]
            views = []
            for k, (gain, bias, jitter) in enumerate(VIEWS):
                pix = np.clip(np.rint(roi * np.asarray(gain) + bias), 0, 255).astype(np.uint8) if k else roi
                mask = cv2.erode(m, K3) if jitter < 0 else (cv2.dilate(m, K3) if jitter > 0 else m)
                views.append(describe(pix, z, (mask if mask.any() else m).astype(bool)))
            yield a["attributes"]["material"], a["attributes"]["object_id"], views


def hold_rows(frames, spec, cache_fold, anchor):
    f = spec["fold"]
    ids = spec["val_real_image_ids"]
    require(set(ids) == set(cache_fold["frames"]), f"Fold {f}: invalid mask coverage")
    info = cache_fold["sources"][f"n640_fold_{f:02d}"]
    mfile, dfile = (ROOT / info[k] for k in ("masks", "detections"))
    for p, key in [(mfile, "masks_sha256"), (dfile, "detections_sha256")]:
        require(digest(p) == info[key], f"Changed input: {p}")
    detections = {d["image_id"]: d for d in map(json.loads, dfile.read_text().splitlines())
                  if d["image_id"] in ids}
    require(set(detections) == set(ids), f"Fold {f}: invalid detection coverage")
    with np.load(mfile, allow_pickle=False) as masks:
        for fid in ids:
            entry = detections[fid]
            h, w = entry["shape"]
            preds = anchor[fid]
            require(masks[fid].shape == (len(preds), h, (w+7)//8), f"{fid}: invalid mask shape")
            packed = masks[fid]
            for got, expected in zip(entry["detections"], preds, strict=True):
                require(all(got[k] == expected[k] for k in ("box", "score", "form")),
                        f"{fid}: OOF mask/detection order mismatch")
            if not any(p["material"] in PAIR for p in preds):
                continue
            im, depth = read_rgbd(frames[fid][0])
            require(depth.shape == (h, w), f"{fid}: invalid RGB-D match")
            for j, (p, mask) in enumerate(zip(preds, packed, strict=True)):
                if p["material"] in PAIR:
                    l, t, r, b = bounds(p["box"], depth.shape)
                    roi = np.unpackbits(mask[t:b], axis=-1, count=w)[:, l:r].astype(bool)
                    yield (fid, j), describe(im[t:b, l:r], depth[t:b, l:r], roi)


def probabilities(rows, hold, augmented):
    y, group, views = map(np.asarray, zip(*rows, strict=True))
    n = len(VIEWS)
    # Paired placebo: only the content of slots 1..n-1 differs.
    # Original rows, weights, object/class mass and preprocessing procedure match exactly.
    x = (views if augmented else np.repeat(views[:, :1], n, axis=1)).reshape(-1, len(FEATURES))
    y, group = np.repeat(y == "metal", n), np.repeat(group, n)

    def fit(mask):
        a, b, g = x[mask], y[mask], group[mask]
        counts = Counter(g)
        obj_class = {o: target for o, target in zip(g, b, strict=True)}
        require(len(set(zip(g, b, strict=True))) == len(obj_class), "Mixed material within object")
        by_class = Counter(obj_class.values())
        require(len(by_class) == 2, "Training requires both materials")
        # Each class has half the total loss mass, each object equal within class.
        # Sum of sample weights is number of physical objects, not number of views.
        w = np.array([len(counts) / (2 * by_class[target] * counts[o])
                      for o, target in zip(g, b, strict=True)])
        scaler = StandardScaler().fit(a, sample_weight=w)
        model = LogisticRegression(C=C, solver="lbfgs", tol=1e-9, max_iter=1000).fit(
            scaler.transform(a), b, sample_weight=w)
        return model.predict_proba(scaler.transform(hold))[:, 1]

    full = fit(np.ones(len(x), dtype=bool))
    jk = np.stack([fit(group != o) for o in sorted(set(group))])
    return np.column_stack([full, jk.min(axis=0), jk.max(axis=0)])


def restore_donors(specs, anchor, base, sources):
    donors = {}
    for spec in specs:
        for fid in spec["val_real_image_ids"]:
            old = anchor[fid]
            raw = sources[fid][f"yolo11l_fold_{spec['fold']:02d}"]
            near = metric._parse(fuse_frame([[dict(p, tail=tail(p)) for p in raw]], [1.], ()), True, fid)
            trace = list(enumerate(old)) + [(j, p) for p, j in additions(old, near)]
            for (j, p), q in zip(trace, base[fid], strict=True):
                require(list(p["box"]) == q["box"] and all(p[k] == q[k] for k in ("score", "form"))
                        and all(old[j][k] == q[k] for k in ("take", "material")), f"{fid}: donor mismatch")
            donors[fid] = [j for j, _ in trace]
    return donors


def decision(old, p, lo, hi):
    if old == "plastic" and p >= MIN_FULL and lo >= MIN_JACKKNIFE:
        return "metal"
    if old == "metal" and p <= 1 - MIN_FULL and hi <= 1 - MIN_JACKKNIFE:
        return "plastic"
    return old


def evaluate(gt, base, proposed, donors):
    before, after = (metric.components(gt, {i: data[i] for i in gt}) for data in (base, proposed))
    require(all(before[k] == after[k] for k in ("D", "V", "S")), "Non-material component changed")
    labels = (*PAIR, "other", "miss")
    matrix = {kind: {m: dict.fromkeys(labels, 0) for m in PAIR} for kind in ("base", "candidate")}
    churn = {f"{a}->{b}": dict.fromkeys(("predictions", "matched", "unmatched", "fixed", "broken",
                                      "still_wrong", "frames", "donors"), 0)
             for a, b in (PAIR, PAIR[::-1])}
    for fid, gold in gt.items():
        a, b = base[fid], proposed[fid]
        index = {id(p): j for j, p in enumerate(a)}
        gold_by_index = {}
        for g, p, q in zip(gold, metric._match(gold, a), metric._match(gold, b), strict=True):
            m = g["material"]
            if m in PAIR:
                for mode, pred in (("base", p), ("candidate", q)):
                    key = "miss" if pred is None else (pred["material"] if pred["material"] in PAIR else "other")
                    matrix[mode][m][key] += 1
            if p is not None:
                gold_by_index[index[id(p)]] = m
        changed = defaultdict(set)
        for j, (p, q) in enumerate(zip(a, b, strict=True)):
            require({k: v for k, v in p.items() if k != "material"} ==
                    {k: v for k, v in q.items() if k != "material"}, f"{fid}/{j}: changed non-material field")
            prev, now = p["material"], q["material"]
            if prev != now:
                key = f"{prev}->{now}"
                ch = churn[key]
                ch["predictions"] += 1
                target = gold_by_index.get(j)
                if target is None:
                    ch["unmatched"] += 1
                else:
                    ch["matched"] += 1
                    ch["fixed" if target == now else "broken" if target == prev else "still_wrong"] += 1
                changed[key].add((fid, donors[fid][j]))
        for key, rows in changed.items():
            churn[key]["frames"] += 1
            churn[key]["donors"] += len(rows)
    return {"base": before, "candidate": after, "delta": after["Score"]-before["Score"],
            "confusion": matrix, "directional_churn": churn}


def run(out):
    input_hashes = {path: digest(path) for path in [*PINNED, MANIFEST, LABEL]}
    require(all(input_hashes[p] == h for p, h in PINNED.items()), "Frozen input changed")
    manifest = read_json(MANIFEST)
    specs = sorted(manifest["folds"], key=lambda s: s["fold"])
    cache = read_json(CACHE / "mask_index.json")
    require(input_hashes[MANIFEST] == cache["fold_manifest_sha256"], "Manifest hash mismatch")
    require(input_hashes[LABEL] == manifest["labels_sha256"]["labels/real_train.json"], "Labels hash mismatch")
    require([s["fold"] for s in specs] == list(range(1, 6)), "Expected all five outer folds")
    base = read_json(BASE)
    anchor = read_json(CACHE / "predictions.json")["n640_rescue"]
    donors = restore_donors(specs, anchor, base, read_json(CACHE / "per_model_predictions.json"))
    data = read_json(LABEL)
    anns = defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append(a)
    frames = {Path(im["file_name"]).parent.name: (im, anns[im["id"]]) for im in data["images"]}
    val_ids = [fid for s in specs for fid in s["val_real_image_ids"]]
    require(len(val_ids) == len(set(val_ids)) and set(val_ids) == set(base) == set(anchor) == set(frames),
            "Expected exact once-only real OOF coverage")
    all_val_objects = [obj for s in specs for obj in s["val_object_ids"]]
    require(len(all_val_objects) == len(set(all_val_objects)), "Object held out in multiple folds")
    variants = {name: {} for name in VARIANTS}
    fold_details, donor_rows = {}, []
    with threadpool_limits(limits=1):
        for spec in specs:
            f = spec["fold"]
            train = set(spec["train_image_ids"]) & frames.keys()
            val = set(spec["val_real_image_ids"])
            tr_obj = {a["attributes"]["object_id"] for i in train for a in frames[i][1]}
            va_obj = {a["attributes"]["object_id"] for i in val for a in frames[i][1]}
            require(not train & val and not tr_obj & set(spec["val_object_ids"]) and
                    tr_obj <= set(spec["train_object_ids"]) and va_obj <= set(spec["val_object_ids"]),
                    f"Fold {f}: group leakage")
            rows = list(train_rows(frames, train))
            cache_fold = next(s for s in cache["folds"] if s["fold"] == f)
            keys, hold = zip(*hold_rows(frames, spec, cache_fold, anchor), strict=True)
            fitted = {name: dict(zip(keys, probabilities(rows, np.asarray(hold), aug), strict=True))
                      for name, aug in VARIANTS.items()}
            for fid, j in keys:
                old = anchor[fid][j]["material"]
                row = {"fold": f, "image_id": fid, "donor_idx": j, "base_material": old}
                for name, probs in fitted.items():
                    p, lo, hi = probs[fid, j]
                    row.update({f"{name}_p_metal": p, f"{name}_min_jk": lo, f"{name}_max_jk": hi,
                                f"{name}_candidate": decision(old, p, lo, hi)})
                donor_rows.append(row)
            for name, probs in fitted.items():
                for fid in spec["val_real_image_ids"]:
                    variants[name][fid] = [
                        dict(p, material=decision(p["material"], *probs[fid, j])) if (fid, j) in probs else p.copy()
                        for p, j in zip(base[fid], donors[fid], strict=True)]
            fold_details[f] = {
                "train_real_frames": len(train), "val_real_frames": len(val), "paired_train_views":
                    {m: sum(target == m for target, _, _ in rows) for m in PAIR}, "paired_train_objects":
                    {m: sorted({o for target, o, _ in rows if target == m}) for m in PAIR},
                "scored_oof_donors": len(keys),
            }
            print(f"Fold {f}: fitted/predicted; train pair views={len(rows)}, OOF pair donors={len(keys)}",
                  flush=True)
    # Evaluation is deliberately after all OOF predictions have been produced.
    gt = {r.image_id: metric._parse(r.objects, False, r.image_id) for r in
          metric.labels_to_solution(LABEL).itertuples(index=False)}
    require(metric.components(gt, base)["Score"] == 72.78019593033471, "Exact baseline metric mismatch")
    report = {"input_sha256": {str(p.relative_to(ROOT)): h for p, h in input_hashes.items()},
              "run_sha256": digest(Path(__file__)),
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "cv2": cv2.__version__, "scikit-learn": version("scikit-learn")},
              "config": {"C": C, "full_threshold": MIN_FULL, "all_jackknife_threshold": MIN_JACKKNIFE,
                         "features": FEATURES, "train_jitter_views_bgr_gain_bias_mask": VIEWS},
              "split_counts": fold_details, "results": {}}
    for name, reference, pred in [*((n, base, p) for n, p in variants.items()),
                                  ("jitter_vs_no_jitter", variants["rgbd_no_jitter"], variants["rgbd_jitter"])]:
        fold_reports = {}
        for s in specs:
            ids = set(s["val_real_image_ids"])
            fold_reports[s["fold"]] = evaluate({i: g for i, g in gt.items() if i in ids}, reference, pred, donors)
        pooled = evaluate(gt, reference, pred, donors)
        report["results"][name] = {"pooled": pooled, "folds": fold_reports}
        print(f"{name}: {pooled['base']['Score']:.4f} -> {pooled['candidate']['Score']:.4f} "
              f"({pooled['delta']:+.4f}), folds {[round(x['delta'], 4) for x in fold_reports.values()]}",
              flush=True)
    out.mkdir(exist_ok=False)
    with (out / "donor_predictions.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(donor_rows[0]))
        w.writeheader()
        w.writerows(donor_rows)
    for name, pred in variants.items():
        (out / f"{name}_oof.json").write_text(json.dumps(pred, separators=(",", ":")) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / "run01")
    args = p.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(HERE) or out.parent != HERE or out.exists():
        p.error("Expected a new direct output subdirectory of this audit")
    cv2.setNumThreads(1)
    run(out)
