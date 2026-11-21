"""Audit metal/plastic overrides on the frozen l960 OOF, never on test labels."""

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from export_joint04 import tail
from l960_anchor import additions
from volume_feature_ab import CACHE, WORK, cv2, joint, np, pd, threadpool_limits
from yolo960_ab import file_sha, fuse_frame

from material_diagnostic import extra_features
from material_features import bounds, polygon_mask, region_features

PAIR = ("metal", "plastic")
EMBEDDINGS = WORK / "sprint2_material_visual_features.npz"


def reflectance(color, depth, mask, *, extended=False):
    """Use only object pixels and their immediate background, not frame metadata."""
    _, s, v = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[mask].T
    invalid = depth[mask] == 0
    result = {}
    for t in (160, 200, 230, 245):
        bright = v >= t
        for saturation in (20, 50, 90):
            spec = bright & (s <= saturation)
            result[f"spec_{t}_{saturation}"] = spec.mean()
            result[f"holes_spec_{t}_{saturation}"] = invalid[spec].mean() if spec.any() else np.nan
    for quantile in (10, 50, 90, 98):
        result[f"s_p{quantile}"] = np.percentile(s, quantile)
        result[f"v_p{quantile}"] = np.percentile(v, quantile)
    core = cv2.erode(mask.astype("uint8"), np.ones((5, 5), "uint8")).astype(bool)
    edge = mask & ~core
    for name, sub in (("core", core), ("edge", edge)):
        result[f"missing_{name}"] = np.mean(depth[sub] == 0) if sub.any() else np.nan
    l, _, stats, _ = cv2.connectedComponentsWithStats((mask & (depth == 0)).astype("uint8"))
    result["depth_hole_largest"] = stats[1:l, cv2.CC_STAT_AREA].max(initial=0) / mask.sum()
    if extended:
        result["v_p99"] = np.percentile(v, 99)
        result["v_max_minus_p50"] = float(v.max()) - np.median(v)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype("float32")
        gx, gy = [cv2.Sobel(gray, cv2.CV_32F, dx, dy, ksize=3) for dx, dy in ((1, 0), (0, 1))]
        grad = cv2.magnitude(gx, gy)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        var = cv2.blur(gray**2, (5, 5)) - cv2.blur(gray, (5, 5))**2
        for k, data in (("sobel", grad), ("laplace", lap), ("local_var", var)):
            x = data[core]
            for q in (50, 90, 99):
                result[f"{k}_p{q}"] = np.percentile(x, q) if x.size else np.nan
        median_z = cv2.medianBlur(depth, 3)
        valid = core & (depth > 0) & (median_z > 0)
        diff = np.abs(depth[valid].astype("int32") - median_z[valid])
        result["depth_spike_p99"] = np.percentile(diff, 99) if diff.size else np.nan
        for t in (15, 30, 80):
            result[f"depth_spike_share_{t}"] = np.mean(diff > t) if diff.size else np.nan
    return result


def extract(path, boxes, packed=False, *, extended=False):
    if not boxes:
        return []
    color = cv2.imread(str(path))
    depth = cv2.imread(str(path.with_name("depth.png")), cv2.IMREAD_UNCHANGED)
    meta = json.loads(path.with_name("meta.json").read_text())
    camera = json.loads((joint.DATA / "camera.json").read_text())
    h, w = depth.shape
    rows = []
    for bbox, mask in boxes:
        l, t, r, b = bounds(bbox, w, h, .1)
        roi = np.unpackbits(mask[t:b], axis=-1, count=w).astype(bool)[:, l:r] if packed else (
            polygon_mask(mask, r-l, b-t, l, t)
        )
        if not roi.any():
            raise ValueError(f"{path}: empty mask")
        im, z = color[t:b, l:r], depth[t:b, l:r]
        text = extra_features(im, roi, {})
        text.pop("meta_holes_share", None)
        rows.append({**region_features(color, depth, bbox, roi, meta, camera, l, t),
                     **text, **reflectance(im, z, roi, extended=extended)})
    return rows


def features(specs, index, anchor, *, extended=False):
    frames = list(joint.load_frames())
    paths = {i: joint.DATA / image["file_name"] for source, i, image, _ in frames if source == "real"}
    train = []
    for _, fid, im, annotations in frames:
        relevant = [a for a in annotations if a["attributes"]["material"] in PAIR]
        rows = extract(joint.DATA / im["file_name"], [(a["bbox"], a["segmentation"]) for a in relevant],
                       extended=extended)
        train.extend(dict(row, image_id=fid, ann_id=a["id"]) for row, a in zip(rows, relevant, strict=True))
    val = []
    for f in specs:
        k = f["fold"]
        entry = next(e for e in index["folds"] if e["fold"] == k)["sources"][f"n640_fold_{k:02d}"]
        archive = Path(joint.W.parent) / entry["masks"]
        if file_sha(archive) != entry["masks_sha256"]:
            raise ValueError(f"Unverified predicted masks: {archive}")
        with np.load(archive) as masks:
            for i in f["val_real_image_ids"]:
                ds = anchor[i]
                boxes = []
                for p, mask in zip(ds, masks[i], strict=True):
                    x, y, r, b = p["box"]
                    boxes.append(((x, y, r-x, b-y), mask))
                rows = extract(paths[i], boxes, packed=True, extended=extended)
                val.extend(dict(row, image_id=i, idx=j, fold=k) for j, row in enumerate(rows))
    return pd.DataFrame(train), pd.DataFrame(val)


def load_embeddings(train, val):
    with np.load(EMBEDDINGS) as cache:
        order = {(i, j): k for k, (i, j) in enumerate(zip(cache["image_id"], cache["idx"], strict=True))}
        idx = [order[i, j] for i, j in zip(val.image_id, val.idx, strict=True)]
        frames = [cache[k][idx] for k in ("d", "c", "df", "cf")]
    with np.load(WORK / "astra_joint04_train_flip.npz") as saved:
        keys = joint.keys(train)
        if keys != saved["ids"].tolist():
            raise ValueError("Training embedding order changed")
        training = [joint.normalize(joint.embeddings(prefix, keys)) for prefix in ("astra_", "astra_clip_")]
    return training, frames


def visual(training, frames, fit, hold, *, binary):
    prob = []
    w = 1 / fit.groupby("object_id").object_id.transform("size") if binary else np.ones(len(fit))
    y = fit.material.eq("metal") if binary else fit.material
    for i in range(2):
        lr = LogisticRegression(C=1, class_weight="balanced", max_iter=1000)
        lr.fit(training[i][fit.index], y, sample_weight=w / w.mean())
        p = (lr.predict_proba(frames[i][hold.index]) + lr.predict_proba(frames[i+2][hold.index])) / 2
        if binary:
            p = p[:, list(lr.classes_).index(True)]
        prob.append(p)
    a = np.mean(prob, axis=0)
    if binary:
        return a
    classes = list(lr.classes_)
    metal, plastic = classes.index("metal"), classes.index("plastic")
    return a[:, metal] / (a[:, metal] + a[:, plastic])


def tabular(train, val, cols):
    w = 1 / train.groupby("object_id").object_id.transform("size")
    y = train.material.eq("metal")
    w = w / y.map(w.groupby(y).sum())
    clf = CatBoostClassifier(iterations=250, depth=4, learning_rate=.045, l2_leaf_reg=6,
                             allow_writing_files=False, verbose=False, thread_count=3, random_seed=42)
    clf.fit(train[cols], y, sample_weight=w / w.mean())
    return clf.predict_proba(val[cols])[:, list(clf.classes_).index(True)]


def build_scores(specs, train, tab, val, original, upstream):
    visual_bank, vision = load_embeddings(train, val)
    cols = val.columns.difference(["image_id", "idx", "fold"]).tolist()
    local = [k for k in cols if k.startswith(("texture_", "spec_", "holes_spec_", "missing_",
                                              "depth_hole_", "v_p", "s_p", "depth_zero", "depth_gradient"))]
    output = {}
    with threadpool_limits(4):
        for spec in specs:
            f = spec["fold"]
            raw = train.loc[train.image_id.isin(spec["train_image_ids"])]
            if set(raw.object_id) != set(spec["train_object_ids"]) or set(raw.object_id) & set(spec["val_object_ids"]):
                raise ValueError(f"Fold {f}: object overlap")
            t = tab.loc[tab.image_id.isin(spec["train_image_ids"])]
            v = val.loc[val.fold.eq(f)]
            paired = raw.loc[raw.material.isin(PAIR)]
            prob0 = visual(visual_bank, vision, raw, v, binary=False)
            prob1 = visual(visual_bank, vision, paired, v, binary=True)
            prob_local = tabular(t, v, local)
            prob_rgbd = tabular(t, v, cols)
            for i, j, p, p1, c, g in zip(v.image_id, v.idx, prob0, prob1, prob_local, prob_rgbd, strict=True):
                output[i, j] = {"original": p, "pair_visual": p1, "local": c, "rgbd": g}
                old = original[i][j]["material"]
                if old in PAIR and (p >= .5) != (old == "metal"):
                    raise ValueError(f"{i}/{j}: does not reproduce frozen material ordering")
    if set(output) != {(i, j) for i, ds in upstream.items() for j in range(len(ds))}:
        raise ValueError("Missing predictions")
    return output


def measure(labels, base, output, specs):
    metric = joint.metric
    old = {i: [p for p, _ in ps] for i, ps in base.items()}
    before = metric.components(labels, old)
    folds = [{i: labels[i] for i in f["val_real_image_ids"]} for f in specs]
    modes = {
        "local": lambda p: p["local"], "rgbd": lambda p: p["rgbd"], "pair_visual": lambda p: p["pair_visual"],
        "local_blend": lambda p: .5*p["local"] + .5*p["original"],
        "rgbd_blend": lambda p: .5*p["rgbd"] + .5*p["original"],
        "pair_visual_blend": lambda p: .5*p["pair_visual"] + .5*p["original"],
        "mixed_blend": lambda p: .25*p["pair_visual"] + .25*p["local"] + .5*p["original"],
    }
    report = {}
    for name, probability in modes.items():
        for threshold in (.55, .65, .8):
            for directions in ("both", "metal", "plastic"):
                new = {}
                for i, ps in base.items():
                    new[i] = []
                    for p, j in ps:
                        m = p["material"]
                        q = probability(output[i, j])
                        if m == "plastic" and q >= threshold and directions != "plastic":
                            m = "metal"
                        elif m == "metal" and q <= 1-threshold and directions != "metal":
                            m = "plastic"
                        new[i].append(dict(p, material=m))
                after = metric.components(labels, new)
                if any(after[k] != before[k] for k in ("D", "V", "S")):
                    raise ValueError("Material-only comparison changed other components")
                delta = [metric.components(g, {i: new[i] for i in g})["Score"]
                         - metric.components(g, {i: old[i] for i in g})["Score"] for g in folds]
                changes = {"fixes": 0, "breaks": 0, "metal_hits_before": 0, "metal_hits_after": 0,
                           "plastic_hits_before": 0, "plastic_hits_after": 0}
                for i, gt in labels.items():
                    for g, pa, pb in zip(gt, metric._match(gt, old[i]), metric._match(gt, new[i]), strict=True):
                        a, b = [p is not None and p["material"] == g["material"] for p in (pa, pb)]
                        changes["fixes"] += int(not a and b)
                        changes["breaks"] += int(a and not b)
                        if g["material"] in PAIR:
                            changes[g["material"] + "_hits_before"] += int(a)
                            changes[g["material"] + "_hits_after"] += int(b)
                key = f"{name}/{threshold}/{directions}"
                report[key] = {"metrics": after, "pooled_delta": after["Score"]-before["Score"],
                               "fold_deltas": delta, **changes}
    return before, report


def run(out):
    index = json.loads((CACHE / "mask_index.json").read_text())
    manifest = WORK / "detector_folds_v1.json"
    if file_sha(manifest) != index["fold_manifest_sha256"]:
        raise ValueError("OOF manifest mismatch")
    specs = json.loads(manifest.read_text())["folds"]
    original = json.loads((CACHE / "predictions.json").read_text())["n640_rescue"]
    source = json.loads((CACHE / "per_model_predictions.json").read_text())
    base = {}
    metric = joint.metric
    for f in specs:
        for i in f["val_real_image_ids"]:
            old = original[i]
            detections = [dict(d, tail=tail(d)) for d in source[i][f"yolo11l_fold_{f['fold']:02d}"]]
            l960 = metric._parse(fuse_frame([detections], [1.], ()), True, i)
            base[i] = [(d, j) for j, d in enumerate(old)] + [
                (dict(d, **{k: old[j][k] for k in ("material", "volume", "take")}), j)
                for d, j in additions(old, l960)
            ]
    train, _ = joint.read_data()
    ext, hold = features(specs, index, original)
    tab = train[["image_id", "ann_id", "object_id", "material"]].merge(ext, on=["image_id", "ann_id"],
                                                                       validate="one_to_one")
    print(f"Features ready: {len(tab)} paired training masks and {len(hold)} OOF predicted masks", flush=True)
    output = build_scores(specs, train, tab, hold, original, original)
    labels = {r.image_id: metric._parse(r.objects, False, r.image_id) for r in metric.labels_to_solution(
        joint.DATA / "labels/real_train.json").itertuples(index=False)}
    before, trials = measure(labels, base, output, specs)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "report.json").open("x") as stream:
        json.dump({"base": before, "trials": trials}, stream, indent=2)
    order = sorted(trials, key=lambda k: -trials[k]["pooled_delta"])
    for k in order[:10]:
        v = trials[k]
        print(f"{k:<34} {v['pooled_delta']:+.4f} | "
              f"{[round(x, 3) for x in v['fold_deltas']]} | fixes {v['fixes']}, breaks {v['breaks']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "sprint3_metal_plastic")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists")
    cv2.setNumThreads(1)
    run(args.out)
