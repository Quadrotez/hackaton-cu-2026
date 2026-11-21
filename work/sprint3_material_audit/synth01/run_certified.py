"""Fixed 80:20 augmentation A/B on foreground-certified, background-allowed synth."""
import argparse
from functools import lru_cache
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run as r

BG = r.ROOT / "work/sprint3_synth_backgrounds"
ALLOW = BG / "allowed_synth.json"
BACKGROUND = BG / "synth_to_real_empty.json"
SYNTH = r.ROOT / "dataset/labels/synth_train.json"
SHARE = .20


def erode(m, n):
    return cv2.erode(m.astype(np.uint8), np.ones((2*n+1, 2*n+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def pack(im, z, m, l, t):
    return np.ascontiguousarray(im), np.ascontiguousarray(z), np.ascontiguousarray(m), l, t


def read_synth(image):
    path = r.ROOT / "dataset" / image["file_name"]
    r.require(path.is_relative_to(r.ROOT / "dataset/synth/train"), str(path))
    im = cv2.imread(str(path))
    z = cv2.imread(str(path.with_name("depth.png")), -1)
    m = cv2.imread(str(path.with_name("mask.png")), -1)
    r.require(im is not None and z is not None and m is not None and
              im.shape[:2] == z.shape == m.shape == (image["height"], image["width"]) and
              z.dtype == np.uint16 and m.dtype == np.uint8, str(path))
    return im, z, m


def crop(im, z, mask, box):
    l, t, end, bot = r.bounds(box, z.shape)
    return pack(im[t:bot, l:end], z[t:bot, l:end], mask[t:bot, l:end], l, t)


def xyxy(a):
    x, y, w, h = a["bbox"]
    return (x, y, x+w, y+h)


def real_crop(image, ann):
    im, z = r.read_rgbd(image)
    l, t, end, bot = r.bounds(xyxy(ann), z.shape)
    m = np.zeros((bot-t, end-l), np.uint8)
    for poly in ann["segmentation"]:
        pts = np.rint(np.asarray(poly).reshape(-1, 2) - (l, t)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return pack(im[t:bot, l:end], z[t:bot, l:end], m, l, t)


def safe_desc(region):
    im, z, m, _, _ = region
    support, inner = erode(m, 4), erode(m, 5)
    if not inner.any():
        return None
    # Mask before any conversion, stencil or order-statistic operation.
    safe_rgb = np.where(support[..., None], im, 0).astype(np.uint8, copy=False)
    safe_z = np.where(support, z, 0).astype(np.uint16, copy=False)
    return r.describe(safe_rgb, safe_z, inner)


def channel_map(a, b):
    # One output per observed input level; preserve monotone intensity ordering.
    mn = np.full(256, 255, np.uint8)
    mx = np.zeros(256, np.uint8)
    np.minimum.at(mn, a, b)
    np.maximum.at(mx, a, b)
    occupied = np.bincount(a, minlength=256) > 0
    return occupied.sum() >= 16 and np.array_equal(mn[occupied], mx[occupied]) and bool(
        np.all(np.diff(mn[occupied].astype(np.int16)) >= 0))


def certify(syn_region, syn_ann, parents, width):
    im, z, m, left, top = syn_region
    y, x = np.where(erode(m, 4))
    if x.size < 512:
        return None
    b, d = im[y, x], z[y, x].astype(np.int32)
    valid = d > 0
    if valid.sum() < 256:
        return None
    # Fixed sparse shortlist. Final test below still checks every support pixel.
    ix = np.linspace(0, len(x)-1, 80, dtype=int)
    sx, sy, sw, sh = syn_ann["bbox"]
    matches = []
    for fid, ann, region, source_interior in parents:
        rx, ry, rw, rh = ann["bbox"]
        if abs(sw-rw) > 4 or abs(sh-rh) > 4 or abs(sy-ry) > 2:
            continue
        rgb, depth, _, l, t = region
        yy = y + top - t
        if yy[0] < 0 or yy[-1] >= depth.shape[0]:
            continue
        for flip in (False, True):
            origin = width - (rx+rw) if flip else rx
            guess = int(round(sx-origin))
            for dx in range(guess-2, guess+3):
                xx = x + left - dx
                if flip:
                    xx = width - 1 - xx
                xx -= l
                if xx.min() < 0 or xx.max() >= depth.shape[1]:
                    continue
                yi, xi = yy[ix], xx[ix]
                if not source_interior[yi, xi].all():
                    continue
                zr = depth[yi, xi].astype(np.int32)
                vi = valid[ix]
                off = d[ix][vi] - zr[vi]
                if not np.array_equal(zr > 0, vi) or not off.size or np.ptp(off) > 1:
                    continue
                if not source_interior[yy, xx].all():
                    continue
                zr = depth[yy, xx].astype(np.int32)
                if not np.array_equal(zr > 0, valid) or np.unique(zr[valid]).size < 16:
                    continue
                off = d[valid] - zr[valid]
                if np.ptp(off) > 1 or np.max(np.abs(off)) > 100:
                    continue
                a = rgb[yy, xx]
                if all(channel_map(a[:, k], b[:, k]) for k in range(3)):
                    matches.append({"real_donor": fid, "dx": dx, "flip_x": flip, "support_pixels": len(x),
                                    "depth_offset_mm": [int(off.min()), int(off.max())]})
    return matches[0] if len(matches) == 1 else None


def frame_data(p):
    d = r.read_json(p)
    by_id = defaultdict(list)
    for ann in d["annotations"]:
        by_id[ann["image_id"]].append(ann)
    return {Path(im["file_name"]).parent.name: (im, by_id[im["id"]]) for im in d["images"]}


def train_sets(real, synthetic, spec, allow, bg, get_real):
    train, val = map(set, (spec["train_object_ids"], spec["val_object_ids"]))
    ids = set(spec["train_image_ids"])
    r.require(not train & val and not ids & set(spec["val_real_image_ids"]), "Group split overlap")
    backgrounds = set(spec["val_real_empty_image_ids"])
    expect = (ids & synthetic.keys()) - set(spec["excluded_mixed_synth_image_ids"])
    expect = {i for i in expect if bg[i] not in backgrounds}
    r.require(expect == allow, f"Background allow-list disagrees for fold {spec['fold']}")
    parents, rows, certs = defaultdict(list), [], {}
    for i in sorted(ids & real.keys()):
        for j, ann in enumerate(real[i][1]):
            at = ann["attributes"]
            parent, label = at["object_id"], at["material"]
            r.require(parent in train, f"{i}: real donor not admitted")
            if label not in r.PAIR:
                continue
            region, inside, features = get_real(i, j)
            if features is not None:
                rows.append((label, parent, False, features))
                parents[label, parent].append((i, ann, region, inside))
    preflight = Counter()
    for i in sorted(allow):
        im, anns = synthetic[i]
        r.require(all(a["attributes"]["object_id"] in train for a in anns), f"{i}: synth donor not admitted")
        if len(anns) != 1 or anns[0]["instance"] != 1:
            preflight["not_one_annotation"] += 1
            continue
        ann, = anns
        at = ann["attributes"]
        parent, label = at["object_id"], at["material"]
        if (label, parent) not in parents:
            preflight["not_usable_pair_parent"] += 1
            continue
        pix, z, m = read_synth(im)
        if set(np.unique(m)) != {0, 1}:
            preflight["not_one_mask_instance"] += 1
            continue
        region = crop(pix, z, m, xyxy(ann))
        preflight["foreground_examined"] += 1
        proof = certify(region, ann, parents[label, parent], im["width"])
        if proof is None:
            preflight["no_unique_foreground_certificate"] += 1
            continue
        v = safe_desc(region)
        if v is None:
            preflight["interior_empty_after_certificate"] += 1
            continue
        rows.append((label, parent, True, v))
        certs[i] = dict(proof, material=label, object_id=parent, background=bg[i])
    return rows, certs, dict(preflight)


def hold_sets(real, spec, index, anchor):
    f = spec["fold"]
    meta = next(x for x in index["folds"] if x["fold"] == f)
    ids = spec["val_real_image_ids"]
    r.require(set(ids) == set(meta["frames"]), f"Fold {f}: invalid mask coverage")
    source = meta["sources"][f"n640_fold_{f:02d}"]
    mfile, dfile = (r.ROOT / source[k] for k in ("masks", "detections"))
    for p, key in [(mfile, "masks_sha256"), (dfile, "detections_sha256")]:
        r.require(r.digest(p) == source[key], f"Changed OOF input: {p}")
    items = {d["image_id"]: d for d in map(json.loads, dfile.read_text().splitlines())}
    found, unscoreable = {}, []
    with np.load(mfile, allow_pickle=False) as masks:
        for fid in ids:
            if not any(p["material"] in r.PAIR for p in anchor[fid]):
                continue
            im, z = r.read_rgbd(real[fid][0])
            h, w = items[fid]["shape"]
            packed = masks[fid]
            r.require(z.shape == (h, w) and packed.shape == (len(anchor[fid]), h, (w+7)//8),
                      f"{fid}: invalid anchor mask shape")
            for j, (q, p, mask) in enumerate(zip(items[fid]["detections"], anchor[fid], packed, strict=True)):
                r.require(all(q[k] == p[k] for k in ("box", "score", "form")), f"{fid}: mask order")
                if p["material"] not in r.PAIR:
                    continue
                m = np.unpackbits(mask, axis=-1, count=w).astype(bool)
                v = safe_desc(crop(im, z, m, p["box"]))
                if v is None:
                    unscoreable.append((fid, j))
                else:
                    found[fid, j] = v
    return found, unscoreable


def predict(rows, xhold, augmented):
    selected = [row for row in rows if augmented or not row[2]]
    labels, objects, domains, x = zip(*selected, strict=True)
    y, group, domain, x = (np.asarray(labels) == "metal", np.asarray(objects), np.asarray(domains), np.asarray(x))

    def fit(mask):
        a, b, g, d = x[mask], y[mask], group[mask], domain[mask]
        n = Counter(zip(g, d, strict=True))
        labels_by_obj = dict(zip(g, b, strict=True))
        by_class = Counter(labels_by_obj.values())
        r.require(len(by_class) == 2 and len(set(zip(g, b, strict=True))) == len(labels_by_obj),
                  "Invalid pair training groups")
        expected = {o: len(labels_by_obj) / (2 * by_class[label]) for o, label in labels_by_obj.items()}
        w = np.array([expected[o] * (SHARE if s else 1-SHARE if (o, True) in n else 1.) / n[o, s]
                      for o, s in zip(g, d, strict=True)])
        r.require(all(np.isclose(w[g == o].sum(), weight) for o, weight in expected.items()),
                  "Parent total weight changed")
        us = StandardScaler().fit(a, sample_weight=w)
        model = LogisticRegression(C=r.C, solver="lbfgs", tol=1e-9, max_iter=1000).fit(
            us.transform(a), b, sample_weight=w)
        return model.predict_proba(us.transform(xhold))[:, 1]

    full = fit(np.ones(len(x), dtype=bool))
    jk = np.stack([fit(group != o) for o in sorted(set(group))])
    return np.column_stack([full, jk.min(axis=0), jk.max(axis=0)])


def obj_counts(rows, is_synth):
    return {m: {"views": sum(label == m and is_synth == syn for label, _, syn, _ in rows),
                "objects": sorted({o for label, o, syn, _ in rows if label == m and syn == is_synth})}
            for m in r.PAIR}


def main(out):
    index = r.read_json(r.CACHE / "mask_index.json")
    manifest = r.read_json(r.MANIFEST)
    specs = manifest["folds"]
    hashes = {path: r.digest(path) for path in [*r.PINNED, r.MANIFEST, r.LABEL, SYNTH, ALLOW, BACKGROUND]}
    r.require(all(hashes[p] == h for p, h in r.PINNED.items()) and
              hashes[r.MANIFEST] == index["fold_manifest_sha256"] and
              all(hashes[p] == manifest["labels_sha256"][f"labels/{p.name}"] for p in (r.LABEL, SYNTH)),
              "Frozen input changed")
    real, synthetic = frame_data(r.LABEL), frame_data(SYNTH)
    allowed, bg = r.read_json(ALLOW), r.read_json(BACKGROUND)
    r.require(set(bg) == set(synthetic) and set(bg.values()) == {i for i in real if not real[i][1]} and
              set(allowed) == {"1", "2", "3", "4", "5"} and all(len(v) == len(set(v)) for v in allowed.values()),
              "Invalid background audit")
    base = r.read_json(r.BASE)
    anchor = r.read_json(r.CACHE / "predictions.json")["n640_rescue"]
    donors = r.restore_donors(specs, anchor, base, r.read_json(r.CACHE / "per_model_predictions.json"))
    val_ids = [i for fold in specs for i in fold["val_real_image_ids"]]
    r.require(len(val_ids) == len(set(val_ids)) and set(val_ids) == set(base) == set(anchor) == set(real),
              "Invalid OOF coverage")

    @lru_cache(maxsize=None)
    def get_real(i, j):
        im, anns = real[i]
        region = real_crop(im, anns[j])
        return region, erode(region[2], 2), safe_desc(region)

    control, candidate = {}, {}
    certificates, fold_counts = {}, {}
    with threadpool_limits(limits=1):
        for spec in specs:
            f = spec["fold"]
            fold_allow = set(allowed[str(f)])
            rows, proof, filters = train_sets(real, synthetic, spec, fold_allow, bg, get_real)
            hold, unscoreable = hold_sets(real, spec, index, anchor)
            score_x = np.asarray(list(hold.values()))
            for pred, aug in [(control, False), (candidate, True)]:
                scores = dict(zip(hold, predict(rows, score_x, aug), strict=True))
                for fid in spec["val_real_image_ids"]:
                    pred[fid] = [dict(p, material=r.decision(p["material"], *scores[fid, j]))
                                 if (fid, j) in scores else p.copy()
                                 for p, j in zip(base[fid], donors[fid], strict=True)]
            certificates[f] = proof
            fold_counts[f] = {"allowed_after_background_filter": len(fold_allow), **filters,
                              "certified_synth_frames": len(proof),
                              "real": obj_counts(rows, False), "certified_synth": obj_counts(rows, True),
                              "unscoreable_hold_donors": unscoreable, "scored_hold_donors": len(hold)}
            print(f"Fold {f}: allowed={len(fold_allow)}, examined={filters.get('foreground_examined', 0)}, "
                  f"certified={len(proof)}, scored donors={len(hold)}", flush=True)
    r.require(any(certificates.values()), "No foreground-certified synth, no estimable treatment")
    gt = {z.image_id: r.metric._parse(z.objects, False, z.image_id)
          for z in r.metric.labels_to_solution(r.LABEL).itertuples(index=False)}
    r.require(r.metric.components(gt, base)["Score"] == 72.78019593033471, "Exact baseline mismatch")
    report = {"input_sha256": {str(p.relative_to(r.ROOT)): h for p, h in hashes.items()},
              "run_sha256": r.digest(Path(__file__)), "base_module_sha256": r.digest(Path(r.__file__)),
              "config": {"synth_fraction": SHARE, "C": r.C, "threshold_full": r.MIN_FULL,
                         "threshold_all_jackknife": r.MIN_JACKKNIFE, "features": r.FEATURES},
              "fold_counts": fold_counts, "comparisons": {}}
    for name, a, b in [("control_vs_frozen", base, control), ("synth_vs_frozen", base, candidate),
                       ("synth_vs_control", control, candidate)]:
        pooled = r.evaluate(gt, a, b, donors)
        fold_reports = {}
        for s in specs:
            ids = set(s["val_real_image_ids"])
            fold_reports[s["fold"]] = r.evaluate({i: g for i, g in gt.items() if i in ids}, a, b, donors)
        report["comparisons"][name] = dict(pooled=pooled, folds=fold_reports)
        print(f"{name}: {pooled['base']['Score']:.4f} -> {pooled['candidate']['Score']:.4f} "
              f"({pooled['delta']:+.4f}); folds {[round(x['delta'], 4) for x in fold_reports.values()]}", flush=True)
    out.mkdir(exist_ok=False)
    for name, obj in [("report", report), ("certificates", certificates),
                      ("real_only_oof", control), ("synth_80_20_oof", candidate)]:
        (out / f"{name}.json").write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "certified")
    args = parser.parse_args()
    if args.out.resolve().parent != HERE or args.out.exists():
        parser.error("--out must be a new direct child of synth01")
    cv2.setNumThreads(1)
    main(args.out)
