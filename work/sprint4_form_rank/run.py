"""Score-only rerank on the frozen bulk_y base; see README.md."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
ROOT = WORK.parent
DATA = ROOT / "dataset"
sys.path[:0] = [str(ROOT), str(WORK), str(ROOT / "fusion")]

import cv2
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits
import score as metric
from material_features import bounds, polygon_mask
from material_split import load_frames
from submission_tools import test_ids, validate_csv

MANIFEST = WORK / "detector_folds_v1.json"
CACHE = WORK / "sprint1_agnostic_oof_v2"
BASE_OOF = WORK / "sprint3_bulk_y_frozen/bulk_y_bottom_oof.json"
BASE_CSV = WORK / "sprint2_volume_bulk_y/submission_807475_bulk_y_bottom.csv"
ANCHOR_TEST = WORK / "sprint1_grisha/submission_fused_max_7897_rescue_small.csv"
FORM_TEST = WORK / "sprint2_l960_anchor/submission_803008_l960_form_only.csv"
WBF = WORK / "test_wbf_v1"
TEST_RAW_CSV = WORK / "submission_wbf6_final.csv"
SCORE = 72.78019593033471
EXPECTED = {
    BASE_OOF: "f0eb64df263afff0130b4e4daf19591031e0f9092374fb9fa6eef1882f1e807e",
    BASE_CSV: "a9d7b2aa53742252c61dc8586c80c105b51945986c1abec1136e3e7cf482b202",
    ANCHOR_TEST: "1b28c7c8a9df5062d3974f87cabe75762cfa4eb10dbe84c70571652144805135",
}
STRENGTH = {"mild": .7, "strong": 1.4}
WEIGHTS = (.4, .4, .2)
MATCH_FIELDS = ("material", "volume", "mass", "take", "take_p")
DINO = WORK / "astra_dino.safetensors"
CLIP = WORK / "astra_clip.bin"


def need(valid, detail):
    if not valid:
        raise ValueError(detail)


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(p.read_text())


def dump(p, x):
    p.write_text(json.dumps(x, allow_nan=False, indent=2) + "\n")


def labelled():
    frames = {fid: (src, im, anns) for src, fid, im, anns in load_frames()}
    label_path = DATA / "labels/real_train.json"
    gt = {i: metric._parse(o, False, i) for i, o, *_ in
          metric.labels_to_solution(label_path).itertuples(index=False, name=None)}
    objects = {i: sorted({a["attributes"]["object_id"] for a in rows})
               for i, (s, _, rows) in frames.items() if s == "real"}
    return frames, gt, objects


def fit_rows(train, spec):
    allow_i, allow_o = set(spec["train_image_ids"]), set(spec["train_object_ids"])
    deny_i = {i for key in ("val_real_image_ids", "val_synth_image_ids",
                            "excluded_mixed_synth_image_ids") for i in spec[key]}
    deny_o = set(spec["val_object_ids"])
    ix = np.array([j for j, (i, _, _, o, _) in enumerate(train) if i in allow_i], dtype=int)
    actual_o = {train[j][3] for j in ix}
    need(actual_o == allow_o and not actual_o & deny_o and
         not {train[j][0] for j in ix} & deny_i, f"Invalid outer train split {spec['fold']}")
    return ix


def training_records(frames):
    return [(fid, im, a, a["attributes"]["object_id"], a["attributes"]["form"])
            for fid, (_, im, anns) in frames.items() for a in anns]


def box_of(p):
    x, y, r, b = p["box"]
    return x, y, r - x, b - y


def records(folder):
    return {r["image_id"]: r for r in map(json.loads, (folder / "base_detections.jsonl").read_text().splitlines())}


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        need(next(reader) == ["image_id", "prediction"], path)
        rows = list(reader)
    need([r[0] for r in rows] == test_ids(), f"{path}: wrong test ID order or coverage")
    return dict(rows)


def test_counts(rows):
    return len(rows), sum(0 if s == "none" else len(s.split(";")) for s in rows.values()), sum(
        s == "none" for s in rows.values())


def same_detection(a, b):
    return tuple(a["box"]) == tuple(b["box"]) and all(a[k] == b[k] for k in ("score", "form"))


def donor(p, anchors):
    overlap, neg = max(((metric._iou(p["box"], a["box"]), -j) for j, a in enumerate(anchors)
                        if a["form"] != p["form"]), default=(0, 0))
    need(overlap >= .5, "Missing alternate form donor")
    return -neg


def prepare_sources(frames, specs, hashes):
    """Frozen boxes plus ordered tuples (image, raw slot, RGB path, raw box, local real mask, shape)."""
    index = read(CACHE / "mask_index.json")
    need(index["fold_manifest_sha256"] == sha(MANIFEST), "Wrong mask manifest")
    anchor_oof = read(CACHE / "predictions.json")["n640_rescue"]
    base_oof = read(BASE_OOF)
    val_sources, oof_map, test_sources, test_map = [], {}, [], {}
    for spec, entry in zip(specs, index["folds"], strict=True):
        f = spec["fold"]
        need(f == entry["fold"] and spec["val_real_image_ids"] == entry["frames"], "Invalid fold index")
        folder = WORK / "detector_oof_v1" / f"fold_{f:02}" / "evaluation"
        meta = entry["sources"][f"n640_fold_{f:02d}"]
        for name, key in (("base_detections.jsonl", "detections_sha256"), ("base_masks.npz", "masks_sha256")):
            path = folder / name
            hashes[path] = sha(path)
            need(hashes[path] == meta[key], f"Unverified OOF {path}")
        raw = records(folder)
        with np.load(folder / "base_masks.npz") as masks:
            for i in spec["val_real_image_ids"]:
                row, old, base = raw[i], anchor_oof[i], base_oof[i]
                h, w = row["shape"]
                arr = masks[i]
                need(row["source"] == "real" and arr.shape == (len(old), h, (w + 7) // 8), i)
                need(len(base) >= len(old), i)
                keys = []
                for j, (r, a) in enumerate(zip(row["detections"], old, strict=True)):
                    need(same_detection(r, a) and same_detection(r, base[j]), (f, i, j))
                    rgb = DATA / frames[i][1]["file_name"]
                    keys.append(len(val_sources))
                    val_sources.append((i, j, rgb, r, arr[j], (h, w)))
                oof_map[i] = (f, keys + [keys[donor(p, old)] for p in base[len(old):]])

    freeze, anchors, wt, form = map(csv_rows, (BASE_CSV, ANCHOR_TEST, TEST_RAW_CSV, FORM_TEST))
    need(test_counts(freeze) == (137, 342, 10), "Unexpected frozen test counts")
    base_test = {i: metric._parse(s, True, i) for i, s in freeze.items()}
    raw = records(WBF)
    need(set(raw) == set(freeze), "WBF test coverage")
    l960_report = read(FORM_TEST.parent / "report.json")
    need(sha(FORM_TEST) == l960_report["test"]["output_sha256"], "Frozen form test hash")
    with np.load(WBF / "base_masks.npz") as masks:
        for i, base in base_test.items():
            a = metric._parse(anchors[i], True, i)
            raw_parsed = metric._parse(wt[i], True, i)
            form_parsed = metric._parse(form[i], True, i)
            need(len(form_parsed) >= len(a), i)
            for p, q in zip(base, form_parsed, strict=True):
                need({k: v for k, v in p.items() if k != "volume"} ==
                     {k: v for k, v in q.items() if k != "volume"}, i)
            row = raw[i]
            h, w = row["shape"]
            arr = masks[i]
            need(arr.shape == (len(raw_parsed), h, (w + 7) // 8), i)
            need(all(same_detection(p, q) for p, q in zip(raw_parsed, row["detections"], strict=True)), i)
            keys = []
            dedup = {}
            for j, p in enumerate(a):
                need(same_detection(p, base[j]), (i, j))
                overlap, k = max((metric._iou(p["box"], q["box"]), k) for k, q in enumerate(raw_parsed)
                                 if q["form"] == p["form"])
                q = raw_parsed[k]
                need(overlap >= .75 and all(q[z] == p[z] for z in ("material", "volume")), (i, j, k))
                if k not in dedup:
                    rgb = next(path for path in (DATA / "real/test" / i).glob("color.*")
                               if path.suffix.lower() in (".png", ".jpg"))
                    dedup[k] = len(test_sources)
                    test_sources.append((i, k, rgb, row["detections"][k], arr[k], (h, w)))
                keys.append(dedup[k])
            test_map[i] = keys + [keys[donor(p, a)] for p in base[len(a):]]
    return base_oof, oof_map, val_sources, freeze, base_test, test_map, test_sources


def mask_features(m):
    # Isotropic downsample for low, comparable sensitivity to annotation/model boundaries.
    x, y, w, h = cv2.boundingRect(m.astype("uint8"))
    need(w > 0 and h > 0, "Empty mask")
    scale = 128 / max(w, h)
    m = cv2.resize(m[y:y+h, x:x+w].astype("float32"),
                   (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA) >= .5
    a = int(m.sum())
    cnts, _ = cv2.findContours(m.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(cnts))
    filled = np.zeros_like(m, dtype="uint8")
    cv2.fillConvexPoly(filled, hull, 1)
    _, _, stats, _ = cv2.connectedComponentsWithStats(m.astype("uint8"), connectivity=8)
    yy, xx = np.nonzero(m)
    pts = np.column_stack((xx, yy)).astype(np.float64)
    ev = np.linalg.eigvalsh((pts - pts.mean(0)).T @ (pts - pts.mean(0)) / a)
    peri = sum(cv2.arcLength(c, True) for c in cnts)
    erode = cv2.erode(np.pad(m.astype("uint8"), 3), np.ones((7, 7), "uint8")).sum()
    short, long = sorted(cv2.minAreaRect(hull)[1])
    return [math.log(w / h), a / m.size, a / max(filled.sum(), a), peri / math.sqrt(a),
            len(cnts), stats[1:, cv2.CC_STAT_AREA].max() / a,
            ev[0] / max(ev[-1], 1e-9), erode / a, a / max(short * long, a)]


def gt_features(train):
    ys = []
    for _, im, a, _, _ in train:
        l, t, r, b = bounds(a["bbox"], im["width"], im["height"], .1)
        ys.append(mask_features(polygon_mask(a["segmentation"], r-l, b-t, l, t)))
    return np.array(ys)


def make_crops(src):
    from astra_deploy import masked_crop
    crops, geom, rgb_signatures = [], [], {}
    prev = None
    for i, j, rgb, p, packed, (h, w) in src:
        if rgb != prev:
            color = cv2.imread(str(rgb))
            need(color is not None and color.shape[:2] == (h, w), (i, rgb))
            rgb_signatures[str(rgb.relative_to(ROOT))] = sha(rgb)
            prev = rgb
        l, t, r, b = bounds(box_of(p), w, h, .1)
        m = np.unpackbits(packed[t:b], axis=-1, count=w)[:, l:r].astype(bool)
        crops.append(masked_crop(color[t:b, l:r], m))
        geom.append(mask_features(m))
    return crops, np.array(geom), rgb_signatures


def encode(crops, device):
    import torch
    import timm
    from astra_clip import open_clip
    from PIL import Image
    from safetensors.torch import load_file
    from timm.data import create_transform, resolve_model_data_config
    torch.set_num_threads(4)
    with torch.inference_mode():
        m = timm.create_model("vit_base_patch16_224.dino", pretrained=False, num_classes=0)
        m.load_state_dict(load_file(str(DINO)))
        transform = create_transform(**resolve_model_data_config(m), is_training=False)
        m.eval().to(device)
        images = [Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) for x in crops]
        d = normalize(np.concatenate([m(torch.stack([transform(x) for x in images[j:j+16]]).to(device))
                                      .cpu().numpy() for j in range(0, len(images), 16)]))
        del m
        m, _, transform = open_clip.create_model_and_transforms("ViT-B-32", pretrained=str(CLIP), device=device)
        m.eval()
        c = normalize(np.concatenate([m.encode_image(torch.stack([transform(x) for x in images[j:j+16]]).to(device))
                                      .cpu().numpy() for j in range(0, len(images), 16)]))
    return d, c


def eval_features(val, test, hashes, device):
    import torch
    src = val + test
    crops, geom, rgb = make_crops(src)
    h = hashlib.sha256()
    for crop in crops:
        h.update(crop.tobytes())
    signature = {"ordered_crops_sha256": h.hexdigest(), "rgb": rgb, "frozen_weights":
                 {p.name: hashes[p] for p in (DINO, CLIP)}, "device": device, "torch": str(torch.__version__)}
    desc, cache = OUT / "eval_cache.json", OUT / "eval_cache.npz"
    if not (cache.exists() and desc.exists() and read(desc) == signature):
        d, c = encode(crops, device)
        np.savez_compressed(cache, dino=d, clip=c)
        dump(desc, signature)
    with np.load(cache) as z:
        d, c = z["dino"], z["clip"]
    need(d.shape == (len(src), 768) and c.shape == (len(src), 512) and
         np.isfinite(d).all() and np.isfinite(c).all(), "Invalid eval cache")
    hashes[desc], hashes[cache] = sha(desc), sha(cache)
    return d, c, geom


def train_features(train, hashes):
    keys = [f"{i}_{a['id']}" for i, _, a, _, _ in train]
    banks = []
    for which in ("astra_", "astra_clip_"):
        ids, features = [], []
        for part in (which, which + "extra_"):
            p, q = WORK / (part + "ids.json"), WORK / (part + "embeddings.npy")
            hashes[p], hashes[q] = sha(p), sha(q)
            new = read(p)
            rows = np.load(q)
            need(len(new) == len(rows), p)
            ids.extend(new)
            features.append(rows)
        lookup = {key: j for j, key in enumerate(ids)}
        need(len(lookup) == len(ids) == len(keys) == 892 and set(lookup) == set(keys), which)
        banks.append(normalize(np.concatenate(features)[[lookup[k] for k in keys]]))
    return *banks, gt_features(train)


def predict(train, tx, vx, ix, hold):
    y = np.array([train[j][4] == "bulk" for j in ix])
    obj = [train[j][3] for j in ix]
    counts = Counter(obj)
    w = np.array([1 / counts[o] for o in obj])
    for label in (False, True):
        w[y == label] /= w[y == label].sum()
    w /= w.mean()
    pred = np.zeros(len(hold))
    for k, (a, b, mix) in enumerate(zip(tx, vx, WEIGHTS, strict=True)):
        lr = LogisticRegression(C=1, max_iter=1000)
        if k == 2:
            cls = make_pipeline(StandardScaler(), lr)
            cls.fit(a[ix], y, standardscaler__sample_weight=w, logisticregression__sample_weight=w)
        else:
            cls = lr.fit(a[ix], y, sample_weight=w)
        pred += mix * cls.predict_proba(b[hold])[:, 1]
    return pred


def rerank(base, prob, strength):
    """Shrink a whole frame's logit update just enough to preserve stable score order."""
    if not base:
        return base
    old = np.array([p["score"] for p in base])
    need(np.all((old > 0) & (old < 1)), "Expected detector score strictly between 0 and 1")
    logits = logit(old)
    prob = np.array(prob)
    need(len(prob) == len(old) and np.all((prob >= 0) & (prob <= 1)), "Posterior out of range")
    is_bulk = np.array([p["form"] == "bulk" for p in base])
    change = strength * (2 * np.where(is_bulk, prob, 1 - prob) - 1)
    order = np.argsort(-old, kind="stable")
    a, b = order[:-1], order[1:]
    closing = change[b] - change[a]
    tied_or_closing = (closing > 0)
    step = min(1., float(np.min(.99 * (logits[a] - logits[b])[tied_or_closing] /
                               closing[tied_or_closing], initial=np.inf)))
    if step == 0:
        return base
    new = expit(logits + step * change)
    if not np.array_equal(order, np.argsort(-new, kind="stable")):
        return base  # exact score ties / roundoff must not perturb _match
    return [dict(p, score=float(s)) for p, s in zip(base, new, strict=True)]


def metrics(gt, pred):
    return dict(metric.components(gt, pred), AP={f: metric._average_precision(gt, pred, f) for f in metric.FORMS})


def paired(gt, old, pred):
    a, b = metrics(gt, {i: old[i] for i in gt}), metrics(gt, {i: pred[i] for i in gt})
    return {"base": a, "candidate": b, "delta": {k: b[k] - a[k] for k in ("Score", "D", "M", "V", "S")}}


def removal(gt, objects, old, pred):
    deltas = {}
    for obj in sorted({o for ids in objects.values() for o in ids}):
        ids = [i for i in gt if obj not in objects[i]]
        g = {i: gt[i] for i in ids}
        deltas[obj] = (metric.components(g, {i: pred[i] for i in ids})["Score"] -
                       metric.components(g, {i: old[i] for i in ids})["Score"])
    worst = min(deltas, key=deltas.get)
    return {"removed_object": worst, "min_delta": deltas[worst], "all_deltas": deltas}


def churn(gt, old, new):
    diff = Counter({k: 0 for k in ("match_slot", "matched_state", *MATCH_FIELDS)})
    for i, gs in gt.items():
        a, b = old[i], new[i]
        need(all({k:v for k,v in p.items() if k != "score"} == {k:v for k,v in q.items() if k != "score"}
                 for p, q in zip(a, b, strict=True)), "OOF non-score change")
        ids_a, ids_b = ({id(p): j for j, p in enumerate(ps)} for ps in (a, b))
        for p, q in zip(metric._match(gs, a), metric._match(gs, b), strict=True):
            diff["match_slot"] += ids_a.get(id(p)) != ids_b.get(id(q))
            diff["matched_state"] += (p is None) != (q is None)
            for k in MATCH_FIELDS:
                diff[k] += (p or {}).get(k) != (q or {}).get(k)
    return dict(diff)


def token_changes(old, new, allowed):
    need(list(old) == list(new) and test_counts(old) == test_counts(new), "Changed test inventory")
    counts = Counter()
    frames = Counter()
    for i, text in old.items():
        if text == "none":
            need(new[i] == text, i)
            continue
        changed = set()
        for a, b in zip(text.split(";"), new[i].split(";"), strict=True):
            for k, (x, y) in enumerate(zip(a.split(), b.split(), strict=True)):
                if x != y:
                    need(k in allowed, (i, k, x, y))
                    counts[k] += 1
                    changed.add(k)
        frames.update(changed)
    return {"token_rows": dict(counts), "token_frames": dict(frames)}


def replace_tokens(text, detections, index_to_key):
    if text == "none":
        need(not detections, "Unexpected none")
        return text
    pieces = []
    for p, d in zip(text.split(";"), detections, strict=True):
        fields = p.split()
        for k, attr in index_to_key.items():
            if float(fields[k]) != d[attr]:
                fields[k] = repr(float(d[attr]))
        pieces.append(" ".join(fields))
    return ";".join(pieces)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("image_id", "prediction"))
        w.writerows(rows.items())


def main(device):
    for p, expected in EXPECTED.items():
        need(sha(p) == expected, f"Frozen input changed: {p}")
    hashes = {p: sha(p) for p in (
        MANIFEST, DATA / "labels/real_train.json", DATA / "labels/synth_train.json", ROOT / "sample_submission.csv",
        CACHE / "mask_index.json", CACHE / "predictions.json", *EXPECTED, FORM_TEST,
        FORM_TEST.parent / "report.json", TEST_RAW_CSV, WBF / "base_masks.npz", WBF / "base_detections.jsonl",
        DINO, CLIP, ROOT / "score.py", Path(__file__).resolve())}
    specs = read(MANIFEST)["folds"]
    frames, gt, objects = labelled()
    base, omap, v, frozen, base_test, tmap, t = prepare_sources(frames, specs, hashes)
    need(len(gt) == len(base) == len(omap) == 263 and set(gt) == set(base) == set(omap), "OOF coverage")
    need(abs(metric.components(gt, base)["Score"] - SCORE) < 1e-12, "Wrong OOF baseline")
    train = training_records(frames)
    tx = train_features(train, hashes)
    print(f"Verified {len(v)} OOF + {len(t)} test anchor crops; caching frozen embeddings ({device})", flush=True)
    vx = eval_features(v, t, hashes, device)
    probability = np.full(len(v), np.nan)
    fit_counts = {}
    with threadpool_limits(4):
        for spec in specs:
            k = spec["fold"]
            ix = fit_rows(train, spec)
            hold = np.array(sorted({j for i in spec["val_real_image_ids"] for j in omap[i][1]}))
            fit_counts[k] = {"gt_train": len(ix), "real_pred_masks": len(hold)}
            need(np.isnan(probability[hold]).all(), f"Fold {k}: duplicate anchor")
            probability[hold] = predict(train, tx, vx, ix, hold)
    need(np.isfinite(probability).all(), "Missing prediction")
    candidates, report = {}, {}
    for key, strength in STRENGTH.items():
        p = {i: rerank(base[i], probability[omap[i][1]], strength) for i in base}
        pooled = paired(gt, base, p)
        f = {s["fold"]: paired({i: gt[i] for i in s["val_real_image_ids"]}, base, p) for s in specs}
        changed = churn(gt, base, p)
        loo = removal(gt, objects, base, p)
        passed = (pooled["delta"]["D"] > 0 and pooled["delta"]["Score"] > 0 and
                  sum(x["delta"]["Score"] > 0 for x in f.values()) >= 3 and loo["min_delta"] > 0 and
                  not any(changed.values()) and all(pooled["delta"][k] == 0 for k in ("M", "V", "S")))
        report[key] = dict(pooled=pooled, folds=f, attr_churn=changed, best_object_removal=loo, passed=passed)
        candidates[key] = p
        print(key, pooled["delta"], "folds", [round(x["delta"]["Score"], 5) for x in f.values()],
              "min LOO", loo["min_delta"], "passed", passed, flush=True)
    selected = max((k for k in STRENGTH if report[k]["passed"]),
                   key=lambda k: report[k]["pooled"]["delta"]["Score"], default="baseline")
    oof = OUT / "selected_oof.json"
    if selected == "baseline":
        oof.write_bytes(BASE_OOF.read_bytes())
    else:
        dump(oof, candidates[selected])
    print(f"OOF ready: {oof.relative_to(ROOT)} [{selected}] {sha(oof)}", flush=True)
    sub = OUT / "submission_selected.csv"
    if selected == "baseline":
        sub.write_bytes(BASE_CSV.read_bytes())
    else:
        with threadpool_limits(4):
            pr = predict(train, tx, vx, np.arange(len(train)), np.arange(len(v), len(v) + len(t)))
        changed = {i: replace_tokens(frozen[i], rerank(base_test[i], pr[tmap[i]], STRENGTH[selected]),
                                     {4: "score"}) for i in frozen}
        write_rows(sub, changed)
    test_diff = token_changes(frozen, csv_rows(sub), {4})
    validate_csv(sub)
    dump(OUT / "report.json", dict(selected=selected, baseline=metrics(gt, base), modes=report,
         folds_fit=fit_counts, scores={"oof": sha(oof), "csv": sha(sub)}, test=test_diff,
         test_counts=list(test_counts(frozen)), weights=WEIGHTS, logit_strengths=STRENGTH, C=1,
         hashes={str(p.relative_to(ROOT)): h for p, h in hashes.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    main(args.device)
