"""Prepare 4-channel crops (RGB + masked normalized depth) for the crop model.

Real annotations use polygon masks; synth uses instance mask.png. Only synth
frames whose object ids are not in the validation fold are kept (no leakage).
Test crops come from the verified 78.97 submission boxes with rectangular masks.
"""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
OUT = ROOT / "work" / "crops4d"
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = json.loads((ROOT / "work" / "team_pipeline" / "work" / "cv_folds.json").read_text())
VAL_OF_OBJ = {}
for f in FOLDS:
    for oid in f["val_object_ids"]:
        VAL_OF_OBJ[oid] = f["fold"]
FORMS = ["bulk", "solid"]
MATERIALS = ["glass", "metal", "mineral", "paper", "plastic", "wood"]


def crop4(color, depth, mask, bbox, size=128):
    h, w = color.shape[:2]
    x, y, bw, bh = bbox
    pad = .1
    l = max(0, int(x - bw * pad)); t = max(0, int(y - bh * pad))
    r = min(w, int(x + bw * (1 + pad))); b = min(h, int(y + bh * (1 + pad)))
    if r <= l or b <= t:
        return None
    cm = mask[t:b, l:r].astype(bool)
    if not cm.any():
        return None
    rgb = color[t:b, l:r]
    dep = depth[t:b, l:r].astype(np.float32)
    dep = np.clip(dep / 2000.0, 0, 1) * 255
    dep = np.where(cm, dep, 0).astype(np.uint8)
    rgb = np.where(cm[..., None], rgb, 0)
    interp = cv2.INTER_AREA if max(rgb.shape[:2]) > size else cv2.INTER_LINEAR
    rgb = cv2.resize(rgb, (size, size), interpolation=interp)
    dep = cv2.resize(dep, (size, size), interpolation=interp)
    return np.concatenate([rgb[..., ::-1], dep[..., None]], axis=-1).transpose(2, 0, 1).astype(np.uint8)


def load_labels(name):
    info = json.loads((DATA / "labels" / name).read_text())
    anns = defaultdict(list)
    for a in info["annotations"]:
        anns[a["image_id"]].append(a)
    return info, anns


records = []
for source, name in (("real", "real_train.json"), ("synth", "synth_train.json")):
    info, anns = load_labels(name)
    for im in info["images"]:
        fid = Path(im["file_name"]).parent.name
        d = DATA / Path(im["file_name"]).parent
        cp = d / "color.png"
        if not cp.exists(): cp = d / "color.jpg"
        color = cv2.imread(str(cp), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(d / "depth.png"), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None: continue
        inst = None
        if source == "synth":
            inst = cv2.imread(str(d / "mask.png"), cv2.IMREAD_UNCHANGED)
        for a in anns[im["id"]]:
            oid = str(a["attributes"]["object_id"])
            fold = VAL_OF_OBJ.get(oid)
            if fold is None: continue
            if source == "synth" and fold in ():
                continue
            m = np.zeros((im["height"], im["width"]), np.uint8)
            if inst is not None:
                m[inst == a["instance"]] = 1
            else:
                for poly in a["segmentation"]:
                    p = np.array(poly, float).reshape(-1, 2).round().astype(np.int32)
                    if len(p) >= 3: cv2.fillPoly(m, [p], 1)
            c = crop4(color, depth, m, a["bbox"])
            if c is None: continue
            key = f"{fid}_{a['id']}"
            np.save(OUT / f"{key}.npy", c)
            records.append({"key": key, "image_id": fid, "ann_id": a["id"],
                            "fold": fold, "source": source, "object_id": oid,
                            "form": a["attributes"]["form"], "material": a["attributes"]["material"]})

with (OUT / "train.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["key", "image_id", "ann_id", "fold", "source", "object_id", "form", "material"])
    w.writeheader(); w.writerows(records)
print("train crops", len(records), "real", sum(r["source"] == "real" for r in records),
      "synth", sum(r["source"] == "synth" for r in records))

# ---- test crops from verified submission boxes ----
trec = []
sub = list(csv.DictReader(open(ROOT / "best_submission.csv")))
for fi, r in enumerate(sub):
    fid = r["image_id"]
    d = DATA / "real" / "test" / fid
    cp = d / "color.png"
    if not cp.exists(): cp = d / "color.jpg"
    color = cv2.imread(str(cp), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(d / "depth.png"), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None: continue
    if r["prediction"] == "none": continue
    for j, obj in enumerate(r["prediction"].split(";")):
        f = obj.split()
        x1, y1, x2, y2 = map(float, f[:4])
        m = np.zeros(color.shape[:2], np.uint8)
        m[int(y1):int(y2), int(x1):int(x2)] = 1
        c = crop4(color, depth, m, (x1, y1, x2 - x1, y2 - y1))
        if c is None: continue
        key = f"{fid}_{j}"
        np.save(OUT / f"test_{key}.npy", c)
        trec.append({"key": key, "image_id": fid, "idx": j, "form": f[5], "material": f[6]})
with (OUT / "test.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["key", "image_id", "idx", "form", "material"])
    w.writeheader(); w.writerows(trec)
print("test crops", len(trec))
