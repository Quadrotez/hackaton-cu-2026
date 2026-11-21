"""Build 4-channel crops for the frozen adapter inventory (boxes-only donor mode)."""
import csv, json, math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
OUT = ROOT / "work" / "inventory_crops"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 128


def crop4(color, depth, box, size=SIZE):
    h, w = color.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    l = max(0, math.floor(x1 - .1 * bw)); t = max(0, math.floor(y1 - .1 * bh))
    r = min(w, math.ceil(x1 + 1.1 * bw)); b = min(h, math.ceil(y1 + 1.1 * bh))
    if r <= l or b <= t: return None
    m = np.zeros((h, w), bool); m[t:b, l:r] = True
    rgb = color[t:b, l:r]
    dep = np.clip(depth[t:b, l:r].astype(np.float32) / 2000.0, 0, 1) * 255
    rgb = np.where(m[t:b, l:r, None], rgb, 0)
    dep = np.where(m[t:b, l:r], dep, 0).astype(np.uint8)
    interp = cv2.INTER_AREA if max(rgb.shape[:2]) > size else cv2.INTER_LINEAR
    rgb = cv2.resize(rgb, (size, size), interpolation=interp)[..., ::-1]
    dep = cv2.resize(dep, (size, size), interpolation=interp)
    return np.concatenate([rgb, dep[..., None]], axis=-1).transpose(2, 0, 1).astype(np.uint8)


def frame_dir(split, fid):
    return DATA / "real" / split / fid


for split, inv in (("oof", ROOT / "work" / "crop_inventory_oof.csv"),
                   ("test", ROOT / "work" / "crop_inventory_test.csv")):
    rows = []
    for r in csv.DictReader(open(inv)):
        fid = r["image_id"]
        d = frame_dir("train" if split == "oof" else "test", fid)
        cp = d / "color.png"
        if not cp.exists(): cp = d / "color.jpg"
        color = cv2.imread(str(cp), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(d / "depth.png"), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None: continue
        c = crop4(color, depth, [r["x1"], r["y1"], r["x2"], r["y2"]])
        if c is None: continue
        key = f"{split}_{fid}_{r['hyp_idx']}"
        np.save(OUT / f"{key}.npy", c)
        rows.append({"key": key, "split": split, "image_id": fid,
                     "hyp_idx": r["hyp_idx"], "fold": r.get("fold", ""),
                     "detection_sha256": r["detection_sha256"], "base_form": r["base_form"]})
    with (OUT / f"{split}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(split, len(rows))
