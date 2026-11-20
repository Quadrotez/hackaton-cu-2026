"""Локальный запуск бейзлайна Астрамис: обучение YOLO, валидация, submission.csv.

Пример:
    python train_and_submit.py --data /home/limo/Nextcloud/dataset --epochs 10
"""

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import score as metric

FORMS = ["bulk", "solid"]
HEAVY_G = 500.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="папка dataset с camera.json, labels/, real/, synth/")
    ap.add_argument("--work", default="work", help="куда писать промежуточные файлы")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--include-synth", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def color_path(frame_dir):
    p = frame_dir / "color.png"
    return p if p.exists() else frame_dir / "color.jpg"


def load_frames(data, labels_file):
    path = data / "labels" / labels_file
    if not path.exists():
        raise SystemExit(f"нет разметки: {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    anns = defaultdict(list)
    for a in info["annotations"]:
        anns[a["image_id"]].append(a)
    frames = []
    for im in info["images"]:
        d = data / Path(im["file_name"]).parent
        if not color_path(d).exists() or not (d / "depth.png").exists():
            continue
        frames.append({"dir": d, "id": d.name, "w": im["width"], "h": im["height"], "anns": anns[im["id"]]})
    return frames


def prepare_yolo(frames_train, frames_val, yolo_dir):
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split, frames in (("train", frames_train), ("val", frames_val)):
        (yolo_dir / "images" / split).mkdir(parents=True)
        (yolo_dir / "labels" / split).mkdir(parents=True)
        for f in frames:
            src = color_path(f["dir"])
            shutil.copy(src, yolo_dir / "images" / split / f"{f['id']}{src.suffix}")
            lines = []
            for a in f["anns"]:
                form = a["attributes"]["form"]
                if form not in FORMS:
                    continue
                seg = max(a["segmentation"], key=len)
                pts = np.array(seg, dtype=float).reshape(-1, 2) / [f["w"], f["h"]]
                lines.append(f"{FORMS.index(form)} " + " ".join(f"{v:.5f}" for v in pts.clip(0, 1).reshape(-1)))
            (yolo_dir / "labels" / split / f"{f['id']}.txt").write_text("\n".join(lines))
    (yolo_dir / "data.yaml").write_text(
        f"path: {yolo_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(FORMS))
    )


def raw_volume_ml(depth_mm, mask, plane, cam, cell=0.004):
    v, u = np.nonzero(mask & (depth_mm > 0))
    if len(v) < 30:
        return None
    z = depth_mm[v, u] / 1000.0
    pts = np.stack([(u - cam["cx"]) / cam["fx"] * z, (v - cam["cy"]) / cam["fy"] * z, z], axis=1)
    n, d = np.array(plane[:3]), plane[3]
    height = pts @ n + d
    e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    keys = np.floor(pts @ e1 / cell).astype(np.int64) * 100000 + np.floor(pts @ e2 / cell).astype(np.int64)
    order = np.argsort(keys)
    keys, height = keys[order], height[order]
    starts = np.r_[0, np.nonzero(np.diff(keys))[0] + 1]
    tops = np.maximum.reduceat(height, starts)
    return float(np.clip(tops, 0, None).sum() * cell * cell * 1e6)


def read_frame(frame_dir):
    depth = cv2.imread(str(frame_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"не читается depth: {frame_dir}")
    plane = json.loads((frame_dir / "meta.json").read_text(encoding="utf-8"))["floor_plane"]
    return depth, plane


def to_string(objs):
    if not objs:
        return "none"
    return ";".join(
        f"{o['box'][0]:.1f} {o['box'][1]:.1f} {o['box'][2]:.1f} {o['box'][3]:.1f} {o['score']:.3f} "
        f"{o['form']} {o['material']} {o['volume']:.0f} {o['mass']:.0f} {o['take']}"
        for o in objs
    )


def main():
    args = parse_args()
    data = Path(args.data)
    if not (data / "camera.json").exists():
        raise SystemExit(f"нет camera.json в {data}")
    random.seed(args.seed)
    cam = json.loads((data / "camera.json").read_text(encoding="utf-8"))
    work = Path(args.work)
    work.mkdir(exist_ok=True)

    real = load_frames(data, "real_train.json")
    objects = sorted({a["attributes"]["object_id"] for f in real for a in f["anns"]})
    val_objects = set(objects[::5])
    is_val = lambda f: any(a["attributes"]["object_id"] in val_objects for a in f["anns"])
    val = [f for f in real if is_val(f)]
    train = [f for f in real if not is_val(f)]
    if args.include_synth:
        synth = load_frames(data, "synth_train.json")
        # убираем синтетику с предметами из валидации
        synth = [f for f in synth if not is_val(f)]
        train += synth
    print(f"train={len(train)} val={len(val)} objects={len(objects)}", flush=True)

    materials = [a["attributes"]["material"] for f in train for a in f["anns"]]
    material_guess = max(set(materials), key=materials.count)
    typical_volume = float(np.median([a["attributes"]["volume_ml"] for f in train for a in f["anns"]]))
    with open(data / "densities.csv", encoding="utf-8") as fh:
        density = {f"{r['form']}_{r['material']}": float(r["density_g_per_ml"]) for r in csv.DictReader(fh)}
    print(f"материал по умолчанию={material_guess}, запасной объём={typical_volume:.0f} мл", flush=True)

    yolo_dir = work / "yolo"
    prepare_yolo(train, val, yolo_dir)
    model = YOLO(args.model)
    model.train(
        data=str(yolo_dir / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
        project=str(work.resolve()), name="seg", exist_ok=True, plots=False, verbose=False,
    )
    model = YOLO(str(work / "seg" / "weights" / "best.pt"))

    def predict(frame_dir):
        result = model(str(color_path(frame_dir)), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        if result.masks is None:
            return []
        try:
            depth, plane = read_frame(frame_dir)
        except Exception:
            return []
        h, w = depth.shape
        out = []
        for box, conf, cls, poly in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
            result.boxes.cls.tolist(), result.masks.xy,
        ):
            form = FORMS[int(cls)]
            material = material_guess
            volume = None
            if len(poly) >= 3:
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [poly.round().astype(np.int32)], 1)
                raw = raw_volume_ml(depth, m > 0, plane, cam)
                if raw and raw > 1:
                    volume = raw
            if volume is None:
                volume = typical_volume
            mass = volume * density.get(f"{form}_{material}", 1.0)
            x1, y1, x2, y2 = box
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            out.append({
                "box": (x1, y1, x2, y2), "score": conf, "form": form, "material": material,
                "volume": volume, "mass": mass, "take": int(mass <= HEAVY_G),
            })
        return out

    gt, pred = {}, {}
    for f in val:
        gt[f["id"]] = [{
            "box": (a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]),
            "form": a["attributes"]["form"], "material": a["attributes"]["material"],
            "volume": a["attributes"]["volume_ml"], "take": int(a["attributes"]["mass_g"] <= HEAVY_G),
        } for a in f["anns"]]
        pred[f["id"]] = predict(f["dir"])
    res = metric.components(gt, pred)
    print(f"VALIDATION Score {res['Score']:.2f} | D {res['D']:.3f} M {res['M']:.3f} V {res['V']:.3f} S {res['S']:.3f}", flush=True)

    test_dirs = sorted(p for p in (data / "real" / "test").iterdir() if p.is_dir())
    rows = [{"image_id": d.name, "prediction": to_string(predict(d))} for d in test_dirs]
    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"ГОТОВО: {out}, кадров {len(rows)}, с объектами {sum(r['prediction'] != 'none' for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
