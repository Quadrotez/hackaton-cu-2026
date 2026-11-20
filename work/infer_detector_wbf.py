"""Fuse cached Astramis detections with per-model weights and optional nano support.

--config reads a JSON list of {"path": "work/model.pt", "weight": 1, "group":
"primary"} records. Paths are relative to the repository root unless absolute.
Use group "nano" for a model that may support but never create a cluster.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from material_features import bounds
from material_split import DATA, WORK
from predict_pipeline import Detection, unpack_yolo
from submission_tools import metric, test_ids


CONF = .025
DETECTOR_IOU = .7
CACHE_VERSION = 1
GRISHA_SHA256 = "24487b0e4f23da64ab082c453c55ace602f71f1c35a045015b991deb605b3ac5"
WEIGHTS = tuple(WORK / "detector_oof_v1" / f"fold_{i:02d}" / "train" / "weights" / "best.pt"
                for i in range(1, 6)) + (WORK / "grisha_yolo.pt",)


@dataclass(frozen=True)
class ModelSpec:
    path: Path
    weight: float = 1.
    group: str = "primary"


def model_specs(config):
    if config is None:
        return [ModelSpec(path) for path in WEIGHTS]
    rows = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON list of models")
    specs = []
    for row in rows:
        if not isinstance(row, dict) or "path" not in row or set(row) - {"path", "weight", "group"}:
            raise ValueError(f"Invalid model config: {row!r}")
        weight, group = row.get("weight", 1.), row.get("group", "primary")
        if (type(weight) not in (int, float) or not np.isfinite(weight) or weight <= 0
                or group not in ("primary", "nano")):
            raise ValueError(f"Expected positive finite weight and primary/nano group: {row!r}")
        specs.append(ModelSpec((WORK.parent / row["path"]).resolve(), float(weight), group))
    if len({s.path for s in specs}) != len(specs) or not any(s.group == "primary" for s in specs):
        raise ValueError("Use distinct checkpoint paths and at least one primary model")
    return specs


@dataclass
class Cluster:
    form: str
    members: dict[int, Detection]
    box: tuple[float, float, float, float]

    def add(self, k, detection, weights):
        self.members[k] = detection
        boxes = np.asarray([d.box for d in self.members.values()])
        votes = [weights[i] * d.score for i, d in self.members.items()]
        self.box = tuple(np.average(boxes, axis=0, weights=votes).tolist())


def fuse(per_model, hw, threshold=.55, *, weights=None, nano=()):
    """One vote per model; nano only joins same-class clusters formed by primary models."""
    n_models = len(per_model)
    weights = tuple(weights) if weights is not None else (1.,) * n_models
    nano = frozenset(nano)
    total_weight = sum(weights)
    if (not n_models or len(weights) != n_models or not all(np.isfinite(w) and w > 0 for w in weights)
            or not np.isfinite(total_weight) or not nano < set(range(n_models))
            or not 0 < threshold < 1):
        raise ValueError("Invalid model weights, nano indices or IoU threshold")
    h, w = hw
    flat = []
    for k, detections in enumerate(per_model):
        for d in detections:
            if not (d.form in metric.FORMS and np.isfinite(d.score) and CONF <= d.score <= 1
                    and d.mask.shape == hw and np.isfinite(d.box).all()):
                raise ValueError(f"Invalid detection from model {k}: {d.box}, {d.score}, {d.form}")
            x0, y0, x1, y1 = np.clip(d.box, (0, 0, 0, 0), (w, h, w, h)).tolist()
            if x1 > x0 and y1 > y0 and usable_mask(d.mask, (x0, y0, x1, y1)):
                flat.append((k, Detection((x0, y0, x1, y1), d.score, d.form, d.mask)))
    groups = {g: [] for g in metric.FORMS}
    ordered = sorted(flat, key=lambda x: -(weights[x[0]] * x[1].score))
    for k, d in ordered:
        if k in nano:
            continue
        best = max(groups[d.form], key=lambda c: metric._iou(c.box, d.box), default=None)
        if best is not None and metric._iou(best.box, d.box) >= threshold:
            if k not in best.members:  # Already present means the higher-score copy won.
                best.add(k, d, weights)
        else:
            groups[d.form].append(Cluster(d.form, {k: d}, d.box))
    if nano:
        # Use the original primary box, never the already nano-shifted box, for
        # eligibility. Multiple nano models cannot pull in an unanchored box by chaining.
        anchors = {form: [(c.box, c) for c in clusters] for form, clusters in groups.items()}
        for k, d in ordered:
            if k not in nano:
                continue
            overlap, c = max(((metric._iou(b, d.box), c) for b, c in anchors[d.form]),
                             key=lambda pair: pair[0], default=(0., None))
            if c is not None and overlap >= threshold and k not in c.members:
                c.add(k, d, weights)
    ranked = []
    for k in metric.FORMS:
        for cluster in groups[k]:
            score = sum(weights[i] * d.score for i, d in cluster.members.items()) / total_weight
            primary = (j for j, d in cluster.members.items() if j not in nano and usable_mask(d.mask, cluster.box))
            chosen = max(primary, key=lambda j: cluster.members[j].score, default=None)
            if chosen is None:
                raise ValueError("No primary member's mask intersects the fused box crop")
            mask = cluster.members[chosen].mask
            ranked.append(({
                "box": cluster.box, "score": score, "form": k, "votes": sorted(cluster.members),
                "mask_model": chosen,
            }, mask))
    return sorted(ranked, key=lambda x: -x[0]["score"])[:metric.MAX_PER_IMAGE]


def usable_mask(mask, box):
    h, w = mask.shape
    x0, y0, x1, y1 = box
    l, t, r, b = bounds((x0, y0, x1 - x0, y1 - y0), w, h, .1)
    return bool(mask[t:b, l:r].any())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_checkpoint(path):
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ.setdefault("YOLO_OFFLINE", "true")
    from ultralytics import YOLO

    classes = []
    for g in torch.serialization.get_unsafe_globals_in_checkpoint(path):
        if not (g.startswith(("torch.nn.modules.", "ultralytics.nn.modules."))
                or g == "ultralytics.nn.tasks.SegmentationModel"):
            raise ValueError(f"Unexpected checkpoint global: {g}")
        mod, name = g.rsplit(".", 1)
        obj = getattr(import_module(mod), name)
        if not isinstance(obj, type) or not issubclass(obj, torch.nn.Module):
            raise ValueError(f"Expected a known network layer: {g}")
        classes.append(obj)
    with torch.serialization.safe_globals(classes):
        model = YOLO(str(path))
    names = dict(model.names)
    if model.task != "segment" or set(names.values()) != set(metric.FORMS) or len(names) != 2:
        raise ValueError(f"Unexpected task/classes in {path}: {model.task}, {names}")
    size = model.ckpt["train_args"]["imgsz"]
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"Unexpected checkpoint inference size in {path}: {size}")
    return model, {"architecture": model.model.yaml.get("yaml_file"), "model_class": type(model.model).__name__,
                   "imgsz": size, "names": names}


def test_image_path(i):
    root = DATA / "real" / "test" / i
    return next(p for p in (root / "color.png", root / "color.jpg") if p.is_file())


def read_image(p):
    color = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"Unreadable image {p}")
    return color


def pack_masks(masks, hw):
    h, w = hw
    return (np.stack([np.packbits(mask, axis=-1) for mask in masks]) if masks
            else np.empty((0, h, (w + 7) // 8), dtype=np.uint8))


def as_key(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_raw_cache(path, checkpoint, frames, key):
    model, meta = load_checkpoint(checkpoint)
    rows, packed = [], {}
    for start in range(0, len(frames), key["batch"]):
        batch = frames[start:start + key["batch"]]
        color = [read_image(p) for _, p in batch]
        preds = model.predict(source=color, imgsz=meta["imgsz"], conf=CONF, iou=DETECTOR_IOU, retina_masks=True,
                              max_det=metric.MAX_PER_IMAGE, device=key["device"], batch=len(color), verbose=False)
        for (i, _), im, p in zip(batch, color, preds, strict=True):
            hw = im.shape[:2]
            raw = unpack_yolo(p, hw)
            rows.append({"image_id": i, "shape": hw,
                         "detections": [{"box": d.box, "score": d.score, "form": d.form} for d in raw]})
            packed[i] = pack_masks([d.mask for d in raw], hw)
    meta.update(boxes=sum(len(r["detections"]) for r in rows), empty=sum(not r["detections"] for r in rows))
    with TemporaryDirectory(prefix=".raw_tmp_", dir=path.parent) as tmp:
        temp = Path(tmp) / path.name
        with temp.open("xb") as f:
            header = json.dumps({"key": key, "model": meta, "frames": rows}, allow_nan=False)
            np.savez_compressed(f, raw_header=np.asarray(header), **packed)
        try:
            path.hardlink_to(temp)
        except FileExistsError:  # Another inference process has just published the same cache.
            pass


def read_raw_cache(stack, cache_dir, checkpoint, frames, key, require_cache):
    path = cache_dir / f"{as_key(key)}.npz"
    if not path.is_file():
        if require_cache:
            raise FileNotFoundError(f"No cache for {checkpoint}: {path}")
        print(f"Inference: {checkpoint}", flush=True)
        write_raw_cache(path, checkpoint, frames, key)
    else:
        print(f"Cache hit: {checkpoint}", flush=True)
    archive = stack.enter_context(np.load(path, allow_pickle=False))
    header = json.loads(archive["raw_header"].item())
    rows = header["frames"]
    ids = [i for i, _ in frames]
    if (header["key"] != key or [r["image_id"] for r in rows] != ids
            or set(archive.files) != set(ids) | {"raw_header"}):
        raise ValueError(f"Invalid raw cache: {path}")
    return archive, {r["image_id"]: r for r in rows}, header["model"], path.name


def cached_detections(archive, row):
    hw = h, w = tuple(row["shape"])
    raw = row["detections"]
    packed = archive[row["image_id"]]
    if packed.dtype != np.uint8 or packed.shape != (len(raw), h, (w + 7) // 8):
        raise ValueError(f"Invalid cached masks for {row['image_id']}")
    masks = np.unpackbits(packed, axis=-1, count=w).astype(bool)
    return hw, [Detection(tuple(d["box"]), d["score"], d["form"], m) for d, m in zip(raw, masks, strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "test_wbf_v1")
    parser.add_argument("--iou", type=float, default=.55)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--config", type=Path, help="Per-model paths, weights and primary/nano roles (JSON list)")
    parser.add_argument("--cache-dir", type=Path, default=WORK / "detector_raw_cache_v1")
    parser.add_argument("--require-cache", action="store_true", help="Do not run detector inference on a cache miss")
    args = parser.parse_args()
    if not (0 < args.iou < 1) or args.batch < 1:
        parser.error("Require 0 < --iou < 1 and --batch >= 1")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    ids = test_ids()
    specs = model_specs(args.config)
    weights = [s.weight for s in specs]
    nano = {i for i, s in enumerate(specs) if s.group == "nano"}
    shas = [digest(s.path) for s in specs]
    if len(set(shas)) != len(specs):
        raise ValueError("Two checkpoints have the same contents; duplicate votes are not allowed")
    for s, sha in zip(specs, shas, strict=True):
        if s.path.name == "grisha_yolo.pt" and sha != GRISHA_SHA256:
            raise ValueError("Grisha's checkpoint is not the inspected Git blob")
    frames = [(i, test_image_path(i)) for i in ids]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    common_key = {
        "version": CACHE_VERSION, "rgb_sha256": as_key([(i, digest(p)) for i, p in frames]),
        "conf": CONF, "detector_iou": DETECTOR_IOU, "retina_masks": True, "max_det": metric.MAX_PER_IMAGE,
        "batch": args.batch, "device": args.device, "ultralytics": version("ultralytics"), "torch": str(torch.__version__),
    }
    packed, result, votes, forms = {}, [], Counter(), Counter()
    metadata = []
    grisha_only = 0
    grisha = {i for i, s in enumerate(specs) if s.path.name == "grisha_yolo.pt"}
    with ExitStack() as stack:
        raw = []
        for i, (s, sha) in enumerate(zip(specs, shas, strict=True)):
            archive, rows, meta, cached = read_raw_cache(stack, args.cache_dir, s.path, frames,
                                                         dict(common_key, checkpoint_sha256=sha), args.require_cache)
            raw.append((archive, rows))
            path = s.path.relative_to(WORK.parent) if s.path.is_relative_to(WORK.parent) else s.path
            metadata.append(dict(meta, id=i, path=str(path), sha256=sha, cache=cached, weight=s.weight, group=s.group))
        for i in ids:
            shapes, candidates = zip(*(cached_detections(a, rows[i]) for a, rows in raw), strict=True)
            hw = h, w = shapes[0]
            if not all(shape == hw for shape in shapes):
                raise ValueError(f"Cached model results disagree on image size: {i}")
            pairs = fuse(candidates, hw, args.iou, weights=weights, nano=nano)
            ds = [p for p, _ in pairs]
            result.append({"image_id": i, "source": "real", "shape": [h, w], "detections": ds})
            packed[i] = pack_masks([mask for _, mask in pairs], hw)
            votes.update(len(p["votes"]) for p in ds)
            forms.update(p["form"] for p in ds)
            grisha_only += sum(len(p["votes"]) == 1 and p["votes"][0] in grisha for p in ds)
            if (n := len(result)) % 32 == 0 or n == len(ids):
                print(f"Fused test frames: {n}/{len(ids)}", flush=True)

    expected_empty = sum(not r["detections"] for r in result)
    report = {
        "frames": len(result), "boxes": sum(votes.values()), "empty_frames": expected_empty,
        "boxes_by_form": dict(forms), "boxes_by_vote_count": dict(sorted(votes.items())),
        "grisha_only_boxes": grisha_only, "model_inputs": metadata,
        "conf_input": CONF, "iou": args.iou,
        "confidence_fusion": "sum(weight * score) over members / sum(weight) over all models",
        "nano_policy": "may join a same-class primary cluster; cannot seed one or supply its mask",
        "mask_source": "original aligned mask from the highest-confidence compatible primary member",
        "max_det": metric.MAX_PER_IMAGE, "device": args.device,
    }
    if [r["image_id"] for r in result] != ids or set(packed) != set(ids):
        raise ValueError("Test image coverage is inconsistent")
    with TemporaryDirectory(prefix="wbf_tmp_", dir=args.out.parent) as tmp:
        tmp = Path(tmp)
        with (tmp / "base_masks.npz").open("xb") as f:
            np.savez_compressed(f, **packed)
        with (tmp / "base_detections.jsonl").open("x", encoding="utf-8") as f:
            for row in result:
                f.write(json.dumps(row, allow_nan=False) + "\n")
        with (tmp / "report.json").open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        args.out.mkdir(exist_ok=False)
        for name in ("base_detections.jsonl", "base_masks.npz", "report.json"):
            (args.out / name).hardlink_to(tmp / name)
    print(f"Saved {args.out}: {report['boxes']} fused boxes, {expected_empty} predicted empty frames.")


if __name__ == "__main__":
    main()
