"""Train fixed-epoch segmentation detectors on the shared outer Astramis folds."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from build_detector_folds import DEFAULT_OUT, digest
from material_split import DATA, WORK, load_frames


FORMS = ("bulk", "solid")
ROOT = WORK.parent


def read_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["version"] != 1 or digest(ROOT / manifest["source_object_folds"]) != manifest["source_object_folds_sha256"]:
        raise ValueError("The source object split no longer matches this manifest")
    for name, sha in manifest["labels_sha256"].items():
        if digest(DATA / name) != sha:
            raise ValueError(f"The annotations changed after manifest creation: {name}")
    return manifest


def frame_map():
    return {i: (source, DATA / im["file_name"], im, anns) for source, i, im, anns in load_frames()}


def copy_yolo_split(root, name, ids, frames):
    images = root / "images" / name
    labels = root / "labels" / name
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in ids:
        _, source, im, anns = frames[i]
        if not source.is_file():
            source = next(p for p in (source.parent / "color.png", source.parent / "color.jpg") if p.is_file())
        # Copy instead of hard-link: image validation must not be able to change the inputs.
        with source.open("rb") as src, (images / (i + source.suffix)).open("xb") as dst:
            shutil.copyfileobj(src, dst)
        lines = []
        for a in anns:
            polygon = max(a["segmentation"], key=len)
            xy = np.asarray(polygon, dtype=float).reshape(-1, 2) / [im["width"], im["height"]]
            if len(xy) < 3:
                raise ValueError(f"Invalid polygon: {i} ann_id={a['id']}")
            form = FORMS.index(a["attributes"]["form"])
            lines.append(f"{form} " + " ".join(f"{v:.5f}" for v in xy.clip(0, 1).ravel()))
        with (labels / (i + ".txt")).open("x", encoding="utf-8") as f:
            f.write("\n".join(lines))


def prepare(root, spec, frames):
    train = spec["train_image_ids"]
    val = set(spec["val_real_image_ids"]) | set(spec["val_synth_image_ids"])
    if set(train) & (val | set(spec["excluded_mixed_synth_image_ids"])):
        raise ValueError(f"Image leakage in fold {spec['fold']}")
    seen = {a["attributes"]["object_id"] for i in train for a in frames[i][3]}
    if seen != set(spec["train_object_ids"]) or seen & set(spec["val_object_ids"]):
        raise ValueError(f"Unexpected training objects in fold {spec['fold']}")

    # Ultralytics needs a val path for its final bookkeeping. This monitor is a
    # subset of train, never the outer holdout, and its score is not reported.
    # All folds use the last fixed epoch, not a checkpoint chosen by the monitor.
    bins = defaultdict(list)
    for i in train:
        src, _, _, anns = frames[i]
        if src == "real":
            bins[anns[0]["attributes"]["form"] if anns else "empty"].append(i)
    monitor = [i for g in ("bulk", "solid", "empty") for i in bins[g][::max(1, len(bins[g]) // 6)][:6]]
    if not all(bins[g] for g in FORMS):
        raise ValueError("Missing monitor form class")
    copy_yolo_split(root, "train", train, frames)
    copy_yolo_split(root, "monitor_from_train", monitor, frames)
    data = root / "data.yaml"
    with data.open("x", encoding="utf-8") as f:
        f.write(f"path: {json.dumps(str(root.resolve()))}\ntrain: images/train\nval: images/monitor_from_train\n"
                + "names:\n" + "".join(f"  {i}: {g}\n" for i, g in enumerate(FORMS)))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out", type=Path, default=WORK / "detector_oof_v1")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 4, 2, 3, 5])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--init", type=Path, default=ROOT / "yolo11n-seg.pt")
    ap.add_argument("--evaluate", action="store_true", help="Export fixed OOF boxes/masks and run form/ranking ablations")
    args = ap.parse_args()
    if not args.init.is_file() or args.epochs <= 0 or args.batch <= 0:
        ap.error("Expected local pretrained weights, positive epochs and positive batch size")
    if digest(args.init) != "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152":
        ap.error("Use the official pretrained YOLO11n-seg weights, not a case-fitted detector")
    manifest = read_manifest(args.manifest)
    configs = {f["fold"]: f for f in manifest["folds"]}
    if len(args.folds) != len(set(args.folds)) or not set(args.folds) <= configs.keys():
        ap.error("Expected distinct fold IDs from the manifest")
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "run.json").open("x", encoding="utf-8") as f:
        json.dump({
            "manifest": str(args.manifest.resolve()), "manifest_sha256": digest(args.manifest),
            "seed": 0, "imgsz": 640, "epochs": args.epochs, "batch": args.batch, "device": args.device,
            "checkpoint": "last.pt", "init": str(args.init.resolve()), "init_sha256": digest(args.init),
            "fold_order": args.folds,
        }, f, indent=2)
        f.write("\n")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    from ultralytics import YOLO

    frames = frame_map()
    for fold in args.folds:
        fold_root = args.out / f"fold_{fold:02d}"
        data = prepare(fold_root / "data", configs[fold], frames)
        print(f"TRAIN fold {fold} ({args.device}, {args.epochs} fixed epochs)", flush=True)
        model = YOLO(str(args.init.resolve()))
        model.train(data=str(data), epochs=args.epochs, imgsz=640, batch=args.batch, seed=0,
                    device=args.device, workers=0, val=False, deterministic=True, plots=False, verbose=False,
                    project=str(fold_root.resolve()), name="train", exist_ok=False)
        checkpoint = fold_root / "train" / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
        print(f"READY fold {fold}: {checkpoint}", flush=True)
        if args.evaluate:
            from detector_fold_eval import evaluate_fold
            evaluate_fold(fold_root, checkpoint, configs[fold], frames, device="cpu")
    if args.evaluate:
        from detector_fold_eval import summarize_pooled
        summarize_pooled(args.out, [configs[f] for f in sorted(args.folds)], frames)


if __name__ == "__main__":
    main()
