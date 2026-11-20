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
DEFAULT_MODEL = ROOT / "yolo11n-seg.pt"


def image_size(value):
    """Keep Ultralytics from silently rounding the configured resolution."""
    size = int(value)
    if size <= 0 or size % 32:
        raise argparse.ArgumentTypeError("imgsz must be a positive multiple of 32")
    return size


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
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 4, 2, 3, 5],
                    help="Fold IDs in execution order, e.g. --folds 1")
    ap.add_argument("--epochs", type=int, default=10, help="Fixed number of epochs per fold")
    ap.add_argument("--batch", type=int, default=8, help="Training batch size")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--model", "--init", dest="model", type=Path, default=DEFAULT_MODEL,
                    help="Local pretrained segmentation .pt, e.g. ./yolo11m-seg.pt; do not use case-fitted weights")
    ap.add_argument("--imgsz", type=image_size, default=640, help="Input size for both training and outer evaluation")
    ap.add_argument("--evaluate", action="store_true", help="Export fixed OOF boxes/masks and run form/ranking ablations")
    ap.add_argument("--eval-device", help="Override the evaluation device; defaults to --device")
    args = ap.parse_args()
    pretrained = args.model.resolve()
    if not pretrained.is_file() or pretrained.suffix != ".pt":
        ap.error(f"Pass a local pretrained segmentation checkpoint via --model: {pretrained}")
    if args.epochs <= 0 or args.batch <= 0:
        ap.error("--epochs and --batch must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = read_manifest(args.manifest)
    configs = {f["fold"]: f for f in manifest["folds"]}
    if len(args.folds) != len(set(args.folds)) or not set(args.folds) <= configs.keys():
        ap.error("Expected distinct fold IDs from the manifest")
    from ultralytics import YOLO

    def fresh_model():
        model = YOLO(str(pretrained))
        if model.task != "segment":
            ap.error("--model must be a segmentation checkpoint")
        # Do not let one of our own Astramis checkpoints leak across outer folds.
        if {str(name).strip().lower() for name in model.names.values()} & set(FORMS):
            ap.error("--model already has Astramis classes; initialize from public pretrained weights")
        return model

    model = fresh_model()
    eval_device = args.eval_device or args.device
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "run.json").open("x", encoding="utf-8") as f:
        json.dump({
            "manifest": str(args.manifest.resolve()), "manifest_sha256": digest(args.manifest),
            "seed": 0, "imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch, "device": args.device,
            "eval_device": eval_device if args.evaluate else None,
            "checkpoint": "last.pt", "init": str(pretrained), "init_sha256": digest(pretrained),
            "fold_order": args.folds,
        }, f, indent=2)
        f.write("\n")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    frames = frame_map()
    for index, fold in enumerate(args.folds):
        if index:
            model = fresh_model()
        fold_root = args.out / f"fold_{fold:02d}"
        data = prepare(fold_root / "data", configs[fold], frames)
        print(f"TRAIN fold {fold} ({pretrained.name}, {args.device}, imgsz={args.imgsz}, "
              f"batch={args.batch}, {args.epochs} fixed epochs)", flush=True)
        model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=0,
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
            evaluate_fold(fold_root, checkpoint, configs[fold], frames, device=eval_device, imgsz=args.imgsz)
    if args.evaluate:
        from detector_fold_eval import summarize_pooled
        summarize_pooled(args.out, [configs[f] for f in sorted(args.folds)], frames)


if __name__ == "__main__":
    main()
