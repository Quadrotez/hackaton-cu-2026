"""RGB-D inference and local submission generation, without network access."""

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import pandas as pd

from material_features import bounds, region_features
from material_split import DATA, WORK
from submission_tools import HEADER, baseline_holdout, ground_truth, metric, parse_prediction, test_ids, validate_csv
from take_train import estimate_mass, predict_take
from take_v2 import take_input, upstream_predictions
from v2_data import UNKNOWN
from volume_calibration import raw_volume_ml
from volume_v2 import predict_volume_v2


FORMS = {"bulk", "solid"}
MAX_DETECTIONS = 30
DEFAULT_BUNDLE = WORK / "submission_models" / "compact.full.pkl"


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    form: str
    mask: np.ndarray


def as_numpy(tensor):
    return np.asarray(tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor)


def unpack_yolo(result, expected_shape):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None:
        raise ValueError("Expected YOLO segmentation masks, not a box-only model")
    boxes = as_numpy(result.boxes.xyxy)
    scores = as_numpy(result.boxes.conf)
    classes = as_numpy(result.boxes.cls).astype(int)
    masks = as_numpy(result.masks.data)
    if masks.shape != (len(boxes), *expected_shape):
        raise ValueError("YOLO masks must be aligned to the input image (retina_masks=True)")
    out = []
    for box, score, class_id, mask in zip(boxes, scores, classes, masks, strict=True):
        form = str(result.names[int(class_id)]).strip().lower()
        if form not in FORMS:
            raise ValueError(f"Unexpected YOLO class {form!r}; expected bulk or solid")
        out.append(Detection(tuple(map(float, box)), float(score), form, mask > 0.5))
    return out


class PredictPipeline:
    def __init__(
        self, model_dir=WORK, take_filename=None, weights=None,
        camera_path=DATA / "camera.json", detector=None, bundle_path=None, form_refiner=None,
    ):
        model_dir = Path(model_dir)
        self.weights = Path(weights) if weights is not None else WORK / "weights" / "best.pt"
        self.camera = json.loads(Path(camera_path).read_text(encoding="utf-8"))
        self.detector = detector
        self.dino = None
        self.crop_for_dino = None
        self.form_refiner = form_refiner
        if take_filename is None:
            with Path(bundle_path or DEFAULT_BUNDLE).open("rb") as f:
                bundle = pickle.load(f)
            self.take, self.volume = bundle["take"], bundle["volume"]
            if bundle["material_kind"] in ("compact", "dino"):
                from astra_deploy import Material, masked_crop

                self.dino = Material(model_dir, compact=bundle["material_kind"] == "compact")
                self.crop_for_dino = masked_crop
                if "head" in bundle:  # Object-disjoint head for validation only.
                    self.dino.head = bundle["head"]
            elif bundle["material_kind"] == "cb":
                self.material = bundle["material"]
            else:
                raise ValueError(f"Unknown material model {bundle['material_kind']!r}")
            return

        with (model_dir / take_filename).open("rb") as f:
            self.take = pickle.load(f)
        with (model_dir / self.take["material_file"]).open("rb") as f:
            self.material = pickle.load(f)
        with (model_dir / self.take["volume_file"]).open("rb") as f:
            self.volume = pickle.load(f)
        if self.take["split_version"] != self.volume["split_version"]:
            raise ValueError("Volume and take bundles must use the same split version")

    def load_detector(self):
        if self.detector is not None:
            return self.detector
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO segmentation weights are not available: {self.weights}")
        os.environ.setdefault("YOLO_OFFLINE", "true")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics in the detector environment to use YOLO") from exc
        self.detector = YOLO(str(self.weights))
        return self.detector

    def read_frame(self, directory):
        path = Path(directory)
        color_path = path / "color.png"
        if not color_path.is_file():
            color_path = path / "color.jpg"
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(path / "depth.png"), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None or color.shape[:2] != depth.shape or depth.dtype != np.uint16:
            raise ValueError(f"Missing or invalid aligned color/depth images in {path}")
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if np.asarray(meta.get("floor_plane")).shape != (4,):
            raise ValueError(f"Missing or invalid floor_plane in {path / 'meta.json'}")
        return color, depth, meta

    def predict_from_detections(self, color, depth, meta, detections):
        height, width = color.shape[:2]
        rows, accepted, crops = [], [], []
        for det in sorted(detections, key=lambda d: -d.score)[:MAX_DETECTIONS]:
            if det.form not in FORMS or not 0 <= det.score <= 1 or not np.isfinite(det.box).all():
                raise ValueError(f"Invalid detection: form={det.form!r}, score={det.score}, box={det.box}")
            x1, y1, x2, y2 = det.box
            x1, x2 = np.clip([x1, x2], 0, width).tolist()
            y1, y2 = np.clip([y1, y2], 0, height).tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            full_mask = np.asarray(det.mask, dtype=bool)
            if full_mask.shape != (height, width):
                raise ValueError("Segmentation mask shape does not match color/depth")
            bbox = (x1, y1, x2 - x1, y2 - y1)
            left, top, right, bottom = bounds(bbox, width, height, .1)
            local_mask = full_mask[top:bottom, left:right]
            if not local_mask.any():
                continue
            if self.crop_for_dino is not None:
                crops.append(self.crop_for_dino(color[top:bottom, left:right], local_mask))
            rows.append({
                **region_features(color, depth, bbox, local_mask, meta, self.camera, left, top),
                "raw_volume_ml": raw_volume_ml(depth, full_mask, meta["floor_plane"], self.camera),
                "form": det.form, "surface": meta.get("surface", UNKNOWN),
                "lighting": meta.get("lighting", UNKNOWN),
            })
            accepted.append(((x1, y1, x2, y2), det))
        if not rows:
            return "none"
        rows = pd.DataFrame(rows)
        if self.form_refiner is not None:
            forms = np.asarray(self.form_refiner(rows.copy(), [det.score for _, det in accepted]))
            if forms.shape != (len(rows),) or not np.isin(forms, list(FORMS)).all():
                raise ValueError("Form refiner must return one bulk/solid label per detection")
            rows["form"] = forms
        if self.dino is None:
            pred = upstream_predictions(rows, self.material, self.volume, self.take["density"])
        else:
            material = self.dino.predict(crops)
            volume = predict_volume_v2(self.volume, rows, material)
            pred = pd.DataFrame({
                "material_pred": material,
                "volume_pred_ml": volume,
                "mass_pred_g": estimate_mass(rows["form"], material, volume, self.take["density"]),
            })
        x = take_input(rows, self.take["steps3"], pred)
        pred["take"] = predict_take(self.take, x)
        return ";".join(
            f"{' '.join(f'{x:.8g}' for x in box)} {det.score:.8g} {form} {p.material_pred} "
            f"{p.volume_pred_ml:.8g} {p.mass_pred_g:.8g} {p.take}"
            for (box, det), form, p in zip(accepted, rows["form"], pred.itertuples(index=False), strict=True)
        )

    def detect_frame(self, directory, *, conf=.25, imgsz=640, device=None):
        detector = self.load_detector()
        color, depth, meta = self.read_frame(directory)
        kwargs = {"device": device} if device is not None else {}
        results = detector.predict(
            source=color, conf=conf, imgsz=imgsz, retina_masks=True, max_det=MAX_DETECTIONS,
            verbose=False, **kwargs,
        )
        if len(results) != 1:
            raise ValueError(f"Expected exactly one YOLO result for {directory}")
        return color, depth, meta, unpack_yolo(results[0], color.shape[:2])

    def predict_frame(self, directory, *, conf=.25, imgsz=640, device=None):
        return self.predict_from_detections(*self.detect_frame(directory, conf=conf, imgsz=imgsz, device=device))


def write_submission(pipeline, output, options):
    images = [(image_id, DATA / "real" / "test" / image_id) for image_id in test_ids()]
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    f = NamedTemporaryFile(dir=output.parent, mode="w", newline="", encoding="utf-8", delete=False)
    temporary = Path(f.name)
    try:
        with f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for index, (image_id, path) in enumerate(images, 1):
                writer.writerow([image_id, pipeline.predict_frame(path, **options)])
                if index % 25 == 0:
                    print(f"Test frames: {index}/{len(images)}", flush=True)
        validate_csv(temporary)
        output.hardlink_to(temporary)  # Publish the complete file without overwriting another run.
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("frame", type=Path, nargs="?")
    mode.add_argument("--out", type=Path, help="Write a new validated test submission")
    mode.add_argument("--validate", action="store_true", help="Score on the baseline's held-out real frames")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--bundle", type=Path, help="New fit_submission_models.py bundle")
    models.add_argument("--take-model", help="Legacy take bundle filename in work/")
    parser.add_argument("--weights", type=Path, default=WORK / "weights" / "best.pt")
    parser.add_argument("--device")
    parser.add_argument("--conf", type=float, default=.10)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    bundle_path = args.bundle
    if args.validate and bundle_path is None and args.take_model is None:
        bundle_path = DEFAULT_BUNDLE.with_name("compact.holdout.pkl")
    cv2.setNumThreads(1)
    pipeline = PredictPipeline(take_filename=args.take_model, weights=args.weights, bundle_path=bundle_path)
    options = {"conf": args.conf, "imgsz": args.imgsz, "device": args.device}
    if args.out:
        write_submission(pipeline, args.out, options)
    elif args.validate:
        holdout = baseline_holdout()
        pred = {image_id: parse_prediction(image_id, pipeline.predict_frame(path, **options))
                for image_id, path, _ in holdout}
        r = metric.components(ground_truth(holdout), pred)
        print(f"VALIDATION ({len(holdout)} frames) Score {r['Score']:.2f} | D {r['D']:.3f} "
              f"M {r['M']:.3f} V {r['V']:.3f} S {r['S']:.3f}", flush=True)
    else:
        print(pipeline.predict_frame(args.frame, **options))


if __name__ == "__main__":
    main()
