"""Write one detector's full CSV using the frozen heads from the 78.41 run."""

import os
import pickle
import sys
from contextlib import ExitStack
from importlib.metadata import version


class Joint04Exporter:
    def __init__(self, root, heads, device, batch):
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        sys.path.insert(0, str(root / "work"))
        import astra_joint04 as joint
        import infer_detector_wbf as raw

        self.joint, self.raw = joint, raw
        self.validate = joint.validate_csv
        joint.cv2.setNumThreads(1)
        joint.torch.set_num_threads(4)
        self.frames = [(i, raw.test_image_path(i)) for i in joint.test_ids()] if heads is not None else None
        self.cache = root / "work" / "detector_raw_cache_v1"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.key = {
            "version": raw.CACHE_VERSION,
            "conf": raw.CONF, "detector_iou": raw.DETECTOR_IOU, "retina_masks": True,
            "max_det": raw.metric.MAX_PER_IMAGE, "batch": batch, "device": device,
            "ultralytics": version("ultralytics"), "torch": str(joint.torch.__version__),
        }
        self.heads = None
        if heads is not None:
            with heads.open("rb") as f:
                self.heads = pickle.load(f)  # Trusted local artifact of the 78.41 pipeline.
        self.encoders = None

    def frame_key(self, frames):
        return self.raw.as_key([(i, self.raw.digest(p)) for i, p in frames])

    def predictions(self, model, sha, frames, heads, rgb_sha256, *,
                    require_cache=False, with_mask_indices=False):
        joint, raw = self.joint, self.raw
        dets = joint.Detections()
        key = dict(self.key, checkpoint_sha256=sha, rgb_sha256=rgb_sha256)
        indices = {}
        with ExitStack() as stack:
            archive, rows, meta, _ = raw.read_raw_cache(stack, self.cache, model.checkpoint, frames, key, require_cache)
            if meta["imgsz"] != model.imgsz:
                raise ValueError(f"{model.tag}: checkpoint imgsz={meta['imgsz']}, expected {model.imgsz}")
            for fid, path in frames:
                (h, w), decoded = raw.cached_detections(archive, rows[fid])
                pairs = []
                indices[fid] = []
                for idx, d in enumerate(decoded):
                    x1, y1, x2, y2 = joint.np.clip(d.box, (0, 0, 0, 0), (w, h, w, h)).tolist()
                    box = (x1, y1, x2, y2)
                    if x1 < x2 and y1 < y2 and d.score > 0 and raw.usable_mask(d.mask, box):
                        pairs.append((dict(box=box, score=d.score, form=d.form), d.mask))
                        indices[fid].append(idx)
                dets.frame(model.tag, fid, path.parent, pairs)
        if not dets.rows:
            return {fid: [] for fid, _ in frames}
        with joint.threadpool_limits(4):
            if self.encoders is None:
                self.encoders = joint.encoders()
            features = [*joint.encode(dets.crops, self.encoders),
                        *joint.encode([joint.cv2.flip(crop, 1) for crop in dets.crops], self.encoders)]
            predictions = dets.output(model.tag, joint.apply(heads, joint.pd.DataFrame(dets.rows), features))
        if with_mask_indices:
            for fid, objects in predictions.items():
                for d, idx in zip(objects, indices[fid], strict=True):
                    d["mask_index"] = idx
        return predictions

    def rows(self, model, sha, rgb_sha256):
        pred = self.predictions(model, sha, self.frames, self.heads, rgb_sha256)
        for fid, _ in self.frames:
            yield fid, ";".join(" ".join(map(str, [*obj["box"], obj["score"], tail(obj)]))
                                for obj in pred[fid]) or "none"


def tail(obj):
    return " ".join(map(str, [obj[k] for k in ("form", "material", "volume", "mass", "take")]))
