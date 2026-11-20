"""Exercise the full stack using stubbed YOLO segmentation results; no weights required."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from material_features import polygon_mask
from material_split import DATA, WORK, load_frames
from predict_pipeline import Detection, PredictPipeline


class Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.xyxy)


class StubYOLO:
    def __init__(self):
        self.result = SimpleNamespace(boxes=None)

    def predict(self, **kwargs):
        assert kwargs["retina_masks"] and kwargs["source"].ndim == 3
        return [self.result]


def smoke_test(take_file="take_v2.pkl", predictions_file="val_oracle_v2.csv"):
    stub = StubYOLO()
    pipeline = PredictPipeline(take_filename=take_file, detector=stub)
    expected = pd.read_csv(WORK / predictions_file).set_index("image_id")["prediction"]
    predicted = {}
    for source, image_id, image, anns in load_frames():
        if source != "real" or image_id not in expected.index:
            continue
        shapes = [ann["attributes"]["form"] for ann in anns]
        masks = [polygon_mask(ann["segmentation"], image["width"], image["height"], 0, 0) for ann in anns]
        boxes = [(x, y, x + w, y + h) for x, y, w, h in (ann["bbox"] for ann in anns)]
        stub.result = SimpleNamespace(
            boxes=Boxes(xyxy=np.asarray(boxes), conf=np.ones(len(boxes)),
                        cls=np.asarray([s == "solid" for s in shapes], dtype=int)),
            masks=SimpleNamespace(data=np.asarray(masks)), names={0: "bulk", 1: "solid"},
        )
        path = (DATA / image["file_name"]).parent
        actual = pipeline.predict_frame(path)
        reference = expected[image_id]
        actual, reference = [s.split() for s in actual.split(";")], [s.split() for s in reference.split(";")]
        for a, b in zip(actual, reference, strict=True):
            assert len(a) == len(b) == 10
            assert a[5:7] == b[5:7] and a[9] == b[9], image_id
            # Oracle output uses the scorer's one-decimal bbox convention.
            np.testing.assert_allclose(np.array(a[:4], float), np.array(b[:4], float), rtol=0, atol=.050002)
            np.testing.assert_allclose(
                np.asarray([a[4], *a[7:9]], dtype=float), np.asarray([b[4], *b[7:9]], dtype=float),
                rtol=1e-6, atol=1e-6,
            )
        predicted[image_id] = True
    assert len(predicted) == sum(expected.index.str.startswith("r_")) == 49

    stub.result = SimpleNamespace(boxes=None)
    assert pipeline.predict_frame(path) == "none"
    color, depth, meta = pipeline.read_frame(path)
    detection = Detection(boxes[0], 1.0, shapes[0], masks[0])
    no_depth = pipeline.predict_from_detections(color, np.zeros_like(depth), meta, [detection])
    assert len(no_depth.split()) == 10
    assert np.isfinite(np.asarray(no_depth.split()[7:9], dtype=float)).all()
    absent = PredictPipeline(take_filename=take_file, weights=WORK / "__absent_test_weights__.pt")
    try:
        absent.load_detector()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing weights were not detected")
    print(f"OK: {len(predicted)} real oracle frames, empty detections, no depth, and missing weights")


if __name__ == "__main__":
    import sys

    smoke_test(*sys.argv[1:])
