import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

def make_tiny_astramis_dataset(root: Path) -> Path:
    frame = root / "real" / "train" / "r_0001"
    (root / "labels").mkdir(parents=True)
    frame.mkdir(parents=True)
    cv2.imwrite(str(frame / "color.png"), np.full((20, 30, 3), (20, 40, 60), dtype=np.uint8))
    cv2.imwrite(str(frame / "depth.png"), np.full((20, 30), 1000, dtype=np.uint16))
    (frame / "meta.json").write_text(json.dumps({"floor_plane": [0, 0, 1, -0.95]}))
    (root / "camera.json").write_text(json.dumps({"fx": 20, "fy": 20, "cx": 15, "cy": 10}))
    labels = {
        "images": [{"id": 1, "file_name": "real/train/r_0001/color.png", "width": 30, "height": 20}],
        "annotations": [{
            "id": 7, "image_id": 1, "bbox": [5, 4, 10, 8], "segmentation": [[5, 4, 15, 4, 15, 12, 5, 12]],
            "attributes": {"object_id": "o1", "form": "solid", "material": "metal", "volume_ml": 600, "mass_g": 700},
        }],
    }
    (root / "labels" / "real_train.json").write_text(json.dumps(labels))
    return root


class BuildInstancesTest(unittest.TestCase):
    def test_preserves_annotation_labels_and_take(self):
        import astramis_improve as ai

        with TemporaryDirectory() as raw:
            rows = ai.build_instances(make_tiny_astramis_dataset(Path(raw)), "real_train.json")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.loc[0, ["ann_id", "object_id", "form", "material", "take"]].to_dict(), {
            "ann_id": 7, "object_id": "o1", "form": "solid", "material": "metal", "take": 0,
        })
        self.assertEqual(rows.loc[0, "mask_area"], 99)
