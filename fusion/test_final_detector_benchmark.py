from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from fusion import final_detector_benchmark as bench


class FinalBenchmarkTests(unittest.TestCase):
    def test_crop_uses_predicted_box_and_padding(self):
        color = np.zeros((128, 128, 3), dtype=np.uint8)
        color[30:80, 30:80] = (100, 110, 120)
        crop = bench.crop_rgb(color, (30, 30, 80, 80))
        self.assertEqual(crop.shape, (128, 128, 3))
        self.assertEqual(crop[0, 0].tolist(), [0, 0, 0])
        self.assertEqual(crop[64, 64].tolist(), [100, 110, 120])

    def test_target_audit_separates_wrong_form_and_missing_hypothesis(self):
        g = {"i": [{"box": (0, 0, 10, 10)}]}
        p = {"i": [{"box": (0, 0, 10, 10), "score": .8, "form": "solid"},
                   {"box": (0, 0, 10, 10), "score": .4, "form": "bulk"}]}
        audit = bench.target_audit(g, {"i": ["b01"]}, p)["b01"]
        self.assertEqual((audit["solid"], audit["missed"], audit["any_bulk_iou50"]), (1, 0, 1))

    def test_serialization_strict_roundtrip(self):
        ids = [f"im{i:03}" for i in range(137)]
        p = {i: [] for i in ids}
        p[ids[0]] = [dict(box=[1, 2, 3, 4], score=.5, form="bulk", material="wood",
                         volume=20., mass=15., take=1)]
        with TemporaryDirectory() as folder, patch("submission_tools.test_ids", return_value=ids), \
                patch.object(bench.joint, "test_ids", return_value=ids):
            result = bench.write_submission(Path(folder) / "submission.csv", p)
            self.assertEqual((result["frames"], result["detections"], result["none"]), (137, 1, 136))


if __name__ == "__main__":
    unittest.main()
