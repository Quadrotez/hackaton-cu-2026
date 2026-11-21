import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from submission_fusion import load
from yolo960_ab import (CONFIG, ROOT, Model, checkpoint_hashes, file_sha, fuse_frame,
                        fused_rows, read_config, write_csv)


def det(box, score, form="solid", attrs="plastic 500 25 1"):
    return {"box": box, "score": score, "form": form, "tail": f"{form} {attrs}"}


class FusionTest(unittest.TestCase):
    def test_base_parity(self):
        inputs = [ROOT / "work" / f"submission_{n}_final.csv" for n in ("wbf6", "grisha_single")]
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old.csv", Path(tmp) / "new.csv"
            subprocess.run(
                [sys.executable, str(CONFIG.with_name("submission_fusion.py")), *map(str, inputs), str(old)],
                check=True, capture_output=True,
            )
            write_csv(new, fused_rows([load(p) for p in inputs], [1., 1.]))
            self.assertEqual(old.read_bytes(), new.read_bytes())

    def test_nano_only_changes_weighted_geometry(self):
        per_model = [
            [det([10, 10, 20, 20], .5)],
            [det([11, 10, 21, 20], .8, attrs="glass 100 40 0")],
            [det([12, 10, 22, 20], 1, attrs="wood 10 10 0"),
             det([50, 50, 60, 60], 1), det([10, 10, 20, 20], 1, form="bulk")],
        ]
        a = fuse_frame(per_model[:2], [1., 1.], ())
        b = fuse_frame(per_model, [1., 1., .3], {2})
        self.assertEqual(a, "10.6 10.0 20.6 20.0 0.800000 solid plastic 500 25 1")
        self.assertEqual(b, "10.9 10.0 20.9 20.0 0.800000 solid plastic 500 25 1")

    def test_fallback_uses_primary_detector_not_nano(self):
        per_model = [[], [det([10, 10, 20, 20], .2, attrs="glass 100 40 0")],
                     [det([10, 10, 20, 20], .9)]]
        self.assertTrue(fuse_frame(per_model, [1., 1., .3], {2}).endswith("0.200000 solid glass 100 40 0"))

    def test_oof_without_s640_uses_highest_confidence_attributes(self):
        sources = [[det([10, 10, 20, 20], .5)],
                   [det([10, 10, 20, 20], .8, attrs="glass 100 40 0")]]
        self.assertTrue(fuse_frame(sources, [1., 1.], (), primary_index=None).endswith("solid glass 100 40 0"))
        self.assertTrue(fuse_frame(sources, [1., 1.], ()).endswith("solid plastic 500 25 1"))

    def test_checkpoint_check_rejects_duplicate_contents(self):
        with tempfile.TemporaryDirectory() as d:
            primary, fold = Path(d) / "s.pt", Path(d) / "fold.pt"
            for p, value in ((primary, "s"), (fold, "m")):
                with ZipFile(p, "w") as archive:
                    archive.writestr("weights/data.pkl", value)
            model = Model("m", 1, fold, 960, 1.)
            expected = file_sha(primary)
            self.assertEqual(set(checkpoint_hashes([model], primary, expected, check_archives=True)),
                             {"yolo11s_primary", model.tag})
            fold.write_bytes(primary.read_bytes())
            with self.assertRaisesRegex(ValueError, "identical contents"):
                checkpoint_hashes([model], primary, expected, check_archives=True)

    def test_frame_cap_and_coverage(self):
        boxes = [det([20 * i, 0, 20 * i + 10, 10], 1 - i * .01) for i in range(32)]
        self.assertEqual(len(fuse_frame([boxes[:16], boxes[16:]], [1., 1.], ()).split(";")), 30)
        with self.assertRaisesRegex(ValueError, "frame coverage"):
            list(fused_rows([{"one": []}, {"two": []}], [1., 1.]))

    def test_config(self):
        _, _, a = read_config(CONFIG, False)
        _, _, b = read_config(CONFIG, True)
        self.assertEqual(a, b[:10])
        self.assertEqual([(m.group, m.imgsz, m.weight) for m in a],
                         [("m", 960, 1.)] * 5 + [("l", 960, 1.)] * 5)
        self.assertEqual([(m.group, m.imgsz, m.weight) for m in b[10:]], [("n", 640, .3)] * 5)


if __name__ == "__main__":
    unittest.main()
