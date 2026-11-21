import unittest

from class_agnostic_wbf import fuse_frame


def obj(left=0., score=.9, form="solid", material="plastic", width=10.):
    return dict(box=(left, 0., left + width, 10.), score=score, form=form,
                tail=f"{form} {material} 100 40 0")


class AgnosticFusionTest(unittest.TestCase):
    def test_votes_sum_and_attributes_follow_winner(self):
        row = fuse_frame([[obj(score=.93)], [obj(score=.55, form="bulk", material="wood")],
                          [obj(score=.52, form="bulk")]], [1., 1., 1.])
        self.assertEqual(row.split()[4:], "0.930000 bulk wood 100 40 0".split())

    def test_one_vote_per_checkpoint(self):
        rows = fuse_frame([[obj(score=.9), obj(score=.8, form="bulk"), obj(score=.7, form="bulk")],
                           [obj(score=.8, form="bulk")]], [1., 1.])
        self.assertEqual(rows.split()[5], "solid")
        self.assertNotIn(";", rows)

    def test_nms_suppresses_overlap_created_by_box_movement(self):
        detections = [[obj(score=.95), obj(left=3.1, score=.9, form="bulk")],
                      [obj(left=1.6, score=.8, form="bulk")]]
        self.assertEqual(len(fuse_frame(detections, [1., 1.], suppress=False).split(";")), 2)
        self.assertEqual(fuse_frame(detections, [1., 1.]).split()[4:6], ["0.950000", "solid"])

    def test_inclusive_iou_threshold_keeps_distant_objects(self):
        rows = fuse_frame([[obj(width=31.)], [obj(left=9., form="bulk", width=31.), obj(left=100.)]],
                          [1., 1.])
        self.assertEqual(len(rows.split(";")), 2)
        self.assertTrue(rows.startswith("4.5 0.0 35.5 10.0"))

    def test_empty_frame(self):
        self.assertEqual(fuse_frame([[], []], [1., 1.]), "none")


if __name__ == "__main__":
    unittest.main()
