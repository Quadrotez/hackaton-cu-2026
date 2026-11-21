import json
import math
import unittest

from attribute_postprocess import AttributePolicy, BulkCalibration, DensityPrior, TakeRule
from yolo960_ab import fuse_frame


class AttributeTest(unittest.TestCase):
    def setUp(self):
        density = DensityPrior.fit([
            ("light", "bulk", "paper", 500., 10.),
            ("heavy", "solid", "metal", 500., 2500.),
        ])
        bulk = BulkCalibration((math.log(50.), math.log(500.)), (math.log(20.), math.log(750.)))
        self.policy = AttributePolicy(density, TakeRule(.8), TakeRule(.8, 300.),
                                      TakeRule(.6, 200.), bulk, "density-small")

    def test_one_vote_per_object_and_volume_monotonicity(self):
        light = ("light", "bulk", "paper", 500., 10.)
        heavy = ("heavy", "solid", "metal", 500., 2500.)
        self.assertEqual(DensityPrior.fit([light] * 100 + [heavy]), self.policy.prior)
        light_p = self.policy.prior.light_probability
        self.assertGreater(light_p("bulk", "paper", 500), light_p("solid", "metal", 500))
        self.assertGreater(light_p("bulk", "paper", 500), light_p("bulk", "paper", 900))
        self.assertTrue(0 < light_p("unseen", "unseen", 500) < 1)

    def test_bulk_calibrator_is_monotone_and_bounded(self):
        vols = [1., 20., 49., 50., 51., 200., 499., 500., 501., 2000.]
        actual = list(map(self.policy.bulk.apply, vols))
        self.assertEqual(actual, sorted(actual))
        self.assertTrue(all(.5 <= a / v <= 2 for a, v in zip(actual, vols, strict=True)))
        self.assertLess(self.policy.bulk.apply(50.), 50.)
        self.assertGreater(self.policy.bulk.apply(500.), 500.)
        self.assertEqual(self.policy.apply("bulk", "solid", "metal", 50., 80., 0), (50., 80., 0))

    def test_policy_roundtrip_only_edits_attributes(self):
        spec = json.loads(json.dumps(self.policy.to_dict()))
        loaded = AttributePolicy.from_dict(spec)
        self.assertEqual(json.dumps(loaded.to_dict()), json.dumps(spec))
        self.assertEqual(loaded.resolve("best"), "density-small")
        raw = fuse_frame([[{"box": [10, 10, 20, 20], "score": .8, "form": "bulk",
                            "tail": "bulk metal 300 750 0"}]], [1.], ())
        fields = raw.split()
        self.assertEqual(loaded.prediction("density", raw), raw)
        self.assertEqual(loaded.prediction("best", raw).split()[:-1], fields[:-1])
        self.assertEqual(loaded.prediction("best", raw).split()[-1], "1")
        both = loaded.prediction("density-small+bulk", raw).split()
        self.assertEqual(both[:7], fields[:7])
        self.assertNotEqual(both[7], fields[7])
        self.assertAlmostEqual(float(both[8]) / float(both[7]), 750 / 300)
        self.assertEqual(both[-1], "1")
        self.assertEqual(loaded.prediction("off", raw), raw)
        self.assertEqual(loaded.prediction("best", "none"), "none")
        self.assertEqual(loaded.apply("best", "bulk", "metal", 0., 0., 0), (0., 0., 0))

    def test_rescue_requires_both_small_volume_and_high_probability(self):
        apply = self.policy.apply
        self.assertEqual(apply("rescue-small", "bulk", "paper", 180., 1., 0)[2], 1)
        self.assertEqual(apply("rescue-small", "bulk", "paper", 201., 1., 0)[2], 0)
        self.assertEqual(apply("rescue-small", "solid", "metal", 180., 600., 0)[2], 0)
        self.assertEqual(apply("rescue-small", "solid", "metal", 180., 600., 1)[2], 1)


if __name__ == "__main__":
    unittest.main()
