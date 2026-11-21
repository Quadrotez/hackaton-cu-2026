import unittest

import numpy as np

from fusion.attribute_postprocess import DensityPrior
from fusion.instance_take_ab import apply_mixture, choose_cut, cluster_ids, consensus, light_scores


class InstanceTakeTest(unittest.TestCase):
    def test_complete_link_does_not_chain(self):
        angles = np.deg2rad([0., 30., 60.])
        vectors = np.column_stack((np.cos(angles), np.sin(angles)))
        groups = cluster_ids(["a", "b", "c"], [[0, 0, 1, 1]] * 3, [1, 1, 1], vectors, vectors)
        self.assertEqual(len(set(groups)), 2)
        self.assertNotEqual(groups[0], groups[2])

    def test_same_frame_does_not_double_vote(self):
        vectors = np.tile([1., 0.], (3, 1))
        groups = cluster_ids(["a", "a", "b"], [[0, 0, 1, 1]] * 3, [1, 1, 1], vectors, vectors)
        self.assertNotEqual(groups[0], groups[1])
        self.assertIn(groups[2], groups[:2])

    def test_consensus_and_singleton(self):
        rows = [dict(score=.9, volume=100., material="metal", take=1),
                dict(score=.2, volume=300., material="wood", take=0),
                dict(score=.8, volume=200., material="glass", take=0)]
        p = np.array([[.7, .3], [.1, .9], [.8, .2]])
        out = consensus(rows, p, ["metal", "wood"], np.array([1, 1, 2]))
        self.assertEqual(out[1], dict(score=.2, volume=200., material="metal", take=1))
        self.assertEqual(out[2], rows[2])
        self.assertEqual(rows[1]["volume"], 300.)

    def test_integrate_full_posterior_and_apply_only_take(self):
        prior = DensityPrior.fit([("a", "solid", "metal", 100., 1000.),
                                  ("b", "solid", "wood", 100., 50.)])
        labels = ["metal", "wood"]
        p = np.array([[.6, .4]])
        mix, hard = light_scores(["solid"], [300.], p, labels, prior)
        self.assertAlmostEqual(hard[0], prior.light_probability("solid", "metal", 300.))
        self.assertAlmostEqual(mix[0], .6 * hard[0] + .4 * prior.light_probability("solid", "wood", 300.))
        self.assertGreater(mix[0], hard[0])
        rows = [dict(box=[0, 0, 10, 10], score=.7, volume=300., form="solid", material="metal", take=0)]
        policy = dict(labels=labels, cut=(mix[0] + hard[0]) / 2, density_entries=prior.entries,
                      temperature=prior.temperature,
                      heads=[dict(coef=[[0, 0], [0, 0]], intercept=np.log(p[0]))] * 2)
        self.assertEqual(apply_mixture(rows, [np.array([[1., 0.]])] * 4, policy), [dict(rows[0], take=1)])
        self.assertEqual(rows[0]["take"], 0)

    def test_cut_maximizes_macro_recall(self):
        p = np.array([.05, .65, .75, .9])
        y = np.array([0, 0, 1, 1])
        cut, s = choose_cut(p, y, np.array(["a", "b", "c", "d"]))
        self.assertEqual(s, 1.)
        np.testing.assert_array_equal(p >= cut, y)


if __name__ == "__main__":
    unittest.main()
