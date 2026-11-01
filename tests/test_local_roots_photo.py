import unittest

from local_roots_photo import OFFSETS, decide, finalize_root
from run_local_roots_photo import heldout_summary


def root(index, costs_a, costs_b, p_a, p_b):
    evidence = {}
    for view, costs, point in (("0012", costs_a, p_a), ("0050", costs_b, p_b)):
        evidence[view] = [dict(offset=list(offset), cost=cost, reason="valid" if cost is not None else "oob",
                               projected_center=point, visibility="unverified")
                          for offset, cost in zip(OFFSETS, costs)]
    return finalize_root([0, index], 2.+index, (0., 0., 2.+index),
                         (0., 0., 1.), False, evidence)


class LocalRootsPhotoTest(unittest.TestCase):
    def test_missing_distant_competitor_can_be_excluded(self):
        best = root(0, [.1]*5, [.12]*5, (10., 10.), (10., 10.))
        far = root(1, [None]*5, [.8]*5, (30., 10.), (30., 10.))
        result = decide([best, far], 1.9)
        self.assertEqual(result["selected_root"], [0, 0])
        self.assertEqual(result["possible_competitor_ids"], [[0, 0]])

    def test_unknown_competitor_is_not_silently_removed(self):
        best = root(0, [.1]*5, [.12]*5, (10., 10.), (10., 10.))
        unknown = root(1, [None]*5, [.08]*5, (30., 10.), (30., 10.))
        result = decide([best, unknown], 1.9)
        self.assertEqual(result["decision"], "conditional_missing_competitor")
        self.assertIsNone(result["selected_root"])
        self.assertEqual(result["output_depth"], 1.9)

    def test_same_position_tie_and_observed_ambiguity(self):
        first = root(0, [.1]*5, [.1]*5, (10., 10.), (10., 10.))
        tied = root(1, [.1]*5, [.1]*5, (10.2, 10.), (10.1, 10.))
        self.assertEqual(decide([tied, first], 1.9)["selected_root"], [0, 0])
        far = root(2, [.1]*5, [.1]*5, (11., 10.), (10.7, 10.))
        self.assertEqual(decide([first, far], 1.9)["decision"],
                         "ambiguous_observed_roots")

    def test_unknown_same_position_and_fixed_weak_threshold(self):
        first = root(0, [.1]*5, [.1]*5, (10., 10.), (10., 10.))
        unknown = root(1, [None]*5, [.08]*5, (10.1, 10.), (10.1, 10.))
        self.assertEqual(decide([first, unknown], 1.9)["selected_root"], [0, 0])
        weak = root(2, [.251]*5, [.251]*5, (10., 10.), (10., 10.))
        self.assertEqual(decide([weak], 1.9)["decision"], "weak_match")
        self.assertEqual(decide([], 1.9)["decision"], "no_geometry_candidate")

    def test_missing_heldout_fixed_set_bounds(self):
        rays = [dict(selected_root=None, heldout={"position_delta": None})
                for _ in range(32)]
        rays[0] = dict(selected_root=[0, 0], heldout={"position_delta": -.2})
        rays[1] = dict(selected_root=[1, 0], heldout={"position_delta": None})
        result = heldout_summary(rays)
        self.assertEqual(result["fixed_indices"], [i for i in range(32) if i != 23])
        self.assertEqual(result["missing_indices"], [1])
        self.assertIsNone(result["formal_mean"])
        self.assertAlmostEqual(result["formal_mean_bounds"][0], -1.2/31)
        self.assertAlmostEqual(result["formal_mean_bounds"][1], .8/31)


if __name__ == "__main__":
    unittest.main()
