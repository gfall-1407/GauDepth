import math
import unittest
from dataclasses import replace

from visible_branch_reference import (Camera, Member, candidate_depth, choose_branch,
                                      intersection, ncc_cost, patch_cost, plane_loss)


O, V = (0., 0., 0.), (0., 0., 1.)


def member(i, z, alpha, T=1., footprint=1., normal=(0., 0., 1.)):
    return Member(i, (0., 0., z), normal, alpha, T, footprint)


class VisibleBranchReferenceTest(unittest.TestCase):
    def test_two_real_camera_patch_warps(self):
        I = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        reference = Camera((0., 0., 0.), I, 10., 10., 20., 20.)
        witnesses = [Camera((.2, 0., 0.), I, 10., 10., 20., 20.),
                     Camera((-.2, 0., 0.), I, 10., 10., 20., 20.)]
        def texture(x, y):
            return math.sin(.63*x) + .22*math.sin(1.37*x) + .1*math.sin(.47*y)
        ref_image = [[texture(x, y) for x in range(50)] for y in range(50)]
        images = [[[texture(x+shift, y) for x in range(50)] for y in range(50)]
                  for shift in (2., -2.)]
        costs = [[patch_cost(member(i, z, .5), reference, ref_image,
                             cam, img, 23., 23.) for i, z in enumerate((1., 2.))]
                 for cam, img in zip(witnesses, images)]
        self.assertLess(costs[0][0], costs[0][1])
        self.assertLess(costs[1][0], costs[1][1])

    def test_two_textured_layers_and_fd(self):
        ms = [member(0, 1., .02), member(1, 1.02, .03), member(2, 2., .9)]
        depths = [intersection(m, O, V)[2] for m in ms]
        projections = [[(0., 0.), (.2, 0.), (8., 0.)],
                       [(0., 0.), (.15, 0.), (9., 0.)]]
        costs = [[0., .01, .7], [0., .01, .8]]
        branch, reason = choose_branch(depths, projections, costs,
                                       [costs, costs, costs, costs])
        self.assertEqual((branch, reason), ((0, 1), "valid"))
        d, valid, _ = candidate_depth(ms, O, V, costs, branch, 1.948)
        self.assertTrue(valid)
        self.assertAlmostEqual(d, 1.011516, places=5)
        for index in branch:
            eps = 1e-5
            plus = ms.copy(); minus = ms.copy()
            plus[index] = replace(ms[index], mu=(0., 0., ms[index].mu[2]+eps))
            minus[index] = replace(ms[index], mu=(0., 0., ms[index].mu[2]-eps))
            grad = (candidate_depth(plus, O, V, costs, branch, 1.948)[0] -
                    candidate_depth(minus, O, V, costs, branch, 1.948)[0])/(2*eps)
            self.assertGreater(grad, 0.)
            self.assertAlmostEqual(grad, [0.424222, 0.575778][index], places=5)
        for index in branch:
            eps = 1e-6
            plus = ms.copy(); minus = ms.copy()
            plus[index] = replace(ms[index], alpha=ms[index].alpha+eps)
            minus[index] = replace(ms[index], alpha=ms[index].alpha-eps)
            grad = (candidate_depth(plus, O, V, costs, branch, 1.948)[0] -
                    candidate_depth(minus, O, V, costs, branch, 1.948)[0])/(2*eps)
            self.assertAlmostEqual(grad, [-.244258, .162838][index], places=5)

    def test_primitive_splitting_preserves_aggregate_weight(self):
        a = 1 - .5**.25
        unsplit = [member(0, 1., .5), member(1, 2., .5, .5)]
        split = [member(i, 1., a, (1-a)**i) for i in range(4)]
        self.assertAlmostEqual(sum(m.alpha*m.transmittance for m in split), .5)
        split.append(member(4, 2., .5, .5))
        c0 = [[0., 0.], [0., 0.]]
        c1 = [[0.]*5, [0.]*5]
        d0 = candidate_depth(unsplit, O, V, c0, (0, 1), 0.)[0]
        d1 = candidate_depth(split, O, V, c1, tuple(range(5)), 0.)[0]
        self.assertAlmostEqual(d0, d1)

    def test_ambiguous_and_perturbation_fallback(self):
        depths = [1., 2.]
        projections = [[(0., 0.), (8., 0.)], [(0., 0.), (9., 0.)]]
        self.assertEqual(choose_branch(depths, projections,
                                       [[.1, .1], [.1, .1]],
                                       [[[.1, .1], [.1, .1]]]*4)[1],
                         "witness_disagreement")
        self.assertEqual(choose_branch(depths, projections,
                         [[0., 1.], [0., 1.]],
                         [[[1., 0.], [0., 1.]]]*4)[1], "unstable_branch")
        self.assertIsNone(ncc_cost([1.]*25, [1.]*25))
        self.assertEqual(choose_branch([1.], [[(0., 0.)], [(0., 0.)]],
                                       [[None], [None]], [[[0.], [0.]]]*4)[1],
                         "invalid_hypothesis_data")
        self.assertEqual(plane_loss([1., 2.], [0., 0.], [False, False],
                                    [1.5, 1.5]), .5)

    def test_audit_split_tie_and_required_perturbations(self):
        depths = [1., 1., 2.]
        projections = [[(0., 0.), (0., 0.), (8., 0.)],
                       [(0., 0.), (0., 0.), (9., 0.)]]
        tied = [[.1, .1, .1], [.1, .1, .1]]
        self.assertEqual(choose_branch(depths, projections, tied,
                                       [tied]*4)[1], "witness_disagreement")
        # Without the split copy, the two distinct surfaces are also ambiguous.
        two = [[.1, .1], [.1, .1]]
        self.assertEqual(choose_branch([1., 2.],
                                       [v[::2] for v in projections], two,
                                       [two]*4)[1], "witness_disagreement")
        single_projection = [[(0., 0.)], [(0., 0.)]]
        single_cost = [[0.], [0.]]
        self.assertEqual(choose_branch([1.], single_projection, single_cost)[1],
                         "missing_perturbations")
        self.assertEqual(choose_branch([1.], single_projection, single_cost,
                                       [single_cost]*3)[1],
                         "missing_perturbations")
        self.assertEqual(choose_branch([1.], single_projection, single_cost,
                                       [[[None], [None]]]*4)[1],
                         "invalid_perturbation_cost")
        self.assertEqual(choose_branch([1.], single_projection, single_cost,
                                       [[[float('nan')], [0.]]]*4)[1],
                         "invalid_perturbation_cost")

    def test_audit_nonfinite_and_invalid_costs(self):
        patch = [float(i) for i in range(25)]
        for bad in (float('nan'), float('inf'), -float('inf')):
            corrupted = patch.copy(); corrupted[0] = bad
            self.assertIsNone(ncc_cost(corrupted, patch))
            self.assertIsNone(ncc_cost(patch, corrupted))
        projections = [[(0., 0.), (8., 0.)], [(0., 0.), (9., 0.)]]
        costs = [[0., .2], [0., .2]]
        for bad in (float('nan'), float('inf')):
            broken = [[(bad, 0.), (8., 0.)], [(0., 0.), (9., 0.)]]
            self.assertEqual(choose_branch([1., 2.], broken, costs,
                                           [costs]*4)[1], "invalid_hypothesis_data")
            self.assertEqual(choose_branch([bad], [[(0., 0.)], [(0., 0.)]],
                                           [[0.], [0.]],
                                           [[[0.], [0.]]]*4)[1],
                             "invalid_hypothesis_data")
        for bad in (None, -.1, float('nan')):
            broken = [[bad, .2], [0., .2]]
            self.assertEqual(choose_branch([1., 2.], projections, broken,
                                           [costs]*4)[1], "invalid_hypothesis_data")

    def test_degenerate_grazing_low_opacity(self):
        self.assertIsNone(intersection(member(0, 1., .5, normal=(0., 0., 0.)), O, V))
        self.assertIsNone(intersection(member(0, 1., .5, normal=(1., 0., 0.)), O, V))
        tiny = [member(0, 1., 1e-6)]
        self.assertEqual(candidate_depth(tiny, O, V, [[0.], [0.]], (0,), 2.)[:2],
                         (1., True))
        # The native accepted-record source may omit this member at alpha<1/255.
        self.assertEqual(candidate_depth(tiny, O, V, [[0.], [0.]], (), 2.)[:2],
                         (2., False))


if __name__ == "__main__":
    unittest.main()
