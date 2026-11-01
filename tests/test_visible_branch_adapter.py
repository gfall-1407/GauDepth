import math
import unittest

from visible_branch_adapter import evaluate_fixed_32, evaluate_ray
from visible_branch_reference import Camera, Member


class AdapterTest(unittest.TestCase):
    def setUp(self):
        eye = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        self.reference = Camera((0., 0., 0.), eye, 10., 10., 20., 20.)
        cameras = [Camera((.2, 0., 0.), eye, 10., 10., 20., 20.),
                   Camera((-.2, 0., 0.), eye, 10., 10., 20., 20.)]
        texture = lambda x, y: math.sin(.63*x)+.22*math.sin(1.37*x)+.1*math.sin(.47*y)
        self.ref_image = [[texture(x, y) for x in range(50)] for y in range(50)]
        images = [[[texture(x+shift, y) for x in range(50)] for y in range(50)]
                  for shift in (2., -2.)]
        self.witnesses = list(zip(cameras, images))
        self.members = [Member(7, (0., 0., 1.), (0., 0., 1.), .1, 1., 1.),
                        Member(9, (0., 0., 2.), (0., 0., 1.), .8, .9, 1.)]

    def test_actual_shifted_patch_costs_and_all_rays(self):
        ray = dict(members=self.members, reference_camera=self.reference,
                   reference_image=self.ref_image, witnesses=self.witnesses,
                   pixel=(23., 23.), expected_depth=1.9)
        result = evaluate_ray(**ray)
        self.assertTrue(result.valid)
        self.assertEqual(result.branch_ids, (7,))
        self.assertAlmostEqual(result.candidate_depth, 1.)
        self.assertLess(result.costs[0][0], result.costs[0][1])
        batch = evaluate_fixed_32([ray]*32)
        self.assertEqual(len(batch), 32)
        self.assertEqual(sum(r.valid for r in batch), 32)

    def test_untextured_fallback_keeps_entry(self):
        flat = [[1.]*50 for _ in range(50)]
        result = evaluate_ray(self.members, self.reference, flat,
                              [(cam, flat) for cam, _ in self.witnesses],
                              (23., 23.), 1.9)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "invalid_hypothesis_data")
        self.assertEqual(result.candidate_depth, 1.9)
        self.assertEqual(result.accepted_count, 2)


if __name__ == "__main__":
    unittest.main()
