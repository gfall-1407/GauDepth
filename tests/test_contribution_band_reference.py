"""CPU checks for the native contribution-band derivative, no torch required."""

import math
import random
import unittest

from utils.contribution_band_reference import band_depth_and_grad


class ContributionBandReferenceTest(unittest.TestCase):
    def finite_difference(self, alpha, depth, field, index, step=1e-6):
        args_plus = [alpha.copy(), depth.copy()]
        args_minus = [alpha.copy(), depth.copy()]
        args_plus[field][index] += step
        args_minus[field][index] -= step
        return (
            band_depth_and_grad(*args_plus)[0]
            - band_depth_and_grad(*args_minus)[0]
        ) / (2 * step)

    def test_single_member(self):
        for alpha in (0.004, 0.1, 0.99):
            value, depth_grad, alpha_grad = band_depth_and_grad([alpha], [2.3])
            self.assertAlmostEqual(value, 2.3)
            self.assertAlmostEqual(depth_grad[0], 1)
            self.assertAlmostEqual(alpha_grad[0], 0)

    def test_partial_endpoints_and_derivatives(self):
        # Both interval boundaries cut through a member.
        alpha = [0.2, 0.4, 0.3, 0.5]
        depth = [1.0, 1.2, 1.6, 2.0]
        value, depth_grad, alpha_grad = band_depth_and_grad(alpha, depth)
        self.assertAlmostEqual(sum(depth_grad), 1)
        self.assertTrue(1.0 < value < 2.0)
        for i in range(len(alpha)):
            self.assertAlmostEqual(
                depth_grad[i], self.finite_difference(alpha, depth, 1, i), delta=1e-8
            )
            self.assertAlmostEqual(
                alpha_grad[i], self.finite_difference(alpha, depth, 0, i), delta=1e-7
            )

    def test_random_fixed_accepted_set_gradient(self):
        rng = random.Random(20261101)
        checked = 0
        for _ in range(200):
            n = rng.randint(2, 8)
            alpha = [rng.uniform(0.02, 0.8) for _ in range(n)]
            depth = [rng.uniform(0.5, 3) for _ in range(n)]
            # At exact interval boundaries the derivative is undefined.
            t = 1.0
            prefix = []
            for a in alpha:
                t *= 1 - a
                prefix.append(1 - t)
            total = prefix[-1]
            if any(min(abs(s / total - .25), abs(s / total - .75)) < 1e-4 for s in prefix):
                continue
            _, depth_grad, alpha_grad = band_depth_and_grad(alpha, depth)
            for i in range(n):
                self.assertAlmostEqual(depth_grad[i], self.finite_difference(alpha, depth, 1, i), delta=1e-8)
                self.assertAlmostEqual(alpha_grad[i], self.finite_difference(alpha, depth, 0, i), delta=1e-7)
            checked += 1
        self.assertGreater(checked, 190)

    def test_separated_surfaces_can_still_average(self):
        value, _, _ = band_depth_and_grad([0.5, 0.5], [1, 2])
        self.assertGreater(value, 1)
        self.assertLess(value, 2)

    def test_invalid_total(self):
        with self.assertRaises(ValueError):
            band_depth_and_grad([0, 0], [1, 2])


if __name__ == "__main__":
    unittest.main()
