"""Check the actual training normal/PM depth route without CUDA imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class NormalDepthRouteTest(unittest.TestCase):
    def test_ratio_and_estimator_route(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "train.py").read_text())
        route = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and "depth_estimator" in ast.dump(node.test))
        selected = compile(ast.Expression(route.test), "train.py", "eval")
        double = compile(ast.Expression(route.orelse[0].test), "train.py", "eval")
        selected_body = " ".join(ast.dump(node) for node in route.body)
        self.assertIn("selected_depth", selected_body)
        self.assertIn("depth_to_normal", selected_body)
        self.assertNotIn("value=Constant(value=None)", selected_body)
        for estimator in ("baseline", "contribution_band", "contribution_band_sg"):
            for ratio in (0., .5, 1.):
                scope = {"pipe": SimpleNamespace(depth_estimator=estimator),
                         "dataset": SimpleNamespace(depth_ratio=ratio)}
                actual = ("selected" if eval(selected, {}, scope) else
                          "double" if eval(double, {}, scope) else "single")
                expected = ("selected" if estimator != "baseline" else
                            "double" if ratio == .5 else "single")
                self.assertEqual(actual, expected, (estimator, ratio))


if __name__ == "__main__":
    unittest.main()
