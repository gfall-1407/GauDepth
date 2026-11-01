"""Full short-run primary/neighbor camera sequence against reviewed f864 selector."""

import ast
import random
from pathlib import Path
from types import SimpleNamespace
import unittest


def train_expression(name):
    tree = ast.parse((Path(__file__).resolve().parents[1] / "train.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return compile(ast.Expression(node.value), "train.py", "eval")
    raise AssertionError(name)


class CameraSequenceTest(unittest.TestCase):
    def test_102_primary_neighbor_pairs_match_reviewed_ab_selector(self):
        cameras = [SimpleNamespace(image_name=str(i),
                                   nearest_id=[j for j in range(8) if j != i])
                   for i in range(8)]
        scene = SimpleNamespace(getTrainCameras=lambda: cameras)
        actual_rng = random.Random(0)
        reference_rng = random.Random(0)
        primary_expr = train_expression("viewpoint_cam")
        neighbor_expr = train_expression("nearest_cam")
        actual_stack, reference_stack = [], []
        actual_pairs, reference_pairs = [], []
        for _ in range(102):  # iterations 7001 through 7102, including final no-step iteration
            if not actual_stack:
                actual_stack = cameras.copy()
            if not reference_stack:
                reference_stack = cameras.copy()
            viewpoint_cam = eval(primary_expr, {"viewpoint_stack": actual_stack,
                                                "ab_rng": actual_rng,
                                                "randint": random.randint})
            nearest_cam = eval(neighbor_expr, {"viewpoint_cam": viewpoint_cam,
                                               "scene": scene,
                                               "ab_rng": actual_rng,
                                               "sample": random.sample})
            actual_pairs.append((viewpoint_cam.image_name, nearest_cam.image_name))

            ref_camera = reference_stack.pop(reference_rng.randrange(len(reference_stack)))
            ref_neighbor = cameras[reference_rng.choice(ref_camera.nearest_id)]
            reference_pairs.append((ref_camera.image_name, ref_neighbor.image_name))

        self.assertEqual(actual_pairs, reference_pairs)
        self.assertEqual(len(actual_pairs), 102)
        self.assertGreater(len(set(actual_pairs)), 8)


if __name__ == "__main__":
    unittest.main()
