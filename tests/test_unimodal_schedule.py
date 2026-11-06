"""CPU-only checks for the training iteration gate, without importing CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]

class UnimodalScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / 'train.py').read_text())
        matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'unimodal_enabled' for t in n.targets)]
        assert len(matches) == 1
        cls.condition = compile(ast.Expression(matches[0].value), 'train.py', 'eval')

    def enabled(self, weight, threshold, iteration):
        return eval(self.condition, {}, {'opt': SimpleNamespace(lambda_unimodal=weight, unimodal_from_iter=threshold), 'iteration': iteration})

    def test_disabled_weight(self):
        self.assertFalse(self.enabled(0, 0, 1))
        self.assertFalse(self.enabled(0, 7000, 7001))

    def test_default_and_boundary(self):
        self.assertTrue(self.enabled(0.1, 0, 1))
        self.assertFalse(self.enabled(0.1, 7000, 6999))
        self.assertTrue(self.enabled(0.1, 7000, 7000))
        self.assertTrue(self.enabled(0.1, 7000, 7001))

    def test_resumed_absolute_iteration(self):
        self.assertTrue(self.enabled(0.1, 7000, 15001))
        self.assertFalse(self.enabled(0.1, 16000, 15001))

if __name__ == '__main__':
    unittest.main()
