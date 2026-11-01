"""CPU-only source/boundary contract; GPU parity still needs the built extension."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "cuda_rasterizer"


def tile(x, width, sampled):
    tiles = (width + 15) // 16
    if sampled:
        return min(tiles - 1, max(0, int((x + 0.5) / 16)))
    return int(x) // 16


class TileContract(unittest.TestCase):
    def test_continuous_boundary(self):
        cases = [
            (0.0, 0, 0), (15.0, 0, 0), (15.49, 0, 0),
            (15.5, 0, 1), (15.75, 0, 1), (16.0, 1, 1),
            (31.5, 1, 2), (31.75, 1, 2), (32.0, 2, 2),
        ]
        for x, render_tile, sample_tile in cases:
            with self.subTest(x=x):
                self.assertEqual(tile(x, 48, False), render_tile)
                self.assertEqual(tile(x, 48, True), sample_tile)
        self.assertEqual(tile(16.0, 17, True), 1)

    def test_source_uses_native_sample_binning_without_rounding_alpha(self):
        native = (CUDA / "rasterizer_impl.cu").read_text()
        query = (CUDA / "support_forward.cu").read_text()
        self.assertIn("(p.x + 0.5f) / BLOCK_X", native)
        self.assertIn("(p.y + 0.5f) / BLOCK_Y", native)
        self.assertIn("(pix.x + 0.5f) / BLOCK_X", query)
        self.assertIn("(pix.y + 0.5f) / BLOCK_Y", query)
        self.assertIn("static_cast<int>(pix.x) / BLOCK_X", query)
        self.assertIn("const float2 d = {mean.x - pix.x, mean.y - pix.y}", query)


if __name__ == "__main__":
    unittest.main()
