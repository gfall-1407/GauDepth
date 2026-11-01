"""Actual train config writer -> mesh argument merge and pipeline extraction."""

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from arguments import ModelParams, PipelineParams, get_combined_args, write_model_config


class DepthConfigRoundtripTest(TestCase):
    def mesh_pipeline(self, model_path, extra=()):
        parser = ArgumentParser()
        ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser, sentinel=True)
        with patch("sys.argv", ["mesh_extract.py", "-m", str(model_path), *extra]):
            return pipeline.extract(get_combined_args(parser))

    def test_candidate_saved_then_loaded_without_cli_flag(self):
        root = Path(__file__).parent.resolve()
        cfg = root / "cfg_args"
        try:
            dataset = SimpleNamespace(model_path=str(root), source_path=str(root), depth_ratio=0.)
            pipe = SimpleNamespace(convert_SHs_python=False,
                                   compute_cov3D_python=False, debug=False,
                                   depth_estimator="contribution_band")
            write_model_config(cfg, dataset, pipe)
            loaded = self.mesh_pipeline(root)
            self.assertEqual(loaded.depth_estimator, "contribution_band")
            self.assertIs(loaded.debug, False)
        finally:
            cfg.unlink(missing_ok=True)

    def test_old_config_cli_candidate_and_complete_defaults(self):
        root = Path(__file__).parent.resolve()
        cfg = root / "cfg_args"
        try:
            cfg.write_text(
                f"Namespace(model_path={str(root)!r}, source_path={str(root)!r}, depth_ratio=0.0)")
            loaded = self.mesh_pipeline(root, ("--depth_estimator", "contribution_band"))
            self.assertEqual(loaded.depth_estimator, "contribution_band")
            self.assertIs(loaded.debug, False)
            self.assertIs(loaded.compute_cov3D_python, False)
            self.assertIs(loaded.convert_SHs_python, False)
        finally:
            cfg.unlink(missing_ok=True)
