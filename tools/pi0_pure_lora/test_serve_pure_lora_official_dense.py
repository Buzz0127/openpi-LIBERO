from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import serve_pure_lora_official_dense as server


class OfficialDenseServerTest(unittest.TestCase):
    def test_parse_args_accepts_only_the_two_o2_runtime_modes(self):
        common = [
            "program", "--openpi-root", "/tmp/openpi", "--base-params", "/tmp/base",
            "--base-manifest", "/tmp/base.json", "--golden", "/tmp/golden.json", "--adapter", "/tmp/adapter",
            "--model-manifest", "/tmp/model.json", "--norm-stats", "/tmp/norm.json", "--config-patch-sha256", "a" * 64,
            "--runtime-recipe", "/tmp/recipe.json", "--port", "8010", "--rng-seed", "0",
        ]
        with mock.patch("sys.argv", common + ["--runtime-mode", "unmerged-lora"]):
            self.assertEqual(server.parse_args().runtime_mode, "unmerged-lora")
        with mock.patch("sys.argv", common + ["--runtime-mode", "not-a-mode"]):
            with self.assertRaises(SystemExit):
                server.parse_args()

    def test_rng_seed_is_injected_after_constructor_without_truth_testing(self):
        class FakeRandom:
            @staticmethod
            def key(seed):
                return ("typed-key", seed)

        class FakeJax:
            random = FakeRandom()

        class FakePolicy:
            _rng = None

        policy = FakePolicy()
        audit = server.inject_policy_rng(policy, FakeJax(), 17)
        self.assertEqual(policy._rng, ("typed-key", 17))
        self.assertEqual(audit["rng_seed"], 17)
        with self.assertRaises(TypeError):
            server.inject_policy_rng(policy, FakeJax(), "17")

    def test_validate_paths_rejects_before_openpi_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = type("Args", (), {
                "openpi_root": root / "openpi", "base_params": root / "params",
                "base_manifest": root / "base.json", "golden": root / "golden.json",
                "adapter": root / "adapter", "model_manifest": root / "model.json",
                "norm_stats": root / "norm.json", "runtime_recipe": root / "recipe.json", "port": 8010,
            })()
            old_preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
            old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            try:
                with self.assertRaises(FileNotFoundError):
                    server.validate_paths(args)
            finally:
                if old_preallocate is None:
                    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
                else:
                    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = old_preallocate
                if old_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_visible


if __name__ == "__main__":
    unittest.main()
