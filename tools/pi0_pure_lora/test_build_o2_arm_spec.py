from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import build_o2_arm_spec as spec


class BuildO2ArmSpecTest(unittest.TestCase):
    def test_exact_order_modes_and_port_uniqueness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file_names = {"python", "server_source", "base_manifest", "golden", "model_manifest", "norm_stats", "runtime_recipe"}
            for name in ("python", "openpi_root", "official_checkpoint", "server_source", "base_params", "base_manifest", "golden", "adapter", "model_manifest", "norm_stats", "runtime_recipe"):
                path = root / name; path.mkdir()
                if name in file_names: path.rmdir(); path.write_text("")
                if name == "official_checkpoint": (path / "params").mkdir()
                if name == "adapter": (path / "manifest.json").write_text("{}")
                if name == "openpi_root": (path / "scripts").mkdir(); (path / "scripts/serve_policy.py").write_text("")
            values = {name: root / name for name in ("python", "openpi_root", "official_checkpoint", "server_source", "base_params", "base_manifest", "golden", "adapter", "model_manifest", "norm_stats", "runtime_recipe")}
            args = argparse.Namespace(**values, config_patch_sha256="a" * 64, rng_seed=0, ports=[18041, 18042, 18043], output=root / "out.json")
            arms = spec.build(args)
            self.assertEqual([arm["name"] for arm in arms], list(spec.ARM_NAMES))
            self.assertIn("unmerged-lora", arms[1]["command"]); self.assertIn("merged-dense", arms[2]["command"])
            args.ports = [18041, 18041, 18043]
            with self.assertRaises(ValueError): spec.build(args)


if __name__ == "__main__":
    unittest.main()
