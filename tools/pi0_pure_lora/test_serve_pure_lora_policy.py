from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import json

import serve_pure_lora_policy as server


class ServePureLoraPolicyTest(unittest.TestCase):
    def test_validate_paths_rejects_missing_before_any_model_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = type("Args", (), {
                "openpi_root": root / "openpi", "base_params": root / "params",
                "base_manifest": root / "base.json", "golden": root / "golden.json",
                "norm_stats": root / "norm.json", "adapter": root / "adapter",
                "model_manifest": root / "model.json", "port": 8000,
            })()
            old_preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
            old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            with self.assertRaises(FileNotFoundError):
                server.validate_paths(args)
            if old_preallocate is None:
                os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
            else:
                os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = old_preallocate
            if old_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_visible

    def test_server_source_has_no_simulator_import(self):
        source = Path(server.__file__).read_text(encoding="utf-8")
        self.assertIn("base_plus_adapter", source)
        self.assertIn('mode not in {"base", "base_plus_adapter"}', source)
        self.assertNotIn("E1 server requires a base_plus_adapter", source)
        self.assertNotIn("OffScreenRenderEnv", source)

    def test_base_identity_comes_from_c0_manifest_not_its_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "base.json"
            expected = "a" * 64
            manifest.write_text(json.dumps({"identities": {"base_manifest_sha256": expected}}), encoding="utf-8")
            self.assertEqual(server.base_identity_from_manifest(manifest), expected)
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                server.base_identity_from_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
