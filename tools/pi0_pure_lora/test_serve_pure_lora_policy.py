from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import json

import numpy as np

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

    def test_base_composition_materializes_zero_adapter_arrays(self):
        """The Base path must not retain eval_shape placeholders in the tree."""
        class FakeShapeDtype:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype

        traverse = types.SimpleNamespace(
            flatten_dict=lambda params, sep: dict(params),
            unflatten_dict=lambda params, sep: dict(params),
        )
        fake_adapter = types.SimpleNamespace(
            flax=types.SimpleNamespace(traverse_util=traverse),
            golden_entries=lambda golden: {entry["path"]: entry for entry in golden["entries"]},
        )
        golden = {
            "entries": [
                {"path": "lora_a", "shape": [2, 3]},
                {"path": "lora_b", "shape": [3, 4]},
            ],
            "review_invariants": {"dtype": "float32", "adapter_leaf_count": 2},
        }
        base = {"kernel": np.array([7.0], dtype=np.float32)}
        reference = {
            **base,
            "lora_a": FakeShapeDtype((2, 3), np.float32),
            "lora_b": FakeShapeDtype((3, 4), np.float32),
        }
        prior = sys.modules.get("adapter_artifact")
        sys.modules["adapter_artifact"] = fake_adapter
        try:
            composed = server.compose_base_with_reference_lora(base, reference, golden)
        finally:
            if prior is None:
                del sys.modules["adapter_artifact"]
            else:
                sys.modules["adapter_artifact"] = prior
        np.testing.assert_array_equal(composed["kernel"], base["kernel"])
        for path, shape in (("lora_a", (2, 3)), ("lora_b", (3, 4))):
            self.assertIsInstance(composed[path], np.ndarray)
            self.assertEqual(composed[path].shape, shape)
            self.assertEqual(composed[path].dtype, np.dtype("float32"))
            self.assertTrue(np.array_equal(composed[path], np.zeros(shape, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
