from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import jax

import adapter_artifact as artifact


def golden() -> dict:
    return {"review_invariants": {"adapter_leaf_count": 2}, "entries": [
        {"path": "m/lora_a", "shape": [2, 1]},
        {"path": "m/lora_b", "shape": [1, 2]},
    ]}


IDENTITIES = {name: char * 64 for name, char in zip(sorted(artifact.REQUIRED_IDENTITIES), "1234", strict=True)}


class AdapterArtifactTest(unittest.TestCase):
    def test_array_hash_accepts_bfloat16_storage(self) -> None:
        try:
            import ml_dtypes
        except ImportError:
            self.skipTest("ml_dtypes is not installed")
        values = np.asarray([1.0, 2.0], dtype=ml_dtypes.bfloat16)
        self.assertEqual(artifact.array_sha256(values), artifact.array_sha256(values.copy()))

    def test_roundtrip_and_non_adapter_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = {"m": {"kernel": np.arange(4, dtype=np.float32).reshape(2, 2)}}
            reference = {"m": {**base["m"], "lora_a": np.ones((2, 1), np.float32), "lora_b": np.zeros((1, 2), np.float32)}}
            before = artifact.array_sha256(base["m"]["kernel"])
            first = artifact.export_adapter(reference, golden(), root / "a", identities=IDENTITIES, train_step=7, train_seed=9)
            combined = artifact.compose_adapter(base, reference, golden(), root / "a", expected_identities=IDENTITIES)
            second = artifact.export_adapter(combined, golden(), root / "b", identities=IDENTITIES, train_step=7, train_seed=9)
            self.assertEqual(first["adapter_identity_sha256"], second["adapter_identity_sha256"])
            self.assertEqual(before, artifact.array_sha256(combined["m"]["kernel"]))

    def test_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = {"m": {"kernel": np.ones((1,), np.float32), "lora_a": np.ones((2, 1), np.float32), "lora_b": np.zeros((1, 2), np.float32)}}
            artifact.export_adapter(reference, golden(), root / "a", identities=IDENTITIES, train_step=0, train_seed=0)
            wrong = dict(IDENTITIES)
            wrong["norm_stats_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                artifact.compose_adapter({"m": {"kernel": np.ones((1,), np.float32)}}, reference, golden(), root / "a", expected_identities=wrong)

    def test_unexpected_artifact_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = {"m": {"kernel": np.ones((1,), np.float32), "lora_a": np.ones((2, 1), np.float32), "lora_b": np.zeros((1, 2), np.float32)}}
            artifact.export_adapter(reference, golden(), root / "a", identities=IDENTITIES, train_step=0, train_seed=0)
            (root / "a" / "extra").write_text("bad")
            with self.assertRaisesRegex(ValueError, "unexpected files"):
                artifact.compose_adapter({"m": {"kernel": np.ones((1,), np.float32)}}, reference, golden(), root / "a", expected_identities=IDENTITIES)

    def test_shape_dtype_struct_reference_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            values = {"m": {"lora_a": np.ones((2, 1), np.float32), "lora_b": np.zeros((1, 2), np.float32)}}
            reference = {"m": {"kernel": jax.ShapeDtypeStruct((1,), np.float32), "lora_a": jax.ShapeDtypeStruct((2, 1), np.float32), "lora_b": jax.ShapeDtypeStruct((1, 2), np.float32)}}
            base = {"m": {"kernel": np.ones((1,), np.float32)}}
            artifact.export_adapter(values, golden(), root / "a", identities=IDENTITIES, train_step=0, train_seed=0)
            combined = artifact.compose_adapter(base, reference, golden(), root / "a", expected_identities=IDENTITIES)
            self.assertEqual(combined["m"]["lora_a"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
