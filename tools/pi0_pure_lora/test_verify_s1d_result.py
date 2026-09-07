#!/usr/bin/env python3

import json
import pathlib
import tempfile
import unittest

import experiment_identity
import verify_s1d_result as verify


class VerifyS1dHelpersTest(unittest.TestCase):
    def test_artifact_manifest_is_stable_and_counts_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "nested").mkdir()
            (root / "a.bin").write_bytes(b"abc")
            (root / "nested/b.bin").write_bytes(b"defg")
            first = verify._artifact_manifest(root)
            second = verify._artifact_manifest(root)
            self.assertEqual(first, second)
            self.assertEqual(first["file_count"], 2)
            self.assertEqual(first["total_bytes"], 7)

    def test_adapter_identity_excludes_only_identity_field(self):
        stable = {
            "schema_version": 1,
            "artifact_type": "pi0_pure_lora_adapter_only",
            "identities": {"a": "b"},
            "train_step": 100,
            "train_seed": 42,
            "entries": [],
        }
        manifest = dict(stable)
        manifest["adapter_identity_sha256"] = experiment_identity.canonical_sha256(stable)
        self.assertEqual(verify._adapter_identity(manifest), manifest["adapter_identity_sha256"])

    def test_partial_and_temporary_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "committed").mkdir()
            self.assertFalse(verify._has_partial_or_temporary_path(root))
            (root / "step.partial-12").mkdir()
            self.assertTrue(verify._has_partial_or_temporary_path(root))

    def test_metric_value_parser_rejects_non_numeric(self):
        self.assertEqual(verify._metric_values([{"loss": 1.0}, {"grad_norm": 2}]), [1.0, 2.0])
        self.assertEqual(verify._metric_values([{"loss": "bad"}]), [])
        self.assertEqual(verify._metric_values({"loss": 1.0}), [])


if __name__ == "__main__":
    unittest.main()
