from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import verify_golden_adapter_manifest as verifier


class VerifyGoldenAdapterManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="g1b-golden-test-")
        self.root = Path(self.temporary_directory.name)
        self.openpi_root = self.root / "openpi"
        source = self.openpi_root / "src/openpi/models/lora.py"
        source.parent.mkdir(parents=True)
        source.write_text("synthetic pinned lora declarations\n", encoding="utf-8")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

        self.tree = self.root / "tree.json"
        self.tree.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "path": "Model/attn/lora_a",
                            "parent_path": "Model/attn",
                            "terminal": "lora_a",
                            "shape": [4, 2],
                            "parameter_count": 8,
                            "variable_type": "flax.nnx.variablelib.Param",
                            "dtype": "float32",
                        },
                        {
                            "path": "Model/attn/kernel",
                            "parent_path": "Model/attn",
                            "terminal": "kernel",
                            "shape": [4, 4],
                            "parameter_count": 16,
                            "variable_type": "flax.nnx.variablelib.Param",
                            "dtype": "float32",
                        },
                    ]
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tree_hash = hashlib.sha256(self.tree.read_bytes()).hexdigest()
        self.manifest = self.root / "manifest.json"
        self.manifest_value = {
            "status": "g1b_golden_frozen",
            "source_identity": {"src/openpi/models/lora.py": source_hash},
            "evidence_identity": {"full_param_tree_sha256": tree_hash},
            "independent_review": {
                "source_declared_terminals": ["lora_a"],
                "legal_parent_paths": ["Model/attn"],
            },
            "review_invariants": {
                "variable_type": "flax.nnx.variablelib.Param",
                "dtype": "float32",
                "adapter_leaf_count": 1,
                "adapter_parameter_count": 8,
                "total_param_leaf_count": 2,
                "total_parameter_count": 24,
                "non_adapter_leaf_count": 1,
                "non_adapter_parameter_count": 16,
            },
            "protected_non_adapter_scopes": ["Model/protected"],
            "entries": [
                {"path": "Model/attn/lora_a", "shape": [4, 2], "parameter_count": 8}
            ],
        }
        self._write_manifest()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_manifest(self) -> None:
        self.manifest.write_text(json.dumps(self.manifest_value, sort_keys=True), encoding="utf-8")

    def _run(self, output_name: str = "verification.json") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(Path(verifier.__file__).resolve()),
                "--tree",
                str(self.tree),
                "--manifest",
                str(self.manifest),
                "--openpi-root",
                str(self.openpi_root),
                "--output",
                str(self.root / output_name),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_exact_frozen_manifest_passes(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "verified_g1b_golden_frozen")

    def test_extra_base_parameter_fails_closed(self) -> None:
        self.manifest_value["entries"].append(
            {"path": "Model/attn/kernel", "shape": [4, 4], "parameter_count": 16}
        )
        self._write_manifest()
        result = self._run("extra.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from independently derived exact path set", result.stderr)

    def test_source_hash_drift_fails_closed(self) -> None:
        source = self.openpi_root / "src/openpi/models/lora.py"
        source.write_text("changed declarations\n", encoding="utf-8")
        result = self._run("drift.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source hash mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
