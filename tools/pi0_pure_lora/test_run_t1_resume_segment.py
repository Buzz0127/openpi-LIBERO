#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_t1_resume_segment as runner
import verify_s1d_result


class T1ResumeSegmentStaticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="t1-resume-")
        self.root = Path(self.temp.name)
        self.checkpoints = self.root / "checkpoints"
        (self.checkpoints / "100").mkdir(parents=True)
        (self.checkpoints / "100/state").write_text("checkpoint\n")
        self.adapters = self.root / "adapters"
        self.adapters.mkdir()
        (self.adapters / "step-00000100.verified.json").write_text("{}\n")
        self.model = self.root / "model.json"; self.model.write_text("{}\n")
        self.golden = self.root / "golden.json"; self.golden.write_text("{}\n")
        self.acceptance = self.root / "acceptance.json"
        checkpoint_tree = verify_s1d_result._artifact_manifest(self.checkpoints)
        adapter_tree = verify_s1d_result._artifact_manifest(self.adapters)
        payload = {"status": "pass", "report_identity_sha256": "a" * 64, "checkpoint_artifact": checkpoint_tree, "adapter_artifact": adapter_tree, "adapter_identity_sha256": "c" * 64}
        self.acceptance.write_text(json.dumps(payload) + "\n")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self) -> argparse.Namespace:
        attempt = self.root / "attempt"
        return argparse.Namespace(
            segment_start=100, segment_end=200, train_seed=42, eval_seed=7,
            checkpoint_dir=self.checkpoints, adapter_root=self.adapters,
            model_manifest=self.model, golden_manifest=self.golden,
            s1d_acceptance_report=self.acceptance,
            expected_s1d_acceptance_sha256=hashlib.sha256(self.acceptance.read_bytes()).hexdigest(),
            expected_s1d_report_identity="a" * 64,
            expected_checkpoint_tree_sha256=json.loads(self.acceptance.read_text())["checkpoint_artifact"]["artifact_tree_sha256"],
            expected_adapter_identity_sha256="c" * 64,
            attempt_dir=attempt, progress=attempt / "progress.json",
            loader_receipt=attempt / "loader.json", output=attempt / "result.json",
        )

    def environment(self) -> dict[str, str]:
        return {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "CUDA_VISIBLE_DEVICES": "1"}

    def test_static_gate_accepts_exact_frozen_inputs(self) -> None:
        args = self.args()
        runner._validate_static(args, self.environment())
        result = runner._verify_resume_artifacts(args)
        self.assertEqual(result["checkpoint"]["file_count"], 1)

    def test_wrong_segment_fails_before_heavy_imports(self) -> None:
        args = self.args(); args.segment_end = 201
        with self.assertRaisesRegex(ValueError, "100->200"):
            runner._validate_static(args, self.environment())

    def test_multi_gpu_mapping_fails(self) -> None:
        environment = self.environment(); environment["CUDA_VISIBLE_DEVICES"] = "0,1"
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            runner._validate_static(self.args(), environment)

    def test_tampered_acceptance_fails(self) -> None:
        args = self.args(); args.expected_s1d_acceptance_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            runner._validate_static(args, self.environment())

    def test_existing_output_fails(self) -> None:
        args = self.args(); args.attempt_dir.mkdir(); args.output.write_text("{}\n")
        with self.assertRaises(FileExistsError):
            runner._validate_static(args, self.environment())

    def test_checkpoint_file_tampering_fails(self) -> None:
        args = self.args()
        runner._validate_static(args, self.environment())
        (self.checkpoints / "100/state").write_text("tampered\n")
        with self.assertRaisesRegex(RuntimeError, "checkpoint files"):
            runner._verify_resume_artifacts(args)


if __name__ == "__main__":
    unittest.main()
