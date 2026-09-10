from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_ft0_formal_training_freeze as builder
import run_formal_segment as runner
import autonomous_stage_orchestrator as orchestrator
from test_build_ft0_formal_training_freeze import Ft0FormalTrainingFreezeTests


class FormalRunnerStaticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="formal-runner-"); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        a = Ft0FormalTrainingFreezeTests().args(); a.checkpoint_root = str(self.root / "checkpoints"); a.adapter_root = str(self.root / "adapters"); self.run_name = a.run_name
        self.freeze = self.root / "ft0.json"; self.freeze.write_text(json.dumps(builder.build(a)))
        self.openpi = self.root / "openpi"; self.openpi.mkdir(); self.model = self.root / "model"; self.model.write_text("{}")
        self.golden = self.root / "golden"; self.golden.write_text("{}")
    def args(self):
        attempt=self.root / "attempt"
        return argparse.Namespace(openpi_root=self.openpi, model_manifest=self.model, golden_manifest=self.golden, ft0_package=self.freeze, attempt_dir=attempt, checkpoint_root=self.root / "checkpoints" / self.run_name, adapter_root=self.root / "adapters" / self.run_name, progress=attempt / "progress.json", loader_receipt=attempt / "loader.json", rng_receipt=attempt / "rng.json", composition_receipt=attempt / "composition.json", output=attempt / "result.json", run_name=self.run_name, segment_start=0, segment_end=1000, previous_result=None, expected_previous_result_sha256=None)
    def test_fresh_ft1_static_gate(self):
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES":"1", "XLA_PYTHON_CLIENT_PREALLOCATE":"false"}, clear=False):
            receipt=runner._validate_local_inputs(self.args())
        self.assertEqual(receipt["resume_mode"], "fresh_pi0_base")
    def test_fresh_ft1_rejects_prior_result(self):
        args=self.args(); args.previous_result=self.root / "old.json"; args.expected_previous_result_sha256="a"*64
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES":"1", "XLA_PYTHON_CLIENT_PREALLOCATE":"false"}, clear=False), self.assertRaisesRegex(ValueError, "must not accept"):
            runner._validate_local_inputs(args)
    def test_later_segment_requires_prior_result_identity(self):
        args=self.args(); args.segment_start=1000; args.segment_end=5000; args.checkpoint_root.joinpath("1000").mkdir(parents=True); args.adapter_root.mkdir(parents=True); args.adapter_root.joinpath("step-00001000.verified.json").write_text("{}")
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES":"1", "XLA_PYTHON_CLIENT_PREALLOCATE":"false"}, clear=False), self.assertRaisesRegex(ValueError, "requires its previous"):
            runner._validate_local_inputs(args)
    def test_initial_progress_schema_is_orchestrator_compatible(self):
        progress = {"current_step": 0, "last_committed_step": 0}
        self.assertTrue(set(progress) <= orchestrator.PROGRESS_KEYS)


if __name__ == "__main__": unittest.main()
