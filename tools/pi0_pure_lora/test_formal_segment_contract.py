from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import build_ft0_formal_training_freeze as builder
import formal_segment_contract as contract
from test_build_ft0_formal_training_freeze import Ft0FormalTrainingFreezeTests


class FormalSegmentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="formal-segment-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        args = Ft0FormalTrainingFreezeTests().args()
        args.checkpoint_root = str(self.root / "checkpoints")
        args.adapter_root = str(self.root / "adapters")
        self.package = self.root / "ft0.json"
        self.package.write_text(json.dumps(builder.build(args), sort_keys=True) + "\n")
        self.checkpoints = self.root / "checkpoints" / args.run_name
        self.adapters = self.root / "adapters" / args.run_name

    def call(self, start: int = 0, end: int = 1000):
        attempt = self.root / "attempt"
        return contract.validate_segment(
            package_path=self.package, segment_start=start, segment_end=end,
            environment={"CUDA_VISIBLE_DEVICES": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
            attempt_dir=attempt,
            outputs={name: attempt / f"{name}.json" for name in ("progress", "loader_receipt", "rng_receipt", "composition_receipt", "result")},
            checkpoint_root=self.checkpoints, adapter_root=self.adapters,
        )

    def test_accepts_new_base_ft1_boundary_without_heavy_imports(self):
        receipt = self.call()
        self.assertEqual(receipt["resume_mode"], "fresh_pi0_base")
        self.assertFalse(receipt["execution_authorized"])

    def test_rejects_non_registered_boundary(self):
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            self.call(0, 999)

    def test_rejects_engineering_root_reuse(self):
        self.checkpoints.mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "fresh FT1"):
            self.call()

    def test_accepts_verified_later_resume_boundary(self):
        (self.checkpoints / "1000").mkdir(parents=True)
        self.adapters.mkdir(parents=True)
        (self.adapters / "step-00001000.verified.json").write_text("{}\n")
        receipt = self.call(1000, 5000)
        self.assertEqual(receipt["resume_mode"], "verified_prior_segment")

    def test_rejects_multi_gpu_environment(self):
        attempt = self.root / "attempt"
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            contract.validate_segment(
                package_path=self.package, segment_start=0, segment_end=1000,
                environment={"CUDA_VISIBLE_DEVICES": "0,1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
                attempt_dir=attempt,
                outputs={name: attempt / f"{name}.json" for name in ("progress", "loader_receipt", "rng_receipt", "composition_receipt", "result")},
                checkpoint_root=self.checkpoints, adapter_root=self.adapters,
            )


if __name__ == "__main__":
    unittest.main()
