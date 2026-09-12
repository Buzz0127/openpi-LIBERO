from __future__ import annotations

import tempfile
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

import run_formal_stage_driver as driver


class FutureDriverTests(unittest.TestCase):
    def preflight(self, root: Path) -> tuple[Path, str]:
        path = root / "preflight.json"
        value = {"sample_count": 30, "samples": [{"gpus": [{"index": 0, "free_memory_percent": 99.0, "utilization_percent": 0.0}]}] * 30,
                 "selected_physical_gpu": 0, "collection_finished_epoch_seconds": 1000.0}
        import experiment_identity
        value["preflight_identity_sha256"] = experiment_identity.canonical_sha256(value)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_runs_bound_runner_then_generic_verifier(self) -> None:
        with tempfile.TemporaryDirectory(prefix="future-driver-") as directory:
            root = Path(directory); calls = []
            preflight, digest = self.preflight(root)
            (root / "progress.json").write_text('{"current_step":10000,"last_committed_step":10000}', encoding="utf-8")
            argv = ["driver", "--runner", str(root / "runner.py"), "--verifier", str(root / "verify.py"),
                    "--prior-acceptance", str(root / "prior.json"), "--prior-result", str(root / "prior-result.json"),
                    "--prior-identity-schema", "formal-canonical-v1", "--result", str(root / "result.json"),
                    "--acceptance-output", str(root / "acceptance.json"), "--progress", str(root / "progress.json"),
                    "--preflight-report", str(preflight), "--expected-preflight-sha256", digest, "--selected-physical-gpu", "0",
                    "--segment-start", "5000", "--segment-end", "10000", "--", "--run"]
            with mock.patch.object(driver.subprocess, "run", side_effect=lambda value, check: calls.append(value)):
                with mock.patch("sys.argv", argv), mock.patch.object(driver.time, "time", return_value=1050.0), mock.patch.object(driver.importlib.util, "find_spec", return_value=object()), mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES":"0", "XLA_PYTHON_CLIENT_PREALLOCATE":"false"}, clear=False):
                    self.assertEqual(driver.main(), 0)
            self.assertEqual(calls[0][1:], [str(root / "runner.py"), "--run"])
            self.assertIn("--prior-identity-schema", calls[1])
            self.assertIn("formal-canonical-v1", calls[1])

    def test_existing_acceptance_fails_before_runner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="future-driver-") as directory:
            root = Path(directory); output = root / "acceptance.json"; output.touch()
            preflight, digest = self.preflight(root)
            with mock.patch("sys.argv", ["driver", "--runner", str(root / "r.py"), "--verifier", str(root / "v.py"),
                "--prior-acceptance", str(root / "p.json"), "--prior-result", str(root / "pr.json"),
                "--prior-identity-schema", "ft2-newline-v1", "--result", str(root / "r.json"),
                "--acceptance-output", str(output), "--progress", str(root / "progress.json"),
                "--preflight-report", str(preflight), "--expected-preflight-sha256", digest, "--selected-physical-gpu", "0",
                "--segment-start", "5000", "--segment-end", "10000", "--", "--x"]):
                with self.assertRaisesRegex(ValueError, "new acceptance"):
                    driver.main()

    def test_rejects_stale_or_wrong_gpu_preflight_before_runner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="future-driver-") as directory:
            root = Path(directory); preflight, digest = self.preflight(root)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                with mock.patch.object(driver.time, "time", return_value=1121.0):
                    driver._validate_fresh_preflight(preflight, digest, 0)
            with self.assertRaisesRegex(RuntimeError, "differs"):
                with mock.patch.object(driver.time, "time", return_value=1050.0):
                    driver._validate_fresh_preflight(preflight, digest, 1)

    def test_rejects_missing_support_package_before_runner(self) -> None:
        with mock.patch.object(driver.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "package parent"):
                driver._validate_support_package_importable()
