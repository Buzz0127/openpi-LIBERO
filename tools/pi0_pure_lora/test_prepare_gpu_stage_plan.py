from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import experiment_identity


SCRIPT = Path(__file__).with_name("prepare_gpu_stage_plan.py")


class GpuStagePlanTest(unittest.TestCase):
    def test_requires_30_samples_and_pins_guard(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            preflight = {"schema_version": 1, "sample_count": 30, "samples": [{}] * 30, "selected_physical_gpu": 1, "selected_gpu_uuid": "GPU-b", "launch_gate": {}, "jax_preallocation_required": False, "guard_thresholds": {}}
            preflight["preflight_identity_sha256"] = experiment_identity.canonical_sha256(preflight)
            (root / "preflight.json").write_text(json.dumps(preflight))
            stage = {"plan_identity_sha256": "a" * 64, "execution_authorized": False}
            (root / "stage.json").write_text(json.dumps(stage))
            guard = root / "guard.py"
            guard.write_text("pass\n")
            output = root / "gpu.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--preflight-report", str(root / "preflight.json"), "--stage-plan", str(root / "stage.json"), "--guard", str(guard), "--expected-guard-sha256", experiment_identity.sha256_file(guard), "--python", sys.executable, "--max-runtime-seconds", "60", "--output", str(output), "--", "echo", "ok"], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(output.read_text())
            self.assertEqual(value["environment"]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertFalse(value["execution_authorized"])
            self.assertTrue(value["external_monitor_outside_child_process_group"])


if __name__ == "__main__":
    unittest.main()
