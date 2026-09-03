from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import experiment_identity


SCRIPT = Path(__file__).with_name("prepare_segment_plan.py")
IDS = {key: char * 64 for key, char in zip(sorted(experiment_identity.IDENTITY_KEYS), "1234")}


class SegmentPlanTest(unittest.TestCase):
    def run_plan(self, root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        openpi = root / "openpi"
        (openpi / "scripts").mkdir(parents=True)
        (openpi / "scripts/train.py").write_text("")
        support = root / "support"
        (support / "pi0_pure_lora").mkdir(parents=True)
        (support / "pi0_pure_lora/adapter_artifact.py").write_text("")
        manifest = experiment_identity.build_model_manifest(model_mode="base", openpi_commit="a" * 64, identities=IDS, adapter_identity_sha256=None, training_seed=None, artifact_purpose="training_init")
        manifest_path = root / "model.json"
        manifest_path.write_text(json.dumps(manifest))
        command = [sys.executable, str(SCRIPT), "--model-manifest", str(manifest_path), "--openpi-root", str(openpi), "--python", sys.executable, "--support-tools-root", str(support), "--allowed-run-root", str(root / "runs"), "--checkpoint-dir", str(root / "runs/checkpoints/x"), "--adapter-root", str(root / "runs/adapters"), "--exp-name", "x", "--segment-start", "0", "--segment-end", "100", "--train-seed", "42", "--eval-seed", "7", "--dataset-revision", "r", "--dataset-manifest-sha256", "d" * 64, "--output", str(root / "plan.json"), *extra]
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_plan_is_non_executing_and_identity_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.run_plan(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((root / "plan.json").read_text())
            self.assertFalse(plan["execution_authorized"])
            self.assertFalse(plan["rotation_policy"]["automatic_pruning_allowed"])

    def test_run_root_escape_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.run_plan(root, "--checkpoint-dir", str(root / "outside"))
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
