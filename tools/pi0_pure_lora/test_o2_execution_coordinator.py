from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import experiment_identity
import o2_execution_coordinator as coordinator


class O2CoordinatorTest(unittest.TestCase):
    def test_rejects_changed_control_and_extracts_runtime_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); control = {"execution_authorized": False}; control["plan_identity_sha256"] = experiment_identity.canonical_sha256({"execution_authorized": False})
            path = root / "control.json"; path.write_text(json.dumps(control))
            self.assertFalse(coordinator.verify_control(path)["execution_authorized"])
            control["execution_authorized"] = True; path.write_text(json.dumps(control))
            with self.assertRaises(ValueError): coordinator.verify_control(path)
        arm = {"command": ["/python", "/server.py", "--openpi-root", "/openpi", "--runtime-mode", "merged-dense", "--port", "18043", "--rng-seed", "0"]}
        source, args = coordinator.runtime_args(arm, mode="merged-dense")
        self.assertEqual(source, Path("/server.py")); self.assertEqual(args["port"], 18043)
        self.assertEqual(args["runtime_mode"], "merged-dense")
        self.assertEqual(args["rng_seed"], 0)


if __name__ == "__main__":
    unittest.main()
