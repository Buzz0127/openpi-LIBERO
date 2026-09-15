from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import experiment_identity
import o2_execution_coordinator as coordinator


class O2CoordinatorTest(unittest.TestCase):
    @staticmethod
    def _exact_report(value: bool = True):
        return {
            "normalized_action_difference": {"exact_sha256_equal": value},
            "physical_action_difference": {"exact_sha256_equal": value},
        }

    def test_equivalence_gate_requires_both_model_and_physical_outputs(self):
        exact = self._exact_report()
        self.assertTrue(coordinator._is_exact_action_equivalence(exact))
        for field in ("normalized_action_difference", "physical_action_difference"):
            changed = {
                "normalized_action_difference": {"exact_sha256_equal": True},
                "physical_action_difference": {"exact_sha256_equal": True},
            }
            changed[field]["exact_sha256_equal"] = False
            self.assertFalse(coordinator._is_exact_action_equivalence(changed))

    def test_direct_equivalence_gate_rejects_before_timing_on_merge_mismatch(self):
        comparisons = {
            "merged-repeat": self._exact_report(),
            "unmerged-vs-merged": self._exact_report(False),
        }
        with self.assertRaisesRegex(RuntimeError, "unmerged-versus-merged"):
            coordinator.require_exact_direct_equivalence(comparisons)
        comparisons["unmerged-vs-merged"] = self._exact_report()
        coordinator.require_exact_direct_equivalence(comparisons)

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
