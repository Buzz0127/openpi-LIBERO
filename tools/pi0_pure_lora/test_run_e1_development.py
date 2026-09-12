from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import run_e1_development as runner


class E1RunnerTest(unittest.TestCase):
    def test_registry_rejects_missing_or_unfrozen_steps(self):
        rows = [{"step": step, "adapter_identity": str(index) * 64} for index, step in enumerate(runner.EXPECTED_STEPS, 1)]
        self.assertEqual(len(runner._registration_candidates({"candidates": rows})), 7)
        with self.assertRaises(ValueError):
            runner._registration_candidates({"candidates": rows[:-1]})

    def test_result_row_marks_model_exception_as_infrastructure(self):
        candidate = {"step": 1000, "adapter_identity_sha256": "a" * 64}
        entry = {"suite": "libero_spatial", "task_id": 0, "initial_state_index": 1}
        result = {"success": False, "failure_reason": "exception"}
        # A real result path is deliberately not needed for this classification helper.
        self.assertEqual(runner._result_row(candidate, entry, result, Path(__file__))["outcome"], "infrastructure_failure")

    def test_base_identity_comes_from_manifest_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "base.json"
            expected = "b" * 64
            manifest.write_text(json.dumps({"identities": {"base_manifest_sha256": expected}}), encoding="utf-8")
            self.assertEqual(runner._base_identity_from_manifest(manifest), expected)


if __name__ == "__main__":
    unittest.main()
