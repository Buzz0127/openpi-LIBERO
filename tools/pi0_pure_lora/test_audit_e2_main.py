from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import audit_e2_main as audit


class AuditE2MainTest(unittest.TestCase):
    def test_collect_rejects_zero_request_infrastructure_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "one_result.json"
            result.write_text(json.dumps({"suite": "s", "task_id": 0, "initial_state_index": 1, "success": False, "policy_requests": 0}))
            with self.assertRaises(ValueError):
                audit._collect(root, [{"suite": "s", "task_id": 0, "initial_state_index": 1}], "base")

    def test_pair_outcomes_are_exhaustive(self):
        rows = [
            {"base_success": False, "pure_lora_success": False},
            {"base_success": False, "pure_lora_success": True},
            {"base_success": True, "pure_lora_success": False},
            {"base_success": True, "pure_lora_success": True},
        ]
        self.assertEqual(audit._summary(rows, "base_success"), {"successes": 2, "denominator": 4, "success_rate_percent": 50.0})
        self.assertEqual(audit._summary(rows, "pure_lora_success"), {"successes": 2, "denominator": 4, "success_rate_percent": 50.0})


if __name__ == "__main__":
    unittest.main()
