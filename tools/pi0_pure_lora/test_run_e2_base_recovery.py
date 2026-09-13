from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import run_e2_base_recovery as recovery
import run_e2_main as e2


class E2BaseRecoveryTest(unittest.TestCase):
    def test_parent_requires_the_known_zero_request_invalid_base_half(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir()
            (root / "summary.json").write_text(json.dumps({"stage": "E2-main-400", "status": "pass", "completed": 400}))
            for index in range(200):
                path = root / "base" / f"result_{index:03d}_result.json"
                path.write_text(json.dumps({"success": False, "policy_requests": 0}))
            parent = recovery._validate_parent(root)
            self.assertEqual(parent["prior_base_result_count"], 200)
            self.assertEqual(parent["prior_base_policy_request_count"], 0)
            (root / "base" / "result_000_result.json").write_text(json.dumps({"success": True, "policy_requests": 1}))
            with self.assertRaises(ValueError):
                recovery._validate_parent(root)

    def test_lock_validation_retains_the_selected_adapter_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = {"stage": "E1-selection-lock", "e1_completed": 280, "e2_authorized": False, "adapter_identity_sha256": "a" * 64}
            lock["lock_identity_sha256"] = e2._canonical(lock)
            lock_path = root / "lock.json"
            selected_path = root / "selected.json"
            lock_path.write_text(json.dumps(lock))
            selected_path.write_text(json.dumps({"model_mode": "base_plus_adapter", "adapter_identity_sha256": "a" * 64}))
            self.assertEqual(recovery._validate_lock(lock_path, selected_path)["lock_identity_sha256"], lock["lock_identity_sha256"])
            selected_path.write_text(json.dumps({"model_mode": "base", "adapter_identity_sha256": "a" * 64}))
            with self.assertRaises(ValueError):
                recovery._validate_lock(lock_path, selected_path)


if __name__ == "__main__":
    unittest.main()
