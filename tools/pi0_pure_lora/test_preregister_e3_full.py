from __future__ import annotations

import unittest

import preregister_e3_full as e3


class E3PreregistrationTest(unittest.TestCase):
    def test_full_manifest_contains_all_frozen_e0_states(self):
        e0 = {"protocol": {"suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"], "tasks_per_suite": 10, "evaluation_seed": 7}, "selection": {"available_initial_states_per_task": 50}, "entries": [{"split": "main", "suite": "libero_goal", "task_id": 3, "initial_state_index": 17}], "manifest_identity_sha256": "a" * 64}
        manifest = e3.build(e0, "b" * 64)
        self.assertEqual(len(manifest["entries"]), 2000)
        self.assertIn({"split": "full", "suite": "libero_goal", "task_id": 3, "initial_state_index": 17}, manifest["entries"])

    def test_rejects_drifted_protocol(self):
        e0 = {"protocol": {"suites": ["libero_spatial"], "tasks_per_suite": 10, "evaluation_seed": 7}, "selection": {"available_initial_states_per_task": 50}, "entries": [], "manifest_identity_sha256": "a" * 64}
        with self.assertRaises(ValueError):
            e3.build(e0, "b" * 64)


if __name__ == "__main__":
    unittest.main()
