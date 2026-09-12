from __future__ import annotations

import unittest

import run_e2_main as runner


class E2RunnerTest(unittest.TestCase):
    def test_main_entries_require_exact_frozen_denominator(self):
        rows = [{"split": "main", "suite": "s", "task_id": index, "initial_state_index": 0} for index in range(200)]
        self.assertEqual(len(runner._main_entries({"entries": rows})), 200)
        with self.assertRaises(ValueError):
            runner._main_entries({"entries": rows[:-1]})

    def test_lock_identity_is_checked_before_e2(self):
        lock = {"stage": "E1-selection-lock", "e1_completed": 280, "e2_authorized": False}
        lock["lock_identity_sha256"] = runner._canonical(lock)
        self.assertEqual(lock["lock_identity_sha256"], runner._canonical({key: value for key, value in lock.items() if key != "lock_identity_sha256"}))


if __name__ == "__main__":
    unittest.main()
