from __future__ import annotations

import unittest

import run_e3_full as e3


class E3RunnerTest(unittest.TestCase):
    def test_entries_require_exact_full_denominator(self):
        rows = [{"split": "full", "suite": "s", "task_id": task, "initial_state_index": state} for task in range(40) for state in range(50)]
        self.assertEqual(len(e3._entries({"entries": rows})), 2000)
        with self.assertRaises(ValueError):
            e3._entries({"entries": rows[:-1]})


if __name__ == "__main__":
    unittest.main()
