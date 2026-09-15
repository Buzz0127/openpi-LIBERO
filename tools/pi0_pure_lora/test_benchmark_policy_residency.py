from __future__ import annotations

import unittest

import numpy as np

import benchmark_policy_residency as benchmark


class BenchmarkPolicyResidencyTest(unittest.TestCase):
    def test_timing_summary_is_stable_and_bounded(self):
        summary = benchmark._summary([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["median"], 3.0)
        self.assertEqual(summary["p95"], 5.0)

    def test_action_hash_rejects_nonfinite_and_wrong_shape(self):
        valid = {"actions": np.ones((50, 7), dtype=np.float32)}
        self.assertEqual(len(benchmark._action_hash(benchmark._action_array(valid))), 64)
        with self.assertRaises(ValueError):
            benchmark._action_array({"actions": np.ones((7,), dtype=np.float32)})
        with self.assertRaises(ValueError):
            benchmark._action_array({"actions": np.full((50, 7), np.nan, dtype=np.float32)})

    def test_action_equivalence_reports_compact_numeric_difference(self):
        host = [np.zeros((2, 7), dtype=np.float32), np.ones((2, 7), dtype=np.float32)]
        device = [np.zeros((2, 7), dtype=np.float32), np.full((2, 7), 1.25, dtype=np.float32)]
        result = benchmark._action_equivalence(host, device)
        self.assertFalse(result["exact_hashes_equal"])
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["first_mismatch"], 1)
        self.assertEqual(result["max_abs_difference"], 0.25)
