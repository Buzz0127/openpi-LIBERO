from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("diagnose_n1b_memory.py")
SPEC = importlib.util.spec_from_file_location("diagnose_n1b_memory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiagnosticHelpersTest(unittest.TestCase):
    def test_query_indices_clamp_at_episode_end(self):
        result = MODULE.query_index_matrix([8, 9], [10, 10], 4)
        self.assertEqual(result, [[8, 9, 9, 9], [9, 9, 9, 9]])

    def test_query_indices_reject_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            MODULE.query_index_matrix([1, 2], [3], 4)

    def test_slope(self):
        samples = [{"batch": 10, "rss_bytes": 100}, {"batch": 30, "rss_bytes": 300}]
        self.assertEqual(MODULE._slope(samples), 10.0)


if __name__ == "__main__":
    unittest.main()
