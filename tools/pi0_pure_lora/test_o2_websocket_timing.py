from __future__ import annotations

import unittest

import numpy as np

import o2_websocket_timing as timing


class O2WebsocketTimingTest(unittest.TestCase):
    def test_summary_and_action_contract_fail_closed(self):
        self.assertEqual(timing._summary([1.0, 2.0, 3.0])["median"], 2.0)
        record = timing._action_record({"actions": np.zeros((50, 7), dtype=np.float32)})
        self.assertEqual(record["shape"], [50, 7])
        self.assertEqual(len(record["sha256"]), 64)
        with self.assertRaises(ValueError):
            timing._action_record({"actions": np.zeros((49, 7), dtype=np.float32)})
        with self.assertRaises(ValueError):
            timing._action_record({"actions": np.full((50, 7), np.nan, dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
