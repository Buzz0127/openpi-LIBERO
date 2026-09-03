from __future__ import annotations

import unittest


class R3PlanTest(unittest.TestCase):
    def test_openpi_full_run_batch_plan(self):
        self.assertEqual(273_465 // 32, 8_545)
        self.assertEqual((273_465 // 32) * 32, 273_440)
        self.assertEqual(273_465 % 32, 25)


if __name__ == "__main__":
    unittest.main()
