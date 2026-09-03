from __future__ import annotations

import automate_n1b_norm_stats as common
import unittest


class R3AutomationTest(unittest.TestCase):
    def test_manifest_environment_is_allowlisted(self):
        overrides = common._offline_overrides()
        self.assertEqual(overrides["HF_HUB_OFFLINE"], "1")
        self.assertEqual(overrides["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("PATH", overrides)


if __name__ == "__main__":
    unittest.main()
