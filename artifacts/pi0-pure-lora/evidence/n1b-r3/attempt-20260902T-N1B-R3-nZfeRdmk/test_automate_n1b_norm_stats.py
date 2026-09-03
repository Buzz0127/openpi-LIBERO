from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("automate_n1b_norm_stats.py")
SPEC = importlib.util.spec_from_file_location("automate_n1b_norm_stats", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareStatsTest(unittest.TestCase):
    def test_manifest_environment_is_allowlisted(self):
        overrides = MODULE._offline_overrides()
        self.assertEqual(overrides["HF_HUB_OFFLINE"], "1")
        self.assertEqual(overrides["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("PATH", overrides)

    def test_equal_and_different_numeric_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "official.json"
            canonical = root / "canonical.json"
            official.write_text(json.dumps({"norm_stats": {"state": {"mean": [1.0, 2.0]}}}))
            canonical.write_text(json.dumps({"norm_stats": {"state": {"mean": [1.0, 3.0]}}}))
            report = MODULE.compare_stats(canonical, official, MODULE.sha256_file(official))
            self.assertFalse(report["all_numeric_equal"])
            self.assertEqual(report["max_abs_difference"], 1.0)
            self.assertEqual(report["canonical_numeric_leaf_count"], 2)

    def test_official_identity_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "official.json"
            canonical = root / "canonical.json"
            official.write_text("{}")
            canonical.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                MODULE.compare_stats(canonical, official, "0" * 64)


if __name__ == "__main__":
    unittest.main()
