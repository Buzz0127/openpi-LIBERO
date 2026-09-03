from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import reconcile_s1b_cache_evidence as reconciliation


class ReconcileS1bCacheEvidenceTest(unittest.TestCase):
    def test_matching_single_roots_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            hf = root / "hf"
            raw_root = hf / "lerobot/org/repo"
            arrow_root = hf / "datasets/parquet/default/0.0.0/fingerprint"
            (raw_root / "meta").mkdir(parents=True)
            (raw_root / "meta/info.json").write_text("{}")
            arrow_root.mkdir(parents=True)
            (arrow_root / "dataset_info.json").write_text("{}")
            (arrow_root / "data.arrow").write_bytes(b"arrow")
            static = root / "static"
            static.mkdir()
            (static / "raw-roots.before.txt").write_text(f"{raw_root.resolve()}\n")
            (static / "arrow-roots.before.txt").write_text(f"{arrow_root.resolve()}\n")
            (static / "raw-roots.after.txt").write_text("")
            (static / "arrow-roots.after.txt").write_text("")
            report_path = root / "d1c.json"
            with mock.patch.object(reconciliation, "_du_bytes", return_value=123):
                file_count = sum(1 for entry in hf.rglob("*") if entry.is_file())
                (static / "hf.du.before.txt").write_text("123\t/hf\n")
                (static / "hf.file-count.before.txt").write_text(f"{file_count}\n")
                report_path.write_text(
                    json.dumps(
                        {
                            "raw_repo_roots_after": [str(raw_root.resolve())],
                            "arrow_tree_roots_after": [str(arrow_root.resolve())],
                            "hf_home_after": {"apparent_bytes": 123, "file_entry_count": file_count},
                            "raw_snapshot_identity": {"actual_repo_file_count": 1, "actual_repo_bytes": 2},
                        }
                    )
                )
                report = reconciliation.reconcile(hf, report_path, static)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
