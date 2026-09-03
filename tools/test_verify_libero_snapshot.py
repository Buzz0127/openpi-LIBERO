from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import verify_libero_snapshot as verifier


class VerifyLiberoSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="snapshot-verify-test-")
        self.tmp = Path(self.temporary_directory.name)
        self.root = self.tmp / "dataset"
        (self.root / "meta").mkdir(parents=True)
        self.info = b'{"ok":true}\n'
        (self.root / "meta/info.json").write_bytes(self.info)
        self.manifest = self.tmp / "manifest.jsonl"
        records = [
            {"path": "meta/info.json", "size": len(self.info)},
            {"path": "data/chunk-000/episode_000000.parquet", "size": 4},
        ]
        self.manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
        self.d1a = self.tmp / "d1a.json"
        self.d1a.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "requested_revision": "abc",
                    "resolved_revision": "abc",
                    "identity": {
                        "metadata": {
                            "meta/info.json": {
                                "sha256": hashlib.sha256(self.info).hexdigest(),
                            }
                        }
                    },
                }
            )
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_partial_accepts_exact_subset_and_incomplete_cache(self) -> None:
        incomplete = self.root / ".cache/huggingface/download/file.incomplete"
        incomplete.parent.mkdir(parents=True)
        incomplete.write_bytes(b"partial")
        report = verifier.verify(self.root, self.manifest, self.d1a, self.tmp / "partial.json", True)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["missing_count"], 1)
        self.assertEqual(report["cache_artifact_count"], 1)

    def test_complete_requires_all_files_and_no_partial(self) -> None:
        parquet = self.root / "data/chunk-000/episode_000000.parquet"
        parquet.parent.mkdir(parents=True)
        parquet.write_bytes(b"data")
        report = verifier.verify(self.root, self.manifest, self.d1a, self.tmp / "complete.json", False)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["actual_repo_file_count"], 2)

    def test_mismatch_fails(self) -> None:
        (self.root / "unexpected.txt").write_text("bad")
        report = verifier.verify(self.root, self.manifest, self.d1a, self.tmp / "bad.json", True)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["unexpected"], ["unexpected.txt"])

    def test_complete_accepts_zero_byte_lock_for_matching_complete_file(self) -> None:
        parquet = self.root / "data/chunk-000/episode_000000.parquet"
        parquet.parent.mkdir(parents=True)
        parquet.write_bytes(b"data")
        lock = self.root / ".cache/huggingface/download/data/chunk-000/episode_000000.parquet.lock"
        lock.parent.mkdir(parents=True)
        lock.write_bytes(b"")
        report = verifier.verify(self.root, self.manifest, self.d1a, self.tmp / "lock-ok.json", False)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["cache_artifact_count"], 0)
        self.assertEqual(report["completed_zero_byte_lock_count"], 1)

    def test_complete_rejects_nonempty_lock(self) -> None:
        parquet = self.root / "data/chunk-000/episode_000000.parquet"
        parquet.parent.mkdir(parents=True)
        parquet.write_bytes(b"data")
        lock = self.root / ".cache/huggingface/download/data/chunk-000/episode_000000.parquet.lock"
        lock.parent.mkdir(parents=True)
        lock.write_bytes(b"active")
        report = verifier.verify(self.root, self.manifest, self.d1a, self.tmp / "lock-bad.json", False)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["cache_artifact_count"], 1)

    def test_complete_rejects_zero_byte_lock_without_matching_file(self) -> None:
        lock = self.root / ".cache/huggingface/download/data/chunk-000/episode_000000.parquet.lock"
        lock.parent.mkdir(parents=True)
        lock.write_bytes(b"")
        report = verifier.verify(self.root, self.manifest, self.d1a, self.tmp / "orphan-lock.json", False)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["cache_artifact_count"], 1)

    def test_complete_rejects_incomplete_and_temp_files(self) -> None:
        parquet = self.root / "data/chunk-000/episode_000000.parquet"
        parquet.parent.mkdir(parents=True)
        parquet.write_bytes(b"data")
        cache = self.root / ".cache/huggingface/download"
        cache.mkdir(parents=True)
        (cache / "file.incomplete").write_bytes(b"partial")
        (cache / "file.tmp").write_bytes(b"partial")
        report = verifier.verify(self.root, self.manifest, self.d1a, self.tmp / "temp-bad.json", False)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["cache_artifact_count"], 2)


if __name__ == "__main__":
    unittest.main()
