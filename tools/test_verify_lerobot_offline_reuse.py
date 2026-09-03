from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import verify_lerobot_offline_reuse as verifier


REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"


class _Meta:
    total_episodes = 1693
    total_frames = 273465
    total_tasks = 40
    fps = 10


class _Dataset:
    def __init__(self, root: Path) -> None:
        self.repo_id = verifier.REPO_ID
        self.revision = REVISION
        self.root = root
        self.meta = _Meta()

    def __len__(self) -> int:
        return 273465


class VerifyLeRobotOfflineReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="offline-reuse-test-")
        self.tmp = Path(self.temporary_directory.name)
        self.hf_home = self.tmp / "huggingface"
        self.lerobot_home = self.hf_home / "lerobot"
        self.datasets_cache = self.hf_home / "datasets"
        self.dataset_root = self.lerobot_home / "physical-intelligence/libero"
        (self.dataset_root / "meta").mkdir(parents=True)
        (self.dataset_root / "meta/info.json").write_text("{}\n")
        self.arrow_root = self.datasets_cache / "parquet/default-test/0.0.0/fingerprint"
        self.arrow_root.mkdir(parents=True)
        (self.arrow_root / "dataset_info.json").write_text("{}\n")
        (self.arrow_root / "train-00000-of-00002.arrow").write_bytes(b"arrow-0")
        (self.arrow_root / "train-00001-of-00002.arrow").write_bytes(b"arrow-1")
        self.raw_report = self.tmp / "raw-report.json"
        self.raw_report.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "mode": "complete",
                    "dataset_root": str(self.dataset_root.resolve()),
                    "revision": REVISION,
                    "expected_file_count": 1,
                    "actual_repo_file_count": 1,
                    "expected_total_bytes": 3,
                    "actual_repo_bytes": 3,
                    "missing_count": 0,
                    "unexpected": [],
                    "size_mismatches": [],
                    "symlinks": [],
                    "cache_artifact_count": 0,
                    "metadata": {"meta/info.json": {"matches": True}},
                }
            )
        )
        self.environment = {
            "HF_HOME": str(self.hf_home),
            "HF_DATASETS_CACHE": str(self.datasets_cache),
            "HF_LEROBOT_HOME": str(self.lerobot_home),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _verify(self, factory=None, output_name: str = "report.json") -> dict[str, object]:
        arrow = verifier._arrow_cache_summary(self.datasets_cache, self.arrow_root)

        def stable_factory(repo_id: str, **kwargs):
            self.assertEqual(repo_id, verifier.REPO_ID)
            self.assertEqual(kwargs["root"], self.dataset_root.resolve())
            self.assertEqual(kwargs["revision"], REVISION)
            self.assertFalse(kwargs["force_cache_sync"])
            self.assertFalse(kwargs["download_videos"])
            return _Dataset(self.dataset_root)

        return verifier.verify(
            dataset_root=self.dataset_root,
            hf_home=self.hf_home,
            hf_datasets_cache=self.datasets_cache,
            hf_lerobot_home=self.lerobot_home,
            raw_snapshot_report=self.raw_report,
            revision=REVISION,
            expected_arrow_root=self.arrow_root,
            expected_arrow_builder="parquet",
            expected_arrow_config="default-test",
            expected_arrow_version="0.0.0",
            expected_arrow_fingerprint="fingerprint",
            expected_arrow_file_count=2,
            expected_arrow_manifest_sha256=arrow["path_size_manifest_sha256"],
            max_new_file_bytes=100_000_000,
            output=self.tmp / output_name,
            dataset_factory=factory or stable_factory,
            environment=self.environment,
        )

    def test_stable_offline_reuse_passes(self) -> None:
        report = self._verify()
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["phase"], "runtime")
        self.assertEqual(report["hf_home_diff"]["added"], [])
        self.assertTrue(all(report["checks"].values()))

    def test_full_hf_home_detects_second_raw_snapshot(self) -> None:
        def duplicate_raw_factory(_repo_id: str, **_kwargs):
            duplicate = self.lerobot_home / "duplicate/libero/meta"
            duplicate.mkdir(parents=True)
            (duplicate / "info.json").write_text("{}\n")
            return _Dataset(self.dataset_root)

        report = self._verify(duplicate_raw_factory)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["checks"]["single_raw_repo_root_after"])
        self.assertFalse(report["checks"]["no_added_hf_home_entries"])

    def test_full_hf_home_detects_second_arrow_tree_and_large_file(self) -> None:
        def duplicate_arrow_factory(_repo_id: str, **_kwargs):
            duplicate = self.datasets_cache / "parquet/other/0.0.0/other-fingerprint/data.arrow"
            duplicate.parent.mkdir(parents=True)
            with duplicate.open("wb") as stream:
                stream.truncate(100_000_001)
            return _Dataset(self.dataset_root)

        report = self._verify(duplicate_arrow_factory)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["checks"]["single_arrow_tree_after"])
        self.assertFalse(report["checks"]["no_new_file_over_limit"])
        self.assertEqual(report["hf_home_diff"]["max_added_file_bytes"], 100_000_001)

    def test_existing_zero_byte_dataset_lock_mtime_touch_is_allowed(self) -> None:
        lock = self.datasets_cache / "builder.lock"
        lock.write_bytes(b"")

        def touch_lock_factory(_repo_id: str, **_kwargs):
            changed_ns = lock.stat().st_mtime_ns + 1_000_000_000
            os.utime(lock, ns=(changed_ns, changed_ns))
            return _Dataset(self.dataset_root)

        report = self._verify(touch_lock_factory)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            report["hf_home_diff"]["allowed_zero_byte_lock_mtime_touches"],
            ["datasets/builder.lock"],
        )
        self.assertEqual(report["hf_home_diff"]["unexpected_changes"], [])

    def test_non_lock_metadata_change_still_fails(self) -> None:
        data = self.arrow_root / "dataset_info.json"

        def touch_data_factory(_repo_id: str, **_kwargs):
            changed_ns = data.stat().st_mtime_ns + 1_000_000_000
            os.utime(data, ns=(changed_ns, changed_ns))
            return _Dataset(self.dataset_root)

        report = self._verify(touch_data_factory)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["checks"]["no_unexpected_hf_home_changes"])
        self.assertEqual(
            report["hf_home_diff"]["unexpected_changes"],
            [
                "datasets/parquet/default-test/0.0.0/fingerprint/dataset_info.json",
            ],
        )

    def test_preflight_failure_does_not_call_loader(self) -> None:
        called = False

        def forbidden_factory(_repo_id: str, **_kwargs):
            nonlocal called
            called = True
            return _Dataset(self.dataset_root)

        self.environment["HTTPS_PROXY"] = "http://127.0.0.1:7890"
        report = self._verify(forbidden_factory)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["phase"], "preflight")
        self.assertFalse(report["checks"]["env_all_proxy_variables_unset"])
        self.assertFalse(called)

    def test_arrow_identity_is_explicit(self) -> None:
        summary = verifier._arrow_cache_summary(self.datasets_cache, self.arrow_root)
        self.assertEqual(summary["builder"], "parquet")
        self.assertEqual(summary["config"], "default-test")
        self.assertEqual(summary["version"], "0.0.0")
        self.assertEqual(summary["fingerprint"], "fingerprint")
        self.assertEqual(summary["arrow_file_count"], 2)
        self.assertEqual(len(summary["path_size_manifest_sha256"]), 64)

    def test_tree_scan_records_symlinks_without_following(self) -> None:
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "secret").write_text("x")
        link = self.hf_home / "linked-dir"
        link.symlink_to(outside, target_is_directory=True)
        snapshot = verifier._scan_tree(self.hf_home)
        self.assertEqual(snapshot["files"]["linked-dir"]["kind"], "symlink")
        self.assertEqual(snapshot["summary"]["symlink_count"], 1)
        self.assertNotIn("linked-dir/secret", snapshot["files"])

    def test_preflight_rejects_hardlinked_arrow_file(self) -> None:
        os.link(
            self.arrow_root / "train-00000-of-00002.arrow",
            self.arrow_root / "hardlinked.arrow",
        )
        report = self._verify()
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["phase"], "preflight")
        self.assertFalse(report["checks"]["arrow_has_no_hardlinks_before"])

    def test_output_is_never_overwritten(self) -> None:
        output = self.tmp / "immutable.json"
        output.write_text("keep")
        with self.assertRaises(FileExistsError):
            self._verify(output_name="immutable.json")
        self.assertEqual(output.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
