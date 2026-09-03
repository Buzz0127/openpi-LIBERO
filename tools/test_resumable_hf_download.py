from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("resumable_hf_download.py")


def write_fake_downloader(path: Path, *, failures: int, message: str) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "state = pathlib.Path(__file__).with_suffix('.count')\n"
        "count = int(state.read_text()) if state.exists() else 0\n"
        "state.write_text(str(count + 1))\n"
        f"failures = {failures}\n"
        "if count < failures:\n"
        f"    print({message!r}, file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "print('complete')\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def run_wrapper(tmp_path: Path, downloader: Path, max_attempts: int = 5) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--downloader",
            str(downloader),
            "--repo-id",
            "physical-intelligence/libero",
            "--revision",
            "deadbeef",
            "--local-dir",
            str(tmp_path / "dataset"),
            "--max-attempts",
            str(max_attempts),
            "--initial-backoff-seconds",
            "0",
            "--max-backoff-seconds",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ResumableDownloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_retries_transient_failure_then_completes(self) -> None:
        downloader = self.tmp_path / "fake_downloader.py"
        write_fake_downloader(downloader, failures=2, message="ProxyError: RemoteDisconnected")
        result = run_wrapper(self.tmp_path, downloader)
        self.assertEqual(result.returncode, 0)
        recorded = events(self.tmp_path / "evidence" / "events.jsonl")
        self.assertEqual([item["event"] for item in recorded].count("attempt_failed"), 2)
        self.assertEqual(recorded[-1]["event"], "completed")
        self.assertEqual((self.tmp_path / "fake_downloader.count").read_text(), "3")

    def test_fatal_failure_does_not_retry(self) -> None:
        downloader = self.tmp_path / "fake_downloader.py"
        write_fake_downloader(downloader, failures=5, message="404 Client Error: Revision Not Found")
        result = run_wrapper(self.tmp_path, downloader)
        self.assertEqual(result.returncode, 1)
        recorded = events(self.tmp_path / "evidence" / "events.jsonl")
        failures = [item for item in recorded if item["event"] == "attempt_failed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["classification"], "fatal")
        self.assertEqual((self.tmp_path / "fake_downloader.count").read_text(), "1")

    def test_unknown_failure_does_not_retry(self) -> None:
        downloader = self.tmp_path / "fake_downloader.py"
        write_fake_downloader(downloader, failures=5, message="unexpected application error")
        result = run_wrapper(self.tmp_path, downloader)
        self.assertEqual(result.returncode, 1)
        recorded = events(self.tmp_path / "evidence" / "events.jsonl")
        self.assertEqual(recorded[-1]["classification"], "unknown")
        self.assertEqual((self.tmp_path / "fake_downloader.count").read_text(), "1")

    def test_refuses_existing_evidence_directory(self) -> None:
        downloader = self.tmp_path / "fake_downloader.py"
        write_fake_downloader(downloader, failures=0, message="unused")
        (self.tmp_path / "evidence").mkdir()
        result = run_wrapper(self.tmp_path, downloader)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("evidence directory already exists", result.stderr)
        self.assertFalse((self.tmp_path / "fake_downloader.count").exists())


if __name__ == "__main__":
    unittest.main()
