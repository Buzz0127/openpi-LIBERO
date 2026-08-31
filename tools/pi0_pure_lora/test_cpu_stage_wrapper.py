from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cpu_stage_wrapper as wrapper


SCRIPT = Path(wrapper.__file__).resolve()


def _command(attempt: Path, *child: str, extra: list[str] | None = None) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--attempt-dir",
        str(attempt),
        "--timeout-seconds",
        "0.35",
        "--sample-interval-seconds",
        "0.05",
        "--term-grace-seconds",
        "0.15",
        "--kill-grace-seconds",
        "0.5",
    ]
    if extra:
        command.extend(extra)
    return command + ["--", *child]


def _status(attempt: Path) -> dict[str, object]:
    return json.loads((attempt / "exit_status.json").read_text(encoding="utf-8"))


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class CpuStageWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="g1a-wrapper-test-")
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_normal_completion_is_reaped(self) -> None:
        attempt = self.tmp_path / "normal"
        result = subprocess.run(
            _command(attempt, sys.executable, "-c", "print('bounded-child-ok')"),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        status = _status(attempt)
        self.assertEqual(status["reason_code"], "completed")
        self.assertIs(status["wait_reaped"], True)
        self.assertFalse(_pid_exists(int(status["child_pid"])))

    def test_timeout_forwards_term_and_reaps_cooperative_child(self) -> None:
        attempt = self.tmp_path / "term"
        child = "import signal,time,sys; signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); time.sleep(30)"
        result = subprocess.run(
            _command(attempt, sys.executable, "-c", child),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, wrapper.EXIT_TIMEOUT, result.stderr)
        status = _status(attempt)
        self.assertEqual(status["reason_code"], "timeout")
        self.assertIs(status["ownership_verified"], True)
        self.assertIs(status["term_sent"], True)
        self.assertIs(status["kill_sent"], False)
        self.assertIs(status["wait_reaped"], True)
        self.assertFalse(_pid_exists(int(status["child_pid"])))

    def test_timeout_escalates_to_kill_and_reaps_ignoring_child(self) -> None:
        attempt = self.tmp_path / "kill"
        child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        result = subprocess.run(
            _command(attempt, sys.executable, "-c", child),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, wrapper.EXIT_TIMEOUT, result.stderr)
        status = _status(attempt)
        self.assertEqual(status["reason_code"], "timeout")
        self.assertIs(status["term_sent"], True)
        self.assertIs(status["kill_sent"], True)
        self.assertIs(status["wait_reaped"], True)
        self.assertFalse(_pid_exists(int(status["child_pid"])))

    def test_external_term_is_forwarded_and_child_is_reaped(self) -> None:
        attempt = self.tmp_path / "external"
        child = "import signal,time,sys; signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); time.sleep(30)"
        process = subprocess.Popen(
            _command(attempt, sys.executable, "-c", child),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 2
        while not (attempt / "run_manifest.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue((attempt / "run_manifest.json").exists())
        os.kill(process.pid, signal.SIGTERM)
        _, stderr = process.communicate(timeout=3)
        self.assertEqual(process.returncode, wrapper.EXIT_GUARD_STOP, stderr)
        status = _status(attempt)
        self.assertEqual(status["reason_code"], "external_signal")
        self.assertEqual(status["external_signal"], signal.SIGTERM)
        self.assertIs(status["term_sent"], True)
        self.assertIs(status["wait_reaped"], True)
        self.assertFalse(_pid_exists(int(status["child_pid"])))

    def test_resource_and_monitor_failure_reason_codes(self) -> None:
        resource_attempt = self.tmp_path / "resource"
        resource_result = subprocess.run(
            _command(
                resource_attempt,
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
                extra=["--max-child-rss-bytes", "1", "--resource-consecutive-samples", "2"],
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(resource_result.returncode, wrapper.EXIT_GUARD_STOP, resource_result.stderr)
        self.assertEqual(_status(resource_attempt)["reason_code"], "resource_max_child_rss")

        monitor_attempt = self.tmp_path / "monitor"
        monitor_result = subprocess.run(
            _command(
                monitor_attempt,
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
                extra=[
                    "--simulate-monitor-failure-after",
                    "0",
                    "--monitor-failure-consecutive-samples",
                    "2",
                ],
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(monitor_result.returncode, wrapper.EXIT_GUARD_STOP, monitor_result.stderr)
        self.assertEqual(_status(monitor_attempt)["reason_code"], "monitor_failure")

    def test_unrelated_process_is_not_signaled(self) -> None:
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            attempt = self.tmp_path / "isolation"
            result = subprocess.run(
                _command(
                    attempt,
                    sys.executable,
                    "-c",
                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, wrapper.EXIT_TIMEOUT, result.stderr)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            try:
                unrelated.wait(timeout=1)
            except subprocess.TimeoutExpired:
                unrelated.kill()
                unrelated.wait(timeout=1)

    def test_existing_attempt_directory_is_rejected(self) -> None:
        attempt = self.tmp_path / "already-there"
        attempt.mkdir()
        result = subprocess.run(
            _command(attempt, sys.executable, "-c", "print('must-not-run')"),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileExistsError", result.stderr)


if __name__ == "__main__":
    unittest.main()
