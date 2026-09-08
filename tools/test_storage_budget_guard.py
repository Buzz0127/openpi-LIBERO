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

import storage_budget_guard as guard


SCRIPT = Path(guard.__file__).resolve()
FAKE_HOST_BOOTSTRAP = (
    f"import sys; sys.path.insert(0, {str(SCRIPT.parent)!r}); "
    "import storage_budget_guard as guard; "
    "guard._read_mem_available_bytes=lambda: 64*(1<<30); "
    "guard._read_child_rss_bytes=lambda pid: 1<<20; "
    "sys.exit(guard.main())"
)


def _command(attempt: Path, root: Path, *child: str, extra: list[str] | None = None) -> list[str]:
    command = [
        sys.executable,
        "-c",
        FAKE_HOST_BOOTSTRAP,
        "--attempt-dir",
        str(attempt),
        "--monitor-root",
        str(root),
        "--existing-billed-bytes",
        "0",
        "--soft-limit-bytes",
        "4096",
        "--hard-limit-bytes",
        "8192",
        "--timeout-seconds",
        "0.6",
        "--sample-interval-seconds",
        "0.03",
        "--near-sample-interval-seconds",
        "0.01",
        "--term-grace-seconds",
        "0.12",
        "--kill-grace-seconds",
        "0.5",
    ]
    if extra:
        command.extend(extra)
    return command + ["--", *child]


def _status(attempt: Path) -> dict[str, object]:
    return json.loads((attempt / "exit_status.json").read_text())


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class StorageBudgetGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="storage-guard-test-")
        self.tmp = Path(self.temporary_directory.name)
        self.root = self.tmp / "metered"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_normal_completion_and_reap(self) -> None:
        attempt = self.tmp / "normal"
        result = subprocess.run(
            _command(attempt, self.root, sys.executable, "-c", "print('ok')"), capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        status = _status(attempt)
        self.assertEqual(status["reason_code"], "completed")
        self.assertTrue(status["wait_reaped"])
        self.assertFalse(_pid_exists(int(status["child_pid"])))

    def test_storage_soft_limit_stops_owned_group(self) -> None:
        attempt = self.tmp / "storage"
        child = (
            "import os,time; "
            f"f=open({str(self.root / 'growth.bin')!r},'wb'); "
            "f.write(os.urandom(1048576)); f.flush(); os.fsync(f.fileno()); f.close(); time.sleep(30)"
        )
        result = subprocess.run(
            _command(attempt, self.root, sys.executable, "-c", child), capture_output=True, text=True
        )
        self.assertEqual(result.returncode, guard.EXIT_GUARD_STOP, result.stderr)
        status = _status(attempt)
        self.assertIn(status["reason_code"], ("storage_soft_limit", "storage_hard_limit"))
        self.assertTrue(status["ownership_verified"])
        self.assertTrue(status["term_sent"])
        self.assertTrue(status["wait_reaped"])

    def test_timeout_escalates_to_kill(self) -> None:
        attempt = self.tmp / "kill"
        child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        result = subprocess.run(
            _command(attempt, self.root, sys.executable, "-c", child), capture_output=True, text=True
        )
        self.assertEqual(result.returncode, guard.EXIT_TIMEOUT, result.stderr)
        status = _status(attempt)
        self.assertTrue(status["term_sent"])
        self.assertTrue(status["kill_sent"])
        self.assertTrue(status["wait_reaped"])

    def test_external_signal_forwards_and_reaps(self) -> None:
        attempt = self.tmp / "external"
        child = "import signal,time,sys; signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); time.sleep(30)"
        process = subprocess.Popen(_command(attempt, self.root, sys.executable, "-c", child))
        deadline = time.monotonic() + 2
        while not (attempt / "run_manifest.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(process.pid, signal.SIGHUP)
        process.wait(timeout=3)
        self.assertEqual(process.returncode, guard.EXIT_GUARD_STOP)
        status = _status(attempt)
        self.assertEqual(status["reason_code"], "external_signal")
        self.assertTrue(status["wait_reaped"])

    def test_monitor_failure_fails_closed(self) -> None:
        attempt = self.tmp / "monitor"
        result = subprocess.run(
            _command(
                attempt,
                self.root,
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
                extra=["--simulate-monitor-failure-after", "0", "--monitor-failure-consecutive-samples", "2"],
            ),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, guard.EXIT_GUARD_STOP, result.stderr)
        self.assertEqual(_status(attempt)["reason_code"], "monitor_failure")

    def test_manifest_write_failure_after_spawn_still_reaps_child(self) -> None:
        attempt = self.tmp / "manifest-failure"
        pid_file = self.tmp / "owned-pid"
        command = _command(attempt, self.root, sys.executable, "-c", "import time; time.sleep(30)")
        bootstrap = (
            f"import pathlib,sys; sys.path.insert(0, {str(SCRIPT.parent)!r})\n"
            "import storage_budget_guard as guard\n"
            "original=guard.subprocess.Popen\n"
            "def launch(*args,**kwargs):\n"
            " child=original(*args,**kwargs)\n"
            f" pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
            " return child\n"
            "def fail(*args,**kwargs): raise OSError('fake manifest disk failure')\n"
            "guard.subprocess.Popen=launch\n"
            "guard._atomic_json=fail\n"
            "sys.exit(guard.main())\n"
        )
        command[2] = bootstrap
        result = subprocess.run(command, capture_output=True, text=True, timeout=4)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fake manifest disk failure", result.stderr)
        self.assertFalse(_pid_exists(int(pid_file.read_text())))

    def test_post_reap_sample_catches_fast_final_write(self) -> None:
        attempt = self.tmp / "fast-final-write"
        gate = self.tmp / "first-periodic-sample"
        child = (
            "import pathlib,time; "
            f"gate=pathlib.Path({str(gate)!r}); "
            "exec('while not gate.exists(): time.sleep(0.005)'); "
            f"pathlib.Path({str(self.root / 'final.bin')!r}).write_bytes(b'x'*1048576)"
        )
        command = _command(attempt, self.root, sys.executable, "-c", child)
        command[2] = self._post_reap_bootstrap(gate, fail_final=False)
        for option in ("--sample-interval-seconds", "--near-sample-interval-seconds"):
            command[command.index(option) + 1] = "0.2"
        result = subprocess.run(command, capture_output=True, text=True, timeout=4)
        self.assertEqual(result.returncode, guard.EXIT_GUARD_STOP, result.stderr)
        status = _status(attempt)
        self.assertEqual(status["reason_code"], "storage_hard_limit")
        self.assertEqual(status["final_storage_sample"]["phase"], "post_reap")
        self.assertGreaterEqual(status["final_billed_lora_bytes"], 1048576)

    def test_post_reap_sampling_failure_fails_closed(self) -> None:
        attempt = self.tmp / "final-sample-failure"
        gate = self.tmp / "first-periodic-sample"
        child = (
            "import pathlib,time; "
            f"gate=pathlib.Path({str(gate)!r}); "
            "exec('while not gate.exists(): time.sleep(0.005)')"
        )
        command = _command(attempt, self.root, sys.executable, "-c", child)
        command[2] = self._post_reap_bootstrap(gate, fail_final=True)
        for option in ("--sample-interval-seconds", "--near-sample-interval-seconds"):
            command[command.index(option) + 1] = "0.2"
        result = subprocess.run(command, capture_output=True, text=True, timeout=4)
        self.assertEqual(result.returncode, guard.EXIT_GUARD_STOP, result.stderr)
        status = _status(attempt)
        self.assertEqual(status["reason_code"], "monitor_failure")
        self.assertIsNone(status["final_storage_sample"])
        self.assertIn("fake post-reap stat failure", status["final_storage_sample_error"])

    def _post_reap_bootstrap(self, gate: Path, *, fail_final: bool) -> str:
        return (
            f"import pathlib,sys; sys.path.insert(0,{str(SCRIPT.parent)!r})\n"
            "import storage_budget_guard as guard\n"
            "guard._read_mem_available_bytes=lambda:64*(1<<30)\n"
            "guard._read_child_rss_bytes=lambda pid:1<<20\n"
            "original=guard._storage_sample\n"
            "calls=0\n"
            "def sample(*args):\n"
            " global calls\n"
            " calls+=1\n"
            f" if calls>1 and {fail_final!r}: raise OSError('fake post-reap stat failure')\n"
            " result=original(*args)\n"
            f" pathlib.Path({str(gate)!r}).touch()\n"
            " return result\n"
            "guard._storage_sample=sample\n"
            "sys.exit(guard.main())\n"
        )

    def test_unrelated_process_is_not_signaled(self) -> None:
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            attempt = self.tmp / "isolation"
            result = subprocess.run(
                _command(attempt, self.root, sys.executable, "-c", "import time; time.sleep(30)"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, guard.EXIT_TIMEOUT, result.stderr)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=2)

    def test_overlapping_roots_rejected(self) -> None:
        child = self.root / "child"
        child.mkdir()
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--attempt-dir",
                str(self.tmp / "overlap"),
                "--monitor-root",
                str(self.root),
                "--monitor-root",
                str(child),
                "--existing-billed-bytes",
                "0",
                "--soft-limit-bytes",
                "4096",
                "--hard-limit-bytes",
                "8192",
                "--timeout-seconds",
                "1",
                "--",
                sys.executable,
                "-c",
                "print('must not run')",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overlap", result.stderr)


if __name__ == "__main__":
    unittest.main()
