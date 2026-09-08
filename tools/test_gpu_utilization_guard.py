import contextlib
import io
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import gpu_utilization_guard as guard


class GpuUtilizationGuardTest(unittest.TestCase):
    def test_log_failure_after_spawn_still_cleans_child(self) -> None:
        children = []
        original_popen = subprocess.Popen

        def launch(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            children.append(child)
            return child

        def emit(_logger, event, **_fields):
            if event == "child_started":
                raise OSError("fake evidence disk failure")

        args = guard.parse_args([
            "--physical-gpu", "0", "--terminate-grace-seconds", "0.1",
            "--kill-grace-seconds", "0.5", "--", sys.executable,
            "-c", "import time; time.sleep(30)",
        ])
        with mock.patch.object(guard, "query_gpu_status", return_value=guard.GpuStatus(1, 10, 100)), \
             mock.patch.object(guard.subprocess, "Popen", side_effect=launch), \
             mock.patch.object(guard.EventLogger, "emit", emit):
            with self.assertRaisesRegex(OSError, "fake evidence disk"):
                guard.run_guarded(args)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertFalse(guard.signal_child_group(children[0], 0))

    def test_signal_during_prelaunch_prevents_spawn(self) -> None:
        def sample(_gpu):
            os.kill(os.getpid(), signal.SIGTERM)
            return guard.GpuStatus(1, 10, 100)

        args = guard.parse_args(["--physical-gpu", "0", "--", sys.executable, "-c", "pass"])
        with mock.patch.object(guard, "query_gpu_status", side_effect=sample), \
             mock.patch.object(guard.subprocess, "Popen") as spawn, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(guard.run_guarded(args), 143)
        spawn.assert_not_called()

    def test_cleanup_kills_worker_after_group_leader_exits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ready = pathlib.Path(directory) / "ready"
            worker = (
                "import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
            )
            leader = (
                "import pathlib,subprocess,sys,time; "
                f"subprocess.Popen([sys.executable,'-c',{worker!r}]); "
                f"ready=pathlib.Path({str(ready)!r}); "
                "exec('while not ready.exists(): time.sleep(0.01)')"
            )
            child = subprocess.Popen([sys.executable, "-c", leader], start_new_session=True)
            try:
                child.wait(timeout=3)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = guard.terminate_child(child, False, 0.1, guard.EventLogger(None), 1.0)
                self.assertTrue(result["term_sent"])
                self.assertTrue(result["kill_sent"])
                self.assertTrue(result["wait_reaped"])
                self.assertTrue(result["group_exit_confirmed"])
            finally:
                guard.signal_child_group(child, signal.SIGKILL)
                child.wait(timeout=2)

    def test_paused_runtime_timeout_resumes_then_kills_with_bounded_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ready = pathlib.Path(directory) / "ready"
            samples = 0

            def sample(_gpu):
                nonlocal samples
                samples += 1
                if samples == 1:
                    return guard.GpuStatus(1, 10, 100)
                deadline = time.monotonic() + 2
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                return guard.GpuStatus(99, 10, 100)

            args = guard.parse_args([
                "--physical-gpu", "0", "--interval-seconds", "0.01",
                "--max-runtime-seconds", "0.2", "--terminate-grace-seconds", "0.1",
                "--kill-grace-seconds", "0.5", "--", sys.executable, "-c",
                "import pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(30)",
            ])
            output = io.StringIO()
            start = time.monotonic()
            with mock.patch.object(guard, "query_gpu_status", side_effect=sample), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(guard.run_guarded(args), 124)
            self.assertLess(time.monotonic() - start, 2)
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertTrue(any(record.get("action") == "paused" for record in records))
            cleanup = next(record for record in records if record["event"] == "child_cleanup")
            self.assertTrue(cleanup["kill_sent"])
            self.assertTrue(cleanup["group_exit_confirmed"])

    def test_pauses_at_95_and_resumes_after_five_safe_samples(self) -> None:
        samples = (
            [guard.GpuStatus(10.0, 50.0, 100.0), guard.GpuStatus(95.0, 50.0, 100.0)]
            + [guard.GpuStatus(85.0, 75.0, 100.0)] * 5
            + [guard.GpuStatus(10.0, 50.0, 100.0)] * 100
        )

        def fake_query(_physical_gpu: int) -> guard.GpuStatus:
            return samples.pop(0) if samples else guard.GpuStatus(10.0, 50.0, 100.0)

        original_query = guard.query_gpu_status
        guard.query_gpu_status = fake_query
        try:
            with tempfile.TemporaryDirectory() as directory:
                log_path = pathlib.Path(directory) / "guard.jsonl"
                args = guard.parse_args(
                    [
                        "--physical-gpu",
                        "0",
                        "--interval-seconds",
                        "0.01",
                        "--max-runtime-seconds",
                        "2",
                        "--log",
                        str(log_path),
                        "--",
                        sys.executable,
                        "-c",
                        "import time; time.sleep(0.25)",
                    ]
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    return_code = guard.run_guarded(args)
                records = [json.loads(line) for line in log_path.read_text().splitlines()]
        finally:
            guard.query_gpu_status = original_query

        events = [record["event"] for record in records]
        self.assertEqual(return_code, 0)
        self.assertIn("child_started", events)
        self.assertIn("gpu_sample", events)
        actions = [
            record["action"]
            for record in records
            if record.get("action") in ("paused", "resumed")
        ]
        self.assertEqual(actions, ["paused", "resumed"])

    def test_pauses_at_15_percent_free_memory(self) -> None:
        samples = (
            [guard.GpuStatus(10.0, 50.0, 100.0), guard.GpuStatus(10.0, 85.0, 100.0)]
            + [guard.GpuStatus(10.0, 80.0, 100.0)] * 5
            + [guard.GpuStatus(10.0, 50.0, 100.0)] * 100
        )

        def fake_query(_physical_gpu: int) -> guard.GpuStatus:
            return samples.pop(0) if samples else guard.GpuStatus(10.0, 50.0, 100.0)

        original_query = guard.query_gpu_status
        guard.query_gpu_status = fake_query
        try:
            with tempfile.TemporaryDirectory() as directory:
                log_path = pathlib.Path(directory) / "guard.jsonl"
                args = guard.parse_args(
                    [
                        "--physical-gpu",
                        "0",
                        "--interval-seconds",
                        "0.01",
                        "--max-runtime-seconds",
                        "2",
                        "--log",
                        str(log_path),
                        "--",
                        sys.executable,
                        "-c",
                        "import time; time.sleep(0.25)",
                    ]
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    return_code = guard.run_guarded(args)
                records = [json.loads(line) for line in log_path.read_text().splitlines()]
        finally:
            guard.query_gpu_status = original_query

        self.assertEqual(return_code, 0)
        actions = [
            record["action"]
            for record in records
            if record.get("action") in ("paused", "resumed")
        ]
        self.assertEqual(actions, ["paused", "resumed"])

if __name__ == "__main__":
    unittest.main()
