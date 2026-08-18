import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

import gpu_utilization_guard as guard


class GpuUtilizationGuardTest(unittest.TestCase):
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
