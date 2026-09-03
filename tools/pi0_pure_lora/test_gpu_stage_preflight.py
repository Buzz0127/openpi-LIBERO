from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("gpu_stage_preflight.py")


class GpuPreflightTest(unittest.TestCase):
    def test_selects_healthier_gpu_after_bounded_samples(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fake = root / "nvidia-smi"
            fake.write_text("#!/bin/sh\nprintf '0, GPU-a, 30, 8000, 10000\\n1, GPU-b, 20, 2000, 10000\\n'\n")
            fake.chmod(0o755)
            proc = root / "proc"
            proc.mkdir()
            (proc / "meminfo").write_text("MemAvailable: 200000000 kB\n")
            (proc / "loadavg").write_text("1.0 1.0 1.0 1/1 1\n")
            (proc / "cpuinfo").write_text("processor : 0\nprocessor : 1\n")
            report = root / "report.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--nvidia-smi", str(fake), "--samples", "3", "--interval-seconds", "0.01", "--test-allow-short", "--proc-root", str(proc), "--output", str(report)], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(report.read_text())
            self.assertEqual(value["selected_physical_gpu"], 1)
            self.assertEqual(value["sample_count"], 3)
            self.assertEqual(value["guard_thresholds"]["resume_consecutive_samples"], 5)

    def test_refuses_short_production_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = subprocess.run([sys.executable, str(SCRIPT), "--samples", "2", "--output", str(Path(raw) / "x")], text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("at least 30", result.stderr)


if __name__ == "__main__":
    unittest.main()
