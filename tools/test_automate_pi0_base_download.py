from __future__ import annotations

import subprocess
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("automate_pi0_base_download.py")


class AutomationTest(unittest.TestCase):
    def test_rejects_missing_inputs_before_stage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--python", sys.executable,
                "--guard", str(Path(raw) / "missing"), "--downloader", str(Path(raw) / "missing2"),
                "--source-manifest", str(Path(raw) / "missing3"), "--evidence-root", raw,
                "--monitor-root", raw, "--existing-billed-bytes", "1", "--soft-limit-bytes", "2",
                "--hard-limit-bytes", "3", "--scratch", str(Path(raw) / "scratch"),
                "--final", str(Path(raw) / "final"), "--lock", str(Path(raw) / "lock"),
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
