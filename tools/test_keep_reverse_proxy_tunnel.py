from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("keep_reverse_proxy_tunnel.py")


class TunnelSupervisorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_retries_failed_fake_ssh_until_timeout(self) -> None:
        fake = self.tmp_path / "fake_ssh.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib\n"
            "state = pathlib.Path(__file__).with_suffix('.count')\n"
            "count = int(state.read_text()) if state.exists() else 0\n"
            "state.write_text(str(count + 1))\n"
            "raise SystemExit(255)\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--ssh",
                str(fake),
                "--evidence-dir",
                str(self.tmp_path / "evidence"),
                "--overall-timeout-seconds",
                "1.0",
                "--initial-backoff-seconds",
                "0",
                "--max-backoff-seconds",
                "0",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        self.assertEqual(result.returncode, 124)
        self.assertGreater(int((self.tmp_path / "fake_ssh.count").read_text()), 1)
        recorded = [
            json.loads(line)
            for line in (self.tmp_path / "evidence" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(recorded[-1]["event"], "overall_timeout")

    def test_refuses_existing_evidence_directory(self) -> None:
        fake = self.tmp_path / "fake_ssh.py"
        fake.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        fake.chmod(0o755)
        evidence = self.tmp_path / "evidence"
        evidence.mkdir()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--ssh", str(fake), "--evidence-dir", str(evidence)],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("evidence directory already exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
