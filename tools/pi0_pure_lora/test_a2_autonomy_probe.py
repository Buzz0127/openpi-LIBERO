#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import a2_autonomy_probe as probe


class A2AutonomyProbeTest(unittest.TestCase):
    def test_requires_offline_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {**os.environ, **probe.REQUIRED_ENVIRONMENT}
            for key in probe.PROXY_VARIABLES:
                environment.pop(key, None)
            result = subprocess.run(
                [
                    sys.executable, str(Path(probe.__file__)), "--progress", str(root / "progress.json"),
                    "--output", str(root / "output.json"), "--steps", "3", "--interval-seconds", "0.01",
                ],
                env=environment,
                start_new_session=True,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "output.json").is_file())

    def test_existing_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output.json").write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                original = sys.argv
                try:
                    sys.argv = ["probe", "--progress", str(root / "progress.json"), "--output", str(root / "output.json"), "--steps", "3", "--interval-seconds", "0.01"]
                    probe.main()
                finally:
                    sys.argv = original


if __name__ == "__main__":
    unittest.main()
