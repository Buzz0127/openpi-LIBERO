from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest import mock

import run_ft1_stage_driver as driver


class DriverTests(unittest.TestCase):
    def test_runs_runner_before_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); calls = []
            with mock.patch.object(driver.subprocess, "run", side_effect=lambda argv, check: calls.append(argv)):
                with mock.patch("sys.argv", ["driver", "--runner", str(root / "r.py"), "--verifier", str(root / "v.py"), "--template", str(root / "t.json"), "--result", str(root / "r.json"), "--acceptance-output", str(root / "a.json"), "--", "--x"]):
                    self.assertEqual(driver.main(), 0)
            self.assertEqual(calls[0][1:], [str(root / "r.py"), "--x"])
            self.assertEqual(calls[1][1:], [str(root / "v.py"), "--template", str(root / "t.json"), "--result", str(root / "r.json"), "--output", str(root / "a.json")])

    def test_rejects_existing_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "a.json"; output.touch()
            with mock.patch("sys.argv", ["driver", "--runner", str(root / "r.py"), "--verifier", str(root / "v.py"), "--template", str(root / "t.json"), "--result", str(root / "r.json"), "--acceptance-output", str(output), "--", "--x"]):
                with self.assertRaisesRegex(ValueError, "new acceptance"):
                    driver.main()
