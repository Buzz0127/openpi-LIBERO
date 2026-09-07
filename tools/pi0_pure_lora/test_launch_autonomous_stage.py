#!/usr/bin/env python3

import pathlib
import tempfile
import unittest

import launch_autonomous_stage as launcher


class AutonomousLauncherTest(unittest.TestCase):
    def test_session_name_is_strict(self) -> None:
        self.assertIsNotNone(launcher.SESSION_RE.fullmatch("pi0-a2.test_01"))
        self.assertIsNone(launcher.SESSION_RE.fullmatch("bad;command"))
        self.assertIsNone(launcher.SESSION_RE.fullmatch("bad name"))

    def test_receipt_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "receipt.json"
            launcher._atomic_new_json(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                launcher._atomic_new_json(path, {"ok": False})

    def test_attempt_binding_is_exact(self) -> None:
        self.assertEqual(
            launcher._bound_attempt(["--stage", "A2", "--attempt-dir", "/tmp/a2-attempt"]),
            pathlib.Path("/tmp/a2-attempt").resolve(),
        )
        with self.assertRaises(ValueError):
            launcher._bound_attempt(["--stage", "A2"])


if __name__ == "__main__":
    unittest.main()
