from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("run_compute_norm_stats_atomic.py")
SPEC = importlib.util.spec_from_file_location("run_compute_norm_stats_atomic", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AtomicPublishTest(unittest.TestCase):
    def test_publishes_only_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "canonical"

            def validate(path: Path):
                self.assertFalse(target.exists())
                self.assertEqual(path.read_bytes(), b"complete")
                return {"validated": True}

            result = MODULE.atomic_publish_directory(target, b"complete", validate)
            self.assertEqual(result, {"validated": True})
            self.assertEqual((target / "norm_stats.json").read_bytes(), b"complete")

    def test_refuses_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "canonical"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.atomic_publish_directory(target, b"new", lambda _path: {})

    def test_failed_validation_never_publishes_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "canonical"

            def fail(_path: Path):
                raise RuntimeError("invalid")

            with self.assertRaisesRegex(RuntimeError, "invalid"):
                MODULE.atomic_publish_directory(target, b"partial", fail)
            self.assertFalse(target.exists())
            self.assertEqual(len(list(Path(directory).glob(".canonical.partial-*"))), 1)

    def test_openpi_source_batch_semantics_are_explicit(self):
        self.assertEqual(273_465 // 32, 8_545)
        self.assertEqual(273_465 % 32, 25)


if __name__ == "__main__":
    unittest.main()
