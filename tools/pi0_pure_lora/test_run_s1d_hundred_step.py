#!/usr/bin/env python3

import pathlib
import tempfile
import unittest

import run_s1d_hundred_step as s1d


class S1dStaticTest(unittest.TestCase):
    def test_exact_stage_bounds_and_required_checkpoint_flow(self):
        self.assertEqual(s1d.TRAINING_STEPS, 100)
        self.assertEqual(s1d.TRAINING_SEED, 42)
        source = pathlib.Path(s1d.__file__).read_text(encoding="utf-8")
        self.assertIn("save_restore_and_export(", source)
        self.assertIn("compose_adapter(", source)
        self.assertIn("max_to_keep=None", source)
        self.assertIn("keep_period=None", source)

    def test_atomic_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            s1d._atomic_json_new(path, {"step": 100})
            with self.assertRaises(FileExistsError):
                s1d._atomic_json_new(path, {"step": 101})

    def test_path_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.assertTrue(s1d._inside(root / "checkpoints/run", root))
            self.assertFalse(s1d._inside(root.parent / "escape", root))


if __name__ == "__main__":
    unittest.main()
