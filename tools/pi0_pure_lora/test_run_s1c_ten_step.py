#!/usr/bin/env python3

import pathlib
import tempfile
import unittest

import run_s1c_ten_step as s1c


class S1cStaticTest(unittest.TestCase):
    def test_exactly_ten_steps_and_no_checkpoint_save(self):
        self.assertEqual(s1c.TRAINING_STEPS, 10)
        source = pathlib.Path(s1c.__file__).read_text(encoding="utf-8")
        self.assertNotIn("save_state(", source)
        self.assertNotIn("export_adapter", source)

    def test_atomic_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            s1c._atomic_json_new(path, {"step": 10})
            with self.assertRaises(FileExistsError):
                s1c._atomic_json_new(path, {"step": 11})


if __name__ == "__main__":
    unittest.main()
