#!/usr/bin/env python3

import hashlib
import pathlib
import tempfile
import unittest

import write_evidence_manifest as manifest


class EvidenceManifestTest(unittest.TestCase):
    def test_manifest_is_relative_sorted_and_collision_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "z").write_bytes(b"z")
            (root / "a").write_bytes(b"a")
            output = root / "final-files.sha256"
            payload = manifest.build_manifest(root, output)
            expected = (
                f"{hashlib.sha256(b'a').hexdigest()}  a\n"
                f"{hashlib.sha256(b'z').hexdigest()}  z\n"
            )
            self.assertEqual(payload, expected)
            output.write_text(payload, encoding="utf-8")
            with self.assertRaises(FileExistsError):
                manifest.build_manifest(root, output)

    def test_output_must_be_inside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "root"
            root.mkdir()
            (root / "file").write_bytes(b"x")
            with self.assertRaises(ValueError):
                manifest.build_manifest(root, root.parent / "manifest")

    def test_excluded_prefix_is_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "keep").write_bytes(b"keep")
            (root / "cache").mkdir()
            (root / "cache/ignored.pyc").write_bytes(b"ignored")
            payload = manifest.build_manifest(root, root / "portable.sha256", (pathlib.Path("cache"),))
            self.assertIn("  keep\n", payload)
            self.assertNotIn("ignored.pyc", payload)


if __name__ == "__main__":
    unittest.main()
