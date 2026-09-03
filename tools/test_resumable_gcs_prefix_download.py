from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("resumable_gcs_prefix_download.py")
SPEC = importlib.util.spec_from_file_location("gcsdl", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class GcsDownloadTest(unittest.TestCase):
    def test_media_url_pins_generation_and_escapes_name(self) -> None:
        url = MODULE.media_url("bucket", {"name": "a/b c", "generation": "123"})
        self.assertIn("a%2Fb%20c", url)
        self.assertTrue(url.endswith("generation=123"))

    def test_complete_partial_promotes_only_after_md5(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "x"
            partial = Path(str(target) + ".part")
            data = b"verified"
            partial.write_bytes(data)
            item = {"name": "p/x", "generation": "1", "size": len(data), "md5_b64": base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode()}
            MODULE.download_one(None, "bucket", item, target)
            self.assertEqual(target.read_bytes(), data)
            self.assertFalse(partial.exists())

    def test_bad_complete_partial_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "x"
            partial = Path(str(target) + ".part")
            partial.write_bytes(b"wrong")
            item = {"name": "p/x", "generation": "1", "size": 5, "md5_b64": base64.b64encode(hashlib.md5(b"right", usedforsecurity=False).digest()).decode()}
            with self.assertRaises(RuntimeError):
                MODULE.download_one(None, "bucket", item, target)
            self.assertTrue(partial.exists())
            self.assertFalse(target.exists())

    def test_atomic_json_refuses_no_data_loss(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "manifest.json"
            MODULE.atomic_json(path, {"x": 1})
            self.assertEqual(json.loads(path.read_text()), {"x": 1})

    def test_report_is_outside_promoted_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scratch, final, report = root / "params.partial", root / "params", root / "report.json"
            data = b"ok"
            item = {"name": "p/x", "relative_path": "x", "generation": "1", "size": len(data), "md5_b64": base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode()}
            scratch.mkdir()
            (scratch / "x").write_bytes(data)
            manifest = {"bucket": "b", "object_count": 1, "total_bytes": len(data), "objects": [item]}
            MODULE.run_download(manifest, scratch, final, None, 1, report)
            self.assertTrue(report.is_file())
            self.assertFalse((final / "download_report.json").exists())


if __name__ == "__main__":
    unittest.main()
