from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import build_o2_input_bundle as bundle


class O2InputBundleTest(unittest.TestCase):
    def test_bundle_is_collision_safe_and_hashes_fixed_preprocessed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            manifest = bundle.write_bundle(output)
            self.assertEqual(manifest["bundle_kind"], "o2-fixed-preprocessed-policy-input-and-noise")
            self.assertEqual(len(manifest["records"]), 4)
            noise = np.load(output / "noise.npy", allow_pickle=False)
            self.assertEqual(noise.shape, (50, 32))
            self.assertEqual(noise.dtype, np.dtype("float32"))
            self.assertEqual(bundle.write_bundle.__name__, "write_bundle")
            with self.assertRaises(FileExistsError):
                bundle.write_bundle(output)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["bundle_identity_sha256"], manifest["bundle_identity_sha256"])


if __name__ == "__main__":
    unittest.main()
