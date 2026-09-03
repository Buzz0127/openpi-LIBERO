from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from openpi.training import pure_lora_checkpointing as runtime


class FakeParams(dict):
    def to_pure_dict(self):
        return dict(self)


def _state(step: int):
    return SimpleNamespace(
        step=np.asarray(step),
        params=FakeParams({"m": {"lora_a": np.arange(2, dtype=np.float32), "kernel": np.ones(3)}}),
        opt_state={"mu": np.arange(2, dtype=np.float32)},
    )


class PureLoraCheckpointingTest(unittest.TestCase):
    def _inputs(self, root: Path) -> None:
        golden = {"review_invariants": {"adapter_leaf_count": 1}, "entries": [{"path": "m/lora_a", "shape": [2]}]}
        identities = {key: char * 64 for key, char in zip(sorted(runtime.adapter_artifact.REQUIRED_IDENTITIES), "1234")}
        (root / "golden.json").write_text(json.dumps(golden))
        identities["golden_manifest_sha256"] = runtime.experiment_identity.sha256_file(root / "golden.json")
        model = runtime.experiment_identity.build_model_manifest(
            model_mode="base", openpi_commit="a" * 40, identities=identities,
            adapter_identity_sha256=None, training_seed=None, artifact_purpose="test",
        )
        (root / "model.json").write_text(json.dumps(model))

    def test_save_restore_then_adapter_publish(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._inputs(root)
            manager = mock.Mock()
            state = _state(100)
            with mock.patch.object(runtime.checkpoints, "save_state") as save, mock.patch.object(runtime.checkpoints, "restore_state", return_value=state) as restore:
                receipt = runtime.save_restore_and_export(manager, state, object(), save_step=100, model_manifest_path=root / "model.json", golden_manifest_path=root / "golden.json", adapter_root=root / "adapters", train_seed=42)
            save.assert_called_once()
            manager.wait_until_finished.assert_called_once()
            restore.assert_called_once()
            self.assertTrue(receipt["checkpoint_restore_succeeded"])
            self.assertTrue((root / "adapters/step-00000100/manifest.json").is_file())
            self.assertTrue((root / "adapters/step-00000100.verified.json").is_file())

    def test_restore_mismatch_blocks_adapter(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._inputs(root)
            manager = mock.Mock()
            with mock.patch.object(runtime.checkpoints, "save_state"), mock.patch.object(runtime.checkpoints, "restore_state", return_value=_state(99)):
                with self.assertRaisesRegex(ValueError, "step mismatch"):
                    runtime.save_restore_and_export(manager, _state(100), object(), save_step=100, model_manifest_path=root / "model.json", golden_manifest_path=root / "golden.json", adapter_root=root / "adapters", train_seed=42)
            self.assertFalse((root / "adapters/step-00000100").exists())


if __name__ == "__main__":
    unittest.main()
