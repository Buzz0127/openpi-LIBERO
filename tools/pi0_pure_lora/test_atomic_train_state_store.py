from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from atomic_train_state_store import AtomicTrainStateStore, tree_sha256


def state(step: int) -> dict:
    return {
        "step": np.asarray(step, dtype=np.int64),
        "params": {"adapter": np.arange(12, dtype=np.float32).reshape(3, 4) + step},
        "opt_state": {"count": np.asarray(step, dtype=np.int32), "momentum": np.full((3, 4), step / 10, np.float32)},
    }


class AtomicTrainStateStoreTest(unittest.TestCase):
    def test_two_verified_steps_coexist_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = AtomicTrainStateStore(Path(raw))
            store.save_verified(0, state(0))
            store.save_verified(100, state(100))
            self.assertEqual(store.committed_steps(), [0, 100])
            self.assertTrue((Path(raw) / "step-00000000").is_dir())
            self.assertEqual(tree_sha256(store.restore_verified(100)), tree_sha256(state(100)))

    def test_interrupted_staging_is_not_discoverable_and_old_survives(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = AtomicTrainStateStore(Path(raw))
            store.save_verified(0, state(0))
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                store.save_verified(100, state(100), fail_after_async_save=True)
            self.assertEqual(store.committed_steps(), [0])
            self.assertEqual(len(store.staging_paths()), 1)
            self.assertEqual(tree_sha256(store.restore_verified(0)), tree_sha256(state(0)))

    def test_tampered_commit_hash_fails_restore(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = AtomicTrainStateStore(root)
            store.save_verified(0, state(0))
            commit = root / "step-00000000.commit.json"
            value = json.loads(commit.read_text())
            value["state_sha256"] = "0" * 64
            commit.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                store.restore_verified(0)

    def test_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = AtomicTrainStateStore(Path(raw))
            store.save_verified(0, state(0))
            with self.assertRaises(FileExistsError):
                store.save_verified(0, state(1))


if __name__ == "__main__":
    unittest.main()
