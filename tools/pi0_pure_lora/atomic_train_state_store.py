"""Atomic, fail-closed train-state saves without automatic checkpoint deletion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp


STEP_RE = re.compile(r"step-(\d{8})$")
COMMIT_RE = re.compile(r"step-(\d{8})\.commit\.json$")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def tree_sha256(tree: Any) -> str:
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    digest = hashlib.sha256(str(treedef).encode())
    for path, leaf in paths_and_leaves:
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(_canonical({"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype)}))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class AtomicTrainStateStore:
    """Stores committed steps; intentionally exposes no delete/prune operation."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_verified(self, step: int, state: Any, *, fail_after_async_save: bool = False) -> dict[str, Any]:
        if step < 0:
            raise ValueError("step must be non-negative")
        final = self.root / f"step-{step:08d}"
        commit = self.root / f"step-{step:08d}.commit.json"
        if final.exists() or commit.exists():
            raise FileExistsError(f"step already exists: {step}")
        staging = self.root / f".step-{step:08d}.staging-{uuid.uuid4().hex}"
        expected_hash = tree_sha256(state)
        handler = ocp.PyTreeCheckpointHandler()
        with ocp.AsyncCheckpointer(handler, timeout_secs=600) as checkpointer:
            checkpointer.save(staging, args=ocp.args.PyTreeSave(state))
            checkpointer.wait_until_finished()
        if fail_after_async_save:
            raise RuntimeError("simulated interruption after async save")
        restored_staging = ocp.PyTreeCheckpointer().restore(staging)
        if tree_sha256(restored_staging) != expected_hash:
            raise ValueError("staging restore hash mismatch")
        os.replace(staging, final)
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        restored_final = ocp.PyTreeCheckpointer().restore(final)
        if tree_sha256(restored_final) != expected_hash:
            raise ValueError("final restore hash mismatch")
        manifest = {
            "schema_version": 1,
            "step": step,
            "state_sha256": expected_hash,
            "async_wait_completed": True,
            "staging_restore_verified": True,
            "final_restore_verified": True,
            "automatic_pruning_enabled": False,
        }
        _atomic_json(commit, manifest)
        return manifest

    def committed_steps(self) -> list[int]:
        steps: list[int] = []
        for path in self.root.iterdir():
            match = COMMIT_RE.fullmatch(path.name)
            if not match:
                continue
            step = int(match.group(1))
            final = self.root / f"step-{step:08d}"
            if not final.is_dir():
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("step") == step and manifest.get("final_restore_verified") is True:
                steps.append(step)
        return sorted(steps)

    def staging_paths(self) -> list[Path]:
        return sorted(path for path in self.root.iterdir() if path.name.startswith(".step-") and ".staging-" in path.name)

    def restore_verified(self, step: int) -> Any:
        if step not in self.committed_steps():
            raise FileNotFoundError(f"step is not committed: {step}")
        manifest = json.loads((self.root / f"step-{step:08d}.commit.json").read_text(encoding="utf-8"))
        restored = ocp.PyTreeCheckpointer().restore(self.root / f"step-{step:08d}")
        if tree_sha256(restored) != manifest["state_sha256"]:
            raise ValueError(f"committed step hash mismatch: {step}")
        return restored
