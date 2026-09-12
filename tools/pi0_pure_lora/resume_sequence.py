#!/usr/bin/env python3
"""Deterministic iterator positioning helpers for segmented pure-LoRA training."""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Iterable, Iterator, TypeVar


T = TypeVar("T")


def canonical_fingerprint(value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


ProgressCallback = Callable[[str, int, int], None]


def positioned_batch(
    loader: Iterable[T], restored_step: int, *, progress: ProgressCallback | None = None, progress_interval: int = 100,
) -> tuple[Iterator[T], T, dict[str, object]]:
    if isinstance(restored_step, bool) or not isinstance(restored_step, int) or restored_step < 0:
        raise ValueError("restored_step must be a non-negative integer")
    iterator = iter(loader)
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if progress is not None:
        progress("positioned_loader", 0, restored_step)
    for index in range(restored_step):
        try:
            next(iterator)
        except StopIteration as error:
            raise RuntimeError("loader ended before restored_step") from error
        if progress is not None and ((index + 1) % progress_interval == 0 or index + 1 == restored_step):
            progress("positioned_loader", index + 1, restored_step)
    try:
        batch = next(iterator)
    except StopIteration as error:
        raise RuntimeError("loader ended at restored_step") from error
    return iterator, batch, {
        "schema_version": 1,
        "restored_step": restored_step,
        "skipped_batch_count": restored_step,
        "first_resumed_batch_index": restored_step,
    }


def verify_position(
    loader_factory: Callable[[], Iterable[T]], restored_step: int, fingerprint: Callable[[T], str],
    *, progress: ProgressCallback | None = None, progress_interval: int = 100,
) -> tuple[Iterator[T], T, dict[str, object]]:
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    reference = iter(loader_factory())
    expected = None
    total = restored_step + 1
    if progress is not None:
        progress("reference_loader", 0, total)
    for index in range(total):
        try:
            expected = next(reference)
        except StopIteration as error:
            raise RuntimeError("reference loader ended before resumed batch") from error
        if progress is not None and ((index + 1) % progress_interval == 0 or index + 1 == total):
            progress("reference_loader", index + 1, total)
    iterator, actual, receipt = positioned_batch(
        loader_factory(), restored_step, progress=progress, progress_interval=progress_interval,
    )
    expected_hash = fingerprint(expected)
    actual_hash = fingerprint(actual)
    if expected_hash != actual_hash:
        raise RuntimeError("resumed batch fingerprint does not match deterministic reference")
    return iterator, actual, {
        **receipt,
        "reference_batch_sha256": expected_hash,
        "resumed_batch_sha256": actual_hash,
        "position_verified": True,
    }
