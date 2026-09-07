#!/usr/bin/env python3
"""Atomically write a collision-safe SHA-256 manifest for an evidence tree."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def _sha256_stable(path: Path) -> str:
    before = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"not a regular evidence file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"evidence file changed while hashing: {path}")
    return digest.hexdigest()


def build_manifest(root: Path, output: Path, exclude_prefixes: tuple[Path, ...] = ()) -> str:
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"evidence root is not a real directory: {root}")
    if output.exists():
        raise FileExistsError(output)
    if root not in output.parents:
        raise ValueError("manifest output must be inside the evidence root")
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if path == output:
            continue
        relative = path.relative_to(root)
        if any(relative == prefix or prefix in relative.parents for prefix in exclude_prefixes):
            continue
        if path.is_symlink():
            raise RuntimeError(f"evidence tree contains a symlink: {path}")
        if path.is_file():
            rows.append(f"{_sha256_stable(path)}  {relative}")
    if not rows:
        raise RuntimeError("evidence tree contains no files")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-prefix", action="append", default=[])
    args = parser.parse_args()
    exclude_prefixes = tuple(Path(value) for value in args.exclude_prefix)
    if any(prefix.is_absolute() or ".." in prefix.parts for prefix in exclude_prefixes):
        raise ValueError("exclude prefixes must be safe relative paths")
    payload = build_manifest(args.root, args.output, exclude_prefixes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(f"files={payload.count(chr(10))} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
