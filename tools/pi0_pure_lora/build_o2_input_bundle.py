"""Create a collision-safe, hashed synthetic O2 Policy input/noise bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def write_bundle(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # These are already LIBERO-client-preprocessed policy inputs: no server-side
    # rotation/resize is permitted. Noise is internal action-space (50, 32).
    values = {
        "image": np.full((224, 224, 3), 127, dtype=np.uint8),
        "wrist_image": np.full((224, 224, 3), 127, dtype=np.uint8),
        "state": np.zeros((8,), dtype=np.float32),
        "noise": np.linspace(-0.25, 0.25, 50 * 32, dtype=np.float32).reshape(50, 32),
    }
    records = []
    for name, value in values.items():
        path = output / (name + ".npy")
        with path.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
        records.append({"name": name, "file": path.name, "dtype": str(value.dtype), "shape": list(value.shape), "array_sha256": array_sha256(value)})
    stable = {
        "schema_version": 1,
        "bundle_kind": "o2-fixed-preprocessed-policy-input-and-noise",
        "prompt": PROMPT,
        "records": records,
        "transform_contract": {
            "image_rotation": "already performed exactly once by LIBERO client",
            "resize": "already padded/resized to 224 by LIBERO client",
            "server_repack": "none",
            "normalization": "performed by canonical Policy transform chain",
            "noise": "explicit internal action-space tensor for direct Policy diagnostics only",
        },
    }
    manifest = dict(stable)
    manifest["bundle_identity_sha256"] = hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = write_bundle(args.output)
    print(json.dumps({"status": "bundle-written", "bundle_identity_sha256": manifest["bundle_identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
