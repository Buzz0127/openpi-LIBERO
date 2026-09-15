"""Run the sealed O2 diagnostics and sequential timing inside one guarded PGID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import experiment_identity

PROJECT_ARMS = ("project_unmerged_bf16", "project_merged_dense_bf16")


def verify_control(path: Path) -> dict[str, Any]:
    control = json.loads(path.read_text(encoding="utf-8"))
    identity = control.pop("plan_identity_sha256", None)
    if control.get("execution_authorized") is not False or identity != experiment_identity.canonical_sha256(control):
        raise ValueError("O2 control plan is altered or authorizing")
    return control


def load_arms(path: Path) -> dict[str, dict[str, Any]]:
    arms = json.loads(path.read_text(encoding="utf-8"))
    expected = ["official_pi0_libero_reference", *PROJECT_ARMS]
    if not isinstance(arms, list) or [item.get("name") for item in arms] != expected:
        raise ValueError("O2 arm order is not pre-registered")
    return {item["name"]: item for item in arms}


def runtime_args(arm: dict[str, Any], *, mode: str) -> tuple[Path, dict[str, Any]]:
    command = arm["command"]
    if not isinstance(command, list) or len(command) < 4:
        raise ValueError("invalid project arm command")
    values: dict[str, Any] = {}
    for index in range(2, len(command), 2):
        if index + 1 >= len(command) or not command[index].startswith("--"):
            raise ValueError("project arm flags must be complete pairs")
        values[command[index][2:].replace("-", "_")] = command[index + 1]
    # The direct diagnostic reconstructs the same runtime namespace as the
    # WebSocket entry point, so validate the mode but retain it in the JSON.
    if values.get("runtime_mode") != mode or "port" not in values:
        raise ValueError("project arm mode/port mismatch")
    values["port"] = int(values["port"])
    if "rng_seed" not in values:
        raise ValueError("project arm lacks rng seed")
    values["rng_seed"] = int(values["rng_seed"])
    return Path(command[1]), values


def _run(command: list[str], stdout: Path, stderr: Path, timeout: int) -> None:
    with stdout.open("xb") as out, stderr.open("xb") as err:
        subprocess.run(command, stdout=out, stderr=err, check=True, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--arm-spec", required=True, type=Path)
    parser.add_argument("--input-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.python.is_absolute() or not args.python.is_file():
        raise ValueError("output must be new and python must be an absolute file")
    control, arms = verify_control(args.control), load_arms(args.arm_spec)
    if control["pre_registered_protocol"]["max_wallclock_seconds"] != 2700:
        raise ValueError("O2 must retain its 45-minute bound")
    args.output_dir.mkdir(parents=True)
    direct = Path(arms[PROJECT_ARMS[0]]["command"][1]).parent / "o2_direct_policy_diagnostic.py"
    compare = direct.with_name("compare_o2_direct_results.py")
    timing = direct.with_name("o2_websocket_timing.py")
    for script in (direct, compare, timing):
        if not script.is_file(): raise FileNotFoundError(script)
    # Reconstruct fresh Policy processes: unmerged once, merged twice for the
    # pre-registered exact-repeatability gate.  Each invocation exits itself.
    direct_roots: dict[str, Path] = {}
    for label, arm_name, mode in (("unmerged", PROJECT_ARMS[0], "unmerged-lora"), ("merged-a", PROJECT_ARMS[1], "merged-dense")):
        server_source, values = runtime_args(arms[arm_name], mode=mode)
        args_json = args.output_dir / f"{label}.server-args.json"
        args_json.write_text(json.dumps(values, sort_keys=True), encoding="utf-8")
        root = args.output_dir / label; direct_roots[label] = root
        command = [str(args.python), str(direct), "--server-source", str(server_source), "--server-args", str(args_json), "--input-bundle", str(args.input_bundle), "--output-dir", str(root)]
        if label == "merged-a":
            direct_roots["merged-b"] = args.output_dir / "merged-b"
            command += ["--repeat-output-dir", str(direct_roots["merged-b"])]
        _run(command, args.output_dir / f"{label}.stdout.log", args.output_dir / f"{label}.stderr.log", 540)
    comparisons = {}
    for label, left, right in (("merged-repeat", "merged-a", "merged-b"), ("unmerged-vs-merged", "unmerged", "merged-a")):
        output = args.output_dir / f"{label}.json"
        _run([str(args.python), str(compare), "--left", str(direct_roots[left]), "--right", str(direct_roots[right]), "--output", str(output)], args.output_dir / f"{label}.stdout.log", args.output_dir / f"{label}.stderr.log", 60)
        comparisons[label] = json.loads(output.read_text(encoding="utf-8"))
    if not (comparisons["merged-repeat"]["normalized_action_difference"]["exact_sha256_equal"] and comparisons["merged-repeat"]["physical_action_difference"]["exact_sha256_equal"]):
        raise RuntimeError("merged-dense exact repeatability gate failed")
    _run([str(args.python), str(timing), "--attempt-dir", str(args.output_dir / "websocket"), "--input-bundle", str(args.input_bundle), "--arm-spec", str(args.arm_spec), "--warmups", "10", "--samples", "50"], args.output_dir / "websocket.stdout.log", args.output_dir / "websocket.stderr.log", 1080)
    experiment_identity.atomic_write_new(args.output_dir / "result.json", {"schema_version": 1, "stage": "O2-execution", "direct_comparisons": comparisons, "websocket_timing": "websocket/timing.json", "status": "pass"})
    print(json.dumps({"status": "pass", "output": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
