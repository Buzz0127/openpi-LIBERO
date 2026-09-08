"""Reproduce only the T1 execution-control CPU fixture suites, with bounded runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

SUITES = [
    "test_run_t1_resume_segment", "test_resume_sequence", "test_gpu_stage_preflight",
    "test_build_t1_execution_template", "test_finalize_t1_execution_plan",
    "test_verify_t1_resume_result", "test_autonomous_stage_orchestrator",
    "test_build_autonomous_stage_plan", "test_launch_autonomous_stage",
    "test_gpu_utilization_guard", "test_storage_budget_guard",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # macOS tar/scp workflows can materialize AppleDouble ``._*.py`` metadata
    # on Linux.  Those are transport metadata, not Python sources.
    sources = sorted(
        path for path in (repo / "tools/pi0_pure_lora").glob("*.py")
        if not path.name.startswith("._")
    ) + [
        repo / "tools" / name for name in (
            "gpu_utilization_guard.py", "storage_budget_guard.py",
            "test_gpu_utilization_guard.py", "test_storage_budget_guard.py",
        )
    ]
    source_hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": os.pathsep.join([str(repo / "tools"), str(repo / "tools/pi0_pure_lora")]),
        "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONDONTWRITEBYTECODE": "1",
    })
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(name, None)
    started = time.time()
    records = []
    with tempfile.TemporaryDirectory(prefix="t1-control-cpu-pycache-") as cache:
        environment["PYTHONPYCACHEPREFIX"] = cache
        commands = [("py_compile", [sys.executable, "-m", "py_compile", *map(str, sources), str(Path(__file__).resolve())])]
        commands += [(name, [sys.executable, "-m", "unittest", name, "-v"]) for name in SUITES]
        for name, command in commands:
            result = subprocess.run(command, cwd=repo, env=environment, text=True, capture_output=True, timeout=180)
            (out / f"{name}.stdout.txt").write_text(result.stdout)
            (out / f"{name}.stderr.txt").write_text(result.stderr)
            match = re.search(r"Ran (\d+) tests? in", result.stderr)
            records.append({"name": name, "command": command, "returncode": result.returncode,
                            "tests_run": int(match.group(1)) if match else 0})
            print(f"{name}: exit={result.returncode}, tests={records[-1]['tests_run']}", flush=True)
    stable = all(hashlib.sha256((repo / p).read_bytes()).hexdigest() == digest for p, digest in source_hashes.items())
    report = {
        "schema_version": 1, "status": "pass" if stable and all(r["returncode"] == 0 for r in records) else "fail",
        "python_executable": sys.executable, "python_version": sys.version,
        "repo_root": str(repo), "source_sha256": source_hashes, "source_stable_during_tests": stable,
        "records": records, "tests_run": sum(r["tests_run"] for r in records),
        "elapsed_seconds": time.time() - started, "scope": "stdlib tests with temporary fake artifacts, GPU samples and child processes",
        "real_gpu_preflight_run": False, "real_runner_run": False,
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
