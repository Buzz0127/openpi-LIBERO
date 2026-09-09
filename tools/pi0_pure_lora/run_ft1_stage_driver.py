#!/usr/bin/env python3
"""Run one FT1 runner and require its independent terminal acceptance report."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--acceptance-output", required=True, type=Path)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner_args = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    if not runner_args or args.acceptance_output.exists():
        raise ValueError("new acceptance output and runner arguments are required")
    subprocess.run([sys.executable, str(args.runner), *runner_args], check=True)
    subprocess.run([
        sys.executable, str(args.verifier), "--template", str(args.template), "--result", str(args.result),
        "--output", str(args.acceptance_output),
    ], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
