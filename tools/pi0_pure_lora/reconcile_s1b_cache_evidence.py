"""Reconcile S1b's empty after-root files against the current offline cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _du_bytes(path: Path) -> int:
    output = subprocess.run(["du", "-sb", str(path)], check=True, capture_output=True, text=True).stdout
    return int(output.split(maxsplit=1)[0])


def _tree_summary(path: Path) -> dict[str, object]:
    stat = path.stat()
    files = [entry for entry in path.rglob("*") if entry.is_file()]
    return {
        "path": str(path),
        "realpath": str(path.resolve(strict=True)),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "du_bytes": _du_bytes(path),
        "file_count": len(files),
        "file_content_bytes": sum(entry.stat().st_size for entry in files),
    }


def _raw_dataset_summary(path: Path) -> dict[str, int]:
    files = [
        entry
        for entry in path.rglob("*")
        if entry.is_file() and ".cache" not in entry.relative_to(path).parts
    ]
    return {
        "file_count": len(files),
        "file_content_bytes": sum(entry.stat().st_size for entry in files),
    }


def reconcile(hf_home: Path, d1c_report_path: Path, s1b_static: Path) -> dict[str, object]:
    hf_home = hf_home.resolve(strict=True)
    d1c = json.loads(d1c_report_path.read_text(encoding="utf-8"))
    raw_roots = sorted(
        {str(info.parent.parent.resolve()) for info in (hf_home / "lerobot").rglob("meta/info.json")}
    )
    arrow_roots = sorted(
        {
            str(info.parent.resolve())
            for info in (hf_home / "datasets").rglob("dataset_info.json")
            if any(info.parent.glob("*.arrow"))
        }
    )
    s1b_raw_before = _lines(s1b_static / "raw-roots.before.txt")
    s1b_arrow_before = _lines(s1b_static / "arrow-roots.before.txt")
    s1b_raw_after = _lines(s1b_static / "raw-roots.after.txt")
    s1b_arrow_after = _lines(s1b_static / "arrow-roots.after.txt")
    current_hf_du = _du_bytes(hf_home)
    current_hf_files = sum(1 for entry in hf_home.rglob("*") if entry.is_file())
    s1b_hf_du_before = int(_lines(s1b_static / "hf.du.before.txt")[0].split()[0])
    s1b_hf_files_before = int(_lines(s1b_static / "hf.file-count.before.txt")[0])
    raw_dataset = _raw_dataset_summary(Path(raw_roots[0])) if len(raw_roots) == 1 else None

    expected_raw = d1c["raw_repo_roots_after"]
    expected_arrow = d1c["arrow_tree_roots_after"]
    checks = {
        "exactly_one_raw_root": len(raw_roots) == 1,
        "exactly_one_arrow_root": len(arrow_roots) == 1,
        "raw_matches_d1c_rb": raw_roots == expected_raw,
        "arrow_matches_d1c_rb": arrow_roots == expected_arrow,
        "raw_matches_s1b_before": raw_roots == s1b_raw_before,
        "arrow_matches_s1b_before": arrow_roots == s1b_arrow_before,
        "s1b_after_raw_capture_is_empty": not s1b_raw_after,
        "s1b_after_arrow_capture_is_empty": not s1b_arrow_after,
        "hf_du_matches_s1b_before": current_hf_du == s1b_hf_du_before,
        "hf_file_count_matches_s1b_before": current_hf_files == s1b_hf_files_before,
        "hf_du_matches_d1c_rb": current_hf_du == d1c["hf_home_after"]["apparent_bytes"],
        "hf_file_count_matches_d1c_rb": current_hf_files == d1c["hf_home_after"]["file_entry_count"],
        "raw_dataset_file_count_matches_d1c_rb": (
            raw_dataset is not None
            and raw_dataset["file_count"] == d1c["raw_snapshot_identity"]["actual_repo_file_count"]
        ),
        "raw_dataset_bytes_match_d1c_rb": (
            raw_dataset is not None
            and raw_dataset["file_content_bytes"] == d1c["raw_snapshot_identity"]["actual_repo_bytes"]
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "conclusion": (
            "S1b after-root files were empty because root collection failed; the raw and Arrow trees remain intact."
            if passed
            else "Current cache identity does not reconcile with D1c-Rb and S1b-before evidence."
        ),
        "checks": checks,
        "hf_home": {
            "path": str(hf_home),
            "realpath": str(hf_home),
            "du_bytes": current_hf_du,
            "file_count": current_hf_files,
        },
        "raw_roots": raw_roots,
        "arrow_roots": arrow_roots,
        "raw_full_tree_summary": _tree_summary(Path(raw_roots[0])) if len(raw_roots) == 1 else None,
        "raw_dataset_summary_excluding_hf_cache_metadata": raw_dataset,
        "arrow_summary": _tree_summary(Path(arrow_roots[0])) if len(arrow_roots) == 1 else None,
        "d1c_report": str(d1c_report_path.resolve(strict=True)),
        "s1b_static": str(s1b_static.resolve(strict=True)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--d1c-report", required=True, type=Path)
    parser.add_argument("--s1b-static", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = reconcile(args.hf_home, args.d1c_report, args.s1b_static)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
