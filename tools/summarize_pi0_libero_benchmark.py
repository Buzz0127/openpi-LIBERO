#!/usr/bin/env python3
"""Validate and summarize the complete four-suite π0-LIBERO evaluation."""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import pathlib
import statistics
from typing import Any, Iterable


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_LABELS = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_object": "LIBERO-Object",
    "libero_goal": "LIBERO-Goal",
    "libero_10": "LIBERO-10",
}
EXPECTED_OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
EXPECTED_LIBERO_COMMIT = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
EXPECTED_NORM_SHA256 = "dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1"
EXPECTED_EVALUATOR_SHA256 = "cfce604677a480534397ddad25358c7a95c982de57287c8df5042ee28d9c4312"
EXPECTED_GUARD_SHA256 = "ba9f7c74b843a61dc7cdf2f48b182390edb59022b5a2e76a489ce9d7048c4594"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/libero-benchmark/pi0_libero_official4_seed7"),
    )
    parser.add_argument(
        "--task0-first",
        type=pathlib.Path,
        default=pathlib.Path(
            "artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/evaluation"
        ),
    )
    parser.add_argument(
        "--task0-second",
        type=pathlib.Path,
        default=pathlib.Path(
            "artifacts/libero-eval/pi0_libero_spatial_task0_init10-49_seed7/evaluation"
        ),
    )
    return parser.parse_args()


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nearest_percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_iso(timestamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(record["success"]) for record in records)
    failures = len(records) - successes
    control_steps = [int(record["control_steps"]) for record in records]
    policy_requests = [int(record["policy_requests"]) for record in records]
    wall_seconds = [float(record["episode_wall_seconds"]) for record in records]
    return {
        "episodes": len(records),
        "successes": successes,
        "failures": failures,
        "success_rate": successes / len(records),
        "total_control_steps": sum(control_steps),
        "mean_control_steps": statistics.fmean(control_steps),
        "p95_control_steps": nearest_percentile(control_steps, 0.95),
        "total_policy_requests": sum(policy_requests),
        "mean_policy_requests": statistics.fmean(policy_requests),
        "total_episode_wall_seconds": sum(wall_seconds),
        "mean_episode_wall_seconds": statistics.fmean(wall_seconds),
    }


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    remaining_root = artifact_root / "raw" / "remaining"
    control_root = artifact_root / "raw" / "control"
    artifact_root.mkdir(parents=True, exist_ok=True)

    result_sources = [
        args.task0_first.resolve() / "results.jsonl",
        args.task0_second.resolve() / "results.jsonl",
        *sorted(remaining_root.glob("*/task_*/evaluation/results.jsonl")),
    ]
    if len(result_sources) != 41:
        raise RuntimeError(f"expected 41 result segments, found {len(result_sources)}")

    located_records: list[tuple[dict[str, Any], pathlib.Path]] = []
    for source in result_sources:
        located_records.extend((record, source) for record in load_jsonl(source))
    records = [record for record, _ in located_records]

    by_task: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_task[(str(record["suite"]), int(record["task_id"]))].append(record)

    expected_tasks = {(suite, task_id) for suite in SUITES for task_id in range(10)}
    if set(by_task) != expected_tasks:
        raise RuntimeError(f"task coverage mismatch: {sorted(set(by_task) ^ expected_tasks)}")
    for key, task_records in by_task.items():
        states = [int(record["initial_state_index"]) for record in task_records]
        if len(task_records) != 50 or set(states) != set(range(50)) or len(states) != len(set(states)):
            raise RuntimeError(f"invalid state coverage for {key}: {states}")
    if len(records) != 2000:
        raise RuntimeError(f"expected 2000 episodes, found {len(records)}")
    if {int(record["seed"]) for record in records} != {7}:
        raise RuntimeError("records contain an unexpected seed")

    config_paths = [
        args.task0_first.resolve() / "run_config.json",
        args.task0_second.resolve() / "run_config.json",
        *sorted(remaining_root.glob("*/task_*/evaluation/run_config.json")),
    ]
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in config_paths]
    for config in configs:
        identity = config["identity"]
        expected = {
            "openpi_commit": EXPECTED_OPENPI_COMMIT,
            "libero_commit": EXPECTED_LIBERO_COMMIT,
            "norm_stats_sha256": EXPECTED_NORM_SHA256,
            "tool_sha256": EXPECTED_EVALUATOR_SHA256,
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise RuntimeError(f"identity mismatch for {key}: {identity.get(key)!r}")
        if config.get("policy_config") != "pi0_libero":
            raise RuntimeError("non-pi0_libero policy found")

    task_rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for suite in SUITES:
        for task_id in range(10):
            task_records = sorted(by_task[(suite, task_id)], key=lambda item: item["initial_state_index"])
            metrics = summarize_records(task_records)
            failed_states = [int(record["initial_state_index"]) for record in task_records if not record["success"]]
            summary = {
                "suite": suite,
                "suite_label": SUITE_LABELS[suite],
                "task_id": task_id,
                "task_description": str(task_records[0]["task_description"]),
                **metrics,
                "failed_initial_states": failed_states,
            }
            task_summaries.append(summary)
            task_rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_description": summary["task_description"],
                    "episodes": metrics["episodes"],
                    "successes": metrics["successes"],
                    "failures": metrics["failures"],
                    "success_rate_percent": f'{100 * metrics["success_rate"]:.2f}',
                    "mean_control_steps": f'{metrics["mean_control_steps"]:.2f}',
                    "p95_control_steps": f'{metrics["p95_control_steps"]:.0f}',
                    "mean_policy_requests": f'{metrics["mean_policy_requests"]:.2f}',
                    "failed_initial_states": ";".join(map(str, failed_states)),
                }
            )

    suite_summaries = []
    for suite in SUITES:
        suite_records = [record for record in records if record["suite"] == suite]
        suite_summaries.append(
            {"suite": suite, "suite_label": SUITE_LABELS[suite], **summarize_records(suite_records)}
        )
    overall = summarize_records(records)

    failure_rows = []
    for record, source in located_records:
        if record["success"]:
            continue
        video = record.get("video", {}).get("filename")
        video_path = source.parent / video if video else None
        failure_rows.append(
            {
                "suite": record["suite"],
                "task_id": record["task_id"],
                "initial_state_index": record["initial_state_index"],
                "failure_reason": record.get("failure_reason"),
                "control_steps": record["control_steps"],
                "policy_requests": record["policy_requests"],
                "episode_wall_seconds": f'{record["episode_wall_seconds"]:.3f}',
                "exception_type": record.get("exception_type") or "",
                "video_path": str(video_path.relative_to(pathlib.Path.cwd())) if video_path else "",
            }
        )

    guard_path = control_root / "benchmark" / "guard.jsonl"
    guard_events = load_jsonl(guard_path)
    guard_counts = collections.Counter(event["event"] for event in guard_events)
    gpu_samples = [event for event in guard_events if event["event"] == "gpu_sample"]
    runtime_seconds = (
        parse_iso(guard_events[-1]["timestamp_utc"]) - parse_iso(guard_events[0]["timestamp_utc"])
    ).total_seconds()
    guard_summary = {
        "sha256": EXPECTED_GUARD_SHA256,
        "physical_gpu": 0,
        "sample_count": len(gpu_samples),
        "peak_utilization_percent": max(event["utilization_percent"] for event in gpu_samples),
        "peak_memory_used_mib": max(event["memory_used_mib"] for event in gpu_samples),
        "minimum_free_memory_percent": min(event["free_memory_percent"] for event in gpu_samples),
        "pause_events": sum(event.get("action") == "paused" for event in gpu_samples),
        "resume_events": sum(event.get("action") == "resumed" for event in gpu_samples),
        "monitor_errors": guard_counts.get("monitor_error", 0),
        "memory_emergencies": guard_counts.get("memory_emergency", 0),
        "child_exit_code": guard_events[-1].get("return_code"),
        "runtime_seconds": runtime_seconds,
    }

    failure_reasons = collections.Counter(
        str(record.get("failure_reason")) for record in records if not record["success"]
    )
    exception_count = sum(record.get("exception_type") is not None for record in records)
    summary = {
        "schema_version": 1,
        "scope": "original pi0_libero on four standard LIBERO suites",
        "protocol": {
            "seed": 7,
            "suites": list(SUITES),
            "tasks_per_suite": 10,
            "initial_states_per_task": 50,
            "episodes": 2000,
            "replan_steps": 5,
            "policy_config": "pi0_libero",
        },
        "identity": {
            "openpi_commit": EXPECTED_OPENPI_COMMIT,
            "libero_commit": EXPECTED_LIBERO_COMMIT,
            "norm_stats_sha256": EXPECTED_NORM_SHA256,
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            "guard_sha256": EXPECTED_GUARD_SHA256,
        },
        "overall": overall,
        "suites": suite_summaries,
        "tasks": task_summaries,
        "failures": {
            "count": overall["failures"],
            "by_reason": dict(sorted(failure_reasons.items())),
            "exception_count": exception_count,
        },
        "guard": guard_summary,
        "raw_evidence": {
            "remaining_results_file_count": sum(1 for path in remaining_root.rglob("*") if path.is_file()),
            "remaining_results_bytes": sum(path.stat().st_size for path in remaining_root.rglob("*") if path.is_file()),
            "control_file_count": sum(1 for path in control_root.rglob("*") if path.is_file()),
            "control_bytes": sum(path.stat().st_size for path in control_root.rglob("*") if path.is_file()),
            "git_ignored": True,
        },
    }

    (artifact_root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(
        artifact_root / "task_results.csv",
        [
            "suite",
            "task_id",
            "task_description",
            "episodes",
            "successes",
            "failures",
            "success_rate_percent",
            "mean_control_steps",
            "p95_control_steps",
            "mean_policy_requests",
            "failed_initial_states",
        ],
        task_rows,
    )
    write_csv(
        artifact_root / "failure_cases.csv",
        [
            "suite",
            "task_id",
            "initial_state_index",
            "failure_reason",
            "control_steps",
            "policy_requests",
            "episode_wall_seconds",
            "exception_type",
            "video_path",
        ],
        failure_rows,
    )

    suite_lines = [
        "| Suite | 成功 / Episodes | 成功率 |",
        "| --- | ---: | ---: |",
    ]
    for suite_summary in suite_summaries:
        suite_lines.append(
            f'| {suite_summary["suite_label"]} | {suite_summary["successes"]} / '
            f'{suite_summary["episodes"]} | {100 * suite_summary["success_rate"]:.2f}% |'
        )
    suite_lines.append(
        f'| **四 suite 总计** | **{overall["successes"]} / {overall["episodes"]}** | '
        f'**{100 * overall["success_rate"]:.2f}%** |'
    )

    task_lines = [
        "| Suite | Task | 成功 / 50 | 成功率 | 失败 states |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for task in task_summaries:
        failures = ", ".join(map(str, task["failed_initial_states"])) or "—"
        task_lines.append(
            f'| {task["suite_label"]} | {task["task_id"]} | {task["successes"]} / 50 | '
            f'{100 * task["success_rate"]:.2f}% | {failures} |'
        )

    hours, remainder = divmod(int(runtime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    report = f"""# 原始 π0-LIBERO 四 Suite 完整评测报告

## 结论

在固定 OpenPI、LIBERO 和原始 `pi0_libero` checkpoint 身份下，本项目完成
4 个标准 suite、40 个 task、每个 task 50 个固定初始状态，共 2,000 个闭环
episode。总成功率为 **{overall['successes']}/{overall['episodes']}（{100 * overall['success_rate']:.2f}%）**。

{chr(10).join(suite_lines)}

这是一组固定版本、固定 seed 7 的完整四-suite评测结果。它不应被表述为多 seed
统计，也不用于声称逐项等同于论文作者的硬件、软件或随机性条件。

## 固定身份与协议

- OpenPI commit：`{EXPECTED_OPENPI_COMMIT}`
- LIBERO commit：`{EXPECTED_LIBERO_COMMIT}`
- Policy：`pi0_libero`
- Checkpoint：`openpi-assets/checkpoints/pi0_libero`
- Norm stats SHA-256：`{EXPECTED_NORM_SHA256}`
- Evaluator SHA-256：`{EXPECTED_EVALUATOR_SHA256}`
- Seed：7
- 每个 task：initial states 0–49
- 模型输出 50 步动作块，客户端执行前 5 步后重新观察与规划

## 逐 Task 结果

{chr(10).join(task_lines)}

机器可读明细见 [`task_results.csv`](task_results.csv)。

## 失败与完整性

- 失败 episode：{overall['failures']}
- `max_control_steps`：{failure_reasons.get('max_control_steps', 0)}
- Python、MuJoCo、WebSocket 或评测器系统异常：{exception_count}
- 40 个 `(suite, task)` 组合均包含且仅包含 initial states 0–49
- 没有重复或缺失 episode

全部失败条目及本地视频路径见 [`failure_cases.csv`](failure_cases.csv)。达到控制步上限
表示 episode 没有及时满足 LIBERO 成功谓词；它属于任务失败，不是程序崩溃。进一步的
行为分类需要逐个检查失败视频，不能只根据 `max_control_steps` 推断原因。

## GPU 与运行保护

- 物理 GPU：0；`CUDA_VISIBLE_DEVICES=0`
- JAX 默认显存预分配：关闭
- 保护器采样：{guard_summary['sample_count']:,} 次
- 峰值 GPU 利用率：{guard_summary['peak_utilization_percent']:.0f}%
- 峰值显存：{guard_summary['peak_memory_used_mib']:.0f} MiB
- 最低剩余显存：{guard_summary['minimum_free_memory_percent']:.2f}%
- 暂停 / 恢复 / 监控错误 / 显存紧急事件：{guard_summary['pause_events']} / {guard_summary['resume_events']} / {guard_summary['monitor_errors']} / {guard_summary['memory_emergencies']}
- 保护器子进程退出码：{guard_summary['child_exit_code']}
- 保护器记录时长：{hours} 小时 {minutes} 分 {seconds} 秒

运行结束后 Policy Server、评测器和保护器均退出，端口 8000 与 GPU 显存均释放。

## 证据与复现边界

可提交证据包括本报告、[`benchmark_summary.json`](benchmark_summary.json)、
[`task_results.csv`](task_results.csv) 和 [`failure_cases.csv`](failure_cases.csv)。完整视频、
逐 episode JSON、动作轨迹、GPU 采样和服务器日志保存在本地 `raw/`，因体积与机器信息
不提交 Git。`raw/` 的结果文件共 {summary['raw_evidence']['remaining_results_file_count']:,} 个，
控制文件共 {summary['raw_evidence']['control_file_count']:,} 个。

本结果最明显的短板是 LIBERO-10（{suite_summaries[-1]['successes']}/500，
{100 * suite_summaries[-1]['success_rate']:.2f}%）。下一步应按 task 和失败视频进行行为级
归类，而不是通过提高控制步上限重新定义本次固定协议。
"""
    (artifact_root / "run_report.md").write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "episodes": overall["episodes"],
                "successes": overall["successes"],
                "success_rate": overall["success_rate"],
                "task_count": len(task_summaries),
                "failure_count": len(failure_rows),
                "output": str(artifact_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
