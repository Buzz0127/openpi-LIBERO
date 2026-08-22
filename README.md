# Original π0 on LIBERO: Reproduction and Evaluation

English | [简体中文](README.zh-CN.md)

This project builds a reproducible, safety-bounded evaluation workflow for the
original Physical Intelligence `pi0_libero` policy in the LIBERO simulator.
It runs directly inside an Ubuntu LXC, without nested Docker, using separate
Python environments for the OpenPI policy server and the LIBERO/MuJoCo client.

The project intentionally selects the historical original π0 checkpoint. It
does not use the current default π0.5-LIBERO policy.

## Current result

| Stage | Scope | Result | Meaning |
| --- | --- | --- | --- |
| Closed-loop smoke test | `libero_spatial`, task 0, initial state 0 | 1/1 success | Confirms that the full simulator-to-policy loop works |
| Mapping calibration | JAX, PyTorch visibility, MuJoCo EGL | Passed | Confirms single-GPU isolation and rendering |
| First bounded evaluation | `libero_spatial`, task 0, initial states 0-9, seed 7 | 9/10 success | First 10-state milestone |
| Complete task-state evaluation | `libero_spatial`, task 0, initial states 0-49, seed 7 | 48/50 success (96.0%) | All 50 fixed states for one task, not a full LIBERO benchmark |
| Complete four-suite evaluation | 4 suites, 40 tasks, 50 states per task, seed 7 | 1845/2000 success (92.25%) | Complete four-suite result for the pinned version and seed |

| Suite | Success / Episodes | Success rate |
| --- | ---: | ---: |
| LIBERO-Spatial | 482 / 500 | 96.40% |
| LIBERO-Object | 489 / 500 | 97.80% |
| LIBERO-Goal | 471 / 500 | 94.20% |
| LIBERO-10 | 403 / 500 | 80.60% |

All 155 failed episodes reached their suite-specific control-step limit, with
no Python, MuJoCo, WebSocket or evaluator exception. The complete report and
per-task evidence are available in
[`artifacts/libero-benchmark/pi0_libero_official4_seed7/run_report.md`](artifacts/libero-benchmark/pi0_libero_official4_seed7/run_report.md).

## Fixed source and model identity

- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- LIBERO submodule commit: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Policy configuration: `pi0_libero`
- Checkpoint: `openpi-assets/checkpoints/pi0_libero`
- LIBERO normalization statistics SHA-256:
  `dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1`

Using only `serve_policy.py --env LIBERO` is deliberately avoided because the
default policy in the pinned OpenPI checkout may not be the original π0 model.

## System architecture

```text
Ubuntu LXC
├── OpenPI environment (Python 3.11, JAX/CUDA)
│   └── original pi0_libero policy server on localhost:8000
├── LIBERO environment (Python 3.8, MuJoCo/EGL)
│   └── bounded sequential evaluator
└── one explicitly selected physical GPU
    ├── CUDA_VISIBLE_DEVICES isolation
    ├── JAX preallocation disabled
    └── utilization/VRAM pause guard
```

The client renders observations with MuJoCo EGL, sends images, robot state and
the language instruction to the WebSocket policy server, receives a 50-step
action chunk, executes the first five actions, and replans until success or the
control-step limit.

## Shared-GPU protection

Every GPU workload is launched through
[`tools/gpu_utilization_guard.py`](tools/gpu_utilization_guard.py). The guard
creates a new process group and signals only that group.

- Pause at 95% total GPU utilization or 15% remaining VRAM.
- Resume only at or below 85% utilization and at or above 20% free VRAM for
  five consecutive samples.
- Stop the task at the 10% free-VRAM emergency floor.
- Pause conservatively after three consecutive monitoring failures.
- Never discover, pause or terminate another user's process.

The remaining 1,950 episodes ran on physical GPU 0. Across 18,103 guard samples,
peak utilization was 54%, peak memory use was 9,987 MiB, and at least 89.80%
VRAM remained free. The guard recorded no pause, resume, monitoring error or
emergency event, and the workload exited with code 0.

## Repository layout

```text
tools/
  eval_libero_bounded.py              bounded evaluator and evidence writer
  gpu_utilization_guard.py            shared-GPU pause/resume guard
  test_gpu_utilization_guard.py       guard behavior tests
  verify_gpu_mapping.sh               JAX/PyTorch/EGL mapping check
  probe_pi0_libero_checkpoint.sh      bounded checkpoint-load readiness probe
  probe_pi0_libero_inference.sh       bounded one-request inference probe
  run_gpu_guarded_supervisor.sh       exact guard wrapper and PID cleanup
  run_pi0_libero_batch_workload.sh    original π0 server + 10-state workload
  run_pi0_libero_remaining_workload.sh resumable 39-task orchestrator
  summarize_pi0_libero_benchmark.py   2,000-episode validation and reporting
config/
  10_nvidia.json                      task-scoped NVIDIA EGL vendor file
docs/
  algorithm_overview.md               beginner-level model and control overview
  deployment_runbook.md               repeatable no-Docker LXC deployment guide
artifacts/
  libero-smoke/                       one-episode functional evidence
  libero-calibration/                 mapping and capacity evidence
  libero-eval/                        staged 10-state and 50-state task evidence
  libero-benchmark/                   four-suite summary, task table and failures
```

Raw server logs, process IDs, machine-specific run configuration and dense
action traces remain local and are excluded from Git. Complete four-suite videos
and per-episode raw evidence remain under the local `raw/` directory; reports,
summaries, per-task results, failure lists and earlier representative videos are
kept as reproducible Git evidence.

## Learning and deployment notes

- [Algorithm overview](docs/algorithm_overview.md) explains the inputs,
  checkpoint transforms, action expert, flow-matching intuition, 50-step
  action chunk and five-step closed-loop replanning without a line-by-line
  source walkthrough.
- [Deployment runbook](docs/deployment_runbook.md) covers the fresh-SSH
  preflight, pinned identities, dual Python environments, CPU-only dry run,
  shared-GPU guard, staged single-episode launch, resume and cleanup.
- [Failure analysis](docs/failure_analysis.md) uses a success control and
  stratified task-8 failures to separate grasp, placement, recovery and stage
  transition evidence from hypotheses that still require validation.

## Evaluation workflow

1. Pin the OpenPI checkout and the LIBERO gitlink commit.
2. Keep one OpenPI environment and one separate LIBERO environment.
3. Restore the original `pi0_libero` checkpoint and its normalization stats.
4. Sample CPU, RAM and both GPUs for at least 30 seconds.
5. Select one GPU, verify CUDA/JAX/PyTorch/EGL mapping and disable JAX
   preallocation.
6. Run a one-episode smoke test.
7. Run the bounded evaluator for 10 fixed initial states.
8. Inspect success and failure videos before expanding the task to 50 states.
9. Use the resumable sequential orchestrator for the other 39 tasks and 1,950 episodes.
10. Verify that all 40 tasks contain exactly states 0-49, then generate the report.

Useful CPU-only checks:

```bash
PYTHONPATH=tools python -m unittest tools/test_gpu_utilization_guard.py
python -m py_compile tools/eval_libero_bounded.py tools/gpu_utilization_guard.py
bash -n tools/verify_gpu_mapping.sh
bash -n tools/run_gpu_guarded_supervisor.sh
bash -n tools/run_pi0_libero_batch_workload.sh
bash -n tools/run_pi0_libero_remaining_workload.sh
python tools/summarize_pi0_libero_benchmark.py
```

GPU commands are intentionally not presented as a one-line quick start: the
baseline sampling, physical-device mapping and guard thresholds must be checked
for the current shared server before every stage.

## Evidence

- [One-episode smoke report](artifacts/libero-smoke/pi0_libero_spatial_task0_episode0_seed7/run_report.md)
- [10-state evaluation report](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/run_report.md)
- [Complete four-suite report](artifacts/libero-benchmark/pi0_libero_official4_seed7/run_report.md)
- [Per-task results](artifacts/libero-benchmark/pi0_libero_official4_seed7/task_results.csv)
- [Failed-episode list](artifacts/libero-benchmark/pi0_libero_official4_seed7/failure_cases.csv)
- [Representative success video](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/evaluation/task_00_init_00_success.mp4)
- [Representative failure video](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/evaluation/task_00_init_01_failure.mp4)

## Limitations and next steps

- The evaluation covers all four suites but uses one fixed seed, not a multi-seed
  confidence interval.
- Source, checkpoint and environment versions must accompany any comparison;
  an average alone does not establish protocol equivalence with another result.
- All 155 structured termination labels are `max_control_steps`. Six LIBERO-10
  task-8 failures now have manual behavior annotations, but they do not replace
  a complete classification of all 37 task-8 failures. Full video annotation
  remains an optional research extension; it is not required to reproduce the
  deployment or the reported benchmark.
