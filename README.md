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
| Bounded evaluation | `libero_spatial`, task 0, initial states 0-9, seed 7 | 9/10 success | A 10-state result for one task, not a full LIBERO benchmark |

The failed episode was initial state 1, which reached the configured limit of
220 control steps without satisfying the LIBERO goal predicate. The complete
10-state report is available in
[`artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/run_report.md`](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/run_report.md).

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

For the 10-state evaluation, the selected GPU reached 92% sampled utilization
and retained at least 37.10% free VRAM. The guard recorded no pause, monitoring
error or emergency event.

## Repository layout

```text
tools/
  eval_libero_bounded.py              bounded evaluator and evidence writer
  gpu_utilization_guard.py            shared-GPU pause/resume guard
  test_gpu_utilization_guard.py       guard behavior tests
  verify_gpu_mapping.sh               JAX/PyTorch/EGL mapping check
  run_gpu_guarded_supervisor.sh       exact guard wrapper and PID cleanup
  run_pi0_libero_batch_workload.sh    original π0 server + 10-state workload
config/
  10_nvidia.json                      task-scoped NVIDIA EGL vendor file
artifacts/
  libero-smoke/                       one-episode functional evidence
  libero-calibration/                 mapping and capacity evidence
  libero-eval/                        10-state structured results and videos
```

Raw server logs, process IDs, machine-specific run configuration and dense
action traces remain local and are excluded from Git. Reports, result summaries,
per-episode outcomes and compact videos are kept as reproducible evidence.

## Evaluation workflow

1. Pin the OpenPI checkout and the LIBERO gitlink commit.
2. Keep one OpenPI environment and one separate LIBERO environment.
3. Restore the original `pi0_libero` checkpoint and its normalization stats.
4. Sample CPU, RAM and both GPUs for at least 30 seconds.
5. Select one GPU, verify CUDA/JAX/PyTorch/EGL mapping and disable JAX
   preallocation.
6. Run a one-episode smoke test.
7. Run the bounded evaluator for 10 fixed initial states.
8. Inspect success and failure videos before expanding to 50 states.

Useful CPU-only checks:

```bash
PYTHONPATH=tools python -m unittest tools/test_gpu_utilization_guard.py
python -m py_compile tools/eval_libero_bounded.py tools/gpu_utilization_guard.py
bash -n tools/verify_gpu_mapping.sh
bash -n tools/run_gpu_guarded_supervisor.sh
bash -n tools/run_pi0_libero_batch_workload.sh
```

GPU commands are intentionally not presented as a one-line quick start: the
baseline sampling, physical-device mapping and guard thresholds must be checked
for the current shared server before every stage.

## Evidence

- [One-episode smoke report](artifacts/libero-smoke/pi0_libero_spatial_task0_episode0_seed7/run_report.md)
- [10-state evaluation report](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/run_report.md)
- [Representative success video](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/evaluation/task_00_init_00_success.mp4)
- [Representative failure video](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/evaluation/task_00_init_01_failure.mp4)

## Limitations and next steps

- The 90% result covers one task and ten fixed initial states only.
- It cannot be compared directly with the official four-suite average.
- The next evaluation milestone is 50 initial states for the same task, after
  analyzing the state-1 failure.
- A full project result requires additional tasks/suites, failure taxonomy,
  latency statistics and a documented comparison with the official baseline.
