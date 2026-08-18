# 原始 π0 在 LIBERO 上的复现与评测

[English](README.md) | 简体中文

本项目为 Physical Intelligence 的原始 `pi0_libero` 策略建立了一套可复现、
具备共享服务器安全边界的 LIBERO 仿真评测流程。项目直接运行在 Ubuntu LXC
内部，不依赖嵌套 Docker；OpenPI Policy Server 与 LIBERO/MuJoCo 客户端分别
使用独立的 Python 环境。

项目显式选择历史公开的原始 π0 checkpoint，不使用当前默认的
π0.5-LIBERO 策略。

## 当前结果

| 阶段 | 范围 | 结果 | 能够说明什么 |
| --- | --- | --- | --- |
| 闭环烟雾测试 | `libero_spatial`、task 0、initial state 0 | 1/1 成功 | 证明仿真器到策略服务器的完整闭环可以运行 |
| 设备映射校准 | JAX、PyTorch 可见性、MuJoCo EGL | 通过 | 证明单 GPU 隔离与离屏渲染正确 |
| 有界评测 | `libero_spatial`、task 0、initial states 0-9、seed 7 | 9/10 成功 | 单个任务的 10 状态结果，不是完整 LIBERO benchmark |

唯一失败的是 initial state 1：episode 达到设定的 220 个控制步上限，仍未满足
LIBERO 成功谓词。完整的 10 状态实验记录位于
[`artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/run_report.md`](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/run_report.md)。

## 固定的源码与模型身份

- OpenPI commit：`15a9616a00943ada6c20a0f158e3adb39df2ccac`
- LIBERO submodule commit：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Policy 配置：`pi0_libero`
- Checkpoint：`openpi-assets/checkpoints/pi0_libero`
- LIBERO 归一化统计 SHA-256：
  `dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1`

项目不会只运行 `serve_policy.py --env LIBERO`，因为固定版本中的默认 LIBERO
策略不一定是原始 π0。服务端必须显式指定 `pi0_libero` 配置与 checkpoint。

## 系统架构

```text
Ubuntu LXC
├── OpenPI 环境（Python 3.11、JAX/CUDA）
│   └── 原始 pi0_libero Policy Server，监听 localhost:8000
├── LIBERO 环境（Python 3.8、MuJoCo/EGL）
│   └── 有界、顺序执行的评测客户端
└── 一张经过明确选择的物理 GPU
    ├── CUDA_VISIBLE_DEVICES 隔离
    ├── 关闭 JAX 默认显存预分配
    └── GPU 利用率与剩余显存保护器
```

客户端通过 MuJoCo EGL 渲染观测，把图像、机器人状态和语言指令发送给
WebSocket Policy Server。模型返回一个 50 步动作块；客户端只执行前 5 步，
然后重新请求策略，直到任务成功或达到控制步上限。

## 共享 GPU 保护

所有 GPU 工作负载都通过
[`tools/gpu_utilization_guard.py`](tools/gpu_utilization_guard.py) 启动。保护器
为任务创建独立进程组，并且只会向这个进程组发送信号。

- GPU 总利用率达到 95% 或剩余显存降至 15% 时暂停任务。
- 只有利用率不高于 85%、剩余显存不低于 20%，并连续满足 5 次采样时才恢复。
- 剩余显存达到 10% 紧急底线时终止本任务。
- 连续 3 次监控失败时保守暂停。
- 绝不查找、暂停或终止其他用户的进程。

10 状态评测中，所选 GPU 的采样峰值利用率为 92%，最低剩余显存为
37.10%。保护器没有发生暂停、监控错误或紧急停止。

## 仓库结构

```text
tools/
  eval_libero_bounded.py              有界评测器与证据记录
  gpu_utilization_guard.py            共享 GPU 暂停/恢复保护器
  test_gpu_utilization_guard.py       保护器行为测试
  verify_gpu_mapping.sh               JAX/PyTorch/EGL 映射检查
  run_gpu_guarded_supervisor.sh       保护器包装与精确 PID 清理
  run_pi0_libero_batch_workload.sh    原始 π0 服务与 10 状态任务
config/
  10_nvidia.json                      任务级 NVIDIA EGL vendor 配置
artifacts/
  libero-smoke/                       单 episode 功能证据
  libero-calibration/                 设备映射与容量校准证据
  libero-eval/                        10 状态结构化结果与视频
```

原始服务器日志、PID、机器相关运行配置和密集动作轨迹只保留在本机，并由
`.gitignore` 排除。实验报告、结果摘要、逐 episode 结果和小体积视频作为
可验证证据保留在 Git 中。

## 评测流程

1. 固定 OpenPI checkout 和 LIBERO gitlink commit。
2. 分别建立 OpenPI 环境与 LIBERO 环境。
3. 恢复原始 `pi0_libero` checkpoint 及其归一化统计。
4. 在每个 GPU 阶段前对 CPU、RAM 和两张 GPU 至少采样 30 秒。
5. 选择一张 GPU，验证 CUDA/JAX/PyTorch/EGL 映射，并关闭 JAX 显存预分配。
6. 完成一个 episode 的闭环烟雾测试。
7. 使用有界评测器运行 10 个固定初始状态。
8. 对比成功和失败视频，再决定是否扩展到 50 个状态。

不使用 GPU 的检查命令：

```bash
PYTHONPATH=tools python -m unittest tools/test_gpu_utilization_guard.py
python -m py_compile tools/eval_libero_bounded.py tools/gpu_utilization_guard.py
bash -n tools/verify_gpu_mapping.sh
bash -n tools/run_gpu_guarded_supervisor.sh
bash -n tools/run_pi0_libero_batch_workload.sh
```

项目没有提供一条可以无条件复制执行的 GPU 快速启动命令。共享服务器的当前
负载、物理设备映射和保护阈值必须在每个阶段开始前重新确认。

## 实验证据

- [单 episode 烟雾测试报告](artifacts/libero-smoke/pi0_libero_spatial_task0_episode0_seed7/run_report.md)
- [10 状态评测报告](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/run_report.md)
- [代表性成功视频](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/evaluation/task_00_init_00_success.mp4)
- [代表性失败视频](artifacts/libero-eval/pi0_libero_spatial_task0_init0-9_seed7/evaluation/task_00_init_01_failure.mp4)

## 局限与下一步

- 当前 90% 结果只覆盖一个任务和 10 个固定初始状态。
- 该结果不能直接与官方四个 suite 的平均成功率比较。
- 下一评测里程碑是在分析 initial state 1 失败原因后，对同一任务评测 50 个
  初始状态。
- 更完整的项目结果还需要扩展任务或 suite、建立失败类型、统计延迟，并与
  官方基线进行有边界的对比。

