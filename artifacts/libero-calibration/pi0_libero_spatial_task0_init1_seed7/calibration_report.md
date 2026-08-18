# 原始 pi0_libero 单 Episode GPU 校准报告

## 1. 结论

本次在物理 GPU 1 上完成了一个严格有界的 calibration episode：

- Suite：`libero_spatial`
- Task ID：`0`
- Initial state：`1`
- Seed：`7`
- Policy：原始 `pi0_libero`
- 结果：成功
- 控制步：`75`
- 策略请求：`15`
- Episode 用时：`35.434 s`

本次只用于测量单 episode 阶段的显存和计算增量，不授权直接扩大到更多 initial state。校准后的正常门禁中，显存门通过、计算门失败，因此没有继续运行。

## 2. 固定身份与设备映射

| 项目 | 值 |
| --- | --- |
| OpenPI commit | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| LIBERO commit | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| Evaluator SHA-256 | `cfce604677a480534397ddad25358c7a95c982de57287c8df5042ee28d9c4312` |
| Checkpoint | `pi0_libero`，19 files，12,014,131,888 bytes |
| norm_stats SHA-256 | `dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1` |
| 物理 GPU | `1` |
| `CUDA_VISIBLE_DEVICES` | `1` |
| JAX 进程内设备 | `cuda:0` |
| MuJoCo EGL device | `1` |
| JAX 预分配 | `XLA_PYTHON_CLIENT_PREALLOCATE=false` |

设备映射测试确认 JAX 和 PyTorch 都只看到一个进程内设备 0。PyTorch wheel 报告未包含 Blackwell `sm_120` 内核，但本次模型推理由 JAX 执行，仿真渲染由 MuJoCo EGL 执行。

## 3. 校准前 30 秒基线

GPU 1 共记录 31 个逐秒样本：

| 指标 | 结果 |
| --- | ---: |
| GPU-Util minimum | `12%` |
| GPU-Util maximum | `54%` |
| GPU-Util mean | `40.419%` |
| memory.used | `42,619 MiB`，全程稳定 |
| minimum free VRAM | `54,629 MiB` |

CPU load 约 `2.4`，约 `250 GiB` RAM 可用；端口 8000、目标目录和任务自有进程均为空。

## 4. 校准监控结果

独立监控从设备映射前持续到 Policy Server、评测器全部退出，共记录 398 个样本：

| 指标 | 结果 |
| --- | ---: |
| runtime peak GPU-Util | `93%` |
| runtime peak memory.used | `52,208 MiB` |
| runtime minimum free VRAM | `45,041 MiB` |
| evaluator peak power | `471.76 W` |
| evaluator peak temperature | `84 C` |
| emergency threshold triggered | `false` |

运行中没有出现一次 100%、连续两次不低于 95%、free VRAM 不高于物理容量 10%、预计 10 秒内耗尽、OOM、ECC 或 Xid 告警。

## 5. E：保守显存需求

相对于任务启动前稳定基线：

```text
observed task memory increment
= 52,208 - 42,619
= 9,589 MiB
```

增加 20% 安全余量：

```text
9,589 * 1.2 = 11,506.8 MiB
```

向上取整后的阶段预算：

```text
E = 12 GiB = 12,288 MiB
```

## 6. C：保守新增计算利用率

评测器在 episode 开始前记录的紧邻基线为 `52%`，运行峰值为 `93%`：

```text
direct observed increment = 93 - 52 = 41 percentage points
direct increment with 20% margin = 49.2
rounded direct C = 50 percentage points
```

由于其他用户负载在校准前 30 秒内于 `12%` 至 `54%` 之间波动，单一紧邻快照可能低估不确定性。本报告采用 30 秒基线均值计算更保守的运行预算：

```text
93 - 40.419 = 52.581 percentage points
52.581 * 1.2 = 63.0972
operational C = 64 percentage points
```

该 `C=64` 用于后续门禁；它是共享 GPU 上的保守工程估计，不声称能够精确分离其他用户每一秒的计算贡献。

## 7. 校准后正常门禁

所有任务进程退出后重新记录 31 个逐秒样本：

| 指标 | 结果 |
| --- | ---: |
| GPU-Util minimum | `8%` |
| GPU-Util maximum `U_max` | `65%` |
| GPU-Util mean | `43.645%` |
| minimum free VRAM `F_min` | `54,629 MiB`，全程稳定 |

显存门：

```text
1.8E = 21.6 GiB
F_min = 53.35 GiB
53.35 >= 21.6  -> PASS
```

计算门：

```text
U_max + C = 65 + 64 = 129%
129 <= 90  -> FAIL
```

即使使用较低的直接估计 `C=50`，`65+50=115%` 仍然失败。因此不得在该基线下扩大到正式阶段或 10 个状态。

## 8. Episode 结果和延迟

- 成功：`true`
- 控制步：`75`
- 保存帧：`75`
- 策略请求：`15`
- 首次 client request：`22,257.4 ms`
- 首次 policy infer：`20,055.1 ms`，包含首次 JAX 编译
- client request p50：`113.289 ms`
- 视频：`38,721 bytes`
- 动作轨迹：`15,424 bytes`
- 完整评测输出目录：约 `84 KiB`

视频 SHA-256：

```text
e8a672e51304b7bae20eb26dba461ce83e1b6743bc1fd6c918f25eb0497ff90c
```

## 9. 进程退出与异常记录

- Evaluator PID：`34081`，退出码 `0`
- Policy Server PID：`30087`，完成后经命令行核实并发送 `SIGTERM`，退出码 `143`
- Monitor PID：`26730`，正常退出，`emergency_file=no`
- 最终端口 8000 无监听
- 最终无 `serve_policy.py`、`eval_libero_bounded.py` 或 LIBERO client 残留进程
- GPU 1 显存恢复至任务前 `42,619 MiB`

第一次 Policy Server 启动因顶层 `--port` 参数放在子命令之后而在参数解析阶段退出，退出码为 2；当时尚未加载 checkpoint 或新增 GPU 显存。随后修正参数顺序并成功完成校准。该记录保留在 `calibration/server.log`。

LIBERO 的 datasets 目录缺失警告不影响本次使用仓库 BDDL 和固定初始状态的仿真。robosuite private macros 与 `OpenGL_accelerate` 警告亦未阻止成功回合。

## 10. 证据文件

- 成功视频：`evaluation/task_00_init_01_success.mp4`
- Episode 结果：`evaluation/task_00_init_01_result.json`
- 运行汇总：`evaluation/run_summary.json`
- 动作轨迹：`evaluation/task_00_init_01_actions.json`
- Evaluator GPU 采样：`evaluation/gpu_samples.jsonl`
- 校准前基线：`calibration/baseline_gpu.csv`
- 完整运行期监控：`calibration/runtime_gpu.csv`
- 校准后基线：`calibration/post_calibration_baseline.csv`
- 服务与评测日志：`calibration/server.log`、`calibration/evaluator.log`

