# 原始 π0-LIBERO Spatial Task 0：50 状态评测报告

## 1. 结果

在固定的 `libero_spatial` task 0、seed 7 下，原始 `pi0_libero` 对全部50个
官方固定初始状态取得：

```text
48 / 50 = 96.0%
```

失败状态为 initial states 1 和35。该结果覆盖一个 task 的完整状态集合，不能
表述为完整 LIBERO Spatial suite 或四个 suite 的 benchmark 成绩。

## 2. 固定身份

| 项目 | 值 |
| --- | --- |
| OpenPI commit | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| LIBERO commit | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| Policy | 原始 `pi0_libero` |
| norm stats SHA-256 | `dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1` |
| replan steps | 5 |
| max control steps | 220 |

## 3. 两段有界运行

为避免重复计算，50个状态由两段身份一致且互不重叠的运行组成：

| Initial states | 成功 | 失败 | 证据 |
| --- | ---: | ---: | --- |
| 0–9 | 9 | 1 | [`results.jsonl`](../pi0_libero_spatial_task0_init0-9_seed7/evaluation/results.jsonl) |
| 10–49 | 39 | 1 | [`results.jsonl`](../pi0_libero_spatial_task0_init10-49_seed7/evaluation/results.jsonl) |

合计4092个控制步、838次策略请求，平均每个 episode 81.84个控制步。

## 4. 失败分析

两个失败 episode 都没有 Python、WebSocket、MuJoCo 或 GPU 异常：

| Initial state | 失败原因 | 控制步 | 策略请求 |
| --- | --- | ---: | ---: |
| 1 | `max_control_steps` | 220 | 44 |
| 35 | `max_control_steps` | 220 | 44 |

state 35 的末段视频显示机械臂已经把碗移动到盘子区域，但随后持续在盘子上方
小范围调整，直到步数上限仍未形成满足 `On(bowl, plate)` 的稳定终态。证据支持
把它归类为末端放置/释放失败或成功谓词未满足，而不是部署故障；仅凭第三人称
视频不能进一步断定接触谓词内部的具体失败项。

- state 1 失败视频：[`task_00_init_01_failure.mp4`](../pi0_libero_spatial_task0_init0-9_seed7/evaluation/task_00_init_01_failure.mp4)
- state 35 失败视频：[`task_00_init_35_failure.mp4`](../pi0_libero_spatial_task0_init10-49_seed7/evaluation/task_00_init_35_failure.mp4)

## 5. 新增 10–49 阶段资源证据

- 动态选择物理 GPU 0；启动前31个样本均为0% utilization；
- 最低启动前剩余显存为97,233 MiB（99.33%）；
- 运行期峰值显存为9,603 MiB；evaluator 内部采样峰值利用率为41%，更外层
  保护器的289个样本捕获到55%，容量报告采用更保守的55%；
- 保护器无暂停、恢复、监控错误或紧急停止；
- 没有 OOM、ECC 或 Xid；
- evaluator 与 guard 退出码均为0；
- 结束后端口8000释放，GPU 0 恢复到16 MiB。

机器上的其他任务负载不计入模型能力结果。共享 GPU 保护只管理本项目创建的
进程组，没有操作其他用户进程。

## 6. 结果边界

这份报告可以用于说明：原始 π0 在固定 task 0 的50个官方初始状态上取得
96.0% 成功率，并且无 Docker LXC 部署、EGL、WebSocket 闭环、单 GPU 隔离和
有界评测链路可以重复运行。

它不能用于声称复现了完整 LIBERO Spatial 或官方四 suite 平均分。下一步若要
扩大模型能力覆盖，应评测新的 task，而不是再次运行 states 0–49。
