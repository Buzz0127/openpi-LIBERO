# 原始 π0-LIBERO 四 Suite 完整评测报告

## 结论

在固定 OpenPI、LIBERO 和原始 `pi0_libero` checkpoint 身份下，本项目完成
4 个标准 suite、40 个 task、每个 task 50 个固定初始状态，共 2,000 个闭环
episode。总成功率为 **1845/2000（92.25%）**。

| Suite | 成功 / Episodes | 成功率 |
| --- | ---: | ---: |
| LIBERO-Spatial | 482 / 500 | 96.40% |
| LIBERO-Object | 489 / 500 | 97.80% |
| LIBERO-Goal | 471 / 500 | 94.20% |
| LIBERO-10 | 403 / 500 | 80.60% |
| **四 suite 总计** | **1845 / 2000** | **92.25%** |

这是一组固定版本、固定 seed 7 的完整四-suite评测结果。它不应被表述为多 seed
统计，也不用于声称逐项等同于论文作者的硬件、软件或随机性条件。

## 固定身份与协议

- OpenPI commit：`15a9616a00943ada6c20a0f158e3adb39df2ccac`
- LIBERO commit：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Policy：`pi0_libero`
- Checkpoint：`openpi-assets/checkpoints/pi0_libero`
- Norm stats SHA-256：`dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1`
- Evaluator SHA-256：`cfce604677a480534397ddad25358c7a95c982de57287c8df5042ee28d9c4312`
- Seed：7
- 每个 task：initial states 0–49
- 模型输出 50 步动作块，客户端执行前 5 步后重新观察与规划

## 逐 Task 结果

| Suite | Task | 成功 / 50 | 成功率 | 失败 states |
| --- | ---: | ---: | ---: | --- |
| LIBERO-Spatial | 0 | 48 / 50 | 96.00% | 1, 35 |
| LIBERO-Spatial | 1 | 50 / 50 | 100.00% | — |
| LIBERO-Spatial | 2 | 50 / 50 | 100.00% | — |
| LIBERO-Spatial | 3 | 50 / 50 | 100.00% | — |
| LIBERO-Spatial | 4 | 43 / 50 | 86.00% | 2, 19, 20, 23, 24, 25, 41 |
| LIBERO-Spatial | 5 | 48 / 50 | 96.00% | 36, 42 |
| LIBERO-Spatial | 6 | 49 / 50 | 98.00% | 8 |
| LIBERO-Spatial | 7 | 48 / 50 | 96.00% | 17, 18 |
| LIBERO-Spatial | 8 | 47 / 50 | 94.00% | 31, 34, 44 |
| LIBERO-Spatial | 9 | 49 / 50 | 98.00% | 0 |
| LIBERO-Object | 0 | 45 / 50 | 90.00% | 0, 6, 24, 38, 43 |
| LIBERO-Object | 1 | 49 / 50 | 98.00% | 43 |
| LIBERO-Object | 2 | 50 / 50 | 100.00% | — |
| LIBERO-Object | 3 | 50 / 50 | 100.00% | — |
| LIBERO-Object | 4 | 48 / 50 | 96.00% | 10, 20 |
| LIBERO-Object | 5 | 49 / 50 | 98.00% | 39 |
| LIBERO-Object | 6 | 50 / 50 | 100.00% | — |
| LIBERO-Object | 7 | 48 / 50 | 96.00% | 31, 45 |
| LIBERO-Object | 8 | 50 / 50 | 100.00% | — |
| LIBERO-Object | 9 | 50 / 50 | 100.00% | — |
| LIBERO-Goal | 0 | 49 / 50 | 98.00% | 19 |
| LIBERO-Goal | 1 | 50 / 50 | 100.00% | — |
| LIBERO-Goal | 2 | 48 / 50 | 96.00% | 36, 42 |
| LIBERO-Goal | 3 | 46 / 50 | 92.00% | 13, 42, 44, 45 |
| LIBERO-Goal | 4 | 50 / 50 | 100.00% | — |
| LIBERO-Goal | 5 | 47 / 50 | 94.00% | 8, 24, 26 |
| LIBERO-Goal | 6 | 50 / 50 | 100.00% | — |
| LIBERO-Goal | 7 | 49 / 50 | 98.00% | 21 |
| LIBERO-Goal | 8 | 47 / 50 | 94.00% | 22, 37, 38 |
| LIBERO-Goal | 9 | 35 / 50 | 70.00% | 2, 4, 7, 9, 11, 16, 19, 20, 26, 28, 32, 36, 40, 44, 49 |
| LIBERO-10 | 0 | 43 / 50 | 86.00% | 15, 18, 23, 25, 28, 33, 39 |
| LIBERO-10 | 1 | 45 / 50 | 90.00% | 17, 33, 39, 41, 46 |
| LIBERO-10 | 2 | 39 / 50 | 78.00% | 1, 9, 11, 12, 16, 17, 20, 26, 31, 37, 49 |
| LIBERO-10 | 3 | 49 / 50 | 98.00% | 19 |
| LIBERO-10 | 4 | 43 / 50 | 86.00% | 1, 10, 11, 16, 18, 22, 49 |
| LIBERO-10 | 5 | 46 / 50 | 92.00% | 23, 29, 33, 44 |
| LIBERO-10 | 6 | 39 / 50 | 78.00% | 11, 13, 14, 16, 21, 23, 27, 29, 43, 46, 47 |
| LIBERO-10 | 7 | 46 / 50 | 92.00% | 7, 16, 23, 44 |
| LIBERO-10 | 8 | 13 / 50 | 26.00% | 0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 16, 17, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 46, 48 |
| LIBERO-10 | 9 | 40 / 50 | 80.00% | 0, 1, 2, 8, 11, 21, 25, 35, 37, 44 |

机器可读明细见 [`task_results.csv`](task_results.csv)。

## 失败与完整性

- 失败 episode：155
- `max_control_steps`：155
- Python、MuJoCo、WebSocket 或评测器系统异常：0
- 40 个 `(suite, task)` 组合均包含且仅包含 initial states 0–49
- 没有重复或缺失 episode

全部失败条目及本地视频路径见 [`failure_cases.csv`](failure_cases.csv)。达到控制步上限
表示 episode 没有及时满足 LIBERO 成功谓词；它属于任务失败，不是程序崩溃。进一步的
行为分类需要逐个检查失败视频，不能只根据 `max_control_steps` 推断原因。

## GPU 与运行保护

- 物理 GPU：0；`CUDA_VISIBLE_DEVICES=0`
- JAX 默认显存预分配：关闭
- 保护器采样：18,103 次
- 峰值 GPU 利用率：54%
- 峰值显存：9987 MiB
- 最低剩余显存：89.80%
- 暂停 / 恢复 / 监控错误 / 显存紧急事件：0 / 0 / 0 / 0
- 保护器子进程退出码：0
- 保护器记录时长：5 小时 39 分 43 秒

运行结束后 Policy Server、评测器和保护器均退出，端口 8000 与 GPU 显存均释放。

## 证据与复现边界

可提交证据包括本报告、[`benchmark_summary.json`](benchmark_summary.json)、
[`task_results.csv`](task_results.csv) 和 [`failure_cases.csv`](failure_cases.csv)。完整视频、
逐 episode JSON、动作轨迹、GPU 采样和服务器日志保存在本地 `raw/`，因体积与机器信息
不提交 Git。`raw/` 的结果文件共 6,007 个，
控制文件共 68 个。

本结果最明显的短板是 LIBERO-10（403/500，
80.60%）。下一步应按 task 和失败视频进行行为级
归类，而不是通过提高控制步上限重新定义本次固定协议。
