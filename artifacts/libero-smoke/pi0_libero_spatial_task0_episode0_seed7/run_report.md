# 原始 π0-LIBERO 单 Episode 闭环复现实验记录

## 1. 实验结论

本次实验在实验室 Ubuntu LXC 中，使用官方 OpenPI 源码和历史原始 `pi0_libero` checkpoint，完成了 LIBERO Spatial task 0、固定初始状态 0 的一次闭环推理。

- 工程闭环：通过
- LIBERO 任务判定：成功（`done=True`）
- 烟雾测试结果：`1/1` 成功
- 结果边界：这只是单 episode 功能验证，不是有统计意义的模型成功率

## 2. 固定版本与运行环境

| 项目 | 固定值 |
| --- | --- |
| OpenPI commit | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| OpenPI branch | `main` |
| LIBERO submodule commit | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| OpenPI Python | `3.11.15` |
| LIBERO Python | `3.8.20` |
| JAX | `0.5.3` |
| Orbax Checkpoint | `0.11.13` |
| MuJoCo | `3.2.3` |
| robosuite | `1.4.1` |
| NumPy（LIBERO 环境） | `1.22.4` |

OpenPI 模型环境与 LIBERO 仿真环境相互隔离，分别位于：

- `$HOME/projects/openpi/.venv`
- `$HOME/projects/openpi/examples/libero/.venv`

## 3. 模型与 checkpoint

- 配置名：`pi0_libero`
- 模型类型：原始 `PI0`
- `pi05=False`
- 内部动作窗口：50
- 内部动作维度：32
- LIBERO 输出动作维度：7
- checkpoint：`$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_libero`
- checkpoint 文件数：19
- checkpoint 总字节数：`12,014,131,888`
- `norm_stats.json` SHA-256：`dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1`

服务端必须显式选择 `pi0_libero`。不能只运行 `serve_policy.py --env LIBERO`，因为当前源码的 LIBERO 默认项指向 `pi05_libero`。

## 4. GPU 隔离与安全边界

- 选定物理 GPU：1
- 隔离变量：`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=1`
- JAX 进程内映射：物理 GPU 1 被重编号为 `CudaDevice(id=0)`
- GPU 0：存在其他任务，视为繁忙，本实验未让进程看到或使用它
- JAX 显存策略：`XLA_PYTHON_CLIENT_PREALLOCATE=false`
- MuJoCo：使用 EGL 离屏渲染，并绑定经过验证的 GPU 1 映射

Policy Server 空载常驻时观测到 GPU 1 显存约 `9,351 MiB`。episode 结束后、服务仍存活时观测到约 `9,427 MiB`。本次没有同步采集全过程峰值，因此不能把这两个观测值表述为闭环推理峰值。

实验结束后：

- GPU 1 恢复到 `P8`、`0%`、`577 MiB`
- 端口 8000 无监听
- Policy Server 与 LIBERO 客户端均无残留进程

## 5. Episode 配置

| 参数 | 值 |
| --- | --- |
| Suite | `libero_spatial` |
| Task ID | 0 |
| Episode / initial-state index | 0 |
| Seed | 7 |
| Task count | 1 |
| Trials per task | 1 |
| Observation resize | 224 × 224 |
| Replan steps | 5 |
| Stabilization steps | 10 |
| Maximum control steps | 220 |

任务语言指令：

> pick up the black bowl between the plate and the ramekin and place it on the plate

BDDL 初始条件中的目标对象是 `akita_black_bowl_1`，其初始位置是 `main_table_between_plate_ramekin_region`。精确成功条件为：

```text
(On akita_black_bowl_1 plate_1)
```

## 6. 闭环调用链

```text
MuJoCo/EGL 渲染观测
→ 主视角与腕部相机图像旋转、缩放至 224×224
→ 末端位置、轴角姿态和夹爪状态拼接为 8 维状态
→ WebSocket 发送图像、状态与语言指令
→ LiberoInputs、归一化与原始 π0 flow-matching 采样
→ 模型产生 50×32 动作块
→ 反归一化并裁剪为 50×7 LIBERO 动作
→ 每次只执行前 5 步后重新规划
→ env.step()
→ LIBERO 检测 BDDL 成功谓词
→ 保存 MP4
```

## 7. 直接观测结果

- 客户端正常退出：`exit_code=0`
- 单 episode 包装器完成：`single_closed_loop_episode=completed`
- LIBERO 成功：视频名后缀为 `_success.mp4`，对应官方代码中的 `done=True`
- 保存控制帧：73
- 视频帧率：10 FPS
- 视频估算时长：7.3 秒
- 视频分辨率：224 × 224 RGB
- 视频文件大小：37,931 字节
- 输出目录大小（加入联系表后）：约 226 KB
- 客户端外层进度显示：约 21.05 秒

根据官方代码“每 5 个控制步重新规划”，73 个保存控制帧对应约 15 次策略请求。这是由控制帧数和源码逻辑推导的估计值，并非本次直接记录的请求计数。

## 8. 视频证据

- 视频：`rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_success.mp4`
- 视频 SHA-256：`e56571dc0cad8cb32dd238014cdb5c45e79b84b1d03a6181cfbf5b57e08fd8f4`
- 联系表：`contact_sheet_frames_00_14_28_42_56_72.png`
- 联系表 SHA-256：`c454177dec5081f72bc3f06098811686afc2c1d22bdef5eecabb1d124d6c4aed`

人工检查可见：机器人选择了位于盘子与 ramekin 之间的目标黑碗，完成接近、抓取、搬运和放置，最终黑碗位于盘子区域。未发现抓取另一个黑碗、明显穿模或画面损坏。

## 9. 警告与限制

1. 本次结果只有 `1/1`，不能写成模型在 LIBERO Spatial 上达到 100% 成功率。
2. 本次没有记录逐次请求延迟、全过程 GPU 峰值、动作轨迹或环境状态日志。
3. 旧版 robosuite/PyOpenGL 在 Python 退出析构时报告 `EGL_NOT_INITIALIZED`；该异常被标记为 `Exception ignored`，episode 已完成且进程退出码为 0，但后续批量评测前应显式关闭环境并确认警告消失。
4. robosuite 提示没有私有 `macros.py`，当前使用默认设置。
5. LIBERO 训练数据集目录不存在；本次仿真评测只使用仓库内 BDDL 和固定初始状态，因此不受影响。
6. 仅导入 MuJoCo 查询版本且未设置任务专用 EGL 环境变量时会打印 `/dev/dri` 权限警告；实际 EGL 渲染使用任务专用 NVIDIA vendor 配置并已通过。
7. 本次使用一次性内存包装器把官方 suite 限制为 task 0；没有修改 OpenPI 源码。官方入口若只设置 `num_trials_per_task=1`，仍会遍历 suite 的全部 10 个任务。

## 10. 当前可以诚实表述的项目成果

> 在共享 NVIDIA GPU 的 Ubuntu LXC 中，以双 Python 环境隔离方式部署 OpenPI 与 LIBERO；固定 OpenPI/LIBERO commit，显式恢复历史原始 `pi0_libero` checkpoint，完成 JAX CUDA、MuJoCo EGL、WebSocket Policy Server 和单 episode 闭环验证，并保留版本、哈希、视频及资源释放证据。

暂时不要表述为“完成完整 LIBERO benchmark”或“复现官方平均成功率”。下一阶段至少应扩展到同一 task 的多个固定初始状态，并记录逐 episode 结果、推理延迟、失败类型和峰值显存。
