# 原始 π0-LIBERO 无 Docker LXC 部署 Runbook

本文记录从全新 SSH 会话到“可以启动单 episode 仿真”的可重复部署流程。所有 GPU 操作开始前都必须重新检查共享服务器状态；本文中的 GPU 编号不是固定选择。

## 1. 已验证的软件身份

| 项目 | 固定值 |
| --- | --- |
| OpenPI commit | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| LIBERO commit | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| Policy config | `pi0_libero` |
| Checkpoint | `openpi-assets/checkpoints/pi0_libero` |
| Norm stats SHA-256 | `dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1` |
| OpenPI Python | 3.11 |
| LIBERO Python | 3.8 |
| Guard SHA-256 | `ba9f7c74b843a61dc7cdf2f48b182390edb59022b5a2e76a489ce9d7048c4594` |

已验证 checkpoint 约 12 GB、OpenPI checkout 与两个环境约 8.6 GB，主要项目占用约 20.6 GB，低于 80 GiB 项目预算。

## 2. 目录与进程边界

```text
$HOME/projects/openpi/                 固定 OpenPI checkout
  .venv/                               Python 3.11 / JAX / Policy Server
  examples/libero/.venv/               Python 3.8 / LIBERO / MuJoCo
  third_party/libero/                  固定 LIBERO submodule
$HOME/.cache/openpi/.../pi0_libero/    单份 checkpoint
$HOME/projects/openpi-eval-tools/      服务器部署副本、控制目录与原始日志
  config/libero/config.yaml            稳定的任务级 LIBERO 路径配置
```

Policy Server 和 LIBERO evaluator 是两个进程，通过 `127.0.0.1:8000` WebSocket 通信。包装器只记录和清理自己创建的 PID；不得使用宽泛的 `pkill`，也不得操作其他用户进程。

## 3. 全新 SSH 会话预检

```bash
ssh <ssh-host-alias>
cd "$HOME/projects/openpi"

whoami
hostname
systemd-detect-virt
printf 'VIRTUAL_ENV=%s\n' "${VIRTUAL_ENV:-unset}"

git rev-parse HEAD
git -C third_party/libero rev-parse HEAD
.venv/bin/python --version
examples/libero/.venv/bin/python --version
```

检查磁盘、端口和当前用户自己的残留进程：

```bash
du -sh \
  "$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_libero" \
  "$HOME/projects/openpi" \
  "$HOME/projects/openpi-eval-tools"
df -h "$HOME"

ss -H -ltn 'sport = :8000'
pgrep -a -u "$USER" -f '[s]erve_policy.py|[e]val_libero_bounded.py|[g]pu_utilization_guard.py'
```

若 commit 不匹配、端口归属不明或发现无法确认的残留任务，应停在这里。

## 4. Checkpoint 与保护器身份

```bash
checkpoint="$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_libero"

sha256sum \
  "$checkpoint/assets/physical-intelligence/libero/norm_stats.json" \
  "$HOME/projects/openpi-eval-tools/gpu_utilization_guard.py"
```

保护器阈值为：

| 事件 | 行为 |
| --- | --- |
| 总 GPU utilization ≥95% | 只暂停保护器创建的任务进程组 |
| 剩余显存 ≤15% | 暂停任务进程组 |
| utilization ≤85% 且剩余显存 ≥20%，连续5次 | 恢复任务 |
| 监控连续失败3次 | 保守暂停 |
| 剩余显存 ≤10% | 紧急停止本任务 |

## 5. CPU-only 配置验证

```bash
env \
  CUDA_VISIBLE_DEVICES="" \
  JAX_PLATFORMS=cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python - <<'PY'
import jax
from openpi.training import config

cfg = config.get_config("pi0_libero")
print(jax.default_backend(), jax.devices())
print(type(cfg.model).__name__, cfg.model.model_type)
print(cfg.model.action_dim, cfg.model.action_horizon)
print(type(cfg.data).__name__, cfg.data.extra_delta_transform)
PY
```

预期为 CPU、`Pi0Config`、`ModelType.PI0`、`action_dim=32`、`action_horizon=50` 和 `extra_delta_transform=True`。

## 6. LIBERO 配置隔离

不要依赖首次导入时生成的 `$HOME/.libero/config.yaml`。使用稳定的任务级目录：

```bash
export LIBERO_CONFIG_PATH="$HOME/projects/openpi-eval-tools/config/libero"
```

`config.yaml` 至少要正确指向固定 checkout 中的：

```text
benchmark_root
bddl_files
init_states
assets
datasets
```

闭环推理需要 BDDL、初始状态和 assets，不需要下载示范训练 datasets；datasets 缺失会产生警告，但不阻塞推理评测。

## 7. 无 GPU evaluator dry-run

下面的 GPU 0 只是参数解析占位，不是实际选卡。`--dry-run` 在 EGL 映射和 GPU 查询之前退出。

```bash
dry_output="$HOME/tmp/openpi-evaluator-dry-run"
test ! -e "$dry_output"

env \
  CUDA_VISIBLE_DEVICES="" \
  JAX_PLATFORMS=cpu \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  PYTHONPATH="$PWD/third_party/libero:$PWD/packages/openpi-client/src" \
  examples/libero/.venv/bin/python \
  "$HOME/projects/openpi-eval-tools/eval_libero_bounded.py" \
  --suite libero_spatial \
  --task-id 0 \
  --initial-states 0:1 \
  --max-episodes 1 \
  --seed 7 \
  --host 127.0.0.1 --port 8000 \
  --server-wait-seconds 30 \
  --replan-steps 5 --max-control-steps 220 \
  --physical-gpu 0 --mujoco-egl-device-id 0 \
  --egl-vendor-file "$HOME/tmp/openpi-setup/egl-vendor/10_nvidia.json" \
  --gpu-sample-interval 1 \
  --max-gpu-memory-fraction 0.90 \
  --max-baseline-gpu-utilization 100 \
  --max-baseline-gpu-memory-fraction 1.0 \
  --policy-config pi0_libero \
  --openpi-root "$HOME/projects/openpi" \
  --checkpoint-dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_libero" \
  --expected-openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
  --expected-libero-commit f78abd68ee283de9f9be3c8f7e2a9ad60246e95c \
  --output-dir "$dry_output" \
  --max-output-bytes 1073741824 \
  --dry-run
```

通过标志是 `dry_run=ok`，且输出目录、端口和 GPU 进程均未创建。

## 8. GPU 启动边界

到这里为止均不需要 GPU。进入设备映射、模型加载或仿真前，必须重新：

1. 至少30秒、约1 Hz采样 CPU、RAM和每张 GPU 的 utilization、used/free/total VRAM。
2. 优先选择明确空闲的卡；若共享繁忙卡，当前剩余显存必须严格大于15%。
3. 估计并记录阶段峰值显存 `E` 与新增利用率 `C`；它们用于报告和异常比较，不是旧的启动硬门禁。
4. 只暴露一张物理 GPU，设置 `CUDA_VISIBLE_DEVICES`，并关闭 JAX 预分配。
5. 通过 `verify_gpu_mapping.sh` 分别核对 JAX、PyTorch 与 EGL 的物理映射。
6. 所有共享 GPU 工作必须由 `gpu_utilization_guard.py` 创建和监管。

不得沿用上一次运行的选卡结论。GPU 0 或 GPU 1 都可以，必须根据本次采样重新决定。

## 9. 分级启动模板

GPU 工作按四个独立阶段递增，每个阶段前重新采样，且都通过保护器启动：

1. `verify_gpu_mapping.sh`：验证 JAX、PyTorch 和 MuJoCo EGL 只看到选定设备；
2. `probe_pi0_libero_checkpoint.sh`：显式恢复原始 checkpoint，端口就绪后停止；
3. `probe_pi0_libero_inference.sh`：发送一次合法观测，要求返回有限的 `50×7` 动作；
4. `run_pi0_libero_batch_workload.sh`：最后才创建 LIBERO 环境并运行有界 episode。

包装器默认保持已验证的10状态范围，但支持通过环境变量缩小或恢复任务。下一次复核应先运行单 episode：

```bash
export LIBERO_SUITE=libero_spatial
export LIBERO_TASK_ID=0
export LIBERO_INITIAL_STATES=0:1
export LIBERO_MAX_EPISODES=1
export LIBERO_SEED=7
export LIBERO_REPLAN_STEPS=5
export LIBERO_MAX_CONTROL_STEPS=220
export LIBERO_MAX_OUTPUT_BYTES=1073741824
export LIBERO_RESUME=0
```

在30秒采样和设备映射通过后，再为本次运行填入：

```text
GPU_ID             选中的物理 GPU
E_MIB              本阶段报告用峰值显存预算
MAX_RUNTIME_SECONDS 固定运行上限
CONTROL_DIR        新建的任务控制目录
OUTPUT_DIR         尚不存在的输出目录
```

受保护启动关系为：

```text
run_gpu_guarded_supervisor.sh
  → gpu_utilization_guard.py
  → run_pi0_libero_batch_workload.sh
      → original pi0_libero Policy Server
      → bounded LIBERO evaluator
```

只有在展示当次采样、选卡、映射和完整命令后才执行。不要把本节当作无需检查即可复制运行的快捷命令。

## 10. 断点恢复

恢复必须使用同一个输出目录，并保持 suite、task、seed、Policy、GPU映射、commits、checkpoint 和 norm stats 一致：

```bash
export LIBERO_RESUME=1
export LIBERO_INITIAL_STATES=<original-scope>
export LIBERO_MAX_EPISODES=<original-limit>
```

评测器读取 `run_config.json` 和 `results.jsonl`，跳过已经完成的 initial-state indices。若身份或参数不一致，它会拒绝恢复，而不是混写结果。

## 11. 结束与验收

包装器正常或异常退出时会先向自己记录的 Policy Server PID发送 `SIGTERM`，等待后才在必要时处理同一个已核实 PID。保护器只管理自己创建的进程组。

每次运行结束后检查：

```bash
ss -H -ltn 'sport = :8000'
pgrep -a -u "$USER" -f '[s]erve_policy.py|[e]val_libero_bounded.py|[g]pu_utilization_guard.py'
du -sh "$OUTPUT_DIR" "$CONTROL_DIR"
```

还应核对：

- `guard.exit_code` 与 `evaluator.exit_code`；
- guard 事件中是否出现 pause/resume/monitor error/emergency；
- server/evaluator 日志是否有 OOM、ECC、Xid 或 traceback；
- 结果 JSONL、摘要、视频与 action trace 是否一致；
- 选中 GPU 的显存是否在进程退出后释放；
- 输出是否仍在1 GiB单次上限和80 GiB总项目预算内。

## 12. 已验证的仿真前状态

2026-08-21 的无 GPU复核结果：

- OpenPI 与 LIBERO commits正确；
- 双 Python 环境可用；
- 原始 `pi0_libero` 配置为 `Pi0Config`，horizon 50；
- checkpoint 共19个文件、12,014,131,888 bytes；
- norm stats 与保护器 SHA-256正确；
- 稳定的任务级 LIBERO 配置可导入6个 suite；
- 使用该稳定配置的单状态完整 evaluator dry-run通过；
- dry-run未创建输出目录；
- 端口8000空闲；
- 没有本任务残留进程；

同日按四阶段流程在物理 GPU 0 完成一次重新部署验收：设备映射、checkpoint
加载和单次 `50×7` 推理均通过；随后运行 `libero_spatial` task 0、initial
state 0、seed 7 的一个 episode，77 个控制步和16次策略请求后成功。episode
阶段采样峰值为9,603 MiB和41%，保护器未暂停且没有监控错误。该 `1/1`
结果只证明闭环部署可重复，不代表 LIBERO benchmark 成功率。

随后在新的批量阶段重新完成31秒 CPU/RAM/逐卡 GPU 采样，只运行尚未覆盖的
initial states 10–49，取得39/40；与已有 states 0–9 合并后，task 0 的50个
固定状态为48/50（96.0%）。失败 states 1 和35 都在220步上限结束且没有系统
异常。该结果仍只覆盖一个 task，不是完整 LIBERO Spatial benchmark。

在此基础上，`run_pi0_libero_remaining_workload.sh` 以同一原始 checkpoint、seed
和变换协议顺序完成其余39个 task。编排器只启动一个 Policy Server，对每个 task
运行 states 0–49，并根据 `run_config.json` 与 `results.jsonl` 跳过已完成 states；
中断后必须在相同物理 GPU 映射下恢复。39个 task 新增1,950 episodes，结合已有
Spatial task 0 后形成2,000-episode四-suite结果：1,845/2,000（92.25%）。完整
汇总见 [`../artifacts/libero-benchmark/pi0_libero_official4_seed7/run_report.md`](../artifacts/libero-benchmark/pi0_libero_official4_seed7/run_report.md)。

长任务通过 `nohup` 与独立 session 脱离 SSH，但 GPU 工作负载本身仍由保护器创建
和监管；Mac 断线不会让任务失去暂停、恢复或紧急停止保护。本次保护器退出码为0，
全部进程、端口8000和GPU显存均在完成后释放。
- 本地保护器两项模拟测试和全部 Shell 语法检查通过。
