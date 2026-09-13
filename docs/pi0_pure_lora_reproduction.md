# pi0 pure-LoRA 主实验复现说明

本说明复现的是已经锁定的 E2 main-200 协议，不允许以此重新选择 adapter 或自动
扩展到 E3。

## 固定输入

- OpenPI source：`/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`，
  commit `3619c35ffdcbfe97ae735de175d91c2fb67a899d`。
- Base 参数：`/home/wengzr/.cache/openpi/openpi-assets/checkpoints/pi0_base/params`。
- canonical norm stats：工具快照下
  `artifacts/pi0-pure-lora/evidence/n1b-r3/attempt-20260902T-N1B-R3-nZfeRdmk/canonical/norm_stats.json`。
- E0 manifest：工具快照下
  `manifests/pi0_pure_lora/e0_task_state_manifest.json`。
- selection lock：`/home/wengzr/projects/openpi-lora-runs/evaluations/attempt-20260912T-E1-DEV-R8-B1w9aV/selection_lock.json`。
- selected adapter：step 25000，identity
  `bd0bb0009ff16eb2588a472d4ff070586df9a8c78ba933dbc7250f7c16960d7d`。

所有路径应先重新哈希；若任一 manifest、commit、adapter、norm 或 E0 identity 不同，
必须新建实验 attempt，不能覆盖或续写本结果。

## 安全执行顺序

1. 用固定 OpenPI Python 做 CPU-only `py_compile`、fake-tree/控制器测试与输入路径检查。
2. 重新采样双卡与 CPU/RAM 至少 30 秒；选择空闲显存严格大于 15% 的物理卡，关闭
   JAX 预分配。
3. 创建 collision-safe control 与 run 目录；检查端口、tmux 名称和输出目录均未存在。
4. 仅通过 verified GPU guard 启动 Base 或 locked-LoRA；guard 只操作自己创建的进程组：
   利用率 >=95% 或空闲显存 <=15% 时暂停，连续五次安全样本才恢复，<=10%/OOM/ECC/Xid
   时终止本任务。
5. 每个 main state 写一条结构化 result；任何 `exception`、零策略请求、重复或缺失 key
   都使配对审计失败关闭。
6. 使用 `tools/pi0_pure_lora/audit_e2_main.py` 对两组 main-200 生成新的审计目录；
   不修改旧结果。

## 已验证的控制入口

- Policy server：`tools/pi0_pure_lora/serve_pure_lora_policy.py`。
- Base recovery：`tools/pi0_pure_lora/run_e2_base_recovery.py` 与
  `tools/pi0_pure_lora/launch_e2_base_recovery.sh`。
- Paired audit：`tools/pi0_pure_lora/audit_e2_main.py`。

恢复 Base 时，服务端必须将 Golden LoRA 叶子实体化为零数组；不可把
`nnx.eval_shape` 的 `ShapeDtypeStruct` 直接传给编译后的 JAX policy。

## 复现实验的解释边界

成功完成 200/200 只证明协议、服务和 evaluator 走通；成功率由 paired audit 的
`comparison_summary.json` 决定。E3 full-2000、重新训练、改变 normalization、修改
E0 states 或以 main 结果换 checkpoint 都是新的授权范围。
