# pi0_base → LIBERO pure-LoRA 对话查阅与下一阶段交接

更新时间：2026-09-09
用途：作为新对话或恢复对话时的第一份入口文档。本文区分“已有证据”、“已实现但未真实执行”和“未开始”，不将工具完成误写为训练完成。

> 2026-09-09 续做记录：T1 `100→200` 工程恢复验证已完成并独立验收；它不是正式候选。FT0 已冻结新的正式 `pi0_base→0→1000` 轨迹和存储上界。提交 `42ec3e8` 的 FT1 非执行模板/终态验收器已部署为 1,025 文件的只读远端工具快照，本地与远端各 31 项 CPU-only fake 测试通过。模板仍为 `execution_authorized=false`；未开始正式训练。旧 GPU 前检的 120 秒有效窗已过，真实启动前必须重新采样。

## 1. 当前一句话停点

pure-LoRA 的数据、normalization、精确冻结、checkpoint/adapter 保存恢复、真实 1/10/100-step smoke、远端自治基础设施和 **T1 execution-control CPU 收尾**均已有证据。

当前停在 **FT1 正式 `0→1000` 启动授权关口**：持久化的 freeze、工具、只读快照、非执行模板、终态验收器和 CPU-only 测试均已完成。下一次真实启动前必须重新做约 30 秒双卡+CPU/RAM 前检并在有效窗内封存 execution plan；这属于时间敏感的启动门禁，不能提前完成或复用。未开始正式候选训练。

## 2. 固定实验定义

- 初始权重：`pi0_base`。
- 主受控比较：`pi0_base weights + LIBERO target-domain normalization (no-gradient baseline)` 对比 `pi0_base + pure LoRA + 同一 canonical normalization`。
- pure-LoRA 真值：独立 Golden manifest 中恰好 20 个 adapter 叶子，49,987,584 个可训练参数；其他 50 个参数叶子、3,238,048,528 个参数全部冻结。
- 验收真值不使用宽泛 `.*lora.*`，而是使用人工审阅后的 `golden_adapter_paths.json`；任一 Golden 叶子缺失或任一非 Golden 参数可训练均失败。
- 官方 `pi0_libero` 继续使用 checkpoint-owned normalization，仅是端到端外部参考，不属于相同 normalization 协议下的主比较。
- 训练 seed 固定为 42，评测 seed 固定为 7，两者分开记录。

## 3. 已完成与已验证内容

| 阶段 | 状态 | 已完成内容 | 对后续的作用 |
|---|---|---|---|
| v1 保护 / R0 | 通过 | 标签 `pi0-libero-v1.0` 指向 `efccc294...`；LoRA 使用独立本地工作树和远端工作树 | 防止 LoRA 工作覆盖旧 `pi0_libero` 部署与 2000-episode 证据 |
| G1/G2/V0 | 通过 | 参数树发现、Golden 枚举、精确 `PathIn` 冻结及 fail-closed 验证 | 建立 pure-LoRA 的可机器检查定义 |
| D1b/D1c-Rb | 通过 | 固定 LIBERO revision；保留一份 raw snapshot 和一份 loader-required Arrow cache，offline loader 复用时无第二份数据增长 | 避免重复下载和重复的约 34.9 GB cache |
| N1a/N1b-R3 | 通过 | 按 `RepackTransform → LiberoInputs(pi0) → DeltaActions → RunningStats` 计算并原子发布 canonical `norm_stats.json` | 使 Base 与 pure-LoRA 共享同一目标域 normalization |
| B1/I1/I2/I3/C0 | 通过 | `pi0_base` 身份、adapter-only 组合、原子 checkpoint 控制面、所有 parameter/optimizer 叶子逐值 restore 与禁止自动 pruning | 保证可 resume，并可将 base + adapter 组合为可验证模型 |
| S1a | 通过 | 真实 `pi0_base` checkpoint load 与 train-state 初始化 | 证明真实模型路径可用 |
| S1b | 通过 | 真实 LIBERO batch 的 1-step 更新；20/20 Golden 变化，50/50 非 Golden 不变 | 证明反向传播与冻结规则在 GPU 上成立 |
| S1c | 通过 | 真实 10-step 稳定性 smoke，loss/grad norm 全部有限，共享 cache 零增长 | 将单步可行性提升为短序列稳定性证据 |
| S1d | 通过 | 真实 100-step smoke；100 组 metrics 有限；保存 step-100 full state 与 adapter，并逐值 restore/组合验证 | 产生 T1 恢复工程段的已知良好起点 |
| E0 | 通过 | 结果盲的 40 dev + 200 main states 预注册，每任务 dev/main 互斥 | 防止使用 main 集选 checkpoint 或调参 |
| A2 | 通过 | 远端自治 orchestrator、immutable plan、心跳/状态、有界日志、有限重试、自有 PGID 回收和断连 smoke | 使长阶段不依赖 Codex 会话、Mac 或前台 SSH |
| T1 freeze | 通过（不授权执行） | 冻结非候选工程段 `100→200`、正式候选 steps、seed、存储上界和禁止删除策略 | 将工程恢复验证与正式研究轨迹分开 |
| T1 resume runner CPU | 通过（未真实执行） | 实现 loader 精确 skip 100、双 loader fingerprint、RNG split 重放 100 次、输入树重哈希与目标冲突门禁；本地/远端各 11 项测试通过 | 填补 OpenPI 通用 `--resume` 不保证 data-loader 位置连续的缺口 |
| T1 execution-control CPU | 通过（已提交；T1 工程段已结束） | 实现静态模板、新鲜前检后封装、独立 terminal verifier 和三层异常回收；本地/远端 fake 测试通过 | 在工程恢复段中 fail closed 地绑定身份、资源、进程组、输出和终态验收 |
| FT0 / FT1 静态准备 | 通过（不授权启动） | 冻结正式候选段，提交 FT1 非执行模板和终态验收器；本地/远端各 31 项 CPU-only fake 测试通过，并部署只读工具快照 | 使正式 `0→1000` 只差一次新鲜前检与用户明确启动授权 |

### S1d 关键实测结果

- 训练：100 steps，batch size 1，workers 0，seed 42；loss 首步 `0.221992`，末步 `0.060486`。这只证明有限性与可训练性，不声称收敛或性能提升。
- full train-state：`5,559,083,375 B`，70 个 parameter 叶子和 42 个 optimizer 叶子逐值恢复一致。
- adapter-only：`199,962,483 B`，恰好 20 个 Golden 叶子；base + adapter 组合后与训练末态逐参数一致。
- GPU guard：30 样本前检后固定物理 GPU 1；运行期最高利用率 78%，最低空闲显存 43.41%，无 pause/resume、monitor error、OOM/ECC/Xid 或本任务残留进程。
- 验收：42 项 fail-closed 检查全部通过。

## 4. 当前固定身份与路径

### Git / 源码

- 本地工作树：`/Users/buzz/MyProjects/openpi-LIBERO-lora`
- 本地分支：`feature/pi0-libero-pure-lora`
- FT1 静态工具快照提交：`42ec3e8a7d59fa0a01192caf404f9cf76618a6d8`（已推送）
- 远端 OpenPI 工作树：`/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`
- 远端分支：`feature/pi0-libero-pure-lora`
- 远端固定 HEAD：`3619c35ffdcbfe97ae735de175d91c2fb67a899d`
- 远端 upstream：`NONE`
- 远端支持工具根：`/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora`
- 固定 Python：`/home/wengzr/projects/openpi/.venv/bin/python`

### 实验身份

- model：`d484ef5fa06bcb92b0dad92d1f221d4b65406f86dd928569362cb8a9106213ac`
- dataset：`44c5fbe41202cbd29cf209d2856b4e95857f89728697a994d9aa14e0f2c5d700`
- canonical norm：`f68a5fafe15e1577b7bb2c6fc4837a7d1669e2e9be3752f2589c3d327c6f8ccf`
- Golden manifest：`3799cf4d053b013089216be97ab0b57d08dde1dd3c4f04744088ce2e93a32029`
- config/runtime：`1c289cc470e064d6717513e149b5cae03ee71b56f7c9634df450e496ca46c958`
- E0 split：`f85350357b75ecaea7330e386f805d024ea44a7a6a5c58f96d64259fbedca283`

### T1 resume 输入

- S1d acceptance identity：`c560bb932159eaa28502f89560ed1a325422f41f9aec249c67e30764541f4d91`
- step-100 checkpoint tree：`abd8fdd33915eae208b2aebe1dcad7fd4675df927ebc504438a448d895ff26a1`
- step-100 adapter identity：`26dc0261e16bb7ca8f77a7ea8113f22c5dab6d264f4e51861cbd003e80b543c1`
- checkpoint root：`/home/wengzr/projects/openpi-lora-runs/checkpoints/pi0_libero_pure_lora/s1d-smoke-seed42-20260904`
- adapter root：`/home/wengzr/projects/openpi-lora-runs/adapters/s1d-smoke-seed42-20260904`
- T1 freeze package identity：`0d1521a326e1f951f4f89f1660090a88ea2e1b162271cf4ef258b0fcdbd44939`

## 5. 现场与限制

- 原始 `/home/wengzr/projects/openpi` 保持只读；LoRA 源码只在专用远端 worktree 操作。
- 已选 raw snapshot 与 Arrow cache 都必须保留，不删除、不移动、不重建。
- 本地存在一个与当前文档任务无关、持续追加的 B1 tunnel `events.jsonl`；不得停止其未知/既有进程，不得将该变化混入本任务。
- LoRA 计费占用的最后实测值为 `91,765,739,008 B`；不包括只读 v1/共享基础快照 `24,574,841,856 B`。
- 存储 review line：`225,000,000,000 B`；soft stop：`240,000,000,000 B`；hard limit：`250,000,000,000 B`；新阶段前保留至少 `20,000,000,000 B` 未承诺空间。
- 任一单项预计新增超过 10 GiB 仍需用户单独决定。
- full-state 轮换必须允许旧+新两份短暂共存；新 step 完整写入、hash 和 restore 验证通过前不得删除最后已知良好状态。
- 当前不存在已授权的 T1 GPU 执行。

## 6. 刚完成阶段：T1 execution-control CPU 收尾

### 6.1 实现结果

- `t1_execution_contract.py`：固定身份、路径、存储、前检、超时和环境合同。
- `build_t1_execution_template.py`：生成无 command、无 environment、无预选 GPU 且不可执行的静态模板。
- `finalize_t1_execution_plan.py`：只将同一 host/boot 且在时间窗内的 30–120 样本前检绑定为精确 plan；plan 仍为 `execution_authorized=false`。
- `verify_t1_resume_result.py`：独立使用 Golden 真值验收 step-100 保留、step-200 原子完成、loader/RNG 连续、参数变化、full restore、adapter 组合、guard、cache 和终态进程。
- guard/orchestrator 修正：即使组长先退出或证据 I/O 失败，也会在有界时间内对自己创建的 PGID 执行 TERM→KILL→wait，不扫描或操作未知 PID。

### 6.2 验证结果

- 本地 macOS Python：`py_compile + 110/110` 项测试通过。
- 远端固定 `/home/wengzr/projects/openpi/.venv/bin/python` 3.11.15：`py_compile + 110/110` 项测试通过。
- 本地与远端 67 个源文件哈希映射完全一致，且测试期间稳定。
- 三层嵌套 guard 的 external-signal 测试验证了每层自有进程组的转发与回收；未知独立进程不被 signal。
- R1/R2 因远端精简测试快照的 AppleDouble/fixture 缺失失败，均已保留；R3 补齐后通过。

### 6.3 部署完成与仍未执行

- execution-control 源码已提交并推送为 `7be18dd407869ae34d521441d7403dbdca5573c5`；67 文件远端快照逐项哈希通过，archive SHA-256 为 `2798b59623fc19b16f4237c25585565c4b63834905d5b681691a1b24a0accc92`。
- 已生成远端静态模板，SHA-256 为 `38b27c6cbd5863a5cef3f0211dea66c8478224a84e45a7bbf1fbec59104b5294`，template identity 为 `1dca5bce43b0f28b27dd566d9f8acc638550ebacd9482334d7e02e850356e3cd`；无 command/environment/GPU 选择，且仍不授权执行。
- 已用一次通过的 30 样本前检封存 exact non-authorizing plan，plan SHA-256 为 `f0e876e4cc6649de8fda5198ab69e5fbc8e38c1e84399b375d9dfe7e1875fcf8`，identity 为 `f113b31048ad2406fad4820c5d3d1084223030c091908e4161ba83d85fd6b128`；当时选择 GPU 1（UUID `GPU-14900654-ea51-b7aa-28d3-b2885502d727`）。plan 未执行，且前检有效窗不能留待之后复用。
- 首次真实启动的独立 run package 也在 launcher 顶层解析处 fail-closed：其 plan 已封存但 `tmux`、训练 attempt 与训练进程均未创建。修正为只在第一个 `--` 前解析顶层 `--attempt-dir`，并新增嵌套 child 参数回归测试；该修正尚未提交或部署。
- 真实 5.5 GB S1d checkpoint 读取或重哈希。
- 真实 LIBERO loader/data 解码、Pi0 import/load、GPU preflight 或训练。
- step-200 checkpoint/adapter 和任何评测。

## 7. 之后的候选路线（均未授权）

1. **T1 `100→200` 工程验证段**：经用户对真实 checkpoint/data/model/GPU 执行的明确授权后，重新执行新鲜 30 秒双卡+CPU/RAM 采样，动态固定一张卡，并在有效窗内封存新的 plan；随后才可在远端自治和双 guard 下恢复 step 100 并训练到 step 200。保留 100 和 200，恢复/组合/终态验收后停止。该段不是候选 checkpoint，不参与性能结论。
2. **T1 正式新轨迹**：从 `pi0_base` 新 run root 启动，候选 steps 已冻结为 `1000/5000/10000/15000/20000/25000/30000`。每段独立授权，full state 用于 resume，候选里程碑发布 adapter-only。
3. **E1 dev-40**：仅在预注册 dev states 上评测固定候选，以成功数最高优先；并列时依次选更早 step、字典序更小 adapter identity。
4. **checkpoint selection lock**：锁定唯一最终 LoRA checkpoint；首个 dev episode 开始后不得追加候选，main 集不用于选模型或调参。
5. **E2 main-200**：共同测试 Base、最终 pure-LoRA 和从已有 2000-episode 结果投影的 official 外部参考；不默认重跑 official。
6. **E3 可选 full-2000**：只有最终 LoRA 在 main-200 上值得扩展时，再单独决定。
7. **F1 结项**：整理成功率、trainable/total params、训练时间、GPU 峰值、存储、制品身份、限制与复现说明。

## 8. 关键证据入口

- 总状态：`docs/pi0_pure_lora_status.md`
- Golden manifest：`manifests/pi0_pure_lora/golden_adapter_paths.json`
- Base/C0 manifest：`manifests/pi0_pure_lora/base_model_manifest_c0.json`
- E0 split：`manifests/pi0_pure_lora/e0_task_state_manifest.json`
- S1d 验收：`artifacts/pi0-pure-lora/evidence/s1d/attempt-20260904T-S1D-FFS04yQr/acceptance_report.json`
- A2 验收：`artifacts/pi0-pure-lora/evidence/a2/attempt-20260907T-A2-AUTONOMY-K4m9Q2/acceptance_report.json`
- T1 冻结包：`artifacts/pi0-pure-lora/evidence/t1-freeze/attempt-20260907T-T1-FREEZE-L7q3R8/freeze_package.json`
- T1 resume runner CPU readiness：`artifacts/pi0-pure-lora/evidence/t1-resume-runner/attempt-20260907T-T1-RESUME-CPU-H6v2N9/readiness.json`
- T1 execution-control CPU readiness：`artifacts/pi0-pure-lora/evidence/t1-execution-control/attempt-20260908T-T1-CONTROL-CPU-2fb7ac5a/readiness.json`
- T1 execution-control 部署/静态模板：`artifacts/pi0-pure-lora/evidence/t1-execution-control-deployment/attempt-20260908T-T1-DEPLOY-7be18dd/static-template.json`
- T1 新鲜前检与 exact non-authorizing plan：`artifacts/pi0-pure-lora/evidence/t1-execution-control-deployment/attempt-20260908T-T1-DEPLOY-7be18dd/fresh-preflight.json`、`artifacts/pi0-pure-lora/evidence/t1-execution-control-deployment/attempt-20260908T-T1-DEPLOY-7be18dd/exact-non-authorizing-plan.json`

## 9. 新对话建议读取顺序

1. 先读本文的第 1、5、6 节，确认当前停点和授权边界。
2. 需要整体状态时读 `docs/pi0_pure_lora_status.md`。
3. 需要机器可验证数值时，只打开第 8 节对应的 JSON 证据，不先扫描大型日志或制品。
4. 任何长任务先生成静态授权包，说明命令、资源、通过条件、停止条件和回滚/保留方案；等待用户本人决定。
5. 未经明确要求，不执行 `git add/commit/push/tag/rebase`。

---

本文中“下一阶段”仅表示后续工作边界，不代表已获真实 checkpoint 读取、模型/数据加载、GPU 或训练授权。
