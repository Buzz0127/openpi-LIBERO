# pi0_base -> LIBERO pure-LoRA：GPT-6 继承入口

> 状态快照：2026-09-10（Asia/Shanghai）
> 用途：供新的 GPT-6 对话直接继承本项目的目标、路线、已完成工作、当前现场、安全边界和下一步候选工作。
> 本文件是交接快照，不是任何未完成阶段的执行授权。

> 2026-09-10 路线补充：读完本入口后，继续阅读 [项目完成路线](pi0_pure_lora_completion_route.md)。它在保留 FT0/E0 冻结协议和本文件授权边界的基础上，细化训练控制、评测接线和最终报告交付，仍为候选路线。第 13 节的 C-FT2 决策尚未由用户作出，不因该补充而跳过。

## 0. 接手时先做什么

GPT-6 接手后应按以下顺序恢复上下文：

1. 完整阅读本文件。
2. 阅读 `docs/pi0_pure_lora_status.md` 和 `docs/pi0_pure_lora_conversation_handoff.md`，但注意这两份旧文档目前滞后于 FT1/FT2 的真实进度。
3. 读取实现对话 `codex://threads/01a04761-9200-7022-82f0-9f24fcd7627a` 的最新若干回合；需要更早证据时再向前翻页。
4. 只读核对本地 LoRA 工作树的 `pwd`、branch、HEAD、upstream、`git status --short`。
5. 只读核对远端相关 `status.json`、`summary.json`、`runner_result.json`、`terminal_acceptance.json` 和当前进程；不得仅依据旧文档推断训练状态。
6. 将现场分成四类再向用户汇报：
   - 已完成且有真实证据；
   - 已实现/已测试但尚未提交或尚未用于后续真实阶段；
   - 历史失败、偏差或受控例外；
   - 未开始、未授权。
7. 未得到用户对具体下一阶段的明确决定前，不启动训练、评测、大下载、checkpoint 删除或 Git 提交。

## 1. 对话与职责

### 1.1 相关对话

- 路线/监督对话：`codex://threads/01a0475a-4c3f-74a0-9f57-802d7969f34a`
- 实现对话：`codex://threads/01a04761-9200-7022-82f0-9f24fcd7627a`
- 补充进度来源：`codex://threads/01a07c74-2da5-72c2-9e72-c2ea3b330c10`
- 更早的 π0-LIBERO 复现与安全基线：
  - `codex://threads/019fea97-c27e-7b40-871e-9f854377badb`
  - `codex://threads/019fea87-bd05-7bc2-89d9-49a1ff105d78`
  - `codex://threads/019fe9b4-5c62-78e3-a5f9-555e03f99242`

### 1.2 GPT-6 的职责

- 延续既定最终目标，不擅自改成官方 low-memory 配置、全量微调、二次微调 `pi0_libero` 或新的研究问题。
- 根据用户指令制定路线或在实现对话执行具体阶段。
- 用户已经明确：是否审查、是否执行由用户本人决定。不得把 GPT 的方案、检查结果或“建议下一阶段”当成授权。
- 每阶段只完成被明确指定的范围；完成后停止并按第 11 节格式汇报。
- 不得因“继续任务”“进行下一阶段”而自动跨越多个大阶段；应结合当前停点解释它对应的唯一阶段。

## 2. 不可改变的最终训练目标

主实验固定为：

```text
pi0_base
  + canonical LIBERO target-domain normalization
  + pure-LoRA-only training on LIBERO
  -> LoRA candidate adapters
  -> preregistered development evaluation
  -> lock one final adapter
  -> preregistered main evaluation
```

核心研究问题：

> 仅训练 π0 中经过独立 Golden manifest 确认的 LoRA A/B 参数，能否以显著更少的可训练参数和可控训练资源，在 LIBERO 上接近官方 `pi0_libero` 的端到端表现？

### 2.1 模型比较口径

1. 主基线：`pi0_base weights + canonical LIBERO target-domain normalization (no-gradient baseline)`。
2. 主实验：同一 `pi0_base`、同一 canonical normalization、只更新 Golden LoRA leaves。
3. 官方参考：官方 `pi0_libero` 使用其 checkpoint 自带 normalization，只作为外部端到端参考。
4. 不把主基线称为“严格 zero-shot”，因为它使用了 LIBERO 目标域统计量。
5. 若 canonical stats 与官方 checkpoint stats 数值不一致，Base 与 pure-LoRA 的受控主比较仍成立；官方结果不得写成严格同协议比较。

### 2.2 pure-LoRA 的机器定义

- 独立 Golden manifest 是唯一真值，不能用 `.*lora.*` 或“路径包含 lora”代替。
- Golden adapter leaves：20 个。
- 机器确认的 trainable 参数：49,987,584。
- 非 Golden leaves：50 个，共 3,238,048,528 参数，必须全部冻结。
- 任一 Golden leaf 缺失、任一非 Golden 参数可训练、任一基座权重在训练后变化，都必须 fail closed。
- 官方 `pi0_libero_low_mem_finetune` 会训练部分非 LoRA 参数，只能作为工程参考，不能冒充本项目的 pure-LoRA 主实验。

## 3. 固定身份与隔离目录

### 3.1 已保存的原复现实验

- v1 tag：`pi0-libero-v1.0`
- v1 commit：`efccc29460535f637a6284c6c130fbbea66357cf`
- v1 及其结果只读保留，不得被 LoRA 工作覆盖。

### 3.2 本地 Mac

- 唯一允许修改的 LoRA 工作树：`/Users/buzz/MyProjects/openpi-LIBERO-lora`
- LoRA branch：`feature/pi0-libero-pure-lora`
- 原 v1 工作树：`/Users/buzz/MyProjects/openpi-LIBERO`，只读保留。
- 每次本地操作前核对 `pwd`、branch、HEAD、upstream；不符即停止。

### 3.3 远端服务器

- SSH 普通登录：`ssh openpi-libero`
- 只有明确需要 `17890 -> 7890` 代理时才使用：`ssh openpi-libero-proxy`
- 固定 Python：`/home/wengzr/projects/openpi/.venv/bin/python`
- 原 OpenPI：`/home/wengzr/projects/openpi`，只读保留，既有 `outputs/` 不触碰。
- LoRA OpenPI worktree：`/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`
- LoRA 辅助工具与 evidence：`/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora`
- LoRA 运行根：`/home/wengzr/projects/openpi-lora-runs`
- LoRA cache 根：`/home/wengzr/projects/openpi-lora-cache`
- canonical assets 根：`/home/wengzr/projects/openpi-lora-assets`

已固定的远端 OpenPI 源码 commit：

```text
3619c35ffdcbfe97ae735de175d91c2fb67a899d
```

## 4. 关键不可变身份

| 对象 | 身份 / SHA-256 |
|---|---|
| LIBERO Hub revision | `a4336d589d589045d1c56423ffdf3b88a0e19b1f` |
| LIBERO 数据身份 | `44c5fbe41202cbd29cf209d2856b4e95857f89728697a994d9aa14e0f2c5d700` |
| 数据规模 | 273,465 frames；1,693 episodes；40 tasks |
| canonical norm stats | `f68a5fafe15e1577b7bb2c6fc4837a7d1669e2e9be3752f2589c3d327c6f8ccf` |
| `pi0_base` manifest | `0a9e72e07255dcdb09b71ebc3daac3c7ab0a8d9d3be72426c4464ca46ef8d8dd` |
| Golden manifest 文件 | `manifests/pi0_pure_lora/golden_adapter_paths.json` |
| Golden experiment identity | `3799cf4d053b013089216be97ab0b57d08dde1dd3c4f04744088ce2e93a32029` |
| model identity | `d484ef5fa06bcb92b0dad92d1f221d4b65406f86dd928569362cb8a9106213ac` |
| config/runtime identity | `1c289cc470e064d6717513e149b5cae03ee71b56f7c9634df450e496ca46c958` |
| preregistered split identity | `f85350357b75ecaea7330e386f805d024ea44a7a6a5c58f96d64259fbedca283` |

任何恢复训练、adapter 组合或评测，都必须在 manifest 中重新绑定相关身份；不能仅依赖目录名。

## 5. 已完成并验证的路线阶段

### 5.1 基础设计与数据链

| 阶段 | 状态 | 已完成内容 |
|---|---|---|
| R0 / G1 / G2 / V0 | 完成 | 抽象参数树发现、精确 pure-LoRA 过滤、独立 Golden 真值、参数计数、负例与回归测试均通过。 |
| D1c-Rb | 完成 | 固定 LIBERO revision，验证 1,693 episodes / 273,465 frames / 40 tasks；证明离线 loader 使用选定的唯一 raw snapshot 和唯一 loader-required Arrow cache。 |
| N1 | 完成 | 使用固定数据、split 和变换链计算 canonical z-score stats，并固化其 hash。 |
| B1 / I1 / I2 / I3 / C0 | 完成 | `pi0_base` 身份、adapter-only 导出/组合、checkpoint/restore、受控评测和远端隔离基础设施已完成相应验收。 |
| E0 | 完成 | 预注册 development 40 episodes 和 main 200 episodes；train seed、split seed、eval seed 分离。 |
| A2 | 完成 | 服务器端 autonomous orchestrator、heartbeat/status/summary、断连存活、进程组回收和 fail-closed 测试完成。 |

### 5.2 Smoke 与工程恢复验证

#### S1d：100-step smoke

- 100 组有限 metrics。
- loss：首项 0.221992，末项 0.060486。
- full state：5,559,083,375 B。
- adapter-only：199,962,483 B。
- restore：70 个 parameter leaves、42 个 optimizer leaves 全值一致。
- 20 个 Golden leaves 改变，50 个非 Golden leaves 不变。
- step-100 adapter identity：`26dc0261e16bb7ca8f77a7ea8113f22c5dab6d264f4e51861cbd003e80b543c1`。
- acceptance identity：`c560bb932159eaa28502f89560ed1a325422f41f9aec249c67e30764541f4d91`。

#### T1：100 -> 200 工程恢复段

- 已真实完成，但 `candidate=false`，仅证明 resume/data-loader/RNG/guard/自治链路。
- 100 组有限 metrics；loss 首项 `0.2410156578`，末项 `0.1311346889`。
- loader 精确 skip 100；RNG replay 100。
- step 100 与 step 200 full state 同时保留，总计 `11,118,291,492 B`。
- 两个 adapter 总计 `399,924,966 B`。
- step-200 adapter identity：`f17493dfdc8decd71b28947abaf0627f696bc1dbfa820f94a7c8834a8db1d279`。
- 20 Golden 改变；50 非 Golden 不变；70 parameter + 42 optimizer leaves restore 通过。
- 最终 terminal acceptance：
  `/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/t1-execution-control/attempt-20260909T044053Z-T1-FINAL-e19eb72/terminal-acceptance.json`
- report identity：`1fcade355ab5c56c454448e407dab30a1f46b1bb1ac92aa5495b07f660f9c6f4`。
- 这不是正式候选、不是收敛证据、不是性能提升证据。

### 5.3 FT0：正式训练冻结

- 正式轨迹从新的 `pi0_base` 根开始，禁止接续 T1 工程 step 200。
- train seed：42。
- batch size：1；workers：0。
- 优化器：AdamW；计划：30k cosine schedule。
- 正式候选 steps：`1000 / 5000 / 10000 / 15000 / 20000 / 25000 / 30000`。
- freeze package identity：`723ede183f745fc45e3709ad91fe8dd39e1e1b487fadb0257db4ddd92e402712`。
- 每一段只能从上一个已经验收的正式候选恢复；每段完成后停止，不自动进入下一段。

### 5.4 FT1：正式 0 -> 1000

真实制品目录：

```text
/home/wengzr/projects/openpi-lora-runs/formal-attempts/
attempt-20260909T-FT1-0-1000-451b757-R2
```

已核验事实：

- `runner_result.json`：`status=pass`。
- 0 -> 1000 共 1,000 组有限 metrics。
- 20 Golden leaves 改变，0 个非 Golden leaves 改变。
- checkpoint history：`[1000]`。
- step-1000 checkpoint restore 成功；70 parameter + 42 optimizer leaves 的 shape/dtype/value 均一致。
- adapter restore 值一致。
- step-1000 adapter identity：`4e4e0970c55a4ede002df49a7cd9f5cad4acf54a699a12ddf02e945eb9644faf`。
- `terminal_acceptance.json`：`status=pass`。
- terminal report identity：`4263e667f1582f71311ed56bf1ce5bdfadd749611e6c630f1db050bec0ab1627`。
- runner result SHA-256：`c72d152b10508f376d430499be610db90432405c70dc6c26b2166d656a91dbb2`。
- `candidate=true`，且 `next_stage_started=false`。

必须同时保留的历史偏差：

- 同一 attempt 的 outer `summary.json` 为 `status=fail`，reason 为 `child_exit_nonzero`。
- outer `status.json` / `progress.json` 停在 step 0。
- 因此不能写成“整个 FT1 编排链无条件通过”；准确说法是“训练 runner 与独立 terminal acceptance 通过，历史 outer orchestrator 记录失败”。

### 5.5 FT2：正式 1000 -> 5000

真实制品目录：

```text
/home/wengzr/projects/openpi-lora-runs/formal-attempts/
attempt-20260910T-FT2-1000-5000-N5v2Pa
```

已核验事实：

- `runner_result.json`：`status=pass`。
- 1000 -> 5000 共 4,000 组有限 metrics。
- 20 Golden leaves 改变，0 个非 Golden leaves 改变。
- checkpoint history：`[1000, 5000]`，旧良好状态未提前删除。
- step-5000 checkpoint restore 成功；70 parameter + 42 optimizer leaves 的 shape/dtype/value 均一致。
- adapter restore 值一致。
- step-5000 adapter identity：`06bcbb7595031f6eace138d65af1017b9cb01cc5f15c3db73cd7e4bbd3933548`。
- `terminal_acceptance.json`：`status=pass`。
- terminal report identity：`336e084bed25c9b48bd472bd9d11c6af9653a44069992286694a98158dfe7397`。
- runner result SHA-256：`07e3c4bfd23e1110b5ae253db9cf7662bf56e232d88641fac11b78dca992911f`。
- `candidate=true`，且 `next_stage_started=false`。
- 训练主体耗时约 1,150 秒；当次 GPU guard 无 pause/OOM/ECC/Xid，任务相关进程已退出。

必须同时保留的历史偏差：

- outer `summary.json` 为 `status=fail`，reason 为 `committed_step_mismatch`。
- outer `status.json` / `progress.json` 停在 step 1000，而真实已提交并恢复验证的是 step 5000。
- 历史 outer summary 不得改写成通过。

### 5.6 FT2 progress 修复与后验复核

已完成但尚未 Git 提交的修复：

- 修改 `tools/pi0_pure_lora/run_formal_segment.py`：在结果原子写入成功后，再原子替换 `progress.json`，把 `current_step` 和 `last_committed_step` 更新到 segment end。
- 修改 `tools/pi0_pure_lora/test_run_formal_segment.py`：加入最终 progress 与 orchestrator 兼容性回归测试。
- 本地 CPU：语法、9/9 单测、`git diff --check` 通过。
- 新远端只读工具快照：约 10.42 MB；runner SHA-256 前缀 `a5dc334`。
- 远端 CPU：9/9 单测通过；固定 OpenPI worktree 复核为干净。
- 使用新快照对既有 FT2 结果重跑独立 verifier，terminal acceptance 仍为 pass；没有重训、没有修改旧 attempt。
- 新的后验观察 identity 前缀：`5bfa198`。
- 后验观察明确保留历史 outer 失败，并标记 `overall_release_ready=false`。

这项修复只保证未来 segment 的 progress 终态，不会、也不应修改 FT1/FT2 的历史失败记录。

## 6. 当前精确停点

截至本快照：

1. FT1 的训练 runner 与独立终态验收通过，但历史 outer orchestrator 失败。
2. FT2 的训练 runner、checkpoint、adapter、restore 与独立终态验收通过，但历史 outer orchestrator 因 stale progress 失败。
3. stale progress 的代码原因已修复并通过本地/远端 CPU 测试；修复尚未 Git 提交。
4. 历史 FT2 已做只读后验复核，没有重训或改写旧证据。
5. FT3 `5000 -> 10000` 未启动、未授权，也不得自动启动。
6. 在继续正式训练前，用户需要决定是否接受以下受控例外：
   - 保留 FT1/FT2 outer summary 的历史失败；
   - 以已通过的 runner result、逐值 restore 和独立 terminal acceptance 作为候选有效性的依据；
   - 仅从未来 FT3 起使用修复后的 progress 逻辑。
7. 若用户不接受该例外，替代路线是另起新的完整正式轨迹；不得静默重跑或覆盖现有 FT1/FT2。

这是一项路线处置决策，不改变最终 pure-LoRA 训练目标。

## 7. 当前 Git 与文件现场

本快照核验时：

```text
branch: feature/pi0-libero-pure-lora
HEAD:   fc30773d15eaf6ae7c4d1e1f3dad4ad05b01f2f5
```

未提交变化：

```text
M artifacts/pi0-pure-lora/evidence/b1-pi0-base/tunnel/
  attempt-20260902T-B1-TUNNEL-screen1/supervisor/events.jsonl
M tools/pi0_pure_lora/run_formal_segment.py
M tools/pi0_pure_lora/test_run_formal_segment.py
```

解释：

- 两个 `run_formal_segment` 文件是本轮 FT2 progress 修复。
- B1 `events.jsonl` 是此前既有、持续变化且本轮未触碰的现场文件；不要把它与修复混在一起提交或恢复。
- 用户规则：除非用户明确提出提交，否则不得运行 `git add`、`git commit`、`git push`、tag、rebase 或 squash。
- 任务执行期间以保存文件、evidence、manifest、status 和 pending-change 记录为主。
- 在 Git 提交前运行的真实任务必须在证据中绑定 HEAD 与相关未提交文件/patch hash，不能只写 branch 名称。

## 8. 存储规则

### 8.1 计费口径

- v1/共享基础快照：`24,574,841,856 B`，从 LoRA 独立额度中排除，但只读保留。
- LoRA 新增存储硬上限：十进制 `250,000,000,000 B`。
- review line：`225,000,000,000 B`。
- soft stop：`240,000,000,000 B`。
- hard stop：`250,000,000,000 B`。
- 始终保留至少 `20,000,000,000 B` 未承诺余量；若无额外决定，阶段峰值预算实际上不能超过 `230,000,000,000 B`。
- 任一预计新增超过 10 GiB 的单项仍需用户单独决定。

FT2 后最近一次确认的 LoRA billed usage：

```text
109,063,857,664 B
```

该值是 FT2 后快照，不可代替下一次大阶段开始前的重新盘点。

### 8.2 数据与 checkpoint 禁则

- 保留已选定的一份 raw LIBERO snapshot 和一份 loader-required Arrow cache。
- 不删除、不移动、不重建这两份数据，也不创建第二个约 34.9 GB 的副本。
- full train-state 安全轮换必须按旧状态与新临时/新提交状态短暂共存计算。
- 新 checkpoint 必须完成写入、完整性检查和真实 restore 后，才可在另行授权下处理旧状态。
- 不得依赖 `max_to_keep=1` 提前删除最后已知良好状态。
- 当前正式候选计划保留候选 adapter-only；full-state 是否删除必须按已冻结策略、当前预算和用户决定执行。

## 9. GPU、共享服务器与长任务规则

### 9.1 每个 GPU 阶段的启动门

每个独立训练或评测段开始前，都要重新进行约 30 秒、约 1 Hz 的双卡及 CPU/RAM 采样：

- 动态选择一张物理 GPU，并固定物理卡与 UUID。
- 启动时所选卡空闲显存必须大于 15%。
- 设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- 不使用“GPU 必须完全空闲”、`U_max + C <= 90%` 或 `F_min >= 1.8E` 的旧门槛。
- 前检有时间敏感性，过期后必须重做；不能提前很久采样后复用。

已验证 guard 只允许控制自身启动的进程组：

- GPU util >= 95% 或 free VRAM <= 15%：暂停本任务组。
- GPU util < 85% 且 free VRAM > 20% 连续 5 个样本：恢复本任务组。
- free VRAM <= 10%、OOM、ECC、Xid 或影响他人任务：只终止本任务组。
- 不得 kill、pause、renice 或修改任何未知进程。

### 9.2 长任务优先自治

对于超过约 15 分钟、重复监控、下载、训练、评测或 checkpoint 操作：

- 优先设计服务器端 autonomous orchestrator、resource guard、heartbeat、`status.json`、`summary.json` 和终态 receipt。
- 任务必须在远端独立运行，不依赖 Mac、Codex、SSH 会话或本地代理持续在线。
- Codex 只负责启动并确认 heartbeat 正常；随后停止高频轮询，用户可以关闭 Codex 或电脑。
- 返回时先读取小型 status/summary/receipt/hash，再按需读取大日志。
- 自动化只完成当前获批阶段，不得自动进入下一训练段或评测段。
- 所有失败 attempt 原样保留；不覆盖、不伪装为成功。

## 10. 后续候选总路线

以下只是候选路线，不是执行授权。

### 10.1 C-FT2：处置历史 outer 失败

目标：明确 FT1/FT2 是否可作为正式候选链继续使用。

需要用户二选一决定：

1. 受控例外继续：
   - 历史 outer summary 继续保留 fail；
   - 以 runner pass、逐值 restore、adapter 组合和独立 terminal acceptance pass 为候选有效依据；
   - 从 FT3 起使用已修复的 progress 原子终态更新；
   - 在最终报告中公开这项工程偏差。
2. 新轨迹重跑：
   - 新建 collision-safe formal run root；
   - 从 `pi0_base` 重新执行 0 -> 1000 -> 5000；
   - 不覆盖、不删除当前正式轨迹；
   - 重新计算峰值存储并分别授权真实 GPU 阶段。

完成标准：用户决定被记录，路线和证据引用清楚；不得仅由模型自行选择。

### 10.2 FT3 及后续正式训练段

若用户接受受控例外并明确授权，按以下 milestone 继续：

```text
FT3:  5000  -> 10000
FT4: 10000  -> 15000
FT5: 15000  -> 20000
FT6: 20000  -> 25000
FT7: 25000  -> 30000
```

每段必须独立完成：

1. 重新盘点存储和 projected peak。
2. 约 30 秒新鲜双卡+CPU/RAM preflight。
3. 绑定上一候选的 result、terminal acceptance、checkpoint、adapter、Golden、base、norm、dataset、config 与工具 hash。
4. 生成不可执行静态计划；用户明确授权后才封装实际 command/environment/GPU。
5. 使用远端 autonomous orchestrator 和双 guard 启动。
6. 训练期间只写当前 collision-safe attempt。
7. 保存新 full train-state 与 adapter-only；旧良好状态在新状态真实 restore 通过前保留。
8. 验证有限 metrics、20 Golden 改变、0 non-Golden 改变、70 param 与 42 optimizer leaves 全值 restore、adapter 组合一致。
9. 独立 terminal acceptance 后停止；`next_stage_started` 必须为 false。
10. 按第 11 节汇报，等待下一段决定。

如 future outer summary 再次与 terminal acceptance 不一致，必须停止，不得继续累积受控例外。

### 10.3 E1：development 40 episodes

- 使用 E0 预注册的 40 tasks x 每 task 1 state。
- 只在 dev set 上比较冻结候选：1000/5000/10000/15000/20000/25000/30000。
- checkpoint 选择主指标：成功 episode 数。
- 并列时优先更早 step；若仍需区分，按已冻结 E0 规则使用 adapter identity 字典序，不额外引入训练成本排序。
- dev 与 main states 不重叠。
- 不得用 main-200 或 full-2000 反复挑 checkpoint。

### 10.4 Selection lock

- 根据 dev-40 锁定唯一最终 LoRA adapter hash。
- 第一个 main episode 开始后不得增加候选、改选择规则或回看 main 调参。
- 记录 base、adapter、config/patch、norm、dataset、split 与 eval identity。

### 10.5 E2：main 200 episodes

- 在预注册 main-200 上只运行：
  - `pi0_base + canonical norm`；
  - 唯一锁定的 pure-LoRA adapter。
- official `pi0_libero` 若要从既有 2000 结果抽取相同 states，必须先验证 task/state/seed/evaluator 身份兼容；否则仅作为不完全同协议外部参考。
- 主报告 success rate，同时保存逐 task / suite 成功数、失败类型、运行资源和 identity receipts。

### 10.6 E3：可选 full 2000

- 现有 E0 仅规定 main 后可选扩展、需要单独授权且不得重选 checkpoint；未冻结数值扩展门槛。如需新增量化门槛，应在 main-200 前单独决定并记录，不能事后称其已预注册。
- 只运行已经锁定的最终 LoRA，不再挑 checkpoint。
- 这是高时长 GPU/仿真阶段，必须使用自治执行并单独预算输出和视频。

### 10.7 F1：最终交付

最终报告至少包括：

- 实验问题与严格口径。
- base / pure-LoRA / official reference 的 normalization 差异。
- trainable/total 参数量、显存、时长、存储。
- dev 选择规则与 main 盲测结果。
- 每 task/suite success rate 和置信边界（如计算）。
- checkpoint/adapter/base/norm/dataset/config/commit/patch/split identities。
- FT1/FT2 历史 outer orchestration 偏差及是否采用受控例外。
- 未证明事项：不得把单 seed、固定协议结果扩大为论文完整复现或普遍结论。

## 11. 每阶段固定汇报格式

实现对话每次完成一个阶段后，必须同时写“本阶段”和“下一阶段”。

### 本阶段

1. 实际完成了什么；明确是真实运行、静态准备、fake test 还是只读检查。
2. 产物与 evidence 的绝对路径、关键 hash/identity。
3. 测试、资源、GPU/CPU/RAM、存储与进程退出情况。
4. 失败 attempt、偏差和未解决问题；不得只报最终成功。
5. 明确没有执行的范围。
6. 当前现场：未提交文件、partial、automation/tmux/guard、远端进程和 Git 状态。

### 下一阶段

1. 阶段名称与目标。
2. 依赖与输入身份。
3. 计划动作及是否涉及模型、数据、GPU、长任务、下载或写入。
4. 预计资源和存储峰值。
5. 风险、停止条件、失败恢复方式。
6. 完成标准。
7. 末尾固定说明：

> 以上是下一阶段说明，不代表已授权执行；等待用户本人决定。

## 12. Git 与证据保存规则

- 任务过程中以文件、evidence、manifest、status、summary 和 pending-change 记录为主。
- 只有用户明确提出“提交”时，才执行 Git staging/commit/push。
- 不按每个小阶段反复提交；用户要求提交时，再按相似阶段集中整理。
- 提交前排除 `.log`、`.pyc`、`.DS_Store`、滚动 events 等不适合进 Git 的运行副产物，除非用户明确指定。
- 不使用 `git reset --hard`、`git checkout --` 或其他破坏性回滚处理用户现场。
- 不覆盖 evidence；新 attempt 必须 collision-safe。
- 旧失败、超时、错误 summary 和中断产物是审计链的一部分，不得删除或改写。

## 13. GPT-6 接手后的第一个决策点

接手后不要直接启动 FT3。先向用户给出当前事实：

```text
FT1 runner/terminal: pass
FT1 outer summary:   fail (child_exit_nonzero)

FT2 runner/terminal: pass
FT2 outer summary:   fail (committed_step_mismatch)

progress 原子更新修复：已实现并通过本地/远端 CPU 测试，尚未 Git 提交
FT3: 未启动
```

然后等待用户决定：

- 接受受控例外，后续从已验收的 step 5000 开始 FT3；或
- 保留现有轨迹并从 `pi0_base` 新建正式轨迹重跑。

如果用户选择继续 FT3，仍需把“FT3 的新鲜前检与正式启动”视为新的具体阶段，并按最新用户指令取得对应决定；不能把路线选择当成 GPU 启动授权。

## 14. 明确禁止的误读

- “terminal acceptance pass”不等于“outer orchestrator 历史记录也 pass”。
- “训练到了 step 5000”不等于“已完成 30k 正式训练”。
- T1 `100 -> 200` 不是候选训练。
- 工具实现、CPU fake test、静态模板或新鲜 preflight 不等于真实训练。
- `pi0_base + LIBERO norm` 不是严格 zero-shot。
- 官方 low-memory LoRA 配置不是 pure-LoRA-only。
- 用户批准存储上限不等于批准下载、训练、评测、checkpoint 删除或 Git 提交。
- 本文件中的下一阶段描述不构成执行授权。

---

GPT-6 应以真实 evidence 和最新实现对话为准；如果它们与本快照冲突，先只读查明时间顺序和身份，再向用户报告，不得静默选择其一。
