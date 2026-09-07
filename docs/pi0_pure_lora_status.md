# pi0_base → LIBERO pure-LoRA 状态

更新：2026-09-07（A2）

## 固定实验定义

- 初始化：`pi0_base`。
- 主比较：Base no-gradient baseline 与 pure-LoRA 使用同一份 canonical LIBERO normalization。
- pure-LoRA 真值：独立 Golden manifest 中恰好 20 个 adapter 叶子、49,987,584 个参数；其他参数全部冻结。
- `pi0_libero` 只作为使用其自带 normalization 的端到端外部参考。

## 已验证

- R0/G1/G2：精确 `PathIn` 冻结规则通过 Golden 校验；20 个 Golden 叶子可训练，50 个非 Golden 叶子冻结。
- D1c-Rb：固定 LIBERO revision 的单份 raw snapshot 与单份 loader-required Arrow cache 可离线复用。
- N1：canonical normalization 已从固定训练 split 和 repack/delta 链生成并验证。
- B1/I1/I2/I3：pi0_base 身份、adapter-only 组合、checkpoint 控制平面与无自动 pruning 规则已验证。
- S1a：真实 checkpoint load 与 train-state 初始化通过。
- S1b：真实 LIBERO batch 的单步更新通过；20 个 Golden 叶子变化，50 个非 Golden 叶子不变。
- C0：远端 OpenPI LoRA 源码已提交为 `3619c35ffdcbfe97ae735de175d91c2fb67a899d`。完整 train-state restore 现在逐叶验证全部 parameter 和 optimizer 值，并在任何 mismatch 时禁止发布 adapter/receipt。
- C0 cache reconciliation：HF 总量仍为 80,604,459,314 B、5,775 文件；唯一 raw dataset 为 1,699 个逻辑文件/34,938,927,454 B，唯一 Arrow tree 为 71 文件/34,941,009,190 B。S1b 的空 after-root 文件属于采集失败，不代表缓存被删除。
- E0：已用 outcome-blind SHA-256 排序预注册 40 条 development 与 200 条 main episode；每个 suite/task 的 development 1 个 state 与 main 5 个 state 严格互斥。实际 evaluator 的 split gate 接受全部 80 个 suite/task/split 分组。
- E0 checkpoint 选择规则：候选 T1 segment-end step 必须在 T1 开始前固定；仅用 development 40 条，以成功数最高为主；并列时依次选择更早 train step、字典序更小的 adapter identity。首次 development episode 后不得追加候选，选择结果在 main 前锁定，main 不得用于重新选 checkpoint 或调参。
- 既有 `pi0_libero` seed-7 汇总投影到相同 E0 states 后为 development 38/40、main 190/200；它继续使用 checkpoint-owned normalization，仅是端到端外部参考，不属于 Base/pure-LoRA 受控 normalization 比较。
- S1c：真实 LIBERO batch 的 10-step 稳定性 smoke 通过。10 组 loss/grad norm 全部有限，train-state 从 step 0 到 10；20/20 Golden adapter 叶子变化，50/50 非 Golden 叶子哈希不变。
- S1c guard：30 样本前检后固定物理 GPU 1；运行期 133 个样本，最高利用率 55%、最低空闲显存 51.78%，无 pause/resume、monitor error 或 memory emergency。child exit 0，无残留本任务进程。
- S1c 未写 full train-state 或 adapter；共享 LIBERO cache、OpenPI cache 与 uv cache 均零增长，本阶段正向存储增量 2,006,046 B（主要为独立 JAX compile cache 与证据）。
- S1d：真实 LIBERO shuffled batch、seed 42、batch size 1、workers 0 的 100-step smoke 通过。100 组 metrics 全部有限，loss 从首步 0.221992 到末步 0.060486（本结果只证明有限性与可训练性，不作为收敛或性能提升声明）。20/20 Golden adapter 叶子变化，50/50 非 Golden 叶子逐值哈希不变。
- S1d full train-state：step 100 checkpoint 为 5,559,083,375 B。相同拓扑上恢复后，70 个 parameter 叶子与 42 个 optimizer 叶子的树、shape、dtype 和每个值全部一致；无自动 pruning、无旧 checkpoint 删除。
- S1d adapter：adapter-only 制品为 199,962,483 B、恰好 20 个 Golden 叶子，identity 为 `26dc0261e16bb7ca8f77a7ea8113f22c5dab6d264f4e51861cbd003e80b543c1`。base + adapter 重新组合后逐参数哈希与训练末态一致。
- S1d guard：30 样本前检后固定物理 GPU 1（`GPU-14900654-ea51-b7aa-28d3-b2885502d727`）；运行期 227 个样本，最高利用率 78%、最低空闲显存 43.41%，无 pause/resume、monitor error 或 memory emergency。两层 guard 均 child exit 0、无 TERM/KILL、子进程已回收，无本任务 GPU 残留。
- S1d 存储：正向增量 5,765,739,008 B，结束时 LoRA 计费占用 91,765,739,008 B；soft/hard headroom 分别为 148,234,260,992 B / 158,234,260,992 B。LIBERO cache、OpenPI cache 与 uv cache 均零增长。远端 checkpoint 和 adapter 保留，未复制进 Git、未删除。
- S1d 独立验收：42 项 fail-closed 检查全部通过；checkpoint 24 个文件与 adapter 22 个文件均完成稳定流式 SHA-256 身份清单。验收报告 identity 为 `c560bb932159eaa28502f89560ed1a325422f41f9aec249c67e30764541f4d91`。
- A2 自治基础设施：实现 immutable plan、精确 source/input/tool identity 绑定、collision-safe attempt、原子 `status/current/heartbeat`、离线环境、有限 timeout/retry、独立 child PGID、只控制自有进程组、TERM→KILL→wait 回收、有界轮转日志和终态 output hash manifest。自动化明确 `next_stage_auto_start=false`。
- A2 断连存活 smoke：服务器 `tmux` 中的 10 秒 synthetic probe 在 launcher 观察到递增心跳 `3→4` 后脱离 SSH；复连时已完成 10/10 步、零重试、child PID=PGID 且已回收，`tmux` 自然退出。未联网、未加载模型/数据/checkpoint、未使用 GPU，也未启动 T1。
- A2 fail-closed 验收：本地完整 A2 测试 11 项通过；远端验证器测试 2 项通过；真实 attempt 的独立 acceptance report 为 `pass`，plan 文件 SHA-256 为 `1653529fef0274d4164e1a54994104b841a2caee896d380e6cbb5d240fa3d719`，内部 identity 为 `9707093c81eb4c39856c6dc0135888c4f1776b9b0fd74d265af1aafd634ee28c`。A2 远端证据总量 289,713 B。

## 失败尝试、警告与偏差（均保留证据）

- 首次 S1b 在 bfloat16 buffer hash 处失败，之后改为 `tobytes(order="C")` 并通过回归测试。
- C0 首次测试命令遗漏工作树根目录的 `PYTHONPATH`，修正后相关 16 个测试通过。
- C0 首次远端 commit 因未配置作者身份失败；随后仅对单次提交使用与本地及父提交一致的显式作者环境变量。
- 远端 `origin` 是官方 Physical-Intelligence 仓库，且非交互 `ls-remote` 超时；C0 commit 未 push，也未绕过凭据或改写 remote。
- S1d stderr 出现两类非阻断警告：固定 LeRobot v2.0 数据仍使用 global stats；Orbax 在同一拓扑 restore 时从 checkpoint 文件补读 sharding metadata。为保持已验证的数据身份与 canonical normalization，未执行数据格式转换；跨拓扑 restore 仍须单独验证。
- S1d 在新的“长任务远端自治”规则到达前已经完成。它有远端双层 guard、独立进程组、collision-safe attempt、结构化日志和完整 restore，但由前台 SSH 启动，且没有持续原子 `status.json/current.json` 与独立心跳，因此不宣称完全符合新规则，也不为补形式重跑。
- A2 首次只读终态命令假定存在 `exit_code.json`，实际 schema 使用 `exit_code.txt`，因此在打印已通过的 `status/summary` 后提前退出。未修改远端 attempt；随后通过文件枚举和独立验证器按真实 schema 完成验收。
- A2 lightweight evidence 首次 `scp` 花括号列表未按预期展开，仅先复制了 plan；随后整目录复制成功。远端原始证据未覆盖，本地 `.log` 文件受 Git ignore 排除。

## 已收敛现场

- 远端唯一 LoRA worktree：`/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`。
- 远端分支：`feature/pi0-libero-pure-lora`，C0 后工作树干净。
- 原始 `/home/wengzr/projects/openpi` 保持只读；既有 `outputs/` 未触碰。
- 稳定初始化 manifest：`manifests/pi0_pure_lora/base_model_manifest_c0.json`。
- C0 runtime identity：`1c289cc470e064d6717513e149b5cae03ee71b56f7c9634df450e496ca46c958`。
- C0 model identity：`d484ef5fa06bcb92b0dad92d1f221d4b65406f86dd928569362cb8a9106213ac`。

## 未开始

- T1：正式分段训练；先为首个有界短分段冻结 committed source、候选 segment-end step、adapter milestone、full-state 安全轮换与存储峰值计划，再单独申请 GPU 执行授权。A2 完成不会自动授权或启动 T1。
- E1 40-episode dev、E2 200-episode main、E3 可选 2,000-episode 评测。

## 当前存储政策

- v1/共享基础快照 24,574,841,856 B 排除且只读。
- LoRA 独立硬上限：250,000,000,000 B。
- 新阶段复核线：225,000,000,000 B。
- soft stop：240,000,000,000 B；hard stop：250,000,000,000 B。
- 新阶段开始前至少保留 20,000,000,000 B 未承诺空间。
- 单项预计新增超过 10 GiB 必须单独决定。
- S1d 实测一份 full train-state 为 5,559,083,375 B；安全轮换必须按旧、新两份短暂共存计入，且新 step 完整写入、hash/restore 验证通过后，才可按另行固定的规则处理旧 step。不得用 `max_to_keep=1` 提前删除最后已知良好状态。
- T1 不得无限保留每个 full state：候选里程碑默认只发布约 0.2 GB 的 adapter；full state 只承担 resume，并按“旧+新”安全轮换。具体候选 step 与轮换/处理授权必须在 T1a 中预注册。
- 两个 2026-09-02 pi0_base 下载 shell launcher 是 historical/not reusable，并已改为 fail closed；历史 evidence 中的 200 GB 数值保持原样。

## 长任务远端自治规则

- 预计墙钟超过 15 分钟，或涉及重复资源轮询、分段重试、长下载、长训练、批量评测、checkpoint 写入/校验时，默认先实现并 smoke 远端自治自动化；只有需要即时人工判断、交互凭据/授权或尚未通过小规模 smoke 的工作才在线陪跑。
- 自动化必须运行在服务器端经审计的持久会话/服务中，不依赖 Mac、Codex、SSH、反向代理或本地 7890；服务器重启后只允许从最后已验证且已原子提交的 checkpoint 恢复，不接受半写 checkpoint。
- 每个阶段固定 pwd/branch/HEAD、模型/数据/norm/Golden/config/训练 seed/评测 seed/split identity、segment 起止、最大墙钟、心跳超时和有限重试。训练/评测保持 offline，不得静默生成第二份数据 cache。
- 主任务使用独立进程组；guard 只控制自己的任务组。GPU 工作继续执行约 30 秒双卡+CPU/RAM 前检、动态固定单卡并关闭 JAX 预分配。non-finite、OOM/ECC/Xid、Golden 外变化、身份漂移、cache 异常、guard 失效或存储越线均 fail closed。
- checkpoint 完整写入并逐值 restore 验证后才发布 adapter；不得自动删除最后已知良好状态。每个 attempt/segment 必须 collision-safe，轻量原子更新 `status.json/current.json` 与 heartbeat，详细指标写有界/轮转日志，结束生成 summary/report、exit code 和输入/输出 hash manifest。
- Codex 启动后只确认远端持久会话、主进程/guard、心跳前进与初始资源正常，随后停止实时轮询；用户可退出 Codex/关闭 Mac。返回后先读小型 status/summary/receipt/hash，只有失败或摘要不足才读日志尾部。自动化完成当前已授权阶段后必须停止，不得进入下一未授权阶段。
- T1 与 E1/E2/E3 在执行前都必须先通过上述自治能力验收。

## Git 集中保存规则

- 阶段执行期间以完成任务和生成 collision-safe 证据为主，不反复 `git add/commit/push`。运行前记录 HEAD、branch、status 和输入 identity，结束记录输出 identity；未提交现场仍须能区分阶段、失败 attempt 和未执行范围。
- 默认每 2–3 个语义相近阶段验收通过后进入一次集中保存节点：运行相关测试、`py_compile`、JSON/manifest 校验与 `git diff --check`，排除 `.log`、`.pyc`、`.DS_Store`、临时文件和既有无关修改，再统一更新状态文档、按职责提交并集中 push。
- 减少提交频率不得覆盖旧 evidence、混入用户修改或弱化测试、身份、恢复与安全标准。训练 segment 的恢复以 checkpoint、adapter、receipt 和 segment manifest 为准，不依赖每个 segment 的 Git commit。
- 只有切换 branch/worktree、即将开始昂贵或不可轻易重复的长任务、已验证重要修复可能被覆盖、用户要求立即保存，或必须用 commit hash 构造身份时，才在安全边界提前提交。
- 远端自治运行前必须绑定最近一个已批准的 committed source identity；自动化运行中不得自行 commit/push，且完成当前授权阶段后必须停止。
- 保存分组：A2 自治基础设施与 T1 首个短分段验证属于“自动化分段训练基础设施”组；T1 后续通常每 2–3 个里程碑或候选集合集中保存轻量摘要；E1 + checkpoint selection + E2 属于评测组；E3 若执行可单独保存；F1 集中保存最终报告、复现说明与结果表。
- A2 源码因要绑定首个自治 T1 短分段的 committed identity，可在 A2 静态/模拟验收后使用“长任务前身份固定”例外先提交；短分段运行期间不提交，其轻量结果在随后保存节点集中整理。
- S1c 已按旧规则提交，S1d 也在本规则到达前提交并推送为 `f85d605d368c44e4812222e099e6561464754797`；不重写、squash 或删除这些历史提交。此后从 A2 安全边界执行新规则。

## 后续路线

先前为落实远端自治而写成的 `T1a/T1b`，现按长期阶段命名收敛为：

`A2(自治 orchestrator + 候选 step/存储轮换预注册与 smoke) → T1(首个短分段验证 → 后续正式 segments) → E1(dev 40) → E2(main 200) → E3(可选 2000)`

这次只是把执行控制阶段统一命名为 A2：A2 仍只补齐自治、预注册和恢复控制，T1 仍是训练本身；pure-LoRA 研究目标、Golden 定义和 250 GB 独立增量政策均不改变。

每个 GPU 阶段仍须重新做约 30 秒双卡与 CPU/RAM 前检、动态固定单卡、关闭 JAX 预分配，并只由已验证 guard 控制其自身进程组。
