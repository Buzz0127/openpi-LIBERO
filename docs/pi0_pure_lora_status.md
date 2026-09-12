# pi0_base → LIBERO pure-LoRA 状态

> 2026-09-11 更新：FT3–FT7 已完成至 step 30000；E-PREP 的 CPU-only 候选登记、episode 记账、selection lock 与自有进程组 fake 回收已实现并测试。远端已发布不可变的 7-candidate index、E1 dev-40 non-executing plan 与 readiness index；E1 尚未启动，仍须重新 GPU 前检与单独授权。训练 result 只保存 `metrics_count/all_metrics_finite`，未持久化逐 step loss/grad-norm 序列；不得补造曲线。

## 2026-09-11 E-PREP completion record (CPU-only)

Remote evidence attempt: `/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/e-prep/attempt-20260911T-EPREP-A1`.

- `registration.json` remains the first immutable seven-candidate registry: steps
  `1000/5000/10000/15000/20000/25000/30000`; its internal identity is
  `c8b07708443588416c5f163fe054256b1b65ff8362a8abdfe5e468028e23babf`.
- `candidate_index.json` binds that registry to the base/norm/Golden/config/
  split/freeze identities, the C-FT2 decision, the E0 manifest and the fixed
  remote source/tool-snapshot identities.  Its identity is
  `98adc74cb217c8bb18f66d4f8c9e6cb1834de1e967271b4021637bb010d7c895`.
- `e1_dev40_plan.json` is an inert plan (`execution_authorized=false`) for
  seven candidates times the same forty E0 development entries, therefore
  exactly 280 future denominator keys.  It binds static Policy Server →
  `127.0.0.1:{port}` WebSocket → evaluator wiring and task-owned PGID cleanup;
  its identity is `88d8be07341ba04859596d739588c7acc05f86178b10adf5dbd732a1fc9ac9b4`
  and file SHA-256 is `1e27a7d5e84429f4cc5899b3fc24154bae3699a5d49c101ec0878f68f6912cef`.
- `readiness_index_repaired.json` is the valid cross-check artifact, with
  status `prepared_not_authorized` and identity
  `fee037677a9d1fb0ab04394ea6c9adf010f37f70b380b6244bd721bf14863384`.
  It preserves the hash of the earlier malformed `readiness_index.json` rather
  than replacing it; only the repaired file is an admissible readiness record.

The local control tests use only temporary files and fake child processes.
They prove registration/plan/ledger/selection-lock semantics and that cleanup
targets only child PGIDs created by the test.  They do not prove a model can
load, a server can listen, WebSocket inference can return finite actions, EGL
can render, or any LIBERO episode can succeed.

> 2026-09-10 当前入口更新：FT1/FT2 runner 与 terminal acceptance 已通过，但 outer summary 分别保留 `child_exit_nonzero` / `committed_step_mismatch` 失败。progress 修复已实现、CPU 测试通过，尚未提交；当前等待 C-FT2 处置决定，FT3 未启动。下文 2026-09-09 内容保留为历史快照，其中“FT1 未开始”等不再代表当前状态。最新事实见 [继承入口](pi0_pure_lora_gpt6_inheritance.md)，后续候选安排见 [项目完成路线](pi0_pure_lora_completion_route.md)。路线规划不构成执行授权；Git 操作仅在用户明确要求提交时进行，历史段落中的提交例外不自动授权。

更新：2026-09-09（T1 工程恢复验证通过；FT0 正式训练启动前静态准备完成）

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
- T1 启动前冻结包：无 GPU 地固定了一个非候选工程验证段 `100→200`，只用于验证 S1d step 100 resume、data-loader 序列连续性和 A2 远端自治组合；正式训练另起新的 base 轨迹，候选 step 固定为 `1000/5000/10000/15000/20000/25000/30000`，训练 seed 42 与评测 seed 7 继续分离。冻结包 identity 为 `0d1521a326e1f951f4f89f1660090a88ea2e1b162271cf4ef258b0fcdbd44939`，文件 SHA-256 为 `8fc786a6ad4f041aa9593785505d21d0fe9095b92c4a6218dd64415dbad80275`。
- T1 存储冻结：当前计费 91,765,739,008 B；按 step 200 工程制品加 7 个正式候选全部不删除、每个 full-state 5,559,083,375 B、adapter 199,962,483 B，并额外预留一份 full-state 原子落盘和 1,000,000,000 B margin，最坏峰值为 144,397,189,247 B。距 soft/hard 仍有 95,602,810,753 B / 105,602,810,753 B，不触及 225,000,000,000 B review line。
- T1 resume runner CPU 阶段：实现 `resume_sequence.py` 和 `run_t1_resume_segment.py`。runner 对两个独立、同 seed loader 做 batch fingerprint 比对，step 100 精确跳过 100 个 batch 后以索引 100 开始，下一 batch 为 101；同时重放 100 次 S1d RNG split，运行前复核 acceptance/manifest 身份，并重新流式哈希 checkpoint 与 adapter 树。
- T1 runner 静态验收：本地、远端固定 Python 各 11 项测试通过；覆盖无限 epoch 定位、第二 loader 漂移、短 loader、负 step、多卡映射、receipt 篡改、checkpoint 文件篡改及目标碰撞。未读取真实 dataset/checkpoint，未导入真实 Pi0，未使用 GPU。测试运行时绑定的是 uncommitted file hashes；用户随后明确要求执行独立 Git consolidation，相关源码与证据纳入包含本状态文档的提交，但既有 readiness evidence 保留其运行时原始表述。
- T1 execution-control CPU 收尾：已实现静态模板、新鲜前检后封装、独立 terminal verifier 及三层 orchestrator→storage guard→GPU guard 异常回收。本地与远端固定 OpenPI Python 各完成 `py_compile + 110` 项 CPU-only fake 测试，67 个源文件的本地/远端哈希映射完全一致，测试期间源码稳定。静态模板无 command/environment/GPU；即使绑定合格 fake preflight，plan 仍为 `execution_authorized=false` 且不自动启动下一阶段。
- T1 execution-control 部署/静态模板：源码与证据已提交并推送为 `7be18dd407869ae34d521441d7403dbdca5573c5`。远端独立工具快照含 67 文件、占 516,160 B，逐项哈希与 archive `2798b59623fc19b16f4237c25585565c4b63834905d5b681691a1b24a0accc92` 一致；静态模板 SHA-256 为 `38b27c6cbd5863a5cef3f0211dea66c8478224a84e45a7bbf1fbec59104b5294`，identity 为 `1dca5bce43b0f28b27dd566d9f8acc638550ebacd9482334d7e02e850356e3cd`。模板仍无 command/environment/GPU 选择，保持 `execution_ready=false` 与 `execution_authorized=false`；未读取/重哈希真实 5.5 GB checkpoint，未解码真实 LIBERO 数据，未导入真实 Pi0，也未训练。
- T1 前检/plan 封存：一次约 34.4 秒的 30 样本双卡+CPU/RAM 前检通过，最终选定 GPU 1（UUID `GPU-14900654-ea51-b7aa-28d3-b2885502d727`，最新样本空闲显存约 99.98%、利用率 0%；CPU 可用内存约 269.7 GB、load/CPU 约 0.0071）。它只用于封存 exact non-authorizing plan，preflight identity 为 `c7117fe7ded147e13c2e1a01af0835cb57c63607ff8588080a5b0bae95a34d4f`，plan SHA-256 为 `f0e876e4cc6649de8fda5198ab69e5fbc8e38c1e84399b375d9dfe7e1875fcf8`，identity 为 `f113b31048ad2406fad4820c5d3d1084223030c091908e4161ba83d85fd6b128`。plan 的 command 未执行，`execution_authorized=false`；前检有效窗为 120 秒，因此真实启动前必须重新采样，不能复用该 GPU 选择。
- T1 `100→200` 工程恢复验证段：已从 S1d step 100 恢复并执行 100 个 step 至 step 200，`status/summary=pass`、`exit_code=0`、`next_stage_started=false`。100 组 metrics 均有限；20/20 Golden adapter 叶子变化、50/50 非 Golden 叶子不变；loader 精确 skip 100 且 RNG 重放 100 次一致。step 100/200 两份 full train-state 和两份 adapter 均保留，未自动 pruning 或删除。最终独立 terminal acceptance 为 `pass`，report identity `1fcade355ab5c56c454448e407dab30a1f46b1bb1ac92aa5495b07f660f9c6f4`。这是预注册的 `candidate=false` 工程恢复验证，不表示正式训练、收敛或性能提升。
- FT0 正式训练冻结：CPU-only 非执行包已将新 `pi0_base` 轨迹、seed 42、batch size 1、workers 0、AdamW 与 30k cosine schedule、候选 step `1000/5000/10000/15000/20000/25000/30000`、adapter-only 发布及 full-state 安全轮换冻结。包 identity 为 `723ede183f745fc45e3709ad91fe8dd39e1e1b487fadb0257db4ddd92e402712`，最坏峰值为 `144,404,504,245 B`，仍低于 `225,000,000,000 B` review line。CPU-only formal-segment contract 已验证新 base `0→1000` 与后续已验证候选恢复边界，拒绝工程 step-200 根复用、多卡映射、未注册边界和缺失前一段制品。actual formal runner 已实现，并在进入 JAX/OpenPI 前执行该合同、身份和 collision-safe 输出门禁；其 GPU 训练路径尚未真实执行或验收。它保持 `execution_authorized=false` 与 `execution_ready=false`；新的 GPU preflight 和 FT1 授权仍是阻塞条件。
- FT1 正式首段静态准备：已生成不可执行的 `0→1000` 模板，绑定 FT0 freeze、固定 OpenPI source、8 个工具 SHA-256、运行/adapter 根、存储政策和“不得自动进入下一段”规则；模板的 command、environment、GPU 均为 `null`，且 `execution_authorized=false`。独立终态验收器会拒绝身份漂移、非有限 metrics、任何非 Golden 叶子变化、错误 checkpoint 历史、未逐值恢复参数/optimizer、adapter 组合缺证或自动启动下一段；报告原子创建且拒绝覆盖。本地与远端固定 Python 均通过 31 项 CPU-only fake 测试。已从提交 `42ec3e8a7d59fa0a01192caf404f9cf76618a6d8` 部署 1,025 文件、11.3 MB 的只读远端快照；前后逐文件 SHA-256 一致，远端 OpenPI 仍为干净的 `3619c35ffdcbfe97ae735de175d91c2fb67a899d`。实际启动时必须重新做约 30 秒双卡+CPU/RAM 前检（旧前检 120 秒后即失效）并由用户单独授权。

## 失败尝试、警告与偏差（均保留证据）

- 首次 S1b 在 bfloat16 buffer hash 处失败，之后改为 `tobytes(order="C")` 并通过回归测试。
- C0 首次测试命令遗漏工作树根目录的 `PYTHONPATH`，修正后相关 16 个测试通过。
- C0 首次远端 commit 因未配置作者身份失败；随后仅对单次提交使用与本地及父提交一致的显式作者环境变量。
- 远端 `origin` 是官方 Physical-Intelligence 仓库，且非交互 `ls-remote` 超时；C0 commit 未 push，也未绕过凭据或改写 remote。
- S1d stderr 出现两类非阻断警告：固定 LeRobot v2.0 数据仍使用 global stats；Orbax 在同一拓扑 restore 时从 checkpoint 文件补读 sharding metadata。为保持已验证的数据身份与 canonical normalization，未执行数据格式转换；跨拓扑 restore 仍须单独验证。
- S1d 在新的“长任务远端自治”规则到达前已经完成。它有远端双层 guard、独立进程组、collision-safe attempt、结构化日志和完整 restore，但由前台 SSH 启动，且没有持续原子 `status.json/current.json` 与独立心跳，因此不宣称完全符合新规则，也不为补形式重跑。
- A2 首次只读终态命令假定存在 `exit_code.json`，实际 schema 使用 `exit_code.txt`，因此在打印已通过的 `status/summary` 后提前退出。未修改远端 attempt；随后通过文件枚举和独立验证器按真实 schema 完成验收。
- A2 lightweight evidence 首次 `scp` 花括号列表未按预期展开，仅先复制了 plan；随后整目录复制成功。远端原始证据未覆盖，本地 `.log` 文件受 Git ignore 排除。
- T1 冻结器首次本地测试使用包式模块路径，但 `tools/pi0_pure_lora` 不是 Python package，测试收集失败且没有执行 T1 逻辑；改用既有 `unittest discover -s tools/pi0_pure_lora` 后本地、远端各 4 项通过。
- 远端源码确认：`restore_state()` 明确丢弃 `data_loader`，通用 `train.py` 在 restore 前重新创建 iterator 并先取首个 batch。因此 `--resume` 不能自动证明 batch 序列连续；在专用 runner 完成确定性 skip/position 校验前，冻结包保持 `execution_ready=false`。
- T1 execution-control 远端 R1/R2 均保留为失败 attempt：R1 因 macOS AppleDouble `._*.py` 和缺少轻量 fixture；R2 已解决 AppleDouble 并通过 `py_compile`，但精简快照仍缺 `base_model_manifest_c0.json` 和 Golden manifest。R3 补齐后 110/110 通过；这些是打包 fixture 问题，不是真实训练或 execution-control 逻辑验收失败。
- T1 首次真实启动 fail-closed：新鲜前检与 exact plan 均已封存，但 `launch_autonomous_stage.py` 曾在真正创建 `tmux` 前将嵌套 storage guard/runner 的多个 `--attempt-dir` 误视为顶层参数，拒绝启动。无 tmux、训练 attempt、模型加载或本任务 GPU 训练进程残留。该解析错误已修复、测试、提交并部署；后续同一工程段已在新 preflight/plan 下完成并通过独立验收。

## 已收敛现场

- 远端唯一 LoRA worktree：`/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`。
- 远端分支：`feature/pi0-libero-pure-lora`，C0 后工作树干净。
- 原始 `/home/wengzr/projects/openpi` 保持只读；既有 `outputs/` 未触碰。
- 稳定初始化 manifest：`manifests/pi0_pure_lora/base_model_manifest_c0.json`。
- C0 runtime identity：`1c289cc470e064d6717513e149b5cae03ee71b56f7c9634df450e496ca46c958`。
- C0 model identity：`d484ef5fa06bcb92b0dad92d1f221d4b65406f86dd928569362cb8a9106213ac`。

## 未开始 / 未授权

- FT1 `0→1000` 正式首段：必须从 FT0 冻结的新 `pi0_base` run root 开始，不能从工程 step 200 继续；在已提交工具的 immutable remote snapshot 上重新做约 30 秒双卡+CPU/RAM 前检，并获得该单段明确授权后才可启动。前检是时间敏感的运行门禁，不能在训练前很早预做或复用。
- FT2 后续正式 segments：`1000→5000→10000→15000→20000→25000→30000` 每段均须单独授权，完成当前段后停止；不得自动进入下一段，也不得在新 full-state restore 通过前删除最后已知良好状态。
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

A2 的自治协议与 synthetic 断连 smoke 已完成；具体 T1 candidate step、adapter milestone 和 full-state 轮换实例必须引用本次 A2 提交后的 committed tool identity，因此移到 T1 启动前的无 GPU 授权包中冻结。这个依赖顺序调整不授权 T1，也不弱化其预注册要求。

每个 GPU 阶段仍须重新做约 30 秒双卡与 CPU/RAM 前检、动态固定单卡、关闭 JAX 预分配，并只由已验证 guard 控制其自身进程组。
