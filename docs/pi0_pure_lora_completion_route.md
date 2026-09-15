# pure-LoRA 项目完成路线：从 step 5000 到可复核总结

更新：2026-09-10（Asia/Shanghai）
状态：候选路线 / 未授权执行。本文是本轮路线规划产物，不是 C-FT2 例外接受书，也不是 FT3、评测、部署、删除或 Git 提交授权。

> 2026-09-13 评测性能路线修订：本节之前的 FT3→E2→F1 历史路线不改写；FT3–FT7、E1、
> selection lock、E2 和 F1 已有的证据及历史失败也不因性能问题失效。E3 的唯一锁定
> step-25000 full-2000 attempt 正在独立 guard/tmux 中运行，严禁热替换、擅停、重选
> checkpoint 或用新协议拼接旧结果。本文件新增 P1→P2→P3，是对“当前 E3 推理远慢于
> 历史官方参考”的因果审计与受控处置路线；不是执行授权。

> 2026-09-14 安全策略修订：未来 pure-LoRA 启动入口使用 memory-only guard policy：GPU
> utilization 是遥测而非硬门禁，启动/选卡只由 CPU/RAM 与空闲显存 >15% 决定；运行期仅
> <=15% pause、>=20% 连续五样本 resume、<=10% terminate，连同 OOM/ECC/Xid/监控失败、
> 墙钟、存储与自有 PGID 故障处理。历史 v1/已有证据不重写。当前 E3 使用的 immutable
> guard 不支持 reload，维持旧 95/85 规则直至自然结束或另行获准的、不可拼接的迁移。
> 其 2026-09-14 审计为 3,697 次 util-only pause/resume、20,940.137 秒暂停（24.261%）；
> 这是实证的速度开销，和 P1 的参数驻留假设分开。

> E3 G2 已执行：用户授权停止旧 attempt 后，382 个完整唯一结果作为 partial 保存（19
> success、0 evaluator exception），不与任何未来 attempt 拼接为 2000。旧 runner 保留
> `status=running` 的历史记录；独立 closeout 记录自有 guard 的 TERM→reap、端口 18001
> 释放及未计入的在途 episode。下一技术候选是 G3 的受控 P1 A/B，但需按路线授权单独启动。

> G3/P1 已执行并停止：host/device 参数驻留 A/B 的模型身份和 70-leaf 参数树一致，但固定
> 60 个请求的 action hash 60/60 不一致（max absolute difference=0.4323568782）。所以观察到的
> host 1384.907 ms 与 device 92.965 ms 稳态 RPC 不能作为速度提升结论，也不能进入 E3/P2。
> 该不通过的有界证据与关闭决定见
> `/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/g3/attempt-20260914T-G3-AB-R4/closeout.json`；
> 后续仅可在新的、明确授权的“解释动作差异”调查包中继续。

> O1 官方 dense 推理适配已完成 CPU 工作包：新入口将已验证的 20 个 adapter leaves 在内存中
> 合并到 ordinary `Pi0Config` 的 10 个 dense kernels，随后采用官方 BF16/JAX/Policy/transform/
> WebSocket 链，同时继续使用项目 canonical norm。固定官方 LoRA 小张量 oracle、零 adapter、
> 坏映射拒绝、RNG 兼容注入与 ordinary dense abstract tree（50 leaves / 3,238,048,528 参数）
> 已通过；没有 checkpoint/model/GPU 操作。这里的 `not-authorized-not-runnable` 是当时 O2
> 控制包的历史状态；后续 R23/R25 closeout 不改写该记录。完整矩阵与当前结论见
> [O1 官方 dense runtime 说明](pi0_pure_lora_official_dense_runtime.md)。

> 2026-09-15 strict-equivalence closeout：O2 的 R23 受保护 GPU 诊断显示 merged-dense
> 同实例重复精确，但 unmerged-versus-merged 的 physical action max-abs 为
> `1.644405388134629`；R25 的固定 OpenPI CPU bf16 oracle 也不逐字节相同。用户选择保持
> strict exact equivalence，故 O2 fused-dense 性能路线已关闭，不重试、不设置事后容差、
> 不产生性能结论。详见 [strict-exact 候选登记](pi0_pure_lora_strict_equivalence_performance_register.md)。

## 0A. P1：参数驻留修复与性能因果验收（已执行，未通过动作等价验收）

### 已知事实与严格表述

- E3 服务快照 `serve_pure_lora_policy.py` 的 SHA-256 为
  `7f233d93df1c6589ea535bfe1af2b123aac20f83598a6371e3aeff74b353cc23`。其 base
  以 `restore_type=np.ndarray` 恢复，composed base + adapter 保持 NumPy tree 并直接
  `config.model.load`；当前 `module_jit` 的 state 仍是动态实参。官方路径则以
  `jax.Array`、bf16 和 device sharding 恢复。故“参数未一次性驻留 GPU、重复 host→device
  传输”是首要源码嫌疑，尚不是已测量因果结论。
- 已有 129 集的排除首请求计时快照：`server_infer` mean 1925.279 ms、median
  1422.954 ms、p95 7071.221 ms；同步 `client_request_wall` mean 1930.597 ms、median
  1427.354 ms、p95 7077.601 ms。每集平均 232.231 s、101.465 requests。历史官方校准的
  随后请求约 0.10–0.13 s 只能说明异常量级：输入、任务组合、负载和实现并未形成受控
  A/B，不能公布固定加速倍率。`policy_infer` 在 NumPy 同步前结束，正式验收以同步
  wall/server/client 时间为准。
- 现有 JIT 与 prefix KV cache 均存在，`sample_actions` 默认 10 steps，E3 的
  `replan_steps=5` 与原协议一致。不得把问题误写成“未 JIT”或“flow steps 过多”。当前
  dtype 策略也应先保持原值；Pi0 计算 dtype 为 bf16，但尚无逐叶 dtype 清单，禁止把当前
  模型笼统称为全量 FP32。
- 多环境并发/动态 batch 移出主线：上游 server 在单一 event loop 中同步调用
  `policy.infer`，没有 batch queue；而 Policy 每请求推进全局 RNG。直接并发既不能保证
  真正 batch，也会改变随机 key 到 episode 的映射。

### P1 的受控实现与真实验收

1. 从活动快照复现加载路径。在完成 Golden、identity、shape、dtype 和内容校验后，仅对
   整棵 composed tree 一次性 `jax.device_put` 到选定单 GPU，再构建 model/Policy。必须
   断言全部运行时参数叶为该 device 上的 `jax.Array`；不得只搬 adapter，且不得重写
   adapter hash。
2. 保持参数值/dtype、模型配置、canonical norm、flow sampling、`replan_steps`、RNG 和
   WebSocket 接口不变；记录逐叶 dtype、bytes、placement 与 runtime identity，并审计是否
   仍有后续转回 NumPy 的路径。
3. CPU 静态检查、最小 diff、单测、证据及文档为一个连续闭环。真实性能结论仅来自另行
   授权的有界单卡 OLD-host vs NEW-device A/B：顺序加载避免双模型显存峰值，同一固定观察、
   相同显式 noise/PRNG keys、10 次 warmup + 50 次同步计时、总墙钟不超过 45 分钟。
   记录冷加载/编译与稳定推理/RPC mean、median、p95、样本数、显存和资源采样；禁止视频、
   大张量落盘、下载、dev/main 消耗和自动扩样。
4. 使用 JAX transfer guard 或有界 profiler 区分正常输入传输与参数规模重复 H2D。若收益
   有限，按“重复编译→主机分页/CPU 压力→共享 GPU 干扰→dtype/LoRA 算子”逐项单变量检查，
   不直接跳到 batch。任一身份漂移、动作非有限/不一致、资源 guard 停止、显存或墙钟越界
   均停止并保留证据。

P1 完成标准：因果诊断报告、修复 diff、runtime/参数身份清单、动作一致性回执和受控前后
timing 表齐全。真实 A/B 前只能标“源码问题已定位、收益待验证”，不得承诺恢复官方绝对速度。

## 0B. P2：E3 衔接与完成（候选 / 未授权）

P1 后才按实测吞吐、已用墙钟和剩余 2000-entry 数量重算 E3 方案。暂停/结束旧 attempt、
启动新 runtime、恢复或重跑均需单独明确授权。旧 E3 attempt 与所有结果必须保留；当前全局
RNG 不允许假定重启后可透明续跑，必须核验准确的请求游标/随机状态。若无法证明续跑等价，
旧结果仅作为 partial evidence，新的协议必须单列新 attempt，不能拼成“完整 2000”。

## 0C. P3：最终报告更新（候选 / 未授权）

保留现有 Base `0/200`、锁定 pure-LoRA `25/200`（+12.5 pp）、E1 lock 和历史失败记录；
报告新增加载实现问题、P1 受控测速与 E3 的最终完成或有界停止结论。官方 checkpoint 继续是
历史端到端参考。若仅设备驻留不足，再另列官方式 bf16 验证候选；不得在 P1 顺手 cast、替换
E1/E2 协议或开展 batch 研究。

## 0. 入口、事实来源和当前停点

先读 [GPT-6 继承入口](pi0_pure_lora_gpt6_inheritance.md) 恢复固定身份与安全边界，再读本文安排后续工作。本文更新执行组织和交付要求；不修改已经冻结的 FT0/E0 manifest。旧 status/handoff 的历史段落不再代表最新进度。

本轮重新读取实现任务 [LoRA微调（实现）](codex://threads/01a04761-9200-7022-82f0-9f24fcd7627a) 最新五轮。最新完成回合仍为 `01a08bdd-ef0e-7d93-bc6e-a772d835a8d6`：FT2 progress 修复与后验复核。没有新的 FT3 启动记录。本对话上一轮还只读核验了远端原始 JSON、结果文件 SHA-256、checkpoint 目录和进程；本轮不将该检查包装成新的 GPU 前检或存储盘点。

| 分类 | 事实与证据 | 尚不能推出的结论 |
|---|---|---|
| 已完成且有真实证据 | FT1 0→1000、FT2 1000→5000 的 runner/terminal 均 pass；分别 1000/4000 组有限 metrics，20 Golden 改变、非 Golden 改变为 0；restore 回执记录 70 parameter + 42 optimizer leaves 逐值一致 | 不等于 30k 完成、性能提高或整条 outer 编排链通过 |
| 已实现/测试，未用于下一真实段 | progress 修复仍未提交；实现任务记录本地/远端 CPU 测试各 9/9；本轮源码 hash `a5dc334f9696cfdaf91c62fd34420cc01e696d22f294c93dd7e278635f30493c` | 单测通过不证明未来完整编排链已通过 |
| 历史失败/偏差 | FT1 outer fail=`child_exit_nonzero`；FT2 outer fail=`committed_step_mismatch`；FT2 后验观察仍为 `overall_release_ready=false`。更早的 venv 路径、canonical 编码、shell 引号失败均保留 | 不能改写历史 summary；不能默认用户已经接受受控例外 |
| 未开始/未授权 | C-FT2 路线处置尚待用户决定；FT3–FT7、候选 dev、锁定后的主比较、可选 E3 均未执行 | 本次“修改规划”不替代这些决定 |

本地基线：`feature/pi0-libero-pure-lora` / `fc30773d15eaf6ae7c4d1e1f3dad4ad05b01f2f5`，upstream 为 `origin/feature/pi0-libero-pure-lora`。本轮开始时已有两个 runner 修复文件、B1 滚动 events 改动和未跟踪的继承文档。它们均不由本轮提交或恢复。

远端证据根：`/home/wengzr/projects/openpi-lora-runs/formal-attempts/`。

| 段 | attempt 目录名 | runner result SHA-256 | terminal report identity |
|---|---|---|---|
| FT1 | `attempt-20260909T-FT1-0-1000-451b757-R2` | `c72d152b10508f376d430499be610db90432405c70dc6c26b2166d656a91dbb2` | `4263e667f1582f71311ed56bf1ce5bdfadd749611e6c630f1db050bec0ab1627` |
| FT2 | `attempt-20260910T-FT2-1000-5000-N5v2Pa` | `07e3c4bfd23e1110b5ae253db9cf7662bf56e232d88641fac11b78dca992911f` | `336e084bed25c9b48bd472bd9d11c6af9653a44069992286694a98158dfe7397` |

## 1. 完成定义：把评测和总结纳入必需交付

主目标不变：从 `pi0_base` 出发，使用 canonical LIBERO normalization，只训练独立 Golden 的 20 个 LoRA leaves；与使用相同 normalization 的 no-gradient Base 做受控比较，官方 `pi0_libero` 仅作端到端外部参考。

完整主项目完成须同时满足：

1. 按 FT0 完成 30k 轨迹，七个候选 step 的身份、restore 和处置状态完整；历史例外如被接受，应可追溯。
2. 七个候选各完成同一 dev-40，共 280 episodes；按冻结规则锁定唯一 adapter。
3. Base 和锁定 LoRA 各完成同一 main-200，共 400 episodes；episode 无遗漏、无重复、无按结果挑选重跑。
4. 生成能追溯到 episode 和制品 hash 的比较表、资源表、失败分析、实验局限和复现入口。

主线评测总量为 680 episodes，不包括另行批准的诊断与 E3。成功率不理想仍可完成实验并如实总结；不能把“必须超过官方结果”设为项目完成条件。若因预算、数据或工程障碍提前停止，应交付带停止原因的阶段报告，明确原 30k 主协议未完成，不静默缩减候选来宣称完整结项。

E3 full-2000 是可选扩展，不阻塞主报告。多 seed、额外超参数搜索、官方模型重跑均不列入本轮主线。

## 2. 相对旧路线的调整与依据

| 旧安排覆盖什么 | 问题或缺口 | 本次候选调整 | 仍未完成的部分 |
|---|---|---|---|
| C-FT2 决定后进入 FT3 | 现有单测主要证明 progress helper 与 reader 兼容；FT2 verifier 固定 `[1000,5000]` | 加入一次有边界的 R-FT CPU 准备，验证未来段的完整成功/失败收尾 | 用户处置决定、后续段工具适配和真实整链验收 |
| 静态包→单独前检包→等用户→真实启动时再次前检 | 时间敏感前检反复过期；临时脚本已有编码/路径错误 | 静态计划一次准备；单段启动获批后在同一自治启动流程内做新鲜前检、绑定、启动 | 实现此组织方式；不降低前检频率或复用过期采样 |
| FT3–FT7 后才笼统进入 E1 | 有 evaluator 不等于 policy server、adapter、EGL、自治生命周期已衔接 | 增加 E-PREP；可在训练段间准备 CPU 协议和结果汇总，不提前看 dev/main | 真实组合模型的评测接线验收 |
| F1 最后才整理 | 参数/资源/失败证据容易到最后才发现缺失 | 每段输出固定小型索引与汇总行；F1 在已有结构上填真实结果 | 汇总文件、报告正文和证据审计 |
| 继承文档笼统描述并列与扩展门槛 | E0 明确并列为早 step→hash；现有 E0 没有 full-2000 数值门槛 | 以冻结 manifest 为准，不添加训练成本排序，不称扩展门槛已冻结 | 如确需数值门槛，应在 main 前另行决定并记录版本 |

依据：[FT2 verifier](../tools/pi0_pure_lora/verify_ft2_result.py)、[progress 单测](../tools/pi0_pure_lora/test_run_formal_segment.py)、[FT0 builder](../tools/pi0_pure_lora/build_ft0_formal_training_freeze.py)、[E0 manifest](../manifests/pi0_pure_lora/e0_task_state_manifest.json)。这是一份路线缺口判断，不是对所有训练源码的全面审计。

## 3. C-FT2：先决定现有候选链如何处置

候选建议：若原始制品身份与恢复证据继续一致，接受有记录的受控例外可以避免仅为修正 outer 状态而重做已有训练。但选择权仍属于用户。

- 接受例外：保留 FT1/FT2 outer fail；记录用户决定、适用 attempt/result/acceptance hash、理由与适用范围；允许这两段成为后续输入。例外只涵盖已知历史编排失败，不涵盖身份、数据、参数冻结或 restore 失败。
- 不接受例外：保留旧轨迹，另建正式 root；FT0 重新绑定新 root，分别授权新的 0→1000、1000→5000；不得覆盖旧产物或把两条轨迹混为一条。

计划产物：新的 `route_decision.json` 或等价决策记录。只有收到用户真实决定才写 accepted/restart 状态；本文件不能充当该记录。完成标准是路线和允许输入明确，不是 GPU 已启动。

## 4. R-FT：一次完成后续段执行准备（候选 / 未授权）

目标：让 FT3–FT7 复用已有 runner、orchestrator 和双 guard，避免每段重新写临时控制器。只补必要适配，不重建通用训练平台。

输入：FT0、上一段 result/acceptance、现有 progress 修复、固定 OpenPI 与工具快照；C-FT2 决定用于绑定 FT3 的合法起点。CPU 工具准备可先做，真实输入可用性必须等决定成立。

交付及验收：

1. 让阶段计划/driver/verifier 从冻结边界读取 FT3–FT7 参数；不能将 FT2 专用 verifier 原样用于 FT3。逐段验证前一段终态、result、checkpoint/adapter 与目标边界的绑定。
2. CPU fake 流程覆盖 runner 结果成功写入→progress 到 end→driver 验收→outer summary pass；另测 result 写入失败、progress 更新失败、verifier 失败、旧/错前段身份均停止且不发布下一段许可。已有 9/9 不能代替这些整链案例。
3. 明确每类 identity 的 canonical schema（有/无末尾换行不能混用），复用对应实现；不重算或改写历史身份。保留 venv 入口路径，不 resolve 到 uv 底层解释器。
4. runner/terminal 是训练结果层；outer/guard/进程退出是执行层。对未来段，二者都通过才标记阶段完成。terminal verifier 读取的是 JSON 回执，不应宣称它独立再做了 tensor restore；真实 restore 由 runner 路径完成并被绑定。
5. 固定只读工具快照与 hash；若尚未提交，保存 HEAD、patch 和相关文件 hash。提交仅在用户明确要求时进行。

计划产物：CPU readiness、不可执行段模板、工具 hash 清单、验收案例结果。没有模型加载、GPU、数据解码或真实 checkpoint restore；预估为小型代码/证据增量，实施前给出确切清单与上界。一次准备通过后复用，只有新增变更或失败才补测。

## 5. FT3–FT7：保持冻结训练，按完整单段交付

| 阶段 | 边界 | 完成后必需产物 |
|---|---|---|
| FT3 | 5000→10000 | step-10000 full state、adapter、结果/restore/terminal/outer 记录 |
| FT4 | 10000→15000 | 同上，目标 15000 |
| FT5 | 15000→20000 | 同上，目标 20000 |
| FT6 | 20000→25000 | 同上，目标 25000 |
| FT7 | 25000→30000 | 同上，目标 30000；七候选完整性索引 |

每段的静态说明先给出输入、预算、命令结构和完成标准。用户明确授权该单段后，自治启动器重新盘点存储，在约 30 秒双卡+CPU/RAM 新鲜采样后立即绑定并启动；前检失败或过期则不启动。不要为了生成一个随后必然过期的不可执行包而重复独立前检。

继续使用单张物理 GPU/UUID、`XLA_PYTHON_CLIENT_PREALLOCATE=false`、自有进程组双 guard、离线数据和有界重试。CPU/RAM 和显存门禁遵循继承入口第 9 节。单段获批不能跨入下一段；默认不自动重启已经进入真实训练后失败的段，先辨明已提交状态并保留失败 attempt。

每段必须满足 5000 条有限 metrics、20 Golden 改变/0 non-Golden 改变、70 parameter + 42 optimizer leaves 逐值恢复一致、adapter 组合一致、旧状态安全保留、progress=end、terminal 与 outer 均 pass、任务进程退出、`next_stage_started=false`。若未来 outer/terminal 再不一致，停止调查，不自动延长历史例外。

资源规划估算：以已记录的 FT2 4000 steps / 1150 秒按比例计算，5000 steps 约 24 分钟，剩余五段约 2 小时；只是粗估，不包含共享卡暂停、重编译、复核和排队，不能作为保证或直接当 timeout。

存储估算采用 FT2 后历史账单 `109,063,857,664 B`，保留全部旧状态，增加五份 full state 与 adapter，再额外预留一份 full state 和 1 GB metadata margin：

`109,063,857,664 + 5 × (5,559,083,375 + 199,962,483) + 5,559,083,375 + 1,000,000,000 = 144,418,170,329 B`。

该数值未计后续评测视频等新增支出，也不是当前盘点。每段启动前重算；遵循 225 GB review、240 GB soft、250 GB hard、20 GB 未承诺余量和单项 >10 GiB 单独决定。主线不依赖删除旧 checkpoint 才能继续；任何删除仍需另行授权。

## 6. E-PREP：在正式评测前补齐接线与协议（候选 / 未授权）

复用 [pure-LoRA bounded evaluator](../tools/eval_libero_pure_lora_bounded.py)，不从头重写。它已有显式 Base/adapter/norm 身份、split gate、finite-action 检查和逐 episode 输出，但不负责启动 policy server；其旧 baseline 参数也需核对是否与当前共享 GPU 规则兼容。

CPU 准备可安排在训练段间，不依赖提前查看模型成功率：

- 明确 policy server 使用固定 OpenPI Python、仿真使用既有 LIBERO 环境；绑定实际 server 启动的 base/adapter/norm 与 evaluator manifest。端口连通不能代替模型身份或有限 action 验证。
- 封装一次候选的 server→WebSocket→evaluator→清理生命周期，自有 PGID 覆盖 server 和 evaluator；断连后仍能完成当前已授权候选，出现身份/基础设施错误停止。
- 固定 seed、task/state、控制步数、replan、预处理、评测次序和随机数初始化。拒绝跨模型复用不相符的 resume 结果；改工具时记录版本和对协议的影响。
- 冻结 episode 记账与重试规则：真实策略失败计失败，不为提高分数重试；基础设施异常保留原记录并只按预设有限规则重试同一 episode；未解决的缺失不得被丢弃或伪装成完整 200。
- 拟定结果汇总、选模锁文件和报告表；用 fake 结果验证去重、分母、身份漂移拒绝和锁后不可换模型。
- 为视频/日志给出每 episode 上限和全阶段预算，记录所有 episode 的结果，视频保留规则在观察结果前确定。避免为每个 adapter 制作额外完整 base 副本。

真实接线检查另属 GPU 阶段。默认在 FT7 后、候选登记冻结后，将第一个已预注册 dev episode 作为 E1 的技术验收点：先确认有限 action、真实 episode 输出、清理与续跑可用；其结果正常计入 dev，不重复刷分。不能先用 dev/main 试跑挑路线、随后把这些结果抹去。若需要更早的真实诊断，必须另列不与 dev/main 重叠的诊断协议并单独授权，不能悄悄消耗评测集。

完成标准：CPU 接线/汇总测试和不可执行评测计划通过，真实模型链的待验证项明确列出；不将 CPU readiness 称为评测成功。评测时长需由首个正式 dev 技术检查补充估算，当前不凭训练吞吐推算仿真时长。

## 7. E1 → Selection lock → E2：产生受控结果

### E1：七候选各 dev-40

输入为 FT0 固定的 `1000/5000/10000/15000/20000/25000/30000`，以及对应实际 adapter hash/terminal/例外处置记录。第一次 dev episode 前锁定候选登记；若候选不全，停止并报告训练未完成，不能默默缩小集合。

每候选授权范围为完整 40 episodes，技术检查计入其中；不拆成每个 episode 都询问。是否批量授权多个候选由用户决定，未批准的候选不自动启动。每个候选使用同一 40 条 E0 development entries，汇总必须恰好 40 条有效唯一结果，七候选总计 280 条。

选择严格按 E0：成功数最高→更早 train step→adapter identity 字典序。不能增加“训练成本”排序，也不因 loss 好看直接选最后 checkpoint。计划产物：`development_results.csv`、`candidate_selection.json`，每个汇总行引用原始 episode/输入 hash。

### Selection lock

锁定唯一 adapter hash、候选集、dev 汇总 hash、选模规则和全部模型/数据/规范身份。main 开始后不得追加候选、换 adapter 或调参；若发生必须重新设计实验的问题，停止当前 main，保留记录并明确协议失效，不能无痕续跑。

### E2：Base 与锁定 LoRA 各 main-200

两种模型共享 canonical norm 和同一 200 条 task/state/seed/evaluator 协议；各自产出 200 条有效唯一结果，再按键配对，共 400 条。按模型分段授权，允许已授权模型内自治完成任务列表。

主表报告成功数/分母、总体成功率、LoRA−Base 百分点差、4 suites 分项、40 tasks 分项及配对成功/失败分布。单 task 只有 5 条，必须展示原始计数，不夸大精度。若加不确定性分析，先固定方法并说明只有一个训练 seed、task 内样本相关；不能将 200 episodes 当作 200 次独立训练实验。

官方参考已有 E0 投影记录：dev 38/40、main 190/200，来自 [E0 report](../artifacts/pi0-pure-lora/evidence/e0/attempt-20260904T-E0-bmW4ae/e0_report.json)。复核来源身份与协议兼容后可引用，但 normalization 不同，必须单列，不能写成严格受控优劣或非劣效结论。未预注册“接近官方”的量化标准时，报告实际差距，不事后发明达标线。

计划产物：`main_results.csv`、`comparison_summary.json`、suite/task 表、失败样例索引、资源汇总。任何基础设施缺失必须显式标出；主结果不好不阻止报告交付。

## 8. F1 必需、E3 可选：让项目有明确结束点

E2 完成后直接完成 F1 主报告，不等待 full-2000。现有 E0 expansion 只规定 E3 可选、单独授权且不重选 checkpoint，没有冻结数值扩展门槛。若确需一个量化触发条件，应在 main 前单独记录；默认不新增门槛、不执行 E3。

F1 最低交付清单（以下文件是待实现产物，不是本轮已生成结果）：

| 文件/内容 | 必须回答的问题 | 验收 |
|---|---|---|
| `docs/pi0_pure_lora_final_report.md` | 为什么做、改了什么、主比较结果如何、哪些结论不成立 | 有方法、协议、结果、工程偏差、局限，所有数字有来源 |
| `docs/pi0_pure_lora_reproduction.md` | 如何复现已锁定模型与评测 | 固定身份、环境、入口、资源、产物路径；不含私密凭据 |
| 训练/候选/评测证据索引 | 每个 checkpoint 和每行成绩来自哪里 | hash、attempt、协议版本、例外和失败均可追溯 |
| 训练资源与曲线 | 训练成本多少、有限性如何 | 指标来自实际日志；区分有效训练耗时、墙钟、暂停及采样显存；loss 不冒充成功率 |
| 失败分析 | 哪些可见行为失败、哪些原因有证据 | 视频/日志明确对应 episode；可见事实与内部原因推测分开 |

参数量口径固定：trainable 49,987,584；frozen 3,238,048,528；总计 3,288,036,112，trainable 约 1.52%。不得将参数比例直接等同训练显存或耗时减少比例。

报告完成标准：核心表覆盖完整主协议，引用可定位，空缺/异常未隐藏，历史 FT1/FT2 outer 失败如实披露；与用户要求一致地准备最终 Git 保存清单，但未获明确提交指令不 staging/commit/push。

## 9. 实现任务的连续工作约定与当前唯一下一步

推荐顺序：`C-FT2 决定 → R-FT 一次准备 → FT3 → FT4 → FT5 → FT6 → FT7 → E1 → 锁定 → E2 → F1`；E-PREP 的 CPU 部分可穿插训练段间，真实接线随首次正式 dev 完成。各阶段仍独立授权。

每次阶段结束更新一个小型证据索引和当前状态，记录完成/失败/待决策/未启动、输入输出 hash、预算与实际值、进程退出和唯一下一阶段。及时更新 status/handoff 的当前摘要；历史 evidence 不改写。不要每段都扩展新的基础设施项目，不因同类单测已通过而反复重做全部测试。

当前唯一决策仍是 C-FT2。用户接受或拒绝后，才将其落为决策记录；R-FT 是建议的下一项实施准备，不能把本次规划请求解释成已批准 R-FT、受控例外或 FT3。

本轮已完成：读取最新五轮实现进展、对照关键本地源码和冻结协议、编写本候选路线并增加入口指引。没有修改训练/评测源码、manifest 或远端 evidence，没有运行模型/训练/评测，没有 Git 提交，也没有向实现任务发送自动执行指令。

以上是下一阶段说明，不代表已授权执行；等待用户本人决定。
