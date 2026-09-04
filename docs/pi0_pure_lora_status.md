# pi0_base → LIBERO pure-LoRA 状态

更新：2026-09-04（E0）

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

## 失败尝试（均保留证据）

- 首次 S1b 在 bfloat16 buffer hash 处失败，之后改为 `tobytes(order="C")` 并通过回归测试。
- C0 首次测试命令遗漏工作树根目录的 `PYTHONPATH`，修正后相关 16 个测试通过。
- C0 首次远端 commit 因未配置作者身份失败；随后仅对单次提交使用与本地及父提交一致的显式作者环境变量。
- 远端 `origin` 是官方 Physical-Intelligence 仓库，且非交互 `ls-remote` 超时；C0 commit 未 push，也未绕过凭据或改写 remote。

## 已收敛现场

- 远端唯一 LoRA worktree：`/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`。
- 远端分支：`feature/pi0-libero-pure-lora`，C0 后工作树干净。
- 原始 `/home/wengzr/projects/openpi` 保持只读；既有 `outputs/` 未触碰。
- 稳定初始化 manifest：`manifests/pi0_pure_lora/base_model_manifest_c0.json`。
- C0 runtime identity：`1c289cc470e064d6717513e149b5cae03ee71b56f7c9634df450e496ca46c958`。
- C0 model identity：`d484ef5fa06bcb92b0dad92d1f221d4b65406f86dd928569362cb8a9106213ac`。

## 未开始

- S1c 10-step 稳定性 smoke。
- S1d 100-step、首个真实 full train-state、adapter export 与 restore 实测。
- T1 正式分段训练。
- E1 40-episode dev、E2 200-episode main、E3 可选 2,000-episode 评测。

## 当前存储政策

- v1/共享基础快照 24,574,841,856 B 排除且只读。
- LoRA 独立硬上限：250,000,000,000 B。
- 新阶段复核线：225,000,000,000 B。
- soft stop：240,000,000,000 B；hard stop：250,000,000,000 B。
- 新阶段开始前至少保留 20,000,000,000 B 未承诺空间。
- 单项预计新增超过 10 GiB 必须单独决定。
- 两个 2026-09-02 pi0_base 下载 shell launcher 是 historical/not reusable，并已改为 fail closed；历史 evidence 中的 200 GB 数值保持原样。

## 后续路线

`S1c(10-step) → S1d(100-step + 首次 full-state/adapter/restore 实测) → T1 → E1(dev 40) → E2(main 200) → E3(可选 2000)`

每个 GPU 阶段仍须重新做约 30 秒双卡与 CPU/RAM 前检、动态固定单卡、关闭 JAX 预分配，并只由已验证 guard 控制其自身进程组。
