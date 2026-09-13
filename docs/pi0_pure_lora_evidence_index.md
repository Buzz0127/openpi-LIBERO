# pi0 pure-LoRA 证据索引

本索引只定位证据，不以路径替代身份校验。复查时先读取小型
`status.json`、`summary.json`、manifest 与 SHA-256，再按需打开逐 episode 文件。

| 范围 | 证据位置 | 结论 / 状态 |
|---|---|---|
| 训练与冻结规则 | [继承入口](pi0_pure_lora_gpt6_inheritance.md) | FT3–FT7 至 step 30000；20 Golden / 50 non-Golden 规则及历史 C-FT2 偏差保留。 |
| E0 split | `/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/tool-snapshots/ft4-control-progress-heartbeat-20260911/manifests/pi0_pure_lora/e0_task_state_manifest.json` | 40 development 与 200 main 键冻结且互斥。 |
| E1 dev 与选模 | `/home/wengzr/projects/openpi-lora-runs/evaluations/attempt-20260912T-E1-DEV-R8-B1w9aV/selection_lock.json` | 280 dev episodes 后锁定 step 25000；lock identity `53833ef…f9066`。 |
| 初始 E2 | `/home/wengzr/projects/openpi-lora-runs/evaluations/attempt-20260912T-E2-MAIN-400-E4z8Pr` | LoRA-200 有效；Base-200 因 `ShapeDtypeStruct`、零策略请求而无效，保留但不计分。 |
| Base recovery | `/home/wengzr/projects/openpi-lora-runs/evaluations/attempt-20260913T-E2-BASE-RECOVERY-200-Q2r5Lm` | 200/200 完成、status pass、guard return code 0；summary identity `44c966…644a5`。 |
| E2 paired audit | `/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/e2-audit/attempt-20260913T-E2A-AUDIT-R1` | `main_results.csv`、suite/task 表与 comparison summary；audit identity `fe8308…ff9fa1`。 |
| 主结论 | [最终报告](pi0_pure_lora_final_report.md) | Base 0/200，pure-LoRA 25/200，受控差值 +12.5 个百分点。 |
| 可重复入口 | [复现说明](pi0_pure_lora_reproduction.md) | 固定输入、控制器、运行门禁、无效结果条件与禁止事项。 |

`E3 full-2000` 已在单独授权后启动，尚未完成；它不会改变已锁定的 adapter，也不应
在终态 summary 出现前作为结果引用。
