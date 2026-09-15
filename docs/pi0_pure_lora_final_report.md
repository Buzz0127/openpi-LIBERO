# pi0_base → LIBERO pure-LoRA：最终主实验报告

## 结论

在预注册、固定 seed=7 的 LIBERO main-200 集合上，`pi0_base + canonical
LIBERO normalization` 成功 `0/200 (0.0%)`；同一 Base 加上开发集锁定的
pure-LoRA adapter（step 25000）成功 `25/200 (12.5%)`。逐 state 配对后，
有 25 条为“Base 失败、LoRA 成功”，175 条均失败，因此受控差值为 **+12.5
个百分点**。

这证明的是该固定协议下、该训练 seed 与评测 seed 的参数高效适配结果，
不是多 seed 统计结论、论文完整复现或对官方 `pi0_libero` 的严格同协议比较。

## 实验定义

- 初始化：发布的 `pi0_base` 权重。
- Base 组：`pi0_base weights + LIBERO target-domain normalization
  (no-gradient baseline)`；不是“zero-shot”。
- LoRA 组：同一 Base、同一 canonical normalization、同一 E0 main task/state
  集合，仅加载 adapter-only 制品。
- pure-LoRA 真值：20 个 Golden adapter 叶子、49,987,584 个可训练参数；
  50 个非 Golden 叶子冻结。总参数 3,288,036,112，训练比例约 1.52%。
- 候选仅用不重叠 development-40 选择：成功数优先、随后更早训练步数、再按
  adapter identity 字典序。7 个候选的开发成功数依次为
  `0,0,1,1,2,6,4`（steps `1000…30000`），故锁定 step 25000。

## 主结果

| Suite | Base | pure-LoRA |
|---|---:|---:|
| `libero_spatial` | 0/50 | 3/50 |
| `libero_object` | 0/50 | 10/50 |
| `libero_goal` | 0/50 | 11/50 |
| `libero_10` | 0/50 | 1/50 |
| **总计** | **0/200** | **25/200** |

两组都包含 200 个有效且唯一的 `(suite, task_id, initial_state_index)` 键。
Base 产生 13,200 次策略请求、LoRA 产生 12,419 次策略请求，均无 evaluator
`exception`；因此 Base 的 0/200 是有效任务失败，而不是服务未响应。

逐 task 的五条原始计数和逐 state 配对结果保存在远端审计证据的
`task_results.json` 与 `main_results.csv`，不以 5 条样本夸大精度。

## 运行、修复与资源事实

首次 E2 的 Base 半边虽然完整执行，但所有 episode 在首次 inference 前因
`ShapeDtypeStruct` 进入 JAX 参数树而失败（零策略请求），故不作为成绩。
该 immutable 失败 attempt 被保留。恢复控制器只重跑 Base 的同一 main-200：
它以 Golden manifest 的形状和 dtype 实体化零 LoRA 叶子，同时保留 Orbax
恢复的 Base 权重。恢复后 Base 200/200 正常完成。

两段均在 GPU 1 的 task-owned guard 下执行，未操作未知进程。逐 episode
采样到的峰值显存为 Base 23,231 MiB、LoRA 27,357 MiB；两者均记录到最高
100% 利用率。它们是采样峰值，不应解释为持续显存占用或训练显存。

## 身份与证据

- E1 selection-lock identity：`53833ef085c169e91a415ecbaeea4674ce8fbd6bbb71fd79ed3c59a09b1f9066`。
- E2 paired audit identity：`fe830800aebabb0aec2e74916c4b0e1d100d35c4f43de53c18957a8df8ff9fa1`。
- Base recovery summary identity：`44c9660d5eb9ca654b9edfb923c738864b3f78a951d1c45ce0975f458df644a5`。
- Paired audit：`/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/e2-audit/attempt-20260913T-E2A-AUDIT-R1`。
- Base recovery：`/home/wengzr/projects/openpi-lora-runs/evaluations/attempt-20260913T-E2-BASE-RECOVERY-200-Q2r5Lm`。
- 锁定 LoRA：`/home/wengzr/projects/openpi-lora-runs/evaluations/attempt-20260912T-E2-MAIN-400-E4z8Pr/pure_lora_step_00025000`。
- E3 partial closeout：`/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/e3/g2/attempt-20260914T-E3-PARTIAL-CLOSEOUT-R1/closeout.json`。
- F2 evidence audit：`/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/f2/attempt-20260915T-F2-REPORT-AUDIT-R1/report-evidence-audit.json`；它逐字段与 SHA-256 复核上述 E2/E3 制品。

官方 `pi0_libero` 的既有 E0 投影为 main `190/200`，但它使用 checkpoint-owned
normalization，不能与本报告的 Base/pure-LoRA 受控比较合并或写作非劣结论。

## 局限与未执行项

- 只有一个训练 seed 与固定评测 seed；task 内五个 state 也不是五次独立训练。
- E3 full-2000 已在本报告生成后单独启动，后经用户授权停止并以 `382` 个有效唯一
  episode、`19` success、`0` evaluator exception 保留为 partial；它不与 main-200
  合并，也不构成完整 2000-episode 结果。没有用 main 结果重选 checkpoint 或调参。
- 参数 device-residency 与 bf16 dense-fusion 的速度诊断都未通过 strict exact-equivalence；
  本报告不包含任何推理加速结论。
- 本报告不把训练 loss、参数比例、采样显存峰值等同于泛化性能、完整训练成本或
  显存节约比例。
- F1 文档没有覆盖、删除或重写历史 FT1/FT2 的 outer orchestration 偏差；其受控
  例外与后续完整证据仍保留在继承入口中。
