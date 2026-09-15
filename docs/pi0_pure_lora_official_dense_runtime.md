# O1：官方 π0-LIBERO dense 推理链适配

状态：O1 CPU 实现与小张量验收完成；O2 的控制包、固定输入和 CPU fake 验收已完成。O2 的真实数值/性能执行仍未启动：GPU 前检、单卡固定与受保护的单一自有进程组是其不可跳过的前置条件。

本版本只改变项目锁定的 `pi0_base + step-25000 pure-LoRA adapter` 的运行时派生方式。它不
替换 base、adapter、canonical LIBERO normalization、训练冻结真值、E1/E2 结论或 E3 的
382-entry partial 证据。官方 `pi0_libero` checkpoint 仅是未来的独立流程/性能参照，不成为
LoRA 的新 base，也不要求两套权重输出相同动作。

## 固定源码矩阵

| 项目 | 冻结参考 | O1 使用方式 | 与 G3 的关系 |
|---|---|---|---|
| 官方运行链 | OpenPI `15a9616a00943ada6c20a0f158e3adb39df2ccac` | 普通 `Pi0Config`、官方 `Policy.module_jit`、官方 transforms、WebSocket server | 取代“仅 `device_put` 后要求旧 FP32 hash 相同”的目标 |
| 官方 LoRA 数学 | `src/openpi/models/lora.py` | Einsum 合并为 `W + scaling_value * A@B`；FeedForward 合并为 `W + A@B` | 不把 FFN 错套 attention 的 `alpha/r` 缩放 |
| 官方 LIBERO 客户端 | `examples/libero/main.py` | 外部客户端只旋转一次、pad/resize 到 224、state 拼接、prompt、每 5 actions replan | server 不再重复旋转、resize、replan 或归一化 |
| 项目 target-domain 资产 | canonical `libero_pi0_delta_zscore_train1693_v1` | 继续由 pure-LoRA 数据配置提供 normalize/unnormalize | 不冒充官方 checkpoint 自带 norm stats |
| 远端运行快照 | `3619c35ffdcbfe97ae735de175d91c2fb67a899d` | 已确认其包含官方 commit，相关链只在 `pi0_config.py` 多出 pure-LoRA 冻结路径 | 普通 dense `Pi0Config()` 可作固定官方链的抽象验收对象 |

G3-R4 仍是历史失败：host/device 两臂模型身份相同，但 60/60 action hash 不同，最大绝对差
`0.4323568782`。它没有被重新解释为浮点舍入，也没有被本路线倒签通过。

另一个已确认事实是：G3 的 `rng_seed=0` 是日志参数，不是显式构造器 key。pinned Policy 采用
`rng or jax.random.key(0)`，因此 typed JAX key 的布尔求值会失败。新入口在 Policy 构造后将
`jax.random.key(seed)` 写入该实例实际由 `infer()` split 的 `_rng` 字段，并将此兼容适配记录进
runtime manifest；这不修改每请求 split 的官方语义。

## 内存派生规则

新工具是 [official_pi0_lora_merge.py](/Users/buzz/MyProjects/openpi-LIBERO-lora/tools/pi0_pure_lora/official_pi0_lora_merge.py)。它将 Golden 的 20 个 adapter leaves 显式映射为 10 个 dense
kernels，拒绝缺叶、多叶、重复路径和形状不匹配：

| dense families | adapter 形态 | 数学 |
|---|---|---|
| `q_einsum`、`kv_einsum`、`attn_vec_einsum`，含 `_1` action-expert 组 | A/B 保留 18-layer、head/KV 轴 | `W += scale * A@B`；scale 从实际 `gemma_*_lora` config 导出 |
| `gating_einsum`、`linear`，含 `_1` 组 | A/B 保留 18-layer 与双 gating 轴 | `W += A@B`；pinned `FeedForward._dot` 没有 `scaling_value` |

合并先在原 FP32 dtype 按一个 leading scan/head chunk 更新，再统一转换为 BF16 并一次性放到受
pin 的单张 GPU。不会写出、缓存或替换一个完整派生 checkpoint。转换后 dense 推理树不再具有
“训练 base 字节不变”的属性；该不变量只适用于训练/adapter artifact，而不是这个易失运行时树。

入口 [serve_pure_lora_official_dense.py](/Users/buzz/MyProjects/openpi-LIBERO-lora/tools/pi0_pure_lora/serve_pure_lora_official_dense.py) 是薄适配层：它先重用 adapter artifact 的原始 dtype/Golden/identity
校验，再使用 ordinary dense `Pi0Config`、canonical transforms 和官方 Policy/WebSocket server。
它不伪称调用 `create_trained_policy`，因为该 factory 的 JAX 路径只接收 checkpoint 目录，无法接收
本项目的内存合并树。

未来运行时只会在服务端 ready event 中产生 schema-v1 manifest；其字段契约为
[official_dense_runtime_manifest_schema_v1.json](/Users/buzz/MyProjects/openpi-LIBERO-lora/manifests/pi0_pure_lora/official_dense_runtime_manifest_schema_v1.json)。它包含 source/adapter identity、10-rule
merge、cast placement 与实际 Policy RNG 注入，不是可保存或可恢复的 checkpoint。

## O1 CPU 证据

远端不可变证据目录：
`/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/o1/attempt-20260914T-O1-OFFICIAL-DENSE-CPU-R3`。

- `JAX_PLATFORMS=cpu`、`CUDA_VISIBLE_DEVICES=`，未加载完整模型或 checkpoint。
- 7 项测试通过：10-rule 映射、scan/head 轴、小张量 scale、零 adapter、坏映射拒绝、RNG 注入、
  以及直接调用固定 OpenPI `lora.Einsum`/`lora.FeedForward` 的 dense 对照。
- `dense_abstract_tree.json` 验证 ordinary tree 为 50 leaves / 3,238,048,528 参数 / 无 LoRA leaves，
  且全部 10 个显式 merge targets 均存在。
- O2-0 控制包与 fixed input/noise bundle 已写出；R4 在 `CUDA_VISIBLE_DEVICES=`、`JAX_PLATFORMS=cpu` 下完成 13 项 fake/静态测试（2 项需要完整 official source 的算子测试按设计跳过）。固定噪声诊断会同时保存模型归一化动作和反归一化物理动作；比较器先复核每个 `.npy` 的 manifest hash，才分别报告两种空间的分量误差。R5 额外通过三臂命令封存器的 fake 验收：它绑定 step‑25000 artifact 的正确 config hash，且不接受重复端口或不完整路径。

## 下一阶段：O2（已获启动授权；GPU 前检尚未运行）

O2 必须重新进行 30 秒双卡、CPU/RAM 前检，使用既有 memory-only guard 和单一自有 PGID。它将
顺序比较官方 checkpoint 参考、项目 unmerged-BF16 参考和项目 merged-dense-BF16，固定 10 warmups
+ 50 timed requests；冷加载/编译、RPC、设备时间、峰值显存、H2D 痕迹和保护等待单列。

O2-0 已固定 synthetic preprocessed observation 与 internal `(50, 32)` noise bundle，identity 为
`bb48db06f160107286395202aa3cf288804f3a9748d9724365ee6b5241895c90`。同 backend 重复性和
相同 dense-BF16 tree 的直接 Policy/适配入口须严格 action hash 一致；unmerged/merged 是新数值
版本的有限、shape 与分量误差报告，不以事后阈值冒充逐位等价。O2 不含 E1/E2/E3、dev/main/
full-2000 或 checkpoint 下载/保存授权。
