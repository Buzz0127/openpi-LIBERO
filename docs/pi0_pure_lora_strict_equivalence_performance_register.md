# strict-exact 推理性能候选登记

状态：2026-09-15 关闭。目标是确保任何所谓“加速”保持相同输入、请求顺序、RNG
演进、模型计算顺序，以及 byte-identical `(50,32)` latent 和 `(50,7)` physical actions。

| 候选 | 结论 | 证据与原因 | 后续处理 |
|---|---|---|---|
| 参数 device residency | 拒绝 | G3/P1 的 60/60 action hash 不同，physical max-abs `0.4323568782`。 | 不重跑，不报告其观察到的 RPC 时间。 |
| bf16 dense fusion | 拒绝 | R23 的 unmerged/merged physical max-abs `1.644405388134629`；R25 CPU oracle 也显示小型 bf16 算子不逐字节相同。 | 不重跑 merged-dense GPU，也不设事后容差。 |
| 多环境并发或动态 batch | 设计上拒绝 | 上游 server 同步 `Policy.infer()`；每请求推进全局 RNG。并发会改变 key 到 episode 的映射。 | 不实现。 |
| 减少冷启动或预热 | 非稳定推理候选 | 当前对比已把 cold load/compile、10 次 warmup 与 stable RPC 分开。 | 可记录，不得称为 steady-state model speedup。 |
| 常驻 server / 连接复用 | 已是基线 | `serve_forever()` 已使一个 Policy 常驻；评测端在单臂内复用一个 client。 | 无新增改动。 |
| 输入/输出序列化微调 | 未形成候选 | 尚无证据表明它是主要耗时；不得在没有固定输入 action 回执的情况下改动协议。 | 如未来提出，需单独建立 identity、动作门禁和有界 A/B。 |

## 当前结论

在 strict-exact 规则下，没有剩余的模型侧 GPU 加速候选。性能支线到此结束，不产生
速度提升结论。项目主线应继续使用已锁定的 pure-LoRA unmerged 运行语义，并转向已有
训练、E1/E2 与 partial E3 证据的报告、复现和版本保存工作。

任何未来放宽为“数值近似等价”的提议均是新的实验协议，不能复用此登记、R23 或 G3 的
结果，也需要用户单独授权。
