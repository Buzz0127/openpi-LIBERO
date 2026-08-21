# 原始 π0-LIBERO 算法概览

本文只保留部署与答辩所需的算法主线，不展开逐函数源码、Gemma/SigLIP 细节或完整数学推导。

## 1. 一句话理解

π0 根据语言指令、第三人称图像、腕部图像和机器人状态，一次生成一段连续机器人动作；LIBERO 执行其中前 5 步，获取新的环境反馈后再次规划。

## 2. 输入是什么

LIBERO 客户端构造四类输入：

| 输入 | 作用 |
| --- | --- |
| `prompt` | 描述要完成的任务 |
| `observation/image` | 第三人称视角，提供场景、物体和空间关系 |
| `observation/wrist_image` | 腕部近景，提供抓取与接触细节 |
| `observation/state` | 8 维状态：末端位置 3 维、姿态 3 维、夹爪关节 2 维 |

`LiberoInputs` 将两张图像映射到模型的 `base_0_rgb` 和 `left_wrist_0_rgb`。LIBERO 没有右腕相机，因此补一张零图；原始 π0 同时把该相机的 mask 设为 false，避免模型把补零误认为真实画面。

## 3. 为什么需要 normalization statistics

机器人位置、旋转和动作各维度的数值范围不同。Policy 从 checkpoint 的 `assets/physical-intelligence/libero/norm_stats.json` 加载与训练时一致的统计量：

```text
LIBERO 原始数值
  → 按 checkpoint 统计量归一化
  → 模型推理
  → 按相同统计量反归一化
  → LIBERO 动作
```

本项目的原始 `pi0_libero` 使用普通 mean/std 归一化，不使用 π0.5/π0-FAST 路线中的 quantile normalization。统计量与 checkpoint 必须配套；模型权重正确但统计量错误，也会产生错误动作尺度。

## 4. 模型内部的角色分工

可以把模型高层理解为两个协作部分：

1. **视觉语言主干**：融合图像和语言，理解当前场景、目标物体以及任务要求。
2. **Action Expert**：结合视觉语言上下文、机器人状态和当前噪声动作，预测如何把噪声逐步变成可执行动作序列。

部署时不需要分别启动这两个部分；它们共同包含在 `pi0_libero` checkpoint 和 OpenPI Policy 中。

## 5. Flow Matching 的最小直觉

训练时可以在真实动作序列 `a` 与高斯噪声 `ε` 之间构造中间状态：

```math
x_t = (1-t)a + t\varepsilon, \qquad t\in[0,1]
```

- `t=0` 时是干净动作 `a`；
- `t=1` 时是随机噪声 `ε`；
- 模型学习在视觉、语言和状态条件下预测使 `x_t` 朝动作方向演化的速度场。

推理从 `t=1` 的噪声动作开始，通过若干数值积分步骤走向 `t=0`，最终得到一段结构化动作。核心直觉是“把随机动作逐步修正成符合当前任务的动作”，而不是逐 token 生成离散控制命令。

## 6. 动作块如何回到 LIBERO

固定版本中的原始 `Pi0Config` 使用：

```text
action_horizon = 50
internal action_dim = 32
```

模型内部用统一的 32 维动作槽位；LIBERO 只需要 7 维，因此反归一化后由 `LiberoOutputs` 保留：

```python
actions[..., :7]
```

最终客户端得到 `50 × 7` 的 action chunk。7 维分别对应末端平移 3 维、末端旋转 3 维和夹爪控制 1 维。

## 7. 为什么动作块仍然是闭环

客户端设置 `replan_steps=5`，只把动作块的前 5 步放入执行队列：

```text
观察环境
  → π0 预测 50 步
  → 执行前 5 步
  → 环境发生变化
  → 获取新图像与状态
  → 再次预测 50 步
```

环境反馈持续进入下一次推理，因此这是每 5 步滚动重规划的闭环控制，不是一次预测后执行到底的开环控制。

## 8. 原始 π0 与默认 π0.5

本项目固定复现原始 π0：

```text
config     = pi0_libero
checkpoint = openpi-assets/checkpoints/pi0_libero
model      = Pi0Config / ModelType.PI0
```

固定 OpenPI checkout 中，单独使用：

```bash
python scripts/serve_policy.py --env LIBERO
```

会进入默认策略分支并选择 `pi05_libero`。因此服务端必须显式使用：

```bash
policy:checkpoint \
  --policy.config pi0_libero \
  --policy.dir <pi0_libero-checkpoint>
```

## 9. 可复述总结

> 原始 π0-LIBERO 接收语言、第三人称图像、腕部图像和 8 维机器人状态。LIBERO 适配器完成相机键映射与缺失相机 mask，并使用 checkpoint 自带统计量归一化输入。视觉语言主干理解任务上下文，Action Expert 通过 flow matching 从噪声得到动作序列。模型内部生成 50×32 的统一动作表示，反归一化后裁剪为 LIBERO 的 50×7 动作块。客户端只执行前 5 步，再使用新观测重新规划。项目显式加载 `pi0_libero`，避免误用 `--env LIBERO` 当前默认的 π0.5。

## 10. 关键入口

- Policy Server：`scripts/serve_policy.py`
- LIBERO client：`examples/libero/main.py`
- 模型与 checkpoint 装配：`src/openpi/policies/policy_config.py`
- LIBERO 输入输出适配：`src/openpi/policies/libero_policy.py`
- 本项目有界评测器：[`../tools/eval_libero_bounded.py`](../tools/eval_libero_bounded.py)
