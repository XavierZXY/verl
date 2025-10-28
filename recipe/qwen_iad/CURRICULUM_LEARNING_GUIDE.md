# Curriculum Learning Guide for Adaptive Reward Function

## 概述

新的 reward 函数实现了**方案4：混合策略（Curriculum Learning）**，通过动态调整奖励标准来解决训练初期模型不愿使用工具的问题。

## 核心改进

### 1. 分阶段训练策略

| 训练阶段 | Progress范围 | 探索奖励 | 质量权重 | IoU阈值 | 特点 |
|---------|-------------|---------|---------|---------|------|
| **早期** | 0.0 - 0.3 | +0.2 | 0.3 | 0.05 | 强烈鼓励使用工具，质量要求宽松 |
| **中期** | 0.3 - 0.6 | +0.1 | 0.4 | 0.1 | 平衡探索和质量 |
| **后期** | 0.6 - 1.0 | 0.0 | 0.5 | 0.1 | 严格质量要求，无探索奖励 |

### 2. 奖励对比表

#### 正样本（有缺陷）场景

| 场景 | 训练早期 | 训练中期 | 训练后期 |
|------|---------|---------|---------|
| 答对 + 不zoom | 0.15 | 0.08 | 0.0 |
| 答对 + zoom(IoU=0.1) | 0.87 | 0.75 | 0.65 |
| 答对 + zoom(IoU=0.5) | 0.95 | 0.88 | 0.82 |
| 答对 + zoom(IoU=0.8) | 1.0 | 1.0 | 1.0 |

#### 负样本（无缺陷）场景

| 场景 | 所有阶段 |
|------|---------|
| 答对 + 不zoom | 0.7 |
| 答对 + zoom | 0.6 |

**解释**：负样本不需要zoom，允许模型直接判断"no"。

## 使用方法

### 方式1：在训练循环中传入 training_progress

修改你的数据集类或训练器，在 `extra_info` 中添加 `training_progress`：

```python
# 在你的训练脚本中
current_step = 100
total_steps = 1000
training_progress = current_step / total_steps  # 0.0 - 1.0

# 传递给 reward 函数
extra_info = {
    "mask_image": mask_bytes,
    "image_width": 1024,
    "image_height": 1024,
    "training_progress": training_progress  # 新增
}

score = compute_score(data_source, solution_str, ground_truth, extra_info)
```

### 方式2：基于 epoch 计算

```python
current_epoch = 2
total_epochs = 8
training_progress = current_epoch / total_epochs

extra_info["training_progress"] = training_progress
```

### 方式3：不传入（默认后期训练）

如果不传入 `training_progress`，系统默认为 `1.0`（后期训练），使用最严格的标准：

```python
# 不传入 training_progress，默认 progress=1.0
extra_info = {
    "mask_image": mask_bytes,
    "image_width": 1024,
    "image_height": 1024,
}
```

## 预期训练效果

### 早期阶段（Epoch 1-3, progress < 0.3）

**目标**：让模型学会使用 zoom 工具

- ✅ 使用工具给予 +0.2 探索奖励
- ✅ IoU 只要 > 0.05 就不惩罚
- ✅ 质量权重较低（0.3），容易拿高分

**预期行为**：
- 模型开始频繁调用 zoom
- zoom 位置可能不准确，但被鼓励继续尝试
- 不使用工具的样本得分极低（0.15）

### 中期阶段（Epoch 4-5, 0.3 ≤ progress < 0.6）

**目标**：提高工具使用质量

- ✅ 探索奖励降至 +0.1
- ✅ 质量权重提升到 0.4
- ✅ IoU < 0.1 开始被惩罚

**预期行为**：
- 模型继续使用工具
- 开始优化 zoom 位置，提高 IoU
- 学会 2-3 次 zoom 的渐进式策略

### 后期阶段（Epoch 6-8, progress ≥ 0.6）

**目标**：精细化高质量工具使用

- ⚠️ 无探索奖励
- ⚠️ 质量权重最高（0.5）
- ⚠️ 严格的 IoU 要求

**预期行为**：
- 模型自然使用工具（已形成习惯）
- 高精度定位缺陷（IoU > 0.7）
- 自适应调整 zoom 次数

## 监控指标

训练时监控以下指标，判断策略是否生效：

```python
# 从返回的字典中提取
result = compute_score(...)

print(f"Training Progress: {result['training_progress']:.2f}")
print(f"Acc Reward: {result['acc_reward']:.2f}")
print(f"Tool Reward: {result['tool_reward']:.2f}")
print(f"Answer Acc: {result['answer_acc']:.2f}")
```

### 健康的训练曲线

```
Epoch 1 (progress=0.12):
  - answer_acc: 0.60  ← 初期准确率较低正常
  - acc_reward: 0.85  ← 使用工具得高分
  - tool_reward: 0.30 ← 工具质量尚可

Epoch 3 (progress=0.37):
  - answer_acc: 0.75  ← 准确率提升
  - acc_reward: 0.88  ← 持续使用工具
  - tool_reward: 0.60 ← 工具质量提升

Epoch 6 (progress=0.75):
  - answer_acc: 0.88  ← 准确率高
  - acc_reward: 0.92  ← 高质量工具使用
  - tool_reward: 1.20 ← 工具质量优秀
```

## 故障排查

### 问题1：训练初期 acc_reward 很低

**症状**：早期 acc_reward < 0.3

**原因**：模型不使用工具或 IoU 极低

**解决**：
- 检查 `training_progress` 是否正确传入
- 检查是否正确设置为 0.0-0.3
- 增大探索奖励（修改源码中的 0.2 → 0.3）

### 问题2：训练后期模型仍然不用工具

**症状**：progress > 0.6 但 zoom_count 经常为 0

**原因**：早期阶段太短，模型没学会使用工具

**解决**：
- 延长早期阶段：将 0.3 阈值改为 0.5
- 增加训练 epoch 数
- 降低早期 answer accuracy 的权重

### 问题3：训练后期工具质量不提升

**症状**：后期 IoU 停留在 0.3-0.4

**原因**：质量奖励信号不够强

**解决**：
- 增加 result_reward 权重（1.5 → 2.0）
- 增加 efficiency_penalty（0.08 → 0.12）
- 检查 gt_mask 是否正确

## 参数调优建议

如果默认参数不适合你的数据集，可以调整以下参数：

### 在 `compute_acc_reward_adaptive` 中

```python
# 行 736-744: 探索奖励
if training_progress < 0.3:
    exploration_bonus = 0.2  # 可调整为 0.1-0.3
elif training_progress < 0.6:
    exploration_bonus = 0.1  # 可调整为 0.05-0.15
else:
    exploration_bonus = 0.0

# 行 770-775: 质量权重
if training_progress < 0.3:
    quality_weight = 0.3  # 可调整为 0.2-0.4
elif training_progress < 0.6:
    quality_weight = 0.4  # 可调整为 0.35-0.45
else:
    quality_weight = 0.5  # 可调整为 0.45-0.6
```

### 在 `compute_progressive_tool_reward` 中

```python
# 行 816-818: 效率惩罚
EFFICIENCY_COST = 0.08  # 可调整为 0.05-0.15

# 行 821-822: 无效zoom惩罚
INEFFECTIVE_THRESHOLD = 0.02  # 可调整为 0.01-0.05
INEFFECTIVE_PENALTY = 0.12    # 可调整为 0.08-0.20
```

## 总结

这个 Curriculum Learning 策略解决了训练初期模型不愿使用工具的问题，通过：

1. ✅ **早期鼓励探索**：高额探索奖励，让模型养成使用工具的习惯
2. ✅ **中期平衡质量**：逐步提高质量要求，引导模型优化
3. ✅ **后期严格标准**：只奖励高质量的工具使用

预期训练 8 个 epoch 后，模型能够：
- 自主使用 zoom 工具（使用率 > 95%）
- 高精度定位缺陷（平均 IoU > 0.7）
- 自适应调整 zoom 次数（2-3次）

祝训练顺利！🚀

