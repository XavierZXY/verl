# Reward Components Tracking

## 概述

现在训练过程中的详细reward组件（format_reward, acc_reward, bbox_reward, count_tool_call_1）会被自动记录并传到wandb。

## 修改说明

### 1. Reward函数返回格式

在 `recipe/qwen_iad/qwen_iad.py` 中，`compute_score` 函数返回包含多个reward组件的字典：

```python
return {
    "score": final_score,           # 最终综合得分
    "format_reward": format_reward,  # 格式正确性奖励
    "acc_reward": acc_reward,        # 答案准确性奖励
    "bbox_reward": bbox_reward,      # 边界框准确性奖励
    "count_tool_call_1": count_tool_call_1,  # 工具调用次数
}
```

### 2. 自动收集机制

- `BatchRewardManager` 会自动提取reward函数返回字典中的所有键值对
- 这些额外信息会被添加到 `batch.non_tensor_batch` 中
- 训练循环会自动处理这些信息

### 3. Metrics统计

在 `verl/trainer/ppo/metric_utils.py` 的 `compute_data_metrics` 函数中，新增了以下统计：

```python
# 对每个reward组件计算统计信息
reward_components = ["format_reward", "acc_reward", "bbox_reward", "count_tool_call_1"]
for component in reward_components:
    if component in batch.non_tensor_batch:
        # 计算 mean, max, min
        metrics[f"reward_components/{component}/mean"]
        metrics[f"reward_components/{component}/max"]
        metrics[f"reward_components/{component}/min"]
```

### 4. WandB日志

这些metrics会通过 `logger.log(data=metrics, step=self.global_steps)` 自动传到wandb，在wandb中可以看到以下指标：

- `reward_components/format_reward/mean`
- `reward_components/format_reward/max`
- `reward_components/format_reward/min`
- `reward_components/acc_reward/mean`
- `reward_components/acc_reward/max`
- `reward_components/acc_reward/min`
- `reward_components/bbox_reward/mean`
- `reward_components/bbox_reward/max`
- `reward_components/bbox_reward/min`
- `reward_components/count_tool_call_1/mean`
- `reward_components/count_tool_call_1/max`
- `reward_components/count_tool_call_1/min`

### 5. Validation支持

在验证(validation)过程中，这些reward组件也会被自动收集和统计：

- `_validate` 方法会收集 `reward_extra_info`
- `process_validation_metrics` 函数会对所有数值型变量进行统计分析
- 结果会包含 mean@N, std@N, best@N 等统计指标

## 使用示例

无需额外配置，只要reward函数返回包含这些组件的字典，系统就会自动记录和统计。

### 训练时查看

在训练日志中可以看到：
```
reward_components/format_reward/mean: 0.85
reward_components/acc_reward/mean: 0.72
reward_components/bbox_reward/mean: 0.68
```

### WandB可视化

在WandB界面中，可以绘制这些指标的变化曲线，用于：
- 监控各个reward组件的训练趋势
- 诊断训练问题（例如某个组件一直很低）
- 对比不同实验的reward组件分布

## 注意事项

1. **非中止样本过滤**: 统计时会过滤掉response_length为0的中止样本，确保统计更准确
2. **自动类型判断**: 只有数值类型的变量会被统计，字符串类型会被跳过
3. **扩展性**: 如果需要添加新的reward组件，只需在reward函数返回字典中添加新的键值对即可

## 相关文件

- `verl/trainer/ppo/metric_utils.py` - Metrics计算逻辑
- `verl/workers/reward_manager/batch.py` - Reward信息收集
- `recipe/qwen_iad/qwen_iad.py` - Reward函数实现
- `verl/trainer/ppo/ray_trainer.py` - 训练主循环

