# 多轮对话日志记录功能

## 功能概述

本功能增强了训练和验证过程中的多轮对话记录能力，能够将每个rollout过程中的完整对话历史（包括assistant的回复、tool的调用和返回的cropped images）记录到wandb和swanlab等追踪平台。

## 主要特性

1. **完整的对话历史追踪**：记录每轮对话的内容，包括：
   - Assistant的回复
   - Tool的调用（包括tool名称、参数、返回结果）
   - Tool返回的cropped images
   - 每轮的reward

2. **多图片支持**：当一个tool返回多张cropped images时，会自动将它们拼接成一张横向排列的图片

3. **训练和验证双支持**：同时支持训练rollout和验证阶段的对话记录

## 修改的文件

### 1. `verl/experimental/agent_loop/tool_agent_loop.py`

**添加的功能：**
- 在 `AgentData` 类中添加了 `conversation_history` 字段来跟踪对话历史
- 在 `_handle_generating_state` 中记录assistant的每次回复
- 在 `_handle_processing_tools_state` 中记录tool的响应和cropped images
- 在 `AgentLoopOutput` 中传递 `conversation_history`

**记录的信息包括：**
```python
{
    "role": "assistant" | "tool",
    "content": "文本内容",
    "turn": 轮次编号,
    "tool_name": "工具名称",  # 仅tool
    "tool_success": True/False,  # 仅tool
    "cropped_images": [PIL.Image, ...],  # 仅tool，包含所有裁剪后的图片
    "tool_reward": 0.0,  # 仅tool
}
```

### 2. `verl/utils/tracking.py`

**添加的功能：**
- `log_multiturn_generations()`: 主入口方法，支持多个追踪后端
- `_log_multiturn_to_wandb()`: wandb实现，创建详细的表格记录
- `_log_multiturn_to_swanlab()`: swanlab实现（保持向后兼容）

**WandB表格列：**
- `step`: 训练步数
- `sample_id`: 样本ID（uid的前12个字符）
- `turn_num`: 对话轮次
- `role`: 角色（assistant/tool）
- `content`: 对话内容（截断至500字符）
- `tool_name`: 工具名称
- `cropped_image`: 裁剪后的图片（wandb.Image对象）
- `tool_reward`: 工具执行的reward
- `score`: 总分（仅在最后一轮显示）

### 3. `verl/trainer/ppo/ray_trainer.py`

**添加的功能：**
- `_log_multiturn_rollout()`: 处理训练rollout的多轮对话记录
- 在训练循环中自动调用多轮对话记录（当multi_turn.enable=True时）

## 使用方法

### 1. 确保multi-turn功能已启用

在你的训练配置中（如 `train_iad.sh`），确保：
```bash
actor_rollout_ref.rollout.multi_turn.enable=True
```

### 2. 配置logger

确保在配置中启用了wandb或swanlab：
```bash
trainer.logger=['wandb','swanlab','console']
```

### 3. 运行训练

正常运行训练脚本，多轮对话将自动记录：
```bash
bash scripts/iad/train_iad.sh
```

## 在WandB中查看结果

1. 登录到你的WandB账户
2. 找到你的项目和实验
3. 在"Tables"标签下，你会看到：
   - `val/multiturn_generations`: 验证阶段的多轮对话
   - `val/multiturn_generations` (训练): 训练rollout的多轮对话

4. 表格中的每一行代表对话的一轮：
   - 查看assistant的回复
   - 查看tool调用的详情
   - **点击 `cropped_image` 列查看tool返回的裁剪图片**
   - 追踪每轮的reward

## 示例输出

假设有如下多轮对话流程：
```
User: 请分析这张图片中的缺陷
Assistant: 我发现图片中有一个可疑区域，让我放大查看 [calls image_zoom_in_tool]
Tool: 返回裁剪后的图片 + 文本描述
Assistant: 经过放大后，我确认这是一个划痕缺陷
```

在WandB表格中会显示为：
```
| step | sample_id | turn | role      | content              | tool_name         | cropped_image | tool_reward | score |
|------|-----------|------|-----------|----------------------|-------------------|---------------|-------------|-------|
| 100  | abc123... | 0    | assistant | 我发现图片中有...      |                   |               |             |       |
| 100  | abc123... | 1    | tool      | Zoomed in on...      | image_zoom_in_tool| [图片]        | 0.0         |       |
| 100  | abc123... | 2    | assistant | 经过放大后...         |                   |               |             | 0.85  |
```

## 注意事项

1. **图片格式**：Tool返回的图片必须是PIL.Image对象
2. **多图片处理**：如果一次tool调用返回多张图片，它们会被水平拼接成一张图
3. **内容截断**：对话内容超过500字符会被截断以提高可读性
4. **性能考虑**：记录图片会占用一定的存储空间，可以通过配置控制记录频率

## 调试

如果对话历史没有被记录：
1. 检查 `conversation_history` 是否在 `batch.non_tensor_batch` 中
2. 确认 `multi_turn.enable=True`
3. 查看日志中是否有 "Warning" 信息
4. 验证图片转换是否成功（检查PIL.Image兼容性）

## 扩展

要支持其他追踪后端（如MLflow、ClearML），可以在 `tracking.py` 的 `log_multiturn_generations()` 方法中添加相应的实现。

## 相关工具

当前支持的工具：
- `image_zoom_in_tool`: 图片缩放工具，返回裁剪后的图片

要添加其他工具的支持，只需确保工具返回的 `ToolResponse` 包含 `image` 字段（PIL.Image对象）。

