# Multi-Zoom Tracking 更新说明

## 问题描述

在实现多轮zoom调用功能后，WandB和SwanLab的日志记录存在以下问题：

1. **原始图像重复显示**：每次zoom都显示相同的原始图像
2. **缺少层级信息**：无法区分第1次zoom、第2次zoom
3. **缺少连续关系**：看不出zoom的递进关系

## 解决方案

### 1. WandB Table列结构更新

**原始列结构：**
```python
columns = ["step", "sample_id", "turn_num", "role", "content", "tool_name", 
           "original_image", "cropped_image", "mask_image", "tool_reward", 
           "score", "bbox_iou", "acc_reward"]
```

**更新后列结构：**
```python
columns = ["step", "sample_id", "turn_num", "role", "content", "tool_name", 
           "zoom_level",  # 新增：显示zoom层级
           "original_image", "cropped_image", "mask_image", "tool_reward", 
           "score", "bbox_iou", "acc_reward"]
```

### 2. 原始图像显示优化

**改进前：**
- 每次zoom tool响应都显示原始图像
- 导致表格中有大量重复的原始图像

**改进后：**
```python
# 只在第一次zoom时显示原始图像
if role == "tool" and tool_name == "image_zoom_in_tool" and zoom_count_so_far == 1 and original_image is not None:
    original_image_obj = wandb.Image(original_image, caption="Original Image (Before Zoom)")
```

**效果：**
- 只有第一次zoom的行会显示original_image
- 后续zoom的行original_image列为空，避免重复

### 3. Zoom层级标注

**实现逻辑：**
```python
# 统计到当前turn为止有多少次zoom调用
zoom_count_so_far = sum(
    1 for prev_turn in conversation_history[:turn_num]
    if prev_turn.get("role") == "tool" and prev_turn.get("tool_name") == "image_zoom_in_tool"
)
if role == "tool" and tool_name == "image_zoom_in_tool":
    zoom_count_so_far += 1  # 包括当前turn
```

**显示效果：**
- zoom_level列显示：`"Zoom #1"`, `"Zoom #2"`, `"Zoom #3"`等
- cropped_image的caption包含层级信息：`"image_zoom_in_tool [Zoom #1]: Successfully zoomed..."`

### 4. SwanLab表格同步更新

**更新内容：**
- 添加`tool_name`列：显示使用的工具名称
- 添加`zoom_level`列：显示zoom层级
- 支持`conversation_history`格式（训练时生成的新格式）

**列结构：**
```python
headers = ["step", "sample_id", "turn_num", "role", "content", 
           "tool_name", "zoom_level",  # 新增
           "score", "bbox_iou", "acc_reward", "has_mask"]
```

## 使用示例

### 单次Zoom场景

| turn_num | role | tool_name | zoom_level | original_image | cropped_image |
|----------|------|-----------|------------|----------------|---------------|
| 0 | assistant | - | - | - | - |
| 1 | tool | image_zoom_in_tool | Zoom #1 | ✅ 显示 | ✅ 第1次裁剪 |
| 2 | assistant | - | - | - | - |

### 双次Zoom场景（推荐）

| turn_num | role | tool_name | zoom_level | original_image | cropped_image |
|----------|------|-----------|------------|----------------|---------------|
| 0 | assistant | - | - | - | - |
| 1 | tool | image_zoom_in_tool | Zoom #1 | ✅ 显示 | ✅ 第1次裁剪 |
| 2 | assistant | - | - | - | - |
| 3 | tool | image_zoom_in_tool | Zoom #2 | ❌ 不显示 | ✅ 第2次裁剪（在第1次基础上） |
| 4 | assistant | - | - | - | - |

### 三次Zoom场景

| turn_num | role | tool_name | zoom_level | original_image | cropped_image |
|----------|------|-----------|------------|----------------|---------------|
| 0 | assistant | - | - | - | - |
| 1 | tool | image_zoom_in_tool | Zoom #1 | ✅ 显示 | ✅ 第1次裁剪 |
| 2 | assistant | - | - | - | - |
| 3 | tool | image_zoom_in_tool | Zoom #2 | ❌ 不显示 | ✅ 第2次裁剪 |
| 4 | assistant | - | - | - | - |
| 5 | tool | image_zoom_in_tool | Zoom #3 | ❌ 不显示 | ✅ 第3次裁剪 |
| 6 | assistant | - | - | - | - |

## 查看效果

### WandB中查看

1. 进入项目的WandB页面
2. 点击"Tables" -> "val/multiturn_generations"或"train/multiturn_generations"
3. 可以看到：
   - **zoom_level列**：清楚显示每次zoom的层级
   - **original_image列**：只在第一次zoom时有内容
   - **cropped_image列**：每次zoom都有对应的裁剪结果
   - **mask_image列**：在最后一行显示ground truth mask

### SwanLab中查看

1. 进入项目的SwanLab页面
2. 查看对应的表格
3. 可以看到tool_name和zoom_level信息

## 技术细节

### Zoom计数逻辑

```python
# 遍历到当前turn之前的所有turns
zoom_count_so_far = sum(
    1 for prev_turn in conversation_history[:turn_num]
    if prev_turn.get("role") == "tool" 
    and prev_turn.get("tool_name") == "image_zoom_in_tool"
)

# 如果当前turn本身是zoom tool，加1
if role == "tool" and tool_name == "image_zoom_in_tool":
    zoom_count_so_far += 1
```

### 向后兼容性

- **messages格式**（旧版validation数据）：仍然支持，但zoom_level为空
- **conversation_history格式**（新版训练数据）：完整支持，包含所有zoom信息

### 数据流

```
tool_agent_loop.py (工具调用)
    ↓ 记录 original_image 和 cropped_images
conversation_history (AgentData)
    ↓ 传递到 extra_fields
trainer.py (_dump_generations)
    ↓ 提取并格式化
tracking.py (_log_multiturn_to_wandb)
    ↓ 计算 zoom_count_so_far
    ↓ 决定是否显示 original_image
    ↓ 添加 zoom_level 标签
WandB Table
```

## 预期收益

1. **更清晰的zoom层级关系**：一眼就能看出哪些是第1次zoom、第2次zoom
2. **减少重复信息**：原始图像不再重复显示，节省空间
3. **更好的分析体验**：可以快速筛选特定zoom层级的样本
4. **渐进式分析**：可以看到模型如何逐步聚焦到缺陷区域

## 配合使用

这些tracking改进应该与以下其他改进一起使用：

1. **image_zoom_in_tool.py**：支持连续crop和mask同步
2. **qwen_iad.py**：zoom调用次数奖励机制
3. **tool_agent_loop.py**：正确传递original_image到conversation_history

## 测试建议

1. **查看单次zoom样本**：验证原始图像和第1次裁剪都正确显示
2. **查看双次zoom样本**：验证第2次裁剪基于第1次裁剪，zoom_level正确
3. **查看zoom_level筛选**：在WandB中按zoom_level列筛选，验证可以快速找到特定层级
4. **对比before/after**：对比改进前后的表格，确认重复已消除

