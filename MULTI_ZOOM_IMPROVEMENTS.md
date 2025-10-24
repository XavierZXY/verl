# Multi-Zoom Tool Improvements

## 概述

本次改进增强了 `image_zoom_in_tool` 以支持多轮连续调用，允许模型进行渐进式的细节检查。同时更新了奖励机制，鼓励模型恰好调用2次zoom工具。

## 改进内容

### 1. 图像和Mask的连续裁剪 (`verl/tools/image_zoom_in_tool.py`)

#### 核心改进：
- **连续裁剪支持**：在上次裁剪的图像上继续裁剪，而非总是在原始图像上操作
- **Mask同步裁剪**：每次裁剪图像时，自动同步裁剪ground truth mask
- **裁剪历史记录**：记录每次裁剪的bbox和偏移量，用于后续的坐标转换

#### 实现细节：

```python
# 实例数据结构
{
    "original_image": img,      # 原始图像（不变）
    "current_image": img,        # 当前图像（会被更新）
    "original_mask": mask_img,   # 原始mask（不变）
    "current_mask": mask_img,    # 当前mask（会被更新）
    "crop_history": [],          # 裁剪历史：[{"bbox": [...], "offset": [...]}]
}
```

#### 工作流程：
1. **第一次zoom**：在原始图像上提供 `bbox1=[100, 100, 200, 200]`
   - 裁剪原始图像得到 `crop1`
   - 同步裁剪原始mask得到 `mask1`
   - 记录裁剪历史
   - 更新 `current_image` 和 `current_mask`

2. **第二次zoom**：在 `crop1` 上提供 `bbox2=[10, 10, 50, 50]`（相对于crop1的坐标）
   - 裁剪 `crop1` 得到 `crop2`
   - 同步裁剪 `mask1` 得到 `mask2`
   - 记录裁剪历史
   - 更新 `current_image` 和 `current_mask`

### 2. Mask Image传递 (`recipe/qwen_iad/qwen_iad.py`)

在 `CustomRLHFDataset.__getitem__` 中，将mask image传递给zoom工具：

```python
tools_kwargs = {
    "image_zoom_in_tool": {
        "create_kwargs": {
            "image": images[0],
            "mask_image": mask_image,  # 新增：传递mask用于同步裁剪
        },
    },
    ...
}
```

### 3. Zoom调用次数奖励 (`recipe/qwen_iad/qwen_iad.py`)

#### 奖励策略：

```python
# 鼓励恰好2次zoom调用
if zoom_call_count == 0:
    zoom_count_reward = 0.0     # 无zoom（可能是good样本）
elif zoom_call_count == 1:
    zoom_count_reward = 0.05    # 至少尝试了
elif zoom_call_count == 2:
    zoom_count_reward = 0.3     # 最佳！恰好2次
elif zoom_call_count == 3:
    zoom_count_reward = 0.1     # 略多
else:
    zoom_count_reward = -0.1    # 过多，惩罚
```

#### 总奖励计算：

```python
final_score = 0.3 * format_reward + acc_reward + tool_reward

tool_reward = 5 * bbox_iou_transformed + tool_diversity_bonus + zoom_count_reward
```

### 4. 多次Zoom的坐标转换 (`recipe/qwen_iad/qwen_iad.py`)

#### 问题：
多次zoom后，最后一个bbox是在裁剪后的图像坐标系中，需要转换回原始图像坐标系才能正确计算IoU。

#### 解决方案：

新增两个辅助函数：

1. **`_extract_zoom_bbox_sequence(solution_str)`**
   - 从模型输出中提取所有zoom调用的bbox
   - 按时间顺序返回bbox列表

2. **`_transform_bbox_to_original_coords(bbox_sequence, image_width, image_height)`**
   - 将多次zoom的bbox序列转换回原始坐标系
   - 考虑每次裁剪时的resize逻辑（保证最小尺寸28x28）
   - 累积计算偏移量

#### 转换示例：

```python
# 第一次zoom: bbox1 = [100, 100, 200, 200] (原始坐标)
# 工具resize后: [100, 100, 200, 200] -> crop1 (100x100)
# 
# 第二次zoom: bbox2 = [10, 10, 50, 50] (crop1坐标)
# 转换到原始坐标: [100+10, 100+10, 100+50, 100+50] = [110, 110, 150, 150]
```

#### IoU计算流程：

```python
# 1. 提取所有zoom bbox
zoom_bbox_sequence = _extract_zoom_bbox_sequence(solution_str)

# 2. 如果有多次zoom，转换最后一个bbox到原始坐标
if len(zoom_bbox_sequence) > 1:
    transformed_bbox = _transform_bbox_to_original_coords(
        zoom_bbox_sequence, image_width, image_height
    )
    pred_boxes = [transformed_bbox]
else:
    pred_boxes = [zoom_bbox_sequence[-1]]

# 3. 用转换后的bbox与原始mask计算IoU
bbox_iou = _compute_mask_iou(pred_boxes, gt_mask_bytes)
```

## 使用示例

### 模型输出示例（两次zoom）：

```xml
<think>我看到一个可疑区域，让我放大看看。</think>
<tool_call>
[{"tool_name": "image_zoom_in_tool", "parameters": {"bbox_2d": [100, 100, 300, 300]}}]
</tool_call>

<tool_response>
Zoomed in on the image to the region [100, 100, 300, 300].
</tool_response>

<think>在放大后的图像中，我在左上角看到一个小缺陷，让我进一步放大。</think>
<tool_call>
[{"tool_name": "image_zoom_in_tool", "parameters": {"bbox_2d": [10, 10, 60, 60]}}]
</tool_call>

<tool_response>
Zoomed in on the image to the region [10, 10, 60, 60].
</tool_response>

<think>现在我可以清楚地看到这是一个裂纹缺陷。</think>
<location>[{"bbox2d": [10, 10, 60, 60]}]</location>
<type>crack</type>
<answer>yes</answer>
```

### 奖励计算：

- `format_reward = 1.0` (格式正确)
- `acc_reward = 1.0` (答案正确)
- `zoom_call_count = 2`
- `zoom_count_reward = 0.3` (恰好2次，最佳)
- `bbox_iou = 0.85` (转换后的bbox与GT mask的IoU)
- `bbox_iou_transformed = 0.85^(1/3) = 0.947`
- `tool_reward = 5 * 0.947 + 0.3 = 5.035`
- `final_score = 0.3 * 1.0 + 1.0 + 5.035 = 6.335`

## 技术细节

### 坐标系转换的正确性

关键在于考虑工具的resize逻辑：

1. **最小尺寸约束**：工具会将小于28x28的bbox扩展到至少28x28
2. **边界处理**：扩展时保持在图像边界内
3. **累积偏移**：每次裁剪的偏移量需要累加

```python
# 伪代码
cumulative_offset = [0, 0]
for each zoom:
    # 在当前坐标系中应用resize逻辑
    resized_bbox = _maybe_resize_bbox(bbox, current_width, current_height)
    
    # 累加偏移量
    cumulative_offset[0] += resized_bbox[0]
    cumulative_offset[1] += resized_bbox[1]
    
    # 更新当前坐标系大小
    current_width = resized_bbox[2] - resized_bbox[0]
    current_height = resized_bbox[3] - resized_bbox[1]
```

### 向后兼容性

- **单次zoom**：行为与之前完全相同
- **无mask的样本**：仍然正常工作（mask为None）
- **旧数据**：不需要修改现有数据格式

## 预期效果

1. **更精确的缺陷定位**：模型可以先粗定位，再细定位
2. **更高的IoU得分**：两次zoom可以更准确地框住缺陷
3. **更好的训练信号**：zoom_count_reward引导模型学习最优策略
4. **更详细的检查过程**：符合真实质检流程（先整体后局部）

## 测试建议

1. **单次zoom测试**：验证向后兼容性
2. **两次zoom测试**：验证连续裁剪和坐标转换
3. **三次zoom测试**：验证奖励惩罚机制
4. **无mask样本测试**：验证good样本的处理
5. **边界情况测试**：验证小bbox的resize逻辑

## 未来改进方向

1. **自适应zoom次数**：根据缺陷大小动态调整最优zoom次数
2. **多区域zoom**：支持同时检查多个可疑区域
3. **zoom历史可视化**：在日志中保存zoom路径的可视化
4. **智能初始定位**：使用attention map引导第一次zoom

