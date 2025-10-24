# Progressive Zoom API 快速参考

## ImageZoomInTool API 变化

### 创建实例 (create)

**旧版 API:**
```python
instance_id, response = await tool.create(
    image=image_data  # PIL Image or URL or base64
)
```

**新版 API:**
```python
instance_id, response = await tool.create(
    image=image_data,        # PIL Image or URL or base64
    mask_image=mask_bytes    # Optional: bytes of PNG mask image
)
```

**参数说明:**
- `image`: 必需，图像数据
- `mask_image`: 可选，mask 图像的 bytes 数据（PNG 格式）
  - 用于 IoU 计算
  - 会随 crop 操作同步更新

### 执行 Zoom (execute)

**API 保持不变，但行为改变:**
```python
response, reward, info = await tool.execute(
    instance_id=instance_id,
    parameters={
        "bbox_2d": [x1, y1, x2, y2],  # 相对于当前可见图像
        "label": "optional_label"      # 可选标签
    }
)
```

**关键变化:**
- `bbox_2d` 现在是**相对于当前 crop 的图像**，而不是原始图像
- 第一次 zoom: 相对于原始图像
- 第二次 zoom: 相对于第一次 crop 的结果
- 第三次 zoom: 相对于第二次 crop 的结果
- 以此类推...

**返回的 info 包含:**
```python
{
    "success": True/False,
    "original_image": PIL.Image,      # 原始图像（不变）
    "current_image": PIL.Image,       # 当前 crop 的图像
    "current_mask": PIL.Image,        # 当前 crop 的 mask（如果有）
    "cumulative_bbox": [x1,y1,x2,y2], # 相对于原始图像的累积 bbox
    "zoom_level": int                 # 当前 zoom 层级（1, 2, 3, ...）
}
```

## 使用示例

### 示例 1: 单次 Zoom (传统用法)
```python
# 创建实例
instance_id, _ = await tool.create(image=my_image)

# 执行单次 zoom
response, _, info = await tool.execute(
    instance_id=instance_id,
    parameters={"bbox_2d": [100, 100, 200, 200]}
)

# info["zoom_level"] = 1
# info["cumulative_bbox"] = [100, 100, 200, 200]
```

### 示例 2: 递进式 Zoom (新用法)
```python
# 创建实例（带 mask）
instance_id, _ = await tool.create(
    image=my_image,
    mask_image=my_mask_bytes
)

# 第一次 zoom: 放大到大致区域
response1, _, info1 = await tool.execute(
    instance_id=instance_id,
    parameters={"bbox_2d": [200, 200, 400, 400]}  # 相对于原始 800x600 图像
)
# info1["zoom_level"] = 1
# info1["current_image"].size = (200, 200)
# info1["cumulative_bbox"] = [200, 200, 400, 400]

# 第二次 zoom: 在上次结果上继续放大
response2, _, info2 = await tool.execute(
    instance_id=instance_id,
    parameters={"bbox_2d": [50, 50, 150, 150]}   # 相对于 200x200 的图像
)
# info2["zoom_level"] = 2
# info2["current_image"].size = (100, 100)
# info2["cumulative_bbox"] = [250, 250, 350, 350]  # 相对于原始图像

# 释放实例
await tool.release(instance_id=instance_id)
```

### 示例 3: Agent 使用场景
```python
# Agent 的思考和行动序列：
<think>我看到图像左上角有可疑区域，先放大看看</think>
<tool_call>
[{"tool_name": "image_zoom_in_tool", "parameters": {"bbox_2d": [0, 0, 400, 400]}}]
</tool_call>

# Tool 返回: zoom level 1, 图像现在是 400x400

<think>在 zoom 后的图像中，我看到中心位置有个小点，再放大确认</think>
<tool_call>
[{"tool_name": "image_zoom_in_tool", "parameters": {"bbox_2d": [150, 150, 250, 250]}}]
</tool_call>

# Tool 返回: zoom level 2, 图像现在是 100x100
# 累积位置: 原始图像的 [150, 150, 250, 250]

<think>现在可以清楚看到是一个缺陷</think>
<location>[{"bbox2d": [150, 150, 250, 250]}]</location>
<answer>yes</answer>
```

## Reward 计算变化

### compute_score() 新参数

函数签名不变，但内部行为改变：
```python
result = compute_score(
    data_source="defect_detection",
    solution_str=model_output,      # 包含多次 tool_call
    ground_truth=gt_dict,
    extra_info={"mask_image": mask_bytes, ...}
)
```

### 返回值扩展
```python
{
    "score": float,                    # 总分
    "format_reward": float,            # 格式奖励
    "acc_reward": float,               # 准确率奖励
    "tool_reward": float,              # 工具奖励（包含下面所有）
    "bbox_iou": float,                 # IoU 得分（变换后）
    "tool_diversity_bonus": float,     # 工具多样性奖励
    "zoom_count": int,                 # 🆕 zoom 次数
    "zoom_count_bonus": float,         # 🆕 zoom 次数奖励
}
```

### Zoom 次数奖励规则
```python
zoom_count = 1  →  bonus = 0.0   # 基准
zoom_count = 2  →  bonus = +0.3  # 最优（鼓励）
zoom_count = 3  →  bonus = +0.2  # 良好
zoom_count > 3  →  bonus = -0.2  # 惩罚
```

### Tool Reward 公式
```python
# 旧版
tool_reward = 5 * iou^(1/3) + diversity_bonus

# 新版
tool_reward = 5 * iou^(1/3) + diversity_bonus + zoom_count_bonus
#             ├─ IoU 质量  ─┤ ├─ 工具多样性 ─┤ ├─ zoom 次数 ─┤
```

## IoU 计算变化

### Progressive Mask IoU

**旧版:** 直接用最后的 bbox 与完整 mask 计算 IoU
```python
iou = _compute_mask_iou(
    pred_boxes=[last_bbox],    # 相对于原始图像
    gt_mask_bytes=mask_bytes
)
```

**新版:** 模拟渐进式 cropping，用最后的 bbox 与 cropped mask 计算 IoU
```python
iou = _compute_mask_iou(
    pred_boxes=all_zoom_bboxes,           # 所有 zoom 的 bbox
    gt_mask_bytes=mask_bytes,
    zoom_history=all_zoom_bboxes[:-1]    # 用前面的 bbox crop mask
)

# 过程:
# 1. Mask 按 zoom_history 渐进式 crop
# 2. 最后的 bbox 与 cropped mask 计算 IoU
# 3. 更准确反映模型的定位精度
```

## 坐标系统说明

### 用户视角（Agent）
- 每次调用 tool 时，**只关心当前看到的图像**
- bbox 坐标相对于当前可见图像
- 不需要计算累积坐标

### 工具内部
- 维护 `current_image` 和 `current_mask`
- 自动计算 `cumulative_bbox`（相对于原始图像）
- 记录 `zoom_history`

### Reward 计算
- 使用 `zoom_history` 重现 cropping 过程
- 对 mask 进行同样的 cropping
- 最后的 bbox 与 cropped mask 计算 IoU

## 最佳实践

1. **渐进式检查策略:**
   ```
   第一次 zoom → 定位大致区域（1/4 到 1/2 图像）
   第二次 zoom → 聚焦可疑点（zoom in 2-3x）
   第三次 zoom → 确认细节（必要时）
   ```

2. **推荐 zoom 次数:** 2-3 次
   - 2 次: 适合大多数情况，效率与精度平衡
   - 3 次: 适合复杂或微小缺陷

3. **避免过度 zoom:**
   - 超过 3 次会有惩罚
   - 过度放大可能失去上下文信息

4. **配合 reference tool:**
   - 先用 reference_tool 了解正常状态
   - 再用 zoom_tool 检查异常区域
   - 可获得 diversity_bonus

## 调试技巧

### 查看 zoom 历史
```python
# 在 tool.execute() 返回的 info 中
info["zoom_level"]         # 当前层级
info["cumulative_bbox"]    # 累积位置

# 从 tool._instance_dict 中（调试用）
zoom_history = tool._instance_dict[instance_id]["zoom_history"]
for i, zoom in enumerate(zoom_history):
    print(f"Zoom {i+1}:")
    print(f"  Requested: {zoom['requested_bbox']}")
    print(f"  Resized: {zoom['resized_bbox']}")
    print(f"  Cumulative: {zoom['cumulative_bbox']}")
```

### 可视化 cropped mask
```python
# 在 reward 计算中，打印 mask 信息
# 已在 _compute_mask_iou() 中包含 debug 日志
# 设置环境变量启用:
# export VERL_LOGGING_LEVEL=DEBUG
```

### 测试脚本
```bash
# 运行提供的测试脚本
python test_progressive_zoom.py
```

## 向后兼容性

- ✅ 单次 zoom 仍然正常工作
- ✅ 不传 mask_image 仍然正常工作
- ✅ 旧的 reward 计算逻辑作为 fallback
- ✅ API 签名未改变，只是增强了功能

## 常见问题

**Q: 如果我只想要传统的单次 zoom 行为？**
A: 直接使用即可，每次 create 新实例就是独立的单次 zoom。

**Q: bbox 坐标错了怎么办？**
A: 工具会自动调用 `_maybe_resize_bbox()` 进行校正和调整。

**Q: 如何知道 cumulative_bbox 是否正确？**
A: 检查 `info["cumulative_bbox"]` 和 `zoom_history`，或者运行测试脚本验证。

**Q: mask_image 是必需的吗？**
A: 不是必需的。如果不提供，IoU 计算会使用 fallback 逻辑（base reward）。

**Q: 可以重置 zoom 状态吗？**
A: 调用 `release()` 然后重新 `create()` 新实例即可。

