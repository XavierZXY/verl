# Progressive Zoom Tool 改进说明

## 概述
本次改进为 `ImageZoomInTool` 添加了递进式放大（progressive cropping）功能，支持多轮调用时在上次 crop 的图片上继续 crop，并同步处理 mask image 用于更准确的 IoU 计算。同时优化了 reward 机制，鼓励模型调用 2-3 次 zoom 工具进行渐进式检查。

## 主要改进

### 1. ImageZoomInTool (`verl/tools/image_zoom_in_tool.py`)

#### 1.1 `create()` 方法改进
- **添加 mask_image 支持**: 在创建实例时可以传入 `mask_image` 参数（bytes 格式）
- **状态跟踪增强**: 维护以下状态信息
  - `original_image`: 原始图像
  - `current_image`: 当前图像（随 crop 更新）
  - `original_mask`: 原始 mask 图像
  - `current_mask`: 当前 mask 图像（随 crop 更新）
  - `zoom_history`: crop 操作历史记录
  - `cumulative_bbox`: 累积的 bbox（相对于原始图像的坐标）

#### 1.2 `execute()` 方法改进
- **递进式 crop**: 基于 `current_image` 而非 `original_image` 进行 crop
- **同步 mask crop**: 在 crop 主图像时，同步 crop mask image
- **坐标变换跟踪**: 
  - 第一次 zoom: bbox 相对于原始图像
  - 后续 zoom: bbox 相对于上次 crop 的图像，自动计算累积坐标
- **zoom 层级反馈**: 在响应文本中显示当前 zoom 层级
- **详细返回信息**: 返回 `current_image`, `current_mask`, `cumulative_bbox`, `zoom_level` 等信息

### 2. 数据集改进 (`recipe/qwen_iad/qwen_iad.py`)

#### 2.1 `CustomRLHFDataset.__getitem__()` 改进
- 从 `extra_info` 中提取 `mask_image`
- 将 `mask_image` 传递给 `image_zoom_in_tool` 的 `create_kwargs`

#### 2.2 新增 `_apply_progressive_cropping()` 函数
- 模拟工具的渐进式 cropping 过程
- 接受 mask image 和 zoom bboxes 序列
- 返回最终 cropped mask 和累积 bbox

#### 2.3 `_compute_mask_iou()` 函数增强
- **新增 `zoom_history` 参数**: 支持传入 zoom 操作序列
- **两种模式**:
  - **Progressive mode**: 使用 zoom_history 对 mask 进行渐进式 crop，然后计算最后一个 bbox 的 IoU
  - **Original mode**: 兼容原有的直接计算方式
- **更准确的 IoU**: 使用 cropped mask 计算，与工具实际执行的结果一致

#### 2.4 `compute_score()` 函数增强
- **Zoom 历史追踪**: 从 solution_str 中提取所有 zoom 操作，构建 `zoom_history`
- **Progressive IoU**: 使用 zoom_history 调用 `_compute_mask_iou()` 进行渐进式 IoU 计算
- **Zoom 次数奖励**: 新增 `zoom_count_bonus`
  - 2 次 zoom: +0.3（最优）
  - 3 次 zoom: +0.2（良好）
  - 1 次 zoom: +0.0（基准）
  - >3 次 zoom: -0.2（惩罚过度放大）
- **增强日志**: 记录 zoom_count 和 zoom_count_bonus
- **返回值扩展**: 添加 `zoom_count` 和 `zoom_count_bonus` 字段

## 工作流程示例

### 单次 Zoom (现有功能)
```
1. 用户: 查看图像，调用 zoom_tool(bbox=[100,100,200,200])
2. 工具: crop 原始图像的 [100,100,200,200] 区域
3. 返回: cropped image
```

### 递进式 Zoom (新功能)
```
1. 用户: 查看图像，调用 zoom_tool(bbox=[100,100,200,200])
   工具: crop 原始图像 → 返回 100x100 的图像
   
2. 用户: 查看放大后的图像，调用 zoom_tool(bbox=[20,20,80,80])
   工具: crop 上次的 100x100 图像 → 返回 60x60 的图像
   累积 bbox: [120,120,180,180]（相对于原始图像）
   
3. IoU 计算: 
   - Mask 也经过两次 crop: [100,100,200,200] → [20,20,80,80]
   - 最终 bbox [20,20,80,80] 与 cropped mask 计算 IoU
   - 更准确反映模型的定位精度
```

## Reward 机制改进

### 原有 Reward 组成
```python
tool_reward = 5 * bbox_iou_transformed + tool_diversity_bonus
# bbox_iou_transformed: cube root of IoU (0.001-1.0)
# tool_diversity_bonus: 0.1 if both zoom and reference tools used
```

### 新 Reward 组成
```python
tool_reward = 5 * bbox_iou_transformed + tool_diversity_bonus + zoom_count_bonus
# bbox_iou_transformed: cube root of IoU (0.001-1.0)
# tool_diversity_bonus: 0.1 if both zoom and reference tools used
# zoom_count_bonus: 
#   - 2 zooms: +0.3 (optimal for progressive inspection)
#   - 3 zooms: +0.2 (good for detailed inspection)
#   - 1 zoom: +0.0 (baseline)
#   - >3 zooms: -0.2 (penalty for excessive)
```

## 优势

1. **更精细的检查**: 支持从粗到细的渐进式检查，适合复杂缺陷检测
2. **更准确的 IoU**: 使用与工具执行一致的 cropped mask 计算，避免坐标不匹配
3. **行为引导**: 通过 reward 机制鼓励合理的多轮探索（2-3次）
4. **向后兼容**: 保持对单次 zoom 和无 mask 场景的兼容性
5. **详细反馈**: zoom_level 信息帮助模型理解当前放大层级

## 测试建议

1. **单次 zoom**: 验证与原有行为一致
2. **两次 zoom**: 验证递进式 crop 和 zoom_count_bonus
3. **三次 zoom**: 验证多层 crop 和奖励
4. **过度 zoom (>3)**: 验证惩罚机制
5. **无 mask 场景**: 验证向后兼容性
6. **坐标变换**: 验证累积 bbox 计算的正确性

## 注意事项

1. **坐标系统**: 
   - 用户提供的 bbox 始终相对于当前可见图像
   - 工具内部维护累积 bbox（相对于原始图像）
   
2. **Mask 同步**: 
   - Mask 必须与原始图像尺寸一致
   - 每次 crop 都会同步处理 mask
   
3. **IoU 计算**: 
   - 使用 progressive mode 时，所有 zoom 都用于 crop mask
   - 最后一个 bbox 用于与 cropped mask 计算 IoU

4. **Reward 平衡**: 
   - zoom_count_bonus 的值可根据实际训练效果调整
   - 当前设置鼓励 2 次 zoom 作为最优策略

