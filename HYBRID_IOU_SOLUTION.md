# 混合 IoU 方案实现说明

## 问题背景

在使用渐进式 crop 计算 IoU 时，存在一个关键问题：**使用 cropped mask 计算 IoU 可能导致模型在很小的区域获得高 IoU，但实际上只覆盖了缺陷的一小部分。**

### 问题示例

```
原始图像 800x600，缺陷区域 100x100
  ↓ Zoom 1: 大致覆盖缺陷 → 局部 IoU = 0.7
  ↓ Zoom 2: 只聚焦缺陷的一个小角 (10x10)
Final: 局部 IoU = 0.9（在小 crop 上）

问题：
- 模型可能学会"钻空子"：不断 zoom 到微小区域获得高局部 IoU
- 但实际只覆盖了 1% 的缺陷区域
- 失去了全局定位能力
```

## 解决方案：局部 + 全局混合 IoU

### 核心思想

**分离评估目标**：
1. **全局 IoU**：评估最终覆盖质量（主要质量指标）
2. **局部 IoU**：评估渐进式改进过程（辅助引导信号）

### 实现架构

```python
# 1. 全局 IoU（主要质量指标）
global_iou = compute_on_original_mask(cumulative_bbox)
# - 使用累积 bbox（相对于原始图像）
# - 在原始 mask 上计算
# - 防止小区域游戏

# 2. 局部 IoU（渐进式改进评估）
local_iou_sequence = [iou1, iou2, iou3, ...]
# - 每步使用 cropped mask
# - 评估改进趋势
# - 引导探索策略

# 3. 面积惩罚（安全约束）
size_penalty = f(coverage_ratio)
# - 惩罚过小的最终区域
# - 确保合理覆盖

# 4. 最终评分
main_score = 5 × global_iou^(1/3) × size_penalty
progressive_bonus = f(local_iou_sequence)
```

## 关键实现

### 1. 累积 Bbox 计算：`calculate_cumulative_bbox()`

```python
def calculate_cumulative_bbox(zoom_history, original_image_size):
    """
    模拟 ImageZoomInTool 的坐标变换，包括 bbox resize 验证。
    
    重要：与工具内部逻辑完全一致
    - 应用 _maybe_resize_bbox() 验证每一步
    - 累积坐标变换
    - 返回最终 bbox（原始图像坐标系）
    """
```

**为什么需要这个函数？**
- 工具内部对 bbox 进行 resize 验证（MIN_DIMENSION = 28）
- IoU 计算必须使用**实际使用的 bbox**，而不是用户请求的 bbox
- 确保预测 bbox 和 IoU 计算的一致性

### 2. 混合 IoU 计算：`compute_global_and_local_iou()`

```python
def compute_global_and_local_iou(zoom_history, mask_bytes, original_image_size):
    """
    同时计算全局和局部 IoU。
    
    Returns:
        - global_iou: 累积 bbox 在原始 mask 上的 IoU
        - local_iou: 最后一步在 cropped mask 上的 IoU
        - coverage_ratio: 累积 bbox 占原始图像的比例
        - size_penalty: 面积惩罚系数
        - adjusted_global_iou: 应用惩罚后的全局 IoU
    """
```

**面积惩罚阈值**：
```python
coverage_ratio < 0.005 (0.5%)  → penalty = 0.3
coverage_ratio < 0.01  (1%)    → penalty = 0.5
coverage_ratio < 0.03  (3%)    → penalty = 0.7
coverage_ratio < 0.05  (5%)    → penalty = 0.85
coverage_ratio >= 0.05         → penalty = 1.0
```

### 3. 在 `compute_score()` 中整合

```python
# 使用混合方法
if original_image_size and len(zoom_history) >= 1:
    global_iou_result = compute_global_and_local_iou(
        zoom_history=zoom_history,
        mask_bytes=gt_mask_bytes,
        original_image_size=original_image_size
    )
    
    # 主要质量指标：全局 IoU（带面积惩罚）
    bbox_iou = global_iou_result['adjusted_global_iou']
    
    # 辅助引导：渐进式改进奖励（基于局部 IoU 序列）
    progressive_improvement_bonus, iou_sequence = compute_zoom_quality_reward(
        zoom_history, 
        mask_bytes
    )
```

### 4. 数据集改进

在 `CustomRLHFDataset.__getitem__()` 中添加图像尺寸：

```python
# 添加原始图像尺寸到 extra_info
if images and images[0] is not None:
    row_dict["extra_info"]["image_width"] = images[0].width
    row_dict["extra_info"]["image_height"] = images[0].height
```

## Reward 结构

### 完整公式

```python
tool_reward = (
    5 × global_iou^(1/3) × size_penalty +  # 主要：全局质量（0-5）
    zoom_count_bonus +                      # Zoom 次数奖励（-0.4 to 0.3）
    progressive_improvement_bonus           # 渐进式改进（-0.3 to 0.4）
)

# 总范围：约 -0.7 ~ 5.7
```

### 各部分作用

| 组件 | 范围 | 作用 | 评估对象 |
|------|------|------|----------|
| Global IoU | 0-5 | **主要质量指标** | 最终覆盖准确性 |
| Size Penalty | 0.3-1.0 | 防止小区域游戏 | 覆盖面积 |
| Zoom Count | -0.4 to 0.3 | 鼓励合理次数 | 探索效率 |
| Progressive Improvement | -0.3 to 0.4 | 引导改进过程 | 探索质量 |

## 效果对比

### 场景 1: 聚焦小角落（被防止）

**旧方法**：
```
Zoom 1: 覆盖整个缺陷 → 局部 IoU = 0.7
Zoom 2: 只聚焦小角 → 局部 IoU = 0.9
→ 系统认为"改进"，给予高分 ❌
```

**新方法**：
```
Zoom 1: 覆盖整个缺陷
Zoom 2: 只聚焦小角
→ 全局 IoU = 0.3（累积 bbox 只覆盖了部分）
→ Coverage ratio = 0.008（太小）
→ Size penalty = 0.5
→ Adjusted global IoU = 0.15
→ 得分很低 ✅
```

### 场景 2: 渐进式改进（被奖励）

**旧方法**：
```
Zoom 1: 大致区域 → 局部 IoU = 0.4
Zoom 2: 精确定位 → 局部 IoU = 0.8
→ 得分：主要基于最后的 0.8
```

**新方法**：
```
Zoom 1: 大致区域
Zoom 2: 精确定位
→ 全局 IoU = 0.75（累积 bbox 覆盖良好）
→ Coverage ratio = 0.08（合理）
→ Size penalty = 1.0
→ Progressive improvement = +0.24（持续改进）
→ 得分：全局质量 + 改进奖励 ✅
```

## 修复的 Bug

### Bug 1: KeyError in agent_loop

**问题**：
```python
KeyError: 'global_iou'
# 在 agent_loop._postprocess 中收集 reward_extra_infos 时出错
```

**原因**：
- 返回字典只在 `if global_iou_result:` 时添加相关字段
- 单次 zoom 或无 mask 时，字段缺失
- agent_loop 期望所有样本都有这些字段

**修复**：
```python
# 始终返回所有字段，使用默认值
result = {
    ...
    "global_iou": global_iou_result['global_iou'] if global_iou_result else 0.0,
    "local_iou": global_iou_result['local_iou'] if global_iou_result else 0.0,
    "coverage_ratio": global_iou_result['coverage_ratio'] if global_iou_result else 0.0,
    "size_penalty": global_iou_result['size_penalty'] if global_iou_result else 1.0,
    "cumulative_bbox": global_iou_result['cumulative_bbox'] if global_iou_result else None,
}
```

### Bug 2: Mask Image Loading

**问题**：
```
Failed to load mask image: '_io.BytesIO' object has no attribute 'startswith'
```

**原因**：
- `fetch_image()` 不支持 BytesIO 对象
- 尝试调用 `startswith()` 方法检查 URL

**修复**：
```python
# 直接使用 PIL.Image.open 而不是 fetch_image
from PIL import Image
mask_img = Image.open(BytesIO(mask_image_bytes))
```

## 验证要点

### 1. 单次 Zoom

```python
zoom_history = [[250, 150, 450, 350]]
→ global_iou: 计算累积 bbox 在原始 mask 上
→ local_iou: 0.0（只有一次 zoom）
→ progressive_improvement_bonus: 0.0
```

### 2. 两次 Zoom - 持续改进

```python
zoom_history = [
    [200, 100, 500, 400],  # 大致区域
    [50, 50, 200, 150]     # 精确定位
]
→ global_iou: 累积 bbox [250, 150, 400, 250] 在原始 mask
→ local_iou_sequence: [0.4, 0.75]
→ progressive_improvement_bonus: +0.24
```

### 3. 两次 Zoom - 过度缩小

```python
zoom_history = [
    [200, 100, 500, 400],
    [10, 10, 20, 20]  # 极小区域
]
→ cumulative_bbox: [210, 110, 220, 120]（10x10）
→ coverage_ratio: 0.0002（0.02%）
→ size_penalty: 0.3
→ adjusted_global_iou: 大幅降低 ✅
```

## 向后兼容性

✅ **单次 Zoom**：正常工作，global_iou = local_iou  
✅ **无 Mask**：返回默认值，不影响其他奖励  
✅ **无图像尺寸**：fallback 到原方法  
✅ **旧数据集**：自动提取图像尺寸

## 使用建议

### 查看详细 IoU 信息

```python
# 设置 DEBUG 日志
export VERL_LOGGING_LEVEL=DEBUG

# 输出示例：
# Global vs Local IoU: global=0.7500, local=0.8200, coverage=0.0800, 
#                      size_penalty=1.00, adjusted_global=0.7500
# Local IoU sequence: 0.400 -> 0.820
```

### 监控指标

重点关注：
- `global_iou` vs `local_iou` 的差异
- `coverage_ratio` 的分布
- `size_penalty` 被触发的频率
- `progressive_improvement_bonus` 的分布

### 调参建议

如果发现模型仍然倾向于过小区域：
```python
# 加强面积惩罚
if coverage_ratio < 0.01:
    size_penalty = 0.3  # 降低到 0.3
elif coverage_ratio < 0.03:
    size_penalty = 0.6  # 降低到 0.6
```

如果发现模型不够精确：
```python
# 提高全局 IoU 的权重
tool_reward = 6 * bbox_iou_transformed + ...  # 从 5 提高到 6
```

## 文件清单

### 修改的文件

1. **`recipe/qwen_iad/qwen_iad.py`**
   - 新增 `calculate_cumulative_bbox()` 函数
   - 新增 `compute_global_and_local_iou()` 函数
   - 修改 `compute_score()` 使用混合 IoU
   - 修改 `CustomRLHFDataset.__getitem__()` 添加图像尺寸
   - 修复返回字典，始终包含所有字段

2. **`verl/tools/image_zoom_in_tool.py`**
   - 修复 mask_image 加载，使用 PIL.Image.open

### 新增文件

- **`HYBRID_IOU_SOLUTION.md`** - 本文档

## 总结

混合 IoU 方案通过分离全局质量评估和局部改进引导，成功解决了"小区域高 IoU"的问题：

✅ **防止游戏规则**：使用全局 IoU 评估最终质量  
✅ **保持引导作用**：使用局部 IoU 评估改进过程  
✅ **面积约束**：通过 size_penalty 惩罚过小区域  
✅ **一致性保证**：累积 bbox 计算与工具行为完全一致  
✅ **向后兼容**：支持单次 zoom 和旧数据格式  

**核心优势**：模型需要在全局覆盖和局部精度之间取得平衡，无法通过只关注微小区域来获得高分。

