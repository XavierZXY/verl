# Progressive Improvement Reward 实现说明

## 概述

实现了**渐进式改进奖励（Progressive Improvement Reward）**机制，用于评估多轮 zoom 操作的质量。该奖励基于一个核心洞察：**有效的 zoom 序列应该在每一步都提升 IoU，而无效的 zoom 会导致 IoU 下降。**

## 核心思想

### 问题定义
在多轮 zoom 场景下：
```
Zoom 1: IoU₁ = 0.3
Zoom 2: IoU₂ = 0.6  →  改进 +0.3 ✅ 有效
Zoom 3: IoU₃ = 0.4  →  下降 -0.2 ❌ 无效
```

**目标**：奖励每一步都带来 IoU 改进的 zoom 序列，惩罚导致 IoU 下降的 zoom。

### 解决方案：混合方案

结合两种评估方式：

1. **逐级改进奖励（权重 0.6）**
   - 评估相邻两次 zoom 之间的 IoU 变化
   - 权重递减：早期改进比后期改进更重要
   - 公式：`∑ᵢ reward(IoUᵢ - IoUᵢ₋₁) × 0.7^(i-1)`

2. **首尾对比奖励（权重 0.4）**
   - 评估第一次和最后一次 zoom 的 IoU 总变化
   - 关注最终结果而非过程
   - 公式：`reward(IoU_last - IoU_first)`

**最终奖励** = `0.6 × 逐级改进奖励 + 0.4 × 首尾对比奖励`

## 实现细节

### 新增函数：`compute_zoom_quality_reward()`

```python
def compute_zoom_quality_reward(zoom_history, mask_bytes):
    """
    计算渐进式改进奖励
    
    Args:
        zoom_history: [bbox1, bbox2, bbox3, ...]
        mask_bytes: Ground truth mask 图像字节
        
    Returns:
        (progressive_improvement_reward, iou_sequence)
    """
```

#### 算法流程

1. **计算每一步的 IoU**
   ```python
   for i in range(len(zoom_history)):
       crop_history = zoom_history[:i]  # 前 i-1 次 zoom 用于 crop mask
       current_bbox = [zoom_history[i]]  # 第 i 次 zoom 用于计算 IoU
       iou_i = _compute_mask_iou(current_bbox, mask_bytes, zoom_history=crop_history)
   ```

2. **逐级改进评分**
   ```python
   for i in range(1, len(iou_sequence)):
       improvement = iou_sequence[i] - iou_sequence[i-1]
       weight = 0.7 ** (i - 1)  # 权重递减
       
       # 分级奖励
       if improvement > 0.05:     step_reward += 0.2 * weight   # 显著改进
       elif improvement > 0.01:   step_reward += 0.12 * weight  # 中等改进
       elif improvement > 0:      step_reward += 0.05 * weight  # 轻微改进
       elif improvement > -0.01:  step_reward -= 0.03 * weight  # 轻微下降
       elif improvement > -0.05:  step_reward -= 0.1 * weight   # 中等下降
       else:                      step_reward -= 0.2 * weight   # 显著下降
   ```

3. **首尾对比评分**
   ```python
   overall_improvement = iou_sequence[-1] - iou_sequence[0]
   
   if overall_improvement > 0.15:    overall_reward = 0.3    # 优秀
   elif overall_improvement > 0.08:  overall_reward = 0.18   # 良好
   elif overall_improvement > 0.03:  overall_reward = 0.08   # 中等
   elif overall_improvement > 0:     overall_reward = 0.02   # 轻微
   elif overall_improvement > -0.05: overall_reward = -0.05  # 轻微下降
   else:                             overall_reward = -0.15  # 显著下降
   ```

4. **综合评分**
   ```python
   progressive_improvement_reward = 0.6 * step_reward + 0.4 * overall_reward
   # 限制范围 [-0.3, 0.4]
   ```

### 整合到 `compute_score()`

在计算 tool_reward 时添加新的组件：

```python
tool_reward = (
    5 * bbox_iou_transformed +       # 主要 IoU 质量 (0-5)
    tool_diversity_bonus +           # 工具多样性 (0-0.1)
    zoom_count_bonus +               # Zoom 次数 (-0.2 to 0.3)
    progressive_improvement_bonus    # 渐进式改进 (-0.3 to 0.4) 🆕
)
```

## 奖励机制详解

### Reward 组成

| 组件 | 范围 | 说明 |
|------|------|------|
| Main IoU | 0 ~ 5 | 最终 IoU 质量（主导） |
| Tool diversity | 0 ~ 0.1 | 使用多种工具 |
| Zoom count | -0.2 ~ 0.3 | 2次最优(+0.3), 3次良好(+0.2), >3惩罚(-0.2) |
| **Progressive improvement** | **-0.3 ~ 0.4** | **渐进式改进质量（新增）** |

**Total tool_reward**: 约 -0.4 ~ 5.8

### 示例场景

#### 场景 1: 两次 Zoom - 持续改进 ✅
```
Zoom 1: IoU = 0.30
Zoom 2: IoU = 0.65  (改进 +0.35)

逐级奖励: +0.2 × 1.0 = +0.2
首尾奖励: +0.3 (优秀改进)
总奖励: 0.6×0.2 + 0.4×0.3 = +0.24

Zoom count bonus: +0.3 (2次最优)
Progressive improvement bonus: +0.24
额外奖励总计: +0.54
```

#### 场景 2: 两次 Zoom - 倒退 ❌
```
Zoom 1: IoU = 0.50
Zoom 2: IoU = 0.30  (下降 -0.20)

逐级奖励: -0.2 × 1.0 = -0.2
首尾奖励: -0.15 (显著下降)
总奖励: 0.6×(-0.2) + 0.4×(-0.15) = -0.18

Zoom count bonus: +0.3 (2次最优)
Progressive improvement bonus: -0.18
额外奖励总计: +0.12 (虽然有 zoom count bonus，但被改进惩罚抵消)
```

#### 场景 3: 三次 Zoom - 持续改进 ✅
```
Zoom 1: IoU = 0.25
Zoom 2: IoU = 0.45  (改进 +0.20)
Zoom 3: IoU = 0.60  (改进 +0.15)

逐级奖励:
  Step 1→2: +0.2 × 1.0 = +0.20
  Step 2→3: +0.2 × 0.7 = +0.14
  总计: +0.34

首尾奖励: +0.3 (优秀改进)
总奖励: 0.6×0.34 + 0.4×0.3 = +0.324

Zoom count bonus: +0.2 (3次良好)
Progressive improvement bonus: +0.324
额外奖励总计: +0.524
```

#### 场景 4: 三次 Zoom - 先好后坏 ⚠️
```
Zoom 1: IoU = 0.20
Zoom 2: IoU = 0.55  (改进 +0.35)
Zoom 3: IoU = 0.40  (下降 -0.15)

逐级奖励:
  Step 1→2: +0.2 × 1.0 = +0.20
  Step 2→3: -0.2 × 0.7 = -0.14
  总计: +0.06

首尾奖励: +0.18 (良好改进)
总奖励: 0.6×0.06 + 0.4×0.18 = +0.108

Zoom count bonus: +0.2 (3次良好)
Progressive improvement bonus: +0.108
额外奖励总计: +0.308 (比持续改进的场景低)
```

## 设计优势

### 1. **精细化反馈**
- 不只看最终结果，还看过程中每一步的质量
- 能够区分"一直改进"和"先好后坏"的序列

### 2. **权重递减**
- 早期改进更重要（找到大致位置）
- 后期微调相对不那么关键
- 符合实际检测流程的直觉

### 3. **混合评估**
- 逐级改进：关注过程，防止无效 zoom
- 首尾对比：关注结果，容忍小波动
- 两者结合，既看过程又看结果

### 4. **适度惩罚**
- 不会过度惩罚探索性 zoom
- 只有明显无效的序列才会被严厉惩罚
- 范围限制 [-0.3, 0.4] 防止奖励失衡

## 与现有机制的关系

### 协同作用

```
Zoom count bonus        → 鼓励合适的 zoom 次数（2-3次）
Progressive improvement → 鼓励有效的 zoom 序列（每次都改进）
```

两者结合：
- **最佳情况**：2次 zoom + 持续改进 → +0.3 + 0.24 = +0.54
- **次优情况**：3次 zoom + 持续改进 → +0.2 + 0.32 = +0.52
- **可接受**：2次 zoom + 轻微改进 → +0.3 + 0.05 = +0.35
- **不理想**：2次 zoom + 下降 → +0.3 - 0.18 = +0.12
- **最差情况**：4次 zoom + 下降 → -0.2 - 0.2 = -0.4

### 互补性

| 机制 | 关注点 | 限制 |
|------|--------|------|
| Zoom count | 次数合理性 | 不考虑质量 |
| Progressive improvement | 质量改进 | 不限制次数 |
| **结合** | **次数 + 质量** | **全面评估** |

## 调参建议

### 当前参数
```python
# 逐级改进阈值
significant_improvement = 0.05    → +0.2
moderate_improvement = 0.01       → +0.12
slight_improvement = 0            → +0.05
slight_degradation = -0.01        → -0.03
moderate_degradation = -0.05      → -0.1
significant_degradation = < -0.05 → -0.2

# 权重递减
weight = 0.7 ** (step - 1)

# 首尾对比阈值
excellent = 0.15   → +0.3
good = 0.08        → +0.18
moderate = 0.03    → +0.08
slight = 0         → +0.02

# 混合权重
step_weight = 0.6
overall_weight = 0.4
```

### 调整方向

**如果希望更强调过程：**
```python
step_weight = 0.7
overall_weight = 0.3
```

**如果希望更强调结果：**
```python
step_weight = 0.5
overall_weight = 0.5
```

**如果希望更严格：**
```python
significant_improvement = 0.08   # 提高阈值
moderate_degradation = -0.03     # 更快进入惩罚区
```

**如果希望更宽容：**
```python
significant_improvement = 0.03   # 降低阈值
moderate_degradation = -0.08     # 减少惩罚触发
```

## 计算成本

### 时间复杂度
- 单次 zoom: 1 次 IoU 计算
- N 次 zoom: N 次 IoU 计算（逐级计算）
- 增加的开销：O(N) 次 mask cropping + IoU 计算

### 优化建议
1. **仅在多次 zoom 时启用**（已实现）
   ```python
   if zoom_count >= 2 and has_tool_usage:
       progressive_improvement_bonus, iou_sequence = compute_zoom_quality_reward(...)
   ```

2. **缓存中间结果**（可选）
   - 如果同一样本多次评估，可缓存 IoU 序列

3. **并行计算**（高级）
   - 多个样本的 IoU 计算可并行

## 实验建议

### 对比实验设计

```
实验 A: Baseline (无 progressive improvement)
       - 只使用 zoom_count_bonus
       
实验 B: + Progressive improvement
       - 添加渐进式改进奖励
       
实验 C: 调整权重
       - 尝试不同的 step_weight / overall_weight

观察指标:
  1. 训练曲线是否更稳定
  2. 模型是否减少无效 zoom
  3. IoU 序列是否更单调递增
  4. 最终检测性能
```

### 预期效果

**短期（训练初期）：**
- 模型学会避免明显无效的 zoom
- 减少"乱跳"行为

**中期（训练中期）：**
- 开始出现渐进式放大的模式
- IoU 序列更平滑

**长期（训练后期）：**
- 形成稳定的 2-3 次 zoom 策略
- 每次 zoom 都有明确目标

## 使用示例

### 查看详细 IoU 序列

```python
# 设置日志级别为 DEBUG
import logging
logging.getLogger("recipe.qwen_iad.qwen_iad").setLevel(logging.DEBUG)

# 计算分数
score = compute_score(data_source, solution_str, ground_truth, extra_info)

# 查看 IoU 序列
print(f"IoU sequence: {score['iou_sequence']}")
print(f"Progressive improvement bonus: {score['progressive_improvement_bonus']}")
```

### 输出示例

```
DEBUG - Zoom step 1: IoU = 0.3000 (crop_history length: 0)
DEBUG - Zoom step 2: IoU = 0.5500 (crop_history length: 1)
DEBUG -   Step 1: IoU 0.3000 → 0.5500, improvement=+0.2500, weight=1.000, contribution=+0.2000
DEBUG -   Overall: IoU 0.3000 → 0.5500, improvement=+0.2500, reward=+0.3000
DEBUG - Progressive improvement: step_reward=+0.2000, overall_reward=+0.3000, final=+0.2400
DEBUG - Progressive improvement bonus: +0.2400, IoU sequence: ['0.300', '0.550']
```

## 文件清单

### 修改的文件
- `recipe/qwen_iad/qwen_iad.py`
  - 新增 `compute_zoom_quality_reward()` 函数
  - 修改 `compute_score()` 整合新奖励

### 新增文件
- `test_progressive_improvement.py` - 测试脚本
- `PROGRESSIVE_IMPROVEMENT_REWARD.md` - 本文档

## 测试方法

```bash
cd /home/takisobe@amd.com/zxy/codes/verl-compare

# 运行测试
python test_progressive_improvement.py

# 查看详细输出（包含 debug 日志）
VERL_LOGGING_LEVEL=DEBUG python test_progressive_improvement.py
```

## 总结

渐进式改进奖励通过评估每一步 zoom 的 IoU 变化，提供了更精细的反馈信号，鼓励模型学习有效的多步探索策略。该机制与现有的 zoom count bonus 互补，共同引导模型在合适的次数内进行高质量的渐进式检测。

**关键优势**：
- ✅ 区分有效和无效的 zoom 序列
- ✅ 权重递减符合实际检测流程
- ✅ 混合评估兼顾过程和结果
- ✅ 适度的奖励/惩罚范围
- ✅ 易于调参和扩展

