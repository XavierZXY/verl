# Image Reference Tool - Implementation Summary

## 概述

成功实现了一个新的 `ImageReferenceTool`，用于在质量检测任务中获取参考图片。这个工具允许 AI agent 在检测过程中查看同类别的无缺陷参考图片，帮助区分正常特征和真实缺陷。

## 实现的文件

### 1. 核心工具实现
**文件**: `verl/tools/image_reference_tool.py`

这是工具的主要实现文件，包含：
- `ImageReferenceTool` 类，继承自 `BaseTool`
- `create()` 方法：从 `extra_info` 中加载参考图片
- `execute()` 方法：返回参考图片给 agent
- `release()` 方法：清理资源
- 完整的错误处理和日志记录

**特点**：
- 比 zoom 工具更简单，因为图片已经预加载
- 不需要复杂的图像处理或 rate limiting
- 可选的 `reason` 参数用于记录为什么需要参考图片

### 2. 训练配置更新
**文件**: `recipe/qwen_iad/qwen_iad.py`

**更改内容**：

#### a. 系统提示词更新（第 31-59 行）
添加了新的"可用工具"部分，说明两个工具：
```python
"### Available Tools\n"
"You have access to the following tools to assist your inspection:\n"
"1. **image_zoom_in_tool**: Zoom in on a specific region...\n"
"2. **image_reference_tool**: Retrieve a defect-free reference image...\n"
```

并添加了使用参考工具的示例。

#### b. tools_kwargs 更新（第 213-232 行）
```python
# Get good reference image from extra_info if available
good_reference_image = row_dict.get("extra_info", {}).get("good_reference_image")

tools_kwargs = {
    "image_zoom_in_tool": {
        "create_kwargs": {"image": images[0]},
    },
    "image_reference_tool": {
        "create_kwargs": {"good_reference_image": good_reference_image},
    }
}
```

### 3. 工具配置文件
**文件**: `recipe/qwen_iad/configs/image_tools_config.yaml`

新建的配置文件，同时包含两个工具的配置：
- `image_zoom_in_tool` 配置（与原来相同）
- `image_reference_tool` 配置（新增）

### 4. 文档和测试
**文件**：
- `verl/tools/README_image_reference_tool.md` - 详细使用文档
- `verl/tools/test_image_reference_tool.py` - 测试脚本

## 数据流程

### 1. 数据准备（convert_data.py）
```
原始数据 → 读取 JSONL → 为每个类别找到 good 样本
                          ↓
                     将 good 图片添加到 extra_info["good_reference_image"]
                          ↓
                    保存到 Parquet/JSONL
```

### 2. 训练时（qwen_iad.py）
```
加载样本 → 从 extra_info 提取 good_reference_image
              ↓
         传递给 tools_kwargs["image_reference_tool"]["create_kwargs"]
              ↓
         工具初始化时加载图片到内存
```

### 3. 推理时（agent 使用）
```
Agent 检测图片 → 需要参考 → 调用 image_reference_tool
                                    ↓
                             获取参考图片
                                    ↓
                             对比分析 → 做出判断
```

## 使用示例

### Agent 调用示例

```xml
<think>
我注意到表面有一些纹理变化。为了确定这是缺陷还是正常的表面图案，
我需要与无缺陷的参考图片进行比较。
</think>
<tool_call>
[{"tool_name": "image_reference_tool", "parameters": {"reason": "to compare surface texture patterns"}}]
</tool_call>
```

### 工具返回

```python
ToolResponse(
    image=[<PIL.Image>],  # 参考图片
    text="Retrieved reference image for: to compare surface texture patterns"
)
```

## 工具配置

### 在训练配置中使用

修改你的训练配置文件（如 `train_config.yaml`），将 `tool_config_path` 指向新的配置：

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      tool_config_path: "recipe/qwen_iad/configs/image_tools_config.yaml"
```

## 优势和特点

### 1. 简单高效
- 无需复杂的图像处理
- 无需 rate limiting（图片已预加载）
- 直接从内存返回

### 2. 易于使用
- 参数可选（只有一个可选的 `reason` 参数）
- 自动从 `extra_info` 获取数据
- 优雅的错误处理

### 3. 与现有系统无缝集成
- 遵循 `BaseTool` 接口
- 使用相同的配置格式
- 兼容现有的 agent loop

### 4. 提升检测能力
- 帮助 agent 区分正常和异常
- 减少误报（将正常特征误判为缺陷）
- 提供对比基准

## 与 Zoom 工具的对比

| 特性 | Image Reference Tool | Image Zoom In Tool |
|------|---------------------|-------------------|
| **目的** | 获取参考图片对比 | 放大特定区域 |
| **输入** | 可选的 reason 字符串 | 必需的 bbox 坐标 |
| **复杂度** | 简单（预加载） | 复杂（图像处理） |
| **使用场景** | 验证正常 vs 缺陷 | 仔细检查细节 |
| **Rate Limiting** | 否 | 是（通过 Ray pool） |
| **数据来源** | extra_info | 当前图片 |

## 测试

运行测试脚本：

```bash
cd verl/tools
python test_image_reference_tool.py
```

测试包括：
1. 基本使用（有参考图片）
2. 无参考图片的情况
3. 无参数调用
4. 无效实例 ID

## 潜在改进

1. **缓存机制**: 如果同一类别的多个样本使用相同参考图片，可以添加缓存
2. **多参考图片**: 支持为每个样本提供多个参考图片
3. **图片预处理**: 可以添加自动对齐、归一化等功能
4. **相似度计算**: 自动计算当前图片与参考图片的相似度

## 注意事项

1. **内存使用**: 每个实例都会在内存中保存一份参考图片，注意内存使用
2. **图片大小**: 数据转换时已经压缩图片，默认质量 85
3. **缺失处理**: 并非所有样本都有参考图片，工具会优雅处理这种情况
4. **格式兼容**: 支持所有 PIL 可读的图片格式

## 相关文件清单

### 新增文件
- `verl/tools/image_reference_tool.py` - 工具实现
- `verl/tools/README_image_reference_tool.md` - 文档
- `verl/tools/test_image_reference_tool.py` - 测试脚本
- `recipe/qwen_iad/configs/image_tools_config.yaml` - 工具配置

### 修改文件
- `recipe/qwen_iad/qwen_iad.py` - 添加工具到 tools_kwargs 和更新系统提示

### 已存在（无需修改）
- `data/tools/v2/convert_data.py` - 已经在准备参考图片数据
- `verl/tools/base_tool.py` - 基类
- `verl/tools/schemas.py` - Schema 定义

## 下一步

1. **测试**: 在实际训练中测试工具效果
2. **监控**: 观察 agent 如何使用这个工具
3. **优化**: 根据使用情况调整提示词
4. **扩展**: 考虑添加更多视觉对比工具

## 总结

成功实现了 `ImageReferenceTool`，这是一个轻量级、易用的工具，允许 AI agent 在质量检测过程中查看无缺陷的参考图片。工具已经完全集成到现有系统中，可以与 `ImageZoomInTool` 配合使用，提升缺陷检测的准确性。

