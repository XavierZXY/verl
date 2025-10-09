# Bbox Step Comparison Tool

这个工具用于比较不同训练步骤（steps）的bounding box预测结果，帮助你可视化模型在训练过程中的bbox预测变化。

## 功能特点

- 并排显示多个训练步骤的bbox预测结果
- 同时显示ground truth（绿色）和预测bbox（红色）
- 显示每个步骤的score和answer信息
- 支持选择特定样本进行对比
- 生成高质量的对比图像

## 安装依赖

```bash
pip install opencv-python numpy
```

## 使用方法

### 方法1：使用Python脚本（推荐）

```bash
python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data data/mvtec/test/test.jsonl \
    --steps 0 40 180 \
    --output-dir logs/rollout/comparison \
    --max-samples 10
```

### 方法2：使用Shell脚本

```bash
# 赋予执行权限
chmod +x scripts/compare_bbox_steps.sh

# 运行
./scripts/compare_bbox_steps.sh --steps 0 40 180 --max-samples 10
```

## 参数说明

### 必需参数

- `--rollout-dir`: rollout结果目录，包含多个步骤的.jsonl文件
  - 例如: `logs/rollout/validation/mvtec/`
  - 该目录应包含如 `0.jsonl`, `40.jsonl`, `180.jsonl` 等文件

- `--test-data`: test.jsonl文件路径，包含ground truth信息和图像路径
  - 例如: `data/mvtec/test/test.jsonl`
  - 用于获取真实的bbox坐标和图像文件名

- `--steps`: 要对比的训练步骤列表
  - 例如: `--steps 0 40 180`
  - 可以指定任意数量的步骤

- `--output-dir`: 输出目录，用于保存对比图像
  - 例如: `logs/rollout/comparison`

### 可选参数

- `--mvtec-root`: MVTec数据集根目录
  - 默认: `/home/zxy/codes/working/RL/verl/data/mvtec`
  - 该目录应包含完整的MVTec数据集图像

- `--max-samples`: 最多处理的样本数量
  - 例如: `--max-samples 10` (只处理前10个样本)
  - 不指定则处理所有样本

- `--sample-indices`: 指定要对比的样本索引
  - 例如: `--sample-indices 0 5 10 15`
  - 优先级高于 `--max-samples`

## 使用示例

### 示例1：对比所有样本的前3个step

```bash
python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data data/mvtec/test/test.jsonl \
    --steps 0 40 180 \
    --output-dir logs/rollout/comparison_all
```

### 示例2：只对比前10个样本

```bash
python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data data/mvtec/test/test.jsonl \
    --steps 0 40 180 \
    --output-dir logs/rollout/comparison_10 \
    --max-samples 10
```

### 示例3：对比特定索引的样本

```bash
python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data data/mvtec/test/test.jsonl \
    --steps 0 40 80 120 180 \
    --output-dir logs/rollout/comparison_selected \
    --sample-indices 0 5 10 15 20
```

### 示例4：对比更多步骤

```bash
python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data data/mvtec/test/test.jsonl \
    --steps 0 20 40 60 80 100 120 140 160 180 \
    --output-dir logs/rollout/comparison_detailed \
    --max-samples 5
```

### 示例5：使用自定义MVTec路径

```bash
python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data /path/to/custom/test.jsonl \
    --mvtec-root /path/to/mvtec/dataset \
    --steps 0 40 180 \
    --output-dir logs/rollout/comparison_custom
```

## 输出说明

生成的对比图像将保存在指定的输出目录中，文件命名格式为：

```
sample_000_comparison.jpg
sample_001_comparison.jpg
sample_002_comparison.jpg
...
```

每张图像包含：
- 多个训练步骤的并排对比
- 顶部显示: Step编号、Score、Answer (yes/no/unknown)
- 绿色框: Ground Truth bbox
- 红色框: 模型预测的bbox
- 底部图例说明

## 数据结构要求

### test.jsonl 文件结构

test.jsonl文件应包含以下字段：

```json
{
    "filename": "bottle/test/broken_large/000.png",
    "clsname": "bottle",
    "label": 1,
    "label_name": "broken_large",
    "bboxes": [[294, 883, 1683, 1738]],
    "bbox_count": 1
}
```

关键字段：
- `filename`: 图像文件的相对路径（相对于mvtec_root）
- `bboxes`: ground truth的bbox坐标列表，格式为 `[x_min, y_min, x_max, y_max]`
- 对于没有缺陷的样本，`bboxes`字段可以为空或不存在

### rollout JSONL文件结构

rollout结果文件应包含模型的预测输出：

```json
{
    "input": "...",
    "output": "...<location>[{\"bbox2d\": [x1, y1, x2, y2]}]</location>...<answer>yes</answer>...",
    "score": 0.728
}
```

关键字段：
- `output`: 模型输出，应包含`<location>`和`<answer>`标签
- `score`: 模型评分
- rollout结果的行号应与test.jsonl的行号一一对应

## 常见问题

### Q: 图像路径找不到怎么办？

A: 确保以下两点：
1. `--mvtec-root` 参数指向正确的MVTec数据集根目录
2. `--test-data` 参数指向正确的test.jsonl文件
3. test.jsonl中的`filename`字段路径是相对于mvtec-root的正确路径

### Q: 某些样本没有生成对比图？

A: 可能原因：
1. test.jsonl中的`filename`字段缺失或格式不正确
2. 图像文件不存在（检查mvtec-root路径）
3. 某个step的rollout文件中缺少该样本
4. rollout结果和test.jsonl的行号不对应

### Q: test.jsonl和rollout结果行号如何对应？

A: 工具假设test.jsonl的第N行对应rollout文件的第N行（从0开始）。确保rollout时按照test.jsonl的顺序处理样本。

### Q: 如何调整图像大小？

A: 目前图像会自动调整到相同高度。如果需要自定义，可以修改 `BboxStepComparator` 类中的 `_combine_images_horizontal` 方法。

### Q: 坐标转换不正确？

A: 工具假设rollout时图像被resize到1024x1024。如果使用了不同的size，可以在初始化时指定：

```python
comparator = BboxStepComparator(
    mvtec_data_root="data/mvtec",
    target_size=(512, 512)  # 修改为实际使用的size
)
```

## 扩展功能

如果需要更多功能，可以扩展 `BboxStepComparator` 类：

1. **添加IoU计算**: 在对比图上显示预测框与GT框的IoU
2. **生成视频**: 将多个step的变化制作成视频
3. **添加统计信息**: 显示整体的bbox准确率变化趋势
4. **支持多种颜色**: 为不同类型的defect使用不同颜色

## 相关工具

- `bbox_visualizer.py`: 单步骤的bbox可视化工具
- `rollout_viewer.py`: TUI界面查看rollout结果
- `run_visualization.py`: 批量可视化工具

## 许可证

Copyright 2025 Bytedance Ltd. and/or its affiliates
Licensed under the Apache License, Version 2.0

