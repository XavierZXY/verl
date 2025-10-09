# Bbox Visualization Tool for Rollout Results

这个工具用于将rollout结果中的bbox显示到原始图像上，支持从1024x1024尺寸到原始图像尺寸的坐标变换。

## 功能特性

- 从rollout结果中提取预测的bounding box坐标
- 处理从1024x1024到原始图像尺寸的坐标变换
- 在原始图像上绘制预测的bbox（红色）和ground truth bbox（绿色）
- 支持批量处理多张图像
- 生成可视化结果图像

## 文件结构

```
rollout/
├── bbox_visualizer.py      # 主要的可视化工具类
├── test_visualizer.py      # 测试脚本
├── run_visualization.py    # 使用示例脚本
└── README.md              # 说明文档
```

## 依赖包

确保安装以下Python包：

```bash
pip install opencv-python pillow numpy
```

## 使用方法

### 1. 命令行使用

```bash
python bbox_visualizer.py \
    --test-data /path/to/test.jsonl \
    --rollout-results /path/to/rollout_results.jsonl \
    --output-dir /path/to/output \
    --mvtec-root /path/to/mvtec/data \
    --max-images 10
```

### 2. 使用示例脚本

```bash
python run_visualization.py
```

### 3. 在Python代码中使用

```python
from bbox_visualizer import BboxVisualizer

# 创建可视化器
visualizer = BboxVisualizer(mvtec_data_root="/path/to/mvtec")

# 运行可视化
visualizer.visualize_rollout_results(
    test_jsonl_path="test.jsonl",
    rollout_jsonl_path="rollout_results.jsonl",
    output_dir="visualization_results",
    max_images=10
)
```

## 数据格式

### Test Data (test.jsonl)

每行包含一个JSON对象：

```json
{
    "filename": "bottle/test/good/004.png",
    "foreground": "bottle/foreground/test/good/004.png",
    "clsname": "bottle",
    "label": 0,
    "label_name": "good",
    "bboxes": [[294, 883, 1683, 1738]],  // 可选，ground truth bbox
    "bbox_count": 1,
    "bbox_valid": true,
    "bbox": [294, 883, 1683, 1738]
}
```

### Rollout Results (rollout_results.jsonl)

每行包含一个JSON对象：

```json
{
    "input": "system prompt and user input...",
    "output": "<think>...</think>\n<location>[{\"bbox2d\": [130, 148, 897, 897]}]</location>\n<type>crack</type>\n<answer>yes</answer>",
    "gts": {"answer": "...", "bboxes": [...]},
    "score": 0.5,
    "step": 60,
    "reward": 0.5
}
```

## 坐标变换

工具会自动处理坐标变换：

1. **输入坐标**：rollout结果中的bbox坐标基于1024x1024的图像
2. **输出坐标**：变换到原始图像尺寸的坐标
3. **变换公式**：
   ```
   scale_x = original_width / 1024
   scale_y = original_height / 1024
   
   new_x = old_x * scale_x
   new_y = old_y * scale_y
   ```

## 可视化结果

- **红色框**：模型预测的bbox
- **绿色框**：Ground truth bbox（如果有的话）
- **标签**：在bbox上方显示"Predicted"或"Ground Truth"

## 测试

运行测试脚本来验证工具功能：

```bash
python test_visualizer.py
```

## 注意事项

1. 确保MVTec数据集的路径正确
2. 图像文件必须存在于指定的路径
3. 工具会跳过没有预测bbox的结果
4. 输出目录会自动创建

## 故障排除

### 常见问题

1. **图像加载失败**
   - 检查图像路径是否正确
   - 确保图像文件存在且可读

2. **坐标变换错误**
   - 验证原始图像尺寸
   - 检查bbox坐标格式

3. **依赖包问题**
   - 安装所需的Python包
   - 检查OpenCV版本兼容性

### 调试模式

在代码中添加调试信息：

```python
# 在visualize_rollout_results方法中添加
print(f"Processing image: {filename}")
print(f"Original size: {original_size}")
print(f"Predicted bboxes: {predicted_bboxes}")
print(f"Transformed bboxes: {transformed_predicted}")
```

## 示例输出

运行成功后，你会看到类似的输出：

```
Processed 1: bottle/test/broken_large/000.png
  Predicted bboxes: 1
  Ground truth bboxes: 1
  Saved to: visualization_results/001_000_visualization.jpg

Processed 2: bottle/test/broken_large/001.png
  Predicted bboxes: 1
  Ground truth bboxes: 1
  Saved to: visualization_results/002_001_visualization.jpg

Visualization complete! Processed 5 images.
Results saved to: visualization_results
```