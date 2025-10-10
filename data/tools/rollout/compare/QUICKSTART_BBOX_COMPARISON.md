# 快速开始：Bbox Step Comparison

## 5分钟上手指南

### 1. 确认数据准备

确保你有以下文件：
- ✅ test.jsonl文件（包含ground truth）
- ✅ 多个step的rollout结果（如0.jsonl, 40.jsonl, 180.jsonl）
- ✅ MVTec数据集图像文件

### 2. 安装依赖

```bash
pip install opencv-python numpy
```

### 3. 运行第一个对比

```bash
# 对比前5个样本，查看步骤0, 40, 180的变化
python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data data/mvtec/test/test.jsonl \
    --steps 0 40 180 \
    --output-dir logs/rollout/my_first_comparison \
    --max-samples 5
```

### 4. 查看结果

生成的图像保存在 `logs/rollout/my_first_comparison/` 目录：
- `sample_000_comparison.jpg`
- `sample_001_comparison.jpg`
- ...

每张图片包含：
- **绿色框**: Ground Truth（真实的缺陷位置）
- **红色框**: 模型预测的bbox
- **顶部**: Step编号、Score、Answer
- **底部**: 图例说明

### 5. 常用命令

#### 对比特定样本

```bash
# 只对比第10, 20, 30个样本
python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data data/mvtec/test/test.jsonl \
    --steps 0 40 180 \
    --output-dir logs/rollout/selected_samples \
    --sample-indices 10 20 30
```

#### 对比更多步骤

```bash
# 对比10个不同的训练步骤
python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data data/mvtec/test/test.jsonl \
    --steps 0 20 40 60 80 100 120 140 160 180 \
    --output-dir logs/rollout/detailed_comparison \
    --max-samples 3
```

#### 使用Shell脚本（更简单）

```bash
chmod +x scripts/compare_bbox_steps.sh

./scripts/compare_bbox_steps.sh --steps 0 40 180 --max-samples 10
```

### 6. 目录结构示例

```
verl/
├── data/
│   └── mvtec/
│       ├── test/
│       │   └── test.jsonl          # ← test data文件
│       ├── bottle/
│       │   └── test/
│       │       └── broken_large/
│       │           └── 000.png     # ← 图像文件
│       └── ...
├── logs/
│   └── rollout/
│       └── validation/
│           └── mvtec/
│               ├── 0.jsonl         # ← rollout结果
│               ├── 40.jsonl
│               └── 180.jsonl
└── scripts/
    └── bbox_step_comparator.py     # ← 对比工具
```

### 7. 验证结果

打开生成的图像，检查：
1. ✅ 是否显示了所有step的预测
2. ✅ 绿色GT框是否在正确位置
3. ✅ 红色预测框是否随训练改进
4. ✅ Score和Answer是否符合预期

### 8. 故障排除

#### 问题：找不到图像文件

```bash
# 检查MVTec root路径是否正确
ls data/mvtec/bottle/test/broken_large/000.png

# 如果路径不对，使用--mvtec-root参数
python scripts/bbox_step_comparator.py \
    --mvtec-root /path/to/your/mvtec \
    --test-data /path/to/test.jsonl \
    ...
```

#### 问题：某些样本没有生成

```bash
# 检查test.jsonl和rollout文件行数是否一致
wc -l data/mvtec/test/test.jsonl
wc -l logs/rollout/validation/mvtec/0.jsonl
```

#### 问题：bbox坐标转换不正确

如果你的rollout使用了不同的图像尺寸（不是1024x1024），需要修改代码：

```python
# 在Python中初始化时指定
comparator = BboxStepComparator(
    mvtec_data_root="data/mvtec",
    target_size=(512, 512)  # 改为你的尺寸
)
```

### 9. 下一步

- 📖 查看完整文档：`BBOX_COMPARISON_README.md`
- 🔧 自定义可视化：修改`draw_single_step_bbox`方法
- 📊 批量处理：编写脚本循环处理多个配置
- 📈 生成报告：解析score和bbox IoU，生成统计图表

### 10. 示例脚本

运行预配置的示例：

```bash
chmod +x scripts/example_compare_bbox.sh
./scripts/example_compare_bbox.sh
```

## 需要帮助？

- 遇到bug？检查错误消息中的文件路径
- 需要新功能？修改`BboxStepComparator`类
- 有疑问？查看`BBOX_COMPARISON_README.md`

Happy comparing! 🎉

