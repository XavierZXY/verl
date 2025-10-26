# 多缺陷图像裁剪工具

这个工具扩展了原始的数据转换脚本，增加了自动检测和裁剪多个缺陷区域的功能。

## 主要功能

- **自动缺陷检测**：使用连通组件分析从mask图中自动检测所有独立的缺陷区域
- **智能裁剪**：为每个检测到的缺陷区域生成独立的裁剪图像
- **同步处理**：同时裁剪原图和mask图，保持坐标一致
- **尺寸控制**：确保裁剪图像至少为768x768（可配置）
- **数据增强**：将一张多缺陷图片转换为多条训练样本

## 工作原理

### 1. 缺陷区域检测

对于每张有缺陷的图片（label=1且存在mask）：
- 加载mask图像并二值化
- 使用OpenCV的连通组件分析检测所有白色区域
- 过滤掉小于最小面积阈值的噪点
- 为每个独立的缺陷区域计算边界框

### 2. 裁剪框计算

对于每个检测到的缺陷区域：
- 获取缺陷的边界框 `[x_min, y_min, x_max, y_max]`
- 添加padding（默认10%的缺陷尺寸）
- 扩展到最小裁剪尺寸（默认768x768）
- 以缺陷中心为基准进行扩展
- 处理边界情况，确保不超出原图范围

### 3. 同步裁剪

使用相同的裁剪坐标同时裁剪：
- 原始图像
- mask图像

### 4. 数据记录生成

原本一条记录变为N条（N = 检测到的缺陷数量），每条记录包含：
- 裁剪后的图像
- 裁剪后的mask
- 原始的prompt和reward信息
- 额外的crop元数据

## 配置说明

在 `data.toml` 文件中配置：

```toml
[crop]
# 启用裁剪功能
enabled = true

# 最小裁剪尺寸（像素）
min_crop_size = 768

# 最小缺陷面积阈值（像素）
# 小于此值的区域会被视为噪点过滤掉
min_defect_area = 100

# 缺陷周围的padding比例
# 0.1 表示在缺陷边界基础上额外增加10%的空间
padding_ratio = 0.1
```

## 使用方法

1. **编辑配置文件** `data.toml`：
   ```toml
   input = "path/to/input.jsonl"
   root = "path/to/dataset"
   output = "path/to/output.parquet"
   
   [crop]
   enabled = true
   min_crop_size = 768
   min_defect_area = 100
   padding_ratio = 0.1
   ```

2. **运行转换脚本**：
   ```bash
   python convert_data.py
   ```

3. **查看输出**：
   - 每张多缺陷图片会被拆分成多条记录
   - 控制台会显示检测到的缺陷数量
   - 输出的parquet/jsonl文件包含所有裁剪后的图像

## 输出数据格式

每条记录的 `extra_info` 中包含以下crop相关字段：

```python
{
    "is_cropped": True,              # 是否为裁剪图像
    "crop_index": 0,                 # 当前缺陷的索引（第几个缺陷）
    "total_crops": 3,                # 原图总共有几个缺陷
    "crop_bbox": [100, 200, 868, 968],  # 裁剪框坐标 [x_min, y_min, x_max, y_max]
    "crop_size": [768, 768],         # 裁剪后的尺寸 [width, height]
    "original_image_size": [1920, 1080],  # 原图尺寸
    "defect_bbox": [150, 250, 350, 450],  # 原始缺陷边界框
    "mask_image": <bytes>,           # 裁剪后的mask图像
    ...
}
```

## 示例场景

### 场景1：单个缺陷
- 输入：1张图片，1个缺陷
- 输出：1条记录（裁剪后的图像，768x768或更大）

### 场景2：多个缺陷（如你的示例图片）
- 输入：1张图片，3个独立的缺陷区域
- 输出：3条记录，每条对应一个缺陷的裁剪图像

### 场景3：良品（无缺陷）
- 输入：1张图片，label=0
- 输出：1条记录（原图，不裁剪）

### 场景4：无mask的缺陷图
- 输入：1张图片，label=1但无mask
- 输出：1条记录（原图，不裁剪）

## 特殊情况处理

| 情况 | 处理方式 |
|------|---------|
| mask不存在 | 保持原图，不crop |
| mask全黑（无白色区域） | 保持原图，记录警告 |
| 多个缺陷重叠 | 分别检测和crop，可能有重叠区域 |
| 缺陷在图像边缘 | 调整crop边界，可能小于min_crop_size |
| 原图 < min_crop_size | 使用原图尺寸，不强制扩展 |
| 缺陷太小（< min_defect_area） | 过滤掉，视为噪点 |

## 依赖要求

```bash
pip install opencv-python numpy pillow pandas rich
```

## 调试技巧

1. **查看检测到的缺陷数量**：
   ```
   [INFO] Detected 3 defect region(s) from mask
   [INFO] [0] Cropping 3 defect region(s) from image
   ```

2. **调整参数**：
   - 如果检测到太多噪点：增大 `min_defect_area`
   - 如果裁剪框太紧：增大 `padding_ratio`
   - 如果需要更大的裁剪图：增大 `min_crop_size`

3. **禁用crop功能**：
   ```toml
   [crop]
   enabled = false
   ```
   这样会回退到原始行为（不裁剪）

## 性能建议

- 启用图像压缩可以显著减小输出文件大小
- 使用 `max_image_size` 在crop之前先缩放原图
- 对于大规模数据集，建议使用parquet格式而非jsonl
- 使用 `limit` 参数先测试小批量数据

## 与原始版本的差异

| 特性 | 原始版本 | Crop版本 |
|------|---------|----------|
| 缺陷图处理 | 保留完整图像 | 自动裁剪为多个小图 |
| 输出记录数 | N张图 → N条记录 | N张图 → M条记录（M≥N） |
| mask用途 | 仅存储 | 用于检测+存储裁剪版本 |
| 数据增强 | 仅prompt变化 | prompt变化 + 多区域裁剪 |
| 配置复杂度 | 简单 | 中等（增加crop配置） |

## 问题排查

### Q: 为什么某些缺陷图没有被裁剪？
A: 可能的原因：
- mask文件不存在或无法读取
- mask是全黑的（没有检测到白色区域）
- 检测到的区域面积小于 `min_defect_area`
- `crop.enabled` 设置为 false

### Q: 裁剪框包含的范围太大/太小？
A: 调整 `padding_ratio` 和 `min_crop_size`：
- 减小 `padding_ratio`：裁剪框更紧凑
- 增大 `min_crop_size`：确保更大的裁剪图像

### Q: 性能太慢？
A: 优化建议：
- 先用 `limit` 参数测试
- 启用 `compress_images` 和设置合理的 `image_quality`
- 使用 `max_image_size` 预先缩放
- 确保使用SSD存储

