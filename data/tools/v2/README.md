# Data Conversion Tool V2

## 主要改动

相比于原始版本，V2 版本做了以下关键改动：

### 1. 保留图片 Resize 和压缩功能
- 继续支持图片压缩和 resize 功能
- 通过 `compress_images`, `image_quality`, `max_image_size` 参数控制

### 2. 直接保存 Mask Image
- **不再计算 bbox_2d 坐标值**
- 直接将 mask image 的字节数据保存到 `extra_info["mask_image"]` 中
- 如果对原图进行了 resize，会对 mask image 应用**相同的 resize 处理**
- 保证 mask image 和原图尺寸一致

### 3. 添加同类别 Good 参考图片
- 为每个样本在 `extra_info["good_reference_image"]` 中添加一张同类别的 good 图片
- 对每个类别，自动从数据集中选择第一个找到的 good 样本作为参考
- Good 参考图片也会应用**相同的 resize 处理**

## 使用方法

### 1. 配置文件

编辑 `data.toml` 文件：

```toml
# 输入 JSONL 文件
input = "../../mvtec/moredata/train.jsonl"

# 数据集根目录
root = "../../mvtec/moredata"

# 输出文件路径
output = "../../mvtec/moredata/train_v2.parquet"

# 输出格式: "parquet" 或 "jsonl"
format = "parquet"

# 限制处理数量 (null 表示处理全部)
limit = null

# 图片压缩设置
compress_images = true
image_quality = 85

# 最大图片尺寸 [width, height] (null 表示不 resize)
max_image_size = [512, 512]
```

### 2. 运行转换

```bash
cd /home/takisobe@amd.com/zxy/codes/verl/data/tools/v2
python convert_data.py
```

## 数据格式

### Extra Info 字段

转换后的数据在 `extra_info` 中包含以下字段：

```python
{
    "answer": {...},                    # 答案信息
    "question": "...",                  # 问题文本
    "prompt_variant_index": 0,          # 提示词变体索引
    "clsname": "grid",                  # 类别名称
    "label": 1,                         # 标签 (0=good, 1=defect)
    "type": "bent",                     # 缺陷类型
    "index": 0,                         # 序号
    "mask_image": b"...",              # Mask 图片字节数据 (仅 label=1 时存在)
    "good_reference_image": b"..."     # 同类别 Good 参考图片字节数据
}
```

### Mask Image
- **格式**: PNG (保持 mask 质量)
- **通道**: 灰度图 (L) 或二值图 (1)
- **尺寸**: 与主图片一致（应用了相同的 resize）
- **存在条件**: 仅当 `label=1` (缺陷样本) 且存在 mask 文件时

### Good Reference Image
- **格式**: JPEG (压缩存储)
- **通道**: RGB
- **尺寸**: 与主图片一致（应用了相同的 resize）
- **来源**: 该类别中第一个 label=0 的样本

## 依赖

```bash
pip install pandas pillow rich tomli  # Python < 3.11
pip install pandas pillow rich        # Python >= 3.11
```

## 与原版本的对比

| 特性 | 原版本 | V2 版本 |
|------|--------|---------|
| Bbox 计算 | ✅ 从 mask 计算 bbox | ❌ 不计算 |
| Mask 保存 | ❌ 不保存 | ✅ 保存原始 mask image |
| Good 参考图 | ❌ 无 | ✅ 保存同类 good 图片 |
| Resize 支持 | ✅ 支持 | ✅ 支持（包括 mask 和 good 图） |
| 压缩支持 | ✅ 支持 | ✅ 支持 |

## 注意事项

1. **Mask Image 处理**：当对图片进行 resize 时，mask image 会自动应用相同的变换
2. **Good 参考图选择**：程序会自动为每个类别选择第一个遇到的 good 样本，建议提前检查数据质量
3. **存储空间**：由于保存了额外的图片数据（mask + good reference），V2 版本的输出文件会比原版本大
4. **处理速度**：由于需要读取和处理额外的图片，处理速度可能会略慢于原版本

## 示例

查看生成的数据：

```python
import pandas as pd

# 读取 parquet 文件
df = pd.read_parquet("../../mvtec/moredata/train_v2.parquet")

# 查看第一行的 extra_info
extra_info = df.iloc[0]["extra_info"]
print(f"Class: {extra_info['clsname']}")
print(f"Label: {extra_info['label']}")
print(f"Type: {extra_info['type']}")
print(f"Has mask: {'mask_image' in extra_info}")
print(f"Has good reference: {'good_reference_image' in extra_info}")

# 如果有 mask，可以这样读取
if 'mask_image' in extra_info:
    from PIL import Image
    from io import BytesIO
    mask_img = Image.open(BytesIO(extra_info['mask_image']))
    mask_img.show()
```

