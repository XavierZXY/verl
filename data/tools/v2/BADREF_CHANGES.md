# 缺陷参考图修改说明

## 修改概述

将原来使用**无缺陷图片**作为参考图的逻辑，改为使用**带红色标注框的有缺陷图片**作为参考图。

## 核心修改

### 1. 新增依赖
```python
from PIL import Image, ImageDraw  # 添加 ImageDraw
import numpy as np  # 用于图像处理
from scipy import ndimage  # 用于连通域检测
```

### 2. 函数替换

#### 原函数: `_build_good_image_map()`
- **功能**: 为每个类别选择一个无缺陷图片
- **返回**: `dict[str, str]` - 类别名 -> 无缺陷图片路径

#### 新函数: `_build_defect_reference_map()`
- **功能**: 为每个类别选择一个有缺陷的图片及其mask
- **返回**: `dict[str, tuple[str, str]]` - 类别名 -> (缺陷图片路径, mask路径)
- **选择逻辑**: 选择每个类别第一个同时存在图片和mask的缺陷样本

### 3. 新增核心函数: `_create_annotated_reference_image()`

该函数负责生成带红色标注框的参考图：

```python
def _create_annotated_reference_image(
    image_path: str,
    mask_path: str,
    compress: bool = True,
    quality: int = 85,
    max_size: Optional[tuple[int, int]] = None,
    box_color: tuple[int, int, int] = (255, 0, 0),  # 红色
    box_width: int = 3,
) -> Optional[bytes]:
```

**处理流程**:
1. 读取原始缺陷图片和mask图片
2. 将图片转换为RGB格式
3. 按需调整图片尺寸（同时调整mask）
4. 使用连通域分析（`scipy.ndimage.label`）从mask中提取所有缺陷区域
5. 为每个缺陷区域计算边界框（bounding box）
6. 在原图上绘制红色边界框
7. 压缩并返回字节数据

**技术细节**:
- 使用 `scipy.ndimage.label()` 进行连通域标记
- 二值化阈值: 128
- 对每个连通域计算最小外接矩形
- 使用 `PIL.ImageDraw.rectangle()` 绘制红框

### 4. convert() 函数修改

**之前**:
```python
# Build good image map for all classes
good_image_map = _build_good_image_map(items, dataset_root)

# Get good reference image for this class
good_image_bytes = _read_image_bytes(good_img_path, ...)
extra_info["good_reference_image"] = good_image_bytes
```

**修改后**:
```python
# Build defect reference image map for all classes
defect_reference_map = _build_defect_reference_map(items, dataset_root)

# Get annotated defect reference image for this class
annotated_reference_bytes = _create_annotated_reference_image(
    defect_img_path, defect_mask_path, ...
)
extra_info["annotated_defect_reference_image"] = annotated_reference_bytes
```

### 5. 数据字段更新

**extra_info 中的字段变化**:
- 统一使用: `reference_image`（替代原来的 `good_reference_image` 和 `annotated_defect_reference_image`）
- 无论是无缺陷参考图还是带标注的缺陷参考图，都使用相同的key，方便工具通用

**JSONL 序列化**:
```python
# 序列化参考图
if "reference_image" in extra_info and isinstance(extra_info["reference_image"], bytes):
    extra_info["reference_image"] = base64.b64encode(
        extra_info["reference_image"]
    ).decode("ascii")
```

## 使用方式

### 运行数据转换

```bash
cd /home/takisobe@amd.com/zxy/codes/verl-compare/data/tools/v2
python convert_data_badref.py
```

确保 `data.toml` 配置文件已正确设置。

### 测试标注功能

```bash
python test_annotated_ref.py
```

需要先在脚本中设置实际的图片和mask路径。

## 配置参数

可以通过修改函数调用来调整标注框的外观：

```python
annotated_reference_bytes = _create_annotated_reference_image(
    image_path=defect_img_path,
    mask_path=defect_mask_path,
    compress=True,
    quality=85,
    max_size=(800, 800),
    box_color=(255, 0, 0),    # RGB颜色，默认红色
    box_width=3,              # 边框宽度，默认3像素
)
```

## 日志输出

程序会输出以下关键信息：

```
Building defect reference image map for all classes...
Selected defect reference image for class 'bottle': /path/to/image.jpg with mask: /path/to/mask.png
Found defect reference images for N classes
```

## 依赖要求

- PIL (Pillow)
- numpy
- scipy
- pandas
- rich

## 注意事项

1. **mask要求**: 每个类别必须至少有一个同时包含图片和mask的缺陷样本
2. **连通域检测**: 使用二值化阈值128，mask中大于128的像素被认为是缺陷区域
3. **多缺陷支持**: 自动检测并标注mask中的所有连通域
4. **图片格式**: 最终输出为JPEG格式（如果compress=True）
5. **错误处理**: 如果创建标注图失败，会记录警告但不会中断处理

## 优势

1. **更明确的参考**: 直接展示缺陷位置，比无缺陷图片更有指导意义
2. **视觉对比**: 模型可以直观看到"这就是缺陷的样子"
3. **标注一致性**: 使用mask确保标注准确
4. **自动化**: 无需人工绘制标注框

