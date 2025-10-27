# Category-Specific Prior Knowledge System

## 概述

该系统为 MVTec 和 VISA 等缺陷检测数据集的每个类别创建了专门的先验知识和针对性的指令提示，以提高缺陷检测模型的性能和准确性。系统支持 18 个不同类别，涵盖工业制造、电子产品、食品质检等多个领域。

## 支持的类别

系统目前支持以下 18 个类别，每个类别都有定制的先验知识：

### MVTec 数据集类别 (13个)

### 1. **Bottle（瓶子）**
- **描述**: 用于饮料包装的玻璃或塑料瓶
- **常见缺陷**: 破损/裂纹、污染、表面损伤
- **检查重点**: 表面完整性、裂纹、异物颗粒

### 2. **Cable（电缆）**
- **描述**: 带有电线和绝缘层的电缆
- **常见缺陷**: 弯曲的电线、电缆交换、切割绝缘、缺失组件、戳孔损伤
- **检查重点**: 电线对齐、绝缘完整性、电缆配置

### 3. **Capsule（胶囊）**
- **描述**: 药用胶囊或药丸容器
- **常见缺陷**: 裂纹、印刷错误、戳孔损伤、划痕、挤压变形
- **检查重点**: 表面光滑度、印刷质量、结构完整性

### 4. **Carpet（地毯）**
- **描述**: 纺织地毯或织物材料
- **常见缺陷**: 颜色变化、切口、孔洞、金属污染、线头问题
- **检查重点**: 颜色均匀性、表面完整性、异物

### 5. **Grid（网格）**
- **描述**: 金属或塑料网格图案
- **常见缺陷**: 弯曲、破损、胶水残留、金属污染、线头损坏
- **检查重点**: 网格对齐、结构完整性、表面清洁度

### 6. **Hazelnut（榛子）**
- **描述**: 榛子坚果质量检查
- **常见缺陷**: 裂纹、切口、孔洞、印刷标记
- **检查重点**: 壳的完整性、表面损伤、孔洞

### 7. **Leather（皮革）**
- **描述**: 制造用皮革材料
- **常见缺陷**: 颜色变化、切口、折痕、胶印、戳孔损伤
- **检查重点**: 表面光滑度、颜色均匀性、结构完整性

### 8. **Pill（药片）**
- **描述**: 药片或片剂
- **常见缺陷**: 划痕、表面损伤
- **检查重点**: 表面光滑度、涂层完整性

### 9. **Screw（螺丝）**
- **描述**: 金属螺丝或紧固件
- **常见缺陷**: 正面操控、头部划痕、颈部划痕、螺纹损伤
- **检查重点**: 螺纹完整性、头部状况、表面划痕

### 10. **Tile（瓷砖）**
- **描述**: 陶瓷或地板砖
- **常见缺陷**: 裂纹、胶条、灰色条纹、油渍、粗糙表面
- **检查重点**: 表面光滑度、裂纹、污渍、纹理均匀性

### 11. **Toothbrush（牙刷）**
- **描述**: 牙刷产品
- **常见缺陷**: 刷毛缺陷、手柄损伤、组装问题
- **检查重点**: 刷毛质量、手柄完整性、整体组装

### 12. **Transistor（晶体管）**
- **描述**: 电子晶体管组件
- **常见缺陷**: 弯曲引脚、切断引脚、损坏外壳、错位
- **检查重点**: 引脚对齐、外壳完整性、组件定位

### 13. **Wood（木材）**
- **描述**: 木材或木制品
- **常见缺陷**: 颜色变化、组合缺陷、孔洞、液体污渍、划痕
- **检查重点**: 表面光滑度、颜色均匀性、孔洞、污渍

### VISA 数据集类别 (5个)

### 14. **Capsules（胶囊 - 带标签）**
- **描述**: 医疗或药品胶囊，带有印刷标签
- **常见缺陷**: 印刷错误、标签缺陷、颜色不一致、变形、裂纹
- **检查重点**: 标签清晰度、印刷质量、形状完整性、表面缺陷

### 15. **Macaroni1（通心粉）**
- **描述**: 通心粉形式的意大利面产品，用于食品质量检查
- **常见缺陷**: 破碎片段、变色、形状变形、异物、烧焦区域
- **检查重点**: 形状均匀性、颜色一致性、结构完整性、污染

### 16. **PCB1（印刷电路板）**
- **描述**: 电子设备用印刷电路板
- **常见缺陷**: 缺失组件、焊接缺陷、走线损伤、短路、组件错位
- **检查重点**: 组件存在性、焊接质量、走线完整性、对齐精度

### 17. **Pipe Fryum（管状油炸食品）**
- **描述**: 管状油炸小食品
- **常见缺陷**: 破碎片段、烧焦区域、变色、变形、油炸不完全
- **检查重点**: 形状完整性、颜色均匀性、油炸质量、结构完整度

### 18. **Fryum（油炸食品）**
- **描述**: 各种形状的油炸小食品
- **常见缺陷**: 破碎片段、烧焦区域、变色、变形、油斑
- **检查重点**: 形状完整性、颜色均匀性、油炸质量、表面缺陷

## 使用方法

### 基本用法

系统会自动根据 `clsname` 字段选择对应的类别知识和提示：

```python
from convert_data import get_category_specific_prompt

# 为 bottle 类别获取特定提示
prompt, index = get_category_specific_prompt("bottle")
print(prompt)
# 输出示例: "<image>.\nInspect this bottle for any cracks, breaks, or contamination..."
```

### 在数据转换中的应用

在 `convert()` 函数中，系统会：

1. **自动选择类别特定提示**: 根据每个样本的 `clsname` 从该类别的 3 个专用提示中随机选择
2. **添加类别知识**: 在 `extra_info` 中存储类别描述、常见缺陷和检查重点
3. **后备机制**: 如果类别未识别，使用通用提示

### 数据结构

每个处理后的记录在 `extra_info` 中包含以下类别知识：

```json
{
  "category_knowledge": {
    "category_description": "Glass or plastic bottles used in beverage packaging",
    "common_defects": ["broken/cracks", "contamination", "surface damage"],
    "inspection_focus": "surface integrity, cracks, foreign particles"
  }
}
```

## 提示多样性

每个类别有 **3 个不同的提示变体**，以增加训练数据的多样性：

- **变体 1**: 关注主要缺陷类型
- **变体 2**: 强调检查流程
- **变体 3**: 综合质量评估

例如，bottle 类别的 3 个提示：
1. "Inspect this bottle for any cracks, breaks, or contamination..."
2. "Examine this bottle image for manufacturing defects such as..."
3. "Analyze the bottle for structural integrity. Look for cracks..."

## 扩展系统

要添加新类别，只需在 `CATEGORY_KNOWLEDGE` 字典中添加新条目：

```python
CATEGORY_KNOWLEDGE = {
    # ... 现有类别 ...
    "new_category": {
        "description": "类别描述",
        "common_defects": ["缺陷1", "缺陷2", "缺陷3"],
        "inspection_focus": "检查重点说明",
        "prompts": [
            "<image>.\n提示变体1...",
            "<image>.\n提示变体2...",
            "<image>.\n提示变体3...",
        ]
    },
}
```

## 优势

1. **针对性强**: 每个类别的提示都针对该类别的特定缺陷类型
2. **知识注入**: 系统提示包含了特定领域的检查知识
3. **数据多样性**: 每个类别有多个提示变体，增加训练样本的多样性
4. **可扩展**: 易于添加新类别和更新现有类别的知识
5. **可追溯**: 每个样本都记录了使用的提示索引和类别知识

## 配置文件

系统使用 `data.toml` 进行配置，无需修改代码即可调整参数：

```toml
input = "data/data-v2/mvtec/mvtec_test.jsonl"
root = "data/data-v2/mvtec"
output = "verl_dataset.parquet"
format = "parquet"
limit = null
compress_images = true
image_quality = 85
max_image_size = [512, 512]
```

## 运行示例

```bash
cd /home/takisobe@amd.com/zxy/codes/verl-compare/data/tools/v2
python convert_data.py
```

系统将：
- 读取 MVTec 或 VISA 数据集
- 为每个样本根据其类别选择特定提示
- 添加类别先验知识
- 生成包含所有信息的 parquet/jsonl 文件

### 多数据集支持示例

**处理 MVTec 数据集：**
```bash
# 修改 data.toml
input = "data/data-v2/mvtec/mvtec_test.jsonl"
root = "data/data-v2/mvtec"
output = "mvtec_verl_dataset.parquet"
```

**处理 VISA 数据集：**
```bash
# 修改 data.toml
input = "data/visa-2k/moredata/visa-test.jsonl"
root = "data/visa-2k/moredata"
output = "visa_verl_dataset.parquet"
```

## 日志输出示例

```
[INFO] Building good image map for all classes...
[INFO] Selected good image for class 'bottle': bottle/train/good/000.png
[INFO] Selected good image for class 'cable': cable/train/good/000.png
...
[INFO] Processed 100 items...
[INFO] Processed 200 items...
...
[INFO] Processed 1823 items (limit=none), wrote parquet with 1823 rows to: verl_dataset.parquet
```

## 技术细节

- **随机选择**: 使用 `random.randint()` 从类别特定提示中随机选择
- **后备机制**: 未知类别使用 `GENERIC_INSTRUCTION_PROMPTS`
- **记录索引**: 每个提示的使用都被记录在 `prompt_variant_index`
- **向后兼容**: 保留 `get_random_instruction_prompt()` 函数以保持兼容性

## 数据增强效果

### MVTec 数据集
- **样本数**: 1823 个
- **类别数**: 13 个（bottle, cable, capsule, carpet, grid, hazelnut, leather, pill, screw, tile, toothbrush, transistor, wood）
- **提示组合**: 13 × 3 = 39 种专用提示
- **领域**: 工业制造、材料检验

### VISA 数据集
- **样本数**: 501 个
- **类别数**: 5 个（capsules, macaroni1, pcb1, pipe_fryum, fryum）
- **提示组合**: 5 × 3 = 15 种专用提示
- **领域**: 食品质检、电子产品

### 整体统计
- **总类别**: 18 个
- **总提示变体**: 54 种（18类别 × 3变体 + 5种通用后备）
- **知识注入**: 每个样本都附带类别描述、常见缺陷、检查重点
- **准确性提升**: 针对性提示比通用提示更能引导模型关注特定缺陷类型

