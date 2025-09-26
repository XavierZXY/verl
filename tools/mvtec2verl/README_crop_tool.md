# MVTec to VERL Crop Tool Data Conversion

This directory contains tools to convert MVTec dataset to VERL format, with a specialized version for crop inspection tool training.

## Files Overview

- `convert_data.py` - Original conversion script for standard VL training
- `convert_data_crop_tool.py` - **NEW** Specialized conversion for crop inspection tool
- `data.toml` - Configuration for standard conversion
- `crop_data.toml` - **NEW** Configuration for crop tool conversion

## Key Differences: Standard vs Crop Tool Version

### System Prompt Comparison

| Aspect                  | Standard Version        | Crop Tool Version                  |
| ----------------------- | ----------------------- | ---------------------------------- |
| **Inspection Method**   | Single-pass analysis    | Progressive multi-round inspection |
| **Tool Usage**          | No tools mentioned      | Explicit crop tool instructions    |
| **Response Format**     | Direct answer format    | Tool-aware response format         |
| **Inspection Strategy** | Basic visual assessment | Strategic cropping methodology     |

### Standard System Prompt (Original)
```
You are a highly precise and meticulous quality control inspector. 
Your primary mission is to analyze images and determine if any defects are present.
[Direct format instructions without tool usage]
```

### Crop Tool System Prompt (New)
```
You are an advanced quality control inspector equipped with crop inspection tools 
for detailed defect analysis. Your mission is to thoroughly analyze images using 
progressive inspection techniques.

**INSPECTION METHODOLOGY:**
1. Initial Assessment: First examine the full image
2. Progressive Cropping: Use crop tool to focus on suspicious regions  
3. Multi-level Analysis: Up to 3 levels of cropping
4. Final Decision: Based on detailed inspection

**AVAILABLE TOOLS:**
- crop_from_location: Crop specific regions for detailed inspection
[Detailed tool usage instructions]
```

### Reward Model Differences

| Component           | Standard            | Crop Tool                       |
| ------------------- | ------------------- | ------------------------------- |
| **Reward Style**    | `"model"`           | `"crop_inspection"`             |
| **Reward Function** | `compute_score`     | `compute_crop_inspection_score` |
| **Tool Awareness**  | Not specified       | `use_crop_tool: true`           |
| **Environment**     | `visual_toolbox_v2` | `crop_inspection_tool`          |
| **Ability Type**    | `vl_chart`          | `vl_crop_inspection`            |

### Extra Info Enhancements

The crop tool version includes additional metadata:

```python
extra_info = {
    # Standard fields
    "answer": reward_model["ground_truth"],
    "question": CROP_INSPECTION_INSTRUCTION_PROMPT,
    "clsname": item.get("clsname"),
    "label": item.get("label"),
    "type": item.get("label_name"),
    "bboxes": bboxes_formatted,
    
    # NEW: Crop tool specific fields
    "tool_enabled": True,
    "crop_tool_available": True,
    "reward_type": "crop_inspection",
}
```

## Usage

### Standard Conversion
```bash
cd tools/mvtec2verl
python convert_data.py
```

### Crop Tool Conversion
```bash
cd tools/mvtec2verl
python convert_data_crop_tool.py
```

## Configuration

### Standard Config (data.toml)
```toml
input = "data.jsonl"
root = "."
output = "verl_dataset.parquet"
format = "parquet"
limit = null
```

### Crop Tool Config (crop_data.toml)
```toml
input = "data.jsonl"
root = "."
output = "verl_crop_dataset.parquet"  # Different output name
format = "parquet"
limit = null

# Additional crop tool configurations
[crop_tool]
max_crop_levels = 3
use_enhanced_evaluation = true

[crop_tool.reward_weights]
accuracy = 0.4
tool_usage = 0.3
localization = 0.2
# ... more weights

[training]
stage = "mid"  # early/mid/late for progressive training
```

## Expected Model Behavior

### Standard Model Response
```xml
<think>I can see a dark area that might be a defect...</think>
<answer>yes</answer>
<location>[{"bbox2d": [100, 150, 200, 250]}]</location>
<type>crack</type>
```

### Crop Tool Model Response
```xml
<think>I need to examine this image carefully. I see a suspicious area that requires closer inspection.</think>

<tool_call>
{"name": "crop_from_location", "arguments": {"location_data": "[{\"bbox2d\": [90, 140, 210, 260]}]", "crop_index": 0}}
</tool_call>

Now I can see the cropped region more clearly. Let me examine it further.

<tool_call>
{"name": "crop_from_location", "arguments": {"location_data": "[{\"bbox2d\": [95, 145, 205, 255]}]", "crop_index": 0}}
</tool_call>

After detailed inspection of the cropped regions, I can confirm there is a crack defect.

<answer>yes</answer>
<location>[{"bbox2d": [100, 150, 200, 250]}]</location>
<type>crack</type>
```

## Training Recommendations

### For Standard Model
- Use original `convert_data.py`
- Focus on direct visual analysis
- Optimize for single-pass accuracy
- Use standard VL reward metrics

### For Crop Tool Model
- Use new `convert_data_crop_tool.py`
- Train with progressive inspection examples
- Emphasize tool usage effectiveness
- Use crop-specific reward function
- Consider curriculum learning approach

## Integration with Training Pipeline

### Standard Integration
```python
from verl.utils.reward_score.vl_agent import compute_score

# Use standard reward computation
score = compute_score(predict_str, ground_truth, extra_info)
```

### Crop Tool Integration
```python
from verl.utils.reward_score.crop_inspection_reward import compute_crop_inspection_score

# Use crop-aware reward computation
score = compute_crop_inspection_score(
    predict_str, 
    ground_truth, 
    extra_info, 
    crop_history,  # Additional crop history
    use_enhanced_evaluation=True
)
```

## Benefits of Crop Tool Version

1. **Enhanced Accuracy**: Progressive inspection allows for more detailed analysis
2. **Tool Usage Training**: Models learn when and how to use inspection tools
3. **Adaptive Inspection**: Can adjust inspection depth based on image complexity
4. **Better Localization**: Multi-round cropping improves bbox precision
5. **Curriculum Learning**: Supports progressive difficulty training

## File Size and Performance

| Metric                 | Standard   | Crop Tool                    |
| ---------------------- | ---------- | ---------------------------- |
| **Prompt Length**      | ~500 chars | ~1500 chars                  |
| **System Complexity**  | Low        | Medium-High                  |
| **Training Time**      | Baseline   | +20-30%                      |
| **Inference Time**     | Baseline   | +50-100% (due to tool calls) |
| **Accuracy Potential** | Good       | Excellent                    |

## Troubleshooting

### Common Issues

1. **Missing crop_data.toml**
   - Script will fallback to data.toml
   - Create crop_data.toml for crop-specific settings

2. **Tool Call Format Errors**
   - Ensure JSON format is valid in tool calls
   - Check bbox coordinate format

3. **Reward Function Not Found**
   - Verify crop_inspection_reward module is available
   - Check import paths in training code

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger("rich").setLevel(logging.DEBUG)
```

## Migration Guide

To migrate from standard to crop tool version:

1. **Update Data Conversion**:
   ```bash
   python convert_data_crop_tool.py
   ```

2. **Update Training Config**:
   - Change `env_name` to `crop_inspection_tool`
   - Update reward function reference
   - Add crop tool configurations

3. **Update Evaluation**:
   - Use crop-specific reward computation
   - Include crop history in evaluation
   - Adjust success metrics for tool usage

4. **Model Architecture**:
   - Ensure model supports tool calling format
   - Add tool usage prediction heads if needed
   - Consider multi-turn conversation handling
