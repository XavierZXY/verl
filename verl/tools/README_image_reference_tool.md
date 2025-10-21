# Image Reference Tool

## Overview

`ImageReferenceTool` is a tool designed for quality control inspection tasks. It provides access to defect-free reference images of the same object class, enabling inspectors to compare current samples against known good samples.

## Features

- **Simple Access**: Directly retrieves pre-loaded reference images from sample metadata
- **No Parameters Required**: Optionally accepts a `reason` parameter for documentation
- **Lightweight**: Much simpler than zoom tool since images are pre-loaded
- **Comparison Aid**: Helps distinguish between normal variations and actual defects

## Tool Schema

```python
{
    "type": "function",
    "function": {
        "name": "image_reference_tool",
        "description": "Retrieve a defect-free reference image of the same object class for comparison.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Optional explanation of why you need the reference image"
                }
            },
            "required": []
        }
    }
}
```

## Usage in Training Pipeline

### 1. Data Preparation

The reference images are prepared during data conversion in `convert_data.py`:

```python
# In convert_data.py, line 351-352
if good_image_bytes is not None:
    extra_info["good_reference_image"] = good_image_bytes
```

Each sample's `extra_info` dictionary contains a `good_reference_image` field with the bytes data of a defect-free sample from the same class.

### 2. Tool Initialization

The tool is initialized in the dataset class (e.g., `qwen_iad.py`):

```python
# Get good reference image from extra_info
good_reference_image = row_dict.get("extra_info", {}).get("good_reference_image")

tools_kwargs = {
    "image_reference_tool": {
        "create_kwargs": {"good_reference_image": good_reference_image},
    }
}
```

### 3. Tool Invocation

Agents can call the tool during inspection:

```xml
<think>
I notice some texture variations on the surface. To determine if this is a defect 
or normal surface pattern, I need to compare it with a defect-free reference image.
</think>
<tool_call>
[{"tool_name": "image_reference_tool", "parameters": {"reason": "to compare surface texture patterns"}}]
</tool_call>
```

The tool returns a `ToolResponse` containing the reference image.

## Return Values

The `execute` method returns a tuple of `(ToolResponse, reward, metrics_dict)`:

- **ToolResponse**: Contains the reference image and descriptive text
- **reward**: 0.0 (neutral reward for information gathering)
- **metrics_dict**: Dictionary with:
  - `success`: Whether the retrieval was successful
  - `reference_available`: Whether a reference image exists
  - `reason`: The provided reason (if any)

## Error Handling

The tool gracefully handles several edge cases:

1. **No Reference Available**: Returns informative message when sample has no reference
2. **Instance Not Found**: Returns error if instance_id is invalid
3. **Image Load Failure**: Logs error and returns unavailable status

## Example Scenarios

### Scenario 1: Successful Retrieval

```python
# Input
parameters = {"reason": "to compare surface finish"}

# Output
ToolResponse(
    image=[<PIL.Image>],
    text="Retrieved reference image for: to compare surface finish"
)
# reward: 0.0
# metrics: {"success": True, "reference_available": True, "reason": "..."}
```

### Scenario 2: No Reference Available

```python
# Output
ToolResponse(
    text="No reference image is available for this sample..."
)
# reward: 0.0
# metrics: {"success": False, "reference_available": False}
```

## Integration with Reward Calculation

The tool itself provides neutral reward (0.0), as it's an information-gathering operation. The actual reward comes from:

1. **Format compliance**: Proper use of tool call tags
2. **Answer accuracy**: Correct defect detection after viewing reference
3. **Bbox accuracy**: If defects are located after comparison

## Comparison with Image Zoom In Tool

| Feature | Image Reference Tool | Image Zoom In Tool |
|---------|---------------------|-------------------|
| Purpose | Compare with good sample | Inspect specific region |
| Input | Optional reason string | Required bbox coordinates |
| Complexity | Simple (pre-loaded) | Complex (image processing) |
| Use Case | Verify normal vs defect | Examine details closely |
| Rate Limiting | No | Yes (via Ray pool) |

## Best Practices

1. **Use Early**: Call reference tool early in inspection to establish baseline
2. **Combine Tools**: Use reference + zoom for comprehensive analysis
3. **Document Reason**: Provide clear reason for better logging
4. **Handle Absence**: Always check if reference is available before comparison

## Configuration

The tool accepts minimal configuration:

```python
config = {}  # No special configuration needed
tool_schema = OpenAIFunctionToolSchema.model_validate({...})
tool = ImageReferenceTool(config, tool_schema)
```

## Logging

The tool logs important events:

- INFO: Successful initialization and image loading
- WARNING: Missing reference images
- ERROR: Image loading failures
- DEBUG: Instance release operations

